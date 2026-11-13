# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Black-box conformance framework (KEP-0001 M1.7).

Three differential layers, run from the warp-metal venv; the golden pass
runs from the Newton venv (Phase 0 harness pattern — golden side generates,
compare side checks, all paths absolute):

  (a) surface    — for every symbol in the fixed reference-frequency table
                   (``wp-symbols-frequency.txt``, 356 symbols): presence
                   (``hasattr``) plus signature cross-check against golden
                   signatures dumped from warp-lang 1.19.0. The missing list
                   drives M1.1–M1.3 implementation order.
  (b) corpus     — collect every ``@wp.kernel`` on the target closure path
                   (Newton kernels + Phase 0 samples), run CPU golden vs
                   Metal with tolerance assertions. Pass count is a ratchet.
  (c) trajectory — same example on ``cpu`` vs ``metal`` for N steps,
                   per-step ``body_q`` trajectory comparison.

Usage:

    # Golden pass (Newton venv — needs warp-lang 1.19.0):
    /Volumes/ssd/main_link/work/newton/.venv/bin/python -I \
        ~/work/warp-metal/tools/conform.py --dump-golden GOLDEN.json

    # Compare pass (warp-metal venv):
    ~/work/warp-metal/.venv/bin/python -I \
        ~/work/warp-metal/tools/conform.py --layer surface \
        --golden GOLDEN.json [--report OUT.md]

    ~/work/warp-metal/.venv/bin/python -I \
        ~/work/warp-metal/tools/conform.py --layer corpus|trajectory|all

The report (``conform-report.md`` in the KEP dir) is append-in-place: each
layer rewrites its own section, keeping the others.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

# --- Absolute paths (uv run --directory resets cwd; never rely on cwd) ----

NEWTON_REPO = Path("/Volumes/ssd/main_link/work/newton")
KEP_DIR = (
    NEWTON_REPO
    / "keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon"
)
SYMBOLS_PATH = KEP_DIR / "wp-symbols-frequency.txt"
DEFAULT_REPORT = KEP_DIR / "conform-report.md"
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_surface.json"
NEWTON_PYTHON = NEWTON_REPO / ".venv/bin/python"
WARP_METAL_REPO = Path.home() / "work/warp-metal"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def read_reference_symbols() -> list[str]:
    """The 356 ``wp.<sym>`` entries, file order kept (rank = line number)."""
    out = []
    for line in SYMBOLS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("wp."):
            out.append(line[len("wp.") :])
    return out


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_head(repo: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(unknown)"


# --------------------------------------------------------------------------
# Symbol classification (same shape on both sides)
# --------------------------------------------------------------------------


def _classify(obj):
    """Map an attribute of the ``wp`` module to a comparable spec.

    Returns ``(kind, sig)`` where ``sig`` is a JSON-able signature string:

    - ``builtin``    — codegen-level Function (warp.context.Function /
                       warp_metal frontend Function): signature = ordered
                       ``input_types`` parameter labels.
    - ``function``   — plain Python callable (host runtime): signature =
                       ``inspect.signature`` parameter names.
    - ``type``       — class object (vec3, array, ...): qualname.
    - ``module``     — submodule / module reference.
    - ``other``      — constant or anything else: ``type(obj).__name__``.
    """
    # Codegen-level Function (warp.context.Function / vendored registry
    # Function), matched by its ``key`` + ``input_types`` pair. getattr is
    # guarded: stub namespaces (wp.utils) raise on any attribute.
    try:
        input_types = getattr(obj, "input_types", None)
    except Exception:
        input_types = None
    if isinstance(input_types, dict) and isinstance(getattr(obj, "key", None), str):
        return "builtin", ",".join(input_types.keys())
    if inspect.isroutine(obj):
        try:
            sig = inspect.signature(obj)
            params = [
                p.name if p.kind != inspect.Parameter.VAR_POSITIONAL else "*args"
                for p in sig.parameters.values()
            ]
        except (ValueError, TypeError):
            params = ["<unintrospectable>"]
        return "function", ",".join(params)
    if inspect.isclass(obj):
        return "type", getattr(obj, "__qualname__", obj.__name__)
    if inspect.ismodule(obj):
        # Compare only the tail so warp.types vs types (vendored, no leading
        # package) do not report as different modules.
        return "module", getattr(obj, "__name__", "?").rsplit(".", 1)[-1]
    return "other", type(obj).__name__


# --------------------------------------------------------------------------
# Layer (a): surface
# --------------------------------------------------------------------------


def dump_golden(out_path: Path) -> None:
    """Golden pass — run under the NEWTON venv (warp-lang 1.19.0)."""
    try:
        import warp
    except ImportError:
        sys.exit("error: --dump-golden must run under the Newton venv (warp-lang)")
    if not warp.__version__.startswith("1.19"):
        sys.exit(f"error: golden side expects warp-lang 1.19.x, got {warp.__version__}")

    entries = {}
    for sym in read_reference_symbols():
        obj = getattr(warp, sym, None)
        if obj is None and not hasattr(warp, sym):
            entries[sym] = {"kind": "missing"}
            continue
        kind, sig = _classify(obj)
        entries[sym] = {"kind": kind, "sig": sig}

    payload = {
        "generator": "conform.py --dump-golden",
        "warp_version": warp.__version__,
        "n_symbols": len(entries),
        "symbols": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"golden: {len(entries)} symbols -> {out_path}")


# Bucket classifier for the missing list (KEP-0001 buckets A–H). Prefix
# heuristics plus explicit sets for irregulars; unknowns land in "?".
_TILE_PREFIXES = ("tile_",)
_NATIVE_PREFIXES = ("bvh_", "mesh_", "hash_grid_", "volume_", "marching cubes_", "marching_cubes")
_DEBUG_NAMES = {"printf", "print", "rand", "rand_init", "randn", "randint", "rand_sample"}
_ADJOINT_NAMES = {"adjoint", "Tape", "grad", "loss"}
_ARRAY_TYPES = {"array", "array2d", "array3d", "array4d", "indexedarray"}
_TYPE_NAMES = {
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bfloat16", "vec2", "vec3", "vec4", "mat22", "mat33",
    "mat44", "quat", "transform", "spatial_vector", "spatial_matrix", "vec2f", "vec2h",
    "vec3f", "vec3h", "vec3i", "vec4f", "vec4i", "vec4h", "mat22f", "mat33f", "mat44f",
    "quath", "quatf", "quatd", "transformf", "transformd", "quati", "Transform", "DType",
    "Scalar", "Float", "Int", "Vector", "Matrix", "Quaternion", "Transformation",
    "bool8", "uint64_t",
}


def classify_bucket(sym: str, kind: str) -> str:
    if sym.startswith(_TILE_PREFIXES) or sym in ("launch_tiled", "blocked_dim"):
        return "E"
    if sym.startswith(_NATIVE_PREFIXES) or sym in {"Bvh", "Mesh", "HashGrid", "Volume", "mesh_get"}:
        return "F"
    if sym in _DEBUG_NAMES or sym.startswith("rand"):
        return "H"
    if sym in _ADJOINT_NAMES or sym.startswith("adj_"):
        return "G"
    if sym in _ARRAY_TYPES:
        return "B"
    if sym in _TYPE_NAMES:
        return "A"
    if kind == "builtin":
        return "C"
    if kind in ("function", "module"):
        return "D"
    return "?"


# Symbols whose kind/sig difference is behavior-equivalent at the Python
# level (black-box tolerance), with the reason recorded in the report.
TOLERATED_MISMATCHES = {
    # upstream exports a plain host decorator; vendored registry exposes the
    # codegen Function — both callable as wp.static(expr) with same semantics.
    "static": "callable-equivalent（上游=普通函数，vendored=registry Function，内核语义一致）",
}


def run_surface(golden_path: Path) -> dict:
    """Compare pass — run under the WARP-METAL venv."""
    try:
        import warp_metal
    except ImportError:
        sys.exit("error: compare pass must run under the warp-metal venv")

    if not golden_path.is_file():
        sys.exit(f"error: golden signatures not found at {golden_path}; run --dump-golden under the Newton venv first")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    rows = []
    for rank, sym in enumerate(read_reference_symbols(), start=1):
        g = golden["symbols"].get(sym)
        if g is None:
            sys.exit(f"error: symbol {sym!r} missing from golden file (stale golden?)")
        obj = getattr(warp_metal, sym, None)
        if obj is None and not hasattr(warp_metal, sym):
            if g["kind"] == "missing":
                # Kernel-only builtin (e.g. wp.tid): not a module attribute in
                # upstream warp either — kernel-text reference, not a Python
                # surface gap. Counted separately, not in the missing list.
                rows.append({"sym": sym, "rank": rank, "golden_kind": "missing", "status": "kernel_only",
                             "bucket": classify_bucket(sym, "builtin")})
            else:
                rows.append({"sym": sym, "rank": rank, "golden_kind": g["kind"], "status": "missing",
                             "bucket": classify_bucket(sym, g.get("kind", "?"))})
            continue
        kind, sig = _classify(obj)
        if g["kind"] == "missing":
            # Golden side has no module attribute (kernel-only builtin like
            # wp.vector) but vendored star-import exports it: harmless extra
            # surface, cannot break a warp-lang-compatible caller.
            status = "extra"
        elif kind == g["kind"] and sig == g.get("sig"):
            status = "ok"
        elif sym in TOLERATED_MISMATCHES:
            status = "tolerated"
        else:
            status = "mismatch"
        rows.append({"sym": sym, "rank": rank, "golden_kind": g["kind"], "golden_sig": g.get("sig"),
                     "kind": kind, "sig": sig, "status": status,
                     "bucket": classify_bucket(sym, g.get("kind", "?"))})

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_mismatch = sum(1 for r in rows if r["status"] == "mismatch")
    n_missing = sum(1 for r in rows if r["status"] == "missing")
    n_kernel_only = sum(1 for r in rows if r["status"] == "kernel_only")
    n_extra = sum(1 for r in rows if r["status"] == "extra")
    n_tolerated = sum(1 for r in rows if r["status"] == "tolerated")
    return {
        "warp_metal_version": warp_metal.__version__,
        "golden_warp_version": golden["warp_version"],
        "golden_sha": sha256_of_file(golden_path),
        "n_symbols": len(rows),
        "n_ok": n_ok,
        "n_mismatch": n_mismatch,
        "n_missing": n_missing,
        "n_kernel_only": n_kernel_only,
        "n_extra": n_extra,
        "n_tolerated": n_tolerated,
        "rows": rows,
    }


def format_surface_section(res: dict) -> str:
    lines = [
        f"- 运行：warp-metal {res['warp_metal_version']} vs golden warp-lang {res['golden_warp_version']}"
        f"（golden sha {res['golden_sha']}）",
        f"- 结果：**{res['n_ok']} ok / {res['n_mismatch']} mismatch / {res['n_missing']} missing**（共 {res['n_symbols']}；"
        f"{res['n_kernel_only']} kernel-only 两侧皆不可 hasattr，{res['n_extra']} extra 导出，{res['n_tolerated']} tolerated，均不计缺失/违规）",
        "",
        "### 缺失符号（M1.1–M1.3 实现顺序依据，rank = 频次表行号）",
        "",
        "| bucket | symbol | golden kind |",
        "|---|---|---|",
    ]
    for r in res["rows"]:
        if r["status"] == "missing":
            lines.append(f"| {r['bucket']} | wp.{r['sym']} | {r['golden_kind']} |")
    lines += ["", "### 签名 mismatch（能力面内黑盒一致性违规项）", ""]
    mismatches = [r for r in res["rows"] if r["status"] == "mismatch"]
    if not mismatches:
        lines.append("（无）")
    else:
        lines += ["| bucket | symbol | golden | local |", "|---|---|---|---|"]
        for r in mismatches:
            g = f"{r['golden_kind']}({r.get('golden_sig')})"
            l = f"{r['kind']}({r.get('sig')})"
            lines.append(f"| {r['bucket']} | wp.{r['sym']} | {g} | {l} |")
    tolerated = [r for r in res["rows"] if r["status"] == "tolerated"]
    if tolerated:
        lines += ["", "### tolerated mismatch（行为等价，黑盒容差内）", ""]
        for r in tolerated:
            lines.append(f"- wp.{r['sym']}：{TOLERATED_MISMATCHES[r['sym']]}")
    extras = [r for r in res["rows"] if r["status"] == "extra"]
    if extras:
        lines += ["", "### extra 导出（golden 无此模块属性，kernel-only builtin 泄漏，无害）", ""]
        lines.append("、".join(f"wp.{r['sym']}" for r in extras))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Layers (b) corpus and (c) trajectory — M1.7(b)/(c)
# --------------------------------------------------------------------------


def run_corpus() -> dict:
    sys.exit("error: corpus layer not implemented yet (KEP-0001 M1.7(b))")


def run_trajectory() -> dict:
    sys.exit("error: trajectory layer not implemented yet (KEP-0001 M1.7(c))")


# --------------------------------------------------------------------------
# Report assembly (each layer owns its section; others preserved)
# --------------------------------------------------------------------------

SECTION_TITLES = {
    "surface": "## Layer (a) — Surface",
    "corpus": "## Layer (b) — Corpus",
    "trajectory": "## Layer (c) — Trajectory",
}
SECTION_STATUS = {
    "surface": "surface",
    "corpus": "corpus",
    "trajectory": "trajectory",
}


def write_report(report_path: Path, sections: dict[str, str], layer_results: dict) -> None:
    existing = {}
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8")
        for key, title in SECTION_TITLES.items():
            if title in text:
                # keep previous non-run sections verbatim
                start = text.index(title)
                nxt = [text.index(t) for t in SECTION_TITLES.values() if t in text[start + len(title):] and text.index(t, start) > start]
                end = min(nxt) if nxt else len(text)
                existing[key] = text[start:end].rstrip() + "\n"

    header = [
        "# warp-metal conformance report（黑盒一致性差分 — KEP-0001 M1.7）",
        "",
        f"生成：{datetime.date.today().isoformat()}；conform.py @ warp-metal {git_head(WARP_METAL_REPO)}",
        "ratchet 规则：各层通过数只许升不许降（ROADMAP M1.7）。",
        "",
    ]
    body = []
    for key in ("surface", "corpus", "trajectory"):
        title = SECTION_TITLES[key]
        if key in layer_results:
            body.append(f"{title}\n\n{layer_results[key]}\n")
        elif key in existing:
            body.append(existing[key])
        else:
            body.append(f"{title}\n\n- 状态：未运行（待 M1.7({'b' if key=='corpus' else 'c'})）\n")

    report_path.write_text("\n".join(header) + "\n".join(body), encoding="utf-8")
    print(f"report -> {report_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", choices=["surface", "corpus", "trajectory", "all"], default="surface")
    ap.add_argument("--dump-golden", metavar="PATH", help="golden pass (Newton venv): dump warp 1.19.0 signatures")
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="golden signature JSON for the compare pass")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    if args.dump_golden:
        dump_golden(Path(args.dump_golden))
        return

    results = {}
    if args.layer in ("surface", "all"):
        res = run_surface(args.golden)
        results["surface"] = format_surface_section(res)
        print(f"surface: {res['n_ok']} ok / {res['n_mismatch']} mismatch / {res['n_missing']} missing")
    if args.layer in ("corpus", "all"):
        run_corpus()
    if args.layer in ("trajectory", "all"):
        run_trajectory()
    write_report(args.report, SECTION_TITLES, results)


if __name__ == "__main__":
    main()
