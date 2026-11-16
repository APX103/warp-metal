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
    $NEWTON_REPO/.venv/bin/python -I \
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
import os
import subprocess
import sys
from pathlib import Path

# --- Absolute paths (uv run --directory resets cwd; never rely on cwd) ----

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for paths.py
import paths  # noqa: E402

NEWTON_REPO = paths.newton_repo()
KEP_DIR = (
    NEWTON_REPO
    / "keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon"
)
SYMBOLS_PATH = KEP_DIR / "wp-symbols-frequency.txt"
DEFAULT_REPORT = KEP_DIR / "conform-report.md"
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_surface.json"
NEWTON_PYTHON = NEWTON_REPO / ".venv/bin/python"
WARP_METAL_REPO = Path(__file__).resolve().parent.parent


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
    # upstream's module attr is a *args/**kwargs unpack wrapper; warp-metal
    # spells the same upstream parameter list explicitly. Same call surface
    # (launch_tiled(kernel, dim, inputs=..., outputs=..., block_dim=...)),
    # adjoint kwargs fail loud per KEP-0001.
    "launch_tiled": "签名等价（上游=*args/**kwargs 解包封装，warp-metal=显式上游参数表；调用面一致，adjoint 实参 fail-loud）",
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

# corpus: (newton module, kernel qualname, builder name in corpus_builders).
# Builders live in tools/corpus_builders.py (pure numpy, both venvs).
CORPUS_TABLE = [
    ("newton._src.solvers.solver", "integrate_bodies", "integrate_bodies"),
    ("newton._src.solvers.solver", "_update_effective_inv_mass_inertia", "update_effective_inv_mass_inertia"),
    ("newton._src.geometry.inertia", "compute_body_inverses_kernel", "compute_body_inverses_kernel"),
    ("newton._src.geometry.inertia", "validate_and_correct_inertia_kernel", "validate_and_correct_inertia_kernel"),
    ("newton._src.geometry.bvh", "compute_enabled_shapes", "compute_enabled_shapes"),
    ("newton._src.geometry.bvh", "compute_shape_world_transforms", "compute_shape_world_transforms"),
    ("newton._src.geometry.bvh", "compute_bvh_group_roots", "compute_bvh_group_roots"),
    (
        "newton._src.geometry.kernels",
        "vertex_triangle_collision_detection_kernel",
        "vertex_triangle_collision_detection_kernel",
    ),
    ("newton._src.viewer.kernels", "update_shape_xforms", "update_shape_xforms"),
    ("newton._src.viewer.kernels", "repack_shape_colors", "repack_shape_colors"),
    ("newton._src.viewer.kernels", "repack_shape_opacities", "repack_shape_opacities"),
    ("newton._src.viewer.kernels", "estimate_world_extents", "estimate_world_extents"),
    ("newton._src.viewer.kernels", "compute_com_positions", "compute_com_positions"),
    ("newton._src.sim.collide", "compute_shape_aabbs", "compute_shape_aabbs"),
    ("newton._src.sim.collide", "_record_reduction_overflow", "record_reduction_overflow"),
    ("newton._src.geometry.broad_phase_nxn", "_nxn_broadphase_precomputed_pairs", "nxn_broadphase_precomputed_pairs"),
    ("newton._src.geometry.narrow_phase", "compute_mesh_plane_vert_counts", "compute_mesh_plane_vert_counts"),
    ("newton._src.geometry.narrow_phase", "verify_narrow_phase_buffers", "verify_narrow_phase_buffers"),
    ("newton._src.solvers.xpbd.kernels", "apply_joint_forces", "apply_joint_forces"),
    ("newton._src.solvers.xpbd.kernels", "apply_body_deltas", "apply_body_deltas"),
    ("newton._src.solvers.xpbd.kernels", "solve_body_joints", "solve_body_joints"),
    ("newton._src.solvers.xpbd.kernels", "solve_body_contact_positions", "solve_body_contact_positions"),
    ("newton._src.solvers.xpbd.kernels", "copy_kinematic_body_state_kernel", "copy_kinematic_body_state_kernel"),
    ("newton._src.geometry.contact_reduction_global", "_clear_active_kernel", "clear_active_kernel"),
    ("newton._src.geometry.contact_reduction_global", "_zero_active_count_kernel", "zero_active_count_kernel"),
    ("newton._src.geometry.contact_reduction_global", "reduce_buffered_contacts_kernel", "reduce_buffered_contacts_kernel"),
    (
        "newton._src.geometry.particle_surface_kernels",
        "compute_particle_bounds",
        "compute_particle_bounds",
    ),
    (
        "newton._src.geometry.narrow_phase",
        "narrow_phase_primitive_kernel",
        "narrow_phase_primitive_kernel",
        ("create_narrow_phase_primitive_kernel", ["write_contact_simple", False, False, False]),
    ),
    ("newton._src.geometry.narrow_phase", "narrow_phase_kernel_gjk_mpr", None),
    ("newton._src.geometry.narrow_phase", "narrow_phase_find_mesh_triangle_overlaps_kernel", None),
    ("newton._src.geometry.narrow_phase", "mesh_triangle_contacts_to_reducer_kernel", None),
    ("newton._src.sim.articulation", "eval_articulation_fk", None),
    # KEP-0001 M2.4 P3+P4: steel tile factories (n_dofs=12, 3 articulations;
    # launch_tiled bd=1 via the builders' __block_dim__ pin — the upstream
    # CPU bd=1 serial scalar-fallback discipline vs the matching Metal
    # threadgroup lowering)
    (
        "newton._src.solvers.steel.solver_steel",
        "_get_cholesky_kernel",
        "steel_cholesky_kernel",
        ("_get_cholesky_kernel", [12, "sm_90", 1]),
    ),
    (
        "newton._src.solvers.steel.solver_steel",
        "_get_triangular_solve_kernel",
        "steel_triangular_solve_kernel",
        ("_get_triangular_solve_kernel", [12, "sm_90", 1]),
    ),
    (
        "newton._src.solvers.steel.solver_steel",
        "_get_hinv_jt_plain_kernel",
        "steel_hinv_jt_plain_kernel",
        ("_get_hinv_jt_plain_kernel", [12, 8, "sm_90", 1, 3, False, True]),
    ),
    (
        "newton._src.solvers.steel.solver_steel",
        "_get_hinv_jt_kernel",
        "steel_hinv_jt_kernel",
        ("_get_hinv_jt_kernel", [12, 8, "sm_90", 1, 3, False, True, False]),
    ),
    (
        "newton._src.solvers.steel.solver_steel",
        "_get_crba_cholesky_kernel",
        "steel_crba_cholesky_kernel",
        ("_get_crba_cholesky_kernel", [12, "sm_90", 1]),
    ),
    (
        "newton._src.solvers.steel.solver_steel",
        "_get_cholesky_inverse_tiled_kernel",
        "steel_cholesky_inverse_tiled_kernel",
        ("_get_cholesky_inverse_tiled_kernel", [12, "sm_90", 1]),
    ),
    # KEP-0001 M3.2 P1: mujoco_warp kernels closing over the identity/diag/
    # matrix/determinant builtin families (M3.1 coverage matrix §4). Factory
    # rows resolve the @cache_kernel kernels with deterministic=False (the
    # Newton call sites' default); fixtures in corpus_builders (_rod_layout /
    # _ellipsoid scene docs there).
    (
        "mujoco_warp._src.passive",
        "_flex_rod_mul",
        "flex_rod_mul",
        ("_flex_rod_mul", [False, 0]),
    ),
    (
        "mujoco_warp._src.passive",
        "_flex_rod_bending",
        "flex_rod_bending",
        ("_flex_rod_bending", [0]),
    ),
    (
        "mujoco_warp._src.derivative",
        "_eff_flex_rod_stiff",
        "eff_flex_rod_stiff",
        ("_eff_flex_rod_stiff", [False]),
    ),
    (
        "mujoco_warp._src.derivative",
        "_qderiv_ellipsoid_fluid",
        "qderiv_ellipsoid_fluid",
        ("_qderiv_ellipsoid_fluid", [False]),
    ),
    # wp.add_inplace / wp.sub_inplace / vec10 rows (KEP-0001 M3.2 package c:
    # the compound-element builtins and the generic vecN def now lower —
    # ACTIVE rows over the mujoco_warp 3.17.0 closures that use them)
    (
        "mujoco_warp._src.passive",
        "_flex_hessian_elem",
        "flex_hessian_elem",
        ("_flex_hessian_elem", [False]),
    ),
    (
        "mujoco_warp._src.passive",
        "_flex_elasticity",
        "flex_elasticity",
        ("_flex_elasticity", [False]),
    ),
    ("mujoco_warp._src.smooth", "_cinert", "cinert"),
    ("mujoco_warp._src.forward", "_implicit_free_body", "implicit_free_body"),
    # KEP-0001 M3.2 package b: the mujoco_warp tile closure (recon B.4) —
    # JTDAJ dense pair (scatter_add / arange / ones / broadcast / matmul
    # chain, solver block_dim=128) and the block-cholesky pair
    # (tile_load_indexed CSR gather with the io.py:1224 sentinel, smooth
    # block_dim=32). TileSet factory args are built per side with the
    # ambient wp package (only .size feeds the @cache_kernel specialization).
    (
        "mujoco_warp._src.solver",
        "_update_gradient_JTDAJ_dense_tiled_compact",
        "jtdaj_dense_compact",
        ("_update_gradient_JTDAJ_dense_tiled_compact", [16, 16, 8]),
    ),
    (
        "mujoco_warp._src.solver",
        "_update_gradient_JTDAJ_dense_tiled",
        "jtdaj_dense",
        ("_update_gradient_JTDAJ_dense_tiled", [16, 16, 8, 12]),
    ),
    (
        "mujoco_warp._src.smooth",
        "_tile_cholesky_factorize_block",
        "tile_cholesky_factorize_block",
        ("_tile_cholesky_factorize_block", [("__tileset__", 4)]),
    ),
    (
        "mujoco_warp._src.smooth",
        "_tile_cholesky_factorize_solve_block",
        "tile_cholesky_factorize_solve_block",
        ("_tile_cholesky_factorize_solve_block", [("__tileset__", 4)]),
    ),
]

_GJK_SKIP = "GJK/MPR factory kernel calls @wp.func_native support functions (Metal translation: Phase 2)"
_ARTICULATION_SKIP = "articulation FK needs a joint-model fixture (builder pending)"
_SKIP_BY_KERNEL = {
    "narrow_phase_kernel_gjk_mpr": _GJK_SKIP,
    "narrow_phase_find_mesh_triangle_overlaps_kernel": _GJK_SKIP,
    "mesh_triangle_contacts_to_reducer_kernel": _GJK_SKIP,
    "eval_articulation_fk": _ARTICULATION_SKIP,
}
_DEFAULT_SEED = 20261113
CORPUS_NPZ = Path(__file__).resolve().parent / "corpus_golden.npz"


def _resolve_kernel(mod, qualname, factory):
    """Module-level kernel by qualname, or a factory-built kernel.

    Factory rows carry ``(factory_name, [args])``; string args resolve as
    module attributes when one exists (e.g. the writer @wp.func) and pass
    verbatim otherwise (literal strings such as device_arch tags), so both
    sides build the identical kernel object.
    """
    if factory is None:
        return getattr(mod, qualname)
    fname, fargs = factory
    fn = getattr(mod, fname)
    args = []
    for a in fargs:
        if isinstance(a, tuple) and a and a[0] == "__tileset__":
            # KEP-0001 M3.2: TileSet factory arg (mjw block-cholesky pair).
            # Built with the ambient wp package (real warp in the golden
            # phase, the shim in the compare phase) — the @cache_kernel
            # specialization reads only tile.size; the dummy arrays stand in
            # for the dataclass fields (launch-time fixtures bind the real
            # elemid/adr arrays as kernel args).
            from mujoco_warp._src.types import TileSet

            import warp as _wp

            size = int(a[1])
            args.append(
                TileSet(
                    adr=_wp.array([0], dtype=_wp.int32),
                    size=size,
                    elemid=_wp.array([0] * (size * size), dtype=_wp.int32),
                )
            )
        elif isinstance(a, str) and hasattr(mod, a):
            args.append(getattr(mod, a))
        else:
            args.append(a)
    return fn(*args)


def _corpus_dtype(pkg, key: str):
    """Element dtype object for a corpus dtype key, per side's wp package."""
    if key == "vec10":
        # mujoco_warp's 10-component vector (cinert/crb): the kernel
        # annotations carry this exact type object on both sides, so the
        # array dtype must be it (a plain vector(10) would fail the launch
        # dtype check on the golden side)
        from mujoco_warp._src.types import vec10

        return vec10
    return {
        "f32": pkg.float32, "i32": pkg.int32, "i64": pkg.int64, "u32": pkg.uint32, "u64": pkg.uint64,
        "bool": pkg.bool, "vec2": pkg.vec2, "vec3": pkg.vec3, "vec4": pkg.vec4, "vec2i": pkg.vec2i,
        "vec3i": pkg.vec3i, "vec4i": pkg.vec4i, "mat33": pkg.mat33,
        "transform": pkg.transform, "spatial_vector": pkg.spatial_vector,
        "spatial_matrix": pkg.spatial_matrix,
    }[key]


def _corpus_launch_dim(args) -> int | tuple:
    """Launch dim: explicit '__dim__' entry, else first output, else first array."""
    explicit = [v for (n, _k, v) in args if n == "__dim__"]
    if explicit:
        return explicit[0]
    for name, key, val in args:
        if key.startswith("out"):
            return val
    for name, key, val in args:
        if key.startswith("arr"):
            return int(val.shape[0])
    raise ValueError("cannot derive corpus launch dim")



def _arr_view(arr):
    """numpy view of an array from either side (real warp vs warp_metal)."""
    v = getattr(arr, "numpy_view", None)
    return v() if v is not None else arr.numpy()


def _bitwise_equal(a, b) -> bool:
    """Bitwise (byte-level) equality of two numpy views.

    KEP-0001 M3.2 §6-1 (R1 A.2): replay-vs-eager compares BITWISE, floats
    included — same PSO, same pointers, same order must reproduce every bit;
    the uint-view criterion dodges -0.0/NaN float reconciliation. uint8 view
    over contiguous buffers generalizes R1's uint32-view rule to every
    element width (int8/uint8/bool rows cannot view as uint32); byte
    equality is the identical semantics.
    """
    import numpy as np

    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def _kernel_atomic_sites(kern, launch_args, block_dim):
    """Transitive atomic call sites in a kernel's emitted MSL.

    KEP-0001 M3.2 §6-1 capture-allowlist criterion (R1 A.2: "closure 符号集
    无 atomic_*"). Scans the kernel body plus every transitively called
    @wp.func body of the exact unit the eager launch compiled — same
    ``_PSO_CACHE`` key as the launch (launch_tiled bakes block_dim, plain
    launches pass None; generic kernels concretize from the same arg list).
    Every real atomic lowering emits the substring "atomic" into a function
    body line (scalar-array ``atomic_fetch_*_explicit``, vec-component /
    2-D ret forms calling the ``wp_atomic_*`` helpers, CAS/exchange). The
    unconditional per-dtype helper DEFINITIONS ride along inside
    vector-dtype struct decls (translate.py ensure_vec_def) — struct text is
    NOT scanned, so a closure that never calls an atomic scans clean.
    Returns ``[(fn_name, line), ...]`` (empty → allowlist-eligible).
    """
    import warp_metal.runtime.host as _wm_host

    k = kern
    if getattr(k, "is_generic", False):
        k = _wm_host._concretize_generic_kernel(k, list(launch_args))
    _pso, unit = _wm_host._unit_for(k, block_dim)
    sites = []
    for fn in (unit.kernel, *unit.functions):
        for ln in fn.lines:
            if "atomic" in ln:
                sites.append((fn.name, ln.strip()))
    return sites


# composite element dtype keys: their numpy packing carries one (or two)
# trailing element axes (vec3 → (..., 3), mat33 → (..., 3, 3)), so the
# array shape is val.shape minus those axes (KEP-0001 M3.2 P1: first arr2
# rows with composite elements — mujoco_warp xipos/ximat/cdof fixtures)
_COMPOSITE_DTYPE_KEYS = {
    "vec2", "vec3", "vec4", "vec2i", "vec3i", "vec4i",
    "mat33", "transform", "spatial_vector", "spatial_matrix",
    "vec10",
}


def _alloc_array(pkg, key, val, device="cpu"):
    """Allocate + fill a WarpArray for an arr:/arr2:/out:/out2: builder key."""
    if key.startswith(("arr:", "io:")):
        off = 4 if key.startswith("arr:") else 3
        elem = _corpus_dtype(pkg, key[off:])
        arr = pkg.zeros(int(val.shape[0]), dtype=elem, device=device)
        view = _arr_view(arr)
        view[:] = val.reshape(view.shape)
        return arr
    if key.startswith("arr2:"):
        elem = _corpus_dtype(pkg, key[5:])
        shape = val.shape[:-1] if key[5:] in _COMPOSITE_DTYPE_KEYS else val.shape
        arr = pkg.zeros(shape, dtype=elem, device=device)
        view = _arr_view(arr)
        view[:] = val.reshape(view.shape)
        return arr
    if key.startswith("arr3:"):
        elem = _corpus_dtype(pkg, key[5:])
        shape = val.shape[:-1] if key[5:] in _COMPOSITE_DTYPE_KEYS else val.shape
        arr = pkg.zeros(shape, dtype=elem, device=device)
        view = _arr_view(arr)
        view[:] = val.reshape(view.shape)
        return arr
    if key.startswith("out3:"):
        elem = _corpus_dtype(pkg, key[5:])
        out_arr = pkg.zeros(val, dtype=elem, device=device)
        return out_arr
    if key.startswith("out2:"):
        elem = _corpus_dtype(pkg, key[5:])
        out_arr = pkg.zeros(val, dtype=elem, device=device)
        return out_arr
    if key.startswith("outinit:"):
        # output array with a deterministic INITIAL value (e.g. the ±1e30
        # bound sentinels the particle-bounds kernels reduce against):
        # val is the numpy fill, val.shape[0] the element count
        elem = _corpus_dtype(pkg, key[8:])
        out_arr = pkg.zeros(int(val.shape[0]), dtype=elem, device=device)
        view = _arr_view(out_arr)
        view[:] = val.reshape(view.shape)
        return out_arr
    if key.startswith("out:"):
        elem = _corpus_dtype(pkg, key[4:])
        shape = val if isinstance(val, tuple) else int(val)
        out_arr = pkg.zeros(shape, dtype=elem, device=device)
        return out_arr
    raise ValueError(f"bad array dtype key {key!r}")


_PACKED_CONTEXTS: list = []
_PACKED_MESHES: dict = {}


def _pack_args(pkg, args, device="cpu", mod=None):
    """(launch_args, dim, out_arrays, out_names) from builder triples.

    Every argument is passed through ``inputs`` in signature order (upstream
    warp treats ``outputs`` as pure convenience — Newton's own call sites
    pass output arrays via ``inputs`` too). ``out_arrays``/``out_names``
    capture the buffers to compare after the launch.

    ``mod`` is the kernel's module (needed to resolve ``struct:<ClassName>``
    annotations); struct values are ``{field: (key, val)}`` dicts using the
    same keys, with ``out:``-prefixed fields registered for comparison as
    ``"<arg>.<field>"``.

    ``__dim__`` and ``__block_dim__`` entries are routing pins, not kernel
    args: ``__block_dim__`` mirrors Newton's own tile launches (plain
    ``wp.launch(..., block_dim=bd)``), which both conformance sides honor so
    the block layout matches the kernel's ``block, lane = wp.tid()`` idiom.
    """
    import numpy as np

    _PACKED_MESHES.clear()
    launch_args, out_arrays, out_names = [], [], []
    dim = _corpus_launch_dim(args)
    for name, key, val in args:
        if name in ("__dim__", "__block_dim__"):
            continue
        if key.startswith("struct:"):
            cls = getattr(mod, key[7:])
            inst = cls()
            for fname, (fkey, fval) in val.items():
                if fkey.startswith("scalar:"):
                    setattr(inst, fname, bool(fval) if fkey[7:] == "bool" else fval)
                elif fkey.startswith("value:"):
                    t = pkg.transform()
                    for i, v in enumerate(np.asarray(fval, dtype=np.float32).ravel()):
                        t[i] = v
                    setattr(inst, fname, t)
                else:
                    farr = _alloc_array(pkg, fkey, fval, device)
                    setattr(inst, fname, farr)
                    if fkey.startswith("out"):
                        out_arrays.append(farr)
                        out_names.append(f"{name}.{fname}")
            launch_args.append(inst)
        elif key.startswith("arr:u64") and isinstance(val, dict):
            # KEP-0001 M2.2: a u64 array whose content is wp.Mesh ids packed
            # earlier in the SAME arg list ("mesh" key) — Newton passes mesh
            # ids to kernels through shape_source-style arrays. None entries
            # stay 0 (the null mesh the kernels' `id != 0` guards select).
            ids = val["mesh_ids"]
            arr = pkg.zeros(len(ids), dtype=pkg.uint64, device=device)
            view = _arr_view(arr)
            for i, ref in enumerate(ids):
                view[i] = 0 if ref is None else _PACKED_MESHES[ref]
            launch_args.append(arr)
        elif key.startswith("arr:") or key.startswith("io:") or key.startswith(
            ("arr2:", "arr3:", "out2:", "out3:", "out:", "outinit:")
        ):
            arr = _alloc_array(pkg, key, val, device)
            launch_args.append(arr)
            if key.startswith("out"):
                out_arrays.append(arr)
                out_names.append(name)
        elif key.startswith("scalar:"):
            launch_args.append(bool(val) if key[7:] == "bool" else val)
        elif key.startswith("value:"):
            # value-typed param: wp.transform-style 7-float sequence
            t = pkg.transform()
            for i, v in enumerate(np.asarray(val, dtype=np.float32).ravel()):
                t[i] = v
            launch_args.append(t)
        elif key == "bvh":
            # KEP-0001 M2.1: build the BVH per conformance phase (golden:
            # real warp CPU tree; compare: warp-metal host tree) over the
            # same deterministic bounds; the kernel arg is the uint64 id
            lowers = np.asarray(val["lowers"], dtype=np.float32)
            uppers = np.asarray(val["uppers"], dtype=np.float32)
            groups = val.get("groups")
            la = pkg.zeros(len(lowers), dtype=pkg.vec3, device=device)
            _arr_view(la)[:] = lowers
            ua = pkg.zeros(len(uppers), dtype=pkg.vec3, device=device)
            _arr_view(ua)[:] = uppers
            ga = None
            if groups is not None:
                ga = pkg.zeros(len(groups), dtype=pkg.int32, device=device)
                _arr_view(ga)[:] = np.asarray(groups, dtype=np.int32)
            bvh = pkg.Bvh(la, ua, constructor="sah", groups=ga, leaf_size=2)
            # keepalive: only the uint64 id escapes this scope; releasing the
            # Bvh destroys the tree context and the launch would dereference
            # a freed handle (bus error on the golden side)
            _PACKED_CONTEXTS.append(bvh)
            launch_args.append(int(bvh.id))
        elif key == "mesh":
            # KEP-0001 M2.2: build the wp.Mesh per conformance phase (golden:
            # real warp CPU mesh with its CPU BVH; compare: warp-metal mesh
            # with the host SAH tree). Only the uint64 id escapes this scope —
            # keepalive holds the Mesh (compare side releases its buffers in
            # __del__; golden side frees its CPU tree).
            points = np.asarray(val["points"], dtype=np.float32)
            indices = np.asarray(val["indices"], dtype=np.int32)
            pa = pkg.zeros(len(points), dtype=pkg.vec3, device=device)
            _arr_view(pa)[:] = points
            ia = pkg.zeros(len(indices), dtype=pkg.int32, device=device)
            _arr_view(ia)[:] = indices
            mesh = pkg.Mesh(points=pa, indices=ia)
            _PACKED_CONTEXTS.append(mesh)
            _PACKED_MESHES[val["name"]] = int(mesh.id)
        else:
            raise ValueError(f"bad corpus dtype key {key!r}")
    return launch_args, dim, out_arrays, out_names


def _corpus_block_dim(args):
    """Explicit block_dim pin from a ``__block_dim__`` builder entry (or None)."""
    for name, _key, val in args:
        if name == "__block_dim__":
            return int(val)
    return None


def _corpus_launch(wp, kern, args, launch_args, device):
    """Launch a corpus kernel the way its builder pinned it.

    Rows with a ``__block_dim__`` pin are tile kernels: both sides go through
    ``wp.launch_tiled`` exactly like Newton's own call sites. Upstream CPU
    resolves block_dim to 1 (``enable_cpu_blocks`` off — the serial bd=1
    scalar-fallback discipline every golden is generated with); warp-metal
    compiles the matching threadgroup-backed lowering under the same block
    size. Unpinned rows keep the plain ``wp.launch`` (per-thread scalar tier).
    """
    dim = _corpus_launch_dim(args)
    bd = _corpus_block_dim(args)
    if bd is None:
        wp.launch(kern, dim, inputs=launch_args, outputs=[], device=device)
    else:
        # upstream launch_tiled requires keyword dim (RuntimeError otherwise)
        wp.launch_tiled(kern, dim=dim if isinstance(dim, (tuple, list)) else [dim],
                        inputs=launch_args, outputs=[], device=device, block_dim=bd)


def _corpus_row_capture(wp, cb, short, builder, mod, kern, args, launch_args,
                        out_names, out_arrays, capture_rows, capture_excluded):
    """Capture replay-equivalence for one corpus row (KEP-0001 M3.2 §6-1, R1 A.2(i)).

    Eligibility = the row's emitted MSL closure (kernel + transitive @wp.func
    bodies) contains no atomic lowering — see ``_kernel_atomic_sites``. Atomic
    rows are excluded: replay-vs-eager bit equality is undefined under
    hardware atomic order (upstream CUDA graph replay gives no atomic-order
    guarantee either).

    For each eligible row: fresh buffers from the SAME builder + seed, a
    ``wp.ScopedCapture`` around the identical ``_corpus_launch``, then
    ``capture_launch`` replay; outputs compare bitwise against the row's
    eager outputs. Run under BOTH batch modes (default batching and
    ``WARP_METAL_BATCH=0`` — the per-record replay path must be bit-equal
    too; guardrail A.4.3). Replay starts immediately after capture_end (no
    eager code touches the captured buffers in between — guardrail A.4.1);
    batch segment counts are never asserted, only final outputs (A.4.2).
    Appends to ``capture_rows`` / ``capture_excluded``; never raises — the
    corpus layer reports, never aborts.
    """
    import numpy as np

    bd = _corpus_block_dim(args)
    try:
        sites = _kernel_atomic_sites(kern, launch_args, bd)
    except Exception as e:  # noqa: BLE001 — eligibility scan reports, never aborts
        capture_excluded.append((short, f"atomic-scan unavailable: {type(e).__name__}: {str(e)[:80]}"))
        return
    if sites:
        fn_name, ln = sites[0]
        capture_excluded.append((short, f"atomic closure ({fn_name}): {ln[:90]}"))
        return

    # eager reference snapshot: the capture pass writes FRESH buffers (same
    # builder + seed re-roll → identical fixture content), so these copies
    # are the bit-equality reference
    ref = [np.ascontiguousarray(_arr_view(a)).copy() for a in out_arrays]
    modes = {}
    for mode, env in (("batch", None), ("unbatched", "0")):
        prev = os.environ.pop("WARP_METAL_BATCH", None)
        if env is not None:
            os.environ["WARP_METAL_BATCH"] = env
        try:
            args2, _skips = cb.BUILDERS[builder](np.random.default_rng(_DEFAULT_SEED))
            launch_args2, _dim, out2, _names2 = _pack_args(wp, args2, device="metal", mod=mod)
            with wp.ScopedCapture() as cap:
                _corpus_launch(wp, kern, args2, launch_args2, "metal")
            # guardrail A.4.1: first replay IMMEDIATELY after capture_end
            wp.capture_launch(cap.graph)
            wp.synchronize()
            modes[mode] = [
                out_names[i]
                for i in range(len(out_names))
                if not _bitwise_equal(_arr_view(out2[i]), ref[i])
            ]
        except Exception as e:  # noqa: BLE001 — capture sub-mode reports per row/mode
            modes[mode] = [f"{type(e).__name__}: {str(e)[:100]}"]
        finally:
            if prev is None:
                os.environ.pop("WARP_METAL_BATCH", None)
            else:
                os.environ["WARP_METAL_BATCH"] = prev
    capture_rows.append((short, modes))


def _format_corpus_capture(capture_rows, capture_excluded) -> list[str]:
    """Report lines for the corpus capture replay-equivalence sub-count."""
    n_ok = sum(
        1
        for _short, modes in capture_rows
        if all(isinstance(v, list) and not v for v in modes.values())
    )
    lines = [
        "",
        "### capture replay-equivalence（KEP-0001 M3.2 §6-1；metal-only 自比较，golden 侧零改动）",
        "",
        "- allowlist 判据：kernel 的 MSL closure（kernel 体 + 传递 @wp.func 体）无 atomic lowering"
        "（`atomic_fetch_*_explicit` / `wp_atomic_*` 调用点；vector-dtype struct 内的无条件 helper"
        " 定义不算——只扫函数体行，见 `_kernel_atomic_sites`）。tile_scatter_add 的 shared 无冲突"
        "累加（无 atomic lowering）在 allowlist 内。原子行不进 allowlist：重放 vs eager 位等无定义。",
        f"- **capture-replay: {n_ok} rows bitwise == eager (batch + unbatched)**"
        "（ratchet 子计数，只升不降，前值见 git 历史；corpus 主计数 ratchet 不变）",
    ]
    if capture_excluded:
        lines += ["", "| kernel | allowlist exclusion |", "|---|---|"]
        for name, why in sorted(capture_excluded):
            lines.append(f"| {name} | {why} |")
    if capture_rows:
        lines += ["", "| kernel | batch | unbatched |", "|---|---|---|"]
        for name, modes in sorted(capture_rows):
            cells = []
            for mode in ("batch", "unbatched"):
                v = modes.get(mode, "<missing>")
                cells.append("==" if isinstance(v, list) and not v else "; ".join(map(str, v)))
            lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    return lines


def run_corpus_golden(out_npz: Path) -> dict:
    """Golden phase (Newton venv, real warp): CPU reference for every entry."""
    import numpy as np

    try:
        import warp  # noqa: F401 — must be the REAL warp-lang
        if "warp_metal" in sys.modules:
            raise ImportError
    except ImportError:
        sys.exit("error: corpus golden phase must run under the Newton venv (real warp)")

    tool_dir = Path(__file__).resolve().parent
    for p in (str(NEWTON_REPO / "src"), str(NEWTON_REPO), str(tool_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import warp as wp  # real warp-lang, CPU(LLVM)

    import corpus_builders as cb

    meta = {}
    payload = {}
    for row in CORPUS_TABLE:
        module, qualname, builder, *rest = row
        factory = rest[0] if rest else None
        entry = f"{module}.{qualname}"
        if builder is None:
            meta[entry] = {"skip": _SKIP_BY_KERNEL[qualname]}
            continue
        args, skips = cb.BUILDERS[builder](np.random.default_rng(_DEFAULT_SEED))
        if skips:
            meta[entry] = {"skip": "; ".join(skips)}
            continue
        mod = importlib.import_module(module)
        kern = _resolve_kernel(mod, qualname, factory)
        launch_args, dim, out_arrays, out_names = _pack_args(wp, args, device="cpu", mod=mod)
        _corpus_launch(wp, kern, args, launch_args, "cpu")
        wp.synchronize()
        for i, name in enumerate(out_names):
            payload[f"{qualname}::{name}"] = _arr_view(out_arrays[i]).copy()
        meta[entry] = {"builder": builder, "dim": dim}
    np.savez(out_npz, **payload)
    (out_npz.parent / "corpus_golden_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _metal_shim_bootstrap() -> None:
    """Alias warp -> warp_metal (mini-M1.5) BEFORE importing newton modules."""
    import warp_metal

    sys.modules["warp"] = warp_metal
    for _name, _mod in list(sys.modules.items()):
        if _name == "warp_metal" or _name.startswith("warp_metal."):
            sys.modules.setdefault("warp" + _name[len("warp_metal"):], _mod)
        if _name.startswith("warp_metal.frontend"):
            _rest = _name[len("warp_metal.frontend"):]
            sys.modules.setdefault("warp." + _rest.lstrip("."), _mod)
            sys.modules.setdefault("warp._src" + _rest, _mod)


def run_corpus(golden_npz: Path) -> str:
    """Compare phase (this venv): Metal launches vs the CPU golden npz."""
    import numpy as np

    if not golden_npz.is_file():
        # two-phase: run the golden side under the Newton venv first
        cmd = [
            str(NEWTON_PYTHON), "-I", str(Path(__file__).resolve()),
            "--layer", "corpus", "--corpus-golden", str(golden_npz),
        ]
        print("golden phase:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    meta = json.loads(golden_npz.with_name("corpus_golden_meta.json").read_text())
    data = np.load(golden_npz)

    tool_dir = Path(__file__).resolve().parent
    for p in (str(NEWTON_REPO / "src"), str(NEWTON_REPO), str(tool_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _metal_shim_bootstrap()
    import warp_metal as wp

    import corpus_builders as cb

    passed, failed, skipped = [], [], []
    # KEP-0001 M3.2 §6-1: capture replay-equivalence sub-mode state (see
    # _corpus_row_capture); the headline ratchet below is untouched by it
    capture_rows, capture_excluded = [], []
    for row in CORPUS_TABLE:
        module, qualname, builder, *rest = row
        factory = rest[0] if rest else None
        entry = f"{module}.{qualname}"
        short = qualname.split(".")[-1]
        m = meta.get(entry, {})
        if "skip" in m:
            skipped.append((short, m["skip"]))
            continue
        args, _skips = cb.BUILDERS[builder](np.random.default_rng(_DEFAULT_SEED))
        try:
            mod = importlib.import_module(module)
            kern = _resolve_kernel(mod, qualname, factory)
            launch_args, dim, out_arrays, out_names = _pack_args(wp, args, device="metal", mod=mod)
            _corpus_launch(wp, kern, args, launch_args, "metal")
            wp.synchronize()
            ok, detail = True, []
            for i, name in enumerate(out_names):
                got = _arr_view(out_arrays[i])
                want = data[f"{qualname}::{name}"]
                if got.dtype.kind in "iub":
                    if not np.array_equal(got.reshape(want.shape), want):
                        ok = False
                        detail.append(name)
                else:
                    try:
                        np.testing.assert_allclose(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)
                    except AssertionError:
                        ok = False
                        detail.append(name)
            (passed if ok else failed).append((short, ",".join(detail) or "all outputs"))
            if ok:
                # capture sub-mode runs only for golden-consistent rows: the
                # sub-count certifies bitwise replay==eager on rows whose
                # eager outputs are known-good vs the CPU golden
                _corpus_row_capture(wp, cb, short, builder, mod, kern, args,
                                    launch_args, out_names, out_arrays,
                                    capture_rows, capture_excluded)
        except Exception as e:  # noqa: BLE001 — corpus entries report, never abort
            failed.append((short, f"{type(e).__name__}: {str(e)[:120]}"))

    lines = [
        f"- golden: {len(meta)} entries, npz `{golden_npz.name}` (seed {_DEFAULT_SEED}); "
        f"Metal compare rtol 1e-5 / atol 1e-6（整数位级相等）",
        f"- **passed: {len(passed)}**（ratchet 只升不降；前值见 git 历史）",
        f"- failed: {len(failed)}; skipped: {len(skipped)}",
        "",
    ]
    if passed:
        lines.append("| kernel | outputs |")
        lines.append("|---|---|")
        for name, detail in sorted(passed):
            lines.append(f"| {name} | {detail} |")
        lines.append("")
    if failed:
        lines.append("| kernel | first divergence / error |")
        lines.append("|---|---|")
        for name, detail in sorted(failed):
            lines.append(f"| {name} | {detail} |")
        lines.append("")
    if skipped:
        lines.append("| kernel | skip reason |")
        lines.append("|---|---|")
        for name, reason in sorted(skipped):
            lines.append(f"| {name} | {reason} |")
    lines += _format_corpus_capture(capture_rows, capture_excluded)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Layer (c) — trajectory: same example, cpu (real warp) vs metal (shim),
# per-step body_q comparison. The golden side runs in the Newton venv WITHOUT
# NEWTON_WARP_BACKEND (real warp, device cpu); the compare side runs in the
# Newton venv WITH NEWTON_WARP_BACKEND=metal. Contact-free motion must match
# at corpus tolerances (rtol 1e-5 / atol 1e-6); once contacts engage, the
# reducer's atomic slot allocation makes per-thread export order a scheduling
# artifact, and clustering picks different representatives — an anticipated
# divergence class (ROADMAP M1.7(c)), bounded by an explicit drift cap.
# --------------------------------------------------------------------------

TRAJECTORY_CASES = {
    # name -> (n_steps, drift cap on per-step max |Δbody_q| component; None = strict only)
    "basic_pendulum": (120, None),  # contact-free: strict everywhere
    # cap recalibrated 0.10 -> 0.15 (user-ratified 2026-11-15, ROADMAP decision log):
    # 3 same-day crossings of the old cap (1.010e-1 / 1.149e-1 / 1.064e-1) sit inside
    # the atomic-order drift envelope (+0-15% over cap), with a clean WARP_METAL_BATCH=0
    # control — chaotic tail, not divergence. p50/p95 drift stats are now reported per
    # run (see TRAJECTORY_DRIFT_HISTORY) so future recalibration is evidence-based.
    "basic_shapes": (100, 0.15),  # primitives strict pre-contact; mesh contacts → cap
    # KEP-0001 M3.3: SolverMuJoCo(metal) single-robot row (R1 recon D.2). mjw
    # solver internals carry atomics; cap starts None (strict-only) and follows
    # the drift-history flow once the drift shape is characterized.
    "robot_cartpole": (90, None),  # 1 world, solver=mujoco via TRAJECTORY_ARGS
}
# R1 recon D.2: the example modules no longer share the basic/ join — cartpole
# lives under robot/ (and any future case carries its own module path).
TRAJECTORY_MODULES = {
    "basic_pendulum": "newton.examples.basic.example_basic_pendulum",
    "basic_shapes": "newton.examples.basic.example_basic_shapes",
    "robot_cartpole": "newton.examples.robot.example_robot_cartpole",
}
# Per-case Example(args) namespace deltas over the XPBD-facing base
# (_traj_example_args): cartpole defaults world_count=100 and solver=mujoco —
# the single-robot row must pass both explicitly (R1 recon D.2).
TRAJECTORY_ARGS = {
    "robot_cartpole": {"solver": "mujoco", "world_count": 1},
}
TRAJECTORY_RTOL = 1e-5
TRAJECTORY_ATOL = 1e-6
TRAJECTORY_NPZ = Path(__file__).resolve().parent / "trajectory_golden.npz"
# Append-only per-run drift records (one JSON line per case per compare run) —
# the p50/p95 drift monitor's backing store for future cap recalibration.
TRAJECTORY_DRIFT_HISTORY = Path(__file__).resolve().parent / "trajectory_drift_history.jsonl"


def _traj_example_args(device: str, extra: dict | None = None):
    import argparse

    ns = argparse.Namespace(
        test=False,
        viewer="null",
        device=device,
        benchmark=False,
        headless=True,
        stage=None,
        num_frames=110,
        render_fps=None,
        solver="xpbd",
        scale=1.0,
    )
    if extra:
        for key, val in extra.items():
            setattr(ns, key, val)
    return ns


def _traj_collect(device: str) -> dict[str, "np.ndarray"]:
    """Run each example for N steps and snapshot state_0.body_q every step."""
    import numpy as np

    if os.environ.get("NEWTON_WARP_BACKEND") == "metal":
        import warp_metal as wp
    else:
        import warp as wp
    wp.set_device(device)

    import newton  # noqa: F401 — must import with the backend switch already set
    from newton.viewer import ViewerNull

    cases = {}
    for name, (n_steps, _cap) in TRAJECTORY_CASES.items():
        mod = importlib.import_module(TRAJECTORY_MODULES[name])
        inst = mod.Example(ViewerNull(), _traj_example_args(device, TRAJECTORY_ARGS.get(name)))
        n_bodies = len(inst.model.body_q)
        frames = [np.asarray(inst.state_0.body_q.numpy(), dtype=np.float64).copy()]
        for _ in range(n_steps):
            inst.step()
            frames.append(np.asarray(inst.state_0.body_q.numpy(), dtype=np.float64).copy())
        cases[name] = np.stack(frames)  # [n_steps+1, n_bodies, 7]
        print(f"  trajectory[{device}] {name}: {n_steps} steps, {n_bodies} bodies")
    return cases


def _append_drift_history(name, *, max_drift, drift_p50, drift_p95, tail_max, n_strict, n_steps, ok) -> None:
    """Record one drift-monitoring sample (per case, per compare run)."""
    record = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "case": name,
        "max": max_drift,
        "p50": drift_p50,
        "p95": drift_p95,
        "tail_max": tail_max,
        "n_strict": n_strict,
        "n_steps": n_steps,
        "ok": ok,
        "batch_env": os.environ.get("WARP_METAL_BATCH", ""),
    }
    with open(TRAJECTORY_DRIFT_HISTORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def run_trajectory() -> str:
    import numpy as np

    side = os.environ.get("NEWTON_WARP_BACKEND", "")
    if side != "metal":
        # golden phase: real warp, CPU
        try:
            import warp  # noqa: F401

            if "warp_metal" in sys.modules:
                raise ImportError
        except ImportError:
            sys.exit("error: trajectory golden phase requires real warp (unset NEWTON_WARP_BACKEND)")
        cases = _traj_collect("cpu")
        payload = {f"{k}::body_q": v for k, v in cases.items()}
        meta = {k: {"n_steps": TRAJECTORY_CASES[k][0], "n_bodies": int(v.shape[1])} for k, v in cases.items()}
        np.savez(TRAJECTORY_NPZ, **payload)
        (TRAJECTORY_NPZ.parent / "trajectory_golden_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"trajectory golden -> {TRAJECTORY_NPZ}")
        return "golden phase (cpu reference) written"

    # compare phase: shim, metal device
    try:
        golden = np.load(TRAJECTORY_NPZ)
    except FileNotFoundError:
        sys.exit(f"error: trajectory golden missing — run the cpu phase first ({TRAJECTORY_NPZ})")

    cases = _traj_collect("metal")
    lines = []
    n_pass = 0
    for name, (n_steps, cap) in TRAJECTORY_CASES.items():
        ref = golden[f"{name}::body_q"]
        got = cases[name]
        if got.shape != ref.shape:
            lines.append(f"- **{name}**: FAIL — shape {got.shape} vs golden {ref.shape}")
            continue
        diff = np.abs(got - ref)
        per_step_max = diff.reshape(n_steps + 1, -1).max(axis=1)
        strict_ok = per_step_max <= TRAJECTORY_ATOL + TRAJECTORY_RTOL * np.abs(ref).reshape(n_steps + 1, -1).max(axis=1)
        n_strict = int(strict_ok.sum())
        max_drift = float(per_step_max.max())
        # Drift-envelope monitoring (user-ratified 2026-11-15 alongside the cap
        # recalibration): p50/p95 over the per-step drift distribution plus the
        # contact-phase tail max (max over non-strict steps), logged per run to
        # TRAJECTORY_DRIFT_HISTORY so cap placement stays evidence-based.
        drift_p50 = float(np.percentile(per_step_max, 50))
        drift_p95 = float(np.percentile(per_step_max, 95))
        tail_steps = per_step_max[~strict_ok]
        tail_max = float(tail_steps.max()) if tail_steps.size else None
        if cap is None:
            ok = bool(strict_ok.all())
            verdict = "PASS（全步 strict）" if ok else "FAIL"
        else:
            # contact phase: every step must be either strict or within the cap
            over = per_step_max[~strict_ok] > cap
            ok = not bool(over.any())
            verdict = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        first_div = int(np.argmax(~strict_ok)) if not strict_ok.all() else "-"
        tail_txt = f"{tail_max:.3e}" if tail_max is not None else "-"
        lines.append(
            f"- **{name}**: {verdict} — steps {n_steps + 1}, bodies {got.shape[1]}; "
            f"strict {n_strict}/{n_steps + 1}（首个非严格步 {first_div}），"
            f"max|Δbody_q|={max_drift:.3e}，cap={cap if cap is not None else 'strict-only'}；"
            f"drift p50={drift_p50:.3e} p95={drift_p95:.3e} tail-max={tail_txt}"
        )
        _append_drift_history(
            name,
            max_drift=max_drift,
            drift_p50=drift_p50,
            drift_p95=drift_p95,
            tail_max=tail_max,
            n_strict=n_strict,
            n_steps=n_steps + 1,
            ok=ok,
        )
    total = len(TRAJECTORY_CASES)
    lines.append(f"")
    lines.append(f"**结果：{n_pass} / {total} 例程通过**（strict rtol={TRAJECTORY_RTOL} / atol={TRAJECTORY_ATOL}；"
                 f"atomic-order 漂移上限见各例程 cap——reducer 原子槽序差异的预期发散类，ROADMAP M1.7(c)）")
    # KEP-0001 M3.2 §6-1 (ii): capture replay==eager sub-count (R1 A.2) —
    # the headline 2/2 above is untouched. Fail-loud: any error surfaces as a
    # visible FAIL line, never a silent skip (guardrail A.4.4).
    lines.append("")
    try:
        lines.append(_run_trajectory_capture())
    except Exception as e:  # noqa: BLE001 — capture sub-mode fails loud into the section
        lines.append(f"- capture replay==eager: FAIL — {type(e).__name__}: {str(e)[:160]}")
    return "\n".join(lines)


def _traj_capture_frames(inst, n_steps) -> list:
    """Per-frame ``state_0.body_q`` snapshots kept at raw float32 (bitwise)."""
    import numpy as np

    frames = [np.asarray(inst.state_0.body_q.numpy(), dtype=np.float32).copy()]
    for _ in range(n_steps):
        inst.step()
        frames.append(np.asarray(inst.state_0.body_q.numpy(), dtype=np.float32).copy())
    return frames


def _run_trajectory_capture() -> str:
    """basic_pendulum eager-vs-replay, same device, per-frame bitwise.

    KEP-0001 M3.2 §6-1 (ii), R1 A.2: the existing 2/2 proves "metal replay ==
    CPU replay" (both sides' step() is graph replay); this sub-mode adds the
    same-device contrast — pass A constructs the Example and clears
    ``inst.graph`` so ``step()`` walks ``simulate()`` eagerly, pass B is a
    fresh instance on the default ``capture_launch`` replay path. Contact-free
    case → strict everywhere, and with no atomic-order divergence class in
    play the per-frame body_q comparison is BITWISE (R1 A.2 先 strict 位级；
    cap 机制只在原子序漂移类出现时启用). Runs under both batch modes
    (guardrail A.4.3). basic_shapes stays deferred: contact-phase atomic
    order voids bit equality (R1 A.2(ii)).
    """
    import numpy as np

    name = "basic_pendulum"
    n_steps = TRAJECTORY_CASES[name][0]
    mod = importlib.import_module(TRAJECTORY_MODULES[name])
    from newton.viewer import ViewerNull

    prev = os.environ.pop("WARP_METAL_BATCH", None)
    try:
        verdicts = {}
        for mode, env in (("batch", None), ("unbatched", "0")):
            if env is None:
                os.environ.pop("WARP_METAL_BATCH", None)
            else:
                os.environ["WARP_METAL_BATCH"] = env
            # pass A: eager — the capture graph built at init stays unused
            inst_e = mod.Example(ViewerNull(), _traj_example_args("metal"))
            inst_e.graph = None
            eager = _traj_capture_frames(inst_e, n_steps)
            del inst_e  # release the unused graph + its records before pass B
            # pass B: replay — fresh instance, default capture_launch path
            # (frame reads between replays are host-visible reads only, so
            # nothing eager rewrites the captured buffers — guardrail A.4.1)
            inst_r = mod.Example(ViewerNull(), _traj_example_args("metal"))
            replay = _traj_capture_frames(inst_r, n_steps)
            del inst_r
            verdicts[mode] = len(eager) == len(replay) and all(
                _bitwise_equal(e, r) for e, r in zip(eager, replay)
            )
    finally:
        if prev is None:
            os.environ.pop("WARP_METAL_BATCH", None)
        else:
            os.environ["WARP_METAL_BATCH"] = prev

    n_ok = 1 if all(verdicts.values()) else 0
    detail = ", ".join(f"{mode}: {'PASS' if verdicts[mode] else 'FAIL'}" for mode in ("batch", "unbatched"))
    return (
        f"- capture replay==eager（{name}，同设备逐帧 body_q 位级：pass A `graph=None` eager vs "
        f"pass B capture_launch 重放，batch + unbatched 两遍，{n_steps + 1} 帧/遍）: "
        f"**{n_ok}/1**（{detail}）"
    )


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
    ap.add_argument("--corpus-golden", type=Path, default=CORPUS_NPZ,
                    help="corpus golden npz (written by the Newton-venv golden phase)")
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
        # KEP-0001 M1.5 note: since the shim, the Newton venv also has
        # warp_metal installed; CORPUS_FORCE_GOLDEN pins this process as the
        # golden phase (warp_metal is never imported, so `import warp`
        # resolves to the real warp-lang below)
        force_golden = bool(os.environ.get("CORPUS_FORCE_GOLDEN"))
        is_golden = force_golden
        if not force_golden:
            try:
                import warp_metal  # noqa: F401 — compare side
            except ImportError:
                is_golden = True
        if is_golden:
            # Newton venv (no warp_metal import): this process IS the golden phase
            run_corpus_golden(args.corpus_golden)
            print(f"corpus golden -> {args.corpus_golden}")
            return
        results["corpus"] = run_corpus(args.corpus_golden)
        print(f"corpus: report section updated (see {args.report})")
    if args.layer in ("trajectory", "all"):
        res = run_trajectory()
        if res.startswith("golden phase"):
            # golden side: nothing to report from this process (like corpus)
            return
        results["trajectory"] = res
        print("trajectory: report section updated (see report)")
    write_report(args.report, SECTION_TITLES, results)


if __name__ == "__main__":
    main()
