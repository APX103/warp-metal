# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Phase 1 target closure analysis (KEP-0001 M1.3 pre-work).

Run under the Newton venv:

    uv run python ~/work/warp-metal/tools/phase1_closure.py [example ...]

Hooks wp.launch to record every kernel actually executed while running the
examples headless, then walks each kernel's source AST transitively
(kernels + @wp.func, including warp builtins) to produce:

  - the executed-kernel set (module, qualname)
  - the transitive wp.* builtin call set with occurrence counts
  - constructs used per kernel (If/For/While/structs/wp.func calls/arrays)

Output: JSON + human summary printed; written to the KEP dir as
phase1-closure.json.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Volumes/ssd/main_link/work/newton")
KEP_DIR = REPO / "keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon"
OUT_PATH = KEP_DIR / "phase1-closure.json"

import warp as wp  # noqa: E402  (Newton venv)

LAUNCHED: list[object] = []
_orig_launch = wp.launch


def _recording_launch(*args, **kwargs):
    func = args[0] if args else kwargs.get("func") or kwargs.get("kernel")
    LAUNCHED.append(func)
    return _orig_launch(*args, **kwargs)


wp.launch = _recording_launch
# some call sites use wp.capture_launch; count those too
CAPTURED: list[object] = []
_orig_capture = getattr(wp, "capture_launch", None)
if _orig_capture is not None:
    def _recording_capture(func, *args, **kwargs):
        CAPTURED.append(func)
        return _orig_capture(func, *args, **kwargs)
    wp.capture_launch = _recording_capture


def run_example(name: str) -> None:
    import runpy
    import sys as _sys

    print(f"--- running {name} (cpu, null viewer) ---")
    argv_backup = _sys.argv
    _sys.argv = ["newton.examples", name, "--viewer", "null", "--device", "cpu"]
    try:
        runpy.run_module("newton.examples", run_name="__main__", alter_sys=True)
    except SystemExit:
        pass
    finally:
        _sys.argv = argv_backup


def symbol_closure(func) -> tuple[Counter, set]:
    """Transitive (wp.* builtin counts, construct set) for one kernel/func."""
    counts: Counter = Counter()
    constructs: set = set()
    seen_funcs: set[int] = set()

    def visit_function(fn, depth):
        if id(fn) in seen_funcs:
            return
        seen_funcs.add(id(fn))
        try:
            src = Path(sys.modules[fn.__module__].__file__).read_text()
        except Exception:
            return
        tree = ast.parse(src)
        target = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
                target = node
                break
        if target is None:
            return
        for node in ast.walk(target):
            if isinstance(node, ast.Attribute):
                root = node.value
                if isinstance(root, ast.Name) and root.id in ("wp", "warp"):
                    counts[node.attr] += 1
            elif isinstance(node, (ast.If, ast.IfExp)):
                constructs.add("if")
            elif isinstance(node, (ast.For, ast.While)):
                constructs.add("loop")
            elif isinstance(node, ast.Subscript):
                constructs.add("index")
        # recurse into user wp.func defined in the same file
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_decorator = any(
                    isinstance(d, ast.Attribute) and d.attr == "func" for d in node.decorator_list
                )
                if has_decorator and node.name in {n.attr for n in ast.walk(target) if isinstance(n, ast.Attribute)}:
                    try:
                        fobj = getattr(sys.modules[fn.__module__], node.name, None)
                        if callable(fobj):
                            visit_function(fobj, depth + 1)
                    except Exception:
                        pass

    visit_function(func, 0)
    return counts, constructs


# builtins that live in the vendored registry (frontend) vs module-level API
REGISTRY_BUILTINS = set()


def main() -> int:
    examples = sys.argv[1:] or ["basic_pendulum", "basic_shapes"]
    for name in examples:
        run_example(name)

    kernels = []
    seen = set()
    for k in LAUNCHED + CAPTURED:
        name = getattr(k, "__qualname__", None) or getattr(k, "__name__", None)
        if name is None or getattr(k, "__module__", None) is None:
            continue  # Graph replays and other non-kernel launch targets
        key = (k.__module__, name)
        if key in seen:
            continue
        seen.add(key)
        kernels.append(k)

    total: Counter = Counter()
    per_kernel = []
    all_constructs: Counter = Counter()
    for k in kernels:
        counts, constructs = symbol_closure(k)
        total.update(counts)
        for c in constructs:
            all_constructs[c] += 1
        per_kernel.append({
            "module": k.__module__,
            "qualname": k.__qualname__,
            "wp_symbols": dict(counts),
            "constructs": sorted(constructs),
        })

    result = {
        "examples": examples,
        "n_launch_calls": len(LAUNCHED),
        "n_capture_calls": len(CAPTURED),
        "n_unique_kernels": len(kernels),
        "kernels": per_kernel,
        "builtin_closure": {k: v for k, v in total.most_common()},
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n=== {len(kernels)} unique kernels executed ===")
    for k in per_kernel:
        print(f"  {k['module'].split('.')[-1]}:{k['qualname']}  constructs={k['constructs']}")
    print(f"\n=== transitive wp.* closure ({len(total)} symbols) ===")
    for sym, n in total.most_common():
        print(f"  {n:5d}  {sym}")
    print(f"\nconstruct usage across kernels: {dict(all_constructs)}")
    print(f"written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
