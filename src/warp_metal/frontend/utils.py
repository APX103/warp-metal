# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``wp.utils`` namespace stub.

Newton references ``wp.utils.segmented_sort_pairs`` / ``wp.utils.array_scan``
only from the kamino solver (not on the rigid-body MVP path). Any access
fails loudly with a clear pointer instead of silently missing.

``GraphColoringAlgorithm`` is vendored (upstream warp/_src/coloring.py):
a pure host-side IntEnum Newton references by type on the rigid path
(graph_coloring.py annotation); the coloring kernels themselves are not
on the MVP closure.
"""

from __future__ import annotations

import enum

import warp_metal as wp  # noqa: E402 — resolved mid-init like frontend/math_funcs.py
from .kernels import func


class GraphColoringAlgorithm(enum.IntEnum):
    """Graph coloring algorithm selection (upstream coloring.py)."""

    MCS = 0
    """Maximum Cardinality Search based coloring algorithm."""

    GREEDY = 1
    """Degree-ordered greedy coloring algorithm."""


@wp.kernel
def _array_scan_kernel(src: wp.array(dtype=wp.int32), dest: wp.array(dtype=wp.int32), inclusive: wp.int32):
    """Sequential exclusive/inclusive prefix sum (upstream array_scan).

    Single-thread scan: Newton's callers scan block-offset arrays of at most
    a few thousand entries, where an O(n) serial pass is negligible next to
    the launch overhead — and it is deterministic (upstream CUDA scans are
    deterministic too). In-place (dest is src) is safe: element i's source
    value is read before dest[i] is written.
    """
    if wp.tid() == 0:
        total = wp.int32(0)
        for i in range(src.shape[0]):
            v = src[i]
            if inclusive != 0:
                total = total + v
                dest[i] = total
            else:
                dest[i] = total
                total = total + v


def array_scan(src, dest, inclusive=True):
    """``wp.utils.array_scan`` (upstream warp/_src/utils.py).

    Device-side so it records into graph capture like any other launch.
    int32 only — Newton's rigid-body closure scans block-offset/range
    counters; other dtypes fail loud (extend if a corpus entry needs them).
    """
    from ..runtime.host import launch as _launch

    for arr, name in ((src, "src"), (dest, "dest")):
        dtype = getattr(arr, "dtype", None)
        if dtype is not wp.int32:
            raise NotImplementedError(
                f"wp.utils.array_scan supports wp.int32 only, got {name} dtype {dtype} "
                "(extend _array_scan_kernel if a corpus entry needs another dtype)"
            )
    _launch(_array_scan_kernel, dim=1, inputs=[src, dest, 1 if inclusive else 0])


@func
def quat_between_vectors(a: wp.vec3, b: wp.vec3) -> wp.quat:
    """Return the rotation quaternion that rotates the direction of ``a`` onto ``b``.

    Verbatim semantics of upstream warp/_src/utils.py (shortest-arc unit
    quaternion; opposite directions are not handled there either). Newton's
    builder calls this host-side for shape planes.
    """
    a = wp.normalize(a)
    b = wp.normalize(b)
    c = wp.cross(a, b)
    d = wp.dot(a, b)
    q = wp.quat(c[0], c[1], c[2], 1.0 + d)
    return wp.normalize(q)


def create_warp_function(func: object) -> tuple:
    """Create a Warp function from a Python function (upstream utils port).

    Newton's example test harness uses this to turn plain Python state
    predicates into launchable @wp.func objects. Uses the vendored codegen;
    lambdas/inline defs are re-extracted from source exactly like upstream.
    """
    import hashlib

    from . import codegen as _codegen
    from .kernels import Function
    from .types import Any as _Any

    def unique_name(code: str):
        return f"func_{hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]}"

    source = None
    argspec = _codegen.get_full_arg_spec(func)
    key = getattr(func, "__name__", None)
    if key is None:
        source, _, _ = _codegen.Adjoint.extract_function_source(func)
        key = unique_name(source)
    elif key == "<lambda>":
        body = _codegen.Adjoint.extract_lambda_source(func, only_body=True)
        if body is None:
            raise ValueError("Could not extract lambda source code")
        key = unique_name(body)
        source = f"def {key}({', '.join(argspec.args)}):\n  return {body}"
    else:
        key = getattr(func, "__qualname__", key)
        key = key.replace(".", "_").replace(" ", "_").replace("<", "").replace(">", "_")

    from .kernels import get_module as _get_module

    module = _get_module(f"map_{key}")
    fn = Function(
        func,
        key=key,
        namespace="",
        module=module,
        source=source,
        overloaded_annotations=dict.fromkeys(argspec.args, _Any),
    )
    return fn, module


def __getattr__(name):
    if name.startswith("__"):
        # let the import machinery probe dunders (__path__ etc.) normally
        raise AttributeError(name)
    raise NotImplementedError(
        f"wp.utils.{name} is not implemented in warp-metal "
        "(host scan/sort helpers arrive with the CPU fallback routing, M1.3+)"
    )
