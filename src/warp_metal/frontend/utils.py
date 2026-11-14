# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``wp.utils`` namespace: host scan/sort utilities + vendored helpers.

``array_scan``, ``radix_sort_pairs`` / ``segmented_sort_pairs`` (numpy on the
shared-memory views: MTLStorageModeShared means a host edit is immediately
device-visible) and ``array_cast`` implement the upstream contracts
(warp/_src/utils.py) that Newton's steel contact paths call.
``GraphColoringAlgorithm`` is vendored (upstream warp/_src/coloring.py):
a pure host-side IntEnum Newton references by type on the rigid path
(graph_coloring.py annotation); the coloring kernels themselves are not
on the MVP closure. Anything else — ``runlength_encode`` /
``graph_coloring_*`` (consumers implicit_mpm/VBD, out of the rigid
closure, Phase 3) — fails loudly through ``__getattr__`` with a clear
pointer instead of silently missing.
"""

from __future__ import annotations

import enum
import operator

import numpy as np

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


# ---------------------------------------------------------------------------
# host sorts: upstream wp.utils.radix_sort_pairs / segmented_sort_pairs
# (numpy on the shared-memory views; upstream's own CPU path is the same kind
# of host LSD radix — warp/native/sort.cpp — so results match bit-for-bit)
# ---------------------------------------------------------------------------


def _float_radix_key(bits):
    """Stereopsis float→unsigned radix-key mapping (warp/native/sort.cpp:261-275):
    ``mask = (unsigned)(-(int)(i >> (w-1))) | (1 << (w-1)); return i ^ mask``.

    Monotonic w.r.t. numeric order except the upstream bit-pattern quirks,
    reproduced exactly: -0.0 maps one step below +0.0, and NaN payloads order
    by bit pattern (positive NaNs above +inf, negative below -inf), not
    numerically.
    """
    w = 8 * bits.dtype.itemsize
    sign = (bits >> bits.dtype.type(w - 1)) != 0
    mask = np.where(sign, bits.dtype.type((1 << w) - 1), bits.dtype.type(0))
    return bits ^ (mask | bits.dtype.type(1 << (w - 1)))


def _bit_range_radix_keys(keys_flat, dtype, begin_bit, end_bit):
    """Unsigned radix keys restricted to bits ``[begin_bit, end_bit)`` — the
    exact integers upstream's 16-bit-per-pass LSD radix ranks
    (warp/native/sort.cpp:118-163). Signed keys are sign-flipped
    (sort.cpp:208/237), floats stereopsis-mapped (sort.cpp:283/298),
    unsigned kept (sort.cpp:222/252). A stable ascending sort of these is
    mathematically equivalent to upstream's pass sequence for any bit subset.
    """
    if dtype in (wp.int32, wp.uint32, wp.float32):
        bits = keys_flat.view(np.uint32)
    else:
        bits = keys_flat.view(np.uint64)
    if dtype in (wp.int32, wp.int64):
        rk = bits ^ bits.dtype.type(1 << (8 * bits.dtype.itemsize - 1))
    elif dtype in (wp.float32, wp.float64):
        rk = _float_radix_key(bits)
    else:
        rk = bits
    if begin_bit:
        rk = rk >> rk.dtype.type(begin_bit)
    nbits = end_bit - begin_bit
    if nbits < 8 * rk.dtype.itemsize:
        rk = rk & rk.dtype.type((1 << nbits) - 1)
    return rk


def radix_sort_pairs(keys, values, count, begin_bit=0, end_bit=None):
    """``wp.utils.radix_sort_pairs`` (upstream warp/_src/utils.py:166-276).

    Stable ascending in-place sort of ``keys[:count]`` with ``values[:count]``
    riding along, ranked by key bits ``[begin_bit, end_bit)`` (Newton's
    contact_sort.py passes ``end_bit=`` subsets). Host-side over the runtime
    array's shared-memory numpy view: one stable argsort of the radix keys,
    the same permutation upstream's LSD radix produces. Runtime ``wp.array``
    (WarpArray) inputs; the ``keys``/``values`` storage must be at least
    ``2*count`` elements, but only ``[:count]`` is read and written.

    Deviations from upstream, all benign:
    - the ``[count:2*count]`` scratch half is left untouched (upstream
      ping-pongs through it, sort.cpp:127-170; the observable contract only
      covers ``[:count]``);
    - float keys order by the stereopsis bit mapping like upstream, so -0.0
      sorts below +0.0 and NaN payloads order by bit pattern (see
      ``_float_radix_key``);
    - not recorded into graph capture (a host-side edit, unlike
      array_scan's launch): fine for the steel closure, whose sort regions
      only run eagerly under metal (solver_steel._run_propagation gates its
      capture_if on device.is_cuda).

    RuntimeError on device mismatch, insufficient storage, unsupported dtypes,
    non-contiguous arrays, or an invalid bit range — upstream messages
    verbatim.
    """
    from .types import type_repr, type_size_in_bytes

    if keys.device != values.device:
        raise RuntimeError(f"Keys and values array storage devices do not match ({keys.device} vs {values.device})")

    # upstream: count==0 returns before storage checks (utils.py:189-190);
    # negative counts reach the C host radix, which no-ops for n <= 0
    # (sort.cpp:114-116) — a silent no-op either way
    if count <= 0:
        return

    if keys.size < 2 * count or values.size < 2 * count:
        raise RuntimeError("Keys and values array storage must be large enough to contain 2*count elements")

    value_size = type_size_in_bytes(values.dtype)
    if value_size not in (4, 8):
        raise RuntimeError(
            f"Unsupported keys and values data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
        )

    if not keys.is_contiguous:
        raise RuntimeError(
            f"radix_sort_pairs() requires a contiguous keys array, got non-contiguous keys with "
            f"data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
        )

    if not values.is_contiguous:
        raise RuntimeError(
            f"radix_sort_pairs() requires a contiguous values array, got non-contiguous values with "
            f"data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
        )

    if keys.dtype in (wp.int32, wp.uint32, wp.float32):
        key_bit_width = 32
    elif keys.dtype in (wp.int64, wp.uint64, wp.float64):
        key_bit_width = 64
    else:
        # upstream reaches the same error through the dispatch fall-through
        # (utils.py:256-259)
        raise RuntimeError(
            f"Unsupported keys and values data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
        )

    if end_bit is None:
        end_bit = key_bit_width

    if not isinstance(begin_bit, int) or not isinstance(end_bit, int):
        raise RuntimeError("begin_bit and end_bit must be integers")

    if not (0 <= begin_bit <= end_bit <= key_bit_width):
        raise RuntimeError(f"Invalid radix sort bit range [{begin_bit}, {end_bit}) for {key_bit_width}-bit keys")

    if begin_bit == end_bit:
        return

    keys_flat = keys.numpy.reshape(-1)
    values_flat = values.numpy.reshape(-1)
    prefix = keys_flat[:count]
    perm = np.argsort(_bit_range_radix_keys(prefix, keys.dtype, begin_bit, end_bit), kind="stable")
    prefix[...] = prefix[perm]
    # permute raw payload bytes so any 4/8-byte value dtype (scalar, vec2f,
    # int64, ...) rides along element-wise — upstream's SortPayload<4|8>
    # (sort.cpp:17-19) does the same bytewise move
    vbytes = values_flat.view(np.uint8).reshape(-1, value_size)
    vbytes[:count] = vbytes[:count][perm]


def segmented_sort_pairs(keys, values, count, segment_start_indices, segment_end_indices=None):
    """``wp.utils.segmented_sort_pairs`` (upstream warp/_src/utils.py:279-475).

    Stable ascending in-place sort of each half-open ``[start, end)`` segment
    of ``keys[:count]`` (int32 or float32 keys, int32 values riding along).
    With ``segment_end_indices=None`` adjacent ``segment_start_indices``
    entries define ``N-1`` segments (broad_phase_sap.py's shape); elements
    outside segment ranges are left untouched; invalid bounds raise
    ValueError (upstream's direct-CPU behavior, error string from
    sort.cpp:315-320). Host-side over the shared-memory views — same capture
    caveat as :func:`radix_sort_pairs` (documented there).

    RuntimeError for the upstream conditions: device mismatch, insufficient
    (>=2*count) storage, non-contiguous arrays, non-1-D/non-int32 segment
    index arrays, mismatched segment-array lengths, unsupported dtypes.
    """
    from .types import type_repr

    # count validation first (upstream utils.py:345-356)
    if isinstance(count, bool):
        raise TypeError("count must be an integer")

    try:
        count = operator.index(count)
    except TypeError as e:
        raise TypeError("count must be an integer") from e

    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count > (1 << 31) - 1:
        raise ValueError(f"count must not exceed {(1 << 31) - 1}, got {count}")

    segment_devices_match = segment_start_indices.device == keys.device
    if segment_end_indices is not None:
        segment_devices_match = segment_devices_match and segment_end_indices.device == keys.device
    if keys.device != values.device or not segment_devices_match:
        raise RuntimeError("Keys, values, and segment index array storage devices must match")

    if count == 0:
        return

    if keys.size < 2 * count or values.size < 2 * count:
        raise RuntimeError("Array storage must be large enough to contain 2*count elements")

    if not keys.is_contiguous:
        raise RuntimeError("segmented_sort_pairs() requires a contiguous keys array")

    if not values.is_contiguous:
        raise RuntimeError("segmented_sort_pairs() requires a contiguous values array")

    if segment_start_indices.dtype != wp.int32:
        raise RuntimeError("segment_start_indices array must be of type int32")

    if segment_start_indices.ndim != 1:
        raise RuntimeError("segment_start_indices must be one-dimensional")

    if not segment_start_indices.is_contiguous:
        raise RuntimeError("segmented_sort_pairs() requires a contiguous segment_start_indices array")

    if segment_end_indices is None:
        num_segments = max(0, segment_start_indices.size - 1)
        if num_segments == 0:
            return
    else:
        if segment_end_indices.dtype != wp.int32:
            raise RuntimeError("segment_end_indices array must be of type int32")

        if segment_end_indices.ndim != 1:
            raise RuntimeError("segment_end_indices must be one-dimensional")

        if not segment_end_indices.is_contiguous:
            raise RuntimeError("segmented_sort_pairs() requires a contiguous segment_end_indices array")

        if segment_start_indices.size != segment_end_indices.size:
            raise RuntimeError(
                "segment_start_indices and segment_end_indices must have the same size "
                f"({segment_start_indices.size} vs {segment_end_indices.size})"
            )

        num_segments = segment_start_indices.size

        if num_segments == 0:
            return

    # dtype dispatch (upstream utils.py:431-451): int32/int32 or float32/int32
    if keys.dtype == wp.int32 and values.dtype == wp.int32:
        float_keys = False
    elif keys.dtype == wp.float32 and values.dtype == wp.int32:
        float_keys = True
    else:
        raise RuntimeError(f"Unsupported data type: {type_repr(keys.dtype)}")

    keys_flat = keys.numpy.reshape(-1)
    values_flat = values.numpy.reshape(-1)
    starts_np = segment_start_indices.numpy.reshape(-1)
    if segment_end_indices is None:
        seg_starts = starts_np[:-1]
        seg_ends = starts_np[1:]
    else:
        seg_starts = starts_np
        seg_ends = segment_end_indices.numpy.reshape(-1)

    # upstream validates every range before sorting anything (sort.cpp:312-321)
    # and surfaces the C error string as ValueError (utils.py:452-453)
    valid = (seg_starts >= 0) & (seg_ends >= seg_starts) & (seg_ends <= count)
    if not bool(valid.all()):
        i = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            f"Warp sort error: Invalid segment range at index {i}: "
            f"[{int(seg_starts[i])}, {int(seg_ends[i])}) for count {count}"
        )

    for i in range(num_segments):
        start = int(seg_starts[i])
        end = int(seg_ends[i])
        if start == end:
            continue
        seg = keys_flat[start:end]
        if float_keys:
            rk = _float_radix_key(seg.view(np.uint32))
        else:
            rk = seg.view(np.uint32) ^ np.uint32(0x80000000)
        perm = np.argsort(rk, kind="stable")
        seg[...] = seg[perm]
        values_flat[start:end] = values_flat[start:end][perm]


def array_cast(in_array, out_array, count=None):
    """``wp.utils.array_cast`` (upstream warp/_src/utils.py:972-1056).

    Element-wise cast of ``in_array`` into ``out_array`` over the first
    ``count`` elements (all of them when ``count`` is None). Same-ndim,
    same-dtype-shape inputs — Newton's mesh.py builds 1-D int64→int32 index
    buffers this way. Upstream's flatten-to-scalar fallback for mismatched
    ndim/dtype-shape (utils.py:1001-1031) is outside the closure and fails
    loud with NotImplementedError. Cast semantics are C-style truncation
    (numpy ``astype``), matching upstream's per-element ``dest.dtype(src)``
    kernel for integer narrowing and float→int rounding toward zero.

    RuntimeError for device mismatch, partial multi-dimensional casts,
    mismatched multi-dim shapes, an out-of-range count, or a negative count
    (upstream fails in the launch there; loud here).
    """
    if in_array.device != out_array.device:
        raise RuntimeError(f"Array storage devices do not match ({in_array.device} vs {out_array.device})")

    in_data_shape = getattr(in_array.dtype, "_shape_", ())
    out_data_shape = getattr(out_array.dtype, "_shape_", ())

    if in_array.ndim != out_array.ndim or in_data_shape != out_data_shape:
        raise NotImplementedError(
            "wp.utils.array_cast requires matching ndim and dtype shape "
            "(upstream flatten-to-scalar fallback warp/_src/utils.py:1001-1031 is "
            "outside the warp-metal closure)"
        )

    if count is None:
        count = in_array.size

    if count < 0:
        # upstream errors inside wp.launch for a negative dim; fail loud here
        raise RuntimeError(f"Cast count must be non-negative, got {count}")

    if in_array.ndim != 1 and count < in_array.size:
        raise RuntimeError("Partial cast is not supported for arrays with more than one dimension")

    if in_array.ndim > 1 and in_array.shape != out_array.shape:
        raise RuntimeError(f"Array shapes do not match ({in_array.shape} vs {out_array.shape})")

    if count > in_array.size or count > out_array.size:
        raise RuntimeError(
            f"Cast count ({count}) exceeds the input size ({in_array.size}) or output size ({out_array.size})"
        )

    comps = 1
    for s in in_data_shape:
        comps *= int(s)

    in_flat = in_array.numpy.reshape(-1)
    out_flat = out_array.numpy.reshape(-1)
    n = count * comps
    out_flat[:n] = in_flat[:n].astype(out_flat.dtype)


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
