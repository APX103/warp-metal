# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""warp-metal host runtime (KEP-0001 M1.2, bucket D).

Built on the M0.4 prototype:
  - ``WarpArray``: one MTLBuffer (storageModeShared) + zero-copy numpy view,
    now with shape (1d-4d flattened) and device tracking
  - array constructors with upstream signatures (shape tuple, device kwarg),
    ``*_like`` variants, ``copy``/``clone``/``from_numpy``
  - ``launch`` with the upstream signature (adjoint kwargs fail loudly)
  - capture passthrough (``capture_begin``/``capture_end``/``capture_launch``)
  - ``init``/``synchronize``/device module

Arrays live in unified memory regardless of the allocating device, so a
``"cpu"`` array is directly usable by Metal kernels — the foundation for
per-builtin CPU fallback routing (M1.3). Whole-kernel CPU execution (LLVM
path) is not wired yet; launching on "cpu" fails loudly.
"""

from __future__ import annotations

import ctypes
import threading

import numpy as np

from ..frontend.types import (
    float32,
    float16,
    int32,
    uint32,
    int8,
    uint8,
    int16,
    uint16,
    int64,
    uint64,
    vec2f,
    vec3f,
    vec4f,
    quatf,
    mat22f,
    mat33f,
    mat44f,
    transformf,
    spatial_vectorf,
)
from .. import native as _mm
from ..msl.emitter import MSLEmitter
from .devices import get_device

_SCALAR_NP = {
    float32: np.float32,
    float16: np.float16,
    int32: np.int32,
    uint32: np.uint32,
    int8: np.int8,
    uint8: np.uint8,
    int16: np.int16,
    uint16: np.uint16,
    int64: np.int64,
    uint64: np.uint64,
}

# composite warp dtypes: contiguous runs of float32
_COMPOSITE_FLOATS = {
    vec2f: 2,
    vec3f: 3,
    vec4f: 4,
    quatf: 4,
    mat22f: 4,
    mat33f: 9,
    mat44f: 16,
    transformf: 7,   # {vec3 p, quat q} imaginary-first quat
    spatial_vectorf: 6,  # {vec3 w, vec3 v}
}

_CTYPES_PTR = {
    np.float32: ctypes.c_float,
    np.float16: ctypes.c_uint16,   # view later
    np.int32: ctypes.c_int32,
    np.uint32: ctypes.c_uint32,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_uint8,
    np.int16: ctypes.c_int16,
    np.uint16: ctypes.c_uint16,
    np.int64: ctypes.c_int64,
    np.uint64: ctypes.c_uint64,
}


def _dtype_layout(dtype):
    """(numpy scalar dtype, element size in bytes, is_composite)."""
    np_t = _SCALAR_NP.get(dtype)
    if np_t is not None:
        return np_t, np.dtype(np_t).itemsize, False
    n = _COMPOSITE_FLOATS.get(dtype)
    if n is not None:
        return np.float32, 4 * n, True
    raise NotImplementedError(f"dtype {dtype} not in spike subset")


def _normalize_shape(shape) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(s) for s in shape)


class WarpArray:
    """MTLBuffer-backed array with a zero-copy numpy view.

    ``shape`` may be ``()`` (zero-dim scalar buffer), an int, or a tuple of
    up to 4 ints; the backing buffer is the flattened product in row-major
    order (warp layout). ``device`` is recorded for dispatch validation but
    allocation is device-independent (storageModeShared unified memory).
    """

    def __init__(self, n, dtype=float32, shape=None, device=None, pinned=False, requires_grad=False, retain_grad=False):
        if requires_grad or retain_grad or pinned:
            # Adjoint/pinned allocation is a KEP-0001 Non-Goal: fail loud
            # rather than silently ignoring the flag (black-box principle).
            raise RuntimeError(
                "warp-metal does not support requires_grad/pinned/retain_grad allocation flags "
                "(adjoint codegen is a KEP-0001 Non-Goal)"
            )
        self.device = get_device(device).name
        self.dtype = dtype
        self.shape = _normalize_shape(shape) if shape is not None else _normalize_shape(n)
        self.n = 1
        for s in self.shape:
            self.n *= s
        self.np_dtype, self.item_bytes, self.is_composite = _dtype_layout(dtype)
        self._buffer = _mm.buffer_create(self.n * self.item_bytes)
        self._ptr = _mm.buffer_ptr(self._buffer)
        c_t = _CTYPES_PTR[self.np_dtype]
        count = self.n * self.item_bytes // np.dtype(self.np_dtype).itemsize
        raw = np.ctypeslib.as_array(
            ctypes.cast(self._ptr, ctypes.POINTER(c_t)), shape=(count,)
        )
        self.numpy = raw.view(self.np_dtype) if c_t is not raw.dtype else raw

    @property
    def buffer(self):
        return self._buffer

    def numpy_view(self):
        """Zero-copy view: self.shape for scalar dtypes, (n, k) for composites."""
        if not self.is_composite:
            return self.numpy.reshape(self.shape)
        return self.numpy.reshape(self.n, -1)

    def to_numpy(self):
        return self.numpy_view().copy()

    def release(self) -> None:
        if getattr(self, "_buffer", None):
            _mm.buffer_release(self._buffer)
            self._buffer = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __len__(self):
        return self.n

    def __repr__(self):
        return (
            f"warp_metal.WarpArray(shape={self.shape}, dtype={getattr(self.dtype, '__name__', self.dtype)}, "
            f"device={self.device!r})"
        )


def _allocate(shape, dtype, device, requires_grad=False, pinned=False, retain_grad=False) -> WarpArray:
    return WarpArray(shape, dtype, shape=shape, device=device, requires_grad=requires_grad, pinned=pinned, retain_grad=retain_grad)


def zeros(shape, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    arr = _allocate(shape, dtype, device, requires_grad, pinned, retain_grad)
    arr.numpy.fill(0)
    return arr


def empty(shape, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    return _allocate(shape, dtype, device, requires_grad, pinned, retain_grad)


def ones(shape, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    arr = _allocate(shape, dtype, device, requires_grad, pinned, retain_grad)
    arr.numpy.fill(1)
    return arr


def full(shape, value, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    arr = zeros(shape, dtype, device, requires_grad, pinned, retain_grad)
    if arr.is_composite:
        arr.numpy_view()[:] = np.asarray(value, dtype=np.float32)
    else:
        arr.numpy.fill(value)
    return arr


def _like_kwargs(src, device, requires_grad, pinned, retain_grad):
    if requires_grad or pinned or retain_grad:
        raise RuntimeError(
            "warp-metal does not support requires_grad/pinned/retain_grad allocation flags "
            "(adjoint codegen is a KEP-0001 Non-Goal)"
        )
    return src.shape, src.dtype, device if device is not None else src.device


def zeros_like(src, device=None, requires_grad=False, pinned=False, retain_grad=False):
    return zeros(*_like_kwargs(src, device, requires_grad, pinned, retain_grad))


def empty_like(src, device=None, requires_grad=False, pinned=False, retain_grad=False):
    return empty(*_like_kwargs(src, device, requires_grad, pinned, retain_grad))


def ones_like(src, device=None, requires_grad=False, pinned=False, retain_grad=False):
    return ones(*_like_kwargs(src, device, requires_grad, pinned, retain_grad))


def full_like(src, value, device=None, requires_grad=False, pinned=False, retain_grad=False):
    return full(src.shape, value, src.dtype, device)


def clone(src, device=None, requires_grad=False, pinned=False, retain_grad=False):
    arr = _allocate(*_like_kwargs(src, device, requires_grad, pinned, retain_grad))
    arr.numpy[:] = src.numpy
    return arr


def copy(dest, src, dest_offset=0, src_offset=0, count=None, stream=None):
    """Copy ``count`` elements (all when None) from ``src`` into ``dest``."""
    if dest.dtype != src.dtype:
        raise TypeError(f"copy dtype mismatch: dest {dest.dtype} vs src {src.dtype}")
    n = src.n - src_offset if count is None else int(count)
    if n < 0 or dest_offset + n > dest.n or src_offset + n > src.n:
        raise ValueError(f"copy out of bounds: count={n}, dest_offset={dest_offset}, src_offset={src_offset}")
    dest.numpy[dest_offset : dest_offset + n] = src.numpy[src_offset : src_offset + n]


def from_numpy(arr, dtype=None, shape=None, device=None, requires_grad=False, retain_grad=False):
    """Create a WarpArray from a numpy array.

    Copies the data into a fresh MTLBuffer (shared storage). A zero-copy
    wrap of foreign memory needs a native ``buffer_from_ptr`` (Phase 2).
    """
    if requires_grad or retain_grad:
        raise RuntimeError("warp-metal does not support grad flags (adjoint is a KEP-0001 Non-Goal)")
    np_arr = np.ascontiguousarray(arr)
    if dtype is None:
        # map the numpy dtype back to a warp dtype
        for wtype, np_t in _SCALAR_NP.items():
            if np_t == np_arr.dtype:
                dtype = wtype
                break
        else:
            raise NotImplementedError(f"from_numpy: numpy dtype {np_arr.dtype} not mapped")
    target_shape = tuple(np_arr.shape) if shape is None else _normalize_shape(shape)
    out = _allocate(target_shape or (1,), dtype, device)
    expected = int(np.prod(target_shape)) if target_shape else 1
    flat = np_arr.reshape(-1)
    if flat.size * np.dtype(out.np_dtype).itemsize * (out.item_bytes // np.dtype(out.np_dtype).itemsize) != out.n * out.item_bytes or flat.size != expected:
        raise ValueError(f"from_numpy shape mismatch: numpy has {flat.size} elements, target {target_shape} needs {expected}")
    out.numpy[:expected] = flat
    return out


def synchronize() -> None:
    _mm.synchronize()


def device_name() -> str:
    return _mm.device_name()


# ---------------------------------------------------------------------------
# launch: Adjoint-SSA translate -> compile -> PSO cache -> synchronous dispatch
# (KEP-0001 M1.3; the M0.4 spike emitter remains for plain-function kernels)
# ---------------------------------------------------------------------------

_PSO_LOCK = threading.Lock()
_PSO_CACHE: dict[int, tuple] = {}  # id(kernel) -> (pso handle, KernelUnit)
_EMITTER: MSLEmitter | None = None


def clear_pso_cache() -> None:
    """Drop all compiled kernel pipelines (used by wp.clear_kernel_cache)."""
    with _PSO_LOCK:
        _PSO_CACHE.clear()
    global _EMITTER
    _EMITTER = None


def _emitter() -> MSLEmitter:
    global _EMITTER
    if _EMITTER is None:
        from ..frontend import registry

        _EMITTER = MSLEmitter(registry.builtin_functions)
    return _EMITTER


def _kernel_emit_target(kernel):
    """(plain func, key) the spike emitter consumes for a kernel argument.

    Accepts a vendored ``Kernel`` record (M1.2+) or a plain decorated
    function (M0.4 spike path).
    """
    func = getattr(kernel, "func", kernel)
    key = getattr(kernel, "key", None)
    name = func.__name__ if key is None else key.rsplit("::", 1)[-1]
    return func, name


def _pso_for(kernel):
    func, name = _kernel_emit_target(kernel)
    with _PSO_LOCK:
        pso = _PSO_CACHE.get(id(func))
        if pso is not None:
            return pso
        source = _emitter().emit(func)
        lib = _mm.library_create(source)
        fn = _mm.function_get(lib, name)
        pso = _mm.pso_create(fn)
        _mm.function_release(fn)
        _mm.library_release(lib)
        _PSO_CACHE[id(func)] = pso
        return pso


# ---- M1.3 Adjoint-SSA pipeline -------------------------------------------

import ctypes as _ctypes

from ..msl.translate import KernelUnit, translate_kernel

# wp_arr_meta mirrors the MSL struct: int offset, ndim, shape[4], strides[4]
# (strides in bytes, warp native array.h semantics)
class _ArrMeta(_ctypes.Structure):
    _fields_ = [
        ("offset", _ctypes.c_int32),
        ("ndim", _ctypes.c_int32),
        ("shape", _ctypes.c_int32 * 4),
        ("strides", _ctypes.c_int32 * 4),
    ]


_CTYPE_OF_SCALAR = {
    "float": _ctypes.c_float,
    "half": _ctypes.c_uint16,
    "char": _ctypes.c_int8,
    "uchar": _ctypes.c_uint8,
    "short": _ctypes.c_int16,
    "ushort": _ctypes.c_uint16,
    "int": _ctypes.c_int32,
    "int64_t": _ctypes.c_int64,
    "uint": _ctypes.c_uint32,
    "uint64_t": _ctypes.c_uint64,
    "bool": _ctypes.c_bool,
}


def _params_struct(unit: KernelUnit):
    """Build the ctypes mirror of the MSL ``Params_<name>`` struct."""
    fields = []
    for a in unit.args:
        if a.kind == "scalar":
            try:
                fields.append((a.name, _CTYPE_OF_SCALAR[a.msl]))
            except KeyError:
                raise RuntimeError(f"launch: scalar param {a.name} has unsupported MSL type {a.msl}")
        elif a.kind == "struct":
            fields.append((a.name, _struct_ctypes(a)))
        elif a.kind == "array":
            fields.append((f"{a.name}_m", _ArrMeta))
    fields.append(("_dims", _ArrMeta))

    class Params(_ctypes.Structure):
        _fields_ = fields or [("_dummy_", _ctypes.c_byte)]

    return Params


def _struct_ctypes(arg_spec):
    """ctypes.Structure class mirroring a vendored struct's field layout."""
    import warp_metal.frontend.codegen as _codegen

    raise RuntimeError(
        "launch: struct kernel params are not wired yet (M1.3 corpus scope: scalar/array params)"
    )


def _arr_meta(arr: WarpArray) -> _ArrMeta:
    shape = arr.shape or (1,)
    ndim = len(shape)
    item = arr.item_bytes
    strides = [0, 0, 0, 0]
    acc = item
    for i in range(ndim - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    meta = _ArrMeta()
    meta.offset = 0
    meta.ndim = ndim
    for i in range(4):
        meta.shape[i] = shape[i] if i < ndim else 0
        meta.strides[i] = strides[i]
    return meta


def _unit_for(kernel):
    """Translate + compile + cache a PSO for a vendored Kernel."""
    with _PSO_LOCK:
        cached = _PSO_CACHE.get(id(kernel))
        if cached is not None:
            return cached
        unit = translate_kernel(kernel)
        source = unit.source()
        lib = _mm.library_create(source)
        fn = _mm.function_get(lib, unit.kernel.name)
        pso = _mm.pso_create(fn)
        _mm.function_release(fn)
        _mm.library_release(lib)
        entry = (pso, unit)
        _PSO_CACHE[id(kernel)] = entry
        return entry


def _launch_ssa(kernel, dim, merged) -> None:
    """Launch a vendored Kernel through the Adjoint-SSA MSL pipeline."""
    pso, unit = _unit_for(kernel)
    Params = _params_struct(unit)

    handles: list[int] = []
    params = Params()
    values = list(merged)
    for a in unit.args:
        value = values.pop(0)
        if a.kind == "array":
            if not isinstance(value, WarpArray):
                raise TypeError(f"kernel array parameter '{a.name}' got {type(value).__name__}, expected WarpArray")
            handles.append(value.buffer)
            meta = _arr_meta(value)
            setattr(params, f"{a.name}_m", meta)
        elif a.kind == "scalar":
            setattr(params, a.name, value)
        else:
            raise RuntimeError(f"launch: struct param '{a.name}' not wired yet (M1.3 fail loud)")
    if values:
        raise ValueError("more launch arguments than kernel parameters")

    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    dims = tuple(int(d) for d in dims)
    threads = 1
    for d in dims:
        threads *= d
    params._dims.offset = 0
    params._dims.ndim = len(dims)
    for i in range(4):
        d = dims[i] if i < len(dims) else 0
        params._dims.shape[i] = d
        params._dims.strides[i] = 0

    handle = _mm.buffer_create(_ctypes.sizeof(params))
    dst = (_ctypes.c_char * _ctypes.sizeof(params)).from_address(_mm.buffer_ptr(handle))
    _ctypes.memmove(dst, _ctypes.byref(params), _ctypes.sizeof(params))
    handles.append(handle)

    try:
        _mm.dispatch(pso, handles, threads)
    finally:
        _mm.buffer_release(handle)


def _scalar_buffer(value) -> int:
    """One 4-byte constant-space buffer per scalar kernel param."""
    packed = np.asarray([value], dtype=np.float32)
    handle = _mm.buffer_create(packed.nbytes)
    dst = np.ctypeslib.as_array(
        _ctypes.cast(_mm.buffer_ptr(handle), _ctypes.POINTER(_ctypes.c_float)),
        shape=(1,),
    )
    dst[:] = packed
    return handle


def launch(
    kernel,
    dim=None,
    inputs=None,
    outputs=None,
    adj_inputs=None,
    adj_outputs=None,
    device=None,
    stream=None,
    adjoint=None,
    record_tape=None,
    record_cmd=None,
    max_blocks=None,
    block_dim=None,
):
    """Launch a kernel (upstream-compatible signature, forward-only).

    ``adj_*``/``adjoint``/``record_tape`` fail loudly (adjoint is a KEP
    Non-Goal); ``stream``/``record_cmd``/``max_blocks``/``block_dim`` are
    accepted for API parity (dispatch is synchronous, batching is Phase 2).
    ``dim`` may be an int or a tuple (multi-dim grids flatten for now; the
    emitter only reads ``tid`` — multi-dim indexing is M1.3 scope). Scalar
    kernel params are passed interleaved in ``inputs``/``outputs`` in
    declaration order, exactly like upstream warp.
    """
    if adj_inputs is not None or adj_outputs is not None or adjoint or record_tape:
        raise RuntimeError(
            "warp-metal does not support adjoint launches (adjoint codegen is a KEP-0001 Non-Goal); "
            "use enable_backward=False / forward-only launches"
        )

    if device is not None and get_device(device).name != "metal":
        # Whole-kernel CPU execution (LLVM path) is not wired (see module docstring).
        raise RuntimeError(
            f"launch device={device!r} is not available yet: the Metal backend dispatches on 'metal'; "
            "'cpu' execution arrives with the CPU fallback routing"
        )

    if dim is None:
        raise TypeError("launch() missing required argument 'dim'")

    merged = list(inputs or []) + list(outputs or [])

    if hasattr(kernel, "adj"):  # vendored Kernel (M1.2+): Adjoint-SSA pipeline
        _launch_ssa(kernel, dim, merged)
        return

    # legacy spike path (M0.4 plain-function kernels)
    if isinstance(dim, (tuple, list)):
        dim = int(np.prod([int(d) for d in dim]))
    dim = int(dim)

    pso = _pso_for(kernel)
    layout = _emitter().param_layout(_kernel_emit_target(kernel)[0])

    handles, temp = [], []
    for _name, is_array, _msl_t in layout:
        if is_array:
            arg = merged.pop(0)
            if not isinstance(arg, WarpArray):
                raise TypeError(f"kernel array parameter got {type(arg).__name__}, expected WarpArray")
            handles.append(arg.buffer)
        else:
            value = merged.pop(0)
            h = _scalar_buffer(value)
            handles.append(h)
            temp.append(h)
    if merged:
        raise ValueError("more launch arguments than kernel parameters")

    _mm.dispatch(pso, handles, dim)
    for h in temp:
        _mm.buffer_release(h)


# ---------------------------------------------------------------------------
# capture passthrough (KEP-0001: MVP 直通 — dispatch is synchronous, so
# "captured" kernels run immediately; the returned graph is an empty record)
# ---------------------------------------------------------------------------


class Graph:
    """Opaque capture record.

    With synchronous passthrough capture, launches between capture_begin and
    capture_end have already executed, so the record is legitimately empty
    and replay is a no-op — never silently wrong results.
    """

    def __init__(self, device=None, n_ops: int = 0):
        self.device = device
        self.n_ops = n_ops

    def __repr__(self):
        return f"warp_metal.Graph(device={self.device!r}, n_ops={self.n_ops})"


_CAPTURE_LOCK = threading.Lock()
_CAPTURE_DEPTH = 0


def capture_begin(device=None, stream=None, force_module_load=False, external=None, apic=None, capture_mode=None):
    """Begin capture (passthrough: kernels run immediately)."""
    global _CAPTURE_DEPTH
    if external is not None or apic:
        raise RuntimeError("warp-metal does not support external/apic capture (CUDA-graph machinery)")
    with _CAPTURE_LOCK:
        _CAPTURE_DEPTH += 1


def capture_end(device=None, stream=None) -> Graph:
    """End capture; returns an empty graph record (ops already executed)."""
    global _CAPTURE_DEPTH
    with _CAPTURE_LOCK:
        _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)
    return Graph(get_device(device).name if device is not None else None, n_ops=0)


def capture_launch(graph, stream=None):
    """Replay a captured graph.

    The passthrough capture model executes kernels eagerly, so the captured
    graph is empty and replay is an honest no-op. Replaying a foreign graph
    object fails loudly.
    """
    if isinstance(graph, Graph):
        return
    raise RuntimeError(f"capture_launch: cannot replay {type(graph).__name__} (warp-metal graphs are capture-passthrough records)")


def capture_if(condition, on_true=None, on_false=None, stream=None, **kwargs):
    """Conditional graph nodes are CUDA-graph machinery — fail loudly."""
    raise NotImplementedError("wp.capture_if is not supported by warp-metal (CUDA-graph construct)")


def capture_while(condition, while_body=None, stream=None, **kwargs):
    """While-loop graph nodes are CUDA-graph machinery — fail loudly."""
    raise NotImplementedError("wp.capture_while is not supported by warp-metal (CUDA-graph construct)")


def init() -> None:
    """Validate the vendored frontend sentinel and lazily probe devices.

    Mirrors ``wp.init()`` as an idempotent bootstrap (safe to call many
    times, as Newton does).
    """
    from ..compat import check_vendored_version

    check_vendored_version()
    from . import devices as _devices

    _devices.is_metal_available()
