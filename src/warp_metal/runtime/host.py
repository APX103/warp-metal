# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""warp-metal host runtime prototype (KEP-0001 M0.4).

Minimal bucket-D surface for the Phase 0 spike:
  - ``WarpArray``: one MTLBuffer (storageModeShared) + zero-copy numpy view
  - ``zeros``/``empty``/``ones``/``full`` array constructors (1D concrete dtypes)
  - ``launch``: emit MSL -> compile -> PSO cache -> synchronous dispatch
  - ``synchronize``

Everything is 1D; scalars pass through per-arg constant-space buffers
(MSL kernels cannot take scalar-value params). The full bucket-D surface
(devices, streams, 2d-4d arrays, copies, capture batching) is M1.2/M2.5.
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
)
from .. import native as _mm
from ..msl.emitter import MSLEmitter

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


class WarpArray:
    """MTLBuffer-backed array with a zero-copy numpy view."""

    def __init__(self, n: int, dtype=float32):
        self.n = int(n)
        self.dtype = dtype
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
        """Zero-copy view: (n,) for scalars, (n, k) for composite dtypes."""
        if not self.is_composite:
            return self.numpy
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
        return f"warp_metal.WarpArray(n={self.n}, dtype={getattr(self.dtype, '__name__', self.dtype)})"


def zeros(n: int, dtype=float32) -> WarpArray:
    arr = WarpArray(n, dtype)
    arr.numpy.fill(0)
    return arr


def empty(n: int, dtype=float32) -> WarpArray:
    return WarpArray(n, dtype)


def ones(n: int, dtype=float32) -> WarpArray:
    arr = WarpArray(n, dtype)
    arr.numpy.fill(1)
    return arr


def full(n: int, value, dtype=float32) -> WarpArray:
    arr = zeros(n, dtype)
    if arr.is_composite:
        arr.numpy_view()[:] = np.asarray(value, dtype=np.float32)
    else:
        arr.numpy.fill(value)
    return arr


def synchronize() -> None:
    _mm.synchronize()


def device_name() -> str:
    return _mm.device_name()


# ---------------------------------------------------------------------------
# launch: emit -> compile -> PSO cache -> synchronous dispatch
# ---------------------------------------------------------------------------

_PSO_LOCK = threading.Lock()
_PSO_CACHE: dict[int, tuple] = {}  # id(kernel) -> pso handle
_EMITTER: MSLEmitter | None = None


def _emitter() -> MSLEmitter:
    global _EMITTER
    if _EMITTER is None:
        from ..frontend import registry

        _EMITTER = MSLEmitter(registry.builtin_functions)
    return _EMITTER


def _pso_for(kernel_func):
    with _PSO_LOCK:
        pso = _PSO_CACHE.get(id(kernel_func))
        if pso is not None:
            return pso
        source = _emitter().emit(kernel_func)
        lib = _mm.library_create(source)
        fn = _mm.function_get(lib, kernel_func.__name__)
        pso = _mm.pso_create(fn)
        _mm.function_release(fn)
        _mm.library_release(lib)
        _PSO_CACHE[id(kernel_func)] = pso
        return pso


def _scalar_buffer(value) -> int:
    """One 4-byte constant-space buffer per scalar kernel param."""
    packed = np.asarray([value], dtype=np.float32)
    handle = _mm.buffer_create(packed.nbytes)
    dst = np.ctypeslib.as_array(
        ctypes.cast(_mm.buffer_ptr(handle), ctypes.POINTER(ctypes.c_float)),
        shape=(1,),
    )
    dst[:] = packed
    return handle


def launch(kernel_func, dim: int, inputs: list[WarpArray], scalars=None, device="metal"):
    """Spike launch: 1D dispatch over ``dim`` threads.

    ``inputs`` are WarpArrays; ``scalars`` are the kernel's scalar params in
    declaration order (buffered individually in the constant address space).
    Handles are passed in kernel-parameter order via the emitter layout.
    """
    if device != "metal":
        raise ValueError(f"spike runtime supports device='metal' only, got {device!r}")
    pso = _pso_for(kernel_func)
    layout = _emitter().param_layout(kernel_func)

    inputs = list(inputs)
    scalars = list(scalars or [])
    scalar_iter = iter(scalars)
    handles, temp = [], []
    for _name, is_array, _msl_t in layout:
        if is_array:
            handles.append(inputs.pop(0).buffer)
        else:
            h = _scalar_buffer(next(scalar_iter))
            handles.append(h)
            temp.append(h)
    if inputs or list(scalar_iter):
        raise ValueError("more launch arguments than kernel parameters")

    _mm.dispatch(pso, handles, int(dim))
    for h in temp:
        _mm.buffer_release(h)
