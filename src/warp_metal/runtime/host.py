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
import functools
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
    bool as warp_bool,
    vec2f,
    vec3f,
    vec4f,
    quatf,
    mat22f,
    mat33f,
    mat44f,
    transformf,
    spatial_vectorf,
    vec2i,
    vec3i,
    vec4i,
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
    warp_bool: np.bool_,
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

# composite warp dtypes: contiguous runs of int32 (vec.h vec_t with int)
_COMPOSITE_INTS = {
    vec2i: 2,
    vec3i: 3,
    vec4i: 4,
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
    np.bool_: ctypes.c_bool,
}

# matrix dtypes carry two trailing numpy axes in warp's host view
_MAT_TRAILING = {
    mat22f: (2, 2),
    mat33f: (3, 3),
    mat44f: (4, 4),
}


def _struct_np_layout(dtype):
    """numpy structured dtype mirroring a warp @wp.struct's ctypes layout.

    The vendored struct decorator exposes ``_ctype`` (a ctypes Structure
    subclass) on instances. Field offsets are taken from ctypes itself and
    given to numpy explicitly — numpy's own alignment rule does not mirror
    ctypes padding (e.g. an 8-aligned array_t after an int32 field).
    """
    ctype_inst = getattr(dtype, "_ctype", None)
    if not isinstance(ctype_inst, ctypes.Structure):
        ctype_inst = getattr(dtype(), "_ctype", None)
    if not isinstance(ctype_inst, ctypes.Structure):
        raise NotImplementedError(f"dtype {dtype} not in spike subset")
    ctype = type(ctype_inst)
    names, formats, offsets = [], [], []
    for name, ftype in ctype._fields_:
        names.append(name)
        formats.append(_field_np_format(ftype))
        offsets.append(getattr(ctype, name).offset)
    np_layout = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                          "itemsize": ctypes.sizeof(ctype)})
    return np_layout, ctypes.sizeof(ctype), True


def _field_np_format(ftype):
    np_fmt = _CTYPES_NP.get(ftype)
    if np_fmt is not None:
        return np_fmt
    if issubclass(ftype, ctypes.Structure):
        return _ctype_to_np(ftype)
    if issubclass(ftype, ctypes.Array):
        elem = _CTYPES_NP.get(ftype._type_)
        if elem is None:
            raise NotImplementedError(f"struct field: {ftype} not mapped")
        return (elem, (ftype._length_,))
    raise NotImplementedError(f"struct field: {ftype} not mapped")


def _ctype_to_np(ctype):
    fields = []
    for name, ftype in ctype._fields_:
        fields.append((name, _field_np_format(ftype)))
    return np.dtype(fields)


_CTYPES_NP = {
    ctypes.c_float: np.float32,
    ctypes.c_double: np.float64,
    ctypes.c_int8: np.int8,
    ctypes.c_uint8: np.uint8,
    ctypes.c_int16: np.int16,
    ctypes.c_uint16: np.uint16,
    ctypes.c_int32: np.int32,
    ctypes.c_uint32: np.uint32,
    ctypes.c_int64: np.int64,
    ctypes.c_uint64: np.uint64,
    ctypes.c_bool: np.bool_,
}


def _dtype_layout(dtype):
    """(numpy dtype, element size in bytes, is_composite)."""
    if dtype is int:
        dtype = int32  # upstream wp.array(dtype=int) alias
    elif dtype is float:
        dtype = float32
    elif dtype is bool:
        dtype = warp_bool  # vendored bool type (1 byte, warp convention)
    np_t = _SCALAR_NP.get(dtype)
    if np_t is not None:
        return np_t, np.dtype(np_t).itemsize, False
    n = _COMPOSITE_FLOATS.get(dtype)
    if n is not None:
        return np.float32, 4 * n, True
    n = _COMPOSITE_INTS.get(dtype)
    if n is not None:
        return np.int32, 4 * n, True
    # generic wp.vector()/wp.matrix() type instances (e.g. vec3d): layout is
    # scalar dtype × shape product (host-side allocation; kernels still fail
    # loud at translation when the element type is not MSL-mappable)
    scalar = getattr(dtype, "_wp_scalar_type_", None)
    shape = getattr(dtype, "_shape_", None)
    if scalar is not None and shape:
        np_s = _SCALAR_NP.get(scalar)
        if np_s is None:
            raise NotImplementedError(f"dtype {dtype} not in spike subset")
        comps = 1
        for s in shape:
            comps *= int(s)
        return np_s, np.dtype(np_s).itemsize * comps, comps > 1
    try:
        return _struct_np_layout(dtype)
    except NotImplementedError:
        raise
    except Exception as error:
        raise NotImplementedError(f"dtype {dtype} not in spike subset") from error


def _normalize_shape(shape) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(s) for s in shape)


class _CallableArray(np.ndarray):
    """ndarray that also answers ``.numpy()`` — real warp's method form."""

    def __call__(self):
        return self


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
        self.offset_elems = 0
        self._owns_buffer = True
        self._buffer = _mm.buffer_create(self.n * self.item_bytes)
        self._ptr = _mm.buffer_ptr(self._buffer)
        self._build_numpy_view()

    @property
    def buffer(self):
        return self._buffer

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def requires_grad(self):
        return False

    @property
    def strides(self):
        """Row-major dense element strides (warp convention)."""
        out = []
        for i in range(len(self.shape)):
            n = 1
            for s in self.shape[i + 1 :]:
                n *= s
            out.append(n)
        return tuple(out)

    @property
    def is_contiguous(self):
        """Runtime arrays are always dense row-major (no strided views)."""
        return True

    @property
    def size(self):
        """Total element count (numpy-like), excluding composite sub-axes."""
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def ptr(self):
        """CPU-side shared-storage pointer (unified memory device address)."""
        return self._ptr

    def __ctype__(self):
        """C-representation for vendored frontend machinery (struct field
        setters, launch arg normalization). ``data`` is the CPU-side shared
        storage pointer; the device-side binding goes through the buffer
        handle at dispatch (see ``_launch_ssa``)."""
        from ..frontend.types import array_t

        return array_t(
            data=ctypes.c_uint64(self._ptr),
            grad=0,
            ndim=len(self.shape),
            shape=self.shape,
            strides=self.strides,
            flags=0,
        )

    def numpy_view(self):
        """Zero-copy host view: ``self.shape`` for scalars, ``self.shape +
        (comps,)`` for vec/quat/mat (mirrors real warp's shaped numpy data)."""
        return self.numpy

    def _build_numpy_view(self):
        """Recreate ``self.numpy`` over the current (possibly offset) pointer.

        Shaped like real warp's host view: ``self.shape`` for scalar dtypes,
        ``self.shape + (comps,)`` for vec/quat/mat, ``(n,)`` for structs.
        """
        np_t = np.dtype(self.np_dtype)
        total = self.n * self.item_bytes
        if np_t.names:
            raw = np.ctypeslib.as_array(
                ctypes.cast(self._ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(total,)
            )
            self.numpy = raw.view(self.np_dtype).view(_CallableArray)
            return
        c_t = _CTYPES_PTR[self.np_dtype]
        count = total // np_t.itemsize
        raw = np.ctypeslib.as_array(
            ctypes.cast(self._ptr, ctypes.POINTER(c_t)), shape=(count,)
        )
        arr = raw if c_t is raw.dtype else raw.view(self.np_dtype)
        if self.is_composite:
            comps = self.item_bytes // np_t.itemsize
            # warp host-view semantics: vec/quat → trailing comps axis;
            # matrices → two trailing axes (mat33: (n, 3, 3), so np.trace etc.
            # work on element reads)
            mshape = _MAT_TRAILING.get(self.dtype)
            target = tuple(self.shape) + (mshape if mshape else (comps,)) if self.n else ((0,) + (mshape if mshape else (comps,)))
            arr = arr.reshape(target)
        elif len(self.shape) > 1:
            arr = arr.reshape(self.shape) if self.n else arr.reshape(0)
        self.numpy = arr.view(_CallableArray)

    def _slice_view(self, start: int, length: int) -> "WarpArray":
        """Host-side array view (warp slicing): shares the base buffer at an
        element offset; the offset reaches device code via wp_arr_meta.offset."""
        view = WarpArray.__new__(WarpArray)
        view.device = self.device
        view.dtype = self.dtype
        view.shape = (length,) + tuple(self.shape[1:])
        view.n = 1
        for s in view.shape:
            view.n *= s
        view.np_dtype = self.np_dtype
        view.item_bytes = self.item_bytes
        view.is_composite = self.is_composite
        view.offset_elems = self.offset_elems + start
        view._buffer = self._buffer  # shared: release is owned by the base
        view._owns_buffer = False
        view._ptr = self._ptr + start * self.item_bytes
        view._build_numpy_view()
        return view

    def __getitem__(self, key):
        if isinstance(key, slice):
            if len(self.shape) == 0:
                raise TypeError("scalar array is not sliceable")
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                raise NotImplementedError("strided array slices are outside the warp-metal closure")
            if start < 0 or stop < start:
                raise IndexError(f"invalid array slice [{start}:{stop}] for shape {self.shape}")
            return self._slice_view(start, stop - start)
        if isinstance(key, tuple):
            # multi-dim element read (host copy via the numpy view)
            return self.numpy_view()[key]
        if isinstance(key, (int, np.integer)):
            if len(self.shape) > 1:
                return self.numpy_view()[key]
            return self.numpy[key]
        raise TypeError(f"unsupported array index {key!r}")

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)) and len(self.shape) == 1:
            self.numpy[key] = value
            return
        self.numpy_view()[key] = value

    def zero_(self):
        """In-place zero fill (warp convention method).

        Upstream this is a stream memset — captured by graph recording. The
        host memset here is likewise recorded so replay re-runs it (SolverXPBD
        re-zeros its atomic-accumulation deltas every iteration).
        """
        self.numpy[...] = 0
        _record_fill(self, 0.0)
        return self

    def fill_(self, value):
        """In-place constant fill (warp convention method; recorded like zero_)."""
        self.numpy[...] = value
        _record_fill(self, value)
        return self

    def assign(self, value) -> None:
        """Copy host data into the array (upstream ``array.assign``).

        Accepts another WarpArray or any host array-like; ``numpy.copyto``
        checks shape and casts dtype the same way upstream does. Storage is
        shared memory, so the write is immediately visible to in-flight and
        captured dispatches.
        """
        if isinstance(value, WarpArray):
            value = value.numpy
        np.copyto(self.numpy, value)

    def to_numpy(self):
        return self.numpy_view().copy()

    def release(self) -> None:
        if getattr(self, "_buffer", None) and getattr(self, "_owns_buffer", True):
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


# the runtime array is the native wp.array family on Metal (see
# frontend.types.register_runtime_array_family)
from ..frontend.types import array as _wp_array, register_runtime_array_family  # noqa: E402

register_runtime_array_family(WarpArray, _wp_array)


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


def full(shape, value, dtype=None, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    if dtype is None:
        # upstream infers the element type from the fill value (wp.full(n,
        # wp.vec3(...)) allocates a vec3 array); scalars default to float32
        dtype = type(value) if hasattr(value, "__len__") and not isinstance(value, (bytes, str)) else float32
        if dtype in (int,):
            dtype = int32
        elif dtype in (float,):
            dtype = float32
    arr = zeros(shape, dtype, device, requires_grad, pinned, retain_grad)
    if arr.is_composite:
        # vec/quat/mat fill: flatten the vector-protocol value componentwise
        # (scalar values broadcast over the components, like upstream)
        comps = arr.item_bytes // np.dtype(arr.np_dtype).itemsize
        if hasattr(value, "__len__"):
            row = np.asarray([value[j] for j in range(len(value))], dtype=arr.np_dtype)
        else:
            row = np.full(comps, value, dtype=arr.np_dtype)
        if len(row) != comps:
            raise ValueError(f"full: {len(row)} components for a {comps}-component dtype")
        arr.numpy_view()[...] = row
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
    if _CAPTURE_RECORDS is not None:
        # upstream stream memcpy: recorded, executed at replay (the record
        # holds both arrays, so neither buffer dies under the graph)
        _CAPTURE_RECORDS.append(("copy", arr, src, 0, 0, None))
    else:
        arr.numpy[:] = src.numpy
    return arr


def copy(dest, src, dest_offset=0, src_offset=0, count=None, stream=None):
    """Copy ``count`` elements (all when None) from ``src`` into ``dest``."""
    if dest.dtype != src.dtype:
        raise TypeError(f"copy dtype mismatch: dest {dest.dtype} vs src {src.dtype}")
    n = src.n - src_offset if count is None else int(count)
    if n < 0 or dest_offset + n > dest.n or src_offset + n > src.n:
        raise ValueError(f"copy out of bounds: count={n}, dest_offset={dest_offset}, src_offset={src_offset}")
    if _CAPTURE_RECORDS is not None:
        _CAPTURE_RECORDS.append(("copy", dest, src, dest_offset, src_offset, n))
        return
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
    if shape is None:
        _np_t, item_bytes_probe, composite_probe = _dtype_layout(dtype)
        if composite_probe and np_arr.ndim > 1:
            comps = item_bytes_probe // np.dtype(_np_t).itemsize
            # trailing numpy axes are the element components (vec3: (n, 3);
            # mat33: (n, 3, 3)) — collapse them (warp semantics)
            product, k = 1, 0
            for ax in reversed(np_arr.shape):
                product *= int(ax)
                k += 1
                if product >= comps:
                    break
            if product == comps:
                target_shape = tuple(np_arr.shape[: np_arr.ndim - k])
    out = _allocate(target_shape or (1,), dtype, device)
    np_t = np.dtype(out.np_dtype)
    if np_t.names:
        # struct element: each source element is a warp struct instance whose
        # ``_ctype`` is a ctypes structure mirroring the numpy layout byte for
        # byte — copy records verbatim (embedded wp.array fields keep their
        # shared-storage pointers, which are the device addresses)
        elements = arr if isinstance(arr, (list, tuple)) else None
        if elements is None:
            raise NotImplementedError("struct-dtype wp.array needs a list of struct instances")
        if len(elements) != out.n:
            raise ValueError(f"struct wp.array: {len(elements)} elements for shape {out.shape}")
        for i, inst in enumerate(elements):
            src = getattr(inst, "_ctype", None)
            if not isinstance(src, ctypes.Structure):
                raise TypeError(f"struct wp.array element {i}: not a warp struct instance")
            ctypes.memmove(out._ptr + i * out.item_bytes, ctypes.addressof(src), out.item_bytes)
        return out
    np_itemsize = np_t.itemsize
    comps = max(1, out.item_bytes // np_itemsize)
    total_scalars = out.n * comps
    flat = np_arr.reshape(-1)
    if flat.size != total_scalars:
        raise ValueError(
            f"from_numpy shape mismatch: numpy has {flat.size} scalars, "
            f"target {out.shape} of {getattr(dtype, '__name__', dtype)} needs {total_scalars}"
        )
    out.numpy[:total_scalars] = flat.reshape(out.numpy.shape)
    return out


def synchronize() -> None:
    _mm.synchronize()


class array:
    """Public ``wp.array``: runtime-backed construction (KEP-0001 M1.6).

    Newton's builder creates arrays host-side with data; the runtime
    WarpArray is the native Metal allocation (shared storage). Subscripting
    yields the vendored annotations and ``isinstance`` accepts both runtime
    and vendored instances, so kernel annotations and struct fields behave
    identically to upstream. ``ptr``/``capacity`` aliasing needs a native
    ``buffer_from_ptr`` (Phase 2) and fails loud; ``copy=False`` lands as a
    copy (Metal cannot alias non-page-aligned host memory — see __new__);
    grad/pinned flags stay Non-Goals.
    """

    def __new__(
        cls,
        data=None,
        dtype=None,
        shape=None,
        device=None,
        copy=True,
        pinned=False,
        requires_grad=False,
        retain_grad=False,
        grad=None,
        ptr=None,
        capacity=None,
        deleter=None,
        ndim=None,
    ):
        from ..frontend.types import array as _vendored_array

        if requires_grad or retain_grad or grad is not None or pinned:
            raise RuntimeError(
                "warp-metal does not support requires_grad/pinned/retain_grad allocation flags "
                "(adjoint codegen is a KEP-0001 Non-Goal)"
            )
        if ptr is not None or capacity is not None or deleter is not None:
            raise NotImplementedError(
                "wp.array ptr/capacity/deleter aliasing needs native buffer_from_ptr (Phase 2)"
            )
        if not copy:
            # Metal cannot alias arbitrary host memory (newBufferWithBytesNoCopy
            # requires page alignment), so copy=False lands as a copy. Newton's
            # copy=False sites hand freshly-built host arrays to the model and
            # never write through the source afterwards, so the data flow is
            # identical; divergence would require a post-wrap write through the
            # original numpy object, which none of the call sites do.
            pass
        if dtype is int:
            dtype = int32
        elif dtype is float:
            dtype = float32
        if data is None and shape is None:
            # annotation mode: wp.array(dtype=..., ndim=...) yields the vendored
            # type descriptor used in kernel/func signatures
            return _vendored_array(
                dtype=dtype if dtype is not None else float32,
                device=device,
                ndim=ndim,
            )
        if data is not None:
            return from_numpy(data, dtype=dtype, shape=shape, device=device)
        if shape is None:
            raise TypeError("wp.array() needs either data or shape")
        return zeros(shape, dtype=dtype if dtype is not None else float32, device=device)

    def __class_getitem__(cls, params):
        from ..frontend.types import _parse_array_subscript, array as _vendored_array

        return _parse_array_subscript(_vendored_array, params)

    def __instancecheck__(cls, instance):
        from ..frontend.types import array as _vendored_array

        return isinstance(instance, WarpArray) or isinstance(instance, _vendored_array)

    def __subclasscheck__(cls, subclass):
        from ..frontend.types import array as _vendored_array

        return issubclass(subclass, (WarpArray, _vendored_array))


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


# wp_mesh mirrors the MSL struct (msl/prelude.py): {device vec3f* data;
# wp_arr_meta m} for points + the int32 form for indices. Natural alignment
# matches the MSL layout (ptr @0/48, meta @8/56; total 96 bytes).
class _MeshArrayF(_ctypes.Structure):
    _fields_ = [
        ("data", _ctypes.c_uint64),
        ("m", _ArrMeta),
    ]


class _MeshArrayI(_ctypes.Structure):
    _fields_ = [
        ("data", _ctypes.c_uint64),
        ("m", _ArrMeta),
    ]


class _MeshEntry(_ctypes.Structure):
    _fields_ = [
        ("points", _MeshArrayF),
        ("indices", _MeshArrayI),
    ]


assert _ctypes.sizeof(_MeshEntry) == 96, _ctypes.sizeof(_MeshEntry)


# the device-side mesh table: one _MeshEntry per registered wp.Mesh (index ==
# mesh id; entry 0 is the null mesh). Kernels reaching wp.mesh_get bind it as
# `device wp_mesh* _wp_mesh_table`. Rebuilt (re-uploaded) whenever a mesh is
# created or its points are replaced — build-time-only work.
_MESH_TABLE: dict = {"handle": None, "size": 0}


def _mesh_meta(arr) -> _ArrMeta:
    """Meta for a WarpArray or a vendored frontend ``array_t`` (mesh data)."""
    if hasattr(arr, "buffer"):  # runtime WarpArray
        return _arr_meta(arr)
    # vendored array_t ctypes struct: data = GPU address, byte strides
    meta = _ArrMeta()
    meta.offset = 0
    meta.ndim = int(arr.ndim)
    for i in range(4):
        meta.shape[i] = int(arr.shape[i]) if i < arr.ndim else 0
        meta.strides[i] = int(arr.strides[i]) if i < arr.ndim else 0
    return meta


def _mesh_data_addr(arr) -> int:
    """GPU address for a WarpArray or a vendored frontend ``array_t``."""
    if hasattr(arr, "buffer"):
        return _mm.buffer_gpu_address(arr.buffer)
    return int(arr.data)


def _mesh_table_upload() -> int:
    """(Re)build the device mesh table; returns the buffer handle."""
    from ..frontend.registry import _CoreShim

    meshes = _CoreShim._HOST_MESHES
    n = _CoreShim._HOST_MESH_SEQ[0] + 1
    entries = (_MeshEntry * n)()
    for mid, rec in meshes.items():
        entry = entries[mid]
        points = rec["points"]
        indices = rec["indices"]
        if points is not None:
            entry.points.data = _mesh_data_addr(points)
            entry.points.m = _mesh_meta(points)
        if indices is not None:
            entry.indices.data = _mesh_data_addr(indices)
            entry.indices.m = _mesh_meta(indices)
    handle = _mm.buffer_create(_ctypes.sizeof(_MeshEntry) * n)
    dst = (_ctypes.c_char * (_ctypes.sizeof(_MeshEntry) * n)).from_address(_mm.buffer_ptr(handle))
    _ctypes.memmove(dst, _ctypes.byref(entries), _ctypes.sizeof(_MeshEntry) * n)
    if _MESH_TABLE["handle"] is not None:
        _mm.buffer_release(_MESH_TABLE["handle"])
    _MESH_TABLE["handle"] = handle
    _MESH_TABLE["size"] = n
    return handle


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


def _builtin_value_fields(msl_t: str) -> list[tuple] | None:
    """ctypes field list for a builtin warp value type (MSL prelude struct).

    Layouts match the prelude exactly: packed 4-byte members, row-major
    ``c[r][col]`` for mats, transform {p, q}, spatial {w, v}.
    """
    f, i = _ctypes.c_float, _ctypes.c_int32
    table = {
        "vec2f": [("x", f), ("y", f)],
        "vec3f": [("x", f), ("y", f), ("z", f)],
        "vec4f": [("x", f), ("y", f), ("z", f), ("w", f)],
        "quatf": [("x", f), ("y", f), ("z", f), ("w", f)],
        "vec2i": [("x", i), ("y", i)],
        "vec3i": [("x", i), ("y", i), ("z", i)],
        "vec4i": [("x", i), ("y", i), ("z", i), ("w", i)],
        "transformf": [("p", "vec3f"), ("q", "quatf")],
        "spatial_vectorf": [("w", "vec3f"), ("v", "vec3f")],
        "mat22f": [("c", (f * 2) * 2)],
        "mat33f": [("c", (f * 3) * 3)],
        "mat44f": [("c", (f * 4) * 4)],
    }
    return table.get(msl_t)


@functools.lru_cache(maxsize=None)
def _builtin_value_struct(msl_t: str):
    fields = _builtin_value_fields(msl_t)
    if fields is None:
        raise RuntimeError(f"launch: no ctypes mirror for builtin value type {msl_t!r}")
    resolved = []
    for fname, ftype in fields:
        if isinstance(ftype, str):  # nested builtin value type (vec3f, quatf, …)
            resolved.append((fname, _builtin_value_struct(ftype)))
        else:
            resolved.append((fname, ftype))
    return type(f"_{msl_t}_ctypes", (_ctypes.Structure,), {"_fields_": resolved})


def _params_struct(unit: KernelUnit):
    """Build the ctypes mirror of the MSL ``Params_<name>`` struct.

    Field order must match ``KernelUnit.kernel_src`` exactly: per arg in
    declaration order — scalar → value; struct → value + one ``wp_arr_meta``
    per split array field; array → one ``wp_arr_meta`` — then ``_dims``.
    """
    fields = []
    for a in unit.args:
        if a.kind == "scalar":
            if a.msl in _CTYPE_OF_SCALAR:
                fields.append((a.name, _CTYPE_OF_SCALAR[a.msl]))
            else:
                # builtin value type (vec/quat/transform/mat/spatial)
                fields.append((a.name, _builtin_value_struct(a.msl)))
        elif a.kind == "struct":
            fields.append((a.name, _user_struct_ctypes(a, unit)))
            for field, _elem_ctype in unit.struct_array_fields.get(a.ctype, []):
                fields.append((f"{a.name}__{field}_m", _ArrMeta))
        elif a.kind == "array":
            fields.append((f"{a.name}_m", _ArrMeta))
    fields.append(("_dims", _ArrMeta))

    class Params(_ctypes.Structure):
        _fields_ = fields or [("_dummy_", _ctypes.c_byte)]

    return Params


def _user_struct_ctypes(arg_spec, unit: KernelUnit):
    """ctypes.Structure class mirroring a vendored struct's field layout."""
    fields: list[tuple] = []
    for label, msl_t in unit.struct_fields[arg_spec.msl]:
        if msl_t in _CTYPE_OF_SCALAR:
            fields.append((label, _CTYPE_OF_SCALAR[msl_t]))
        elif msl_t in ("vec2i", "vec3i", "vec4i"):
            fields.append((label, _builtin_value_struct(msl_t)))
        elif _builtin_value_fields(msl_t) is not None:
            fields.append((label, _builtin_value_struct(msl_t)))
        elif msl_t in unit.struct_fields:  # nested user struct
            nested_spec = type("_Ref", (), {"msl": msl_t})
            fields.append((label, _user_struct_ctypes(nested_spec, unit)))
        else:
            raise RuntimeError(f"launch: struct field {label!r}: unsupported MSL type {msl_t!r}")
    return type(f"_{arg_spec.msl}_ctypes", (_ctypes.Structure,), {"_fields_": fields})


def _pack_value(dest, msl_t: str, value):
    """Pack a Python-side warp value into its ctypes mirror ``dest``.

    Vector-protocol values (vec/quat/transform/spatial: ``len(v)`` +
    ``v[j]``) flatten componentwise; mats flatten row-major linear
    (mat.h initializer_array order), matching the prelude layouts.
    """
    fields = _builtin_value_fields(msl_t)
    if fields is None:
        raise RuntimeError(f"launch: cannot pack MSL value type {msl_t!r}")
    if msl_t.startswith("mat"):
        comps = [value[r][c] for r in range(len(value)) for c in range(len(value[0]))]
    else:
        comps = [value[j] for j in range(len(value))]
    cur = [0]
    _fill_struct(dest, fields, comps, cur)
    if cur[0] != len(comps):
        raise RuntimeError(f"launch: {msl_t} packing consumed {cur[0]} of {len(comps)} components")


def _fill_struct(dest, fields, comps, cur) -> None:
    for fname, ftype in fields:
        if isinstance(ftype, str):  # nested builtin value type
            _fill_struct(getattr(dest, fname), _builtin_value_fields(ftype), comps, cur)
        elif isinstance(ftype, type) and issubclass(ftype, _ctypes.Array):
            _fill_array(getattr(dest, fname), comps, cur)
        else:
            setattr(dest, fname, comps[cur[0]])
            cur[0] += 1


def _fill_array(arr, comps, cur) -> None:
    for k in range(len(arr)):
        item = arr[k]
        if isinstance(item, _ctypes.Array):
            _fill_array(item, comps, cur)
        else:
            arr[k] = comps[cur[0]]
            cur[0] += 1


def _pack_user_struct(dest, msl_t: str, value, unit: KernelUnit):
    """Pack a vendored struct instance (attribute per ``.vars`` order)."""
    for label, f_msl in unit.struct_fields[msl_t]:
        field_value = getattr(value, label)
        if f_msl in _CTYPE_OF_SCALAR:
            setattr(dest, label, field_value)
        elif _builtin_value_fields(f_msl) is not None:
            _pack_value(getattr(dest, label), f_msl, field_value)
        elif f_msl in unit.struct_fields:
            _pack_user_struct(getattr(dest, label), f_msl, field_value, unit)
        else:
            raise RuntimeError(f"launch: struct field {label!r}: unsupported MSL type {f_msl!r}")


def _arr_meta(arr: WarpArray | None) -> _ArrMeta:
    if arr is None:
        # null array argument: zero shape/strides, ndim 0 (truthiness reads it)
        meta = _ArrMeta()
        meta.offset = 0
        meta.ndim = 0
        return meta
    shape = arr.shape or (1,)
    ndim = len(shape)
    item = arr.item_bytes
    strides = [0, 0, 0, 0]
    acc = item
    for i in range(ndim - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    meta = _ArrMeta()
    meta.offset = arr.offset_elems * item
    meta.ndim = ndim
    for i in range(4):
        meta.shape[i] = shape[i] if i < ndim else 0
        meta.strides[i] = strides[i]
    return meta


def _unit_for(kernel, block_dim: int | None = None):
    """Translate + compile + cache a PSO for a vendored Kernel.

    ``block_dim`` keys the cache alongside the kernel: a tiled translation
    bakes the block size into tid mapping and threadgroup reductions.
    """
    with _PSO_LOCK:
        cached = _PSO_CACHE.get((id(kernel), block_dim))
        if cached is not None:
            return cached
        opts = {"block_dim": block_dim} if block_dim is not None else None
        unit = translate_kernel(kernel, opts)
        source = unit.source()
        lib = _mm.library_create(source)
        fn = _mm.function_get(lib, unit.kernel.name)
        pso = _mm.pso_create(fn)
        _mm.function_release(fn)
        _mm.library_release(lib)
        entry = (pso, unit)
        _PSO_CACHE[(id(kernel), block_dim)] = entry
        return entry


def _concretize_generic_kernel(kernel, args):
    """Resolve a generic kernel overload from launch argument values.

    Upstream launch does the same dance (``kernel.is_generic`` → infer →
    ``_add_normalized_overload``); the vendored ``infer_argument_types``
    speaks upstream-array dialects, so the warp-metal ``WarpArray`` maps to
    the vendored lightweight array annotation here — the exact type object
    upstream's infer produces for a real array.
    """
    from ..frontend.types import _ArrayAnnotation, type_to_warp

    declared = [a.type for a in kernel.adj.args]

    def _compiler_type(v, decl):
        if v is None:
            # upstream allows passing None for array args: keep the declared
            # annotation type (the kernel guards with `if arr:` at runtime)
            return decl
        if isinstance(v, WarpArray):
            return _ArrayAnnotation(v.dtype, len(v.shape))
        if isinstance(v, bool):
            return type_to_warp(bool)
        if isinstance(v, int):
            return type_to_warp(int)
        if isinstance(v, float):
            return type_to_warp(float)
        if hasattr(v, "_cls"):  # vendored struct instance
            return v._cls
        return type(v)  # vector/matrix value types pass as classes

    fwd_types = [_compiler_type(v, d) for v, d in zip(args, declared)]
    return kernel._add_normalized_overload(fwd_types)


def _null_buffer() -> int:
    """Shared 1-byte buffer standing in for None array arguments (upstream
    binds a null device pointer; a real buffer keeps Metal bindings valid,
    and array truthiness reads the meta's ndim==0 instead)."""
    global _NULL_BUF
    if _NULL_BUF is None:
        _NULL_BUF = _mm.buffer_create(1)
    return _NULL_BUF


_NULL_BUF: int | None = None


def _launch_ssa(kernel, dim, merged, block_dim: int | None = None) -> None:
    """Launch a vendored Kernel through the Adjoint-SSA MSL pipeline."""
    if getattr(kernel, "is_generic", False):
        kernel = _concretize_generic_kernel(kernel, merged)
    from .. import host_fallback

    fb = host_fallback.get(kernel)
    if fb is not None:
        if block_dim is not None:
            raise RuntimeError(
                f"launch_tiled: kernel '{kernel.key}' has a CPU fallback registered but tiled "
                "ports are not supported (fallback ports take a flat thread count)"
            )
        if _CAPTURE_RECORDS is not None:
            # record the host port: replay re-runs it on the same buffers
            _CAPTURE_RECORDS.append(("fallback", kernel, dim, list(merged), fb))
            return
        # closure-external kernel with an explicit verified CPU fallback
        # (unified memory: same shared-storage buffers the device sees)
        fb(kernel, dim, merged)
        return
    pso, unit = _unit_for(kernel, block_dim)
    Params = _params_struct(unit)

    handles: list[int] = []
    # strong references to every WarpArray bound into this dispatch — under
    # graph capture the dispatch outlives the launching Python scope, and a
    # released buffer would leave the record reading freed memory (upstream
    # CUDA graphs keep stream-capture pool allocations alive for the graph's
    # lifetime; this is the equivalent guarantee)
    array_refs: list = []
    # device-pointer bindings in emitter order (name, buffer handle); when the
    # kernel uses the argument-buffer form these become GPU addresses in a
    # single shared buffer, in exactly the emitter's binding order
    device_bindings: list[tuple[str, int]] = []
    params = Params()
    values = list(merged)
    if len(values) != len(unit.args):
        raise ValueError(
            f"launch: kernel '{kernel.key}' expects {len(unit.args)} argument(s), got {len(values)}"
        )
    for a in unit.args:
        value = values.pop(0)
        if a.kind == "array":
            if value is not None and not isinstance(value, WarpArray):
                raise TypeError(f"kernel array parameter '{a.name}' got {type(value).__name__}, expected WarpArray")
            device_bindings.append((f"{a.name}_data", value.buffer if value is not None else _null_buffer()))
            if value is not None:
                array_refs.append(value)
            meta = _arr_meta(value)
            setattr(params, f"{a.name}_m", meta)
        elif a.kind == "scalar":
            if a.msl in _CTYPE_OF_SCALAR:
                setattr(params, a.name, value)
            else:  # builtin value type (vec/quat/transform/mat/spatial)
                _pack_value(getattr(params, a.name), a.msl, value)
        elif a.kind == "struct":
            _pack_user_struct(getattr(params, a.name), a.msl, value, unit)
            # split array fields: one buffer binding + one Params meta each
            for field, _elem_ctype in unit.struct_array_fields.get(a.ctype, []):
                field_arr = getattr(value, field)
                if field_arr is not None and not isinstance(field_arr, WarpArray):
                    raise TypeError(
                        f"kernel struct parameter '{a.name}' field '{field}' got "
                        f"{type(field_arr).__name__}, expected WarpArray"
                    )
                device_bindings.append(
                    (f"{a.name}__{field}_data", field_arr.buffer if field_arr is not None else _null_buffer())
                )
                if field_arr is not None:
                    array_refs.append(field_arr)
                setattr(params, f"{a.name}__{field}_m", _arr_meta(field_arr))
        else:
            raise RuntimeError(f"launch: unknown arg kind {a.kind!r}")
    if values:
        raise ValueError("more launch arguments than kernel parameters")

    # Metal caps plain buffer bindings at 30; beyond that the emitter switches
    # the kernel to an argument buffer of device pointers (tier 2)
    argbuf_handle = None
    if unit.uses_argbuf:
        argbuf_handle = _mm.buffer_create(8 * len(device_bindings))
        dst = (_ctypes.c_uint64 * len(device_bindings)).from_address(_mm.buffer_ptr(argbuf_handle))
        expected_order = [b for b, _ in unit.device_bindings]
        actual_order = [b for b, _ in device_bindings]
        if expected_order != actual_order:
            raise RuntimeError(
                f"launch: device binding order mismatch vs MSL: {actual_order} != {expected_order}"
            )
        for i, (_name, h) in enumerate(device_bindings):
            dst[i] = _mm.buffer_gpu_address(h)
        handles.append(argbuf_handle)
    else:
        handles.extend(h for _name, h in device_bindings)

    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    dims = tuple(int(d) for d in dims)
    threads = 1
    for d in dims:
        threads *= d
    if block_dim is not None:
        # tiled launch: `threads` is the BLOCK count — the grid is
        # threads*block_dim threads in block_dim-sized threadgroups
        threads *= block_dim
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

    # per-thread arena for in-device wp.zeros() (fixedarray stack arrays):
    # one slice per thread, bound LAST to match the emitter's buffer index.
    # Fresh zeros every launch — the kernel's zero-fill loop handles init.
    arena_handle = None
    if unit.arena_bytes_per_thread > 0:
        arena_handle = _mm.buffer_create(unit.arena_bytes_per_thread * threads)
        handles.append(arena_handle)

    # host-built mesh table for wp.mesh_get users (bound after the arena,
    # matching the emitter's buffer index; the handle is global and never
    # released, so graph records can reference it safely)
    if getattr(unit, "needs_mesh", False):
        handles.append(_mesh_table_upload() if _MESH_TABLE["handle"] is None else _MESH_TABLE["handle"])

    try:
        if _CAPTURE_RECORDS is not None:
            # graph recording: keep the dispatch alive for replay — the
            # params/argbuf temporaries are owned by the Graph now, and the
            # bound arrays are referenced by the record so their buffers
            # cannot be released under it
            owned = [handle] + ([argbuf_handle] if argbuf_handle is not None else [])
            if arena_handle is not None:
                owned.append(arena_handle)
            _CAPTURE_RECORDS.append(("metal", pso, list(handles), threads, block_dim, owned, array_refs))
            return
        import os as _os

        if _os.environ.get("WARP_METAL_DEBUG_LAUNCH"):
            print(
                f"[wmm-launch] kernel={getattr(kernel, 'key', kernel)} threads={threads} "
                f"block_dim={block_dim} n_buffers={len(handles)} mesh={getattr(unit, 'needs_mesh', False)}",
                flush=True,
            )
        if block_dim is not None:
            _mm.dispatch_tiled(pso, handles, threads, block_dim)
        else:
            _mm.dispatch(pso, handles, threads)
    finally:
        if _CAPTURE_RECORDS is None:
            _mm.buffer_release(handle)
            if argbuf_handle is not None:
                _mm.buffer_release(argbuf_handle)
            if arena_handle is not None:
                _mm.buffer_release(arena_handle)


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



def launch_tiled(
    kernel,
    dim=None,
    inputs=None,
    outputs=None,
    block_dim=None,
    device=None,
    stream=None,
    record_cmd=None,
    max_blocks=None,
    record_tape=None,
    adjoint=None,
):
    """Tiled launch (upstream-compatible signature, forward-only).

    ``dim`` is the BLOCK count (int, or a length-1 sequence); the kernel
    reads ``wp.tid()`` as ``(block_index, lane)`` with ``wp.block_dim()``
    folding to ``block_dim`` (default 256, matching upstream's default).
    The vendored parse consumes ``block_dim`` for tile shapes, and the
    emitter lowers tile_reduce to a barriered threadgroup tree reduction —
    which requires a power-of-two ``block_dim`` (upstream warp makes the
    same assumption for its register-tile layout).
    """
    if adjoint or record_tape:
        raise RuntimeError(
            "warp-metal does not support adjoint launches (adjoint codegen is a KEP-0001 Non-Goal)"
        )
    if device is not None and get_device(device).name != "metal":
        raise RuntimeError(
            f"launch_tiled device={device!r} is not available: the Metal backend dispatches on 'metal'"
        )
    if stream is not None or record_cmd is not None or max_blocks is not None:
        # accepted-for-parity in launch(); there is no batching here at all,
        # so tiled variants fail loud instead of ignoring silently
        raise RuntimeError("launch_tiled does not accept stream/record_cmd/max_blocks (synchronous dispatch)")
    if dim is None:
        raise TypeError("launch_tiled() missing required argument 'dim'")

    if block_dim is None:
        block_dim = 256  # upstream default (warp.config block dim)
    block_dim = int(block_dim)
    if block_dim & (block_dim - 1):
        raise ValueError(f"launch_tiled block_dim must be a power of two, got {block_dim}")

    if isinstance(dim, (tuple, list)):
        if len(dim) != 1:
            raise ValueError(
                f"launch_tiled dim must be an int or length-1 sequence (block count), got {dim!r}"
            )
        dim = int(dim[0])
    else:
        dim = int(dim)

    merged = list(inputs or []) + list(outputs or [])
    _launch_ssa(kernel, dim, merged, block_dim=block_dim)


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
        _launch_ssa(kernel, dim, merged, block_dim=block_dim)
        return

    # legacy spike path (M0.4 plain-function kernels)
    if _CAPTURE_RECORDS is not None:
        raise RuntimeError("capture of legacy spike-path kernels is not supported (record the SSA path)")
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
    """Replayable capture record.

    Capture records each launch (Metal dispatch or CPU-fallback port) in
    order; ``capture_launch`` re-dispatches every record synchronously.
    Recorded param/arg-buffer temporaries are owned (and released) by the
    graph, and each record holds strong references to the arrays it binds —
    their buffers stay alive for the graph's lifetime even if the launching
    scope's temporaries die (upstream stream-capture pool semantics).
    """

    def __init__(self, device=None, n_ops: int = 0):
        self.device = device
        self.n_ops = n_ops
        self._records: list = []
        self._owned: list[int] = []

    def __del__(self):
        try:
            for record in getattr(self, "_records", []):
                if record[0] == "metal":
                    for h in record[5]:
                        _mm.buffer_release(h)
            self._records = []
        except Exception:
            pass

    def __repr__(self):
        return f"warp_metal.Graph(device={self.device!r}, n_ops={self.n_ops})"


class ScopedTimer:
    """Upstream-compatible host timing context (wall-clock only).

    Mirrors upstream ``warp._src.utils.ScopedTimer``: ``active`` (second
    positional — Newton passes ``False``) disables the timer entirely,
    ``print`` gates the exit print, ``dict`` receives per-name elapsed
    lists. Warp's profiler plumbing (CUDA events, NVTX, cProfile) is out
    of scope; timing lands on this process's wall clock, which is exact
    for the synchronous Metal dispatch model.
    """

    enabled = True

    def __init__(
        self,
        name: str | None = None,
        active: bool = True,
        print: bool = True,  # noqa: A002 — upstream parameter name
        detailed: bool = False,
        dict: dict[str, list[float]] | None = None,  # noqa: A002 — upstream
        use_nvtx: bool = False,
        color: int | str = "rapids",
        synchronize: bool = False,
        cuda_filter: int = 0,
        report_func=None,
        skip_tape: bool = False,
    ):
        self.name = name or "unnamed"
        self.active = active
        self.print_enabled = print
        self.dict = dict
        self.elapsed = None
        self._start = None
        if detailed or cuda_filter or report_func is not None:
            raise RuntimeError(
                "ScopedTimer: detailed/cuda_filter/report_func profiling is not "
                "supported by warp-metal (KEP-0001 Non-Goal; warp profiling plumbing)"
            )

    def __enter__(self):
        if not (type(self).enabled and self.active):
            return self
        import time as _time

        self._start = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is None:
            return False
        import time as _time

        self.elapsed = _time.perf_counter() - self._start
        if self.dict is not None:
            self.dict.setdefault(self.name, []).append(self.elapsed)
        if self.print_enabled:
            print(f"{self.name} took {self.elapsed * 1000.0:.2f} ms")
        return False


class ScopedCapture:
    """Context manager mirroring upstream: ``with wp.ScopedCapture() as c:``.

    Passthrough (KEP-0001 MVP): the captured launches run immediately and
    ``capture.graph`` is an empty record; ``capture_launch`` replays nothing.
    """

    def __init__(self, device=None, stream=None, force_module_load=False, **kwargs):
        if kwargs:
            raise RuntimeError(f"ScopedCapture does not support {sorted(kwargs)} (KEP-0001 passthrough capture)")
        self.device = device
        self.stream = stream
        self.force_module_load = force_module_load
        self.graph = None

    def __enter__(self):
        capture_begin(device=self.device, stream=self.stream, force_module_load=self.force_module_load)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.graph = capture_end(device=self.device, stream=self.stream)
        return False


_CAPTURE_LOCK = threading.Lock()
_CAPTURE_DEPTH = 0
# active record list while capturing; None outside capture
_CAPTURE_RECORDS: list | None = None


def _record_fill(arr, value) -> None:
    """Record an in-place fill as a graph op (upstream stream-memset semantics).

    The record holds the array itself, so its buffer stays alive for the
    graph's lifetime. Host-only fills skipped outside capture.
    """
    if _CAPTURE_RECORDS is not None:
        _CAPTURE_RECORDS.append(("fill", arr, value))


def capture_begin(device=None, stream=None, force_module_load=False, external=None, apic=None, capture_mode=None):
    """Begin capture: launches are RECORDED, not executed (upstream graph
    semantics). Replay re-dispatches each recorded op synchronously."""
    global _CAPTURE_DEPTH, _CAPTURE_RECORDS
    if external is not None or apic:
        raise RuntimeError("warp-metal does not support external/apic capture (CUDA-graph machinery)")
    with _CAPTURE_LOCK:
        if _CAPTURE_DEPTH == 0:
            _CAPTURE_RECORDS = []
        else:
            raise RuntimeError("nested capture is not supported by warp-metal")
        _CAPTURE_DEPTH += 1


def capture_end(device=None, stream=None) -> Graph:
    """End capture; returns the recorded graph."""
    global _CAPTURE_DEPTH, _CAPTURE_RECORDS
    with _CAPTURE_LOCK:
        _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)
        records = _CAPTURE_RECORDS
        _CAPTURE_RECORDS = None
    graph = Graph(get_device(device).name if device is not None else None, n_ops=len(records) if records else 0)
    graph._records = records or []
    return graph


def capture_launch(graph, stream=None):
    """Replay a captured graph: re-dispatch every recorded operation in
    order (synchronous, like the original launches)."""
    if isinstance(graph, Graph):
        for record in getattr(graph, "_records", []):
            kind = record[0]
            if kind == "metal":
                _, pso, handles, threads, block_dim, _owned, _refs = record
                if block_dim is not None:
                    _mm.dispatch_tiled(pso, handles, threads, block_dim)
                else:
                    _mm.dispatch(pso, handles, threads)
            elif kind == "fill":
                _, arr, value = record
                arr.numpy[...] = value
            elif kind == "copy":
                _, dest, src, dest_offset, src_offset, count = record
                n = src.n - src_offset if count is None else int(count)
                dest.numpy[dest_offset : dest_offset + n] = src.numpy[src_offset : src_offset + n]
            elif kind == "fallback":
                _, kernel, dim, merged, fb = record
                fb(kernel, dim, merged)
            else:
                raise RuntimeError(f"capture_launch: unknown record kind {kind!r}")
        return
    raise RuntimeError(f"capture_launch: cannot replay {type(graph).__name__}")


def capture_if(condition, on_true=None, on_false=None, stream=None, **kwargs):
    """Upstream-faithful eager branch outside graph capture.

    Upstream semantics: outside a CUDA/APIC capture (the only mode this
    backend has), ``capture_if`` reads the condition element and invokes the
    chosen callback eagerly with ``**kwargs``. Graph objects as callbacks
    (replay inside a recorded conditional) remain the M2.5 boundary and fail
    loud.
    """
    if on_true is None and on_false is None:
        return
    if not isinstance(condition, WarpArray) or condition.numpy.dtype != np.int32:
        raise TypeError("Condition must be a Warp array of int32 with a single element")
    condition_value = bool(condition.numpy[0])
    if condition_value:
        if on_true is None:
            return
        if callable(on_true):
            on_true(**kwargs)
        else:
            raise TypeError("on_true must be a Callable or a Graph (Graph replay: Phase 2, M2.5)")
    else:
        if on_false is None:
            return
        if callable(on_false):
            on_false(**kwargs)
        else:
            raise TypeError("on_false must be a Callable or a Graph (Graph replay: Phase 2, M2.5)")


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
