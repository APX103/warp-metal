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
  - graph capture (``capture_begin``/``capture_end``/``capture_launch``):
    launches inside a capture are recorded, not executed; ``capture_launch``
    replays the records in order, batching consecutive kernel records plus
    eligible copies / zero fills into one Metal command buffer per run
    (kernels as compute sections, copies/zero-fills as blit sections),
    pre-marshaled once at ``capture_end`` (M2.6)
  - ``init``/``synchronize``/device module

Eager dispatch model (M2.6): launches ENCODE into one persistent command
buffer (one process-lifetime device + queue, native side) and execute at
the next explicit flush — ``flush()``/``synchronize()``, and implicitly at
every host-visible read of array data (``.numpy``/``.numpy_view()``/
``__getitem__`` element reads/``to_numpy``/``copy``/``clone`` host paths),
at every host write through an existing array (``__setitem__``/``assign``/
host-path ``fill_``), and at the capture_begin/end/launch boundaries.
``zero_``/``fill_`` and eligible array-to-array copies default to the
round-1 ORDERED HOST PATH (flush — pending dispatches commit and complete
first — then the host write; dispatches launched afterwards observe it).
Round 2 encoded these ops as blit sections and round 2b as compute
sections in the pending command buffer; both measured healthy at micro
scale (tests/test_deferred_micro.py) but round 2 regressed the Newton
conveyor ~35x and the 2b compute encoding is unproven at Newton scale
(mechanism unknown — lead: flush cadence vs Newton collide() ``.numpy``
reads turning a long pending pipeline into serialized full commit+wait
cycles), so per KEP D2 the GPU encoding ships only behind the opt-in knob
``WARP_METAL_EAGER_GPU_FILLS=1`` (read per call, like every other knob).
With the knob set, ``zero_``/``fill_`` encode COMPUTE sections on the
same compute encoder as the launches around them (upstream CUDA
semantics: device-side memsets/memcpys on the stream, ordered with the
launches around them) — no host write, no flush, no encoder transition.
``fill_`` values whose element bytes are one replicable 8-bit pattern
encode on the GPU (0 anywhere, -1 in int/uint, True in bool); anything
else segments to the ordered host path (flush + write), the M2.5
commit-2 rule for non-representable fills. Eligible copies (equal dtype,
same device, no same-buffer overlap — the shared ``_blit_copy_payload``
rule) encode as compute copies; ineligible copies keep the host path.
Host reads of a GPU-filled/copied array go through the flush-on-read
property (shared storage shows the GPU-written bytes only once the buffer
completes).
``WARP_METAL_SYNC_EAGER=1`` reverts to per-dispatch commit+wait with
ordered host writes for ``zero_``/``fill_``/``copy`` regardless of the
fill knob (its check comes first, so the two knobs compose), for triage.
Held-view hazard (upstream-async equivalent): a numpy view grabbed BEFORE
a launch — or before a knob-enabled GPU-encoded fill (``v =
arr.numpy_view()``) — bypasses the flush-on-read property: it reads stale
bytes until some later flush point, exactly like holding a host view
across an async CUDA launch. (The default host-path fills flush before
writing, so a view held across them sees the write — it aliases the same
shared bytes.)

Arrays live in unified memory regardless of the allocating device, so a
``"cpu"`` array is directly usable by Metal kernels — the foundation for
per-builtin CPU fallback routing (M1.3). Explicit ``"cpu"`` launches run
the same pipeline: kernels with a verified CPU port (``host_fallback``)
execute their port; everything else lowers to MSL with the per-thread
scalar tile semantics upstream's CPU path forces (``block_dim`` resolves
to 1, ``context.py _resolve_launch_block_dim``). Whole-kernel CPU
execution (LLVM path) stays unwired — a cpu launch is not a separate
executor, it is the metal pipeline over shared storage.
"""

from __future__ import annotations

import ctypes
import functools
import os
import threading
import weakref

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
    type_repr,
)
from .. import native as _mm
from ..msl.emitter import MSLEmitter
from .devices import Device, get_device

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

# inverse map: numpy scalar types → warp scalars (upstream wp.array accepts
# numpy dtypes, warp/_src/types.py array.__init__ canonicalize block)
_NP_SCALAR_TO_WARP = {v: k for k, v in _SCALAR_NP.items()}

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

    def numpy(self):
        return self


class WarpArray:
    """MTLBuffer-backed array with a zero-copy numpy view.

    ``shape`` may be ``()`` (zero-dim scalar buffer), an int, or a tuple of
    up to 4 ints; the backing buffer is the flattened product in row-major
    order (warp layout). ``device`` is recorded for dispatch validation but
    allocation is device-independent (storageModeShared unified memory).
    """

    # Base-allocation ptr registry for ``wp.array(ptr=...)`` reinterpret
    # views (KEP-0001 M3.3): the shared-storage pointer IS the unified
    # device address, so mjw's efc staging (constraint.py:5356 — contact
    # frame (n,9)f32 reinterpreted as (n,3)vec3) can share the source
    # MTLBuffer without a native buffer_from_ptr. Weak values: a dead
    # allocation's ptr stops resolving (its buffer is released), while an
    # alive alias keeps the source — and the buffer — alive through a
    # strong reference of its own.

    @staticmethod
    def _resolve_allocation(addr: int) -> "WarpArray | None":
        """Find the live base allocation containing byte address ``addr``.

        Newton's selection.py points ``wp.array(ptr=...)`` at INTERIOR
        addresses (``attrib.ptr + layout.offset * value_stride``), so an
        exact registry hit is not enough — scan the live allocations' byte
        spans. The registry holds only base allocations (views never
        register), so this is a small scan."""
        if addr in WarpArray._PTR_ALIASES:
            return WarpArray._PTR_ALIASES[addr]
        for src in WarpArray._PTR_ALIASES.values():
            if src._ptr <= addr < src._ptr + src.n * src.item_bytes:
                return src
        return None

    _PTR_ALIASES: "weakref.WeakValueDictionary[int, WarpArray]" = weakref.WeakValueDictionary()

    def __init__(self, n, dtype=float32, shape=None, device=None, pinned=False, requires_grad=False, retain_grad=False):
        if requires_grad or retain_grad or pinned:
            # Adjoint/pinned allocation is a KEP-0001 Non-Goal: fail loud
            # rather than silently ignoring the flag (black-box principle).
            raise RuntimeError(
                "warp-metal does not support requires_grad/pinned/retain_grad allocation flags "
                "(adjoint codegen is a KEP-0001 Non-Goal)"
            )
        # upstream array.device IS the Device object (warp/_src/types.py
        # stores the resolved device; Newton reads e.g.
        # graph_edge_indices.device.is_cpu in graph_coloring.py:284) — the
        # Device compares equal to its name string and hashes by name, so
        # string comparisons/dict keys keep working.
        self.device = get_device(device)
        # canonicalize dtype (upstream array.__init__ canonicalize block,
        # warp/_src/types.py:3559-3565: the Python builtins map to their
        # 32-bit warp types, e.g. Newton's `wp.full(..., dtype=float)`;
        # numpy scalar types map the same way — upstream accepts
        # wp.array(dtype=np.float32) and reports wp.float32 back)
        if dtype is int or dtype is np.int64:
            dtype = int32
        elif dtype is float or dtype is np.float64:
            dtype = float32
        elif dtype is bool or dtype is np.bool_:
            dtype = warp_bool
        elif isinstance(dtype, type) and issubclass(dtype, np.number):
            mapped = _NP_SCALAR_TO_WARP.get(dtype)
            if mapped is None:
                raise NotImplementedError(
                    f"dtype {dtype} not in spike subset"
                )
            dtype = mapped
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
        self._PTR_ALIASES[self._ptr] = self
        self._build_numpy_view()

    @property
    def buffer(self):
        return self._buffer

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def requires_grad(self):
        return getattr(self, "_requires_grad_flag", False)

    @requires_grad.setter
    def requires_grad(self, value) -> None:
        # upstream stores the flag and only fails later at grad-mode use
        # (Tape/grad tape); the shim's ratified Non-Goal boundary fails loud
        # at the earliest adjoint surface instead — allocation-time
        # requires_grad=True already raises the same way. Setting False is
        # the plain-inference path (warp_nn Parameter) and must work.
        if value:
            raise RuntimeError(
                "warp-metal does not support requires_grad=True (adjoint codegen "
                "is a KEP-0001 Non-Goal)"
            )
        self._requires_grad_flag = False

    @property
    def strides(self):
        """Byte strides per dimension (upstream convention — warp
        _src/types.py ``strides_from_shape``: the last dim's stride is the
        element size in bytes; Newton's selection.py multiplies these
        directly). Views store their own strides; dense arrays compute
        row-major defaults."""
        stored = getattr(self, "_strides", None)
        if stored is not None:
            return stored
        out = []
        acc = self.item_bytes
        for i in range(len(self.shape) - 1, -1, -1):
            out.append(acc)
            acc *= self.shape[i]
        return tuple(reversed(out))

    def _dense_strides(self) -> tuple[int, ...]:
        out = []
        acc = self.item_bytes
        for i in range(len(self.shape) - 1, -1, -1):
            out.append(acc)
            acc *= self.shape[i]
        return tuple(reversed(out))

    @property
    def is_contiguous(self):
        """True when the strides match the dense row-major layout."""
        return getattr(self, "_strides", None) is None or self.strides == self._dense_strides()

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
        if getattr(self, "_ptr_is_none", False):
            return None
        return self._ptr

    @ptr.setter
    def ptr(self, value) -> None:
        # Newton's selection.py nulls the ptr of empty selection results;
        # downstream code then reads ``attrib.ptr is None`` to skip them.
        if value is None:
            self._ptr_is_none = True
            return
        raise NotImplementedError(
            "warp-metal arrays cannot be re-pointed at foreign memory "
            "(wp.array(ptr=...) aliasing over live allocations is the only path)"
        )

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

    @property
    def numpy(self) -> np.ndarray:
        """Zero-copy host view over the shared storage.

        Every access flushes the pending eager dispatches first (M2.6
        flush-on-read: a host read of shared storage is only exact once the
        uncommitted command buffer has completed — upstream eager semantics
        order host reads after launched kernels, and this is the same
        contract under deferred commits). The flush is a no-op when nothing
        is pending. Host writes through this view therefore also land after
        pending kernels, never under them."""
        _mm.flush()
        return self._numpy

    @numpy.setter
    def numpy(self, view) -> None:
        self._numpy = view

    def numpy_view(self) -> np.ndarray:
        """Zero-copy host view (flush-on-read, see ``numpy``)."""
        return self.numpy

    def _build_numpy_view(self):
        """Recreate ``self.numpy`` over the current (possibly offset) pointer.

        Shaped like real warp's host view: ``self.shape`` for scalar dtypes,
        ``self.shape + (comps,)`` for vec/quat/mat, ``(n,)`` for structs.
        Strided views (``_strides`` set, non-dense) build an as_strided view
        over the accessed byte span instead — element strides are already in
        bytes, exactly numpy's unit."""
        np_t = np.dtype(self.np_dtype)
        stored = getattr(self, "_strides", None)
        if stored is not None and stored != self._dense_strides():
            if np_t.names:
                raise NotImplementedError(
                    "strided views over struct-dtype arrays are outside the "
                    "warp-metal closure"
                )
            span = self.item_bytes
            for s, st in zip(self.shape, stored):
                if st % np_t.itemsize != 0:
                    raise NotImplementedError(
                        f"strided view: stride {st} bytes is not a multiple of "
                        f"the {np_t.itemsize}-byte component size"
                    )
                span += max(0, s - 1) * st
            raw = np.ctypeslib.as_array(
                ctypes.cast(self._ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(span,)
            )
            flat = raw.view(np_t)  # component-level view over the span
            tail_shape, tail_strides = (), ()
            if self.is_composite:
                comps = self.item_bytes // np_t.itemsize
                mshape = _MAT_TRAILING.get(self.dtype)
                tail_shape = mshape if mshape else (comps,)
                # element-internal layout stays dense row-major
                tail_acc = np_t.itemsize
                tail_strides = []
                for d in reversed(tail_shape):
                    tail_strides.append(tail_acc)
                    tail_acc *= d
                tail_strides = tuple(reversed(tail_strides))
            shape = tuple(self.shape) + tail_shape
            strides = tuple(stored) + tail_strides
            self.numpy = np.lib.stride_tricks.as_strided(flat, shape=shape, strides=strides).view(_CallableArray)
            return
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

    def _leading_index_view(self, idx: int) -> "WarpArray":
        """Upstream ``arr[i]`` on ndim>1 (warp/_src/types.py __getitem__):
        the int key occupies the leading dim and the remaining dims keep
        full slices — the view drops the leading dim entirely, i.e. shape
        ``shape[1:]`` and strides ``strides[1:]``, sharing the base buffer
        at the element offset. Newton's per-world rows
        (``_row_dropped_all[0]``, VBD ``_body_contact_scratch[0]``) read
        kernels with 1-D annotations, so the view shape MUST match
        upstream or the kernel's stride math reads wrong elements."""
        parent_strides = getattr(self, "_strides", None)
        if parent_strides is not None:
            lead_bytes = parent_strides[0]
            view = WarpArray.__new__(WarpArray)
            view.device = self.device
            view.dtype = self.dtype
            view.shape = tuple(self.shape[1:])
            view.n = 1
            for s in view.shape:
                view.n *= s
            view.np_dtype = self.np_dtype
            view.item_bytes = self.item_bytes
            view.is_composite = self.is_composite
            view._strides = parent_strides[1:] if parent_strides[1:] != view._dense_strides_from(view.shape) else None
            view.offset_elems = 0
            view._offset_bytes = self._meta_offset_bytes() + idx * lead_bytes
            view._buffer = self._buffer  # shared: release is owned by the base
            view._owns_buffer = False
            view._alias_src = getattr(self, "_alias_src", None) or self
            view._ptr = self._ptr + idx * lead_bytes
            view._build_numpy_view()
            return view
        lead = 1
        for s in self.shape[1:]:
            lead *= s
        view = WarpArray.__new__(WarpArray)
        view.device = self.device
        view.dtype = self.dtype
        view.shape = tuple(self.shape[1:])
        view.n = lead
        view.np_dtype = self.np_dtype
        view.item_bytes = self.item_bytes
        view.is_composite = self.is_composite
        view.offset_elems = self.offset_elems + idx * lead
        view._buffer = self._buffer  # shared: release is owned by the base
        view._owns_buffer = False
        view._ptr = self._ptr + idx * lead * self.item_bytes
        view._build_numpy_view()
        return view

    def _slice_view(self, start: int, length: int) -> "WarpArray":
        """Host-side array view (warp slicing): shares the base buffer at an
        element offset; the offset reaches device code via wp_arr_meta.offset."""
        parent_strides = getattr(self, "_strides", None)
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
        if parent_strides is not None:
            # strided parent: slicing dim 0 keeps every stride; the start
            # offset is a byte offset (element strides, not dense counts)
            view._strides = parent_strides if parent_strides != view._dense_strides_from(view.shape) else None
            view.offset_elems = 0
            view._offset_bytes = self._meta_offset_bytes() + start * parent_strides[0]
            view._alias_src = getattr(self, "_alias_src", None) or self
            byte_start = start * parent_strides[0]
        else:
            view.offset_elems = self.offset_elems + start
            byte_start = start * self.item_bytes
        view._buffer = self._buffer  # shared: release is owned by the base
        view._owns_buffer = False
        view._ptr = self._ptr + byte_start
        view._build_numpy_view()
        return view

    @classmethod
    def _ptr_alias_view(cls, src: "WarpArray", ptr: int, dtype, shape, device, capacity=None, strides=None) -> "WarpArray":
        """Zero-copy reinterpret view over a live allocation's buffer.

        KEP-0001 M3.3 (mjw constraint.py:5356): the efc contact staging
        reinterprets ``(naconmax, 9)f32`` as ``(naconmax, 3)vec3`` — the same
        byte span under a different element type. Under unified memory the
        shared-storage pointer IS the device address, so the view shares the
        source MTLBuffer (never releases it — ``_owns_buffer`` False plus a
        strong source reference) and rebuilds the numpy host view over the
        same pointer range. Upstream ``wp.array(ptr=...)`` semantics: the view
        starts at ``ptr`` with the requested dtype/shape/strides (bytes);
        Newton's selection.py points ``ptr`` INSIDE a live allocation and
        passes byte strides — the interior offset reaches device code through
        ``_offset_bytes`` (the buffer binding is the whole allocation).
        """
        if dtype is int:
            dtype = int32
        elif dtype is float:
            dtype = float32
        elif dtype is bool:
            dtype = warp_bool
        view = WarpArray.__new__(WarpArray)
        view.device = get_device(device) if device is not None else src.device
        if view.device != src.device:
            raise ValueError(
                f"wp.array(ptr=...) aliasing: device {view.device!r} does not match the "
                f"source allocation's {src.device!r}"
            )
        view.dtype = dtype
        view.shape = _normalize_shape(shape) if shape is not None else _normalize_shape(src.n)
        view.n = 1
        for s in view.shape:
            view.n *= s
        view.np_dtype, view.item_bytes, view.is_composite = _dtype_layout(dtype)
        view.offset_elems = 0
        if strides is not None:
            if np.dtype(view.np_dtype).names:
                raise NotImplementedError(
                    "wp.array(ptr=..., strides=...) over struct dtypes is outside "
                    "the warp-metal closure"
                )
            strides = tuple(int(s) for s in strides)
            if len(strides) != len(view.shape):
                raise ValueError(
                    f"wp.array(strides=...): {len(strides)} strides for a "
                    f"{len(view.shape)}-dimensional shape {view.shape} (upstream "
                    "strides are in bytes, one per dimension)"
                )
            if strides == view._dense_strides_from(view.shape):
                view._strides = None
            else:
                view._strides = strides
        else:
            view._strides = None
        if capacity is not None and int(capacity) < view.n:
            raise ValueError(
                f"wp.array(ptr=...) aliasing: capacity {capacity} < element count {view.n}"
            )
        base = int(ptr) - src._ptr  # byte offset of the alias inside the source span
        span = src.n * src.item_bytes - base
        need = view.item_bytes if view.n else 0
        if view.n:
            st = view._strides if view._strides is not None else view._dense_strides()
            for s, s_bytes in zip(view.shape, st):
                need += max(0, s - 1) * s_bytes
        if base < 0 or need > span:
            raise ValueError(
                f"wp.array(ptr=...) aliasing: {need} bytes needed at byte offset {base}, "
                f"source allocation has {span} available (upstream reinterpret semantics "
                "require the same byte span)"
            )
        view._buffer = src._buffer  # shared: release is owned by the source
        view._owns_buffer = False
        view._alias_src = src  # strong ref: the shared buffer must outlive the view
        view._offset_bytes = base  # device addressing: buffer base + this offset
        view._ptr = int(ptr)
        view._build_numpy_view()
        return view

    def reshape(self, shape):
        """Return a reshaped zero-copy view (upstream array.reshape semantics,
        warp/_src/types.py:4860: same pointer, new shape, base buffer kept
        alive through the shared reference; mjw solver's _cholesky_
        factorize_solve calls ``grad.reshape(shape=(...))``).

        Contiguous arrays only (upstream check kept for parity — runtime
        arrays here are always dense row-major). One ``-1`` dimension may be
        inferred from the element count; the total must match. Upstream also
        transfers the grad/read flags; this runtime rejects grad allocation,
        so there is nothing to carry.
        """
        if not self.is_contiguous:
            raise RuntimeError("Reshaping non-contiguous arrays is unsupported.")

        # convert shape to tuple (upstream: int → 1-tuple, other iterables
        # via tuple(); mjw passes shape=(...) as a keyword)
        if shape is None:
            raise RuntimeError("shape parameter is required.")
        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)

        if len(shape) > 4:
            # upstream ARRAY_MAX_DIMS
            raise RuntimeError(
                f"Arrays may only have 4 dimensions maximum, trying to create array with {len(shape)} dims."
            )

        # check for -1 dimension and reformat (upstream verbatim)
        if -1 in shape:
            idx = self.size
            denom = 1
            minus_one_count = 0
            for i, d in enumerate(shape):
                if d == -1:
                    idx = i
                    minus_one_count += 1
                else:
                    denom *= d
            if minus_one_count > 1:
                raise RuntimeError("Cannot infer shape if more than one index is -1.")
            new_shape = list(shape)
            new_shape[idx] = int(self.size / denom)
            shape = tuple(new_shape)

        size = 1
        for d in shape:
            size *= d

        if size != self.size:
            raise RuntimeError(
                f"cannot reshape array of shape {self.shape} (size {self.size}) into shape {shape} (size {size})"
            )

        view = WarpArray.__new__(WarpArray)
        view.device = self.device
        view.dtype = self.dtype
        view.shape = shape
        view.n = self.n
        view.np_dtype = self.np_dtype
        view.item_bytes = self.item_bytes
        view.is_composite = self.is_composite
        view.offset_elems = self.offset_elems
        view._buffer = self._buffer  # shared: release is owned by the base
        view._owns_buffer = False
        view._ptr = self._ptr
        view._build_numpy_view()
        return view

    def to(self, device, requires_grad=None):
        """Return a Warp array with this array's data moved to the specified
        device, no-op if already on device (upstream array.to,
        warp/_src/types.py:4828). All warp-metal arrays live in shared
        unified memory, so the cross-device path is a clone onto the target
        device's allocator; the same-device path returns ``self`` exactly
        like upstream. ``requires_grad`` keeps upstream's clone argument —
        the clone fails loud on grad flags (KEP-0001 Non-Goal)."""
        device = get_device(device)
        if self.device == device:
            return self
        return clone(self, device=device, requires_grad=requires_grad)

    def flatten(self):
        """Return a zero-copy view of the array collapsed to 1-D (upstream
        array.flatten, warp/_src/types.py:4835). Only supported for
        contiguous arrays; runtime arrays here are always dense row-major.
        Upstream also transfers the read flag and grad — this runtime
        rejects grad allocation, so there is nothing to carry; the shared
        ``_buffer`` reference keeps the base allocation alive exactly like
        upstream's ``_ref`` back-ref."""
        if self.ndim == 1:
            return self
        if not self.is_contiguous:
            raise RuntimeError("Flattening non-contiguous arrays is unsupported.")
        # reshape gives the exact (size,) 1-D zero-copy view (upstream builds
        # array(ptr=self.ptr, shape=(self.size,)) — same shared buffer)
        return self.reshape(self.size)

    def view(self, dtype):
        """Return a zero-copy view of this array's memory with a different
        data type (upstream array.view, warp/_src/types.py:4925).

        Upstream branches on same-byte-size reinterpret, vec/mat→scalar and
        scalar→vec/mat conversions, carrying explicit strides. This runtime
        is dense row-major only, so every representable case reduces to a
        dense view with the new dtype and the upstream-adjusted shape; the
        upstream stride-compatibility checks are vacuous here and the
        non-representable (strided) cases fall out of the byte-span capacity
        check in ``_ptr_alias_view``. Grad flags have nothing to carry (this
        runtime rejects grad allocation)."""
        from ..frontend.types import type_is_scalar, type_size_in_bytes

        if type_size_in_bytes(dtype) == type_size_in_bytes(self.dtype):
            # a simple "reinterpret cast"
            result_shape = self.shape
        elif type_is_scalar(dtype) and hasattr(self.dtype, "_wp_scalar_type_"):
            # cast from vec/mat type to scalar type
            if type_size_in_bytes(dtype) == type_size_in_bytes(self.dtype._wp_scalar_type_):
                result_shape = (*self.shape, *self.dtype._shape_)
            else:
                raise TypeError(
                    f"cannot create an array view with dtype {type_repr(dtype)} "
                    f"from source dtype {type_repr(self.dtype)}: source scalar dtype "
                    f"{type_repr(self.dtype._wp_scalar_type_)} has size "
                    f"{type_size_in_bytes(self.dtype._wp_scalar_type_)} bytes, "
                    f"but target dtype has size {type_size_in_bytes(dtype)} bytes"
                )
        elif type_is_scalar(self.dtype) and hasattr(dtype, "_wp_scalar_type_"):
            # cast from scalar type to vec/mat type
            if type_size_in_bytes(self.dtype) == type_size_in_bytes(dtype._wp_scalar_type_):
                # ensure that the shape is compatible with the requested
                # vec/mat type (the inner-strides check is vacuous — dense
                # row-major storage makes matching trailing dims dense)
                dtype_ndim = len(dtype._shape_)
                if self.ndim < dtype_ndim or self.shape[-dtype_ndim:] != dtype._shape_:
                    raise RuntimeError(
                        f"The shape of the source array {self.shape} is not compatible with the requested data type {type_repr(dtype)}"
                    )
                # Corner case: if the array contains a single vector or
                # matrix, the result would be a 0-dimensional array, which
                # is not supported in Warp currently. In this case, we return
                # a 1D array with one element. (upstream verbatim)
                result_shape = self.shape[:-dtype_ndim] or (1,)
            else:
                raise TypeError(
                    f"cannot create an array view with dtype {type_repr(dtype)} "
                    f"from source dtype {type_repr(self.dtype)}: source dtype has size "
                    f"{type_size_in_bytes(self.dtype)} bytes, but target scalar dtype "
                    f"{type_repr(dtype._wp_scalar_type_)} has size "
                    f"{type_size_in_bytes(dtype._wp_scalar_type_)} bytes"
                )
        else:
            raise TypeError(
                f"cannot create an array view with dtype {type_repr(dtype)} "
                f"from source dtype {type_repr(self.dtype)}: source dtype has size "
                f"{type_size_in_bytes(self.dtype)} bytes, but target dtype has size "
                f"{type_size_in_bytes(dtype)} bytes"
            )

        return WarpArray._ptr_alias_view(self, self._ptr, dtype, result_shape, self.device)

    def _tuple_slice_view(self, key) -> "WarpArray":
        """Upstream multi-dim slicing (warp/_src/types.py __getitem__): tuple
        keys containing slices produce a zero-copy view sharing the base
        buffer — int keys drop their dimension (offset advance), slice keys
        select a range (stride multiplied by the step), missing trailing
        keys pad with full slices. Newton's selection.py slices every
        selected attribute array this way (``attrib[slices]``)."""
        if len(key) > len(self.shape):
            raise IndexError(
                f"too many indices for array of shape {self.shape}: {len(key)} keys"
            )
        shape = []
        strides = []
        offset_bytes = 0
        for i, k in enumerate(key):
            dim = self.shape[i]
            st = self.strides[i]
            if isinstance(k, slice):
                start, stop, step = k.indices(dim)
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                length = len(range(start, stop, step))
                shape.append(length)
                strides.append(st * step)
                offset_bytes += start * st
            elif isinstance(k, (int, np.integer)):
                idx = int(k)
                if idx < 0:
                    idx += dim
                if idx < 0 or idx >= dim:
                    raise IndexError(f"index {key[i]} out of range for dim {dim} of shape {self.shape}")
                offset_bytes += idx * st  # dim dropped
            else:
                raise TypeError(f"unsupported array index {k!r}")
        # missing trailing keys pad with full slices (upstream semantics)
        for i in range(len(key), len(self.shape)):
            shape.append(self.shape[i])
            strides.append(self.strides[i])
        view = WarpArray.__new__(WarpArray)
        view.device = self.device
        view.dtype = self.dtype
        view.shape = tuple(shape)
        view.n = 1
        for s in view.shape:
            view.n *= s
        view.np_dtype = self.np_dtype
        view.item_bytes = self.item_bytes
        view.is_composite = self.is_composite
        view._strides = tuple(strides) if tuple(strides) != self._dense_strides_from(shape) else None
        view.offset_elems = 0
        view._offset_bytes = self._meta_offset_bytes() + offset_bytes
        view._buffer = self._buffer  # shared: release is owned by the base
        view._owns_buffer = False
        view._alias_src = getattr(self, "_alias_src", None) or self
        view._ptr = self._ptr + offset_bytes
        view._build_numpy_view()
        return view

    def _meta_offset_bytes(self) -> int:
        stored = getattr(self, "_offset_bytes", None)
        if stored is not None:
            return stored
        return self.offset_elems * self.item_bytes

    def _dense_strides_from(self, shape) -> tuple[int, ...]:
        out = []
        acc = self.item_bytes
        for i in range(len(shape) - 1, -1, -1):
            out.append(acc)
            acc *= shape[i]
        return tuple(reversed(out))

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
            # upstream __getitem__ (warp/_src/types.py:4278): int keys drop
            # their dimension and missing trailing keys pad with full slices
            # (len(key) < ndim → view, Newton's [0, 0] on 3-D attribute
            # arrays); slices select ranges. Only all-int len==ndim keys are
            # element reads — upstream forbids those on wp.array, the shim
            # keeps the historical numpy element read (host convenience).
            if any(isinstance(k, slice) for k in key) or len(key) < len(self.shape):
                return self._tuple_slice_view(key)
            return self.numpy_view()[key]
        if isinstance(key, (int, np.integer)):
            if len(self.shape) == 0:
                raise TypeError("scalar array is not indexable")
            if len(self.shape) == 1:
                return self.numpy[key]
            # leading-index view: upstream arr[i] on ndim>1 drops the
            # leading dim (key=[i] padded with full slices) — a view of one
            # leading slice sharing the buffer (Newton steel per-world rows,
            # VBD scratch rows); only 1-D indexing yields the element value
            idx = int(key)
            if idx < 0:
                idx += self.shape[0]
            if idx < 0 or idx >= self.shape[0]:
                raise IndexError(f"array index {key} out of range for shape {self.shape}")
            return self._leading_index_view(idx)
        raise TypeError(f"unsupported array index {key!r}")

    def __setitem__(self, key, value):
        # host write into an existing buffer: pending kernels must run first
        # (the .numpy property flush covers this; explicit for clarity)
        _mm.flush()
        if isinstance(key, (int, np.integer)) and len(self.shape) == 1:
            self._numpy[key] = value
            return
        self._numpy[key] = value

    def zero_(self):
        """In-place zero fill (warp convention method).

        DEFAULT (round-1 semantics): an ordered host write — pending eager
        dispatches flush first (commit + wait), the zeros land on shared
        storage, and dispatches launched afterwards observe them (upstream
        eager semantics: a host-side memset synchronized with the stream).
        ``WARP_METAL_EAGER_GPU_FILLS=1`` instead encodes a tiny fill-kernel
        COMPUTE section on the same compute encoder as the launches around
        it (round 2b — no host write, no flush, no encoder transition; a
        dependent launch observes the fill in stream order; SolverXPBD
        re-zeros its atomic-accumulation deltas every iteration). Round 2
        encoded this as a ``fillBuffer`` blit section, which regressed the
        Newton conveyor ~35x; the 2b compute encoding measured healthy at
        micro scale (tests/test_deferred_micro.py) but is unproven at
        Newton scale, so the conveyor-safe round-1 host path stays the
        default and the GPU encoding is opt-in (KEP D2: a known regression
        never ships as the default). Host reads of a knob-filled array
        flush first (flush-on-read: shared storage shows the GPU-written
        bytes only once the buffer completes). Inside a capture the fill
        is recorded for replay, plus the record-time host write documented
        on ``ScopedCapture``; ``WARP_METAL_SYNC_EAGER=1`` keeps the flush +
        ordered host write regardless of the knob.
        """
        if _CAPTURE_RECORDS is not None:
            _mm.flush()  # boundary safety: capture_begin already drained
            self._numpy[...] = 0
            _record_fill(self, 0.0)
            return self
        if _sync_eager() or not _eager_gpu_fills():
            # round-1 default (and the SYNC_EAGER escape hatch): ordered
            # host write — pending dispatches complete first, the write
            # lands, later dispatches observe it (no-op flush in sync mode,
            # where nothing ever pends)
            _mm.flush()
            self._numpy[...] = 0
            return self
        if self.n == 0:
            return self  # upstream skips the memset for empty arrays
        _mm.compute_fill(
            self.buffer, self.offset_elems * self.item_bytes, self.n * self.item_bytes, 0
        )
        return self

    def fill_(self, value):
        """In-place constant fill (warp convention method; recorded like zero_).

        DEFAULT (round-1 semantics): the ordered host path for every value —
        pending dispatches flush first, the write lands, later dispatches
        observe it. ``WARP_METAL_EAGER_GPU_FILLS=1`` encodes a GPU
        compute-section fill (see ``zero_``) when the value's per-element
        bytes are one replicable 8-bit pattern (``0`` in any dtype, ``-1``
        in int/uint dtypes, ``True`` in bool — Newton's per-iteration fills
        are dominated by exactly these), keeping everything else on the
        ordered host path, the M2.5 commit-2 rule for non-representable
        fills — SolverSteel's ``INT32_MAX`` status sentinels and arbitrary
        float inits pay the flush cost only for those fills. Ordering and
        host-visibility semantics are ``zero_``'s; ``WARP_METAL_SYNC_EAGER=1``
        and capture keep the host write for every value."""
        if _CAPTURE_RECORDS is not None:
            _mm.flush()
            self._numpy[...] = value
            _record_fill(self, value)
            return self
        if _sync_eager() or not _eager_gpu_fills():
            # round-1 default (and SYNC_EAGER): ordered host write, any value
            _mm.flush()
            self._numpy[...] = value
            return self
        if self.n == 0:
            return self  # upstream fill_ converts nothing for empty arrays
        byte = _exact_fill_byte(self, value)
        if byte is not None:
            _mm.compute_fill(
                self.buffer, self.offset_elems * self.item_bytes, self.n * self.item_bytes, byte
            )
            return self
        # not byte-replicable (INT32_MAX sentinels, -0.0, arbitrary scalars):
        # segment to the ordered host path — pending dispatches complete
        # first, the write lands, later dispatches observe it
        _mm.flush()
        self._numpy[...] = value
        return self

    def assign(self, value) -> None:
        """Copy host data into the array (upstream ``array.assign``).

        Accepts another WarpArray or any host array-like; ``numpy.copyto``
        checks shape and casts dtype the same way upstream does. Storage is
        shared memory, so the write is immediately visible to in-flight and
        captured dispatches. Pending dispatches flush first (host write must
        not land under a kernel that has not executed yet)."""
        if isinstance(value, WarpArray):
            value = value.numpy
        _mm.flush()
        np.copyto(self._numpy, value)

    def to_numpy(self):
        """Host copy of the array data (flush-on-read; see ``numpy``)."""
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
        device = self.device.name if isinstance(self.device, Device) else self.device
        return (
            f"warp_metal.WarpArray(shape={self.shape}, dtype={getattr(self.dtype, '__name__', self.dtype)}, "
            f"device={device!r})"
        )


def _allocate(shape, dtype, device, requires_grad=False, pinned=False, retain_grad=False) -> WarpArray:
    return WarpArray(shape, dtype, shape=shape, device=device, requires_grad=requires_grad, pinned=pinned, retain_grad=retain_grad)


# the runtime array is the native wp.array family on Metal (see
# frontend.types.register_runtime_array_family)
from ..frontend.types import array as _wp_array, register_runtime_array_family  # noqa: E402

register_runtime_array_family(WarpArray, _wp_array)


def zeros(shape, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    arr = _allocate(shape, dtype, device, requires_grad, pinned, retain_grad)
    arr._numpy.fill(0)  # internal view: fresh buffer, no pending dispatch can reference it
    return arr


def empty(shape, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    return _allocate(shape, dtype, device, requires_grad, pinned, retain_grad)


def ones(shape, dtype=float32, device=None, requires_grad=False, pinned=False, retain_grad=False, **kwargs):
    arr = _allocate(shape, dtype, device, requires_grad, pinned, retain_grad)
    arr._numpy.fill(1)  # internal view: fresh buffer (see zeros)
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
        arr._numpy[...] = row  # internal view: fresh buffer (see zeros)
    else:
        arr._numpy.fill(value)
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
        return arr
    payload = _blit_copy_payload(arr, src, 0, 0, src.n) if src.is_contiguous else None
    if payload is not None and src.n > 0 and not _sync_eager() and _eager_gpu_fills():
        # GPU-encoded stream memcpy (opt-in knob; same eligibility as the
        # replay path — equal dtype, same device, fresh dest never
        # overlaps), encoded as a compute section (round 2b). The fresh
        # destination's host view is stale until the usual flush-on-read —
        # the upstream-async equivalent.
        _mm.compute_copy(payload[1], payload[2], payload[3], payload[4], payload[5])
        return arr
    _mm.flush()  # host copy: read src after pending kernels, not under them
    arr._numpy[:] = src.numpy  # dest is fresh (internal view); src read flushes
    return arr


def copy(dest, src, dest_offset=0, src_offset=0, count=None, stream=None):
    """Copy ``count`` elements (all when None) from ``src`` into ``dest``.

    Upstream contract (context.py ``copy``): both sides must be arrays, and
    the copy is a raw memcpy valid whenever the element sizes match in bytes
    — dtype identity is not required (float32 ↔ float canonicalize to the
    same 4-byte element upstream; int32 ↔ float32 same-size copies are
    equally legal there).

    DEFAULT (round-1 semantics): the ordered host copy — pending dispatches
    flush first (the copy reads src's post-kernel state), the write lands,
    later dispatches observe dest. ``WARP_METAL_EAGER_GPU_FILLS=1`` encodes
    the copy as a GPU compute section (round 2b: a tiny copy-kernel dispatch
    on the same compute encoder as the launches around it — the round-2
    ``copyFromBuffer`` blit encoding regressed the Newton conveyor ~35x)
    when it matches the replay path's copy eligibility (equal dtype — dense
    row-major layouts are then byte-identical, composite dtypes included —
    same device, no same-buffer overlap; the shared ``_blit_copy_payload``
    rule), killing the host flush entirely and ordering the copy with the
    launches around it. Ineligible copies keep the host path under the knob
    too, as does ``WARP_METAL_SYNC_EAGER=1``.
    """
    src_is_warp = isinstance(src, WarpArray)
    dest_is_warp = isinstance(dest, WarpArray)
    if not (src_is_warp and dest_is_warp):
        if not src_is_warp and not dest_is_warp:
            # vendored frontend arrays (texture staging: array(data,
            # device="cpu") in Texture.__init__ → _init_from_data → this
            # copy) are host-side ptr-backed arrays, not runtime WarpArrays.
            # Upstream copy() copies between two array objects with a raw
            # memcpy — do the same on the host when both sides are
            # contiguous and element-size compatible; anything else fails
            # loud rather than silently corrupting.
            from ..frontend.types import type_size_in_bytes

            if (
                getattr(dest, "ptr", None) is not None
                and getattr(src, "ptr", None) is not None
                and getattr(dest, "is_contiguous", False)
                and getattr(src, "is_contiguous", False)
                and type_size_in_bytes(dest.dtype) == type_size_in_bytes(src.dtype)
                and dest.size == src.size
            ):
                if dest_offset or src_offset or count is not None:
                    raise NotImplementedError(
                        "vendored-array host copy supports full-range copies only "
                        "(the texture staging path passes no offsets)"
                    )
                import ctypes as _ctypes

                _ctypes.memmove(dest.ptr, src.ptr, dest.capacity)
                return
        # upstream: "Copy source and destination must be arrays" (context.py:15197)
        raise RuntimeError("Copy source and destination must be arrays")
    if src.item_bytes != dest.item_bytes:
        # upstream element-size check + message verbatim (context.py:15294-15299)
        raise RuntimeError(
            "Incompatible array data types: "
            f"destination dtype={type_repr(dest.dtype)}, "
            f"source dtype={type_repr(src.dtype)}"
        )
    n = src.n - src_offset if count is None else int(count)
    if n < 0 or dest_offset + n > dest.n or src_offset + n > src.n:
        raise ValueError(f"copy out of bounds: count={n}, dest_offset={dest_offset}, src_offset={src_offset}")
    if _CAPTURE_RECORDS is not None:
        _CAPTURE_RECORDS.append(("copy", dest, src, dest_offset, src_offset, n))
        return
    if n == 0:
        return  # zero-size copy: nothing to do, no flush either
    if not _sync_eager() and _eager_gpu_fills():
        payload = _blit_copy_payload(dest, src, dest_offset, src_offset, n)
        if payload is not None:
            # GPU-encoded stream memcpy as a compute section (round 2b,
            # opt-in knob): in-stream ordered, no flush
            _mm.compute_copy(payload[1], payload[2], payload[3], payload[4], payload[5])
            return
    # host fallback (the default, SYNC_EAGER, or a copy outside copy
    # eligibility under the knob — cross-dtype same-size, overlapping
    # ranges): flush so the copy reads src's post-kernel state and lands
    # before any later dispatch observes dest
    _mm.flush()
    dest.numpy[dest_offset : dest_offset + n] = src.numpy[src_offset : src_offset + n]


def from_numpy(arr, dtype=None, shape=None, strides=None, device=None, requires_grad=False, retain_grad=False):
    """Create a WarpArray from a numpy array.

    Copies the data into a fresh MTLBuffer (shared storage). A zero-copy
    wrap of foreign memory needs a native ``buffer_from_ptr`` (Phase 2).
    ``strides`` (bytes, upstream convention) reinterprets the copied raw
    buffer with caller-provided strides — the numpy source's raw memory is
    copied verbatim, so values land exactly where upstream's alias
    semantics would read them.
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
    if strides is not None:
        return _from_numpy_strided(arr, dtype, target_shape, strides, device)
    out = _allocate(target_shape or (1,), dtype, device)
    np_t = np.dtype(out.np_dtype)
    if np_t.names:
        # struct element: each source element is a warp struct instance whose
        # ``_ctype`` is a ctypes structure mirroring the numpy layout byte for
        # byte — copy records verbatim (embedded wp.array fields keep their
        # shared-storage pointers, which are the device addresses)
        elements = arr if isinstance(arr, (list, tuple)) else None
        if elements is None and isinstance(np_arr, np.ndarray) and np_arr.dtype == np_t:
            # numpy structured source (Newton picking.py: np.empty(1,
            # dtype=PickingState.numpy_dtype())) — the record layout equals
            # the struct's byte for byte, so the records copy verbatim, the
            # same contract as the struct-instance path below. Anything not
            # exactly the struct's own numpy dtype falls through to the
            # fail-loud branch (black-box principle).
            if np_arr.shape[0] != out.n:
                raise ValueError(f"struct wp.array: numpy has {np_arr.shape[0]} records for shape {out.shape}")
            ctypes.memmove(out._ptr, np_arr.ctypes.data, out.n * out.item_bytes)
            return out
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
    out._numpy[:total_scalars] = flat.reshape(out._numpy.shape)  # fresh buffer (see zeros)
    return out


def _from_numpy_strided(src_np, dtype, shape, strides, device):
    """``wp.array(numpy, shape=..., strides=...)`` — upstream aliases the
    numpy buffer with the given byte strides; Metal cannot alias
    non-page-aligned host memory, so the raw source bytes are copied into a
    fresh MTLBuffer and reinterpreted with the same strides (identical
    values). The strides must mirror the numpy source's own layout — a
    mismatched reinterpret would silently misread, which fails loud here
    (black-box principle)."""
    if not isinstance(src_np, np.ndarray):
        raise TypeError("wp.array(data=..., strides=...) needs a numpy array source")
    if dtype is None:
        for wtype, np_t in _SCALAR_NP.items():
            if np_t == src_np.dtype:
                dtype = wtype
                break
        else:
            raise NotImplementedError(f"from_numpy: numpy dtype {src_np.dtype} not mapped")
    np_t, item_bytes, is_composite = _dtype_layout(dtype)
    np_t = np.dtype(np_t)
    if shape is None:
        target = tuple(src_np.shape)
        if is_composite:
            comps = item_bytes // np_t.itemsize
            product, k = 1, 0
            for ax in reversed(src_np.shape):
                product *= int(ax)
                k += 1
                if product >= comps:
                    break
            if product == comps:
                target = tuple(src_np.shape[: src_np.ndim - k])
    else:
        target = _normalize_shape(shape)
    strides = tuple(int(s) for s in strides)
    if len(strides) != len(target):
        raise ValueError(
            f"wp.array(strides=...): {len(strides)} strides for a {len(target)}-dim shape {target}"
        )
    expect_tail = () if not is_composite else (np_t.itemsize,) * (src_np.ndim - len(target))
    if tuple(src_np.strides) != tuple(src_np.strides[: len(target)]) + expect_tail:
        raise NotImplementedError(
            f"wp.array(strides={strides}): the numpy source layout {src_np.strides} is "
            "not the plain element layout the strides describe — a mismatched "
            "reinterpret would silently misread; warp-metal fails loud instead"
        )
    span = item_bytes + sum(max(0, s - 1) * st for s, st in zip(target, strides))
    if span > src_np.nbytes:
        raise ValueError(
            f"wp.array(strides=...): strided span {span} bytes exceeds the numpy "
            f"source's {src_np.nbytes}"
        )
    out = WarpArray.__new__(WarpArray)
    out.device = get_device(device)
    out.dtype = dtype
    out.shape = target
    out.n = 1
    for s in target:
        out.n *= s
    out.np_dtype, out.item_bytes, out.is_composite = _dtype_layout(dtype)
    out.offset_elems = 0
    out._strides = strides if strides != out._dense_strides_from(target) else None
    out._owns_buffer = True
    out._buffer = _mm.buffer_create(span)
    out._ptr = _mm.buffer_ptr(out._buffer)
    WarpArray._PTR_ALIASES[out._ptr] = out
    src_addr = src_np.__array_interface__["data"][0]
    ctypes.memmove(out._ptr, src_addr, min(span, src_np.nbytes))
    out._build_numpy_view()
    return out


def synchronize() -> None:
    """Wait for outstanding eager dispatches (upstream ``wp.synchronize``).

    Flushes the pending eager command buffer (commit + watchdog-bounded
    wait); raises MetalError naming the pending-dispatch count on failure."""
    _mm.flush()


def flush() -> None:
    """Commit the pending eager command buffer and wait (no-op when idle).

    Same native wait as :func:`synchronize`; the explicit name for the
    M2.6 deferred eager model. Also available as ``wp.flush``."""
    _mm.flush()


class _ArrayMeta(type):
    """Metaclass for the public ``wp.array`` facade (KEP-0001 M3.2).

    Upstream ``isinstance(x, wp.array)`` accepts every ``wp.array`` instance,
    including the annotation-mode descriptors ``wp.array(ndim=..., dtype=...)``
    returns (mujoco_warp builds its array specs exactly that way — types.py:950
    — and probes them with ``isinstance(typ, wp.array)``, warp_util.py:26).
    The facade is a plain wrapper class, though: neither runtime ``WarpArray``
    values nor vendored annotation descriptors are Python subclasses of it, so
    the default type() check answers False for both — mujoco_warp's put_model
    then routed every one of the 521 Model fields to the scalar ``f.type(value)``
    branch and raised ``TypeError: 'array' object is not callable`` (matrix
    §5-1). The checks here restore the upstream answers: a runtime WarpArray or
    a vendored array descriptor passes; everything else keeps the plain
    subclass answer.
    """

    def __instancecheck__(cls, instance):
        from ..frontend.types import array as _vendored_array

        return isinstance(instance, WarpArray) or isinstance(instance, _vendored_array)

    def __subclasscheck__(cls, subclass):
        from ..frontend.types import array as _vendored_array

        if subclass is cls:
            return True
        return issubclass(subclass, (WarpArray, _vendored_array))


class array(metaclass=_ArrayMeta):
    """Public ``wp.array``: runtime-backed construction (KEP-0001 M1.6).

    Newton's builder creates arrays host-side with data; the runtime
    WarpArray is the native Metal allocation (shared storage). Subscripting
    yields the vendored annotations, and ``isinstance`` accepts both runtime
    and vendored instances via the ``_ArrayMeta`` metaclass above (a bare
    in-class ``__instancecheck__`` never fires — ``isinstance`` dispatches on
    the metaclass — so the facade check must live on ``_ArrayMeta``; M3.2),
    so kernel annotations, struct fields, and downstream isinstance-based
    spec probes (mujoco_warp ``is_array_spec``) behave identically to
    upstream. ``ptr``/``capacity`` aliasing needs a native
    ``buffer_from_ptr`` (Phase 2) and fails loud; ``copy=False`` lands as a
    copy (Metal cannot alias non-page-aligned host memory — see __new__);
    grad/pinned flags stay Non-Goals.
    """

    def __new__(
        cls,
        data=None,
        dtype=None,
        shape=None,
        strides=None,
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
        if strides is not None:
            # upstream strides kwarg is in BYTES (warp/_src/types.py
            # strides_from_shape; Newton's selection.py passes bytes)
            strides = tuple(int(s) for s in strides)
        if ptr is not None or capacity is not None or deleter is not None:
            if ptr is None or deleter is not None:
                # foreign-memory wrap (deleter-owned storage) is unbuildable
                # without the native entry point — keep the Phase-2 boundary
                raise NotImplementedError(
                    "wp.array capacity/deleter aliasing needs native buffer_from_ptr (Phase 2)"
                )
            # KEP-0001 M3.3: reinterpret an existing warp-metal allocation
            # through its shared-storage pointer (mjw constraint.py:5356
            # reinterprets contact.frame (n,9)f32 as (n,3)vec3). The ptr IS
            # the unified device address, so the view shares the same
            # MTLBuffer; Newton's selection.py points INSIDE a live
            # allocation — resolve by byte-span containment, exact hits
            # first. Pointers outside every live allocation keep the
            # Phase-2 fail-loud.
            src = WarpArray._resolve_allocation(int(ptr))
            if src is None:
                raise NotImplementedError(
                    f"wp.array(ptr={hex(int(ptr))}) aliasing: pointer is not a live warp-metal "
                    "allocation — foreign-memory wrap needs native buffer_from_ptr (Phase 2)"
                )
            if shape is None:
                shape = (src.n,)
            return WarpArray._ptr_alias_view(src, int(ptr), dtype, shape, device, capacity, strides)
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
            return from_numpy(data, dtype=dtype, shape=shape, strides=strides, device=device)
        if strides is not None:
            raise NotImplementedError(
                "wp.array(shape=..., strides=...) allocation is outside the "
                "warp-metal closure (allocate dense and slice instead)"
            )
        if shape is None:
            raise TypeError("wp.array() needs either data or shape")
        return zeros(shape, dtype=dtype if dtype is not None else float32, device=device)

    def __class_getitem__(cls, params):
        from ..frontend.types import _parse_array_subscript, array as _vendored_array

        return _parse_array_subscript(_vendored_array, params)

    # isinstance/issubclass parity lives on _ArrayMeta (the metaclass): an
    # __instancecheck__ defined in this class body would never fire, because
    # isinstance() dispatches on type(cls), i.e. the metaclass.


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


# wp_mesh mirrors the MSL struct (msl/prelude.py _MESH_QUERY_OPS — the
# upstream mesh.h struct Mesh port): three {device data ptr; wp_arr_meta}
# array payloads (48 bytes each), per-triangle bounds pointers, counts, the
# EMBEDDED BVH (112-byte wp_bvh_t image written in place — upstream embeds
# the BVH by value and mesh_get_bvh addresses the member), context and
# average_edge_length. 304 bytes total, natural alignment.
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


class _MeshBVH(_ctypes.Structure):
    _fields_ = [
        ("node_lowers", _ctypes.c_uint64),
        ("node_uppers", _ctypes.c_uint64),
        ("node_parents", _ctypes.c_uint64),
        ("node_counts", _ctypes.c_uint64),
        ("primitive_indices", _ctypes.c_uint64),
        ("root", _ctypes.c_uint64),
        ("item_lowers", _ctypes.c_uint64),
        ("item_uppers", _ctypes.c_uint64),
        ("item_groups", _ctypes.c_uint64),
        ("max_depth", _ctypes.c_int32),
        ("max_nodes", _ctypes.c_int32),
        ("num_nodes", _ctypes.c_int32),
        ("num_leaf_nodes", _ctypes.c_int32),
        ("num_items", _ctypes.c_int32),
        ("leaf_size", _ctypes.c_int32),
        ("constructor_type", _ctypes.c_int32),
        ("_pad0", _ctypes.c_int32),
        ("context", _ctypes.c_uint64),
    ]


assert _ctypes.sizeof(_MeshBVH) == 112, _ctypes.sizeof(_MeshBVH)


class _MeshEntry(_ctypes.Structure):
    _fields_ = [
        ("points", _MeshArrayF),
        ("velocities", _MeshArrayF),
        ("indices", _MeshArrayI),
        ("lowers", _ctypes.c_uint64),
        ("uppers", _ctypes.c_uint64),
        ("solid_angle_props", _ctypes.c_uint64),
        ("num_points", _ctypes.c_int32),
        ("num_tris", _ctypes.c_int32),
        ("bvh", _MeshBVH),
        ("context", _ctypes.c_uint64),
        ("average_edge_length", _ctypes.c_float),
        ("_pad0", _ctypes.c_int32),
    ]


assert _ctypes.sizeof(_MeshEntry) == 304, _ctypes.sizeof(_MeshEntry)

# MSL layout contract (prelude _MESH_QUERY_OPS struct wp_mesh): the embedded
# BVH starts at byte 176 — mesh_get_bvh routes ids to &entry.bvh via exactly
# this offset.
_MESH_BVH_OFF = getattr(_MeshEntry, "bvh").offset
assert _MESH_BVH_OFF == 176, _MESH_BVH_OFF


# the device-side mesh table: one _MeshEntry per registered wp.Mesh (index ==
# mesh id; entry 0 is the null mesh). Kernels reaching wp.mesh_get / the mesh
# query family bind it as `device wp_mesh* _wp_mesh_table`. Rebuilt
# (re-uploaded) whenever a mesh is created or destroyed — build-time-only work.
_MESH_TABLE: dict = {"handle": None, "size": 0}


class _FaceBounds:
    """numpy master + shared-storage Metal buffer pair for the per-triangle
    bounds arrays (the mesh's lowers/uppers — also the embedded BVH's item
    arrays, matching upstream mesh.cpp which builds the BVH directly over
    m->lowers/m->uppers). Implements the .numpy()/.buffer surface HostBVH
    expects from a caller array."""

    def __init__(self, mm, data: np.ndarray):
        self._mm = mm
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self.buffer = mm.buffer_create(max(self._data.nbytes, 16))
        self.flush()

    def numpy(self) -> np.ndarray:
        return self._data

    def flush(self) -> None:
        dst = (_ctypes.c_char * self._data.nbytes).from_address(self._mm.buffer_ptr(self.buffer))
        _ctypes.memmove(dst, self._data.view(np.uint8).reshape(-1).tobytes(), self._data.nbytes)

    def release(self) -> None:
        if self.buffer:
            self._mm.buffer_release(self.buffer)
            self.buffer = 0


def _mesh_meta(arr) -> _ArrMeta:
    """Meta for a WarpArray or a vendored frontend ``array_t`` (mesh data)."""
    if hasattr(arr, "buffer"):  # runtime WarpArray (incl. _FaceBounds)
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


def _mesh_array_payload(arr):
    """(gpu address, meta) for a mesh record array; None arrays stay null."""
    if arr is None:
        return 0, None
    return _mesh_data_addr(arr), _mesh_meta(arr)


def _mesh_assets_build(rec: dict) -> None:
    """Per-triangle bounds + embedded-layout BVH for a mesh record (upstream
    mesh.cpp mesh_create_host port): triangle bounds from the points/indices,
    HostBVH over those bounds (constructor/groups/leaf_size verbatim), and
    average_edge_length = total edge length / (num_tris * 3)."""
    from .bvh import HostBVH

    points, indices = rec["points"], rec["indices"]
    num_tris = rec["num_tris"]
    num_points = rec["num_points"]
    pts = points.numpy().reshape(-1, 3)[:num_points]
    idx = indices.numpy().ravel()[: num_tris * 3].astype(np.int64)
    tri = pts[idx.reshape(num_tris, 3)] if num_tris else np.zeros((0, 3, 3), dtype=np.float32)

    lowers = np.ascontiguousarray(tri.min(axis=1), dtype=np.float32)
    uppers = np.ascontiguousarray(tri.max(axis=1), dtype=np.float32)
    lb = _FaceBounds(_mm, lowers)
    ub = _FaceBounds(_mm, uppers)
    bvh = HostBVH(lb, ub, rec.get("groups"), rec["bvh_constructor"], rec["bvh_leaf_size"], _mm)

    # edge-length sum (mesh.cpp verbatim: per-triangle 3 edges, summed, then
    # divided once); 0 tris mirrors the C++ 0.0f/0.0f NaN
    if num_tris:
        p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
        el = np.sqrt(((p0 - p1) ** 2).sum(axis=1)) + np.sqrt(((p0 - p2) ** 2).sum(axis=1)) + np.sqrt(((p2 - p1) ** 2).sum(axis=1))
        avg = np.float32(el.sum() / np.float32(num_tris * 3))
    else:
        avg = np.float32("nan")

    rec["face_lowers"] = lb
    rec["face_uppers"] = ub
    rec["bvh"] = bvh
    rec["average_edge_length"] = float(avg)


def _mesh_assets_refit(rec: dict) -> None:
    """mesh.cpp mesh_refit_host port: recompute the per-triangle bounds from
    the CURRENT points, then refit the tree over them (node buffers re-uploaded
    in place; the embedded context is address-stable, no table rebuild).
    Flushes first: the bounds/BVH host writes must not land under pending
    eager dispatches that dereference the same buffers."""
    _mm.flush()
    points, indices = rec["points"], rec["indices"]
    num_tris, num_points = rec["num_tris"], rec["num_points"]
    pts = points.numpy().reshape(-1, 3)[:num_points]
    idx = indices.numpy().ravel()[: num_tris * 3].astype(np.int64)
    tri = pts[idx.reshape(num_tris, 3)] if num_tris else np.zeros((0, 3, 3), dtype=np.float32)

    lb: _FaceBounds = rec["face_lowers"]
    ub: _FaceBounds = rec["face_uppers"]
    lb.numpy()[:] = tri.min(axis=1)
    ub.numpy()[:] = tri.max(axis=1)
    lb.flush()
    ub.flush()
    rec["bvh"].refit()


def _mesh_assets_release(rec: dict) -> None:
    """Release the record's face-bounds buffers and BVH node buffers (each
    leaves the residency set in native buffer_release)."""
    if rec.get("face_lowers") is not None:
        rec["face_lowers"].release()
        rec["face_uppers"].release()
        rec["bvh"].destroy()
    rec["face_lowers"] = rec["face_uppers"] = rec["bvh"] = None


def _fill_mesh_entry(entry: _MeshEntry, rec: dict) -> None:
    for field in ("points", "velocities", "indices"):
        addr, meta = _mesh_array_payload(rec.get(field))
        slot = getattr(entry, field)
        slot.data = addr
        if meta is not None:
            slot.m = meta
    entry.lowers = _mm.buffer_gpu_address(rec["face_lowers"].buffer)
    entry.uppers = _mm.buffer_gpu_address(rec["face_uppers"].buffer)
    entry.solid_angle_props = 0  # winding-number family is fail-loud (M2.2)
    entry.num_points = rec["num_points"]
    entry.num_tris = rec["num_tris"]
    entry.context = 0
    entry.average_edge_length = rec["average_edge_length"]
    # embedded wp_bvh_t image (upstream Mesh embeds the BVH by value)
    ctx = rec["bvh"].embedded_context(_mm).view(np.uint8).reshape(-1).tobytes()
    dst = (_ctypes.c_char * 112).from_address(_ctypes.addressof(entry) + _MESH_BVH_OFF)
    _ctypes.memmove(dst, ctx, 112)


def _mesh_table_upload() -> int:
    """(Re)build the device mesh table; returns the buffer handle."""
    from ..frontend.registry import _CoreShim

    meshes = _CoreShim._HOST_MESHES
    n = _CoreShim._HOST_MESH_SEQ[0] + 1
    entries = (_MeshEntry * n)()
    for mid, rec in meshes.items():
        _fill_mesh_entry(entries[mid], rec)
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
    strides = arr.strides  # bytes per dim (views carry their own; device
    # wp_address indexes a char* with these — warp native array.h semantics)
    meta = _ArrMeta()
    meta.offset = arr._meta_offset_bytes()
    meta.ndim = ndim
    for i in range(4):
        meta.shape[i] = shape[i] if i < ndim else 0
        meta.strides[i] = int(strides[i]) if i < ndim else 0
    return meta


def _unit_for(kernel, block_dim: int | None = None):
    """Translate + compile + cache a PSO for a vendored Kernel.

    ``block_dim`` keys the cache alongside the kernel: it bakes parse-time
    tile shapes and wp.block_dim() folding into the MSL — callers pass it
    ONLY for launch_tiled (the sole per-block shared-semantics path). Plain
    launches arrive with ``block_dim=None`` regardless of what the caller
    supplied: their tile ops lower per-thread scalar (one-lane blocks, the
    CPU-golden semantics), producing launch-layout-independent MSL. The tid
    mapping is routing-independent (runtime decode off ``_dims``), so every
    plain launch of a kernel shares one PSO.
    """
    with _PSO_LOCK:
        cached = _PSO_CACHE.get((id(kernel), block_dim))
        if cached is not None:
            return cached
        opts = {"block_dim": block_dim} if block_dim is not None else None
        try:
            unit = translate_kernel(kernel, opts)
            source = unit.source()
        except Exception as e:
            # fail loud naming the kernel: the translation error itself names
            # only the offending construct (e.g. an fp64 declaration with no
            # MSL mapping); annotate it with the kernel identity without
            # changing the exception type (tests match on both)
            key = getattr(kernel, "key", getattr(kernel, "__name__", "<kernel>"))
            module = getattr(getattr(kernel, "func", None), "__module__", "<spike>")
            if e.args:
                e.args = (f"{e.args[0]}\ntranslating kernel '{key}' ({module})", *e.args[1:])
            raise
        dump_dir = os.environ.get("WARP_METAL_DUMP_MSL")
        if dump_dir:
            # debugging aid: the Metal diagnostics buffer clips long error
            # lists — dumping the exact MSL lets a standalone
            # newLibraryWithSource probe recover the full diagnostics
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"{kernel.__name__ or 'kernel'}.msl")
            with open(dump_path, "w") as f:
                f.write(source)
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


def _launch_ssa(kernel, dim, merged, block_dim: int | None = None, tiled: bool = False) -> None:
    """Launch a vendored Kernel through the Adjoint-SSA MSL pipeline.

    Routing (upstream context.py semantics):
    - plain launch (``tiled=False``): total threads = prod(dim); ``block_dim``
      is only the threadgroup scheduling size (CUDA's threads-per-block) and
      never multiplies the thread count.
    - tiled launch (``tiled=True``): ``block_dim`` is appended as the trailing
      launch dimension (``launch_tiled`` semantics), so the grid is
      prod(dim)*block_dim threads in block_dim-sized threadgroups and the
      kernel's tid mapping reads the block_dim off ``_dims``.
    """
    if getattr(kernel, "is_generic", False):
        kernel = _concretize_generic_kernel(kernel, merged)
    from .. import host_fallback

    fb = host_fallback.get(kernel)
    if fb is not None:
        if tiled:
            raise RuntimeError(
                f"launch_tiled: kernel '{kernel.key}' has a CPU fallback registered but tiled "
                "ports are not supported (fallback ports take a flat thread count)"
            )
        # ports take a flat thread count; a grid tuple flattens row-major
        # (cpu launch_tiled routes here as a plain launch with block_dim=1,
        # so prod(dims) is exactly its one-thread-per-block count)
        flat_dim = dim
        if isinstance(dim, (tuple, list)):
            flat_dim = 1
            for d in dim:
                flat_dim *= int(d)
        if _CAPTURE_RECORDS is not None:
            # record the host port: replay re-runs it on the same buffers
            _CAPTURE_RECORDS.append(("fallback", kernel, flat_dim, list(merged), fb))
            return
        # closure-external kernel with an explicit verified CPU fallback
        # (unified memory: same shared-storage buffers the device sees)
        fb(kernel, flat_dim, merged)
        return

    # launch bounds: for a tiled launch block_dim joins as the trailing dim
    # (upstream launch_tiled appends it), then threads = prod(bounds).
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    dims = tuple(int(d) for d in dims)
    if any(d < 0 for d in dims):
        raise ValueError("Launch dimensions must be non-negative")
    # upstream plain-launch-with-block_dim semantics (context.py launch: CPU
    # invoke_cpu_blocks / CUDA blockDim.x): the trailing launch dim is the
    # block axis — mujoco_warp's sparse JTDAJ launches
    # dim=(nworld, groups, threads_per_group) with block_dim ==
    # threads_per_group. Keep the dims UNCHANGED (the tid helpers read the
    # trailing coordinate as the tile lane off _dims, so the mapping equals
    # upstream's exactly) and only switch the lowering/dispatch to the
    # blocked launch_tiled form keyed by block_dim.
    blocked = (
        not tiled and block_dim is not None and block_dim > 1 and bool(dims) and dims[-1] == block_dim
    )
    if tiled:
        dims = dims + (block_dim,)
    threads = 1
    for d in dims:
        threads *= d
    if threads == 0:
        # upstream launch() skips entirely when total_dim_size == 0
        return
    # translation key: plain launches without a block_dim kwarg lower every
    # tile per-thread scalar (one-lane blocks — the CPU-golden semantics
    # upstream forces on the CPU path), so their MSL does not depend on the
    # requested block_dim at all; launch_tiled AND the blocked plain launch
    # (trailing dim == block_dim, above) bake the per-block shared lowering
    # keyed by block_dim
    pso, unit = _unit_for(kernel, block_dim if (tiled or blocked) else None)
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

    # _dims is the launch-bounds image the device tid helpers read (upstream
    # launch_bounds_t): for tiled launches it includes the trailing
    # block_dim, so scalar wp.tid() resolves to the block index and the 2-D
    # form to (block, lane).
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
            _CAPTURE_RECORDS.append(
                ("metal", pso, list(handles), threads, block_dim, owned, array_refs, tiled or blocked)
            )
            return
        import os as _os

        if _os.environ.get("WARP_METAL_DEBUG_LAUNCH"):
            print(
                f"[wmm-launch] kernel={getattr(kernel, 'key', kernel)} threads={threads} "
                f"block_dim={block_dim} tiled={tiled} n_buffers={len(handles)} "
                f"mesh={getattr(unit, 'needs_mesh', False)}",
                flush=True,
            )
        if tiled or blocked:
            _mm.dispatch_tiled(pso, handles, threads, block_dim)
        else:
            # tg 0 = native auto threadgroup size. Plain launches never bake
            # a block layout into the MSL (per-thread scalar tile lowering,
            # no shared arrays, no barriers), so the auto choice is always
            # correct and no scheduling hint is needed
            _mm.dispatch(pso, handles, threads, 0)
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

    ``dim`` is the BLOCK-grid shape (int, or a 1-3 dim sequence); upstream
    appends ``block_dim`` as the trailing launch dimension, so the grid runs
    prod(dim)*block_dim threads in block_dim-sized threadgroups and the
    kernel reads ``wp.tid()`` as the block coordinate (scalar = block index,
    2-D = ``(block_index, lane)``) with ``wp.block_dim()`` folding to
    ``block_dim`` (default 256, matching upstream's default). The vendored
    parse consumes ``block_dim`` for tile shapes, and the emitter lowers
    tile_reduce to a barriered threadgroup tree reduction — which requires a
    power-of-two ``block_dim`` (upstream warp makes the same assumption for
    its register-tile layout).
    """
    if adjoint or record_tape:
        raise RuntimeError(
            "warp-metal does not support adjoint launches (adjoint codegen is a KEP-0001 Non-Goal)"
        )
    dev_name = get_device(device).name if device is not None else "metal"
    if dev_name not in ("metal", "cpu"):
        raise RuntimeError(
            f"launch_tiled device={device!r} is not available: the Metal backend dispatches on 'metal'"
        )
    if stream is not None or record_cmd is not None or max_blocks is not None:
        # accepted-for-parity in launch(); there is no batching here at all,
        # so tiled variants fail loud instead of ignoring silently
        raise RuntimeError(
            "launch_tiled does not accept stream/record_cmd/max_blocks "
            "(deferred eager dispatch flushes at host-visible reads; stream batching is a Non-Goal)"
        )
    if dim is None:
        raise TypeError("launch_tiled() missing required argument 'dim'")

    if isinstance(dim, (tuple, list)):
        dims = tuple(int(d) for d in dim)
        if not 1 <= len(dims) <= 3:
            # upstream: "wp.launch_tiled() requires a grid with fewer than 4 dimensions"
            raise ValueError(f"launch_tiled dim must be an int or a 1-3 dim grid, got {dim!r}")
    else:
        dims = (int(dim),)

    if dev_name == "cpu":
        # CPU-golden semantics: upstream resolves every CPU block_dim request
        # to 1 (context.py _resolve_launch_block_dim — values >1 are opt-in
        # behind wp.config.enable_cpu_blocks, and launch_tiled appends the
        # resolved value as the trailing launch dim), so a tiled launch on
        # cpu is prod(dims) one-lane blocks. That is definitionally the
        # per-thread scalar tile lowering plain launches use (same module:
        # _launch_ssa), and every array lives in shared storage, so the
        # port routing + MSL pipeline inside _launch_ssa apply unchanged —
        # kernels with a verified CPU port (host_fallback) run their port,
        # everything else dispatches through the per-thread-scalar MSL.
        # Upstream only bounds the CPU request: "block_dim must be at most
        # 1024 on CPU" (_resolve_launch_block_dim).
        block_dim = int(block_dim) if block_dim is not None else 1
        if block_dim > 1024:
            raise ValueError(f"block_dim must be at most 1024 on CPU, got {block_dim}")
        if block_dim > 1:
            from ..frontend.config import enable_cpu_blocks

            if enable_cpu_blocks:
                # upstream would run cooperative CPU fibers here; warp-metal has
                # no fiber execution, so fail loud instead of silently forcing 1
                raise NotImplementedError(
                    "wp.config.enable_cpu_blocks CPU fibers are not supported by the Metal backend; "
                    "cpu launches run one-lane blocks (upstream's default resolution)"
                )
        merged = list(inputs or []) + list(outputs or [])
        _launch_ssa(kernel, dims, merged, block_dim=1, tiled=False)
        return

    if block_dim is None:
        block_dim = 256  # upstream default (warp.config block dim)
    block_dim = int(block_dim)
    if block_dim & (block_dim - 1):
        raise ValueError(f"launch_tiled block_dim must be a power of two, got {block_dim}")

    merged = list(inputs or []) + list(outputs or [])
    _launch_ssa(kernel, dims, merged, block_dim=block_dim, tiled=True)


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
    Non-Goal); ``stream``/``record_cmd``/``max_blocks`` are accepted for API
    parity (eager dispatches encode into the pending command pipeline and
    flush at host-visible reads; stream batching is a Non-Goal). ``dim`` may
    be an int or a tuple: total threads = prod(dim), exactly upstream. An
    explicit ``block_dim`` on a plain launch selects per-thread tile
    semantics (every tile op collapses to the thread's own element — the
    CPU-golden convention); per-block shared tile semantics require
    ``launch_tiled``. Scalar kernel params are passed interleaved in
    ``inputs``/``outputs`` in declaration order, exactly like upstream warp.
    """
    if adj_inputs is not None or adj_outputs is not None or adjoint or record_tape:
        raise RuntimeError(
            "warp-metal does not support adjoint launches (adjoint codegen is a KEP-0001 Non-Goal); "
            "use enable_backward=False / forward-only launches"
        )

    if device is not None and get_device(device).name not in ("metal", "cpu"):
        raise RuntimeError(
            f"launch device={device!r} is not available: the Metal backend dispatches on 'metal'"
        )
    # An explicit "cpu" device runs on the same shared-storage buffers the
    # device dispatches on: upstream CPU launches resolve block_dim to 1
    # (context.py _resolve_launch_block_dim), so the per-thread scalar tile
    # lowering below IS the CPU-golden execution. _launch_ssa routes kernels
    # with a verified CPU port (host_fallback) to their port; everything else
    # lowers to MSL over the unified-memory arrays.

    if dim is None:
        raise TypeError("launch() missing required argument 'dim'")

    merged = list(inputs or []) + list(outputs or [])

    if hasattr(kernel, "adj"):  # vendored Kernel (M1.2+): Adjoint-SSA pipeline
        _launch_ssa(kernel, dim, merged, block_dim=block_dim, tiled=False)
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

    _mm.dispatch(pso, handles, dim, 0)  # 0 = auto threadgroup size (spike kernels are 1-D)
    for h in temp:
        _mm.buffer_release(h)


def get_suggested_block_size(kernel, device: DeviceLike = None) -> tuple[int, int]:
    """Suggest launch geometry for ``kernel`` (upstream context.py parity).

    mujoco_warp's collision_driver/collision_convex/solver launch paths call
    ``wp.get_suggested_block_size(kernel[, device])`` before launching.
    Upstream queries CUDA occupancy (``cuOccupancyMaxPotentialBlockSize``)
    and, for any non-CUDA device, returns ``(1, 1)`` — the shim has no CUDA,
    so every resolved device takes that upstream non-CUDA branch, matching
    the CPU-golden one-lane block resolution the launch surface already
    applies (``launch_tiled`` cpu path, context.py
    ``_resolve_launch_block_dim``). The kernel type check and device
    resolution still run first so a bad kernel or device identifier fails
    loud exactly like upstream. Upstream calls ``init()`` before resolving;
    the shim skips that (device resolution here is lazy), so CPU-only hosts
    and host-only test processes can call this without creating a Metal
    device. Upstream's generic-kernel rejection sits behind the non-CUDA
    return, so generic overloads are accepted here too — same as upstream
    on a CPU device. There is deliberately no ``get_suggested_block_count``:
    that name does not exist in upstream warp 1.19 nor in mujoco_warp.
    """
    from ..frontend.kernels import Kernel

    if not isinstance(kernel, Kernel):
        raise TypeError(f"get_suggested_block_size() expected a wp.Kernel, got {type(kernel)}")

    dev = get_device(device)  # ValueError on unknown identifiers, upstream parity
    if not dev.is_cuda:
        return (1, 1)

    # Unreachable today: every warp-metal device reports is_cuda False
    # (runtime/devices.py). If a device ever reports CUDA-capable, fail loud
    # rather than guess — there is no Metal analogue of the CUDA occupancy
    # query this function mirrors on CUDA hardware.
    raise RuntimeError(
        f"get_suggested_block_size() has no occupancy analogue for device '{dev}' (warp-metal is Metal-only)"
    )


# ---------------------------------------------------------------------------
# graph capture (record/replay): ops inside capture_begin..capture_end are
# appended to a record list instead of executing; capture_launch re-executes
# the records in order, batching consecutive eligible records (kernels,
# array copies, zero fills) into one command buffer per run — compute
# sections plus blit sections, one commit + one watchdog wait per batch
# ---------------------------------------------------------------------------


class CaptureStateError(RuntimeError):
    """A graph construct needs CUDA-conditional-node machinery that the
    active warp-metal record/replay capture cannot express (KEP-0001 M2.5:
    raised by ``capture_if`` inside an active capture; ``capture_while``
    keeps the blanket fail-loud everywhere)."""


class Graph:
    """Replayable capture record.

    Capture records each launch (Metal dispatch or CPU-fallback port) in
    order; ``capture_launch`` replays them, batching consecutive eligible
    records into one Metal command buffer (kernels as compute sections,
    eligible copies / zero fills as blit sections — one commit + watchdog
    wait per batch) and applying host records between batches in record
    order. The batch segments are PRE-MARSHALED into native
    ``wmm_batch_record`` payloads once (at ``capture_end`` — the records are
    immutable from there; M2.6), so every replay is one pre-marshaled native
    call per batch.
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
        # pre-marshaled replay segments: [("batch", BatchPayload, base_index)
        # | ("host", record), ...] — built at capture_end (or lazily at the
        # first replay); None = not built yet
        self._segments: list | None = None
        self._segments_len = -1

    def _prebuild_replay(self) -> None:
        """Pre-marshal the batched replay path (M2.6).

        Walks the records once, classifying each as batch-eligible (compute
        section, blit copy / zero fill — the same eligibility matrix as the
        per-replay walk had) or a host-record batch splitter, and marshals
        each eligible run into a native payload reused at every replay. The
        WARP_METAL_BATCH=0 escape hatch bypasses these segments entirely
        (it replays record-by-record through the synchronous entries).
        """
        segments: list = []
        batch: list = []
        batch_base = 0
        records = self._records
        for index, record in enumerate(records):
            kind = record[0]
            if kind == "metal":
                _, pso, handles, threads, block_dim, _owned, _refs, tiled = record
                batch.append(
                    ("compute", pso, handles, threads,
                     block_dim if tiled else 0, 1 if tiled else 0)
                )
                continue
            blit = _blit_payload(record) if kind in ("copy", "fill") else None
            if blit is not None:
                batch.append(blit)
                continue
            if batch:
                segments.append(("batch", _mm.make_batch_payload(batch), batch_base))
                batch = []
            if kind not in ("copy", "fill", "fallback"):
                raise RuntimeError(f"capture_launch: unknown record kind {kind!r}")
            segments.append(("host", record))
            batch_base = index + 1
        if batch:
            segments.append(("batch", _mm.make_batch_payload(batch), batch_base))
        self._segments = segments
        self._segments_len = len(records)

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
    of scope; timing lands on this process's wall clock. Eager dispatches
    flush at the timed section's host-visible reads, so a completed section
    includes their execution.
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

    Record/replay (upstream CUDA graph semantics; KEP-0001 M2.5). Kernel
    launches, ``zero_``/``fill_``, ``wp.copy``/``clone`` and CPU-fallback
    launches inside the block are appended to ``c.graph`` in order and are
    NOT executed here (deferred until replay, like a CUDA capture).
    ``wp.capture_launch(c.graph)`` executes the records: consecutive
    batch-eligible records replay through one Metal command buffer (single
    commit + watchdog wait per run) — kernels as compute sections, eligible
    copies and zero fills as blit sections on the same command buffer.
    Non-zero fills, copies that fail blit eligibility (overlapping ranges,
    cross-device) and fallback ports are host ops that split the batch.
    One documented divergence from upstream: a fill also writes its array
    immediately at record time (shared storage makes that harmless, and
    replay re-runs it in order).
    """

    def __init__(self, device=None, stream=None, force_module_load=False, **kwargs):
        if kwargs:
            raise RuntimeError(f"ScopedCapture does not support {sorted(kwargs)} (KEP-0001 record/replay capture)")
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
# active record list while capturing; None outside capture (aliases the
# in-progress Graph's record list)
_CAPTURE_RECORDS: list | None = None
# the Graph being recorded (created at capture_begin so the device
# capture-parity attributes can expose it, upstream
# ``device.captures[stream]``-style) and the device the capture was begun on
_CAPTURE_GRAPH: Graph | None = None
_CAPTURE_DEVICE = None


def active_capture(device=None) -> Graph | None:
    """The Graph currently being recorded for ``device``'s implicit stream,
    else ``None``.

    Runtime-internal accessor and single source of truth behind
    ``Device.is_capturing``/``Device.captures``/``Stream.is_capturing``
    (KEP-0001 M2.5 commit 3). Warp-metal allows one non-nested capture,
    begun on the device passed to (or current at) ``capture_begin``; like
    upstream's per-device ``captures`` registration, it reports only on that
    device. With ``device=None`` the active capture is reported regardless
    of the device it was begun on.
    """
    with _CAPTURE_LOCK:
        if _CAPTURE_DEPTH == 0 or _CAPTURE_GRAPH is None:
            return None
        if (
            device is not None
            and _CAPTURE_DEVICE is not None
            and get_device(device).name != _CAPTURE_DEVICE.name
        ):
            return None
        return _CAPTURE_GRAPH


def _record_fill(arr, value) -> None:
    """Record an in-place fill as a graph op (upstream stream-memset semantics).

    The record holds the array itself, so its buffer stays alive for the
    graph's lifetime. Host-only fills skipped outside capture.
    """
    if _CAPTURE_RECORDS is not None:
        _CAPTURE_RECORDS.append(("fill", arr, value))


def capture_begin(device=None, stream=None, force_module_load=False, external=None, apic=None, capture_mode=None):
    """Begin capture: launches are RECORDED, not executed (upstream graph
    semantics). The in-progress Graph is registered for the target device's
    implicit stream (observable through ``Device.captures``/``Device.is_capturing``);
    replay batches consecutive eligible records into one command buffer.
    Pending eager dispatches flush at the boundary (upstream capture_begin
    syncs the stream; here it guarantees capture-time host writes — e.g.
    recorded fills' immediate host writes — never land under pre-capture
    kernels)."""
    global _CAPTURE_DEPTH, _CAPTURE_RECORDS, _CAPTURE_GRAPH, _CAPTURE_DEVICE
    if external is not None or apic:
        raise RuntimeError("warp-metal does not support external/apic capture (CUDA-graph machinery)")
    _mm.flush()
    with _CAPTURE_LOCK:
        if _CAPTURE_DEPTH == 0:
            dev = get_device(device)
            graph = Graph(dev.name)
            graph._records = []
            _CAPTURE_GRAPH = graph
            _CAPTURE_RECORDS = graph._records
            _CAPTURE_DEVICE = dev
        else:
            raise RuntimeError("nested capture is not supported by warp-metal")
        _CAPTURE_DEPTH += 1


def capture_end(device=None, stream=None) -> Graph:
    """End capture; returns the recorded graph — the same Graph object the
    device's ``captures`` view exposed while recording. The graph's replay
    batches are pre-marshaled here (records are immutable from this point)."""
    global _CAPTURE_DEPTH, _CAPTURE_RECORDS, _CAPTURE_GRAPH, _CAPTURE_DEVICE
    _mm.flush()  # boundary: nothing pending may straddle capture end
    with _CAPTURE_LOCK:
        _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)
        graph = _CAPTURE_GRAPH
        records = _CAPTURE_RECORDS
        _CAPTURE_RECORDS = None
        _CAPTURE_GRAPH = None
        _CAPTURE_DEVICE = None
    if graph is None:
        # capture_end without capture_begin: keep the historical lenient
        # empty-graph return (upstream raises here)
        return Graph(get_device(device).name if device is not None else None)
    graph.n_ops = len(records) if records else 0
    graph._prebuild_replay()
    return graph


def _batching_enabled() -> bool:
    """``WARP_METAL_BATCH=0`` replays record-by-record (one commit+wait per
    dispatch, the pre-batching path) for fault triage; any other value — or
    unset — batches consecutive records into one command buffer."""
    return os.environ.get("WARP_METAL_BATCH", "") != "0"


def _sync_eager() -> bool:
    """Python-side mirror of the native SYNC_EAGER check.

    Read per call, exactly the way the native dispatch entries read the
    environment, so eager host ops and the deferred GPU encodes can never
    disagree about the active mode within one call sequence.
    """
    s = os.environ.get("WARP_METAL_SYNC_EAGER")
    return bool(s) and s != "0"


def _eager_gpu_fills() -> bool:
    """The eager GPU-encode knob: ``WARP_METAL_EAGER_GPU_FILLS=1`` only.

    Round 2b encoded eager ``zero_``/``fill_``/``copy``/``clone`` as
    compute sections in the pending command buffer; it measures healthy at
    micro scale (tests/test_deferred_micro.py) but is implicated in the
    unexplained ~35x Newton conveyor regression (round-2 lead: flush
    cadence vs collide()'s ``.numpy`` reads), so per KEP D2 the round-1
    ordered host path is the DEFAULT and this knob opts into the GPU
    encoding. Read per call, exactly like :func:`_sync_eager` (any other
    value — unset included — keeps the host path), so in-process tests can
    toggle it; SYNC_EAGER keeps precedence (checked first at every call
    site), so setting both knobs means per-dispatch commit+wait with host
    writes.
    """
    return os.environ.get("WARP_METAL_EAGER_GPU_FILLS") == "1"


def _is_zero_value(value) -> bool:
    """True when ``value`` is an exact zero for the fill eligibility checks.

    ``-0.0`` is excluded: it compares equal to zero but its float bit
    pattern is not all-zero bytes, so a blit fill would corrupt it.
    """
    if isinstance(value, (bool, np.bool_)):
        return not value
    if isinstance(value, (int, np.integer)):
        return value == 0
    if isinstance(value, (float, np.floating)):
        return value == 0.0 and not np.signbit(value)
    return False


def _exact_fill_byte(arr, value) -> int | None:
    """The single 8-bit pattern whose replication over one ``arr`` element
    produces exactly ``value`` (Metal ``fillBuffer`` semantics), or None
    when no exact pattern exists — the fill must then stay on a host path
    (equal-to-zero values) or fail loud (everything else, eager mode).

    Scalar values only: composite/list fills broadcast per component and
    are not byte-replicable. Struct dtypes keep the M2.5 rule (only an
    exact zero matches the default-constructed all-zero bytes).
    """
    np_t = np.dtype(arr.np_dtype)
    if np_t.names:  # struct element: default construction is all-zero bytes
        return 0 if _is_zero_value(value) else None
    if hasattr(value, "__len__"):
        return None
    try:
        raw = np.array(value, dtype=np_t).tobytes()
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None
    if not raw:
        return None
    first = raw[0]
    return first if all(b == first for b in raw) else None


def _blit_copy_payload(dest, src, dest_offset, src_offset, n):
    """GPU-side blit payload for an array→array copy, or None when the copy
    must stay on the host path (splitting a replay batch / keeping the
    eager host copy). Shared by the eager ``wp.copy``/``clone`` GPU path
    and the graph-replay records, so both sides enforce the same
    eligibility matrix. Element offsets/sizes become buffer byte
    offsets/sizes — ``offset_elems`` (sliced/row views share the base
    buffer) folds in on both sides.

    Eligibility (M2.5 commit-2 rule): same device (single-device runtime —
    checked anyway), same dtype/element size (``wp.copy`` enforces equal
    element sizes at record time, and runtime arrays are dense row-major,
    so equal dtype means byte-identical element layout — composite dtypes
    included — and the byte range copy is exact), and no overlapping
    src/dst byte ranges within one buffer (blit aliasing is undefined —
    those go back to the host numpy copy).
    """
    if dest.device != src.device:
        return None
    if dest.dtype != src.dtype or dest.item_bytes != src.item_bytes:
        return None
    if n < 0:
        return None  # unreachable (wp.copy validates); host path anyway
    src_off = (src.offset_elems + src_offset) * src.item_bytes
    dst_off = (dest.offset_elems + dest_offset) * dest.item_bytes
    nbytes = n * dest.item_bytes
    if src.buffer == dest.buffer and (
        src_off < dst_off + nbytes and dst_off < src_off + nbytes
    ):
        return None  # overlapping ranges in one buffer: host copy
    return ("copy", src.buffer, dest.buffer, src_off, dst_off, nbytes)


def _blit_payload(record):
    """GPU-side blit payload for a ``"copy"``/``"fill"`` graph record, or
    None when the record must stay on the host replay path (splitting the
    batch).

    Eligibility (commit-2 brief; anything else segments to the host path,
    preserving the pre-batching behavior):
    - fill: value must be an exact zero (``fillBuffer`` takes an 8-bit
      pattern; a non-zero or ``-0.0`` float pattern is not representable —
      Newton's captured fills are zero re-zeroings). Zero-size ranges are
      eligible but encode nothing. (The eager path additionally accepts
      other byte-exact patterns through ``_exact_fill_byte`` — the replay
      rule stays conservative.)
    - copy: delegated to ``_blit_copy_payload`` (the shared eager/replay
      eligibility).
    """
    kind = record[0]
    if kind == "fill":
        _, arr, value = record
        if not _is_zero_value(value):
            return None
        return ("fill", arr.buffer, arr.offset_elems * arr.item_bytes,
                arr.n * arr.item_bytes, 0)
    if kind == "copy":
        _, dest, src, dest_offset, src_offset, count = record
        n = src.n - src_offset if count is None else int(count)
        return _blit_copy_payload(dest, src, dest_offset, src_offset, n)
    return None


def _replay_batch(batch, base_index: int, total: int) -> None:
    """Replay one run of consecutive batch-eligible records (compute
    sections + blit copy/zero-fill) through a single command buffer
    (``wmm_dispatch_batch``). On native failure, extend the error with the
    graph-record range the run covered (the native message already reports
    the batch size and points at record-by-record replay for isolation)."""
    try:
        _mm.dispatch_batch(batch)
    except _mm.MetalError as e:
        raise _mm.MetalError(
            f"{e} [batched run covered graph records {base_index}.."
            f"{base_index + len(batch) - 1} of {total}]"
        ) from None


def _replay_host_record(record) -> None:
    """Execute one host-path replay record (batch-ineligible copy/fill or a
    CPU-fallback port) — the ops that split the native batches."""
    kind = record[0]
    if kind == "fill":
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


def capture_launch(graph, stream=None):
    """Replay a captured graph: execute every recorded operation in order.

    Consecutive batch-eligible records replay through one command buffer
    (single commit + watchdog wait per run — this is what gives a captured
    Newton steel step its one-commit replay): kernel records encode as
    compute sections, eligible copies and zero fills as blit sections on
    the same command buffer (``_blit_payload`` decides; ineligible copies /
    non-zero fills / fallback ports are host operations that split the
    batch, applying their host side effects between batch waits so record
    order is preserved exactly).
    The batches are pre-marshaled at ``capture_end`` (M2.6): replay loops
    the ready-made native payload segments — no per-record Python marshaling
    on the hot path.
    Pending eager dispatches flush at the boundary first: the replay batch
    runs on its own command queue, so draining the eager pipeline is what
    preserves global order.
    ``WARP_METAL_BATCH=0`` replays record-by-record (one commit+wait per
    dispatch through the synchronous entries, the pre-M2.6 path) for fault
    triage; it bypasses the pre-marshaled segments entirely.
    """
    if isinstance(graph, Graph):
        _mm.flush()
        records = getattr(graph, "_records", [])
        total = len(records)
        if _batching_enabled():
            if graph._segments is None or graph._segments_len != total:
                graph._prebuild_replay()  # lazy path: graphs made before capture_end wiring
            for entry in graph._segments:
                if entry[0] == "batch":
                    _replay_batch(entry[1], entry[2], total)
                else:
                    _replay_host_record(entry[1])
            return
        # escape hatch: the pre-batching path, one commit+wait per record
        for record in records:
            kind = record[0]
            if kind == "metal":
                _, pso, handles, threads, block_dim, _owned, _refs, tiled = record
                if tiled:
                    _mm.dispatch_tiled_sync(pso, handles, threads, block_dim)
                else:
                    _mm.dispatch_sync(pso, handles, threads, 0)
            else:
                _replay_host_record(record)
        return
    raise RuntimeError(f"capture_launch: cannot replay {type(graph).__name__}")


def capture_if(condition, on_true=None, on_false=None, stream=None, **kwargs):
    """Upstream-faithful eager branch outside graph capture; fail-loud inside.

    Outside capture (upstream's own defined non-CUDA behavior): read the
    condition element and invoke the chosen callback eagerly with
    ``**kwargs``. The condition read is a host-visible read of shared
    storage, so pending eager dispatches flush before it (M2.6 flush-on-read). Inside an active capture, upstream CUDA builds a
    conditional graph node, but warp-metal's record/replay capture cannot
    express one — recording the eager read would silently pin the branch to
    its record-time value for every replay — so this raises
    :class:`CaptureStateError`: capture the branch outside the graph (eager
    semantics) or restructure so the condition is read on the host between
    replays. Graph objects as callbacks (replay inside a recorded
    conditional) remain the M2.5 boundary and fail loud.
    """
    if _CAPTURE_RECORDS is not None:
        raise CaptureStateError(
            "wp.capture_if inside an active capture: upstream CUDA builds a "
            "conditional graph node here, but warp-metal's record/replay "
            "capture would have to eager-read the condition once and replay "
            "that fixed branch on every launch. Capture the branch outside "
            "the graph (eager semantics), or restructure so the condition is "
            "evaluated on the host between replays."
        )
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
    """Upstream-faithful eager loop outside graph capture; fail-loud inside.

    Outside capture, upstream's ``capture_while`` takes its no-graph branch
    (context.py: "since no graph is active, just execute the kernels
    directly"): read the condition element on the host each iteration and
    run the callable body until it clears — no CUDA-graph machinery
    involved, and the same branch upstream CPU devices always take. mjw's
    solver reaches ``wp.capture_while`` only outside a capture (Newton never
    capture_begins around mjw steps), so this eager form reproduces the
    upstream CPU semantics exactly. Capture-INTERNAL while nodes stay the
    rejected candidate (KEP-0001 graph_conditional disposition; see
    capture_if) and fail loud, matching the APIC-record path upstream uses
    for CPU captures.
    """
    if _CAPTURE_RECORDS is not None:
        raise CaptureStateError(
            "wp.capture_while inside an active capture: upstream CPU records "
            "an APIC while-op here, but warp-metal's record/replay capture "
            "would have to eager-read the condition once and replay that "
            "fixed iteration count on every replay. Capture the loop outside "
            "the graph (host while-loop between replays)."
        )
    if not isinstance(condition, WarpArray) or condition.numpy.dtype != np.int32:
        raise TypeError("Condition must be a Warp array of int32 with a single element")
    if while_body is None:
        return
    while True:
        if not bool(condition.numpy[0]):
            break
        if callable(while_body):
            while_body(**kwargs)
        else:
            raise TypeError("while_body must be a Callable (Graph replay: Phase 2, M2.5)")


def init() -> None:
    """Validate the vendored frontend sentinel and lazily probe devices.

    Mirrors ``wp.init()`` as an idempotent bootstrap (safe to call many
    times, as Newton does).
    """
    from ..compat import check_vendored_version

    check_vendored_version()
    from . import devices as _devices

    _devices.is_metal_available()
