# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""ctypes binding for the warp-metal native runtime (libwarp_metal.dylib).

Mirrors the C API in warp_metal.mm. Buffers are opaque handles; the Python
side holds numpy views over the shared-storage contents (zero copy).
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path

from .build_native import DYLIB

_LIB = None
_LOCK = threading.Lock()

# wmm_batch_record kind tags (warp_metal.mm); Python payload tuples use the
# readable strings, marshalled here
_KIND = {"compute": 0, "copy": 1, "fill": 2}


class _BatchRecord(ctypes.Structure):
    """ctypes mirror of the native ``wmm_batch_record`` (one replay-batch
    record: a compute dispatch section or a blit copy / zero-fill)."""

    _fields_ = [
        ("pso", ctypes.c_void_p),        # compute: pipeline state
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),  # compute: handle array
        ("src", ctypes.c_void_p),        # blit copy: source buffer
        ("dst", ctypes.c_void_p),        # blit copy/fill: destination buffer
        ("src_off", ctypes.c_uint64),    # blit copy: source byte offset
        ("dst_off", ctypes.c_uint64),    # blit copy/fill: destination byte offset
        ("nbytes", ctypes.c_uint64),     # blit: byte count (0 = encode nothing)
        ("n_buffers", ctypes.c_uint32),  # compute: handle count
        ("threads_x", ctypes.c_uint32),  # compute: grid threads
        ("tg_x", ctypes.c_uint32),       # compute: threadgroup size (0 = auto)
        ("require_tg", ctypes.c_uint8),  # compute: exact-tg flag (tiled)
        ("kind", ctypes.c_uint8),        # _KIND_*
        ("value", ctypes.c_uint8),       # blit fill: 8-bit pattern (0)
    ]


def _load() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        with _LOCK:
            if _LIB is None:
                if not DYLIB.exists():
                    from .build_native import build
                    build()
                lib = ctypes.CDLL(str(DYLIB))
                # ---- signatures ----
                lib.wmm_last_error.restype = ctypes.c_char_p
                lib.wmm_last_error.argtypes = []
                lib.wmm_device_name.restype = ctypes.c_char_p
                lib.wmm_device_name.argtypes = []
                lib.wmm_buffer_create.restype = ctypes.c_void_p
                lib.wmm_buffer_create.argtypes = [ctypes.c_uint64]
                lib.wmm_buffer_ptr.restype = ctypes.c_void_p
                lib.wmm_buffer_ptr.argtypes = [ctypes.c_void_p]
                lib.wmm_buffer_gpu_address.restype = ctypes.c_uint64
                lib.wmm_buffer_gpu_address.argtypes = [ctypes.c_void_p]
                lib.wmm_buffer_length.restype = ctypes.c_uint64
                lib.wmm_buffer_length.argtypes = [ctypes.c_void_p]
                lib.wmm_buffer_release.restype = None
                lib.wmm_buffer_release.argtypes = [ctypes.c_void_p]
                lib.wmm_library_create.restype = ctypes.c_void_p
                lib.wmm_library_create.argtypes = [ctypes.c_char_p]
                lib.wmm_library_release.restype = None
                lib.wmm_library_release.argtypes = [ctypes.c_void_p]
                lib.wmm_function_get.restype = ctypes.c_void_p
                lib.wmm_function_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                lib.wmm_function_release.restype = None
                lib.wmm_function_release.argtypes = [ctypes.c_void_p]
                lib.wmm_pso_create.restype = ctypes.c_void_p
                lib.wmm_pso_create.argtypes = [ctypes.c_void_p]
                lib.wmm_pso_release.restype = None
                lib.wmm_pso_release.argtypes = [ctypes.c_void_p]
                lib.wmm_pso_max_threads_per_group.restype = ctypes.c_uint64
                lib.wmm_pso_max_threads_per_group.argtypes = [ctypes.c_void_p]
                lib.wmm_dispatch.restype = ctypes.c_int
                lib.wmm_dispatch.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                ]
                lib.wmm_dispatch_tiled.restype = ctypes.c_int
                lib.wmm_dispatch_tiled.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                ]
                lib.wmm_dispatch_batch.restype = ctypes.c_int
                lib.wmm_dispatch_batch.argtypes = [
                    ctypes.POINTER(_BatchRecord),  # per-record payloads (kind-tagged)
                    ctypes.c_uint32,               # record count
                ]
                lib.wmm_batch_record_stride.restype = ctypes.c_uint64
                lib.wmm_batch_record_stride.argtypes = []
                # ABI guard: a stride mismatch would silently misread every
                # record payload — fail loud at load instead
                native_stride = int(lib.wmm_batch_record_stride())
                if native_stride != ctypes.sizeof(_BatchRecord):
                    raise RuntimeError(
                        f"wmm_batch_record ABI mismatch: native stride {native_stride}, "
                        f"ctypes mirror {ctypes.sizeof(_BatchRecord)} — rebuild the dylib"
                    )
                lib.wmm_synchronize.restype = None
                lib.wmm_synchronize.argtypes = []
                _LIB = lib
    return _LIB


class MetalError(RuntimeError):
    """Raised when a native runtime call fails."""


def _check(handle):
    if not handle:
        raise MetalError(_load().wmm_last_error().decode())
    return handle


def device_name() -> str:
    name = _load().wmm_device_name()
    if not name:
        raise MetalError(_load().wmm_last_error().decode())
    return name.decode()


def buffer_create(bytes_len: int):
    return _check(_load().wmm_buffer_create(bytes_len))


def buffer_ptr(handle) -> int:
    return _load().wmm_buffer_ptr(handle)


def buffer_gpu_address(handle) -> int:
    return _load().wmm_buffer_gpu_address(handle)


def buffer_length(handle) -> int:
    return _load().wmm_buffer_length(handle)


def buffer_release(handle) -> None:
    _load().wmm_buffer_release(handle)


def library_create(msl_source: str):
    return _check(_load().wmm_library_create(msl_source.encode()))


def library_release(handle) -> None:
    _load().wmm_library_release(handle)


def function_get(lib_handle, name: str):
    return _check(_load().wmm_function_get(lib_handle, name.encode()))


def function_release(handle) -> None:
    _load().wmm_function_release(handle)


def pso_create(fn_handle):
    return _check(_load().wmm_pso_create(fn_handle))


def pso_release(handle) -> None:
    _load().wmm_pso_release(handle)


def pso_max_threads_per_group(pso_handle) -> int:
    return _load().wmm_pso_max_threads_per_group(pso_handle)


def dispatch(pso_handle, buffer_handles, threads_x: int, tg_x: int = 0) -> None:
    """Dispatch ``threads_x`` threads (1-D grid).

    ``tg_x`` is the threadgroup size: 0 (default) picks the PSO maximum
    clamped to the grid; a positive value pins it (a plain launch's
    ``block_dim`` scheduling hint). The completion wait is bounded by the
    dispatch watchdog (WARP_METAL_DISPATCH_TIMEOUT_S, default 120 s; 0
    disables it) — a hung/faulted dispatch raises MetalError instead of
    blocking forever.
    """
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch(pso_handle, arr, len(buffer_handles), threads_x, tg_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def dispatch_tiled(pso_handle, buffer_handles, threads_x: int, tg_x: int) -> None:
    """Tiled dispatch with an exact threadgroup size (wp.launch_tiled).

    Same watchdog-bounded completion wait as :func:`dispatch`."""
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch_tiled(pso_handle, arr, len(buffer_handles), threads_x, tg_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def dispatch_batch(records) -> None:
    """Batched dispatch: one command queue + one command buffer for a whole
    run of kind-tagged records (capture replay):

      - ``("compute", pso, buffer_handles, threads_x, tg_x, require_tg)``
        — a kernel launch, encoded as a compute section;
      - ``("copy", src_handle, dst_handle, src_off_b, dst_off_b, nbytes)``
        — an array-to-array copy, encoded on a blit encoder (byte
        offsets/size, offset_elems already folded in by the caller);
      - ``("fill", dst_handle, off_b, nbytes, 0)`` — a zero fill on a blit
        encoder (``fillBuffer`` takes an 8-bit pattern, so only zero fills
        are batch-eligible; the caller segments anything else).

    Records encode in list order (compute sections merge into one compute
    encoder, blits into one blit encoder; never two open at once) and
    complete under a single commit + watchdog wait
    (WARP_METAL_DISPATCH_TIMEOUT_S, default scaled by the record count; 0
    disables). Sections execute sequentially, so record i observes record
    i-1's writes exactly like per-record dispatch. Per-record attribution
    inside one command buffer is impossible — a failure names the batch;
    replay with ``WARP_METAL_BATCH=0`` (host-runtime escape hatch) to
    isolate the faulting record.
    """
    if not records:
        return
    n = len(records)
    payload = (_BatchRecord * n)()
    handle_arrays = []  # keep per-compute-record handle arrays alive for the call
    for i, rec in enumerate(records):
        kind = rec[0]
        entry = payload[i]
        if kind == "compute":
            _, pso, handles, threads_x, tg_x, require_tg = rec
            harr = (ctypes.c_void_p * len(handles))(*handles)
            handle_arrays.append(harr)
            entry.pso = pso
            entry.buffers = ctypes.cast(harr, ctypes.POINTER(ctypes.c_void_p))
            entry.n_buffers = len(handles)
            entry.threads_x = threads_x
            entry.tg_x = tg_x
            entry.require_tg = require_tg
            entry.kind = _KIND["compute"]
        elif kind == "copy":
            _, src, dst, src_off, dst_off, nbytes = rec
            entry.src = src
            entry.dst = dst
            entry.src_off = src_off
            entry.dst_off = dst_off
            entry.nbytes = nbytes
            entry.kind = _KIND["copy"]
        elif kind == "fill":
            _, dst, off, nbytes, value = rec
            entry.dst = dst
            entry.dst_off = off
            entry.nbytes = nbytes
            entry.value = value
            entry.kind = _KIND["fill"]
        else:
            raise ValueError(f"dispatch_batch: unknown record kind {kind!r}")
    rc = _load().wmm_dispatch_batch(payload, n)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def synchronize() -> None:
    _load().wmm_synchronize()
