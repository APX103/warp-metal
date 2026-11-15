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
                lib.wmm_dispatch_sync.restype = ctypes.c_int
                lib.wmm_dispatch_sync.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                ]
                lib.wmm_dispatch_tiled_sync.restype = ctypes.c_int
                lib.wmm_dispatch_tiled_sync.argtypes = [
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
                lib.wmm_blit_fill.restype = ctypes.c_int
                lib.wmm_blit_fill.argtypes = [
                    ctypes.c_void_p,  # dst buffer
                    ctypes.c_uint64,  # byte offset into dst
                    ctypes.c_uint64,  # byte count
                    ctypes.c_uint8,   # 8-bit fill pattern (replicated)
                ]
                lib.wmm_blit_copy.restype = ctypes.c_int
                lib.wmm_blit_copy.argtypes = [
                    ctypes.c_void_p,  # src buffer
                    ctypes.c_void_p,  # dst buffer
                    ctypes.c_uint64,  # byte offset into src
                    ctypes.c_uint64,  # byte offset into dst
                    ctypes.c_uint64,  # byte count
                ]
                lib.wmm_compute_fill.restype = ctypes.c_int
                lib.wmm_compute_fill.argtypes = [
                    ctypes.c_void_p,  # dst buffer
                    ctypes.c_uint64,  # byte offset into dst
                    ctypes.c_uint64,  # byte count
                    ctypes.c_uint8,   # 8-bit fill pattern (replicated)
                ]
                lib.wmm_compute_copy.restype = ctypes.c_int
                lib.wmm_compute_copy.argtypes = [
                    ctypes.c_void_p,  # src buffer
                    ctypes.c_void_p,  # dst buffer
                    ctypes.c_uint64,  # byte offset into src
                    ctypes.c_uint64,  # byte offset into dst
                    ctypes.c_uint64,  # byte count
                ]
                lib.wmm_synchronize.restype = None
                lib.wmm_synchronize.argtypes = []
                lib.wmm_flush.restype = ctypes.c_int
                lib.wmm_flush.argtypes = []
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
    """Dispatch ``threads_x`` threads (1-D grid) into the eager pipeline.

    Deferred (M2.6): the launch ENCODES into the persistent command buffer
    without waiting; execution happens at the next :func:`flush` (host.py
    flushes at every host-visible read/sync boundary). Every bound buffer is
    pinned native-side until the flush completes, so Python may release its
    handles immediately. ``WARP_METAL_SYNC_EAGER=1`` reverts to the pre-2.6
    per-dispatch commit+wait (triage). The completion wait is bounded by the
    dispatch watchdog (WARP_METAL_DISPATCH_TIMEOUT_S, default 120 s per
    pending launch, scaled by the pending count; 0 disables) — the window
    bounds a whole flush, and a hung/faulted flush raises MetalError instead
    of blocking forever.
    """
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch(pso_handle, arr, len(buffer_handles), threads_x, tg_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def dispatch_tiled(pso_handle, buffer_handles, threads_x: int, tg_x: int) -> None:
    """Tiled dispatch with an exact threadgroup size (wp.launch_tiled).

    Deferred like :func:`dispatch`; the same SYNC_EAGER / watchdog semantics."""
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch_tiled(pso_handle, arr, len(buffer_handles), threads_x, tg_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def dispatch_sync(pso_handle, buffer_handles, threads_x: int, tg_x: int = 0) -> None:
    """Synchronous single dispatch (the pre-M2.6 ``dispatch`` behavior).

    Commits and waits within one watchdog-bounded call; pending deferred
    work is flushed first so ordering never inverts across modes. This is
    the WARP_METAL_BATCH=0 replay-triage entry point."""
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch_sync(pso_handle, arr, len(buffer_handles), threads_x, tg_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def dispatch_tiled_sync(pso_handle, buffer_handles, threads_x: int, tg_x: int) -> None:
    """Synchronous tiled single dispatch (pre-M2.6 ``dispatch_tiled``).

    Exact-threadgroup validation and the same flush-first / watchdog
    semantics as :func:`dispatch_sync`."""
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch_tiled_sync(pso_handle, arr, len(buffer_handles), threads_x, tg_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


class BatchPayload:
    """A pre-marshaled batch payload (M2.6 graph-replay pre-marshaling).

    Holds the native ``wmm_batch_record`` ctypes array plus the per-compute
    handle arrays built ONCE from a payload-record list (typically at
    ``capture_end``, since a Graph's records are immutable then); every
    replay reuses it instead of re-marshaling. Iterates the source payload
    records, so callers and test spies see exactly the tuples
    :func:`dispatch_batch` has always accepted — only the marshaling moved.
    """

    __slots__ = ("records", "payload", "_handle_arrays")

    def __init__(self, records):
        self.records = list(records)
        n = len(self.records)
        payload = (_BatchRecord * n)()
        handle_arrays = []  # keep per-compute-record handle arrays alive for the call
        for i, rec in enumerate(self.records):
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
        self.payload = payload
        self._handle_arrays = handle_arrays

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def __repr__(self):
        return f"warp_metal.native.BatchPayload(n_records={len(self.records)})"


def make_batch_payload(records) -> BatchPayload:
    """Pre-marshal a batch payload-record list into a reusable :class:`BatchPayload`."""
    return BatchPayload(records)


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

    ``records`` is either the payload-record list (marshaled here) or a
    pre-built :class:`BatchPayload` (marshaled once upstream — the M2.6
    replay path). Records encode in list order (compute sections merge into
    one compute encoder, blits into one blit encoder; never two open at
    once) and complete under a single commit + watchdog wait
    (WARP_METAL_DISPATCH_TIMEOUT_S, default scaled by the record count; 0
    disables). Sections execute sequentially, so record i observes record
    i-1's writes exactly like per-record dispatch. Per-record attribution
    inside one command buffer is impossible — a failure names the batch;
    replay with ``WARP_METAL_BATCH=0`` (host-runtime escape hatch) to
    isolate the faulting record.
    """
    if not records:
        return
    if isinstance(records, BatchPayload):
        payload = records.payload
        n = len(records)
    else:
        pre = BatchPayload(records)
        payload = pre.payload
        n = len(pre)
    rc = _load().wmm_dispatch_batch(payload, n)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def blit_fill(dst_handle, offset_bytes: int, nbytes: int, value: int) -> None:
    """Eager GPU-encoded fill — blit TRIAGE/MEASURE entry (M2.6 round 2b).

    Encodes one ``fillBuffer:range:value:`` blit section into the pending
    command buffer without committing. OFF THE PRODUCTION EAGER PATH since
    round 2b — the conveyor regression diagnosis kept this encoding only so
    :mod:`warp_metal` tests can keep measuring the blit mechanism against
    the knob-gated compute-section encoding (:func:`compute_fill` behind
    ``WARP_METAL_EAGER_GPU_FILLS=1``; microbench configs B/B2/E in
    tests/test_deferred_micro.py).

    Upstream semantics are unchanged either way: ``zero_``/``fill_`` are
    device-side memsets on the stream, so the section orders with the
    launches around it — a later launch observes the fill with no host
    flush, and host reads go through the usual flush-on-read. ``value`` is
    the replicated 8-bit pattern — the Python caller has already validated
    it against the array's element bytes. Zero-size fills encode nothing.
    Every referenced buffer is pinned native-side until the flush
    completes, like compute dispatches. Deferred mode only
    (``WARP_METAL_SYNC_EAGER=1`` keeps the host write, host.py's call)."""
    rc = _load().wmm_blit_fill(dst_handle, offset_bytes, nbytes, value)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def blit_copy(src_handle, dst_handle, src_off_bytes: int, dst_off_bytes: int, nbytes: int) -> None:
    """Eager GPU-encoded array→array copy — blit TRIAGE/MEASURE entry
    (M2.6 round 2b; the knob-gated encoding is :func:`compute_copy`).

    Encodes one ``copyFromBuffer:sourceOffset:toBuffer:destinationOffset:
    size:`` blit section into the pending command buffer without
    committing. Byte offsets/size; eligibility (equal dtype/element size,
    same device, no same-buffer overlap) is the Python caller's — the
    native side re-checks bounds and aliasing fail-loud. Zero-size copies
    encode nothing. Pins both buffers until the flush completes, like
    compute dispatches. Deferred mode only (``WARP_METAL_SYNC_EAGER=1``
    keeps the host copy)."""
    rc = _load().wmm_blit_copy(src_handle, dst_handle, src_off_bytes, dst_off_bytes, nbytes)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def compute_fill(dst_handle, offset_bytes: int, nbytes: int, value: int) -> None:
    """Eager GPU-encoded fill as a COMPUTE section (M2.6 round 2b — the
    eager ``zero_``/``fill_`` GPU encoding, OPT-IN via
    ``WARP_METAL_EAGER_GPU_FILLS=1`` since round 2c: the eager default is
    host.py's ordered host write, the conveyor-safe round-1 path).

    Encodes a tiny fill-kernel dispatch into the SAME compute encoder as
    the launches around it — no encoder transition, no blit/DMA engine —
    with the fill value riding the command stream via ``setBytes``. Stream
    semantics are exactly the blit encoding's: the section orders with the
    launches around it (a later launch observes the fill with no host
    flush), host reads go through the usual flush-on-read. One thread per
    replicated byte (per 4-byte word on the aligned fast path); the PSO is
    compiled lazily once per process. ``value`` is the replicated 8-bit
    pattern — the Python caller has already validated it against the
    array's element bytes. Zero-size fills encode nothing. Every referenced
    buffer is pinned native-side until the flush completes, like compute
    dispatches. Deferred mode only (``WARP_METAL_SYNC_EAGER=1`` keeps the
    host write, host.py's call)."""
    rc = _load().wmm_compute_fill(dst_handle, offset_bytes, nbytes, value)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def compute_copy(src_handle, dst_handle, src_off_bytes: int, dst_off_bytes: int, nbytes: int) -> None:
    """Eager GPU-encoded array→array copy as a COMPUTE section (M2.6 round
    2b — the eager ``wp.copy``/``clone`` GPU encoding, OPT-IN via
    ``WARP_METAL_EAGER_GPU_FILLS=1`` since round 2c; the eager default is
    host.py's ordered host copy).

    Encodes a tiny copy-kernel dispatch into the SAME compute encoder as
    the launches around it — no encoder transition, no blit/DMA engine —
    with the byte offsets riding the command stream via ``setBytes``.
    Upstream ``wp.copy`` array→array is a stream memcpy: the section orders
    with the launches around it, killing the copy's host flush. Byte
    offsets/size; eligibility (equal dtype/element size, same device, no
    same-buffer overlap) is the Python caller's — the native side re-checks
    bounds and aliasing fail-loud (per-byte thread copies have arbitrary
    per-byte read/write order, so overlap stays on the host path). Zero-size
    copies encode nothing. Pins both buffers until the flush completes,
    like compute dispatches. Deferred mode only (``WARP_METAL_SYNC_EAGER=1``
    keeps the host copy)."""
    rc = _load().wmm_compute_copy(src_handle, dst_handle, src_off_bytes, dst_off_bytes, nbytes)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def flush() -> None:
    """Commit the pending eager command buffer and wait for completion.

    Watchdog-bounded (WARP_METAL_DISPATCH_TIMEOUT_S bounds the whole flush
    window; the error names how many dispatches were pending). No-op when
    nothing is pending. host.py calls this at every host-visible
    read/sync boundary; WARP_METAL_SYNC_EAGER=1 makes it a permanent no-op
    (dispatches then commit+wait per call instead)."""
    rc = _load().wmm_flush()
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def synchronize() -> None:
    """Flush the eager pipeline, raising on failure (legacy name — see :func:`flush`)."""
    flush()
