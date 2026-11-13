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
                ]
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


def dispatch(pso_handle, buffer_handles, threads_x: int) -> None:
    arr = (ctypes.c_void_p * len(buffer_handles))(*buffer_handles)
    rc = _load().wmm_dispatch(pso_handle, arr, len(buffer_handles), threads_x)
    if rc != 0:
        raise MetalError(_load().wmm_last_error().decode())


def synchronize() -> None:
    _load().wmm_synchronize()
