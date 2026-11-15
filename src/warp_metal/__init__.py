# Copyright 2026 Jaylin Lee
# Portions Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""warp-metal: Warp-compatible DSL with a native Apple Metal (MSL) backend.

Entry point is ``import warp_metal as wp`` (see :mod:`warp_metal.compat`).

Components (KEP-0001 "库形态与组件边界"):

- ``warp_metal.frontend`` — vendored Warp compiler frontend (frozen at 1.19.0)
- ``warp_metal.msl``      — MSL code generator (Warp AST → Metal Shading Language)
- ``warp_metal.runtime``  — native Metal runtime + module/device hub
- ``warp_metal.compat``   — ``wp`` compatibility layer + version sentinel
"""

import sys as _sys

# Vendored codegen recurses per SSA line and per emitted statement; Newton's
# convex-contact function chains (GJK support maps, deep @wp.func call
# graphs) exceed CPython's default 1000-frame ceiling during Adjoint builds.
# Upstream warp's emitter is shallower per call; mirror the effect of
# upstream's wp.config.recursive_limit by raising the ceiling once here.
_sys.setrecursionlimit(max(_sys.getrecursionlimit(), 10000))

from .compat import (
    EXPECTED_VENDORED_WARP_VERSION,
    VendoredVersionError,
    __version__,
    check_vendored_version,
)
from .shim import install_as_warp  # noqa: F401

# Importing the frontend registers the full builtin table, attaches the
# exported ``wp.*`` callables to this package, and wires the host module
# (see frontend/__init__.py).
from .frontend import builtins as _builtins  # noqa: F401

# Public type surface (scalar/vector/matrix/quat/array types): upstream
# re-exports these per-name from warp/_src/types; a star import is
# equivalent since types.py carries the same public names.
from .frontend.types import *  # noqa: F401,F403

# @wp.struct decorator (M1.1, ported from upstream context.py). Imported
# AFTER the star import on purpose: the vendored types.py leaks the stdlib
# ``struct`` module, which would otherwise shadow the decorator.
from .frontend.structs import struct  # noqa: F401
from .frontend.codegen import Struct  # noqa: F401
from .frontend.registry import Function  # noqa: F401

# Kernel/function decorators + Kernel record (M1.2, ported from upstream context.py).
from .frontend.kernels import (  # noqa: F401
    Kernel,
    kernel,
    func,
    overload,
    func_native,
    func_grad,
    func_replay,
)

# Math constants (upstream warp exports plain floats of the same values).
import math as _math  # noqa: E402

pi = _math.pi  # noqa: F401
PI = _math.pi  # noqa: F401
tau = _math.tau  # noqa: F401
inf = _math.inf  # noqa: F401
nan = _math.nan  # noqa: F401
HALF_PI = _math.pi / 2  # noqa: F401

# Log levels (upstream logger constants).
LOG_DEBUG = 10  # noqa: F401
LOG_INFO = 20  # noqa: F401
LOG_WARNING = 30  # noqa: F401
LOG_ERROR = 40  # noqa: F401

# Host runtime (M1.2): devices, module hub, buffers, launch, capture.
from .runtime.devices import (  # noqa: F401
    Device,
    Stream,
    Event,
    ScopedDevice,
    ScopedStream,
    get_device,
    get_current_device,
    set_device,
    get_preferred_device,
    get_stream,
    set_stream,
    synchronize_device,    synchronize_stream,
    record_event,
    get_event_elapsed_time,
    get_cuda_device_count,
    get_cuda_devices,
    get_devices,
    is_cuda_available,
    is_metal_available,
    is_mempool_enabled,
    get_mempool_used_mem_current,
    get_mempool_used_mem_high,
    can_access_device,
)
from .runtime.modules import (  # noqa: F401
    Module,
    get_module,
    _get_module,
    set_module_options,
    get_module_options,
    clear_kernel_cache,
    clear_lto_cache,
)
from .runtime.host import (  # noqa: F401
    WarpArray,
    zeros,
    empty,
    ones,
    full,
    zeros_like,
    empty_like,
    ones_like,
    full_like,
    clone,
    copy,
    from_numpy,
    launch,
    launch_tiled,
    synchronize,
    flush,
    capture_begin,
    capture_end,
    ScopedCapture,
    ScopedTimer,
    capture_launch,
    capture_if,
    capture_while,
    CaptureStateError,
    Graph,
    init,
    device_name,
)

# DeviceLike: upstream context.py type alias Newton annotates with
# (``from warp import DeviceLike``); Device | str | None.
DeviceLike = Device | str | None  # noqa: F401

# Upstream-internals aliases Newton relies on (``wp._src.context.runtime.core``).
from .frontend import context as _frontend_context  # noqa: F401
from .frontend import config as config  # noqa: F401
from .frontend import utils as utils  # noqa: F401
from .frontend.utils import quat_between_vectors  # noqa: F401

# public wp.array: runtime-backed constructor (vendored class stays as
# frontend.types.array for the internal machinery); must come after both the
# star import (annotations) and the runtime host import (zeros/from_numpy)
from .runtime.host import array as array  # noqa: F401
from .frontend import types as types  # noqa: F401  (upstream: wp.types.vector(...))
from .frontend.config import DeterministicMode  # noqa: F401  (newton module_options)
from .frontend.builtins import static as static  # noqa: F401  (upstream: plain identity fn, wp.static(expr))

_src = _sys.modules["warp_metal.frontend"]  # upstream: warp._src
_src.__name__ = "warp_metal._src"  # display parity with the upstream alias

# Vendored frontend modules occasionally ``import warp`` / ``import warp.x``
# at runtime (deterministic.py static-call resolution, logger verbosity, …).
# Alias this package and every ``warp_metal.*`` submodule under the ``warp``
# name: warp-metal exposes the same symbol surface, and the CUDA-only
# fallback paths those modules guard against are dormant (KEP-0001).
_sys.modules.setdefault("warp", _sys.modules[__name__])
for _name, _mod in list(_sys.modules.items()):
    if _name == "warp_metal" or _name.startswith("warp_metal."):
        _sys.modules.setdefault("warp" + _name[len("warp_metal") :], _mod)
    if _name.startswith("warp_metal.frontend"):
        # upstream layout parity in both spellings: warp/<rest> (top-level,
        # e.g. warp.config ← frontend/config) and warp._src/<rest> (e.g.
        # warp._src.thirdparty.appdirs ← frontend/thirdparty/appdirs).
        _rest = _name[len("warp_metal.frontend") :]
        _sys.modules.setdefault("warp." + _rest.lstrip("."), _mod)
        _sys.modules.setdefault("warp._src" + _rest, _mod)

# Vendored codegen.resolve_path() looks unknown roots up on the ``warp``
# module global (upstream: the installed package object). Bind it here so the
# vendored file stays byte-identical to upstream.
from .frontend import codegen as _frontend_codegen  # noqa: E402

_frontend_codegen.warp = _sys.modules[__name__]

# Host-side transform helper (upstream signature: transform_expand(t)).
from .frontend.types import transform as _transform_type  # noqa: E402


def transform_expand(t) -> _transform_type:
    """Expand a 7-element sequence (px,py,pz,qx,qy,qz,qw) into a transform."""
    out = _transform_type()
    vals = [float(v) for v in t]
    if len(vals) != 7:
        raise ValueError(f"transform_expand expects 7 elements, got {len(vals)}")
    for i, v in enumerate(vals):
        out[i] = v
    return out


# Vendored upstream math helpers (warp 1.19 warp/_src/math.py verbatim).
# Upstream attaches these @wp.func definitions to the wp namespace at import
# time (warp/__init__.py:602 "from warp._src.math import *"); imported last
# so every attribute their bodies reference exists, then re-exported so
# kernels resolve wp.quat_to_euler & co. as user functions — and host code
# calls them (Newton import_usd.py: wp.transform_compose(p, q, s)).
from .frontend.math_funcs import (  # noqa: E402
    norm_huber,
    norm_l1,
    norm_l2,
    norm_pseudo_huber,
    quat_from_euler,
    quat_to_euler,
    quat_to_rpy,
    quat_twist,
    quat_twist_angle,
    quat_twist_angle_signed,
    smooth_normalize,
    transform_compose,
    transform_decompose,
    transform_from_matrix,
    transform_to_matrix,
    transform_twist,
    transform_wrench,
    velocity_at_point,
)

__all__ = [
    "__version__",
    "EXPECTED_VENDORED_WARP_VERSION",
    "VendoredVersionError",
    "check_vendored_version",
    "install_as_warp",
    "quat_between_vectors",
    # decorators + type machinery
    "kernel",
    "func",
    "overload",
    "func_native",
    "func_grad",
    "func_replay",
    "struct",
    "Struct",
    "Kernel",
    "Function",
    "Module",
    # constants
    "pi",
    "PI",
    "tau",
    "inf",
    "nan",
    "HALF_PI",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARNING",
    "LOG_ERROR",
    # devices
    "Device",
    "Stream",
    "Event",
    "ScopedDevice",
    "ScopedStream",
    "get_device",
    "get_current_device",
    "set_device",
    "get_preferred_device",
    "get_stream",
    "set_stream",
    "synchronize_device",
    "synchronize_stream",
    "record_event",
    "get_event_elapsed_time",
    "get_cuda_device_count",
    "get_cuda_devices",
    "get_devices",
    "is_cuda_available",
    "is_metal_available",
    "is_mempool_enabled",
    "get_mempool_used_mem_current",
    "get_mempool_used_mem_high",
    "can_access_device",
    # module hub
    "get_module",
    "set_module_options",
    "get_module_options",
    "clear_kernel_cache",
    "clear_lto_cache",
    # arrays + launch + capture
    "WarpArray",
    "zeros",
    "empty",
    "ones",
    "full",
    "zeros_like",
    "empty_like",
    "ones_like",
    "full_like",
    "clone",
    "copy",
    "from_numpy",
    "launch",
    "launch_tiled",
    "synchronize",
    "flush",
    "capture_begin",
    "capture_end",
    "ScopedCapture",
    "ScopedTimer",
    "capture_launch",
    "capture_if",
    "capture_while",
    "Graph",
    "init",
    "device_name",
    "transform_expand",
    "transform_compose",
    "transform_decompose",
    "transform_from_matrix",
    "transform_to_matrix",
    "config",
    "utils",
]
