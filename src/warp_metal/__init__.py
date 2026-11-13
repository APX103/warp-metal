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
- ``warp_metal.runtime``  — native Metal runtime + LLVM CPU fallback
- ``warp_metal.compat``   — ``wp`` compatibility layer + version sentinel
"""

from .compat import (
    EXPECTED_VENDORED_WARP_VERSION,
    VendoredVersionError,
    __version__,
    check_vendored_version,
)

# Importing the frontend registers the full builtin table, attaches the
# exported ``wp.*`` callables to this package, and wires the host module
# (see frontend/__init__.py).
from .frontend import builtins as _builtins  # noqa: F401

# Public type surface (scalar/vector/matrix/quat/array types): upstream
# re-exports these per-name from warp/_src/types; a star import is
# equivalent since types.py carries the same public names.
from .frontend.types import *  # noqa: F401,F403

# Host runtime (M0.4 prototype): buffers, launch, synchronize. These also
# serve as the registry's lazy runtime_symbol targets for vendored code.
from .runtime import (  # noqa: F401
    WarpArray,
    zeros,
    empty,
    ones,
    full,
    launch,
    synchronize,
    device_name,
)

__all__ = [
    "__version__",
    "EXPECTED_VENDORED_WARP_VERSION",
    "VendoredVersionError",
    "check_vendored_version",
    "WarpArray",
    "zeros",
    "empty",
    "ones",
    "full",
    "launch",
    "synchronize",
    "device_name",
]
