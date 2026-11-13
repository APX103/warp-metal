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

__all__ = [
    "__version__",
    "EXPECTED_VENDORED_WARP_VERSION",
    "VendoredVersionError",
    "check_vendored_version",
]
