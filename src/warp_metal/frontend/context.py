# Copyright 2026 Jaylin Lee
# Portions Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Vendored-context namespace (upstream ``warp._src.context`` stand-in).

Newton reaches into warp internals via ``wp._src.context.runtime.core`` to
call native host builtins without Python-scope overload resolution
(``newton/_src/core/host_transform.py``). This module exposes the warp-metal
equivalents: ``runtime`` is the frontend runtime shim whose ``core`` carries
the numpy-implemented host transform builtins.
"""

from .registry import runtime, builtin_functions  # noqa: F401
from .module_registry import register_module_source, register_module_constructs  # noqa: F401

__all__ = ["runtime", "builtin_functions", "register_module_source", "register_module_constructs"]
