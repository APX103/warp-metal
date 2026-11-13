# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Stub for upstream warp._src.build (native compile/nvrtc/LTO machinery).

Kept so vendored code paths that reference it import cleanly. Any actual
use raises: warp-metal compiles MSL via its own runtime (M0.4), not the
upstream C++/CUDA toolchain. The libmathdx LTO dispatch paths in vendored
builtins.py are permanently dormant (registry.runtime.core gate).
"""


def _reject(name):
    def fn(*args, **kwargs):
        raise RuntimeError(
            f"warp_metal.frontend.build.{name} is not available: native CUDA "
            f"build machinery is not vendored (KEP-0001)"
        )
    return fn


def __getattr__(name):
    return _reject(name)
