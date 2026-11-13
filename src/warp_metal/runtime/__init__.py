# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Native Metal runtime + LLVM CPU fallback.

M0.4 prototype: ObjC++ native module (native/), WarpArray over
storageModeShared MTLBuffers with zero-copy numpy views, emit->compile->
PSO-cache->dispatch launch path (host.py). The CPU LLVM fallback lands
with bucket D in M1.2.
"""

from .host import (
    WarpArray,
    zeros,
    empty,
    ones,
    full,
    launch,
    synchronize,
    device_name,
)
from .modules import Module  # noqa: F401  (placeholder hub, M1.2 replaces)

__all__ = [
    "WarpArray",
    "zeros",
    "empty",
    "ones",
    "full",
    "launch",
    "synchronize",
    "device_name",
    "Module",
]
