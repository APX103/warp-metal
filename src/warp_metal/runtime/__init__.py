# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Native Metal runtime + LLVM CPU fallback.

Placeholder (M0.1 skeleton). Original work (not vendored). M0.4 lands the
ObjC++ prototype:

- ``MTLDevice`` / ``MTLBuffer`` management with ``storageModeShared``
  (unified memory; CPU-side numpy zero-copy views over buffer contents)
- PSO cache; runtime MSL → metallib compilation
- single kernel dispatch + synchronize
- (later phases) command buffer batching: per-step deferred commit

The CPU backend reuses Warp's LLVM execution path, built without CUDA
(KEP-0001 Goals: debugging backend and per-kernel fallback routing target).
Build style follows warp native: pybind-style bindings, compiled by the
package build; no CUDA toolkit required anywhere.
"""
