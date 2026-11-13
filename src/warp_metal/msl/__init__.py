# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""MSL code generator: Warp AST → Metal Shading Language.

Placeholder (M0.1 skeleton). Original work (not vendored). M0.3 lands the
minimal emitter covering the 10 representative spike kernels: ``tid``,
scalar/vector ops, array load/store, ``atomic_add``, ``mat33`` multiply and
inverse, ``select``, ``sin``/``cos``, ``dot``/``cross``, ``length``/
``normalize``. MSL baseline: 2.3+ (macOS 26+/Metal 4 API at the runtime layer,
fallback Metal 3).

Key mapping seeds (KEP-0001 Design Details):
  ``wp.tid()``                  → ``gid`` (dispatch_threads grid coordinate)
  ``wp.atomic_add(arr, i, v)``  → ``atomic_fetch_add_explicit`` (fp32, Metal 2+)
  ``wp.tile_load``              → threadgroup memory + simdgroup transfers (Phase 2)
"""
