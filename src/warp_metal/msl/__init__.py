# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""MSL code generator: Warp AST → Metal Shading Language.

Original work (not vendored). The spike emitter (M0.3) covers the 10
representative spike kernels in ``spike_kernels.py``. MSL baseline: 2.3+
(macOS 26+/Metal 4 API at the runtime layer, fallback Metal 3).

Key mapping seeds (KEP-0001 Design Details):
  ``wp.tid()``                  → ``gid`` (dispatch_threads grid coordinate)
  ``wp.atomic_add(arr, i, v)``  → ``atomic_fetch_add_explicit`` (fp32, Metal 2+)
  ``wp.tile_load``              → threadgroup memory + simdgroup transfers (Phase 2)
"""

from .emitter import MSLEmitter, msl_type_name
from .spike_kernels import SPIKE_KERNELS

__all__ = ["MSLEmitter", "msl_type_name", "SPIKE_KERNELS"]
