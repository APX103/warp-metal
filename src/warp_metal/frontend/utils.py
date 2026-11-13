# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``wp.utils`` namespace stub.

Newton references ``wp.utils.segmented_sort_pairs`` / ``wp.utils.array_scan``
only from the kamino solver (not on the rigid-body MVP path). Any access
fails loudly with a clear pointer instead of silently missing.
"""

from __future__ import annotations


def __getattr__(name):
    raise NotImplementedError(
        f"wp.utils.{name} is not implemented in warp-metal "
        "(host scan/sort helpers arrive with the CPU fallback routing, M1.3+)"
    )
