# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Placeholder module hub (KEP-0001 M1.1).

The vendored frontend resolves runtime-land names (``Module``,
``_get_module``, ...) against the host package through lazy bindings
(``registry._LazySymbol``). This placeholder gives those bindings a target
so ``@wp.struct`` can register structs at decoration time.

M1.2 replaces this with the real device/module registry (bucket D).
"""

from __future__ import annotations


class Module:
    """Placeholder for the runtime-side module hub.

    Tracks structs (and later kernels/functions/constants) per Warp module
    so codegen can discover them. Keyed by source-module name.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.functions: dict = {}
        self.kernels: dict = {}
        self.constants: dict = {}
        self.structs: dict = {}
        self.hashes: set = set()

    def register_struct(self, struct) -> None:
        """Track a Struct defined against this module (codegen.Struct)."""
        self.structs[struct.key] = struct
