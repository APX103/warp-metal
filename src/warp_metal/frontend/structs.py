# Copyright 2026 Jaylin Lee
# Portions Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``@wp.struct`` decorator (KEP-0001 M1.1, ported from warp 1.19 context.py).

Upstream keeps this in ``context.py`` against the full runtime Module hub;
the vendored frontend carries the placeholder :class:`Module`, so the
decorator resolves user classes to a placeholder module keyed by
``__module__``. The module table is replaced by the M1.2 host runtime.

Supported call forms (matching upstream):

    @wp.struct
    class A: ...

    @wp.struct(module="unique")
    class B: ...
"""

from __future__ import annotations

import functools

from . import codegen
from .registry import Module

# Per-source-module placeholder modules so that structs from the same Python
# module land in the same Module (upstream: get_module(c.__module__)).
_MODULES: dict[str, Module] = {}


def _placeholder_module(name: str) -> Module:
    mod = _MODULES.get(name)
    if mod is None:
        mod = Module(name)
        _MODULES[name] = mod
    return mod


def struct(c: type | None = None, *, module: Module | str | None = None):
    """Decorator to define a Warp struct for use in kernels and functions."""

    def wrapper(c, *args, **kwargs):
        if module is None:
            m = _placeholder_module(c.__module__)
        elif module == "unique":
            m = Module(c.__name__)
        elif isinstance(module, str):
            m = _placeholder_module(module)
        else:
            m = module
        s = codegen.Struct(key=codegen.make_full_qualified_name(c), cls=c, module=m)
        s = functools.update_wrapper(s, c)
        return s

    if c is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(c)
