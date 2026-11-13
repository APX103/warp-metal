# Copyright 2026 Jaylin Lee
# Portions Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``@wp.struct`` decorator (KEP-0001 M1.1, ported from warp 1.19 context.py).

Module resolution goes through the host runtime's module hub (``get_module``
lazy bindings, :mod:`warp_metal.runtime.modules`). Supported call forms
(matching upstream):

    @wp.struct
    class A: ...

    @wp.struct(module="unique")
    class B: ...
"""

from __future__ import annotations

import functools

from . import codegen
from .registry import _LazySymbol

_get_module = _LazySymbol("_get_module")
get_module = _LazySymbol("get_module")
Module = _LazySymbol("Module")


def struct(c: type | None = None, *, module=None):
    """Decorator to define a Warp struct for use in kernels and functions."""

    def wrapper(c, *args, **kwargs):
        if module is None:
            m = _get_module(c.__module__, c.__qualname__)
        elif module == "unique":
            m = Module(c.__name__, None)
        elif isinstance(module, str):
            m = get_module(module)
        else:
            m = module
        s = codegen.Struct(key=codegen.make_full_qualified_name(c), cls=c, module=m)
        s = functools.update_wrapper(s, c)
        return s

    if c is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(c)
