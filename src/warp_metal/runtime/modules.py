# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Warp module hub (KEP-0001 M1.2, bucket D).

The vendored frontend resolves runtime-land names (``Module``,
``get_module``, ``_get_module`` ...) against the host package through lazy
bindings (``registry._LazySymbol``). This module is that target: one
:class:`Module` per Warp module name tracking kernels, functions, structs,
constants, and per-module compile options.

Unlike upstream warp there is no build artifact per module here yet —
codegen to MSL happens per-kernel at launch time (M1.3), so modules are
bookkeeping only.
"""

from __future__ import annotations

from ..frontend.module_registry import resolve_module_name


class Module:
    """Runtime-side module hub.

    Tracks the kernels / functions / structs / constants declared against
    this module plus its compile options. Keyed by Warp module name (usually
    the defining Python module's name).
    """

    def __init__(self, name: str = "", source_module: str | None = None):
        self.name = name
        self.source_module = source_module
        self.functions: dict = {}
        self.kernels: dict = {}
        self.constants: dict = {}
        self.structs: dict = {}
        self.hashes: set = set()
        self.options: dict = {}
        # Compilation metadata filled in by later milestones (M1.3 codegen).
        self.exec_descriptor = None

    def register_struct(self, struct) -> None:
        """Track a Struct defined against this module (codegen.Struct)."""
        self.structs[struct.key] = struct

    def register_kernel(self, kernel) -> None:
        """Track a Kernel defined against this module (@wp.kernel)."""
        self.kernels[kernel.key] = kernel

    def register_function(self, function, scope_locals=None, skip_adding_overload=False) -> None:
        """Track a Function defined against this module (@wp.func).

        Overload semantics mirror upstream: functions sharing a key stack up
        in insertion order; the most recently added overload is returned by
        lookup so callers see the latest definition.
        """
        existing = self.functions.get(function.key)
        if existing is not None and not skip_adding_overload:
            existing.add_overload(function)
            self.functions[function.key] = function
        else:
            self.functions[function.key] = function


_MODULES: dict[str, Module] = {}


def get_module(name: str) -> Module:
    """Return the Warp module ``name``, creating it on first reference."""
    mod = _MODULES.get(name)
    if mod is None:
        mod = Module(name)
        _MODULES[name] = mod
    return mod


def _get_module(name: str, qualname: str | None = None) -> Module:
    """Resolve the module a construct belongs to (module_registry aware)."""
    return get_module(resolve_module_name(name, qualname))


def set_module_options(options: dict, module=None) -> None:
    """Set compilation options on a module (upstream-compatible signature).

    ``module`` may be a :class:`Module`, a module name string, or ``None``
    (sets the options on the calling Python module, mirroring upstream).
    """
    if module is None:
        import inspect

        module = inspect.currentframe().f_back.f_globals.get("__name__", "")
    if isinstance(module, str):
        module = get_module(module)
    if not isinstance(options, dict):
        raise TypeError(f"module options must be a dict, got {type(options).__name__}")
    module.options.update(options)


def get_module_options(module) -> dict:
    """Return a copy of the module's compile options."""
    if isinstance(module, str):
        module = get_module(module)
    return dict(module.options)


def clear_kernel_cache() -> None:
    """Drop all registered kernels (module registry + compiled artifacts)."""
    for mod in _MODULES.values():
        mod.kernels.clear()
        mod.exec_descriptor = None
    # Compiled PSO cache lives in runtime.host; imported lazily to avoid a cycle.
    from .host import clear_pso_cache

    clear_pso_cache()


def clear_lto_cache() -> None:
    """No-op: LTO distribution is dormant in warp-metal (CUDA path stubbed)."""
