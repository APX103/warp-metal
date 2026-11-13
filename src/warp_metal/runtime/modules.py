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
        """Track a Function defined against this module.

        Overload semantics mirror upstream ``context.py`` exactly:

        - functions from *different scopes* that share a key only replace the
          module's top-level pointer — each keeps **private** overload
          tables. This is what lets factory-composed generic functions
          (Newton's ``shape_support`` / ``support_map`` chains) coexist: a
          captured outer variant resolving a call to an inner variant must
          find the inner variant's own template, never the outer one that is
          currently building, or the build recurses forever.
        - same-scope redefinition with an identical signature replaces the
          top-level function and inherits the old tables (Python shadowing).
        - different signatures become genuine overloads on the existing
          function's tables.
        """
        func_existing = None
        if scope_locals is not None:
            raw = getattr(function, "func", None)
            obj = scope_locals.get(getattr(raw, "__name__", None))
            if obj is not None and type(obj).__name__ == "Function":
                func_existing = obj
        self.functions[function.key] = func_existing or function
        if func_existing is None:
            return

        from ..frontend.types import get_signature, is_generic_signature

        sig = get_signature(
            list(function.input_types.values()),
            func_name=function.key,
            arg_names=list(function.input_types.keys()),
        )
        sig_existing = get_signature(
            list(func_existing.input_types.values()),
            func_name=func_existing.key,
            arg_names=list(func_existing.input_types.keys()),
        )
        if sig == sig_existing:
            # same-scope redefinition with identical signature: the new
            # function replaces the top-level one and inherits the old
            # tables ("replace the top-level function, but keep existing
            # overloads" upstream)
            function.user_templates = dict(func_existing.user_templates)
            if is_generic_signature(sig):
                for k, v in func_existing.user_overloads.items():
                    if getattr(v, "generic_parent", None) != func_existing:
                        function.user_overloads[k] = v
                function.user_templates[sig] = function
            else:
                function.user_overloads = dict(func_existing.user_overloads)
                function.user_overloads[sig] = function
            self.functions[function.key] = function
        elif not skip_adding_overload:
            # a generic overload replacing an existing generic purges the
            # concrete specializations instantiated from the old one
            if is_generic_signature(sig):
                old_generic = func_existing.user_templates.get(sig)
                if old_generic is not None:
                    for k, v in list(func_existing.user_overloads.items()):
                        if getattr(v, "generic_parent", None) == old_generic:
                            del func_existing.user_overloads[k]
            func_existing.add_overload(function)


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


def _resolve_module_like(module):
    """Accept a Module hub, a module name string, or a Python module object."""
    if module is None or isinstance(module, Module) or isinstance(module, str):
        return module
    name = getattr(module, "__name__", None)
    if isinstance(name, str):
        return get_module(name)
    return module


def set_module_options(options: dict, module=None) -> None:
    """Set compilation options on a module (upstream-compatible signature).

    ``module`` may be a :class:`Module`, a module name string, a Python
    module object, or ``None`` (sets the options on the calling Python
    module, mirroring upstream).
    """
    module = _resolve_module_like(module)
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
    module = _resolve_module_like(module)
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
