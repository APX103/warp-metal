# Copyright 2026 Jaylin Lee
# Portions Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""``@wp.kernel`` / ``@wp.func`` / ``@wp.overload`` decorators (KEP-0001 M1.2).

Ported from upstream warp 1.19 context.py against the vendored frontend's
codegen machinery (``codegen.Adjoint``, ``codegen.make_full_qualified_name``,
``registry.Function``). Backend specifics:

- CUDA-only kernel options (``launch_bounds``, ``cuda_max_registers``,
  ``cluster_dim``, ``enable_cuda_smem_spilling``) are validated and stored
  but have no Metal effect yet.
- ``@wp.func_native`` fails loudly: embedded CUDA source has no Metal
  translation (KEP Non-Goal adjacent; revisit at Phase 3 for mujoco_warp).
- ``@wp.func_grad`` / ``@wp.func_replay`` accept and record (forward-only
  codegen never reads them; adjoint is a KEP Non-Goal).
"""

from __future__ import annotations

import copy
import functools
import inspect

from . import codegen
from . import types as _types

# Lazy host bindings (resolved at call time against the warp_metal package).
from .registry import Function, _LazySymbol

_get_module = _LazySymbol("_get_module")
get_module = _LazySymbol("get_module")
Module = _LazySymbol("Module")


def _shallowcopy(obj):
    """Shallow-copy a Kernel, mirroring upstream ``copy_shallow`` semantics:
    copies instance ``__dict__`` without deep-copying codegen state."""
    clone = copy.copy(obj)
    clone.__dict__ = dict(obj.__dict__)
    return clone


class Kernel:
    """Warp kernel object, typically created by decorating a Python function
    with :func:`@wp.kernel <warp_metal.kernel>`.

    Kernels can be launched on Metal devices using
    :func:`warp_metal.launch`.
    """

    def __init__(self, func, key=None, module=None, options=None, code_transformers=None, source=None):
        self.func = func

        if module is None:
            self.module = _get_module(func.__module__, func.__qualname__)
        else:
            self.module = module

        if key is None:
            self.key = codegen.make_full_qualified_name(func)
        else:
            self.key = key

        self.options = {} if options is None else options

        # Cached so launch() can reject non-Warp entry points without a dict
        # lookup on every launch.
        self.uses_warp_entry_point_abi = self.options.get("entry_point_abi", "warp") == "warp"

        if code_transformers is None:
            code_transformers = []

        self.adj = codegen.Adjoint(func, transformers=code_transformers, source=source)
        self.return_annotation = self.adj.arg_types.pop("return", None)
        if self.return_annotation is type(None):
            self.return_annotation = None

        # check if generic
        self.is_generic = False
        for arg_type in self.adj.arg_types.values():
            if _types.type_is_generic(arg_type):
                self.is_generic = True
                break

        # unique signature (used to differentiate instances of generic kernels during codegen)
        self.sig = ""

        # known overloads for generic kernels, indexed by type signature
        self.overloads = {}

        # generic kernel that was used to instantiate this overload
        self.generic_parent = None

        # argument indices by name
        self.arg_indices = {a.label: i for i, a in enumerate(self.adj.args)}

        # hash and mangled name will be computed when the module is built
        self._hash: bytes | None = None
        self._mangled_name: str | None = None

        # effective grid_stride, resolved when the module is built and read at launch
        self.grid_stride: bool | None = None

        if self.module:
            self.module.register_kernel(self)

    def infer_argument_types(self, args):
        template_types = list(self.adj.arg_types.values())

        if len(args) != len(template_types):
            raise RuntimeError(f"Invalid number of arguments for kernel {self.key}")

        arg_names = list(self.adj.arg_types.keys())

        try:
            return _types.infer_argument_types(args, template_types, arg_names)
        except TypeError as error:
            raise TypeError(f"Error launching kernel '{self.key}': {error}") from error

    def add_overload(self, arg_types):
        if len(arg_types) != len(self.adj.arg_types):
            raise RuntimeError(f"Invalid number of arguments for kernel {self.key}")

        # Never retain live-array descriptors or application subclass
        # identities in the compiler's specialization types.
        arg_names = list(self.adj.arg_types.keys())
        normalized_types = []
        for arg_name, arg_type in zip(arg_names, arg_types, strict=True):
            try:
                normalized_types.append(_types._normalize_array_argument_type(arg_type))
            except TypeError as error:
                raise TypeError(f"Kernel '{self.key}', argument '{arg_name}': {error}") from error
        return self._add_normalized_overload(normalized_types)

    def _add_normalized_overload(self, arg_types):
        """Look up or create an overload from already normalized compiler types."""
        sig = _types.get_signature(arg_types, func_name=self.key)
        ovl = self.overloads.get(sig)
        if ovl is not None:
            return ovl

        arg_names = list(self.adj.arg_types.keys())
        template_types = list(self.adj.arg_types.values())

        # make sure all argument types are concrete and match the kernel parameters
        for i in range(len(arg_types)):
            if not _types.type_matches_template(arg_types[i], template_types[i]):
                if _types.type_is_generic(arg_types[i]):
                    raise TypeError(
                        f"Kernel {self.key} argument '{arg_names[i]}' cannot be generic, got {arg_types[i]}"
                    )
                raise TypeError(
                    f"Kernel {self.key} argument '{arg_names[i]}' type mismatch: "
                    f"expected {template_types[i]}, got {arg_types[i]}"
                )

        overload_annotations = dict(zip(arg_names, arg_types, strict=True))

        # instantiate this kernel with the given argument types
        ovl = _shallowcopy(self)
        ovl.adj = codegen.Adjoint(self.func, overload_annotations, source=self.adj.source)
        ovl.is_generic = False
        ovl.overloads = {}
        ovl.sig = sig
        ovl.generic_parent = self

        self.overloads[sig] = ovl

        return ovl

    def get_overload(self, arg_types):
        sig = _types.get_signature(arg_types, func_name=self.key)
        return self.overloads.get(sig)

    @property
    def hash(self) -> bytes | None:
        return self._hash

    @hash.setter
    def hash(self, value: bytes | None) -> None:
        self._hash = value


def _normalize_cuda_max_registers(value) -> int:
    if isinstance(value, bool):
        raise TypeError(f"cuda_max_registers must be a positive int, got bool {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"cuda_max_registers must be a positive int, got {value!r}")
    if value < 1:
        raise ValueError(f"cuda_max_registers must be >= 1, got {value!r}")
    return value


def _normalize_cluster_dim(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"cluster_dim must be a positive int in [1, 16], got {value!r}")
    if not 1 <= value <= 16:
        raise ValueError(f"cluster_dim must be in [1, 16], got {value!r}")
    return value


def kernel(
    f=None,
    *,
    name: str | None = None,
    enable_backward: bool | None = None,
    launch_bounds=None,
    cuda_max_registers: int | None = None,
    enable_cuda_smem_spilling: bool | None = None,
    cluster_dim: int | None = None,
    module=None,
    module_options: dict | None = None,
    entry_point_abi: str | None = None,
    grid_stride: bool | None = None,
):
    """Decorator to register a Warp kernel from a Python function.

    Semantics mirror upstream warp. CUDA-only options are validated and
    recorded; adjoint generation is a KEP-0001 Non-Goal so ``enable_backward``
    is recorded but never acted on (the Metal backend emits forward kernels
    only).
    """
    kernel_options: dict = {}

    if name is not None:
        if not codegen.is_valid_cpp_identifier(name):
            raise ValueError(f"Kernel name '{name}' is not a valid identifier")
        kernel_options["custom_kernel_name"] = name

    if enable_backward is not None:
        if not isinstance(enable_backward, bool):
            raise TypeError(f"enable_backward must be a bool, got {type(enable_backward).__name__}")
        kernel_options["enable_backward"] = enable_backward

    if launch_bounds is not None:
        kernel_options["launch_bounds"] = launch_bounds

    if entry_point_abi is not None:
        if entry_point_abi not in ("warp", "external_constant_params"):
            raise ValueError(f"entry_point_abi must be 'warp' or 'external_constant_params', got {entry_point_abi!r}")
        if entry_point_abi != "warp":
            kernel_options["entry_point_abi"] = entry_point_abi

    if entry_point_abi == "external_constant_params" and enable_backward is not False:
        raise ValueError("entry_point_abi='external_constant_params' requires enable_backward=False")

    if cuda_max_registers is not None:
        if launch_bounds is not None:
            raise ValueError("cuda_max_registers and launch_bounds cannot be specified together")
        kernel_options["cuda_max_registers"] = _normalize_cuda_max_registers(cuda_max_registers)

    if enable_cuda_smem_spilling is not None:
        if not isinstance(enable_cuda_smem_spilling, bool):
            raise TypeError(f"enable_cuda_smem_spilling must be a bool, got {type(enable_cuda_smem_spilling).__name__}")
        if enable_cuda_smem_spilling:
            kernel_options["enable_cuda_smem_spilling"] = True

    if grid_stride is not None:
        kernel_options["grid_stride"] = bool(grid_stride)

    if cluster_dim is not None:
        kernel_options["cluster_dim"] = _normalize_cluster_dim(cluster_dim)

    def wrapper(f, *args, **kwargs):
        # Resolve the module for this kernel (upstream semantics)
        if module is None:
            m = _get_module(f.__module__, f.__qualname__)
        elif module == "unique":
            m = Module(name or f.__name__, None)
        elif isinstance(module, str):
            m = get_module(module)
        else:
            m = module

        if module_options is not None:
            if not isinstance(module_options, dict):
                raise TypeError(
                    f"@wp.kernel for '{f.__name__}': module_options must be a dict, got {type(module_options).__name__}."
                )
            if module != "unique":
                raise ValueError(
                    f"@wp.kernel for '{f.__name__}': module_options requires module=\"unique\". "
                    "For shared modules use wp.set_module_options() instead."
                )
            m.options.update(module_options)

        k = Kernel(func=f, key=name, module=m, options=kernel_options)
        return functools.update_wrapper(k, f)

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


def func(
    f=None,
    *,
    name: str | None = None,
    module=None,
    inline: bool | None = None,
):
    """Decorator to define a Warp function callable from kernels and other
    Warp functions. Mirrors upstream warp 1.19 semantics."""

    if inline is not None and not isinstance(inline, bool):
        raise TypeError(f"inline must be a bool or None, got {type(inline).__name__}: {inline!r}")

    inline_hint = None if inline is None else ("forceinline" if inline else "noinline")

    frame = inspect.currentframe()
    try:
        if frame is None or frame.f_back is None:
            scope_locals = {}
        else:
            scope_locals = frame.f_back.f_locals
    finally:
        # Avoid a reference cycle that would retain the caller stack.
        del frame

    def wrapper(f, *args, **kwargs):
        if name is None:
            key = codegen.make_full_qualified_name(f)
        else:
            key = name

        if module is None:
            m = _get_module(f.__module__, f.__qualname__)
        elif module == "unique":
            m = Module(f.__name__, None)
        elif isinstance(module, str):
            m = get_module(module)
        else:
            m = module

        doc = getattr(f, "__doc__", "") or ""
        Function(
            func=f,
            key=key,
            namespace="",
            module=m,
            value_func=None,  # inferred during Adjoint build
            scope_locals=scope_locals,
            doc=doc.strip(),
            inline_hint=inline_hint,
        )

        # use the top of the list of overloads for this key
        g = m.functions[key]
        # copy over the function attributes, including docstring
        return functools.update_wrapper(g, f)

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


def overload(kernel, arg_types: dict | list | None = None):
    """Overload a generic kernel with the given argument types.

    Can be called directly or used as a function decorator.
    """
    if isinstance(kernel, Kernel):
        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded. Kernel {kernel.key} is not generic")

        if isinstance(arg_types, list):
            arg_list = arg_types
        elif isinstance(arg_types, dict):
            # substitute named args
            arg_list = [a.type for a in kernel.adj.args]
            for arg_name, arg_type in arg_types.items():
                idx = kernel.arg_indices.get(arg_name)
                if idx is None:
                    raise RuntimeError(f"Invalid argument name '{arg_name}' in overload of kernel {kernel.key}")
                arg_list[idx] = arg_type
        elif arg_types is None:
            arg_list = []
        else:
            raise TypeError("Kernel overload types must be given in a list or dict")

        # return new kernel overload
        return kernel.add_overload(arg_list)

    elif callable(kernel):
        # handle cases where user calls us as a function decorator (@wp.overload)

        module_name = kernel.__module__
        m = _get_module(module_name, kernel.__qualname__)

        parent = _find_generic_kernel(kernel, m)
        if parent is None:
            raise RuntimeError(
                f"Overload for function '{kernel.__qualname__}' does not match any generic kernel "
                f"in module '{module_name}'"
            )
        adj = codegen.Adjoint(kernel, is_user_function=True)
        arg_list = [adj.arg_types[name] for name in parent.adj.arg_types.keys()]
        return parent.add_overload(arg_list)

    raise TypeError(f"Invalid kernel argument for overload: {kernel}")


def _find_generic_kernel(fn, m):
    """Find the generic kernel in ``m`` that ``fn`` overloads (by name)."""
    for k in m.kernels.values():
        if k.func.__name__ == fn.__name__ and k.is_generic:
            return k
    return None


def func_native(snippet: str, adj_snippet: str | None = None, replay_snippet: str | None = None):
    """Register a Warp function implemented by a native C++/CUDA snippet.

    The registration itself succeeds (upstream parity: module imports and
    non-GJK paths keep working — Newton imports support_function.py at the
    geometry package level), but the snippet is CUDA/C++ source with no MSL
    translation. Any kernel that CALLS the function fails loudly at
    translate time with a clear message (KEP-0001: fail loud at the
    capability boundary, not at unrelated import time).
    """
    frame = inspect.currentframe()
    try:
        if frame is None or frame.f_back is None:
            scope_locals = {}
        else:
            scope_locals = frame.f_back.f_locals
    finally:
        # Avoid a reference cycle that would retain the caller stack.
        del frame

    def wrapper(f, *args, **kwargs):
        key = codegen.make_full_qualified_name(f)
        m = _get_module(f.__module__, f.__qualname__)

        Function(
            func=f,
            key=key,
            namespace="",
            module=m,
            native_snippet=snippet,
            adj_native_snippet=adj_snippet,
            replay_snippet=replay_snippet,
            scope_locals=scope_locals,
        )  # value_type not known yet, will be inferred during Adjoint.build()

        g = m.functions[key]
        # copy over the function attributes, including docstring
        return functools.update_wrapper(g, f)

    return wrapper


def func_grad(forward_fn):
    """Register the decorated function as the gradient of ``forward_fn``.

    Mirrors upstream warp 1.19 ``wp.func_grad``: the grad function is wrapped
    as a ``Function`` (same key as the forward, ``custom_reverse_mode=True``)
    so callers that inspect ``custom_grad_func.adj`` — deferred parse in
    codegen, deterministic reverse-call args — see the same object shape as
    upstream. The Metal backend is forward-only: the custom grad is parsed but
    its reverse codegen is never emitted.
    """
    def wrapper(grad_fn):
        m = _get_module(forward_fn.__module__, forward_fn.__qualname__)
        target = m.functions.get(codegen.make_full_qualified_name(forward_fn))
        if target is None:
            return grad_fn  # forward not (yet) registered: record nothing
        target.custom_grad_func = Function(
            grad_fn,
            key=target.key,
            native_func=target.native_func,
            namespace=target.namespace,
            module=target.module,
            skip_forward_codegen=True,
            custom_reverse_mode=True,
            custom_reverse_num_input_args=len(target.input_types),
            skip_adding_overload=False,
            code_transformers=list(getattr(target.adj, "transformers", None) or []),
        )
        target.adj.skip_reverse_codegen = True
        return grad_fn

    return wrapper


def func_replay(forward_fn):
    """Register the decorated function as the replay variant of ``forward_fn``.

    Mirrors upstream: the replay function is wrapped as a ``Function``
    (``replay_<key>``) so caller-side inspection of ``custom_replay_func.adj``
    — deferred parse — sees the upstream object shape. The Metal backend is
    forward-only: the replay body is parsed but never emitted.
    """
    def wrapper(replay_fn):
        m = _get_module(forward_fn.__module__, forward_fn.__qualname__)
        target = m.functions.get(codegen.make_full_qualified_name(forward_fn))
        if target is None:
            return replay_fn  # forward not (yet) registered: record nothing
        target.custom_replay_func = Function(
            replay_fn,
            key=f"replay_{target.key}",
            namespace=target.namespace,
            input_types=target.input_types,
            value_func=target.value_func,
            export_func=target.export_func,
            dispatch_func=target.dispatch_func,
            module=target.module,
            skip_reverse_codegen=True,
            skip_adding_overload=True,
            code_transformers=list(getattr(target.adj, "transformers", None) or []),
            inline_hint=target.inline_hint,
        )
        return replay_fn

    return wrapper
