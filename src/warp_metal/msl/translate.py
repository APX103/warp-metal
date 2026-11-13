# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Adjoint-SSA → MSL translator (KEP-0001 M1.3).

Strategy (per ROADMAP decision log 2026-11-13): the vendored warp 1.19
``Adjoint`` walker is the sole language front end — control flow (if/elif
merges, range/while loops), ``@wp.func`` inlining, overload resolution,
constant folding, multi-dim ``tid`` and SSA construction are upstream
battle-tested machinery that warp-metal does not re-implement. This module
consumes the machine-generated C++ SSA text (``codegen_func_forward`` /
``codegen_func`` output, a small regular dialect) and translates it to MSL.

The translation is line-oriented over that dialect:

- types: ``wp::vec_t<3, wp::float32>`` → ``vec3f`` etc. (scalarized
  warp-native layouts from :mod:`warp_metal.msl.prelude`);
- control flow: warp emits loops as ``start_<id>:;`` / ``goto end_<id>;``
  label pairs; these rewrite mechanically to ``while (true) { ... }`` with
  ``break``/``continue`` (identical semantics, MSL-valid);
- memory: warp's ``wp::address``/``wp::load``/``wp::copy`` pointer dance
  collapses into direct array loads (Metal device pointers bound per arg);
  ``&((var).field)`` references resolve to member reads — generated MSL
  holds no address-space-qualified pointer variables;
- builtins: each ``wp::<name>(...)`` maps through a table to prelude
  functions ported 1:1 from warp native headers.

Anything outside the observed dialect fails loud (TranslationError) — the
black-box principle from the KEP: silent miscompiles are worse than a
rejected kernel.
"""

from __future__ import annotations

import re

from .prelude import PRELUDE


class TranslationError(NotImplementedError):
    """A construct outside the supported SSA dialect was encountered."""


# ---------------------------------------------------------------------------
# C++ (warp native dialect) → MSL type mapping
# ---------------------------------------------------------------------------

_SCALAR_CTYPES = {
    "wp::float32": "float",
    "wp::float16": "half",
    "wp::int8": "char",
    "wp::uint8": "uchar",
    "wp::int16": "short",
    "wp::uint16": "ushort",
    "wp::int32": "int",
    "wp::int64": "int64_t",
    "wp::uint32": "uint",
    "wp::uint64": "uint64_t",
    "bool": "bool",
}

_VEC_SIZES = {2: "vec2f", 3: "vec3f", 4: "vec4f", 6: "spatial_vectorf"}
_VECI_SIZES = {2: "vec2i", 3: "vec3i", 4: "vec4i"}
_MAT_SIZES = {(2, 2): "mat22f", (3, 3): "mat33f", (4, 4): "mat44f"}

_RE_VEC_T = re.compile(r"^wp::vec_t<(\d+),\s*wp::(float32|int32)>$")
_RE_MAT_T = re.compile(r"^wp::mat_t<(\d+),\s*(\d+),\s*wp::float32>$")
_RE_QUAT_T = re.compile(r"^wp::quat_t<wp::float32>$")
_RE_TRANSFORM_T = re.compile(r"^wp::transform_t<wp::float32>$")
_RE_ARRAY_T = re.compile(r"^wp::array_t<(.+)>$")


def msl_type(ctype: str) -> str:
    """Map a warp-native C++ type string from the SSA text to an MSL type."""
    ctype = ctype.strip()
    if ctype in _SCALAR_CTYPES:
        return _SCALAR_CTYPES[ctype]
    if m := _RE_VEC_T.match(ctype):
        n = int(m.group(1))
        sizes = _VEC_SIZES if m.group(2) == "float32" else _VECI_SIZES
        if n in sizes:
            return sizes[n]
        raise TranslationError(f"wp::vec_t<{n}> has no MSL mapping")
    if m := _RE_MAT_T.match(ctype):
        key = (int(m.group(1)), int(m.group(2)))
        if key in _MAT_SIZES:
            return _MAT_SIZES[key]
        raise TranslationError(f"wp::mat_t<{key[0]},{key[1]}> has no MSL mapping")
    if _RE_QUAT_T.match(ctype):
        return "quatf"
    if _RE_TRANSFORM_T.match(ctype):
        return "transformf"
    if ctype == "wp::range_t":
        return "wp_range"
    if ctype == "wp::shape_t":
        return "int4"  # loaded shape vectors index via wp_extract(int4, i)
    if ctype == "wp::spatial_vector_t<wp::float32>":
        return "spatial_vectorf"
    if m := _RE_ARRAY_T.match(ctype):
        # array args become (device data ptr, wp_arr_meta) pairs; the type
        # itself never appears as a value in translated MSL
        raise TranslationError(f"wp::array_t used as a value type: {ctype}")
    if ctype.endswith("*"):
        raise TranslationError(f"pointer value {ctype!r} not supported (aliases must collapse)")
    # user struct types (Name_hash) pass through — valid MSL identifiers
    if re.match(r"^[A-Za-z_]\w*$", ctype):
        return ctype
    raise TranslationError(f"no MSL mapping for C++ type {ctype!r}")


def array_elem_msl(ctype: str) -> str:
    """Elem MSL type of a ``wp::array_t<...>`` annotation (device pointer)."""
    m = _RE_ARRAY_T.match(ctype.strip())
    if not m:
        raise TranslationError(f"expected wp::array_t<...>, got {ctype!r}")
    return msl_type(m.group(1))


# ---------------------------------------------------------------------------
# builtin call translation tables
# ---------------------------------------------------------------------------

_CALL_TABLE = {
    # arithmetic (scalar + vector overloads; matmul via wp_mul)
    "add": "wp_add",
    "sub": "wp_sub",
    "mul": "wp_mul",
    "div": "wp_div",
    "mod": "wp_mod",
    "neg": "wp_neg",
    "abs": "wp_abs",
    "sign": "wp_sign",
    "min": "wp_min",
    "max": "wp_max",
    "clamp": "wp_clamp",
    "nonzero": "wp_nonzero",
    # bit ops (builtin.h bit_and/bit_or/bit_xor/lshift/rshift)
    "bit_and": "wp_bit_and",
    "bit_or": "wp_bit_or",
    "bit_xor": "wp_bit_xor",
    "lshift": "wp_lshift",
    "rshift": "wp_rshift",
    # rounding/trig/scalar
    "sqrt": "wp_sqrt",
    "sin": "wp_sin",
    "cos": "wp_cos",
    "tan": "wp_tan",
    "asin": "wp_asin",
    "acos": "wp_acos",
    "isfinite": "wp_isfinite",
    "nan": "wp_nan",
    # vector
    "dot": "wp_dot",
    "cross": "wp_cross",
    "length": "wp_length",
    "length_sq": "wp_length_sq",
    "normalize": "wp_normalize",
    "ddot": "wp_ddot",
    "transpose": "wp_transpose",
    "inverse": "wp_inverse",
    "extract": "wp_extract",
    "where": "wp_where",
    "assign_inplace": "wp_assign_inplace",
    # quat
    "quat_identity": "wp_quat_identity",
    "quat_inverse": "wp_quat_inverse",
    "quat_rotate": "wp_quat_rotate",
    "quat_rotate_inv": "wp_quat_rotate_inv",
    "quat_to_matrix": "wp_quat_to_matrix",
    # transform
    "transform_identity": "wp_transform_identity",
    "transform_get_translation": "wp_transform_get_translation",
    "transform_get_rotation": "wp_transform_get_rotation",
    "transform_point": "wp_transform_point",
    "transform_vector": "wp_transform_vector",
    "transform_multiply": "wp_transform_multiply",
    "transform_inverse": "wp_transform_inverse",
    # spatial
    "spatial_top": "wp_spatial_top",
    "spatial_bottom": "wp_spatial_bottom",
}

# scalar casts: wp::<scalar>(x) → ((T)(x))
_CAST_TABLE = {
    "wp::float": "float",
    "wp::float32": "float",
    "wp::int": "int",
    "wp::int32": "int",
    "wp::int64": "int64_t",
    "wp::uint": "uint",
    "wp::uint32": "uint",
    "wp::uint64": "uint64_t",
}

# constructors: wp::<cpp_type>(args) → make_<msl>(args)
_CTOR_TABLE = {
    "wp::vec_t<2, wp::float32>": "make_vec2f",
    "wp::vec_t<3, wp::float32>": "make_vec3f",
    "wp::vec_t<4, wp::float32>": "make_vec4f",
    "wp::vec_t<6, wp::float32>": "make_spatial_vectorf",
    "wp::vec_t<2, wp::int32>": "make_vec2i",
    "wp::vec_t<3, wp::int32>": "make_vec3i",
    "wp::vec_t<4, wp::int32>": "make_vec4i",
    "wp::quat_t<wp::float32>": "make_quatf",
    "wp::mat_t<2, 2, wp::float32>": "make_mat22f",
    "wp::mat_t<3, 3, wp::float32>": "make_mat33f",
    "wp::mat_t<4, 4, wp::float32>": "make_mat44f",
    "wp::transform_t<wp::float32>": "make_transformf",
}

# zero-arg template constructors: wp::x<wp::float32>() → <msl>
_ZEROARG_TABLE = {
    "wp::quat_identity<wp::float32>": "wp_quat_identity()",
    "wp::transform_identity<wp::float32>": "wp_transform_identity()",
}


def _find_balanced(s: str, start: int) -> int:
    """Index of the paren closing the one at ``s[start]`` (must be '(')."""
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise TranslationError(f"unbalanced parens in {s!r}")


_OPEN = {"(", "[", "<"}
_CLOSE = {")", "]", ">"}


def split_args(src: str) -> list[str]:
    """Split a C++ argument list on depth-0 commas (template angles count)."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in src:
        if ch in _OPEN:
            depth += 1
            cur.append(ch)
        elif ch in _CLOSE:
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


# ---------------------------------------------------------------------------
# parsing helpers for the codegen C++ SSA text
# ---------------------------------------------------------------------------

_RE_DECL = re.compile(r"^\s*(const\s+)?(\S.*?)\s+(var_\w+)\s*(?:=\s*(.+?))?;\s*$")
_RE_LABEL = re.compile(r"^\s*(start|end)_(for|while)_(\d+):\s*;\s*$")
_RE_GOTO = re.compile(r"goto\s+((?:start|end)_(?:for|while)_\d+)\s*;")
_RE_IF_OPEN = re.compile(r"^\s*if\s*\((.*)\)\s*\{\s*$")
_RE_ELSE = re.compile(r"^\s*\}\s*else\s*\{\s*$")
_RE_BRACE_CLOSE = re.compile(r"^\s*\}\s*$")
_RE_RETURN = re.compile(r"^\s*return\s*(.*?);\s*$")


class ArgSpec:
    """One kernel/function argument, classified for MSL emission."""

    def __init__(self, name: str, kind: str, ctype: str):
        self.name = name  # launch-level arg name (from _wp_args->name)
        self.kind = kind  # 'array' | 'scalar' | 'struct'
        self.ctype = ctype  # warp-native C++ type string from the SSA text

    @property
    def msl_elem(self) -> str:
        return array_elem_msl(self.ctype)

    @property
    def msl(self) -> str:
        return msl_type(self.ctype)


class Param:
    """MSL parameter of a device function."""

    def __init__(self, kind: str, msl: str, name: str):
        self.kind = kind  # 'value' | 'array' | 'ref' (thread T&)
        self.msl = msl  # MSL type (elem type for arrays)
        self.name = name


class MSLFunction:
    """A translated device function or kernel entry."""

    def __init__(self, name: str, ret_msl: str | None):
        self.name = name
        self.ret_msl = ret_msl  # None → void
        self.params: list[Param] = []
        self.lines: list[str] = []


class KernelUnit:
    """A complete MSL translation unit for one kernel launch."""

    def __init__(self, kernel_name: str):
        self.structs: dict[str, str] = {}  # native_name -> MSL struct def
        self.struct_fields: dict[str, list[tuple[str, str]]] = {}  # name -> [(field, msl type)]
        self.functions: list[MSLFunction] = []  # callees first
        self.kernel = MSLFunction(kernel_name, None)
        self.args: list[ArgSpec] = []  # kernel args in order
        self.param_fields: list[tuple[str, str]] = []  # (msl type, field name)

    def source(self) -> str:
        parts = ["// generated by warp-metal MSL translator (Adjoint SSA)\n", PRELUDE, "\n"]
        for definition in self.structs.values():
            parts.append(definition + "\n")
        for fn in self.functions:
            parts.append(fn_src(fn))
            parts.append("\n")
        parts.append(self.kernel_src())
        return "".join(parts)

    def kernel_src(self) -> str:
        fn = self.kernel
        params, idx = [], 0
        for a in self.args:
            if a.kind == "array":
                params.append(f"    device {a.msl_elem}* {a.name}_data [[buffer({idx})]]")
                idx += 1
        params.append(f"    constant Params_{self.kernel.name}& _wp_args [[buffer({idx})]]")
        params.append("    uint3 _gid [[thread_position_in_grid]]")
        fields = []
        for a in self.args:
            if a.kind == "scalar":
                fields.append(f"    {a.msl} {a.name};")
            elif a.kind == "struct":
                fields.append(f"    {a.msl} {a.name};")
            elif a.kind == "array":
                fields.append(f"    wp_arr_meta {a.name}_m;")
        fields.append("    wp_arr_meta _dims;  // launch bounds for multi-dim tid")
        body = "\n".join(fn.lines) or "    ;"
        return (
            f"struct Params_{self.kernel.name} {{\n" + "\n".join(fields) + "\n};\n\n"
            + f"kernel void {self.kernel.name}(\n" + ",\n".join(params) + ")\n{\n"
            + "    const int _idx = (int)_gid.x;\n"
            + body
            + "\n}\n"
        )


def fn_src(fn: MSLFunction) -> str:
    params = []
    for p in fn.params:
        if p.kind == "array":
            params.append(f"    device {p.msl}* {p.name}_data")
            params.append(f"    constant wp_arr_meta& {p.name}_m")
        elif p.kind == "ref":
            params.append(f"    thread {p.msl}& {p.name}")
        else:
            params.append(f"    {p.msl} {p.name}")
    body = "\n".join(fn.lines) or "    ;"
    ret = "void" if fn.ret_msl is None else fn.ret_msl
    return f"static {ret} {fn.name}(\n" + ",\n".join(params) + ")\n{\n" + body + "\n}\n"


# ---------------------------------------------------------------------------
# expression + statement translation
# ---------------------------------------------------------------------------

class _BodyTranslator:
    def __init__(self, unit: KernelUnit, args: dict[str, ArgSpec], user_funcs: dict[str, "MSLFunction"]):
        self.unit = unit
        self.array_args = {f"var_{a.name}": a for a in args.values() if a.kind == "array"}
        self.user_funcs = user_funcs  # native name -> translation (param kinds)
        self.aliases: dict[str, tuple[bool, str]] = {}  # var -> (is_pointer, value expr)

    # -- expressions ---------------------------------------------------------

    def expr(self, src: str) -> str:
        src = src.strip()
        m = re.match(r"^builtin_tid([1234])d\s*\((.*)\)$", src)
        if m:
            tid_src = self._builtin_tid(m.group(1), m.group(2))
            return tid_src
        idx = src.find("wp::")
        if idx < 0:
            return src  # plain var / already-valid MSL op expression
        if idx != 0:
            # wp:: inside a larger expression (e.g. casts inside ops) — recurse
            # around it: only single-call or operator forms occur in the SSA
            head, tail = src[:idx], src[idx:]
            return head + self.expr(tail)
        return self._call(src)

    def _builtin_tid(self, n: str, arg_src: str) -> str:
        args = split_args(arg_src) if arg_src.strip() else []
        if n == "1":
            if args:
                raise TranslationError("builtin_tid1d takes no args")
            return "_idx"
        out = ", ".join(a.strip() for a in args)
        return f"wp_tid{n}({out}, _idx, _wp_args._dims)"

    def _call(self, src: str) -> str:
        m = re.match(r"^(wp::[A-Za-z_]\w*(?:<[^()]*>)?)\(", src)
        if not m:
            raise TranslationError(f"cannot parse call expression {src!r}")
        key = m.group(1)
        op = m.end() - 1
        close = _find_balanced(src, op)
        inner = src[op + 1 : close]
        rest = src[close + 1 :].strip()
        if rest:
            raise TranslationError(f"trailing text after call in {src!r}")
        args = split_args(inner)

        if key in _ZEROARG_TABLE:
            if args:
                raise TranslationError(f"{key} expected no args, got {args}")
            return _ZEROARG_TABLE[key]
        if key in _CAST_TABLE:
            return f"(({_CAST_TABLE[key]})({self.expr(args[0])}))"
        if key in _CTOR_TABLE:
            return f"{_CTOR_TABLE[key]}({', '.join(self.expr(a) for a in args)})"
        if key == "wp::copy":
            return self.expr(args[0])
        if key == "wp::load":
            return self._load(args[0])
        if key == "wp::address":
            raise TranslationError("wp::address outside assignment (aliasing context)")
        if key == "wp::range":
            n = len(args)
            if n not in (1, 2, 3):
                raise TranslationError(f"wp::range with {n} args")
            return f"wp_range_{n}({', '.join(self.expr(a) for a in args)})"
        if key in ("wp::iter_cmp", "wp::iter_next"):
            return key.replace("wp::", "wp_") + f"({self.expr(args[0])})"
        if key[4:].startswith("atomic_"):
            # value-returning context (warp atomics return the old value)
            return self._atomic(key[4:], args, ret=True)
        if key in ("wp::print", "wp::printf"):
            raise TranslationError(
                "wp.printf: MSL printf output is host-tool-dependent; not wired in M1.3 (fail loud)"
            )
        if key in ("wp::mesh_get", "wp::bvh_get_group_root", "wp::hash_grid_query", "wp::mesh_query"):
            raise TranslationError(
                f"{key}: BVH/Mesh device builtins are Phase 2 (KEP-0001); not available in M1.3"
            )
        if key[4:] in _CALL_TABLE:
            return f"{_CALL_TABLE[key[4:]]}({', '.join(self.expr(a) for a in args)})"
        raise TranslationError(f"builtin {key!r} has no MSL translation")

    def _load(self, var: str) -> str:
        var = var.strip()
        alias = self.aliases.get(var)
        if alias is None:
            raise TranslationError(f"wp::load of non-alias var {var!r}")
        is_pointer, expr = alias
        return f"(*({expr}))" if is_pointer else expr

    # -- statements ----------------------------------------------------------

    def stmt(self, line: str) -> list[str]:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            return []

        # control flow: label/goto → while(true)/break/continue
        if m := _RE_LABEL.match(line):
            return ["while (true) {"] if m.group(1) == "start" else ["}"]
        if _RE_GOTO.search(line):
            def _repl(mo):
                return "continue;" if mo.group(1).startswith("start_") else "break;"

            out = _RE_GOTO.sub(_repl, stripped)
            # bare iter_cmp/iter_next on the range-loop condition line
            return [_RE_ITER.sub(lambda mo: f"wp_{mo.group(1)}(", out)]
        if m := _RE_IF_OPEN.match(line):
            return [f"if ({self.expr(m.group(1))}) {{"]
        if _RE_ELSE.match(line):
            return ["} else {"]
        if _RE_BRACE_CLOSE.match(line):
            return ["}"]
        if m := _RE_RETURN.match(line):
            return [f"return {self.expr(m.group(1))};".replace("return ;", "return;")]

        # plain assignment `name = expr;` — must be tried before the decl
        # regex, whose lazy `(\S.*?)` can otherwise swallow `ret_0 =` as a
        # "type" on multi-output ref-assign lines (T & ret_0 params)
        if m := re.match(r"^\s*(\w+)\s*=\s*(.+?);\s*$", line):
            target, rhs = m.group(1), m.group(2)
            if rhs.startswith("wp::address(") or rhs.startswith("&("):
                self._alias(target, rhs)
                return []  # alias collapsed: no MSL line, no decl (pointer decl was dropped)
            return [f"{target} = {self.expr(rhs)};"]

        # alias creation: var_p = wp::address(...) / var_p = &(expr)
        if m := _RE_DECL.match(line):
            const, ctype, name, init = m.group(1), m.group(2), m.group(3), m.group(4)
            if init is not None and (init.startswith("wp::address(") or init.startswith("&(")):
                self._alias(name, init)
                return []
            # value declarations pass through with mapped types; pointer decls
            # are dropped (address-of aliases collapse to direct reads)
            if ctype.rstrip().endswith("*"):
                return []
            if not ctype:
                raise TranslationError(f"EMPTY TYPE on line {line!r}")
            try:
                msl_t = msl_type(ctype)
            except TranslationError as e:
                raise TranslationError(f"decl line {line!r}: {e}")
            if init is not None:
                return [f"{'const ' if const else ''}{msl_t} {name} = {self.expr(init)};"]
            return [f"{'const ' if const else ''}{msl_t} {name};"]

        if m := re.match(r"^\s*(.+?);\s*$", line):
            return [self.call_stmt(m.group(1)) + ";"]

        raise TranslationError(f"unparsed statement {line!r}")

    def _alias(self, var: str, init: str) -> None:
        if init.startswith("wp::address("):
            args = split_args(init[len("wp::address(") : -1])
            arr_var = args[0].strip()
            spec = self.array_args.get(arr_var)
            if spec is None:
                raise TranslationError(f"wp::address on non-array arg {arr_var!r}")
            idx = ", ".join(self.expr(a) for a in args[1:])
            expr = f"wp_address({spec.name}_data, _wp_args.{spec.name}_m{', ' + idx if idx else ''})"
            self.aliases[var] = (True, expr)
            return
        # &(expr) reference forms
        inner = init[2:-1].strip()
        shape_of = re.match(r"^\(?(var_\w+)\.shape\)?$", inner)
        if shape_of:
            spec = self.array_args.get(shape_of.group(1))
            if spec is None:
                raise TranslationError(f"&(...shape) on non-array arg {inner!r}")
            s = f"_wp_args.{spec.name}_m.shape"
            expr = f"int4({s}[0], {s}[1], {s}[2], {s}[3])"
            self.aliases[var] = (False, expr)
            return
        # struct member reference &((var).field) — value alias
        member = re.match(r"^\((\w+)\)(\.\w+)+$", inner)
        if not member:
            raise TranslationError(f"unsupported reference form {init!r}")
        self.aliases[var] = (False, self.expr(inner))

    def call_stmt(self, src: str) -> str:
        src = src.strip()
        m = re.match(r"^builtin_tid([1234])d\s*\((.*)\)$", src)
        if m:
            return self._builtin_tid(m.group(1), m.group(2))
        m = re.match(r"^(wp::[A-Za-z_]\w*(?:<[^()]*>)?)\(", src)
        if not m:
            # user function call (multi-output) — pass through with name check
            name = src.split("(", 1)[0]
            if name in self.user_funcs or re.match(r"^\w+$", name):
                return self._call_user(src)
            raise TranslationError(f"cannot parse statement call {src!r}")
        key = m.group(1)
        op = m.end() - 1
        close = _find_balanced(src, op)
        args = split_args(src[op + 1 : close])
        if src[close + 1 :].strip():
            raise TranslationError(f"trailing text after call stmt {src!r}")

        if key == "wp::assign":
            if len(args) != 2:
                raise TranslationError(f"wp::assign with {len(args)} args")
            return f"{args[0].strip()} = {self.expr(args[1])}"
        if key == "wp::array_store":
            arr_var = args[0].strip()
            spec = self.array_args.get(arr_var)
            if spec is None:
                raise TranslationError(f"wp::array_store on non-array arg {arr_var!r}")
            idx = ", ".join(self.expr(a) for a in args[1:-1])
            value = self.expr(args[-1])
            return f"*wp_address({spec.name}_data, _wp_args.{spec.name}_m{', ' + idx if idx else ''}) = {value}"
        if key and key[4:].startswith("atomic_"):
            return self._atomic(key[4:], args)
        # value-producing builtins as statements: emit as expression statement
        return self._call(src)

    def _atomic(self, op: str, args: list[str], ret: bool = False) -> str:
        arr_var = args[0].strip()
        spec = self.array_args.get(arr_var)
        if spec is None:
            raise TranslationError(f"wp::{op} on non-array arg {arr_var!r}")
        idx = ", ".join(self.expr(a) for a in args[1:-1])
        value = self.expr(args[-1])
        suffix = "_ret_at" if ret else "_at"
        return f"wp_{op}{suffix}({spec.name}_data, _wp_args.{spec.name}_m{', ' + idx if idx else ''}, {value})"

    def _call_user(self, src: str) -> str:
        name = src.split("(", 1)[0]
        close = _find_balanced(src, len(name))
        args = split_args(src[len(name) + 1 : close])
        fn = self.user_funcs.get(name)
        if fn is None:
            # struct value constructor (Name_hash(...)) — rewrite to make_<T>
            if name in self.unit.structs:
                return f"make_{name}({', '.join(self.expr(a) for a in args)})"
            return f"{name}({', '.join(self.expr(a) for a in args)})"
        out: list[str] = []
        vi = 0
        for p in fn.params:
            if vi >= len(args):
                raise TranslationError(f"call {name}: missing argument for param {p.name}")
            arg = args[vi]
            vi += 1
            if p.kind == "array":
                spec = self.array_args.get(arg.strip())
                if spec is None:
                    raise TranslationError(f"array param {p.name} of {name} got non-array {arg!r}")
                out.append(f"{spec.name}_data")
                out.append(f"_wp_args.{spec.name}_m")
            else:
                out.append(self.expr(arg))
        if vi != len(args):
            raise TranslationError(f"call {name}: {len(args)} args for {len(fn.params)} params")
        return f"{name}({', '.join(out)})"


_RE_ITER = re.compile(r"\b(iter_cmp|iter_next)\(")
_RE_SEC = re.compile(r"^\s*//\s*(argument vars|primal vars|forward)\s*$")


def _classify_ctype(ctype: str) -> str:
    """'array' | 'struct' | 'scalar' classification of a C++ type string."""
    ctype = ctype.strip().rstrip("*").strip()
    if _RE_ARRAY_T.match(ctype):
        return "array"
    if ctype in _SCALAR_CTYPES or _RE_VEC_T.match(ctype) or _RE_MAT_T.match(ctype):
        return "scalar"
    if _RE_QUAT_T.match(ctype) or _RE_TRANSFORM_T.match(ctype):
        return "scalar"
    if ctype == "wp::spatial_vector_t<wp::float32>":
        return "scalar"
    if ctype in ("wp::range_t", "wp::shape_t"):
        return "scalar"
    if re.match(r"^[A-Za-z_]\w*$", ctype):
        return "struct"
    raise TranslationError(f"cannot classify type {ctype!r}")


def translate_kernel(kernel, options=None) -> KernelUnit:
    """Translate a vendored ``Kernel`` to a complete MSL translation unit.

    The vendored ``Adjoint`` walks the kernel (enable_backward is forced off:
    adjoint codegen is a KEP-0001 Non-Goal), and this module translates the
    resulting SSA C++ text — kernel plus every transitively called
    ``@wp.func`` — into MSL.
    """
    from ..frontend import codegen as _codegen
    from ..frontend.config import DeterministicMode

    opts = dict(options or {})
    opts["enable_backward"] = False
    # The deterministic builder branches on `options.get("deterministic")`;
    # a missing key reads as None != NOT_GUARANTEED and turns ON the det
    # codegen path (WP_DET_SCATTER_OR_FALLBACK macros). Force plain mode.
    opts["deterministic"] = DeterministicMode.NOT_GUARANTEED

    adj = kernel.adj
    adj.build(None, opts)

    # kernel MSL entry name (valid C identifier)
    base = kernel.key.rsplit("::", 1)[-1]
    name = re.sub(r"\W", "_", base)

    unit = KernelUnit(name)

    # ---- classify kernel args from the vendored type objects ----
    from ..frontend.codegen import Var as _Var

    for var in adj.args:
        ctype = _Var.type_to_ctype(var.type)
        kind = _classify_ctype(ctype)
        spec = ArgSpec(var.label, kind, ctype)
        unit.args.append(spec)
        if kind == "struct":
            _emit_struct_def(var.type, unit, _codegen)

    args_by_var = {f"var_{a.name}": a for a in unit.args}

    # ---- transitive user functions, callees first ----
    user_funcs: dict[str, MSLFunction] = {}

    def visit(f):
        fname = f.native_func
        if fname in user_funcs:
            return
        snippet = getattr(f, "native_snippet", None)
        if snippet:
            raise TranslationError(
                f"@wp.func_native {fname!r} has no MSL translation: embedded CUDA/C++ source is "
                f"not supported by the Metal backend (KEP-0001). Rewriting it as a plain @wp.func "
                f"is the supported path; the calling kernel cannot launch on 'metal'."
            )
        for callee in f.adj.called_user_functions:
            visit(callee)
        fn = _translate_function(f, unit, user_funcs, _codegen)
        user_funcs[fname] = fn
        unit.functions.append(fn)

    for uf in adj.called_user_functions:
        visit(uf)

    # ---- kernel body ----
    src = _codegen.codegen_func_forward(adj, func_type="kernel", device="cpu")
    tr = _BodyTranslator(unit, args_by_var, user_funcs)
    body: list[str] = []
    section = None
    for line in src.splitlines():
        if line.lstrip().startswith("#line"):
            continue
        if m := _RE_SEC.match(line):
            section = m.group(1)
            continue
        if section == "argument vars":
            continue  # arg bindings are generated from classified specs
        if section is None:
            continue
        body.extend(tr.stmt(line))

    # arg bindings (scalars/structs read from the Params buffer)
    bindings = [f"{a.msl} var_{a.name} = _wp_args.{a.name};" for a in unit.args if a.kind in ("scalar", "struct")]
    unit.kernel.lines = bindings + body
    return unit


def _translate_function(f, unit: KernelUnit, user_funcs: dict, _codegen) -> MSLFunction:
    """Translate one built ``@wp.func`` (callee-first order guaranteed)."""
    from ..frontend.codegen import Var as _Var

    fname = f.native_func
    src = _codegen.codegen_func(f.adj, fname, device="cpu", forward_only=True)
    # strip #line directives and path/source-echo comments, but KEEP the
    # section markers ("// primal vars" / "// forward") — stmt() only emits
    # lines inside a recognized section, so dropping the markers would drop
    # the entire function body.
    lines = [
        ln
        for ln in src.splitlines()
        if not ln.lstrip().startswith("#line")
        and (not ln.lstrip().startswith("//") or _RE_SEC.match(ln))
    ]

    # header: lines up to the opening brace
    hdr: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        hdr.append(line.strip())
        if line.strip() == "{":
            body_start = i + 1
            break
    header = " ".join(hdr)
    header = header[len("static ") :]
    open_idx = header.index("(")
    close_idx = header.rindex(")")
    ret_and_name = header[:open_idx].strip()
    ret_cpp, fn_name = ret_and_name.rsplit(None, 1)
    assert fn_name == fname, f"function name mismatch: {fn_name} vs {fname}"
    ret_msl = None if ret_cpp == "void" else msl_type(ret_cpp)

    fn = MSLFunction(fn_name, ret_msl)
    args_by_var: dict[str, ArgSpec] = {}

    for arg_src in split_args(header[open_idx + 1 : close_idx]):
        if "&" in arg_src:
            ctype, pname = arg_src.split("&")
            fn.params.append(Param("ref", msl_type(ctype), pname.strip()))
            continue
        ctype, pname = arg_src.rsplit(None, 1)
        if _classify_ctype(ctype) == "array":
            fn.params.append(Param("array", array_elem_msl(ctype), pname))
            args_by_var[f"var_{pname}"] = ArgSpec(pname, "array", ctype)
        else:
            fn.params.append(Param("value", msl_type(ctype), pname))
            if _classify_ctype(ctype) == "struct":
                for var in f.adj.args:
                    if var.label == pname:
                        _emit_struct_def(var.type, unit, _codegen)

    tr = _BodyTranslator(unit, args_by_var, user_funcs)
    section = None
    for line in lines[body_start:]:
        if m := _RE_SEC.match(line):
            section = m.group(1)
            continue
        if line.strip() == "}":
            break
        if section in (None, "argument vars"):
            continue  # args are the MSL signature params already
        fn.lines.extend(tr.stmt(line))
    return fn


def _emit_struct_def(s, unit: KernelUnit, _codegen) -> None:
    """Emit an MSL struct definition (warp-native layout) + constructors."""
    from ..frontend.codegen import Var as _Var

    if s.native_name in unit.structs:
        return
    fields: list[tuple[str, str]] = []
    nested: list = []
    for label, var in s.vars.items():
        ctype = _Var.type_to_ctype(var.type)
        if _RE_ARRAY_T.match(ctype):
            raise TranslationError(
                f"struct {s.native_name} field {label!r}: array fields are not supported by the "
                "M1.3 MSL emitter (fail loud; needed by mesh/narrow-phase paths in M1.4+)"
            )
        if isinstance(var.type, type(s)):
            nested.append(var.type)
            _emit_struct_def(var.type, unit, _codegen)
        fields.append((label, msl_type(ctype)))

    lines = [f"struct {s.native_name} {{"]
    for label, msl_t in fields:
        lines.append(f"    {msl_t} {label};")
    lines.append("};")
    sig = ", ".join(f"{t} {l}" for l, t in fields) or "void"
    lines.append(f"static inline {s.native_name} make_{s.native_name}({sig}) {{")
    if fields:
        lines.append(f"    {s.native_name} v;")
        for label, _ in fields:
            lines.append(f"    v.{label} = {label};")
    else:
        lines.append(f"    {s.native_name} v = {{}};")
    lines.append("    return v;")
    lines.append("}")
    lines.append(f"static inline {s.native_name} make_{s.native_name}() {{")
    lines.append(f"    {s.native_name} v = {{}};")
    lines.append("    return v;")
    lines.append("}")
    unit.structs[s.native_name] = "\n".join(lines)
    # host-side ctypes mirror spec (packed in declaration order, matching MSL)
    unit.struct_fields[s.native_name] = fields
