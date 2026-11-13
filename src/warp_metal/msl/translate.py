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
_RE_GEN_MAT = re.compile(r"^mat(\d+)x(\d+)f$")
_RE_QUAT_T = re.compile(r"^wp::quat_t<wp::float32>$")
_RE_TRANSFORM_T = re.compile(r"^wp::transform_t<wp::float32>$")
_RE_ARRAY_T = re.compile(r"^wp::array_t<(.+)>$")


def ensure_mat_def(unit: KernelUnit, msl_t: str) -> None:
    """Emit the struct def + row-extract for a generic ``mat<M>x<N>f``.

    The fixed 2x2/3x3/4x4 mats live in the prelude; sizes beyond those
    (Newton's FACE_NORMALS icosahedron table is 20x3) are generated per
    translation unit on first use. Rows extract to vec2/3/4 — anything else
    fails loud.
    """
    m = _RE_GEN_MAT.match(msl_t)
    if m is None or msl_t in unit.structs:
        return
    M, N = int(m.group(1)), int(m.group(2))
    vec = _VEC_SIZES.get(N)
    if vec is None:
        raise TranslationError(f"generic mat {msl_t}: row extract has no vec{N} mapping")
    lines = [f"struct {msl_t} {{ float c[{M}][{N}]; }};"]
    ctor_args = ", ".join(
        f"float a{r}{c}" for r in range(M) for c in range(N)
    )
    ctor_body = "\n".join(
        f"    v.c[{r}][{c}] = a{r}{c};" for r in range(M) for c in range(N)
    )
    lines.append(f"static inline {msl_t} make_{msl_t}({ctor_args}) {{")
    lines.append(f"    {msl_t} v;")
    lines.append(ctor_body)
    lines.append("    return v;")
    lines.append("}")
    lines.append(f"static inline {msl_t} make_{msl_t}() {{")
    lines.append(f"    {msl_t} v = {{}};")
    lines.append("    return v;")
    lines.append("}")
    comps = ", ".join(f"m.c[i][{j}]" for j in range(N))
    lines.append(
        f"static inline {vec} wp_extract({msl_t} m, int i) {{ return make_{vec}({comps}); }}"
    )
    # element form (warp extract(m, i, j) -> scalar)
    lines.append(
        f"static inline float wp_extract({msl_t} m, int i, int j) {{ return m.c[i][j]; }}"
    )
    # row assign (warp assign_inplace(m, i, vec))
    assign_body = "; ".join(f"m.c[i][{j}] = v.{comp}" for j, comp in enumerate(
        ["x", "y", "z", "w"][:N]))
    lines.append(
        f"static inline void wp_assign_inplace(thread {msl_t}& m, int i, {vec} v) {{ {assign_body}; }}"
    )
    # scalar element assign (warp assign_inplace(m, i, j, scalar))
    lines.append(
        f"static inline void wp_assign_inplace(thread {msl_t}& m, int i, int j, float v) {{ m.c[i][j] = v; }}"
    )
    lines.append(
        f"static inline {msl_t} wp_where(bool c, {msl_t} a, {msl_t} b) {{ return c ? a : b; }}"
    )
    unit.structs[msl_t] = "\n".join(lines)


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
        return f"mat{key[0]}x{key[1]}f"  # generic size; def emitted per unit
    if _RE_QUAT_T.match(ctype):
        return "quatf"
    if _RE_TRANSFORM_T.match(ctype):
        return "transformf"
    if ctype == "wp::range_t":
        return "wp_range"
    if ctype == "wp::shape_t":
        return "int4"  # loaded shape vectors index via wp_extract(int4, i)
    if ctype == "wp::Mesh":
        return "wp_mesh"  # Phase-1 stub (see prelude) — never executed
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
    "floordiv": "wp_floordiv",
    "cw_mul": "wp_cw_mul",
    "cw_div": "wp_cw_div",
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
    "invert": "wp_invert",
    "unot": "wp_unot",
    # rounding/trig/scalar
    "sqrt": "wp_sqrt",
    "sin": "wp_sin",
    "cos": "wp_cos",
    "tan": "wp_tan",
    "asin": "wp_asin",
    "acos": "wp_acos",
    "atan": "wp_atan",
    "atan2": "wp_atan2",
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
    "matrix_from_cols": "wp_matrix_from_cols",
    "matrix_from_rows": "wp_matrix_from_rows",
    "extract": "wp_extract",
    "where": "wp_where",
    "assign_inplace": "wp_assign_inplace",
    # quat
    "quat_identity": "wp_quat_identity",
    "quat_inverse": "wp_quat_inverse",
    "quat_rotate": "wp_quat_rotate",
    "quat_rotate_inv": "wp_quat_rotate_inv",
    "quat_to_matrix": "wp_quat_to_matrix",
    "quat_from_matrix": "wp_quat_from_matrix",
    "quat_from_axis_angle": "wp_quat_from_axis_angle",
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

    def __init__(self, name: str, kind: str, ctype: str, array_fields: list[tuple[str, str]] | None = None):
        self.name = name  # launch-level arg name (from _wp_args->name)
        self.kind = kind  # 'array' | 'scalar' | 'struct'
        self.ctype = ctype  # warp-native C++ type string from the SSA text
        # for kind == 'struct': [(field, elem ctype)] array fields split out
        # into their own buffer bindings (not part of the MSL struct value)
        self.array_fields = array_fields or []

    @property
    def msl_elem(self) -> str:
        return array_elem_msl(self.ctype)

    @property
    def msl(self) -> str:
        return msl_type(self.ctype)


class Param:
    """MSL parameter of a device function."""

    def __init__(self, kind: str, msl: str, name: str, array_fields: list[tuple[str, str]] | None = None):
        self.kind = kind  # 'value' | 'array' | 'ref' (thread T&) | 'struct'
        self.msl = msl  # MSL type (elem type for arrays)
        self.name = name
        # for kind == 'struct': [(field, elem ctype)] split array fields
        # expanded into extra data+meta params right after the value param
        self.array_fields = array_fields or []


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
        # struct name -> [(field, elem ctype)] for array fields SPLIT OUT of
        # the MSL struct: each becomes its own buffer binding + Params meta
        self.struct_array_fields: dict[str, list[tuple[str, str]]] = {}
        self.functions: list[MSLFunction] = []  # callees first
        self.kernel = MSLFunction(kernel_name, None)
        self.args: list[ArgSpec] = []  # kernel args in order
        self.param_fields: list[tuple[str, str]] = []  # (msl type, field name)
        # set by kernel_src: True when device pointers bind through a tier-2
        # argument buffer (Metal plain-buffer cap is 30)
        self.uses_argbuf = False
        self.device_bindings: list[tuple[str, str]] = []  # [(binding, elem msl)]

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
        # Metal caps plain buffer bindings at index 30. Kernels with many
        # arrays (Newton's primitive narrow phase: 34 device pointers) bind
        # them through a tier-2 argument buffer instead: one constant struct
        # of `device T*` members (GPU addresses written host-side), plus
        # textual defines so every `{binding}_data` reference in the emitted
        # body keeps working unchanged.
        self.device_bindings = list(self._iter_device_bindings())
        use_argbuf = len(self.device_bindings) > 30
        self.uses_argbuf = use_argbuf

        params, idx = [], 0
        defines: list[str] = []
        for a in self.args:
            if a.kind == "array":
                binding = f"{a.name}_data"
                if use_argbuf:
                    defines.append(f"#define {binding} _abuf.{binding}")
                else:
                    params.append(f"    device {a.msl_elem}* {binding} [[buffer({idx})]]")
                    idx += 1
            elif a.kind == "struct":
                for field, elem_msl in self._struct_field_elems(a):
                    binding = f"{a.name}__{field}_data"
                    if use_argbuf:
                        defines.append(f"#define {binding} _abuf.{binding}")
                    else:
                        params.append(f"    device {elem_msl}* {binding} [[buffer({idx})]]")
                        idx += 1
        if use_argbuf:
            params.insert(0, f"    constant Args_{self.kernel.name}& _abuf [[buffer({idx})]]")
            idx += 1
        params.append(f"    constant Params_{self.kernel.name}& _wp_args [[buffer({idx})]]")
        params.append("    uint3 _gid [[thread_position_in_grid]]")
        fields = []
        for a in self.args:
            if a.kind == "scalar":
                fields.append(f"    {a.msl} {a.name};")
            elif a.kind == "struct":
                fields.append(f"    {a.msl} {a.name};")
                for field, _ in self._struct_field_specs(a):
                    fields.append(f"    wp_arr_meta {a.name}__{field}_m;")
            elif a.kind == "array":
                fields.append(f"    wp_arr_meta {a.name}_m;")
        fields.append("    wp_arr_meta _dims;  // launch bounds for multi-dim tid")
        body = "\n".join(fn.lines) or "    ;"
        defines_text = "\n".join(defines) + "\n" if defines else ""
        return (
            self._argbuf_struct(use_argbuf) + "\n\n"
            + f"struct Params_{self.kernel.name} {{\n" + "\n".join(fields) + "\n};\n\n"
            + defines_text
            + f"kernel void {self.kernel.name}(\n" + ",\n".join(params) + ")\n{\n"
            + "    const int _idx = (int)_gid.x;\n"
            + body
            + "\n}\n"
        )

    def _iter_device_bindings(self):
        """(binding, elem msl) for every device-pointer binding, in order."""
        for a in self.args:
            if a.kind == "array":
                yield f"{a.name}_data", a.msl_elem
            elif a.kind == "struct":
                for field, elem_msl in self._struct_field_elems(a):
                    yield f"{a.name}__{field}_data", elem_msl

    def _argbuf_struct(self, use_argbuf: bool) -> str:
        if not use_argbuf:
            return ""
        lines = [f"struct Args_{self.kernel.name} {{"]
        for binding, elem in self._iter_device_bindings():
            lines.append(f"    device {elem}* {binding};")
        lines.append("};")
        return "\n".join(lines)

    def _struct_field_specs(self, a: ArgSpec) -> list[tuple[str, str]]:
        """[(field, elem ctype)] of ``a``'s split array fields."""
        return self.struct_array_fields.get(a.ctype, [])

    def _struct_field_elems(self, a: ArgSpec) -> list[tuple[str, str]]:
        """[(field, elem MSL type)] of ``a``'s split array fields."""
        return [(field, msl_type(elem_ctype)) for field, elem_ctype in self._struct_field_specs(a)]


def fn_src(fn: MSLFunction) -> str:
    params = []
    for p in fn.params:
        if p.kind == "array":
            params.append(f"    device {p.msl}* {p.name}_data")
            params.append(f"    constant wp_arr_meta& {p.name}_m")
        elif p.kind == "ref":
            params.append(f"    thread {p.msl}& {p.name}")
        elif p.kind == "struct" and p.array_fields:
            # split array fields expand into extra data+meta params here, at
            # emission — fn.params mirrors the C++ SSA parameter list so the
            # call-site argument zip stays aligned
            params.append(f"    {p.msl} {p.name}")
            for field, elem_ctype in p.array_fields:
                params.append(f"    device {msl_type(elem_ctype)}* {p.name}__{field}_data")
                params.append(f"    constant wp_arr_meta& {p.name}__{field}_m")
        else:
            params.append(f"    {p.msl} {p.name}")
    body = "\n".join(fn.lines) or "    ;"
    ret = "void" if fn.ret_msl is None else fn.ret_msl
    return f"static {ret} {fn.name}(\n" + ",\n".join(params) + ")\n{\n" + body + "\n}\n"


# ---------------------------------------------------------------------------
# expression + statement translation
# ---------------------------------------------------------------------------

class _BodyTranslator:
    def __init__(self, unit: KernelUnit, array_args: dict[str, ArgSpec], user_funcs: dict[str, "MSLFunction"],
                 scope: str = "kernel"):
        self.unit = unit
        self.scope = scope  # "kernel" (Params buffer) | "func" (meta params)
        # array bindings visible in this scope, keyed by every SSA form that
        # can reference them: ``var_<arg>`` for top-level arrays,
        # ``var_<struct>.<field>`` for split struct array fields, and (added
        # by _alias) the ``var_N = &((var_x).field)`` alias vars themselves
        self.array_args = array_args
        self.user_funcs = user_funcs  # native name -> translation (param kinds)
        self.aliases: dict[str, tuple[bool, str]] = {}  # var -> (is_pointer, value expr)
        # `wp::Any var_N;` decls (wp.static temps) — typed at first assignment
        self.deferred_any: set[str] = set()

    def _meta_ref(self, spec: ArgSpec) -> str:
        """wp_arr_meta reference for an array arg in the current scope."""
        if self.scope == "kernel":
            return f"_wp_args.{spec.name}_m"
        return f"{spec.name}_m"  # fn_src emits `constant wp_arr_meta& {name}_m`

    # -- expressions ---------------------------------------------------------

    def expr(self, src: str) -> str:
        src = src.strip()
        m = re.match(r"^builtin_tid([1234])d\s*\((.*)\)$", src)
        if m:
            tid_src = self._builtin_tid(m.group(1), m.group(2))
            return tid_src
        if src in self.array_args:
            # array used as a value (phi merges of `if arr:` blocks, bool
            # contexts): upstream tests array_t's non-null data pointer, and
            # warp-metal always binds a real buffer → statically true
            return "true"
        # value aliases (`&(x).field`, array-truthiness temps) resolve inline
        alias = self.aliases.get(src)
        if alias is not None:
            if alias[0]:
                raise TranslationError(f"pointer alias {src!r} used outside wp::load")
            return alias[1]
        # short-circuit bool ops: SSA keeps C++ `&&`/`||` on bool temps
        for op in ("&&", "||"):
            parts = _split_bool_op(src, op)
            if parts is not None:
                return f"({self.expr(parts[0])} {op} {self.expr(parts[1])})"
        idx = src.find("wp::")
        if idx != 0:
            # user function call in expression context (RHS of assignment):
            # needs param-aware expansion (array params → data + meta args)
            m = re.match(r"^([A-Za-z_]\w*)\(", src)
            if m and (
                m.group(1) in self.user_funcs
                or m.group(1) in self.unit.structs
                or m.group(1) in ("printf", "print")
            ):
                close = _find_balanced(src, len(m.group(1)) - 1)
                if not src[close + 1 :].strip():
                    return self._call_user(src[: close + 1])
            if idx < 0:
                return src  # plain var / already-valid MSL op expression
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

        # aggregate init forms arrive as one braced wrapper (`wp::spatial_vector_t<f32>({a,...,f})`)
        if inner.startswith("{") and inner.endswith("}"):
            args = split_args(inner[1:-1])

        if key in _ZEROARG_TABLE:
            if args:
                raise TranslationError(f"{key} expected no args, got {args}")
            return _ZEROARG_TABLE[key]
        if key in _CAST_TABLE:
            return f"(({_CAST_TABLE[key]})({self.expr(args[0])}))"
        if key in _CTOR_TABLE:
            return f"{_CTOR_TABLE[key]}({', '.join(self.expr(a) for a in args)})"
        m_ctor = re.match(r"^wp::mat_t<(\d+), (\d+), wp::float32>$", key)
        if m_ctor:  # generic-size matrix constructor (e.g. FACE_NORMALS tables)
            msl_t = f"mat{m_ctor.group(1)}x{m_ctor.group(2)}f"
            ensure_mat_def(self.unit, msl_t)
            return f"make_{msl_t}({', '.join(self.expr(a) for a in args)})"
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
            # wp.print is debug stdout only — it cannot affect output arrays,
            # so elide it in MSL (Metal device printf is host-tool-dependent).
            # The generated line keeps a marker so the elision stays visible.
            return "/* wp.print elided (Metal) */ 0"
        if key == "wp::mesh_get":
            # Phase-1 stub (see prelude): returns a zeroed wp_mesh. Provable
            # no-op — mesh registration doesn't exist in Phase 1, so every
            # caller is guarded by mesh_id != 0.
            return "wp_mesh_get(" + self.expr(args[0]) + ")"
        if key in ("wp::bvh_get_group_root", "wp::hash_grid_query", "wp::mesh_query"):
            raise TranslationError(
                f"{key}: BVH/Mesh device builtins are Phase 2 (KEP-0001); not available in M1.4"
            )
        if key == "wp::static":
            # wp.static(x): compile-time constant expression — emit the arg
            # (the `wp::Any` decl it feeds is deferred to the first assignment)
            if len(args) != 1:
                raise TranslationError(f"wp::static with {len(args)} args")
            return self.expr(args[0])
        # generic calls carry a template suffix (`wp::matrix_from_cols<wp::float32>`);
        # strip it for the plain-call table (cast/ctor tables keep the full key)
        base = key[4:].split("<", 1)[0]
        if base in _CALL_TABLE:
            return f"{_CALL_TABLE[base]}({', '.join(self.expr(a) for a in args)})"
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
            cond = m.group(1).strip()
            # array truthiness (`if arr:` — upstream emits `if (array_var)`
            # via array_t's non-null data check; warp-metal always binds a
            # real device buffer, so the check is statically true)
            if cond in self.array_args:
                return ["if (true) {"]
            return [f"if ({self.expr(m.group(1))}) {{"]
        if _RE_ELSE.match(line):
            return ["} else {"]
        if _RE_BRACE_CLOSE.match(line):
            return ["}"]
        if m := _RE_RETURN.match(line):
            return [f"return {self.expr(m.group(1))};".replace("return ;", "return;")]

        # plain assignment `name = expr;` (targets may be struct members:
        # `var_contact_data.field = ...`) — must be tried before the decl
        # regex, whose lazy `(\S.*?)` can otherwise swallow `ret_0 =` as a
        # "type" on multi-output ref-assign lines (T & ret_0 params)
        if m := re.match(r"^\s*(\w+(?:\.\w+)*)\s*=\s*(.+?);\s*$", line):
            target, rhs = m.group(1), m.group(2)
            if rhs.startswith("wp::address(") or rhs.startswith("&("):
                self._alias(target, rhs)
                return []  # alias collapsed: no MSL line, no decl (pointer decl was dropped)
            if rhs.startswith("wp::load("):
                spec = self._resolve_array_load(rhs)
                if spec is not None:
                    # array-typed value copy (`t = wp::load(arr_alias)`): the
                    # target becomes another name for the same binding pair
                    self.array_args[target] = spec
                    return []
                # otherwise fall through: a value-alias load (normal path)
            if rhs in self.array_args:
                # array value copy (`t = arr;`): register as another name for
                # the same binding pair. Bool contexts still resolve (expr
                # treats any array binding as statically true — upstream tests
                # the data pointer, always bound here).
                self.array_args[target] = self.array_args[rhs]
                return []
            if target in self.deferred_any:
                # first assignment to a `wp::Any` temp (wp.static lowering):
                # declare with a deduced `auto` type at the point of init
                self.deferred_any.discard(target)
                return [f"const auto {target} = {self.expr(rhs)};"]
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
            if _RE_ARRAY_T.match(ctype.strip()):
                # array-typed value decls carry no data (the binding pair is
                # registered when the var is assigned a load of another array)
                if init is not None and init.startswith("wp::load("):
                    spec = self._resolve_array_load(init)
                    if spec is None:
                        raise TranslationError(f"array decl from non-array load {init!r}")
                    self.array_args[name] = spec
                return []
            if ctype.strip() == "wp::Any":
                # wp.static temps: type-erased in the SSA; declared as
                # `const auto` at the first (only) assignment instead
                self.deferred_any.add(name)
                return []
            if ctype.strip() == "wp::str":
                # string-literal temps feed only wp.print, which is elided
                return []
            if not ctype:
                raise TranslationError(f"EMPTY TYPE on line {line!r}")
            try:
                msl_t = msl_type(ctype)
            except TranslationError as e:
                raise TranslationError(f"decl line {line!r}: {e}")
            if _RE_GEN_MAT.match(msl_t):
                ensure_mat_def(self.unit, msl_t)
            # warp constant tables arrive as flattened brace initializers
            # (row-major linear, matching the struct's c[M][N] layout); MSL
            # wants per-row braces, so reshape flat → rows for generic mats
            if init is not None and init.startswith("wp::initializer_array<"):
                brace = init.index("{")
                elems = init[brace + 1 : -1].strip()
                m_gen = _RE_GEN_MAT.match(msl_t)
                if m_gen:
                    rows, cols = int(m_gen.group(1)), int(m_gen.group(2))
                    flat = [e.strip() for e in elems.split(",")]
                    if len(flat) != rows * cols:
                        raise TranslationError(
                            f"initializer_array<{len(flat)}> does not fill {msl_t} ({rows}x{cols})"
                        )
                    row_inits = ", ".join(
                        "{" + ", ".join(flat[r * cols : (r + 1) * cols]) + "}" for r in range(rows)
                    )
                    # struct → c[M][N] member → rows: full bracing silences
                    # -Wmissing-braces (Metal surfaces warnings before errors
                    # and the 1KB error buffer would clip the real failure)
                    elems = "{" + row_inits + "}"
                return [f"{'const ' if const else ''}{msl_t} {name} = {{{elems}}};"]
            if init is not None:
                return [f"{'const ' if const else ''}{msl_t} {name} = {self.expr(init)};"]
            # zero-init: Metal promotes -Wsometimes-uninitialized to an error
            # on phi-merged temps; SSA assigns before every reachable read,
            # so a static zero is never observable
            if msl_t in ("vec2f", "vec3f", "vec4f", "quatf", "mat22f", "mat33f", "mat44f",
                         "transformf", "spatial_vectorf", "vec2i", "vec3i", "vec4i",
                         "wp_range", "wp_mesh") or msl_t in self.unit.structs:
                return [f"{'const ' if const else ''}{msl_t} {name} = {{}};"]
            return [f"{'const ' if const else ''}{msl_t} {name} = 0;"]

        if m := re.match(r"^\s*(.+?);\s*$", line):
            return [self.call_stmt(m.group(1)) + ";"]

        raise TranslationError(f"unparsed statement {line!r}")

    def _resolve_array_load(self, rhs: str) -> ArgSpec | None:
        """Spec behind ``wp::load(<var>)`` when the load yields an array."""
        inner = rhs[len("wp::load(") : -1].strip() if rhs.endswith(")") else None
        if inner is None:
            return None
        return self.array_args.get(inner)

    def _alias(self, var: str, init: str) -> None:
        if init.startswith("wp::address("):
            args = split_args(init[len("wp::address(") : -1])
            arr_var = args[0].strip()
            spec = self.array_args.get(arr_var)
            if spec is None:
                raise TranslationError(f"wp::address on non-array arg {arr_var!r}")
            idx = ", ".join(self.expr(a) for a in args[1:])
            expr = f"wp_address({spec.name}_data, {self._meta_ref(spec)}{', ' + idx if idx else ''})"
            self.aliases[var] = (True, expr)
            return
        # &(expr) reference forms
        inner = init[2:-1].strip()
        shape_of = re.match(r"^\(?(var_\w+)\.shape\)?$", inner)
        if shape_of:
            spec = self.array_args.get(shape_of.group(1))
            if spec is None:
                raise TranslationError(f"&(...shape) on non-array arg {inner!r}")
            s = f"{self._meta_ref(spec)}.shape"
            expr = f"int4({s}[0], {s}[1], {s}[2], {s}[3])"
            self.aliases[var] = (False, expr)
            return
        # struct member reference &((var).field)
        member = re.match(r"^\((\w+)\)((?:\.\w+)+)$", inner)
        if not member:
            # pointer-dereference member form &(var_x->field): var_x is a
            # value alias → (alias_expr).field; var_x is an array binding →
            # the array's shape vector (same as the &(arr.shape) form)
            arrow = re.match(r"^(\w+)->(\w+)$", inner)
            if arrow:
                base = arrow.group(1)
                if arrow.group(2) == "shape" and base in self.array_args:
                    s = f"{self._meta_ref(self.array_args[base])}.shape"
                    self.aliases[var] = (False, f"int4({s}[0], {s}[1], {s}[2], {s}[3])")
                    return
                alias = self.aliases.get(base)
                if alias is None or alias[0]:
                    raise TranslationError(f"&(...->...) on non-value alias {inner!r}")
                self.aliases[var] = (False, f"{alias[1]}.{arrow.group(2)}")
                return
            raise TranslationError(f"unsupported reference form {init!r}")
        # split struct array fields bind like arrays: the alias var becomes a
        # name for the field's own (data ptr, meta) binding pair. Scalar
        # fields fall through to a plain value alias of the member expression.
        key = member.group(1) + member.group(2)
        spec = self.array_args.get(key)
        if spec is not None:
            self.array_args[var] = spec
            return
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
            return f"*wp_address({spec.name}_data, {self._meta_ref(spec)}{', ' + idx if idx else ''}) = {value}"
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
        return f"wp_{op}{suffix}({spec.name}_data, {self._meta_ref(spec)}{', ' + idx if idx else ''}, {value})"

    def _array_arg_spec(self, arg: str) -> ArgSpec | None:
        """Binding spec for an SSA array argument (direct var or load of one)."""
        spec = self.array_args.get(arg.strip())
        if spec is not None:
            return spec
        if arg.strip().startswith("wp::load("):
            return self._resolve_array_load(arg.strip())
        return None

    def _call_user(self, src: str) -> str:
        name = src.split("(", 1)[0]
        close = _find_balanced(src, len(name))
        args = split_args(src[len(name) + 1 : close])
        fn = self.user_funcs.get(name)
        if fn is None:
            # struct value constructor (Name_hash(...)) — rewrite to make_<T>
            if name in self.unit.structs:
                return f"make_{name}({', '.join(self.expr(a) for a in args)})"
            if name in ("printf", "print"):
                # wp.printf lowers to a native printf call; elided like wp::print
                return "/* wp.print elided (Metal) */ 0"
            return f"{name}({', '.join(self.expr(a) for a in args)})"
        out: list[str] = []
        vi = 0
        for p in fn.params:
            if vi >= len(args):
                raise TranslationError(f"call {name}: missing argument for param {p.name}")
            arg = args[vi]
            vi += 1
            if p.kind == "array":
                spec = self._array_arg_spec(arg)
                if spec is None:
                    raise TranslationError(f"array param {p.name} of {name} got non-array {arg!r}")
                out.append(f"{spec.name}_data")
                out.append(self._meta_ref(spec))
            elif p.kind == "struct" and p.array_fields:
                # struct value + one (data, meta) pair per split array field
                out.append(self.expr(arg))
                for field, _elem_ctype in p.array_fields:
                    spec = self.array_args.get(f"{arg.strip()}.{field}")
                    if spec is None:
                        raise TranslationError(
                            f"struct param {p.name} of {name}: array field {field!r} has no binding"
                        )
                    out.append(f"{spec.name}_data")
                    out.append(self._meta_ref(spec))
            else:
                out.append(self.expr(arg))
        if vi != len(args):
            raise TranslationError(f"call {name}: {len(args)} args for {len(fn.params)} params")
        return f"{name}({', '.join(out)})"


_RE_ITER = re.compile(r"\b(iter_cmp|iter_next)\(")
_RE_SEC = re.compile(r"^\s*//\s*(argument vars|primal vars|forward)\s*$")


def _split_bool_op(src: str, op: str) -> tuple[str, str] | None:
    """Split `a && b` / `a || b` at the top level (outside parens/brackets)."""
    depth = 0
    i = 0
    while i < len(src) - 1:
        c = src[i]
        if c in "(<[{":
            depth += 1
        elif c in ")>]}" and depth > 0:
            depth -= 1
        elif depth == 0 and src.startswith(op, i):
            return src[:i].strip(), src[i + len(op):].strip()
        i += 1
    return None


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

    array_args: dict[str, ArgSpec] = {}
    for var in adj.args:
        ctype = _Var.type_to_ctype(var.type)
        kind = _classify_ctype(ctype)
        spec = ArgSpec(var.label, kind, ctype)
        unit.args.append(spec)
        if kind == "struct":
            spec.array_fields = _emit_struct_def(var.type, unit, _codegen, split_arrays=True)
            for field, elem_ctype in spec.array_fields:
                array_args[f"var_{var.label}.{field}"] = ArgSpec(
                    f"{var.label}__{field}", "array", elem_ctype
                )
        elif kind == "array":
            array_args[f"var_{var.label}"] = spec

    # ---- transitive user functions, callees first ----
    user_funcs: dict[str, MSLFunction] = {}

    def visit(f):
        fname = f.native_func
        if fname in user_funcs:
            return
        snippet = getattr(f, "native_snippet", None)
        if snippet:
            fn = _translate_native_function(f, snippet, unit, _codegen)
            user_funcs[fname] = fn
            unit.functions.append(fn)
            return
        for callee in f.adj.called_user_functions:
            visit(callee)
        fn = _translate_function(f, unit, user_funcs, _codegen)
        user_funcs[fname] = fn
        unit.functions.append(fn)

    for uf in adj.called_user_functions:
        visit(uf)

    # ---- kernel body ----
    src = _codegen.codegen_func_forward(adj, func_type="kernel", device="cpu")
    tr = _BodyTranslator(unit, array_args, user_funcs)
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


def _translate_native_function(f, snippet: str, unit: KernelUnit, _codegen) -> MSLFunction:
    """Include a ``@wp.func_native`` snippet verbatim as an MSL function.

    func_native snippets are portable C++ by construction (upstream runs the
    same text on the CPU backend). MSL is C++14-based, so a plain-C++ snippet
    compiles verbatim; one using CUDA intrinsics fails loudly at Metal
    library compile — never silently. The registered snippet references the
    Python parameter names, so the MSL signature keeps them.
    """
    from ..frontend.codegen import Var as _Var

    fname = f.native_func
    params: list[Param] = []
    for var in f.adj.args:
        ctype = _Var.type_to_ctype(var.type)
        params.append(Param("value", msl_type(ctype), var.label))
    ret_ctype = _Var.type_to_ctype(f.adj.arg_types["return"]) if "return" in f.adj.arg_types else None
    ret_msl = None if ret_ctype is None else msl_type(ret_ctype)
    fn = MSLFunction(fname, ret_msl)
    fn.params = params
    body = snippet.strip("\n")
    # MSL dialect adapter: references must carry an address space qualifier;
    # a local lvalue's only legal address space is `thread` (semantics equal
    # to the unqualified CPU C++ form the snippets are written against)
    body = re.sub(r"reinterpret_cast<\s*([^<>]+?)\s*&\s*>", r"reinterpret_cast<thread \1&>", body)
    fn.lines = [f"    {ln}" for ln in body.splitlines()]
    return fn


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
    array_args: dict[str, ArgSpec] = {}

    for arg_src in split_args(header[open_idx + 1 : close_idx]):
        if "&" in arg_src:
            ctype, pname = arg_src.split("&")
            fn.params.append(Param("ref", msl_type(ctype), pname.strip()))
            continue
        ctype, pname = arg_src.rsplit(None, 1)
        if _classify_ctype(ctype) == "array":
            fn.params.append(Param("array", array_elem_msl(ctype), pname))
            # C++ SSA function params already carry the var_ prefix
            key = pname if pname.startswith("var_") else f"var_{pname}"
            array_args[key] = ArgSpec(pname, "array", ctype)
        elif _classify_ctype(ctype) == "struct":
            if ctype in unit.struct_array_fields:
                split_fields = unit.struct_array_fields[ctype]
            else:
                struct_type = next(
                    (v.type for v in f.adj.args if v.label == pname.removeprefix("var_")), None
                )
                if struct_type is None:
                    raise TranslationError(
                        f"function {fname}: struct param {pname!r} ({ctype}) has no matching adjoint arg"
                    )
                split_fields = _emit_struct_def(struct_type, unit, _codegen, split_arrays=True)
            fn.params.append(Param("struct", msl_type(ctype), pname, array_fields=split_fields))
            for field, elem_ctype in split_fields:
                # composite key so the body resolves field array references
                binding = f"{pname}__{field}"
                array_args[f"{pname}.{field}"] = ArgSpec(binding, "array", elem_ctype)
        else:
            fn.params.append(Param("value", msl_type(ctype), pname))

    tr = _BodyTranslator(unit, array_args, user_funcs, scope="func")
    section = None
    depth = 0  # emitted-brace depth; 0 → the C++ function's closing brace
    for line in lines[body_start:]:
        if m := _RE_SEC.match(line):
            section = m.group(1)
            continue
        if section in (None, "argument vars"):
            continue  # args are the MSL signature params already
        if line.strip() == "}" and depth == 0:
            break  # function-closing brace (early `return` blocks nest deeper)
        out = tr.stmt(line)
        for emitted in out:
            s = emitted.strip()
            if s.endswith("{"):
                depth += 1
            elif s == "}":
                depth -= 1
        fn.lines.extend(out)
    return fn


def _emit_struct_def(s, unit: KernelUnit, _codegen, split_arrays: bool = False) -> list[tuple[str, str]]:
    """Emit an MSL struct definition (warp-native layout) + constructors.

    With ``split_arrays`` (parameter context), array-typed fields are excluded
    from the MSL struct — Metal cannot store device pointers inside a
    ``constant`` buffer struct — and returned as ``[(field, elem ctype)]`` for
    per-field buffer binding. Nested struct fields always emit recursively
    with ``split_arrays=False``: a nested array field has no binding path and
    fails loud.
    """
    from ..frontend.codegen import Var as _Var

    if s.native_name in unit.structs:
        return unit.struct_array_fields.get(s.native_name, [])
    fields: list[tuple[str, str]] = []
    split: list[tuple[str, str]] = []
    for label, var in s.vars.items():
        ctype = _Var.type_to_ctype(var.type)
        if _RE_ARRAY_T.match(ctype):
            if not split_arrays:
                raise TranslationError(
                    f"struct {s.native_name} field {label!r}: array fields are only supported on "
                    "kernel/function parameters, where each field gets its own buffer binding "
                    "(KEP-0001 M1.4); a struct value cannot embed an array"
                )
            # record the ELEMENT type (the array_t wrapper becomes a binding)
            split.append((label, _RE_ARRAY_T.match(ctype).group(1)))
            continue
        if isinstance(var.type, type(s)):
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
    if split:
        unit.struct_array_fields[s.native_name] = split
    return split
