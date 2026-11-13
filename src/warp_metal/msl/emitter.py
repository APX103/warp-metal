# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""MSL emitter, spike subset (KEP-0001 M0.3).

Original work (not vendored). Compiles a straight-line kernel function
(plain Python function with warp type annotations) to an MSL compute
kernel, resolving ``wp.*`` calls against the vendored builtin registry.

Spike scope: ``tid``, scalar/vector arithmetic, array load/store,
``atomic_add`` (fp32 device atomics), ``mat33`` multiply/inverse,
``select``/``where``, ``sin``/``cos``, ``dot``/``cross``, ``length``/
``normalize``. No control flow, no adjoint, no tiles — Phase 1+ migrates
onto the vendored Adjoint walker once the module builder (M1.2) exists;
this file is the reference for the MSL dialect that migration targets.

Numerics note: vector/matrix types are emitted as plain structs with the
same layout as warp's native ``vec_t``/``mat_t`` (row-major ``c[r][col]``
matrices) and math is ported from warp's native implementations, so Metal
results are bit-comparable with the CPU path within fp32 tolerance.
``sin``/``cos`` map to ``precise::`` intrinsics for the same reason.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from ..frontend.types import (
    float32,
    int32,
    mat22f,
    mat33f,
    mat44f,
    quatf,
    vec2f,
    vec3f,
    vec4f,
)

# ---------------------------------------------------------------------------
# warp type -> MSL type mapping
# ---------------------------------------------------------------------------

_SCALAR_TO_MSL = {
    "float32": "float",
    "float16": "half",
    "int32": "int",
    "int64": "int64_t",
    "uint32": "uint",
    "uint64": "uint64_t",
    "int8": "char",
    "uint8": "uchar",
    "int16": "short",
    "uint16": "ushort",
}

_VEC_FIELDS = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "w"),
}

# mat inverse port source: warp native mat_t<N,N,T>::inverse (cofactor form)
_MAT_SIZES = {mat22f: 2, mat33f: 3, mat44f: 4}


def msl_type_name(warp_type) -> str:
    """Map a concrete warp value type to its MSL struct/scalar name."""
    name = getattr(warp_type, "__name__", None)
    if name in _SCALAR_TO_MSL:
        return _SCALAR_TO_MSL[name]
    if name in ("vec2f", "vec3f", "vec4f", "quatf", "mat22f", "mat33f", "mat44f"):
        return name
    raise NotImplementedError(f"MSL mapping for type {warp_type} not in spike subset")


def _dtype_of_array(array_type) -> type:
    dtype = getattr(array_type, "dtype", None)
    if dtype is None:
        raise TypeError(f"array annotation without dtype: {array_type}")
    return dtype


def _array_type_name(t) -> str:
    return getattr(type(t), "__name__", getattr(t, "__name__", ""))


def _is_array_type(t) -> bool:
    return _array_type_name(t) in ("array", "array_1d") and hasattr(t, "dtype")


# ---------------------------------------------------------------------------
# MSL preamble: structs (warp-native layout) + math (ported from warp native)
# ---------------------------------------------------------------------------

_PREAMBLE = """\
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ---- scalarized value types (warp-native layout) ----
struct vec2f { float x, y; };
struct vec3f { float x, y, z; };
struct vec4f { float x, y, z, w; };
struct quatf { float x, y, z, w; };
struct mat22f { float c[2][2]; };  // c[row][col], warp row-major
struct mat33f { float c[3][3]; };
struct mat44f { float c[4][4]; };

// ---- constructors ----
static inline vec2f make_vec2f(float x, float y) { vec2f v; v.x = x; v.y = y; return v; }
static inline vec3f make_vec3f(float x, float y, float z) { vec3f v; v.x = x; v.y = y; v.z = z; return v; }
static inline vec4f make_vec4f(float x, float y, float z, float w) { vec4f v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }

// ---- componentwise operators ----
static inline vec2f operator+(vec2f a, vec2f b) { return make_vec2f(a.x+b.x, a.y+b.y); }
static inline vec2f operator-(vec2f a, vec2f b) { return make_vec2f(a.x-b.x, a.y-b.y); }
static inline vec2f operator*(vec2f a, vec2f b) { return make_vec2f(a.x*b.x, a.y*b.y); }
static inline vec2f operator*(float s, vec2f b) { return make_vec2f(s*b.x, s*b.y); }
static inline vec2f operator*(vec2f a, float s) { return make_vec2f(a.x*s, a.y*s); }
static inline vec2f operator/(vec2f a, vec2f b) { return make_vec2f(a.x/b.x, a.y/b.y); }
static inline vec2f operator/(vec2f a, float s) { return make_vec2f(a.x/s, a.y/s); }
static inline vec3f operator+(vec3f a, vec3f b) { return make_vec3f(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3f operator-(vec3f a, vec3f b) { return make_vec3f(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3f operator*(vec3f a, vec3f b) { return make_vec3f(a.x*b.x, a.y*b.y, a.z*b.z); }
static inline vec3f operator*(float s, vec3f b) { return make_vec3f(s*b.x, s*b.y, s*b.z); }
static inline vec3f operator*(vec3f a, float s) { return make_vec3f(a.x*s, a.y*s, a.z*s); }
static inline vec3f operator/(vec3f a, vec3f b) { return make_vec3f(a.x/b.x, a.y/b.y, a.z/b.z); }
static inline vec3f operator/(vec3f a, float s) { return make_vec3f(a.x/s, a.y/s, a.z/s); }
static inline vec4f operator+(vec4f a, vec4f b) { return make_vec4f(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
static inline vec4f operator*(float s, vec4f b) { return make_vec4f(s*b.x, s*b.y, s*b.z, s*b.w); }
static inline vec4f operator*(vec4f a, float s) { return make_vec4f(a.x*s, a.y*s, a.z*s, a.w*s); }
static inline vec4f operator-(vec4f a, vec4f b) { return make_vec4f(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }

// ---- geometry (ported from warp native warp.cpp) ----
static inline float dot(vec2f a, vec2f b) { return a.x*b.x + a.y*b.y; }
static inline float dot(vec3f a, vec3f b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float dot(vec4f a, vec4f b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
static inline vec3f cross(vec3f a, vec3f b) {
    return make_vec3f(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float length(vec2f a) { return sqrt(dot(a, a)); }
static inline float length(vec3f a) { return sqrt(dot(a, a)); }
static inline float length(vec4f a) { return sqrt(dot(a, a)); }
static inline vec2f normalize(vec2f a) { float l = length(a); return make_vec2f(a.x/l, a.y/l); }
static inline vec3f normalize(vec3f a) { float l = length(a); return make_vec3f(a.x/l, a.y/l, a.z/l); }
static inline vec4f normalize(vec4f a) { float l = length(a); return make_vec4f(a.x/l, a.y/l, a.z/l, a.w/l); }

// ---- mat33 (ported from warp native mat_t<3,3> multiply/inverse) ----
static inline vec3f mat33_mul_vec(mat33f m, vec3f v) {
    return make_vec3f(
        m.c[0][0]*v.x + m.c[0][1]*v.y + m.c[0][2]*v.z,
        m.c[1][0]*v.x + m.c[1][1]*v.y + m.c[1][2]*v.z,
        m.c[2][0]*v.x + m.c[2][1]*v.y + m.c[2][2]*v.z);
}
static inline mat33f mat33_mul_mat(mat33f a, mat33f b) {
    mat33f r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.c[i][j] = a.c[i][0]*b.c[0][j] + a.c[i][1]*b.c[1][j] + a.c[i][2]*b.c[2][j];
    return r;
}
// cofactor (adjugate) inverse, matches warp native mat_inverse for 3x3
static inline mat33f mat33_inverse(mat33f m) {
    float c00 =  (m.c[1][1]*m.c[2][2] - m.c[1][2]*m.c[2][1]);
    float c01 = -(m.c[1][0]*m.c[2][2] - m.c[1][2]*m.c[2][0]);
    float c02 =  (m.c[1][0]*m.c[2][1] - m.c[1][1]*m.c[2][0]);
    float det = m.c[0][0]*c00 + m.c[0][1]*c01 + m.c[0][2]*c02;
    float inv_det = 1.0f / det;
    mat33f r;
    r.c[0][0] = c00 * inv_det;
    r.c[0][1] = (m.c[0][2]*m.c[2][1] - m.c[0][1]*m.c[2][2]) * inv_det;
    r.c[0][2] = (m.c[0][1]*m.c[1][2] - m.c[0][2]*m.c[1][1]) * inv_det;
    r.c[1][0] = c01 * inv_det;
    r.c[1][1] = (m.c[0][0]*m.c[2][2] - m.c[0][2]*m.c[2][0]) * inv_det;
    r.c[1][2] = (m.c[0][2]*m.c[1][0] - m.c[0][0]*m.c[1][2]) * inv_det;
    r.c[2][0] = c02 * inv_det;
    r.c[2][1] = (m.c[0][1]*m.c[2][0] - m.c[0][0]*m.c[2][1]) * inv_det;
    r.c[2][2] = (m.c[0][0]*m.c[1][1] - m.c[0][1]*m.c[1][0]) * inv_det;
    return r;
}

// ---- fp32 device atomics (Metal 2+; replaces warp's CUDA atomic path) ----
static inline void atomic_add_float(device atomic_float* addr, float value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline void atomic_add_int(device atomic_int* addr, int value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline void atomic_add_uint(device atomic_uint* addr, uint value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
"""


def _atomic_addr_cast(array_elem_msl: str) -> str:
    return {
        "float": "atomic_float",
        "int": "atomic_int",
        "uint": "atomic_uint",
    }[array_elem_msl]


# ---------------------------------------------------------------------------
# builtins -> MSL emission
# ---------------------------------------------------------------------------

# key -> (preamble function name) for identity-mappable builtins
_IDENTITY_BUILTINS = {
    "dot": "dot",
    "cross": "cross",
    "length": "length",
    "normalize": "normalize",
    "min": "min",
    "max": "max",
    "abs": "abs",
    "sqrt": "sqrt",
    "floor": "floor",
    "ceil": "ceil",
}

_PRECISE_BUILTINS = {"sin": "precise::sin", "cos": "precise::cos"}

# key -> {arg type: preamble function} for builtins whose MSL name depends
# on the argument type (generic overloads resolved against the registry)
_TYPED_BUILTINS = {
    "inverse": {
        mat33f: "mat33_inverse",
        mat22f: "mat22_inverse",
        mat44f: "mat44_inverse",
    },
}


class _KernelSignature:
    def __init__(self, func):
        self.name = func.__name__
        self.args = []  # (py_name, warp_type, is_array, dtype)
        import typing

        ann = typing.get_type_hints(func)  # evaluates string annotations
        params = inspect.signature(func).parameters
        for pname in params:
            t = ann.get(pname)
            if t is None:
                raise TypeError(f"kernel '{self.name}' param '{pname}' missing type annotation")
            if _is_array_type(t):
                self.args.append((pname, t, True, _dtype_of_array(t)))
            else:
                self.args.append((pname, t, False, None))


class MSLEmitter:
    """Compile one straight-line kernel function to MSL source."""

    def __init__(self, builtin_functions):
        self.builtin_functions = builtin_functions

    # -- public entry -------------------------------------------------------

    def emit(self, func) -> str:
        sig = _KernelSignature(func)
        self._sig = sig
        self._lines: list[str] = []
        self._tmp = 0
        self._env: dict[str, tuple[str, type]] = {}  # py name -> (msl type, warp type)
        self._array_args: set[str] = set()

        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        body = []
        for n in fn_node.body:
            # drop the docstring only
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                continue
            body.append(n)

        for name, wtype, is_array, dtype in sig.args:
            msl_t = msl_type_name(dtype) if is_array else msl_type_name(wtype)
            self._env[name] = (msl_t, dtype if is_array else wtype)
            if is_array:
                self._array_args.add(name)

        for node in body:
            self._emit_statement(node)

        args = []
        for i, (name, wtype, is_array, dtype) in enumerate(sig.args):
            msl_t = msl_type_name(dtype) if is_array else msl_type_name(wtype)
            args.append(f"    device {msl_t}* {name} [[buffer({i})]],")
        args.append(f"    uint3 _gid [[thread_position_in_grid]]")

        body_src = "\n".join(self._lines) or "    ;"
        return (
            f"// generated by warp-metal msl emitter — kernel {sig.name}\n"
            + _PREAMBLE
            + f"\nkernel void {sig.name}(\n" + "\n".join(args) + "\n) {\n"
            + "    const int tid = int(_gid.x);\n"
            + body_src
            + "\n}\n"
        )

    # -- statements ----------------------------------------------------------

    def _emit_statement(self, node):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise NotImplementedError("tuple assignment not in spike subset")
            value_src, vtype = self._emit_expr(node.value)
            self._emit_target(node.targets[0], value_src, vtype)
        elif isinstance(node, ast.AugAssign):
            value_src, vtype = self._emit_expr(node.value)
            target_src, ttype = self._emit_lvalue(node.target)
            if not _is_array_type(ttype):
                op = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/"}[type(node.op).__name__]
                self._lines.append(f"    {target_src} = {target_src} {op} {value_src};")
            else:
                raise NotImplementedError("aug-assign into array not in spike subset")
        elif isinstance(node, ast.Expr):
            src, _ = self._emit_expr(node.value)
            if src:
                self._lines.append(f"    {src};")
        else:
            raise NotImplementedError(f"statement {type(node).__name__} not in spike subset")

    def _emit_target(self, target, value_src, vtype):
        if isinstance(target, ast.Name):
            if target.id in self._env:
                self._lines.append(f"    {target.id} = {value_src};")
            else:
                msl_t = msl_type_name(vtype)
                self._env[target.id] = (msl_t, vtype)
                self._lines.append(f"    {msl_t} {target.id} = {value_src};")
        elif isinstance(target, ast.Subscript):
            arr_src, arr_type = self._emit_lvalue(target.value)
            idx_src, _ = self._emit_expr(target.slice)
            self._lines.append(f"    {arr_src}[{idx_src}] = {value_src};")
        elif isinstance(target, ast.Attribute):
            base_src, _ = self._emit_lvalue(target.value)
            self._lines.append(f"    {base_src}.{target.attr} = {value_src};")
        else:
            raise NotImplementedError(f"assign target {type(target).__name__} not in spike subset")

    def _emit_lvalue(self, node) -> tuple[str, type]:
        if isinstance(node, ast.Name):
            return node.id, self._env[node.id][1]
        if isinstance(node, ast.Subscript):
            arr_src, arr_type = self._emit_lvalue(node.value)
            idx_src, _ = self._emit_expr(node.slice)
            dtype = arr_type
            return f"{arr_src}[{idx_src}]", dtype
        if isinstance(node, ast.Attribute):
            base_src, base_type = self._emit_lvalue(node.value)
            return f"{base_src}.{node.attr}", base_type
        raise NotImplementedError(f"lvalue {type(node).__name__} not in spike subset")

    # -- expressions ---------------------------------------------------------

    def _emit_expr(self, node) -> tuple[str, type]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return ("true" if node.value else "false"), type(None)
            if isinstance(node.value, int):
                return f"{node.value}", self._int_hint()
            if isinstance(node.value, float):
                return f"{node.value!r}f", float32
            raise NotImplementedError(f"constant {node.value!r}")
        if isinstance(node, ast.Name):
            entry = self._env.get(node.id)
            if entry is None:
                raise NameError(f"unresolved kernel variable '{node.id}'")
            return node.id, entry[1]
        if isinstance(node, ast.Attribute):
            base_src, base_type = self._emit_expr(node.value)
            if _array_type_name(base_type) in ("vec2f", "vec3f", "vec4f", "quatf"):
                return f"{base_src}.{node.attr}", float32
            raise NotImplementedError(f"attribute {node.attr} on {base_type}")
        if isinstance(node, ast.BinOp):
            return self._emit_binop(node)
        if isinstance(node, ast.UnaryOp):
            src, t = self._emit_expr(node.operand)
            op = {ast.USub: "-", ast.UAdd: "+"}[type(node.op)]
            return f"({op}{src})", t
        if isinstance(node, ast.Compare):
            left, lt = self._emit_expr(node.left)
            if len(node.ops) != 1:
                raise NotImplementedError("chained compare not in spike subset")
            right, _ = self._emit_expr(node.comparators[0])
            op = {
                ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                ast.Gt: ">", ast.GtE: ">=",
            }[type(node.ops[0])]
            return f"({left} {op} {right})", type(None)
        if isinstance(node, ast.BoolOp):
            parts = [self._emit_expr(v)[0] for v in node.values]
            op = "&&" if isinstance(node.op, ast.And) else "||"
            return "(" + f" {op} ".join(parts) + ")", type(None)
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name) and base.id in self._array_args:
                arr_msl, dtype = self._env[base.id]
                idx_src, _ = self._emit_expr(node.slice)
                return f"{base.id}[{idx_src}]", dtype
            arr_src, arr_type = self._emit_expr(node.value)
            raise NotImplementedError("subscript on non-array not in spike subset")
        if isinstance(node, ast.Call):
            return self._emit_call(node)
        raise NotImplementedError(f"expression {type(node).__name__} not in spike subset")

    def _emit_binop(self, node) -> tuple[str, type]:
        left, lt = self._emit_expr(node.left)
        right, rt = self._emit_expr(node.right)
        opname = type(node.op).__name__
        if opname == "MatMult":
            if lt is mat33f and rt is vec3f:
                return f"mat33_mul_vec({left}, {right})", vec3f
            if lt is mat33f and rt is mat33f:
                return f"mat33_mul_mat({left}, {right})", mat33f
            raise NotImplementedError(f"@ for {lt} @ {rt}")
        op = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%"}[opname]
        return f"({left} {op} {right})", lt

    def _int_hint(self):
        return int32

    def _resolve_overload(self, key, arg_warp_types):
        func = self.builtin_functions.get(key)
        if func is None:
            raise NameError(f"'{key}' is not a registered warp builtin")
        for ov in func.overloads:
            it = ov.input_types
            if len(it) != len(arg_warp_types):
                continue
            if all(it[k] is t for k, t in zip(it, arg_warp_types)):
                return ov
        return None

    def _emit_call(self, node) -> tuple[str, type]:
        resolved = self._resolve_call(node)
        if resolved is not None:
            return resolved
        raise NotImplementedError(f"call {ast.unparse(node)} not in spike subset")

    def _resolve_call(self, node):
        func_node = node.func
        # spike kernels address builtins as wp.<key> or bare <key>
        if isinstance(func_node, ast.Attribute):
            key = func_node.attr
        elif isinstance(func_node, ast.Name):
            key = func_node.id
        else:
            return None
        if key == "tid" and not node.args:
            return "tid", int32
        arg_results = [self._emit_expr(a) for a in node.args]
        arg_types = [t for _, t in arg_results]
        ov = self._resolve_overload(key, arg_types)

        if key == "atomic_add":
            arr_name, arr_type = self._emit_expr(node.args[0])
            elem = msl_type_name(arr_type)
            idx_src, _ = self._emit_expr(node.args[1])
            val_src, _ = self._emit_expr(node.args[2])
            return (
                f"atomic_add_{elem}(reinterpret_cast<device {_atomic_addr_cast(elem)}*>(&{arr_name}[{idx_src}]), {val_src})",
                self._overload_return_type(ov, arg_types) if ov is not None else None,
            )
        if key in ("select", "where"):
            cond_src, _ = self._emit_expr(node.args[0])
            a_src, at = self._emit_expr(node.args[1])
            b_src, _ = self._emit_expr(node.args[2])
            return f"({cond_src} ? {a_src} : {b_src})", at
        if key in _PRECISE_BUILTINS:
            a_src, at = self._emit_expr(node.args[0])
            return f"{_PRECISE_BUILTINS[key]}({a_src})", at
        if key in _TYPED_BUILTINS:
            table = _TYPED_BUILTINS[key]
            a_src, at = arg_results[0]
            fn = table.get(at)
            if fn is None or ov is None:
                raise NotImplementedError(f"{key}({at}) not in spike subset")
            return f"{fn}({a_src})", self._overload_return_type(ov, arg_types)
        if key in _IDENTITY_BUILTINS:
            args_src = ", ".join(s for s, _ in arg_results)
            return f"{_IDENTITY_BUILTINS[key]}({args_src})", (
                self._overload_return_type(ov, arg_types) if ov else None
            )
        return None

    @staticmethod
    def _overload_return_type(ov, arg_types) -> type:
        if ov.value_type is not None:
            return ov.value_type
        if ov.value_func is not None:
            return ov.value_func(dict(zip(ov.input_types, arg_types)), None)
        raise TypeError(f"overload for {ov.key} has no resolvable return type")


# (atomic helpers live in _PREAMBLE above; nothing else is appended post-hoc)

