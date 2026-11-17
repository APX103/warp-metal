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

_VEC_SIZES = {2: "vec2f", 3: "vec3f", 4: "vec4f", 5: "vec5f", 6: "spatial_vectorf"}
_VECI_SIZES = {2: "vec2i", 3: "vec3i", 4: "vec4i"}
_MAT_SIZES = {(2, 2): "mat22f", (3, 3): "mat33f", (4, 4): "mat44f"}

_RE_VEC_T = re.compile(r"^wp::vec_t<(\d+),\s*wp::(float32|int32)>$")
_RE_MAT_T = re.compile(r"^wp::mat_t<(\d+),\s*(\d+),\s*wp::(float32|int32)>$")
_RE_GEN_MAT = re.compile(r"^mat(\d+)x(\d+)([fi])$")
_RE_GEN_VEC = re.compile(r"^vec(\d+)([fi])$")
# MSL struct value types (fixed family + generated) — operands of this kind
# route infix `==` through wp_equal (see the expr() comparison hook)
_RE_STRUCT_T = re.compile(r"^(?:spatial_vectorf|transformf|quatf|vec\d+[fi]|mat\d+x\d+[fi]|mat\d+[fi])$")
_RE_QUAT_T = re.compile(r"^wp::quat_t<wp::float32>$")
_RE_TRANSFORM_T = re.compile(r"^wp::transform_t<wp::float32>$")
_RE_ARRAY_T = re.compile(r"^wp::array_t<(.+)>$")
# in-device wp.zeros(): a fixed-size stack array. Functions receiving one are
# specialized with this param type; it lowers exactly like array_t — a
# (device elem* data, wp_arr_meta meta) pair. The ALLOCATION itself (the
# producing function's local) lowers to a per-thread arena slice; see the
# fixedarray ctor hook in _BodyTranslator.stmt.
_RE_FIXEDARRAY_T = re.compile(r"^wp::fixedarray_t<(\d+),\s*(.+)>$")


def ensure_mat_def(unit: KernelUnit, msl_t: str) -> None:
    """Emit the struct def + accessors for a generic ``mat<M>x<N><f|i>``.

    The fixed 2x2/3x3/4x4 float mats live in the prelude; sizes beyond those
    (Newton's FACE_NORMALS icosahedron table is 20x3) are generated per
    translation unit on first use. Rows extract to vec2/3/4 of the element
    type — anything else fails loud.

    int mats (mujoco_warp's 6x2 flex-edge endpoint tables) get the
    storage / extract / assign / where surface only: they are constant
    lookup tables, never arithmetic operands — an int-mat mul/trace would
    fail loud at Metal compile (no wp_mul/wp_trace overload), matching the
    fail-loud closure discipline.
    """
    m = _RE_GEN_MAT.match(msl_t)
    if m is None or msl_t in unit.structs:
        return
    M, N = int(m.group(1)), int(m.group(2))
    elem = "float" if m.group(3) == "f" else "int"
    vec = (_VEC_SIZES if elem == "float" else _VECI_SIZES).get(N)
    if vec is None:
        raise TranslationError(f"generic mat {msl_t}: row extract has no vec{N} mapping")
    lines = [f"struct {msl_t} {{ {elem} c[{M}][{N}]; }};"]
    ctor_args = ", ".join(
        f"{elem} a{r}{c}" for r in range(M) for c in range(N)
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
    # scalar fill (mat.h mat_t(Type s) constructor, e.g. zero-inertia guards)
    lines.append(f"static inline {msl_t} make_{msl_t}({elem} s) {{")
    lines.append(f"    {msl_t} v;")
    lines.append(f"    for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j) v.c[i][j] = s;")
    lines.append("    return v;")
    lines.append("}")
    # copy ctor (C++ mat_t(mat_t): exact memberwise copy). A single-arg SSA
    # ctor call may be the scalar fill above OR a copy of an aggregate — the
    # translator cannot type-check the arg, so MSL overload resolution
    # mirrors upstream C++ (see the prelude copy-ctor block)
    lines.append(f"static inline {msl_t} make_{msl_t}({msl_t} m) {{ return m; }}")
    comps = ", ".join(f"m.c[i][{j}]" for j in range(N))
    lines.append(
        f"static inline {vec} wp_extract({msl_t} m, int i) {{ return make_{vec}({comps}); }}"
    )
    # element form (warp extract(m, i, j) -> scalar)
    lines.append(
        f"static inline {elem} wp_extract({msl_t} m, int i, int j) {{ return m.c[i][j]; }}"
    )
    # column extract (warp extract(m, slice, j) -> vec of M comps, the
    # codegen's `m[:, j]` lowering via wp_matcol). Mats whose row count has
    # no vector form (the 6x2 int flex tables, the 20x3 icosahedron) get no
    # overload — a column extract on them fails loud at the call.
    colvec = (_VEC_SIZES if elem == "float" else _VECI_SIZES).get(M)
    if colvec is not None:
        colcomps = ", ".join(f"m.c[{i}][j]" for i in range(M))
        lines.append(
            f"static inline {colvec} wp_matcol({msl_t} m, int j) {{ return make_{colvec}({colcomps}); }}"
        )
    # row assign (warp assign_inplace(m, i, vec)); wp_extract covers every
    # row type (vec2f-5f, spatial_vectorf) and any width N (mat6x6 rows)
    assign_body = "; ".join(f"m.c[i][{j}] = wp_extract(v, {j})" for j in range(N))
    lines.append(
        f"static inline void wp_assign_inplace(thread {msl_t}& m, int i, {vec} v) {{ {assign_body}; }}"
    )
    # scalar element assign (warp assign_inplace(m, i, j, scalar))
    lines.append(
        f"static inline void wp_assign_inplace(thread {msl_t}& m, int i, int j, {elem} v) {{ m.c[i][j] = v; }}"
    )
    # scalar element add/sub (warp add/sub_inplace(m, i, j, scalar) — the
    # codegen's `m[i,j] += x` compound-assign lowering for local mats;
    # mat.h:841/1262 negative-index wrap, mutation through the thread ref)
    lines.append(
        f"static inline void wp_add_inplace(thread {msl_t}& m, int i, int j, {elem} v) {{ "
        f"if (i < 0) i += {M}; if (j < 0) j += {N}; m.c[i][j] += v; }}"
    )
    lines.append(
        f"static inline void wp_sub_inplace(thread {msl_t}& m, int i, int j, {elem} v) {{ "
        f"if (i < 0) i += {M}; if (j < 0) j += {N}; m.c[i][j] -= v; }}"
    )
    lines.append(
        f"static inline {msl_t} wp_where(bool c, {msl_t} a, {msl_t} b) {{ return c ? a : b; }}"
    )
    if elem == "float":
        if M == N:
            diag = " + ".join(f"m.c[{i}][{i}]" for i in range(N))
            lines.append(f"static inline float wp_trace({msl_t} m) {{ return {diag}; }}")
            # square-mat ops (prelude covers 2x2..4x4; generic sizes like Newton's
            # spatial-inertia mat6x6f get the same warp-mul/transpose semantics:
            # row-major r.c[i][j] = sum_k a.c[i][k]*b.c[k][j])
            lines.append(f"static inline {msl_t} wp_mul({msl_t} a, {msl_t} b) {{")
            lines.append(f"    {msl_t} r;")
            terms = " + ".join(f"a.c[i][{k}]*b.c[{k}][j]" for k in range(N))
            lines.append(f"    for (int i = 0; i < {N}; ++i)")
            lines.append(f"        for (int j = 0; j < {N}; ++j)")
            lines.append(f"            r.c[i][j] = {terms};")
            lines.append("    return r;")
            lines.append("}")
            terms_mv = " + ".join(f"m.c[i][{j}]*wp_extract(v, {j})" for j in range(N))
            lines.append(f"static inline {vec} wp_mul({msl_t} m, {vec} v) {{")
            lines.append(f"    {vec} r = make_{vec}();")
            lines.append(f"    for (int i = 0; i < {N}; ++i)")
            lines.append(f"        r = wp_vec_insert(r, i, {terms_mv});")
            lines.append("    return r;")
            lines.append("}")
            lines.append(f"static inline {msl_t} wp_mul(float s, {msl_t} m) {{")
            lines.append(f"    {msl_t} r;")
            lines.append(f"    for (int i = 0; i < {N}; ++i)")
            lines.append(f"        for (int j = 0; j < {N}; ++j)")
            lines.append("            r.c[i][j] = s*m.c[i][j];")
            lines.append("    return r;")
            lines.append("}")
            lines.append(f"static inline {msl_t} wp_mul({msl_t} m, float s) {{ return wp_mul(s, m); }}")
            lines.append(f"static inline {msl_t} wp_transpose({msl_t} m) {{")
            lines.append(f"    {msl_t} r;")
            lines.append(f"    for (int i = 0; i < {N}; ++i) for (int j = 0; j < {N}; ++j) r.c[i][j] = m.c[j][i];")
            lines.append("    return r;")
            lines.append("}")
            # value-returning component atomics (warp atomic_add/sub on arrays of
            # matrix dtype, e.g. featherstone body_I_c accumulation): row-major
            # float components are independently atomic
            for op, fname in (("add", "wp_atomic_add_ret_at"), ("sub", "wp_atomic_sub_ret_at")):
                lines.append(
                    f"static inline {msl_t} {fname}(device {msl_t}* d, wp_arr_meta m, int i, {msl_t} v) {{"
                )
                lines.append(
                    f"    device float* base = (device float*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));"
                )
                lines.append(f"    {msl_t} r;")
                lines.append(f"    for (int a = 0; a < {M}; ++a)")
                lines.append(f"        for (int b = 0; b < {N}; ++b)")
                lines.append(
                    f"            r.c[a][b] = atomic_fetch_{op}_explicit((device atomic_float*)(base + a*{N} + b), v.c[a][b], memory_order_relaxed);"
                )
                lines.append("    return r;")
                lines.append("}")
        # elementwise div, both operand orders (mat.h:3148/3161)
        lines.append(f"static inline {msl_t} wp_div({msl_t} m, float s) {{")
        lines.append(f"    {msl_t} r;")
        lines.append(f"    for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j) r.c[i][j] = m.c[i][j] / s;")
        lines.append("    return r;")
        lines.append("}")
        lines.append(f"static inline {msl_t} wp_div(float s, {msl_t} m) {{")
        lines.append(f"    {msl_t} r;")
        lines.append(f"    for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j) r.c[i][j] = s / m.c[i][j];")
        lines.append("    return r;")
        lines.append("}")
        # elementwise neg/add/sub (mat.h operator surface)
        lines.append(f"static inline {msl_t} wp_neg({msl_t} m) {{")
        lines.append(f"    {msl_t} r;")
        lines.append(f"    for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j) r.c[i][j] = -m.c[i][j];")
        lines.append("    return r;")
        lines.append("}")
        lines.append(f"static inline {msl_t} wp_add({msl_t} a, {msl_t} b) {{")
        lines.append(f"    {msl_t} r;")
        lines.append(f"    for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j) r.c[i][j] = a.c[i][j] + b.c[i][j];")
        lines.append("    return r;")
        lines.append("}")
        lines.append(f"static inline {msl_t} wp_sub({msl_t} a, {msl_t} b) {{")
        lines.append(f"    {msl_t} r;")
        lines.append(f"    for (int i = 0; i < {M}; ++i) for (int j = 0; j < {N}; ++j) r.c[i][j] = a.c[i][j] - b.c[i][j];")
        lines.append("    return r;")
        lines.append("}")
    unit.structs[msl_t] = "\n".join(lines)


def ensure_vec_def(unit: KernelUnit, msl_t: str) -> None:
    """Emit the struct def + accessors for a generic ``vecN<f|i>``.

    Mirror of :func:`ensure_mat_def` on the vector side: sizes beyond the
    fixed ``vec2f``-``vec5f``/``spatial_vectorf`` family (mujoco_warp's vec10
    cinert/crb, Newton mujoco-glue vec12 eq_data) are generated per
    translation unit on first use — same machine, sizes driven by what the
    kernels use. The component layout is the packed ``c[N]`` array (the
    prelude's named-component forms stop at 5); every helper carries
    vec.h's negative-index wrap (`if (idx < 0) idx += Length`).

    Componentwise arithmetic (operator+/dot/normalize...) has no live closure
    and is deliberately NOT generated: a call fails loud at Metal compile
    (implicit declaration) instead of silently mistranslating.
    """
    m = _RE_GEN_VEC.match(msl_t)
    if m is None or msl_t in unit.structs:
        return
    n = int(m.group(1))
    elem = "float" if m.group(2) == "f" else "int"
    # the fixed prelude family keeps its named-component layout (vec2f-5f,
    # vec2i-4i): their names match the generic pattern too, so exclude them
    # explicitly — a per-unit def here would collide with the prelude struct
    if n in (_VEC_SIZES if elem == "float" else _VECI_SIZES):
        return
    lines = [f"struct {msl_t} {{ {elem} c[{n}]; }};"]
    # zero-arg (vec.h default ctor: aggregate zero)
    lines.append(f"static inline {msl_t} make_{msl_t}() {{ {msl_t} v = {{}}; return v; }}")
    # scalar fill (vec.h vec_t(Type s) broadcast)
    lines.append(f"static inline {msl_t} make_{msl_t}({elem} s) {{")
    lines.append(f"    {msl_t} v;")
    lines.append(f"    for (int i = 0; i < {n}; ++i) v.c[i] = s;")
    lines.append("    return v;")
    lines.append("}")
    # element ctor (wp.vector(...) linear)
    ctor_args = ", ".join(f"{elem} a{i}" for i in range(n))
    ctor_body = "\n".join(f"    v.c[{i}] = a{i};" for i in range(n))
    lines.append(f"static inline {msl_t} make_{msl_t}({ctor_args}) {{")
    lines.append(f"    {msl_t} v;")
    lines.append(ctor_body)
    lines.append("    return v;")
    lines.append("}")
    # copy ctor (C++ vec_t(vec_t) memberwise copy — see the prelude copy block)
    lines.append(f"static inline {msl_t} make_{msl_t}({msl_t} v) {{ return v; }}")
    # component read (vec.h extract + component_ref, negative wrap)
    lines.append(
        f"static inline {elem} wp_extract({msl_t} v, int i) {{ if (i < 0) i += {n}; return v.c[i]; }}"
    )
    if elem == "float":
        # dot (vec.h operator*(vec,vec) reduction — the closure's live use is
        # the mjw octree SDF weight dot `wp.dot(weights[0], oct_coeff[node])`
        # over vec_t<8>; other componentwise arithmetic stays ungenerated,
        # failing loud at Metal compile instead of silently mistranslating)
        dot_body = " + ".join(f"a.c[{i}] * b.c[{i}]" for i in range(n))
        lines.append(f"static inline float wp_dot({msl_t} a, {msl_t} b) {{ return {dot_body}; }}")
    # register-vector element store (wp::store(v, i, x))
    lines.append(
        f"static inline {msl_t} wp_vec_insert({msl_t} v, int i, {elem} x) {{ "
        f"{msl_t} r = v; if (i < 0) i += {n}; r.c[i] = x; return r; }}"
    )
    # device-space lvalue element access — the inline
    # `&(wp::index(arr, ...).component_ref(k))` store target lowers to
    # `wp_vec_ref(*wp_address(...), k) = v` (see the prelude device block);
    # the generated element type lives in device memory
    lines.append(
        f"static inline device {elem}& wp_vec_ref(device {msl_t}& v, int i) {{ "
        f"if (i < 0) i += {n}; return v.c[i]; }}"
    )
    # element assign/add/sub (vec.h assign/add/sub_inplace: mutation through
    # the thread ref, negative wrap)
    lines.append(
        f"static inline void wp_assign_inplace(thread {msl_t}& v, int i, {elem} x) {{ "
        f"if (i < 0) i += {n}; v.c[i] = x; }}"
    )
    lines.append(
        f"static inline void wp_add_inplace(thread {msl_t}& v, int i, {elem} x) {{ "
        f"if (i < 0) i += {n}; v.c[i] += x; }}"
    )
    lines.append(
        f"static inline void wp_sub_inplace(thread {msl_t}& v, int i, {elem} x) {{ "
        f"if (i < 0) i += {n}; v.c[i] -= x; }}"
    )
    # lvalue element access (wp::indexref `&v[i]`, matching wp_vec_ref)
    lines.append(
        f"static inline thread {elem}& wp_vec_ref(thread {msl_t}& v, int i) {{ "
        f"if (i < 0) i += {n}; return v.c[i]; }}"
    )
    # atomic add, 2-D arrays (upstream vec.h:1831-1839 lowers wp.atomic_add on
    # vector dtypes per component through the scalar cores, returning the old
    # vector; mjw crb accumulates vec10 mass matrices on array2d). Only the
    # add family has a live closure: other ops / arities fail loud as
    # implicit declarations at Metal compile, exactly like the arithmetic
    # above — extend here when a kernel exercises them.
    if elem == "float":
        lines.append(
            f"static inline void wp_atomic_add_at(device {msl_t}* d, wp_arr_meta m, int i, int j, {msl_t} v) {{"
        )
        lines.append(
            "    device atomic_float* c = (device atomic_float*)(("
            "device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1]));"
        )
        for k in range(n):
            lines.append(f"    wp_atomic_add(c + {k}, v.c[{k}]);")
        lines.append("}")
        lines.append(
            f"static inline {msl_t} wp_atomic_add_ret_at(device {msl_t}* d, wp_arr_meta m, int i, int j, {msl_t} v) {{"
        )
        lines.append(
            "    device atomic_float* c = (device atomic_float*)(("
            "device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1]));"
        )
        lines.append(f"    {msl_t} old;")
        for k in range(n):
            lines.append(f"    old.c[{k}] = wp_atomic_add_ret(c + {k}, v.c[{k}]);")
        lines.append("    return old;")
        lines.append("}")
    unit.structs[msl_t] = "\n".join(lines)


_RE_TUPLE_T = re.compile(r"^wp::tuple_t<(.+)>$")


def ensure_tuple_def(unit: "KernelUnit", elems: list[str]) -> str:
    """Emit the struct def + tuple ctor for a ``wp::tuple_t<T0, T1, ...>``
    value tuple (upstream tuple.h tuple_t) and return its MSL struct name.

    ``elems`` are the MSL element type names in order. The closure's only
    value tuples are mjw collision_sdf's ``Tuple[vec8, vec8, vec8]`` octree
    weight triples, passed by reference through the multi-return ref-param
    lowering; member reads lower inline as ``.v{i}`` (see the wp::extract
    tuple intercept). Shape tuples (all-int ``wp::tuple(x)`` temps) never
    reach here — they stay metadata-only aliases.
    """
    name = "wp_tuple_" + "_".join(elems)
    if name in unit.structs:
        return name
    for e in elems:
        if e.startswith("wp_tuple_"):
            continue  # nested tuple: its def was ensured when it was built
        if _RE_GEN_VEC.match(e):
            ensure_vec_def(unit, e)
        elif _RE_GEN_MAT.match(e):
            ensure_mat_def(unit, e)
    lines = [f"struct {name} {{ " + " ".join(f"{e} v{i};" for i, e in enumerate(elems)) + " };"]
    ctor_args = ", ".join(f"{e} a{i}" for i, e in enumerate(elems))
    lines.append(f"static inline {name} make_{name}({ctor_args}) {{")
    lines.append(f"    {name} t;")
    for i in range(len(elems)):
        lines.append(f"    t.v{i} = a{i};")
    lines.append("    return t;")
    lines.append("}")
    unit.structs[name] = "\n".join(lines)
    return name


def msl_type(ctype: str) -> str:
    """Map a warp-native C++ type string from the SSA text to an MSL type."""
    ctype = ctype.strip()
    if ctype in _SCALAR_CTYPES:
        return _SCALAR_CTYPES[ctype]
    if m := _RE_TUPLE_T.match(ctype):
        return "wp_tuple_" + "_".join(msl_type(e.strip()) for e in split_args(m.group(1)))
    if m := _RE_VEC_T.match(ctype):
        n = int(m.group(1))
        sizes = _VEC_SIZES if m.group(2) == "float32" else _VECI_SIZES
        if n in sizes:
            return sizes[n]
        # generic size; def emitted per unit (ensure_vec_def — same machine
        # as the generic mat flow below)
        return f"vec{n}{'f' if m.group(2) == 'float32' else 'i'}"
    if m := _RE_MAT_T.match(ctype):
        key = (int(m.group(1)), int(m.group(2)))
        if m.group(3) == "float32" and key in _MAT_SIZES:
            return _MAT_SIZES[key]
        # generic size; def emitted per unit (ensure_mat_def)
        return f"mat{key[0]}x{key[1]}{'f' if m.group(3) == 'float32' else 'i'}"
    if _RE_QUAT_T.match(ctype):
        return "quatf"
    if _RE_TRANSFORM_T.match(ctype):
        return "transformf"
    if ctype == "wp::range_t":
        return "wp_range"
    if ctype == "wp::shape_t":
        return "int4"  # loaded shape vectors index via wp_extract(int4, i)
    if ctype == "wp::Mesh":
        return "wp_mesh"  # full Mesh struct (M2.2; prelude _MESH_QUERY_OPS)
    if ctype == "wp::bvh_query_t":
        return "wp_bvh_query_t"  # M2.1 query family (prelude _BVH_QUERY_OPS)
    if ctype == "wp::mesh_query_point_t":
        return "wp_mesh_query_point_t"  # M2.2 struct-return query result
    if ctype == "wp::mesh_query_ray_t":
        return "wp_mesh_query_ray_t"  # M2.2 struct-return ray hit
    if ctype == "wp::mesh_query_aabb_t":
        return "wp_mesh_query_aabb_t"  # M2.2 aabb/sphere iterator (prelude)
    if ctype == "wp::mesh_query_aabb_thread_block_t":
        # tile_mesh_query_aabb query handle: the M2.2 per-lane scalar iterator.
        # Under the cooperative lowering every lane holds the struct but only
        # lane 0's copy is advanced (results hand off through shared slots —
        # see _meshq_construct/_meshq_next); on a plain launch each thread
        # drives its own iterator, exactly upstream's CPU fiber semantics
        return "wp_mesh_query_aabb_t"
    if ctype.startswith("wp::mesh_query"):
        raise TranslationError(f"{ctype!r} has no MSL mapping (outside the KEP-0001 closure)")
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


# sizeof an MSL elem type, for fixedarray arena accounting and view strides
_ELEM_BYTES = {
    "float": 4, "int": 4, "uint": 4, "uchar": 1, "char": 1, "bool": 1,
    "uint64_t": 8, "int64_t": 8, "half": 2,
    "vec2f": 8, "vec3f": 12, "vec4f": 16, "vec5f": 20,
    "vec2i": 8, "vec3i": 12, "vec4i": 16,
    "quatf": 16, "transformf": 28, "spatial_vectorf": 24,
    "mat22f": 16, "mat33f": 36, "mat44f": 64,
}


def _shr_elem_bytes(elem: str, unit: "KernelUnit") -> int:
    """Byte size of a shared-array element: scalar/vector table first, then
    user structs (fields packed in declaration order — the MSL struct def is
    exactly the declaration-order field sequence; nested structs recurse).
    Unknown elements fail loud (upstream struct tiles carry padding/reordering
    this size model cannot prove)."""
    b = _ELEM_BYTES.get(elem)
    if b is not None:
        return b
    fields = unit.struct_fields.get(elem)
    if fields is None:
        raise TranslationError(
            f"tile element {elem!r} has no byte size (struct tiles are outside the closure)"
        )
    return sum(_shr_elem_bytes(t, unit) for _label, t in fields)


def array_elem_msl(ctype: str) -> str:
    """Elem MSL type of a ``wp::array_t<...>`` / ``wp::fixedarray_t<...>``
    annotation (device pointer)."""
    ctype = ctype.strip()
    m = _RE_ARRAY_T.match(ctype)
    if not m:
        m = _RE_FIXEDARRAY_T.match(ctype)
        if m:
            return msl_type(m.group(2))
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
    "log": "wp_log",
    "exp": "wp_exp",
    "sin": "wp_sin",
    "cos": "wp_cos",
    "tan": "wp_tan",
    "asin": "wp_asin",
    "acos": "wp_acos",
    "atan": "wp_atan",
    "atan2": "wp_atan2",
    "isfinite": "wp_isfinite",
    "nan": "wp_nan",
    "floor": "wp_floor",
    "ceil": "wp_ceil",
    "round": "wp_round",
    "trunc": "wp_trunc",
    "frac": "wp_frac",
    "lerp": "wp_lerp",
    "degrees": "wp_degrees",
    "step": "wp_step",
    "pow": "wp_pow",
    "isinf": "wp_isinf",
    "isnan": "wp_isnan",
    # random (upstream rand.h PCG port in the prelude; MSL resolves the 1/2/3
    # arg overloads like the upstream C++ overloads. The draw family takes
    # `thread uint& state` — SSA passes the state variable by name and the
    # reference binding advances it in place, matching upstream semantics)
    "rand_init": "wp_rand_init",
    "randi": "wp_randi",
    "randu": "wp_randu",
    "randf": "wp_randf",
    "randn": "wp_randn",
    # matrix
    "skew": "wp_skew",
    "diag": "wp_diag",
    "determinant": "wp_determinant",
    # vector
    "dot": "wp_dot",
    "outer": "wp_outer",
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
    # compound element assign on LOCAL aggregates (codegen subscript-augassign
    # lowering: `v[i] += x` / `m[i,j] -= x`); fixed vec/quat/mat overloads in
    # the prelude, generic mat/vec sizes generated per unit (see
    # ensure_mat_def / ensure_vec_def) — KEP-0001 M3.2 package c
    "add_inplace": "wp_add_inplace",
    "sub_inplace": "wp_sub_inplace",
    # quat
    "quat_identity": "wp_quat_identity",
    "quat_inverse": "wp_quat_inverse",
    "quat_rotate": "wp_quat_rotate",
    "quat_rotate_inv": "wp_quat_rotate_inv",
    "quat_to_matrix": "wp_quat_to_matrix",
    "quat_from_matrix": "wp_quat_from_matrix",
    "quat_from_axis_angle": "wp_quat_from_axis_angle",
    "quat_slerp": "wp_quat_slerp",
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
    # tile_mesh_query_aabb / tile_query_valid / tile_mesh_query_aabb_next are
    # NOT table calls: they lower through dedicated statement/expression
    # branches (_meshq_construct / _meshq_next) that own the shared handoff
    # slots and barriers (KEP-0001 M2.4 P5)
    "untile": "wp_untile",
    # BVH device query family (KEP-0001 M2.1; prelude _BVH_QUERY_OPS —
    # upstream bvh.h port). The SSA resolves wp.bvh_query_next to the
    # per-kind *_next builtin by the query's static type; all of them map
    # to the one runtime-kind-dispatch MSL function.
    "bvh_query_aabb": "wp_bvh_query_aabb",
    "bvh_query_ray": "wp_bvh_query_ray",
    "bvh_query_capsule": "wp_bvh_query_capsule",
    "bvh_query_sphere": "wp_bvh_query_sphere",
    "bvh_query_next": "wp_bvh_query_next",
    "bvh_query_ray_next": "wp_bvh_query_next",
    "bvh_query_capsule_next": "wp_bvh_query_next",
    "bvh_query_sphere_next": "wp_bvh_query_next",
    "bvh_query_next_dynamic": "wp_bvh_query_next",
}

# Mesh device query family (KEP-0001 M2.2; prelude _MESH_QUERY_OPS — upstream
# mesh.h port). Routed through _call's wp::mesh_* hook: the SSA call's id
# argument is expanded to `_wp_mesh_table[id]` (a wp_mesh VALUE read from the
# per-launch mesh table) and the remaining args pass through. mesh.h variants
# that need sign resolution machinery NOT in this slice raise TranslationError
# here (fail-loud Non-Goals discipline) — everything upstream taking uint64 id
# that isn't in this table is rejected.
_MESH_CALL_TABLE = {
    # struct-return + ref-out query forms share the MSL name (MSL overloads by
    # parameter count, exactly like the upstream C++ overloads)
    "mesh_query_point_no_sign": "wp_mesh_query_point_no_sign",
    "mesh_query_ray": "wp_mesh_query_ray",
    # aabb / sphere iterators: the SSA resolves wp.mesh_query_next to the
    # per-kind native builtin by the query's static type
    "mesh_query_aabb": "wp_mesh_query_aabb",
    "mesh_query_sphere": "wp_mesh_query_sphere",
    "mesh_query_aabb_next": "wp_mesh_query_aabb_next",
    "mesh_query_sphere_next": "wp_mesh_query_sphere_next",
    "mesh_query_next": "wp_mesh_query_next_dynamic",
    "mesh_query_next_dynamic": "wp_mesh_query_next_dynamic",
    # interpolation + accessors
    "mesh_eval_position": "wp_mesh_eval_position",
    "mesh_eval_velocity": "wp_mesh_eval_velocity",
    "mesh_eval_face_normal": "wp_mesh_eval_face_normal",
    "mesh_get_point": "wp_mesh_get_point",
    "mesh_get_velocity": "wp_mesh_get_velocity",
    "mesh_get_index": "wp_mesh_get_index",
}

# iterator family: query-first signatures (upstream mesh.h:2327-2346 take the
# `mesh_query_aabb_t&` by reference — no id argument to expand to the table)
_MESH_QUERY_FIRST_CALLS = {
    "mesh_query_aabb_next",
    "mesh_query_sphere_next",
    "mesh_query_next",
    "mesh_query_next_dynamic",
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
    "wp::vec_t<5, wp::float32>": "make_vec5f",
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

# multi-output builtins: upstream codegen appends the SSA output vars as
# trailing call arguments (codegen.py forward call = func(fwd_args +
# output_list)), so the SSA statement has more args than the builtin's
# inputs and no return value. Each entry: number of leading input args,
# the output vars' MSL types in order, and the prelude out-param function.
# Statement-context only — a value-context tuple use still fails loud
# through the generic no-translation path.
_MULTI_OUT_BUILTINS = {
    "wp::quat_to_axis_angle": {
        "inputs": 1,
        "outputs": ("vec3f", "float"),
        "msl": "wp_quat_to_axis_angle",
    },
}

# zero-arg template constructors: wp::x<wp::float32>() → <msl>
_ZEROARG_TABLE = {
    "wp::quat_identity<wp::float32>": "wp_quat_identity()",
    "wp::transform_identity<wp::float32>": "wp_transform_identity()",
    # wp.identity(n, dtype) lowers to n-constant zero-arg factories (KEP-0001
    # M3.2 P1; upstream mat.h identity(): unit diagonal). Other sizes/dtypes
    # fail loud through the generic no-translation path below.
    "wp::identity<2, wp::float32>": "wp_identity2()",
    "wp::identity<3, wp::float32>": "wp_identity3()",
    "wp::identity<4, wp::float32>": "wp_identity4()",
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


def _tile_scalar_lit(elem: str, value) -> str:
    """Format a compile-time scalar as an MSL literal for a tile element type
    (float/half need the suffix; ints and bools are bare)."""
    if elem in ("float", "half"):
        return f"{float(value):.9g}{'f' if elem == 'float' else 'h'}"
    if elem == "bool":
        return "true" if value else "false"
    return str(int(value))


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


# ---------------------------------------------------------------------------
# tile variables (KEP-0001 M2.4): threadgroup-backed tile model
# ---------------------------------------------------------------------------

# Tile SSA decl (`wp::tile_register_t<T, wp::tile_layout_register_t<
# wp::tile_shape_t<D0,...>>> var = ...` / `wp::tile_shared_t<T,
# wp::tile_layout_strided_t<wp::tile_shape_t<D0,...>, wp::tile_stride_t<...>>,
# ALIGN> var = ...`).
_RE_TILE_DECL = re.compile(
    r"^\s*(?:const\s+)?wp::tile_(register|shared)_t<([A-Za-z0-9_:<>,\s]+)>\s+(\w+)\s*=\s*([^;]*);\s*$"
)
_RE_TILE_SHAPE = re.compile(r"wp::tile_shape_t<([\d,\s]+)>")

# Apple silicon exposes 32 KB of threadgroup memory per threadgroup
# (MTLDevice maxThreadgroupMemoryLength; the runtime never overrides it).
# Every shared tile array and reduction scratch array is statically
# allocated at kernel scope, so the total is known at translation time and
# oversubscription fails loud instead of faulting in the Metal compiler.
_TILE_SHARED_BUDGET = 32 * 1024


class TileVar:
    """Translation-time representation of an SSA tile variable.

    kind == "shr":    the tile lives in a kernel-scope ``threadgroup`` array
                      ``_wp_tile_shr_{slot}`` of ``size`` elements — every
                      ``tile_shared_t`` decl, every register decl whose shape
                      is not the per-thread one-element form, and every
                      reduction result (a shape-(1,) shared tile).
                      ``slot == -1`` marks a non-owning ALIAS decl (the
                      ``= nullptr`` form a ``wp::tile_transpose/reshape/view``
                      result is declared with): no storage of its own — the
                      producing assignment binds ``slot``/``base`` to the
                      source tile's array (upstream Owner=false shared view).
    kind == "scalar": the tile is the thread's own element value — the
                      ``wp.tile(v)`` broadcast / per-thread 1-element register
                      form, kept on the M1 per-thread-scalar lowering.

    ``strides`` are the per-dimension element strides into the backing array
    (row-major of ``dims`` for owner tiles; the swapped template strides for
    transpose aliases) and ``base`` the MSL element-offset expression of the
    view origin (``"0"`` for owners and transpose/reshape aliases, the
    offset-coordinate dot product for ``tile_view``). Storage address of an
    element is ``base + sum(coord_i * strides_i)``; a contiguous tile
    (base "0", row-major strides) keeps the identity linear mapping.
    """

    __slots__ = ("kind", "elem", "slot", "size", "dims", "strides", "base")

    def __init__(
        self,
        kind: str,
        elem: str,
        slot: int = -1,
        size: int = 0,
        dims: list[int] | None = None,
        strides: list[int] | None = None,
        base: str = "0",
    ):
        self.kind = kind
        self.elem = elem
        self.slot = slot
        self.size = size
        self.dims = dims or []
        self.strides = strides if strides is not None else _rowmajor_strides(self.dims)
        self.base = base

    def is_contig(self) -> bool:
        return self.base == "0" and self.strides == _rowmajor_strides(self.dims)

    def coord_addr(self, coords: list[str]) -> str:
        """Storage address expression for one element given per-dim coordinate
        MSL expressions (len == len(dims))."""
        parts = []
        if self.base != "0":
            parts.append(f"({self.base})")
        for c, s in zip(coords, self.strides):
            if s == 1:
                parts.append(f"({c})")
            elif s != 0:
                parts.append(f"({c}) * {s}")
        if not parts:
            return "0"
        return parts[0] if len(parts) == 1 else " + ".join(parts)

    def lin_addr(self, lin: str, prefix: str = "_wp_v") -> tuple[list[str], str]:
        """(coordinate-decl lines, storage address) for one element given its
        tile-linear index expression. Contiguous tiles map the linear index
        straight through; strided views unravel coordinates first."""
        if self.is_contig():
            return [], lin
        lines: list[str] = []
        coords: list[str] = []
        div = 1
        for d in self.dims[1:]:
            div *= d
        rem = lin
        for i, d in enumerate(self.dims):
            if i == len(self.dims) - 1:
                coords.append(rem)
            else:
                name = f"{prefix}{i}"
                # no trailing semicolon: every caller wraps the decl itself
                lines.append(f"int {name} = ({rem}) / {div}")
                coords.append(name)
                rem = f"({rem}) - {name} * {div}"
                div //= d if d else 1
        return lines, self.coord_addr(coords)


def _swap_line(elem: str, arr: str, i: str, j: str) -> str:
    """One-line compare-exchange of two array slots (any MSL value type)."""
    return f"{{ {elem} _wp_tmp = {arr}[{i}]; {arr}[{i}] = {arr}[{j}]; {arr}[{j}] = _wp_tmp; }}"


# tile_sort padding identity per key MSL type: the LARGEST value as an MSL
# literal (the Metal environment has no INT64_MAX/UINT64_MAX macros; INFINITY
# is available). Padded scratch slots sort past every real key and are
# dropped on copy-back.
_SORT_PAD = {
    "float": "INFINITY",
    "half": "INFINITY",
    "int": "2147483647",
    "uint": "4294967295u",
    "int64_t": "9223372036854775807ll",
    "uint64_t": "18446744073709551615ull",
}


class StackVar:
    """Translation-time representation of an SSA tile_stack variable.

    tier == "shr" (launch_tiled): the stack lives in three kernel-scope
    threadgroup arrays — ``_wp_tile_shr_{data_slot}`` (capacity elements of
    the payload type), ``_wp_tile_shr_{count_slot}`` (one int), and
    ``_wp_tile_shr_{stage_slot}`` (block_dim ints of push-rank staging).
    tier == "scalar" (plain launch): per-thread ``{name}_d[{capacity}]`` /
    ``{name}_c`` locals (one-lane block — the stack is the thread's own).
    """

    __slots__ = ("tier", "elem", "cap", "data_slot", "count_slot", "stage_slot", "name")

    def __init__(self, tier: str, elem: str, cap: int, name: str):
        self.tier = tier
        self.elem = elem
        self.cap = cap
        self.data_slot = -1
        self.count_slot = -1
        self.stage_slot = -1
        self.name = name


def _rowmajor_strides(dims: list[int]) -> list[int]:
    """Row-major element strides of ``dims`` (last dim contiguous)."""
    strides: list[int] = []
    acc = 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    return strides[::-1]


def _tile_shape_dims(layout_src: str) -> list[int]:
    dims_text = _RE_TILE_SHAPE.search(layout_src)
    if dims_text is None:
        raise TranslationError(
            f"tile layout {layout_src!r} carries no compile-time tile_shape_t (outside the closure)"
        )
    return [int(d) for d in dims_text.group(1).replace(" ", "").split(",") if d]


# vector element types: make-function, component count, float-vs-int flavor
_VEC_IDENT = {
    "vec2f": ("make_vec2f", 2, "f"),
    "vec3f": ("make_vec3f", 3, "f"),
    "vec4f": ("make_vec4f", 4, "f"),
    "vec5f": ("make_vec5f", 5, "f"),
    "spatial_vectorf": ("make_spatial_vectorf", 6, "f"),
    "vec2i": ("make_vec2i", 2, "i"),
    "vec3i": ("make_vec3i", 3, "i"),
    "vec4i": ("make_vec4i", 4, "i"),
}

_SCALAR_IDENT = {
    # (min identity, max identity) — ±INFINITY for floats so a tile whose
    # true extreme is infinite still reduces correctly
    "float": ("INFINITY", "-INFINITY"),
    "half": ("INFINITY", "-INFINITY"),
    "int": ("2147483647", "(-2147483647 - 1)"),
    "uint": ("4294967295u", "0u"),
    "int64_t": ("INT64_MAX", "INT64_MIN"),
    "uint64_t": ("UINT64_MAX", "0u"),
    "bool": ("true", "false"),
    "char": ("127", "(-127 - 1)"),
    "uchar": ("255u", "0u"),
    "short": ("32767", "(-32768)"),
    "ushort": ("65535u", "0u"),
}


def _reduce_identity_init(elem: str, combine: str) -> str:
    """Initializer for the neutral staging element of a tree-reduction fold.

    Threads that own no tile element (tile smaller than the block, or a
    partial threadgroup's absent lanes) stage the identity so the guarded
    log2 tree stays exact for every combine.
    """
    if combine in ("+", "|"):
        return "{}"
    scalar = _SCALAR_IDENT.get(elem)
    if scalar is not None:
        return scalar[0] if combine == "wp_min" else scalar[1]
    vec = _VEC_IDENT.get(elem)
    if vec is not None:
        make, n, flavor = vec
        comp = ("INFINITY" if flavor == "f" else "2147483647") if combine == "wp_min" else (
            "-INFINITY" if flavor == "f" else "(-2147483647 - 1)"
        )
        return f"{make}({', '.join([comp] * n)})"
    raise TranslationError(
        f"no neutral identity for a reduction over tile element {elem!r} "
        "(struct tiles are outside the closure)"
    )


class ArgSpec:
    """One kernel/function argument, classified for MSL emission."""

    def __init__(self, name: str, kind: str, ctype: str, array_fields: list[tuple[str, str]] | None = None):
        self.name = name  # launch-level arg name (from _wp_args->name)
        self.kind = kind  # 'array' | 'scalar' | 'struct'
        self.ctype = ctype  # warp-native C++ type string from the SSA text
        # for kind == 'struct': [(field, elem ctype)] array fields split out
        # into their own buffer bindings (not part of the MSL struct value)
        self.array_fields = array_fields or []
        # phi-materialized local binding (`a if c else b` over two arrays):
        # the (data, meta) pair lives in locals `{name}_data` / `{name}_m`,
        # not in the kernel's argument buffer
        self.local = False

    @property
    def msl_elem(self) -> str:
        return array_elem_msl(self.ctype)

    @property
    def msl(self) -> str:
        return msl_type(self.ctype)


class Param:
    """MSL parameter of a device function."""

    def __init__(
        self,
        kind: str,
        msl: str,
        name: str,
        array_fields: list[tuple[str, str]] | None = None,
        slot: int = -1,
    ):
        self.kind = kind  # 'value' | 'array' | 'ref' (thread T&) | 'struct' | 'tile_binding'
        self.msl = msl  # MSL type (elem type for arrays)
        self.name = name
        # for kind == 'struct': [(field, elem ctype)] split array fields
        # expanded into extra data+meta params right after the value param
        self.array_fields = array_fields or []
        # for kind == 'tile_binding': the caller-side threadgroup slot whose
        # ``_wp_tile_shr_{slot}`` array carries this tile-typed argument
        # (clone-per-site monomorphization of template<typename tile_...>
        # callees; fn_src skips the param — the tile travels through the
        # hidden shr-slot pointer, not a value param). On a template stub,
        # ``msl`` holds the template TYPE name instead of an MSL type.
        self.slot = slot


class MSLFunction:
    """A translated device function or kernel entry."""

    def __init__(self, name: str, ret_msl: str | None):
        self.name = name
        self.ret_msl = ret_msl  # None → void
        self.params: list[Param] = []
        self.lines: list[str] = []
        # template<typename tile_...> stub pending clone-per-site
        # instantiation (warp's C++ for tile-typed @wp.func params); the
        # concrete clones live under per-binding suffixed names
        self.is_template = False
        self.template_params: list[str] = []  # template param type names
        self.template_func = None  # the original warp func, for instantiation
        # tile-returning @wp.func: (elem, dims, slot) — the value travels in
        # the callee's claimed shared slot; callers rebind their result var
        # to it (the `= nullptr` alias mechanism one step wider)
        self.tile_return: tuple[str, list[int], int] | None = None
        # tile-using function: takes a hidden trailing ``int _wp_lane``
        # parameter plus one ``threadgroup elem* _wp_tile_shr_{k}`` per
        # reduction slot (fn.params stays SSA-aligned; fn_src/_call_user add
        # them). MSL forbids function-scope threadgroup storage in non-kernel
        # functions, so the kernel owns the arrays and passes pointers.
        self.needs_lane = False
        self.shr_slots: list[tuple[int, str]] = []
        # allocates wp::fixedarray_t stack arrays: takes a trailing
        # `device uchar* _wp_arena` parameter (the kernel passes its own
        # binding at every call site; kernel entry binds the arena buffer)
        self.needs_arena = False
        # reaches wp.mesh_get: takes a trailing `device wp_mesh*
        # _wp_mesh_table` parameter (same call-site forwarding pattern)
        self.needs_mesh = False


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
        # launch block size when one was supplied: wp.launch_tiled only
        # (default 256). Plain launches NEVER carry one — their tile ops
        # lower per-thread scalar (one-lane blocks, the CPU-golden
        # convention) regardless of any caller-supplied block_dim.
        # Parse-time tile shapes (vendored builtins read
        # codegen.options["block_dim"]), wp.block_dim() folding and the
        # threadgroup reduction lowering all key off it; the tid mapping
        # reads the bounds off _dims instead (host appends block_dim for
        # tiled launches), so it is routing-independent.
        self.block_dim: int | None = None
        # [(elem msl, op msl)] per wp::tile_reduce site, in body order —
        # signals that a barriered tree fold exists (drives _wp_live preamble
        # emission and the _gsize binding; the scratch arrays themselves are
        # claimed in shr_arrays like every other shared tile array)
        self.tile_reduces: list[tuple[str, str]] = []
        # True when any non-reduce lowering also needs the _wp_live partial
        # threadgroup guard (tile_stack push/pop rank walks): joins
        # tile_reduces in driving the _wp_live preamble definition and the
        # _gsize kernel binding
        self.uses_live = False
        # every kernel-scope threadgroup array in claim order (shared tile
        # vars, reduction scratch arrays, extract scratch slots):
        # [(elem msl, size)] — kernel_src emits one
        # ``threadgroup {elem} _wp_tile_shr_{k}[{size}]`` per entry and runs
        # the 32 KB threadgroup-memory budget check over the total
        self.shr_arrays: list[tuple[str, int]] = []
        # True when the unit reaches any tile construct (tile vars,
        # tile_reduce/sum/min/max, tile_zeros/extract/scatter, tile mesh
        # query stubs). Introspection only — plain wp.launch of a tile kernel
        # is accepted, matching upstream's degenerate tile semantics (the
        # per-thread lowering is exact when every threadgroup lane runs, e.g.
        # dim a multiple of block_dim; see tests/test_launch_routing.py).
        self.uses_tile_ops = False
        # a callee forwards _wp_live (its tile reduces need the fold guard):
        # the kernel preamble must define it (kernel_src)
        self.callee_needs_live = False
        # in-device wp.zeros() (wp::fixedarray_t) allocations: every site in
        # the unit (kernel + all @wp.func bodies) claims a unique byte offset
        # in the per-thread arena (`device uchar* _wp_arena`, bytes_per_thread
        # × threads). Sites get disjoint static offsets, so nested live
        # allocations (caller's buffer passed into a callee that allocates
        # its own) can never overlap — unique-offset over-allocates vs true
        # stack scoping but is always correct. kernel_src reads this to size
        # the arena binding.
        self.arena_next_off = 0
        self.arena_bytes_per_thread = 0
        # True when the kernel body or any transitively called @wp.func
        # reaches wp.mesh_get: the kernel binds `device wp_mesh*
        # _wp_mesh_table` (host-built mesh table; runtime/host.py) LAST
        self.needs_mesh = False

    def source(self) -> str:
        parts = [
            "// generated by warp-metal MSL translator (Adjoint SSA)\n",
            PRELUDE,
            "\n",
            # zero meta bound by dead-branch struct-array element calls (see
            # _call_user): never executed, keeps the reference param bindable
            "constant wp_arr_meta _wp_dead_meta_g = {};\n\n",
        ]
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
        idx += 1
        if self.arena_next_off > 0:
            # in-device wp.zeros() (fixedarray) allocations: one per-thread
            # arena slice. Every allocation site in the unit
            # (kernel or @wp.func) holds a unique static offset.
            self.arena_bytes_per_thread = (self.arena_next_off + 15) & ~15
            params.append(f"    device uchar* _wp_arena [[buffer({idx})]]")
            idx += 1
        if self.needs_mesh:
            # wp.mesh_get users read the host-built mesh table (last binding)
            params.append(f"    device wp_mesh* _wp_mesh_table [[buffer({idx})]]")
            idx += 1
        params.append("    uint3 _gid [[thread_position_in_grid]]")
        if self.tile_reduces or self.uses_live or self.callee_needs_live:
            # grid extent for the partial-threadgroup fold guard (_wp_live);
            # the guard can also live in a callee's body (JTDAJ_sparse's
            # hessian_entry template) — the binding must fire whenever the
            # preamble definition does
            params.append("    uint3 _gsize [[threads_per_grid]]")
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
        if "_wp_live" in body and not (
            self.block_dim is not None
            and (self.tile_reduces or self.uses_live or self.callee_needs_live)
        ):
            # static self-check (mirrors the one in fn_src): a kernel-body
            # fold or stack rank walk needs the preamble definition, which
            # fires exactly when block_dim is set and a _wp_live site exists
            raise TranslationError(
                f"kernel {self.kernel.name}: body uses _wp_live without the preamble "
                "definition (live-guarded tile op lowered without block_dim — translation bug)"
            )
        needs_lane = bool(self.shr_arrays or self.uses_tile_ops)
        if "_wp_lane" in body and not needs_lane:
            # static self-check: strided tile loops (load/store/map/zeros and
            # the fold staging) need the preamble lane index, which fires
            # exactly when a tile op exists (explicit-block_dim launches use
            # _idx % block_dim; plain launches pin the degenerate lane 0)
            raise TranslationError(
                f"kernel {self.kernel.name}: body uses _wp_lane without the preamble "
                "definition (tile op lowered without block_dim — translation bug)"
            )
        # 32 KB threadgroup-memory budget: every shared tile array is a
        # static kernel-scope allocation, so the total is exact at
        # translation time (see _TILE_SHARED_BUDGET)
        budget_bytes = 0
        for elem, size in self.shr_arrays:
            budget_bytes += _shr_elem_bytes(elem, self) * size
        if budget_bytes > _TILE_SHARED_BUDGET:
            raise TranslationError(
                f"kernel {self.kernel.name}: threadgroup tile storage budget exceeded — "
                f"{budget_bytes} bytes across {len(self.shr_arrays)} shared tile arrays > "
                f"{_TILE_SHARED_BUDGET} (maxThreadgroupMemoryLength on Apple silicon); "
                "shrink the tile shapes or split the kernel"
            )
        # tiled launch support: one threadgroup array per claimed slot
        # (function-scope threadgroup storage; sizes are compile-time). The
        # fold is guarded by _wp_live so partial threadgroups (plain wp.launch
        # with dim % block_dim != 0) never read uninitialized shared slots.
        preamble = "    const int _idx = (int)_gid.x;\n"
        if self.block_dim is not None:
            for k, (elem, size) in enumerate(self.shr_arrays):
                preamble += f"    threadgroup {elem} _wp_tile_shr_{k}[{size}];\n"
            if self.shr_arrays or self.uses_tile_ops:
                # scalar-carve-out tile ops (the wp.tile(v) per-thread form)
                # use the lane index without claiming a shared array
                preamble += f"    const int _wp_lane = _idx % {self.block_dim};\n"
            if self.tile_reduces or self.uses_live or self.callee_needs_live:
                preamble += f"    const int _wp_live = min((int)_gsize.x - (_idx - _wp_lane), {self.block_dim});\n"
        elif self.uses_tile_ops:
            # plain launch without block_dim: per-thread degenerate tile
            # mode — the thread is its own one-lane block (Newton's non-CUDA
            # block_dim=1 convention, the semantics every plain-launch CPU
            # golden was generated with). The lane index is constant 0, no
            # shared arrays and no barriers are emitted, so the kernel is
            # correct under ANY dispatch threadgroup size (including the
            # native auto choice) — scheduling cannot change its output.
            preamble += "    const int _wp_lane = 0;\n"
        defines_text = "\n".join(defines) + "\n" if defines else ""
        # brace-balance static self-check over the emitted kernel body (the
        # surrounding wrapper text is static templates): an unbalanced body
        # surfaces at Metal compile time as "extraneous closing brace" —
        # fail loud here instead, naming the kernel
        depth = 0
        for ch in body:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    raise TranslationError(
                        f"kernel {self.kernel.name}: extraneous closing brace in emitted "
                        "MSL body (brace imbalance — translation bug)"
                    )
        if depth != 0:
            raise TranslationError(
                f"kernel {self.kernel.name}: unbalanced braces in emitted MSL body "
                f"({depth} unclosed opens — translation bug)"
            )
        return (
            self._argbuf_struct(use_argbuf) + "\n\n"
            + f"struct Params_{self.kernel.name} {{\n" + "\n".join(fields) + "\n};\n\n"
            + defines_text
            + f"kernel void {self.kernel.name}(\n" + ",\n".join(params) + ")\n{\n"
            + preamble
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
        if p.kind == "tile_binding":
            # clone-per-site template binding: the tile travels through the
            # hidden `_wp_tile_shr_{slot}` trailing param (needs_lane), not a
            # value param — nothing to emit here
            continue
        if p.kind == "array":
            params.append(f"    device {p.msl}* {p.name}_data")
            params.append(f"    wp_arr_meta {p.name}_m")
        elif p.kind == "ref":
            params.append(f"    thread {p.msl}& {p.name}")
        elif p.kind == "struct" and p.array_fields:
            # split array fields expand into extra data+meta params here, at
            # emission — fn.params mirrors the C++ SSA parameter list so the
            # call-site argument zip stays aligned
            params.append(f"    {p.msl} {p.name}")
            for field, elem_ctype in p.array_fields:
                params.append(f"    device {msl_type(elem_ctype)}* {p.name}__{field}_data")
                params.append(f"    wp_arr_meta {p.name}__{field}_m")
        else:
            params.append(f"    {p.msl} {p.name}")
    body = "\n".join(fn.lines) or "    ;"
    ret = "void" if fn.ret_msl is None else fn.ret_msl
    # Hidden parameters go into the same list (order must mirror _call_user's
    # argument forwarding: lane/live/shr slots, then arena, then mesh table).
    # Never append to a params *string*: for a signatureless function (empty
    # fn.params) `s += ",\n int _wp_lane"` emitted a leading comma and tore
    # the parameter list loose at file scope.
    if fn.needs_lane:
        params.append("    int _wp_lane")
        params.append("    int _wp_live")
        for k, elem in fn.shr_slots:
            params.append(f"    threadgroup {elem}* _wp_tile_shr_{k}")
    if fn.needs_arena:
        params.append("    device uchar* _wp_arena")
    if fn.needs_mesh:
        params.append("    device wp_mesh* _wp_mesh_table")
    params_text = ",\n".join(params)
    # shape self-check: every parameter line is 4-space indented, so a comma
    # at the start of the list or of any line means a malformed signature
    if params_text.startswith(",") or "\n," in params_text:
        raise TranslationError(f"function {fn.name}: malformed parameter list (misplaced comma)")
    if "_wp_live" in body and "int _wp_live" not in params_text:
        # static self-check: every emission path that lowers _tile_reduce
        # (kernel body / @wp.func body / loops / transitive calls) must pair
        # each _wp_live use with a definition in scope — kernel preamble or
        # the hidden needs_lane parameter here
        raise TranslationError(
            f"function {fn.name}: body uses _wp_live but its signature does not carry it "
            "(tile_reduce lowered outside the needs_lane forwarding — translation bug)"
        )
    if "_wp_lane" in body and not fn.needs_lane:
        # plain-scalar tile ops inside a @wp.func: no caller forwards a lane
        # (the unit lowered per-thread degenerate), so provide the constant
        # zero lane locally — one-lane block per thread
        body = "    const int _wp_lane = 0;\n" + body
    text = f"static {ret} {fn.name}(\n{params_text})\n" + "{\n" + body + "\n}\n"
    # brace-balance static self-check: an unbalanced emission surfaces at
    # Metal compile time as "extraneous closing brace" — fail loud here
    # instead, naming the function
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise TranslationError(
                    f"function {fn.name}: extraneous closing brace in emitted MSL "
                    "(brace imbalance — translation bug)"
                )
    if depth != 0:
        raise TranslationError(
            f"function {fn.name}: unbalanced braces in emitted MSL ({depth} unclosed "
            "opens — translation bug)"
        )
    return text


# ---------------------------------------------------------------------------
# expression + statement translation
# ---------------------------------------------------------------------------

class _BodyTranslator:
    def __init__(self, unit: KernelUnit, array_args: dict[str, ArgSpec], user_funcs: dict[str, "MSLFunction"],
                 scope: str = "kernel", codegen=None):
        self.unit = unit
        self.scope = scope  # "kernel" (Params buffer) | "func" (meta params)
        # warp codegen handle — kept so _call_user can instantiate
        # template-typed callees (clone-per-site monomorphization) lazily at
        # the first call site
        self.codegen = codegen
        # array bindings visible in this scope, keyed by every SSA form that
        # can reference them: ``var_<arg>`` for top-level arrays,
        # ``var_<struct>.<field>`` for split struct array fields, and (added
        # by _alias) the ``var_N = &((var_x).field)`` alias vars themselves
        self.array_args = array_args
        self.user_funcs = user_funcs  # native name -> translation (param kinds)
        self.aliases: dict[str, tuple[bool, str]] = {}  # var -> (is_pointer, value expr)
        # declared var -> MSL type (from the SSA decl lines) — drives the
        # struct-comparison routing (`a == b` on struct values must rewrite
        # to wp_equal; MSL structs carry no operator==)
        self.var_types: dict[str, str] = {}
        # wp::slice_t values (warp codegen's `m[begin:end, j]` lowering):
        # name -> (begin, end, step) — pure metadata; the only consumer is
        # the sliced wp::extract (the rangefinder depth term's mat33[:, 2])
        self._slices: dict[str, tuple[int, int, int]] = {}
        # threadgroup slots this body uses, [(k, elem msl, combine)] — the
        # kernel preamble / function preamble declare each slot's shared
        # array in its own function scope
        self.used_shr: list[tuple[int, str, str]] = []
        # `wp::Any var_N;` decls (wp.static temps) — typed at first assignment
        self.deferred_any: set[str] = set()
        # all-int wp::tuple_t decls dropped by the shape-tuple heuristic —
        # the multi-arg wp::tuple ctor re-emits decl+init for these
        # (mjw _dof_chain_advance's genuine Tuple[int, int])
        self._tuple_decl_dropped: set[str] = set()
        # provable block-uniformity: SSA vars/ranges whose value every lane
        # of the block computes identically (constants, scalar params, range
        # loop indices over uniform bounds, arithmetic over those). Drives
        # the barrier-uniformity checkpoint's provable branch — tile ops
        # under an `if` whose condition is uniform cannot diverge (upstream
        # warp's own uniformity rule; mujoco's blocked cholesky guards
        # tile_assign with `end == matrix_size`). Conservative: unknown
        # provenance (array loads, tid-derived values, opaque calls) is
        # divergent.
        self.uniform_vars: set[str] = set()
        self.uniform_ranges: set[str] = set()
        self._tid_vars: set[str] = set()
        # uniformity scope snapshots: the SSA declares each scope's own var
        # numbers in a decl block (shadowing any outer number), while defs of
        # function-scope vars inside a branch persist past the close — so on
        # close only the scope's DECLARED names roll back to the snapshot
        self._uniform_scopes: list[tuple[frozenset[str], frozenset[str], set[str]]] = []
        # uniform-address load tracking: ptr var → array arg it addresses
        # (all indices provably block-uniform), and the array args that
        # receive a store anywhere in the kernel (those lose the privilege:
        # a lane could change the value between lanes' loads)
        self._uniform_addr: dict[str, str] = {}
        self._stored_arrays: set[str] = set()
        # tiled kernels: SSA tile var -> TileVar (threadgroup-backed shared
        # array or the per-thread one-element register form); plus
        # compile-time SSA constants (const var = literal) for tile_extract
        self.tile_vars: dict[str, TileVar] = {}
        self.const_vars: dict[str, str] = {}
        # SSA tile_stack vars (wp.tile_stack decls) -> StackVar: cooperative
        # block-wide LIFO stacks (launch_tiled: shared data/count/rank-stage
        # arrays) or the per-thread arena-tier degenerate (plain launches)
        self.stack_vars: dict[str, "StackVar"] = {}
        # SSA tile_mesh_query_aabb query vars -> (result slot, latch slot):
        # the shared handoff slots backing the cooperative traversal
        self.meshq_slots: dict[str, tuple[int, int]] = {}
        # unique suffix for bare kernel-scope index decls (register-tile
        # stores emit `int _wp_s{N}` OUTSIDE any loop scope — two stores in
        # one kernel must not redeclare the same name)
        self._store_uid = 0
        # lexically open braces, ["if" | "if-const" | "loop", ...]: cooperative
        # tile ops (barrier emitters) must sit in uniform control flow — a
        # reduce / scatter / shared-fill site lexically inside an `if` fails
        # loud unless the branch condition folds to a compile-time constant
        # (block-uniform by construction; see _check_uniform). Upstream relies
        # on documentation discipline; KEP-0001 M2.4 adds the translation-time
        # checkpoint.
        self._block_stack: list[str] = []
        # compile-time folds of SSA vars (wp.constant materialization chain):
        # name -> folded value, fed by _const_note, read by _const_eval
        self._const_vars: dict[str, object] = {}
        # SSA vars holding a loaded wp_mesh points stub (dead mesh branch —
        # see _alias / the wp::load branch)
        self.mesh_stub_vars: set[str] = set()
        # set when a dead-branch struct call bound "_wp_dead_meta": the
        # enclosing body then needs the zero-meta lvalue declared
        self.dead_meta_emitted = False
        # this body allocates wp::fixedarray_t stack arrays (needs the
        # `_wp_arena` parameter threaded through)
        self.used_arena = False
        # a callee needs the arena: this body must take the parameter too so
        # it can forward it (transitive propagation, like needs_lane)
        self.callee_needs_arena = False
        # a callee needs the lane + threadgroup arrays: merge its slots into
        # used_shr so this body's signature forwards them transitively
        self.callee_needs_lane = False
        # a callee forwards _wp_live (its tile reduces need the fold guard):
        # the kernel preamble must define it
        self.callee_needs_live = False
        # this body reads wp.mesh_get (needs the `_wp_mesh_table` binding)
        self.used_mesh = False
        # a callee reaches wp.mesh_get: forward the table binding transitively
        self.callee_needs_mesh = False
        # SSA vars holding a wp.Mesh value (mesh_get results and their value
        # copies): their `.points`/`.indices` members bind as real arrays
        self.mesh_vars: set[str] = set()
        # function-scope decls for local array bindings (fixedarray arena
        # slices, wp.array views, where-phi merges): the SSA declares every
        # var at function scope, but these materialize at their ASSIGN site —
        # often inside a branch — so the decls hoist to the top and sites
        # emit plain assignments (keeps downstream phis in scope)
        self.hoisted: list[str] = []

    def _meta_ref(self, spec: ArgSpec) -> str:
        """wp_arr_meta reference for an array arg in the current scope."""
        if getattr(spec, "local", False):
            return f"{spec.name}_m"
        mm = getattr(spec, "mesh_member", None)
        if mm is not None:
            # real wp.Mesh member array: meta lives inside the mesh struct
            return f"({mm[0]}.{mm[1]}.m)"
        if self.scope == "kernel":
            return f"_wp_args.{spec.name}_m"
        return f"{spec.name}_m"  # fn_src emits a by-value `wp_arr_meta {name}_m`

    def _data_expr(self, spec: ArgSpec) -> str:
        """Device data-pointer expression for an array binding."""
        mm = getattr(spec, "mesh_member", None)
        if mm is not None:
            return f"{mm[0]}.{mm[1]}.data"
        return f"{spec.name}_data"

    def _claim_shr(self, elem: str, size: int) -> int:
        """Claim a kernel-scope threadgroup array slot (unit-global numbering,
        shared by kernel body and all @wp.func bodies via the needs_lane
        forwarding). The kernel preamble declares the array; function bodies
        receive ``threadgroup elem*`` pointers. Struct elements require their
        MSL def to have been emitted first (translate_kernel pre-passes
        tile_stack payload structs), so the byte size is derivable."""
        _shr_elem_bytes(elem, self.unit)
        k = len(self.unit.shr_arrays)
        self.unit.shr_arrays.append((elem, size))
        self.used_shr.append((k, elem, "+"))
        return k

    def _require_block_dim(self, what: str) -> int:
        if self.unit.block_dim is None:
            raise TranslationError(
                f"{what} requires a launch with block_dim (wp.launch_tiled, or wp.launch "
                "with an explicit block_dim): the threadgroup block layout is undefined otherwise"
            )
        return self.unit.block_dim

    # -- compile-time constant folding (uniformity-checkpoint exemption) -----

    _CONST_BINOPS: tuple[tuple[str, object], ...] = (
        ("!=", lambda a, b: a != b),
        ("==", lambda a, b: a == b),
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        ("&&", lambda a, b: bool(a) and bool(b)),
        ("||", lambda a, b: bool(a) or bool(b)),
        (">", lambda a, b: a > b),
        ("<", lambda a, b: a < b),
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    )

    @staticmethod
    def _split_top_op(expr: str, op: str) -> list[str]:
        """Split expr at top-level occurrences of op (outside parens and
        brackets); returns the operand segments (len 1 when absent)."""
        segs: list[str] = []
        cur: list[str] = []
        depth = 0
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if depth == 0 and expr.startswith(op, i):
                segs.append("".join(cur))
                cur = []
                i += len(op)
                continue
            cur.append(ch)
            i += 1
        segs.append("".join(cur))
        return segs

    @staticmethod
    def _paren_atom(expr: str) -> bool:
        """True when expr is one paren group: the first `(`'s matching `)` is
        the final char (`(a != b)` yes, `(int)(x)` no)."""
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i == len(expr) - 1
        return False

    def _const_eval(self, expr: str):
        """Fold an SSA expression to a compile-time constant, or None.

        wp.constant(N) materializes as ``const t var = N;`` and derived tests
        as ``var = (const != const);`` — every leaf resolves, so the value is
        block-uniform. Any runtime leaf (array load, wp.tid, SSA var without
        a tracked fold) makes the whole expression None: the fold must never
        claim constancy it cannot prove."""
        e = expr.strip()
        if e in self._const_vars:
            return self._const_vars[e]
        if re.fullmatch(r"-?\d+", e):
            return int(e)
        if re.fullmatch(r"-?\d*\.\d+f?", e):
            return float(e.rstrip("f"))
        if e == "true":
            return True
        if e == "false":
            return False
        if e.startswith("(") and e.endswith(")") and self._paren_atom(e):
            return self._const_eval(e[1:-1])
        if e.startswith("!"):
            v = self._const_eval(e[1:])
            return None if v is None else not v
        for op, fn in self._CONST_BINOPS:
            segs = self._split_top_op(e, op)
            if len(segs) > 1:
                vals = [self._const_eval(s) for s in segs]
                if any(v is None for v in vals):
                    return None
                out = vals[0]
                for v in vals[1:]:
                    out = fn(out, v)
                return out
        return None

    def _const_note(self, stripped: str) -> None:
        """Track SSA vars whose value folds at compile time (the wp.constant
        materialization chain: const literal decls, literal folds of tracked
        vars, and plain copies of tracked vars)."""
        m = re.match(r"^const [\w:]+ (\w+) = (.*?);?$", stripped)
        if m is None:
            m = re.match(r"^(?:bool|int|float)\w* (\w+) = (.*?);?$", stripped)
        if m is None:
            m = re.match(r"^(\w+) = (.*?);?$", stripped)
        if m is None:
            return
        name, rhs = m.group(1), m.group(2)
        value = self._const_eval(rhs)
        if value is None or (name in self._const_vars and self._const_vars[name] != value):
            # not foldable, or an SSA reassignment that changes the value:
            # the var must not be trusted as constant
            self._const_vars.pop(name, None)
            return
        self._const_vars[name] = value

    def _is_uniform_expr(self, expr: str) -> bool:
        """True when every SSA identifier in expr is provably block-uniform.
        Qualified builtin names (``wp::add`` — the ``::`` makes the bare word
        its own token) are skipped; their arguments carry the classification.
        Anything not tracked as uniform — array loads, tid-derived values,
        opaque calls — classifies divergent."""
        # numeric literals first: their exponent/hex/suffix letters
        # (`9.5367432e-07`, `0x1F`, `2.0f`) would otherwise tokenize as
        # untracked identifiers and poison constant-only expressions
        expr = re.sub(r"\b\d[\w.]*", " ", expr)
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            tok = m.group(0)
            if expr[max(0, m.start() - 2) : m.start()] == "::":
                continue  # qualified name (wp::add, wp::int32 …)
            if tok.startswith("wp") or tok in ("true", "false"):
                continue
            if tok in self.uniform_vars or tok in self.uniform_ranges:
                continue
            return False
        return True

    def _note_uniformity(self, stripped: str) -> None:
        """Track provable block-uniformity from raw SSA assignment lines
        (before statement routing — the classification is SSA-level, not
        emission-level)."""
        # scope tracking: the SSA declares each scope's own var numbers in a
        # decl block (shadowing any outer number), while defs of
        # function-scope vars inside a branch persist past the close — so on
        # close only the scope's DECLARED names roll back to the snapshot
        m_lbl = _RE_LABEL.match(stripped)
        if _RE_IF_OPEN.match(stripped) or (m_lbl and m_lbl.group(1) == "start"):
            snap = (frozenset(self.uniform_vars), frozenset(self.uniform_ranges))
            self._uniform_scopes.append((snap[0], snap[1], set()))
            return
        if _RE_ELSE.match(stripped) or _RE_BRACE_CLOSE.match(stripped) or (
            m_lbl and m_lbl.group(1) == "end"
        ):
            if not self._uniform_scopes:
                return
            saved_vars, saved_ranges, decls = self._uniform_scopes[-1]
            # else: the `}` ends the then-scope and `{` opens the else-scope —
            # restore its decl'd names but keep the frame for the else close
            pop = not _RE_ELSE.match(stripped)
            for n in decls:
                if n in saved_vars:
                    self.uniform_vars.add(n)
                else:
                    self.uniform_vars.discard(n)
                if n in saved_ranges:
                    self.uniform_ranges.add(n)
                else:
                    self.uniform_ranges.discard(n)
            if pop:
                self._uniform_scopes.pop()
            return
        m = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
            r"\s*=\s*wp::tid\(\);",
            stripped,
        )
        m_b = re.match(r"^builtin_tid(\d)d\(([A-Za-z0-9_,\s]+)\)\s*;?$", stripped)
        m_ba = re.match(r"^(var_\w+)\s*=\s*builtin_tid(\d)d\(\)\s*;?$", stripped)
        if m or m_b or m_ba:
            if m_b:
                names = [t.strip() for t in m_b.group(2).split(",") if t.strip()]
            elif m_ba:
                names = [m_ba.group(1)]
            else:
                names = [t.strip() for t in m.group(1).split(",")]
            for t in names:
                self._tid_vars.add(t)
            # launch_tiled (plain launches never pass block_dim — see
            # _unit_for): the runtime tid helper hands out
            # (block coords ..., lane) — every coordinate except the LAST is
            # shared verbatim by every lane of the block, so the mujoco
            # `worldid, dof_block_id, tid = wp.tid()` heads classify
            # block-uniform and only the lane stays divergent. The 1-D
            # assign form is the degenerate tiled read of the BLOCK index
            # (wp_tid1: `_idx / block_dim`) — uniform by the same token.
            if self.unit.block_dim is not None and self.unit.block_dim > 1 and (
                len(names) > 1 or m_ba
            ):
                for t in names[:-1] if len(names) > 1 else names:
                    self.uniform_vars.add(t)
            return
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?);?$", stripped)
        if not m:
            # typed decl with init (`const wp::int32 var_k = 0;` /
            # `wp::vec_t<3, wp::float32> var_v = wp::vec_t<...>(...);`) — the
            # block_size/border_size constants feeding range bounds and the
            # linesearch's vec-component constants arrive this way; a missed
            # seed would poison the whole chain
            m = re.match(
                r"^(?:const\s+)?[A-Za-z_][A-Za-z0-9_:]*(?:<[^;=]*>)?"
                r"\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?);?$",
                stripped,
            )
        if not m:
            # bare typed decl (`wp::range_t var_73;` / `wp::float32* var_p;` /
            # `wp::vec_t<3, wp::float32> var_v;`) — a fresh var number whose
            # scope owns this name: register it with the open scope (the
            # close rolls its classification back) and invalidate any
            # classification from the block that used this number before
            m = re.match(
                r"^(?:const\s+)?[A-Za-z_][A-Za-z0-9_:]*(?:<[^;]*>)?\s*\**\s*"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$",
                stripped,
            )
            if m:
                if self._uniform_scopes:
                    self._uniform_scopes[-1][2].add(m.group(1))
                self.uniform_vars.discard(m.group(1))
                self.uniform_ranges.discard(m.group(1))
            return
        name, rhs = m.group(1), m.group(2).strip()
        # SSA var numbers are block-scoped and get REUSED across basic
        # blocks (var_73 is a range iterator in one loop, a tile load in the
        # next) — every reassignment must invalidate the previous
        # classification, not just potentially add a new one. Self-referencing
        # phis (`flag = flag && cond;`) read the pre-line value, so snapshot
        # the membership BEFORE the discard
        self_phi = name in self.uniform_vars
        self.uniform_vars.discard(name)
        self.uniform_ranges.discard(name)
        # uniform-address load: `var_p = wp::address(var_arr, i0, i1, ...)` —
        # every lane computing the SAME address reads the same bytes, so a
        # load through this pointer is block-uniform unless the array is
        # stored to within this kernel (a lane could change the value
        # between lanes' loads). This is what admits mujoco's parameter
        # loads (`opt_tolerance[worldid % shape]`) — worldid is a block
        # coordinate — while lane-indexed loads (`efc_D_in[worldid, tid]`)
        # stay divergent.
        m_ad = re.match(
            r"^(var_\w+)\s*=\s*wp::address\(var_(\w+)\s*(?:,\s*(.*?))?\)\s*;?$",
            stripped,
            re.DOTALL,
        )
        if m_ad:
            arr = "var_" + m_ad.group(2)
            idxs = split_args(m_ad.group(3)) if m_ad.group(3) else []
            # a reused SSA number re-addresses: the stale entry must not
            # survive a divergent re-address
            self._uniform_addr.pop(m_ad.group(1), None)
            if arr not in self._stored_arrays and all(self._is_uniform_expr(a) for a in idxs):
                self._uniform_addr[m_ad.group(1)] = arr
            return
        m_ld = re.match(r"^(var_\w+)\s*=\s*wp::load\((var_\w+)\)\s*;?$", stripped)
        if m_ld:
            addr = self._uniform_addr.get(m_ld.group(2))
            if addr is not None and addr not in self._stored_arrays:
                self.uniform_vars.add(m_ld.group(1))
            return
        # shape metadata pointer: `var_p = &(var_arr.shape);` — launch-bound
        # metadata is identical for every lane, so loads through it
        # (`var_n = wp::load(var_p)` → the dims vector feeding
        # `worldid % shape[0]`) are block-uniform
        m_sp = re.match(r"^(var_\w+)\s*=\s*&\((var_\w+)\.shape\)\s*;?$", stripped)
        if m_sp:
            self._uniform_addr[m_sp.group(1)] = "__shape__"
            return
        # loop-carried phi move: `wp::assign(var_dst, var_src);` (the 2-arg
        # scalar form — the 4-arg form is the tile element store) — the dst
        # takes the src's classification (SSA numbers are reused, so the
        # stale classification must be replaced, not kept)
        m_as = re.match(r"^wp::assign\(var_(\w+),\s*var_(\w+)\)\s*;?$", stripped)
        if m_as:
            dst = f"var_{m_as.group(1)}"
            src = f"var_{m_as.group(2)}"
            self.uniform_vars.discard(dst)
            if src in self.uniform_vars:
                self.uniform_vars.add(dst)
            return
        m_r = re.match(r"^wp::range\((.*)\)$", rhs, re.DOTALL)
        if m_r:
            if self._is_uniform_expr(m_r.group(1)):
                self.uniform_ranges.add(name)
            return
        # a block-reduce result is definitionally block-uniform: the emitted
        # fold leaves the value in the tile's own one-element scratch and
        # every lane reads the same broadcast (see _tile_reduce). The mjw
        # linesearch guards its second reduction with `if not ls_converged`
        # where ls_converged chains entirely through gauss_sum[0]/p0_sum[0].
        if re.match(r"^wp::tile_(?:reduce|sum|min|max|argmin|argmax)\(", rhs):
            self.uniform_vars.add(name)
            return
        # component extraction of a tracked-uniform source stays uniform
        # (tile_extract reads the reduce scratch; extract indexes a uniform
        # vector/tuple): same address — or an index-selected lane of the same
        # value — on every thread. The SOURCE must itself be tracked, never
        # assumed; extraction of a load-derived value stays divergent.
        m_e = re.match(r"^wp::(?:tile_)?extract\(([A-Za-z_][A-Za-z0-9_]*),\s*(.+)\)$", rhs, re.DOTALL)
        if m_e and m_e.group(1) in self.uniform_vars and self._is_uniform_expr(m_e.group(2)):
            self.uniform_vars.add(name)
            return
        # a user @wp.func call with every argument uniform is itself uniform:
        # warp funcs are pure functions of their arguments (a callee that
        # touches memory must take an array param, and array args are never
        # tracked uniform, so the call stays divergent above); native
        # snippets in the closure are void tile-syncs and are never assigned.
        # Runtime intrinsics (builtin_tidNd() — the thread index — and any
        # wp_ helper) are NOT pure functions of their (often empty) argument
        # list and never take this path
        m_u = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", rhs, re.DOTALL)
        if m_u and not m_u.group(1).startswith(("builtin_", "wp_")):
            args = split_args(m_u.group(2)) if m_u.group(2).strip() else []
            if all(self._is_uniform_expr(a) for a in args):
                self.uniform_vars.add(name)
                return
        # tile-typed values are never scalar-uniform (tile_load/view/store
        # results — the reduce/extract forms above returned already)
        if "wp::tile_" in rhs:
            return
        # data-dependent loads and opaque calls stay divergent (conservative)
        if "wp::address(" in rhs or "wp::load(" in rhs or rhs.startswith("*"):
            return
        # a range iterator's induction values advance identically on every
        # lane of the block when the range bounds are uniform — the mjw
        # blocked-cholesky loop indices (`end = k + block_size`) come from
        # exactly this form
        m_i = re.match(r"^wp::iter_next\(([A-Za-z_][A-Za-z0-9_]*)\)$", rhs)
        if m_i:
            if m_i.group(1) in self.uniform_ranges:
                self.uniform_vars.add(name)
            return
        # self-referencing phi (`flag = flag && cond;`): the rhs reads the
        # dst's OWN pre-line value, which was classified at its earlier def
        # — substitute it out before the generic check (the discard above
        # would otherwise poison every loop-carried uniform flag)
        gen_rhs = rhs
        if self_phi and re.search(rf"\b{name}\b", gen_rhs):
            gen_rhs = re.sub(rf"\b{name}\b", "1", gen_rhs)
        if self._is_uniform_expr(gen_rhs):
            self.uniform_vars.add(name)

    def _check_uniform(self, what: str) -> None:
        """Barrier-uniformity checkpoint: cooperative tile ops emit
        threadgroup barriers, which diverge if any lane takes a different
        control-flow path. Upstream documents the discipline; this rejects
        the lexically-nested case at translation time — except branches
        whose condition folds to a compile-time constant (wp.constant
        derives, e.g. steel's ``if COMPUTE_DIAG != 0:``): every lane folds
        the same value, so the branch is taken or skipped block-wide and
        cannot diverge. Single-lane threadgroups (Newton's non-CUDA
        block_dim=1 convention) cannot diverge — their barriers are vacuous
        — so the checkpoint does not apply there either."""
        if self.unit.block_dim == 1:
            return
        if "if" in self._block_stack:
            raise TranslationError(
                f"{what} sits inside non-uniform control flow (an `if` branch): the emitted "
                "threadgroup barriers would diverge across lanes. Hoist the call to uniform "
                "control flow or split the kernel (KEP-0001 M2.4 barrier-uniformity checkpoint)"
            )

    def _tvar(self, name: str, what: str) -> TileVar:
        tvar = self.tile_vars.get(name)
        if tvar is None:
            raise TranslationError(f"{what}: source {name!r} is not a tile variable")
        if tvar.kind == "shr" and tvar.slot < 0:
            raise TranslationError(
                f"{what}: source {name!r} is an unbound tile alias (no producing "
                "tile_transpose/reshape/view assignment reached it)"
            )
        return tvar

    # -- expressions ---------------------------------------------------------

    def expr(self, src: str) -> str:
        src = src.strip()
        if src in ("True", "False"):
            # Python bool constants reach expression positions through call
            # arguments (e.g. tile_scatter_masked's mask); MSL spells them
            # lowercase — a raw `True` compiles as an undeclared identifier
            return src.lower()
        tvar = self.tile_vars.get(src)
        if tvar is not None and tvar.kind == "shr":
            # a threadgroup-backed tile only exists as a kernel-scope array —
            # it is reachable through the tile builtins, never as a value
            raise TranslationError(
                f"tile variable {src!r} used outside a tile op (threadgroup-backed tiles "
                "are only reachable through tile builtins)"
            )
        if "builtin_block_dim()" in src:
            # wp.block_dim() folds to the launch's compile-time block size;
            # plain launches run one-lane blocks (the CPU-golden convention),
            # so the degenerate block size there is 1
            src = src.replace(
                "builtin_block_dim()",
                str(self.unit.block_dim if self.unit.block_dim is not None else 1),
            )
        m = re.match(r"^builtin_tid([1234])d\s*\((.*)\)$", src)
        if m:
            tid_src = self._builtin_tid(m.group(1), m.group(2))
            return tid_src
        if src in self.array_args:
            # array used as a value (phi merges of `if arr:` blocks, bool
            # contexts): upstream tests array_t's non-null data pointer;
            # warp-metal binds a real buffer even for None args, so read
            # the meta's ndim (0 only for a null binding)
            spec = self.array_args[src]
            return f"({self._meta_ref(spec)}.ndim != 0)"
        m_ptr = re.match(r"^(\w+)\.ptr$", src)
        if m_ptr and m_ptr.group(1) in self.array_args:
            # .ptr on an array binding: the raw device address. Local stack
            # arrays / views carry it in `{name}_data`; bound kernel arrays
            # have the same shape (their binding is the buffer pointer), and
            # ref-copy aliases share the original spec's name (same address).
            spec = self.array_args[m_ptr.group(1)]
            return f"((uint64_t)((device char*){self._data_expr(spec)}))"
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
        # struct equality: the vendored codegen lowers `a == b` on struct
        # values to infix `==` (vec.h/quat.h operator== — all components
        # equal, bool result; spatial_vector_t is vec_t<6>), but MSL structs
        # carry no operator==, so route through the prelude wp_equal when
        # either operand is struct-typed (mjw _apply_ft's
        # `ft_body == wp.spatial_vector()` zero test). Upstream defines no
        # struct operator!= — the same form fails to compile upstream too,
        # so `!=` on struct operands fails loud.
        e_cmp = src
        while self._paren_atom(e_cmp):
            e_cmp = e_cmp[1:-1].strip()
        for cmp_op in ("==", "!="):
            cmp_parts = self._split_top_op(e_cmp, cmp_op)
            if len(cmp_parts) != 2:
                continue
            t_left = self.var_types.get(cmp_parts[0].strip())
            t_right = self.var_types.get(cmp_parts[1].strip())
            if _RE_STRUCT_T.match(t_left or "") or _RE_STRUCT_T.match(t_right or ""):
                if cmp_op == "!=":
                    raise TranslationError(
                        "struct '!=' has no upstream native definition (vec.h/quat.h "
                        "define operator== only) — rewrite as not(wp_equal(a, b))"
                    )
                return f"wp_equal({self.expr(cmp_parts[0])}, {self.expr(cmp_parts[1])})"
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
        """Translate ``builtin_tidNd`` to the upstream grid-coordinate form.

        Upstream semantics (context.py ``launch`` / ``launch_tiled`` +
        builtin.h ``tid()``): ``_idx`` is the flat THREAD index; ``wp.tid()``
        returns coordinates of the row-major unravel over the launch bounds
        (last dim fastest), with threads beyond the coordinate folded into
        one bucket (``coord_mult``). ``_wp_args._dims`` carries the bounds —
        the user dim for a plain launch, grid + ``[block_dim]`` for a tiled
        launch — so a shared runtime helper covers every routing:
          1-D plain: the flat index; N-D plain: leading coordinate
          (dim=(a,b): ``_idx / b``);
          tiled scalar: the BLOCK index (``_idx / block_dim``);
          tiled 2-D: the (block, lane) pair.
        """
        args = split_args(arg_src) if arg_src.strip() else []
        if n == "1":
            if args:
                raise TranslationError("builtin_tid1d takes no args")
            return "wp_tid1(_idx, _wp_args._dims)"
        if len(args) != int(n):
            raise TranslationError(f"builtin_tid{n}d takes {n} out args, got {len(args)}")
        out = ", ".join(a.strip() for a in args)
        return f"wp_tid{n}({out}, _idx, _wp_args._dims)"

    def _tile_reduce(self, target: str, rhs: str) -> list[str]:
        """Lower a tile reduction site (``tile_reduce(op, var)`` /
        ``tile_sum/min/max(var)`` / ``tile_argmin/argmax(var)``) to a
        barriered threadgroup tree over a kernel-scope scratch array.

        Every thread in the block participates; a log2(block_dim) barriered
        pairwise fold leaves the result at scratch[0]. Requires uniform
        control flow around the call (checked) and a power-of-two block_dim
        (validated at launch). The fold is guarded by
        ``_wp_lane + _wp_s < _wp_live`` so partial threadgroups never read
        unwritten slots. Sources:

        - threadgroup-backed tile: strided per-thread accumulate over the
          shared array, first-element seeded — threads that own no element
          (tile smaller than the block) stage the neutral identity so the
          tree stays exact for every combine;
        - per-thread scalar broadcast tile: the M1 gather path (each lane
          stages its own value; its linear index is the lane, which is what
          the argmin/argmax index tree stages).

        argmin/argmax run an int index tree alongside the value tree:
        comparison key = value, ties resolve to the EARLIER linear index —
        a deterministic refinement of upstream's "unspecified" tie order
        (upstream tile_reduce.h ValueAndIndex). The result (for argmin the
        INDEX, upstream returns a shape-(1,) int32 tile) is uniform across
        the block and lands in the target tile's own one-element shared
        array; every ``tile_extract(result, 0)`` is a direct read.
        """
        open_paren = rhs.index("(")
        close = _find_balanced(rhs, open_paren)
        parts = split_args(rhs[open_paren + 1 : close])
        # convenience forms carry the op in the name: tile_sum/min/max/argmin/
        # argmax(var); tile_reduce takes the operator as its first argument
        m_conv = re.match(r"^wp::tile_(sum|min|max|argmin|argmax)\(", rhs)
        arg_op: str | None = None
        if m_conv:
            if len(parts) != 1:
                raise TranslationError(f"{rhs.split('(')[0]} with {len(parts)} args (axis form is outside the closure)")
            src = parts[0].strip()
            op_name = m_conv.group(1)
            if op_name in ("argmin", "argmax"):
                arg_op = op_name
                combine = "wp_min" if op_name == "argmin" else "wp_max"
            else:
                combine = {"sum": "+", "min": "wp_min", "max": "wp_max"}[op_name]
        else:
            if len(parts) != 2:
                raise TranslationError(f"tile_reduce with {len(parts)} args (axis form is outside the closure)")
            op, src = parts[0].strip(), parts[1].strip()
            op_msl = _CALL_TABLE.get(op[4:] if op.startswith("wp::") else op)
            if op_msl not in ("wp_min", "wp_max", "wp_bit_or", "wp_add"):
                raise TranslationError(
                    f"tile_reduce operator {op!r} is outside the closure (add/min/max/bit_or only)"
                )
            combine = {"wp_min": "wp_min", "wp_max": "wp_max", "wp_bit_or": "|", "wp_add": "+"}[op_msl]
        tvar = self._tvar(src, "tile_reduce")
        elem = tvar.elem
        if arg_op and elem not in _SCALAR_IDENT:
            raise TranslationError(
                f"tile_{arg_op} over tile element {elem!r}: only scalar tiles are in the closure"
            )
        if self.unit.block_dim is None:
            # plain launch without block_dim: per-thread degenerate lowering.
            # The tile is the thread's own element (one-lane block), so the
            # reduction is the identity and argmin/argmax resolve to the
            # tile's only index (0) — exactly the plain-launch CPU semantics.
            # No shared scratch, no barrier, no tree: the result never
            # depends on the dispatch threadgroup layout.
            self.unit.uses_tile_ops = True
            self._tvar(target, "tile_reduce")  # validate the target tile
            if arg_op:
                return [f"{target} = 0;"]
            return [f"{target} = {src};"]
        bd = self._require_block_dim(rhs.split("(")[0])
        self._check_uniform(rhs.split("(")[0])
        self.unit.uses_tile_ops = True
        self.unit.tile_reduces.append((elem, combine))  # a fold exists (_wp_live preamble)
        j = self._claim_shr(elem, bd)  # value scratch
        if arg_op:
            j_idx = self._claim_shr("int", bd)  # index scratch (int32, upstream's tile dtype)
        lines: list[str] = []
        if tvar.kind == "shr":
            # strided per-thread accumulate over the shared array,
            # first-element seeded; threads that own no element stage the
            # neutral identity so the tree stays exact for every combine.
            # Reads go through the tile's linear->storage mapping (a
            # transpose/view source reorders coordinates).
            arr = f"_wp_tile_shr_{tvar.slot}"
            r_lines, r_addr = tvar.lin_addr("_wp_t", prefix="_wp_r")
            lines += [
                f"{elem} _wp_ident_{j} = {_reduce_identity_init(elem, combine)};",
                f"{elem} _wp_acc_{j} = _wp_ident_{j};",
                f"bool _wp_has_{j} = false;",
                f"int _wp_first_{j} = 0;",
                f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
                *(f"    {ln};" for ln in r_lines),
            ]
            stage = f"    if (!_wp_has_{j}) {{ _wp_acc_{j} = {arr}[{r_addr}]; _wp_has_{j} = true; _wp_first_{j} = _wp_t; }}"
            if arg_op:
                # argmin/argmax: the staged index must track the improving
                # extreme (not just the lane's first element), or a lane whose
                # minimum occurs at a later element biases the global tie
                # toward a wrong earlier index
                cmp = "<" if arg_op == "argmin" else ">"
                lines.append(stage)
                lines.append(
                    f"    else {{ if ({arr}[{r_addr}] {cmp} _wp_acc_{j}) {{ _wp_acc_{j} = {arr}[{r_addr}]; _wp_first_{j} = _wp_t; }} }}"
                )
            elif combine == "+":
                lines.append(stage)
                if elem in _SCALAR_IDENT:
                    lines.append(f"    else {{ _wp_acc_{j} = _wp_acc_{j} + {arr}[{r_addr}]; }}")
                else:
                    # struct elements have no operator+ on threadgroup
                    # lvalues — route through the same prelude helper the
                    # builtin mat add lowers to (componentwise wp_add)
                    lines.append(f"    else {{ _wp_acc_{j} = wp_add(_wp_acc_{j}, {arr}[{r_addr}]); }}")
            elif combine == "|":
                lines.append(stage)
                lines.append(f"    else {{ _wp_acc_{j} = _wp_acc_{j} | {arr}[{r_addr}]; }}")
            else:
                lines.append(stage)
                lines.append(f"    else {{ _wp_acc_{j} = {combine}(_wp_acc_{j}, {arr}[{r_addr}]); }}")
            lines += [
                "}",
            ]
            if arg_op:
                # BOTH trees stage: the value scratch feeds the comparison
                # tree, the index scratch the tie-break — staging only the
                # index leaves the value tree reading uninitialized
                # threadgroup memory (the argmin/argmax elem is scalar —
                # non-scalar was rejected above — so the ternary is exact)
                lines.append(f"_wp_tile_shr_{j}[_wp_lane] = _wp_has_{j} ? _wp_acc_{j} : _wp_ident_{j};")
                lines.append(f"_wp_tile_shr_{j_idx}[_wp_lane] = _wp_has_{j} ? _wp_first_{j} : 2147483647;")
            elif elem in _SCALAR_IDENT:
                lines.append(f"_wp_tile_shr_{j}[_wp_lane] = _wp_has_{j} ? _wp_acc_{j} : _wp_ident_{j};")
            else:
                # the ternary operator has no struct form in MSL — branch
                lines += [
                    f"if (_wp_has_{j}) {{ _wp_tile_shr_{j}[_wp_lane] = _wp_acc_{j}; }}",
                    f"else {{ _wp_tile_shr_{j}[_wp_lane] = _wp_ident_{j}; }}",
                ]
        else:
            # scalar broadcast tile: each lane stages its own element
            lines += [
                f"_wp_tile_shr_{j}[_wp_lane] = {src};",
            ]
            if arg_op:
                lines.append(f"_wp_tile_shr_{j_idx}[_wp_lane] = _wp_lane;")
        lines.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        if arg_op:
            # two-array tree: value is the comparison key, ties take the
            # earlier linear index
            if arg_op == "argmin":
                take = (
                    f"(_wp_tile_shr_{j}[_wp_lane + _wp_s] < _wp_tile_shr_{j}[_wp_lane]"
                    f" || (_wp_tile_shr_{j}[_wp_lane + _wp_s] == _wp_tile_shr_{j}[_wp_lane]"
                    f" && _wp_tile_shr_{j_idx}[_wp_lane + _wp_s] < _wp_tile_shr_{j_idx}[_wp_lane]))"
                )
            else:
                take = (
                    f"(_wp_tile_shr_{j}[_wp_lane + _wp_s] > _wp_tile_shr_{j}[_wp_lane]"
                    f" || (_wp_tile_shr_{j}[_wp_lane + _wp_s] == _wp_tile_shr_{j}[_wp_lane]"
                    f" && _wp_tile_shr_{j_idx}[_wp_lane + _wp_s] < _wp_tile_shr_{j_idx}[_wp_lane]))"
                )
            lines += [
                f"for (int _wp_s = {bd} >> 1; _wp_s > 0; _wp_s >>= 1) {{",
                "    if (_wp_lane < _wp_s && _wp_lane + _wp_s < _wp_live) {",
                f"        if ({take}) {{",
                f"            _wp_tile_shr_{j}[_wp_lane] = _wp_tile_shr_{j}[_wp_lane + _wp_s];",
                f"            _wp_tile_shr_{j_idx}[_wp_lane] = _wp_tile_shr_{j_idx}[_wp_lane + _wp_s];",
                "        }",
                "    }",
                "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                "}",
            ]
        else:
            shr_lhs = f"_wp_tile_shr_{j}[_wp_lane]"
            shr_rhs = f"_wp_tile_shr_{j}[_wp_lane + _wp_s]"
            if combine == "+" and elem not in _SCALAR_IDENT:
                # struct elements: no operator+ on threadgroup lvalues — the
                # prelude's componentwise helper (same form the builtin mat
                # add lowers to)
                combined = f"wp_add({shr_lhs}, {shr_rhs})"
            elif combine in ("+", "|"):
                combined = f"({shr_lhs} {combine} {shr_rhs})"
            else:
                combined = f"{combine}({shr_lhs}, {shr_rhs})"
            lines += [
                f"for (int _wp_s = {bd} >> 1; _wp_s > 0; _wp_s >>= 1) {{",
                "    if (_wp_lane < _wp_s && _wp_lane + _wp_s < _wp_live) {",
                f"        {shr_lhs} = {combined};",
                "    }",
                "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                "}",
            ]
        # the reduction lands in the TARGET tile's own one-element array (the
        # shape-(1,) result var), never in the source tile — writing the
        # source slot would clobber its element 0 and leave the result
        # uninitialized. argmin/argmax publish the INDEX (j_idx), not the
        # folded value. Every lane writes the same uniform value, so the
        # subsequent extract sees a consistent value without another barrier.
        tgt = self._tvar(target, "tile_reduce")
        res_val = f"_wp_tile_shr_{j_idx}[0]" if arg_op else f"_wp_tile_shr_{j}[0]"
        if tgt.kind == "shr":
            lines.append(f"_wp_tile_shr_{tgt.slot}[0] = {res_val};")
        else:
            lines.append(f"{target} = {res_val};")
        return lines

    # -- tile load/store/map (KEP-0001 M2.4 P2) ------------------------------
    #
    # Every shared-storage tile is a full-shape threadgroup array read and
    # written with per-thread strided passes (lane, lane+BD, ...) over the
    # linear row-major tile index — a simplified version of upstream's
    # three-channel (global -> shared -> register) tile.h:2115-2139 lowering:
    # global <-> shared directly, register-ization deferred (correctness
    # first). Bounds handling mirrors upstream tile.h: tile_load zero-fills
    # coordinates at or past the source extent (upper bound only — negatives
    # wrap through wp_address like every array access), tile_store silently
    # drops them, both selected by the compile-time bounds_check template.

    def _elem_addr(self, spec: ArgSpec, idx_srcs: list[str]) -> str:
        """Element pointer through wp_address (data + meta byte math). Shared
        by the tile load/store passes and the inline ``&(wp::index(...)
        .component_ref(k))`` store target (chained subscript stores)."""
        meta = self._meta_ref(spec)
        return f"wp_address({self._data_expr(spec)}, {meta}, {', '.join(idx_srcs)})"

    # Historical alias (tile-only era); kept so external callers don't break.
    _tile_addr = _elem_addr

    def _parse_tile_io(self, rhs: str, what: str, with_shape: bool) -> tuple[str, bool, list[int], list[str]]:
        """(elem, bounds_check, dims, call args) behind a tile_load/tile_store
        template call. tile_load carries the shape in its template
        (``<elem, bc, aligned, D0, ...>``); tile_store's template is only
        ``<elem, bc, aligned>`` and the shape comes from the source tile var."""
        open_p = rhs.index("(")
        m_t = re.match(r"^wp::tile_(load|store)<(.+)>$", rhs[:open_p].strip())
        if m_t is None:
            raise TranslationError(f"cannot parse {what} template {rhs!r}")
        targs = split_args(m_t.group(2))
        n_template = 3 + (1 if with_shape else 0)
        if len(targs) < n_template:
            raise TranslationError(f"{what} template {rhs!r} is malformed")
        elem = msl_type(targs[0])
        bc = targs[1].strip() == "true"
        dims = [int(a.strip()) for a in targs[3:]] if with_shape else []
        if with_shape and (not dims or len(dims) > 2):
            raise TranslationError(
                f"{what} with shape {dims}: only 1-D/2-D tiles are in the closure (KEP-0001 M2.4)"
            )
        close = _find_balanced(rhs, open_p)
        cargs = split_args(rhs[open_p + 1 : close])
        if with_shape and len(cargs) != 1 + len(dims):
            raise TranslationError(f"{what} needs one offset per array dimension, got {len(cargs) - 1}")
        return elem, bc, dims, cargs

    def _tile_io_spec(self, arg: str, what: str) -> ArgSpec:
        spec = self._array_arg_spec(arg)
        if spec is None:
            raise TranslationError(f"{what} source {arg!r} is not an array binding")
        return spec

    def _tile_load(self, target: str, rhs: str) -> list[str]:
        elem, bc, dims, cargs = self._parse_tile_io(rhs, "tile_load", with_shape=True)
        spec = self._tile_io_spec(cargs[0], "tile_load")
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_load target {target!r} is not a tile variable")
        if tvar.elem != elem:
            raise TranslationError(f"tile_load elem {elem} does not match target tile {tvar.elem}")
        self.unit.uses_tile_ops = True
        bd = None
        if tvar.kind == "shr":
            # shared storage only exists with a defined block layout; the
            # per-thread scalar form (plain launches and wp.tile(v)
            # broadcasts) needs no block geometry at all
            bd = self._require_block_dim("wp::tile_load")
            self._check_uniform("wp::tile_load into a shared tile")
        meta = self._meta_ref(spec)
        offs = [self.expr(a) for a in cargs[1:]]
        n = tvar.size
        # bounds_check=False emits no guard and never reads the zero scratch:
        # only open the scratch block when the bounds checks will reference it
        # (bounds_check=False loads otherwise leave an unused-variable warning
        # per site)
        lines = [f"{{ {elem} _wp_z = {{}};"] if bc else []
        # bounds_check=True: zero-fill coordinates at or past the upper
        # extent; bounds_check=False: unconditional access (the caller
        # guarantees in-bounds, matching upstream)
        def _load_if(lhs: str, addr: str, s_checks: list[str]) -> str:
            if bc:
                ok = " && ".join(s_checks)
                return f"if ({ok}) {{ {lhs} = *{addr}; }} else {{ {lhs} = _wp_z; }}"
            return f"{lhs} = *{addr};"

        if tvar.kind == "shr":
            arr = f"_wp_tile_shr_{tvar.slot}"
            if len(dims) == 1:
                lines += [
                    f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
                    f"    int _wp_s0 = ({offs[0]}) + _wp_t;",
                    f"    {_load_if(f'{arr}[_wp_t]', self._tile_addr(spec, ['_wp_s0']), [f'_wp_s0 < {meta}.shape[0]'])}",
                    "}",
                ]
            else:
                cols = dims[1]
                lines += [
                    f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
                    f"    int _wp_i0 = _wp_t / {cols};",
                    f"    int _wp_i1 = _wp_t - _wp_i0 * {cols};",
                    f"    int _wp_s0 = ({offs[0]}) + _wp_i0;",
                    f"    int _wp_s1 = ({offs[1]}) + _wp_i1;",
                    "    " + _load_if(
                        f"{arr}[_wp_t]",
                        self._tile_addr(spec, ["_wp_s0", "_wp_s1"]),
                        [f"_wp_s0 < {meta}.shape[0]", f"_wp_s1 < {meta}.shape[1]"],
                    ),
                    "}",
                ]
            tail = ["threadgroup_barrier(mem_flags::mem_threadgroup);"]
            if bc:
                tail.append("}")
            lines += tail
        else:
            # per-thread one-element register form: this thread owns linear
            # element _wp_lane
            if len(dims) != 1:
                raise TranslationError("tile_load into a per-thread tile is 1-D only in the closure")
            lines += [
                f"int _wp_s0 = ({offs[0]}) + _wp_lane;",
                "    " + _load_if(target, self._tile_addr(spec, ["_wp_s0"]), [f"_wp_s0 < {meta}.shape[0]"]),
            ]
            if bc:
                lines.append("}")
        return lines

    def _tile_load_indexed(self, target: str, rhs: str) -> list[str]:
        """``var = wp::tile_load_indexed<D0...>(src, idx_tile, axis, offsets...);``
        — tile.h:2963 gather along one axis through a 1-D index tile:
        element ``c`` reads ``src[offset + idx[c[axis]]]`` with BOTH bounds
        checked — a negative or past-the-end index (the mjw io.py:1224
        structurally-absent sentinel ``elemid == nC``) reads ``T{}`` zero.
        Upstream always stages the indices tile in shared storage; the
        closure form is 1-D sources, axis 0."""
        open_p = rhs.index("(")
        m_t = re.match(r"^wp::tile_load_indexed<(.+)>$", rhs[:open_p].strip())
        if m_t is None:
            raise TranslationError(f"cannot parse tile_load_indexed template {rhs!r}")
        dims = [int(a.strip()) for a in split_args(m_t.group(1))]
        if len(dims) != 1:
            raise TranslationError(f"tile_load_indexed over {len(dims)} dims: 1-D sources only in the closure")
        cargs = split_args(rhs[open_p + 1 : _find_balanced(rhs, open_p)])
        if len(cargs) < 3:
            raise TranslationError(f"tile_load_indexed needs (src, indices, axis[, offset]): {rhs!r}")
        spec = self._tile_io_spec(cargs[0], "tile_load_indexed")
        try:
            elem = spec.msl_elem
        except TranslationError as e:
            raise TranslationError(f"tile_load_indexed source element type unavailable: {e}") from e
        idx_name = cargs[1].strip()
        idx_tvar = self.tile_vars.get(idx_name)
        if idx_tvar is None:
            raise TranslationError(f"tile_load_indexed indices operand {idx_name!r} is not a tile variable")
        if idx_tvar.elem != "int":
            raise TranslationError(f"tile_load_indexed indices tile must be int, got {idx_tvar.elem}")
        axis = self._const_eval(cargs[2])
        if axis != 0:
            raise TranslationError(f"tile_load_indexed axis {axis}: only axis 0 is in the closure")
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_load_indexed target {target!r} is not a tile variable")
        if tvar.elem != elem:
            raise TranslationError(f"tile_load_indexed elem {elem} does not match target tile {tvar.elem}")
        offs = [self.expr(a) for a in cargs[3:]] or ["0"]
        meta = self._meta_ref(spec)
        self.unit.uses_tile_ops = True
        if tvar.kind == "shr":
            bd = self._require_block_dim("wp::tile_load_indexed")
            self._check_uniform("wp::tile_load_indexed")
            if idx_tvar.kind != "shr" or idx_tvar.slot < 0:
                raise TranslationError(
                    f"tile_load_indexed indices tile {idx_name!r} must be bound shared storage "
                    "(upstream promotes register index tiles)"
                )
            if tvar.size != dims[0]:
                raise TranslationError(f"tile_load_indexed template size {dims[0]} != target tile {tvar.size}")
            arr = f"_wp_tile_shr_{tvar.slot}"
            idx_arr = f"_wp_tile_shr_{idx_tvar.slot}"
            return [
                f"{{ {elem} _wp_z = {{}};",
                f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
                f"    int _wp_li = {idx_arr}[_wp_t];",
                f"    int _wp_s0 = ({offs[0]}) + _wp_li;",
                f"    if (_wp_s0 >= 0 && _wp_s0 < {meta}.shape[0]) {{ {arr}[_wp_t] = *{self._tile_addr(spec, ['_wp_s0'])}; }} else {{ {arr}[_wp_t] = _wp_z; }}",
                "}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
                "}",
            ]
        # per-thread one-element form: the thread's element gathers through
        # its own index-tile element (plain-launch one-lane degradation)
        if idx_tvar.kind != "scalar":
            raise TranslationError(
                f"tile_load_indexed: per-thread target with shared indices tile {idx_name!r} is outside the closure"
            )
        s = f"({offs[0]}) + {idx_name}"
        return [
            f"{{ {elem} _wp_z = {{}};",
            f"if ({s} >= 0 && {s} < {meta}.shape[0]) {{ {target} = *{self._tile_addr(spec, [s])}; }} else {{ {target} = _wp_z; }}",
            "}",
        ]

    def _tile_store(self, rhs: str) -> list[str]:
        elem, bc, _tpl_dims, cargs = self._parse_tile_io(rhs, "tile_store", with_shape=False)
        # the template arg arrives as a Python bool; MSL spells it lowercase
        # (a raw `True` interpolates as an undeclared identifier)
        bc_msl = "true" if bc else "false"
        if len(cargs) < 2:
            raise TranslationError(f"tile_store needs (array, offsets..., source): {rhs!r}")
        spec = self._tile_io_spec(cargs[0], "tile_store")
        src_name = cargs[-1].strip()
        tvar = self.tile_vars.get(src_name)
        if tvar is None:
            raise TranslationError(f"tile_store source {src_name!r} is not a tile variable")
        if tvar.elem != elem:
            raise TranslationError(f"tile_store elem {elem} does not match source tile {tvar.elem}")
        self.unit.uses_tile_ops = True
        meta = self._meta_ref(spec)
        offs = [self.expr(a) for a in cargs[1:-1]]
        nd = len(offs)
        if nd > 2:
            raise TranslationError(f"tile_store with {nd} offsets: only 1-D/2-D stores are in the closure")
        # no barrier: every thread writes only global-memory coordinates
        # derived from its own strided slots (no cross-lane visibility need)
        if tvar.kind == "shr":
            if tvar.slot < 0:
                raise TranslationError(
                    f"tile_store source {src_name!r} is an unbound tile alias (no producing "
                    "tile_transpose/reshape/view assignment reached it)"
                )
            bd = self._require_block_dim("wp::tile_store")
            arr = f"_wp_tile_shr_{tvar.slot}"
            n = tvar.size
            if nd == 1:
                s_lines, s_addr = tvar.lin_addr("_wp_t", prefix="_wp_st")
                return [
                    f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
                    *(f"    {ln};" for ln in s_lines),
                    f"    int _wp_s0 = ({offs[0]}) + _wp_t;",
                    (
                        f"    if (_wp_s0 < {meta}.shape[0] || !{bc_msl}) "
                        f"{{ *{self._tile_addr(spec, ['_wp_s0'])} = {arr}[{s_addr}]; }}"
                    ),
                    "}",
                ]
            cols = tvar.dims[1] if len(tvar.dims) == 2 else 0
            if cols <= 0:
                raise TranslationError(
                    f"tile_store source tile {src_name!r} has no 2-D shape to unravel against"
                )
            s_lines, s_addr = tvar.lin_addr("_wp_t", prefix="_wp_st")
            return [
                f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
                *(f"    {ln};" for ln in s_lines),
                f"    int _wp_s0 = ({offs[0]}) + _wp_t / {cols};",
                f"    int _wp_s1 = ({offs[1]}) + _wp_t - _wp_t / {cols} * {cols};",
                (
                    f"    if ((_wp_s0 < {meta}.shape[0] && _wp_s1 < {meta}.shape[1]) || !{bc_msl}) "
                    f"{{ *{self._tile_addr(spec, ['_wp_s0', '_wp_s1'])} = {arr}[{s_addr}]; }}"
                ),
                "}",
            ]
        if nd != 1:
            raise TranslationError("tile_store from a per-thread tile is 1-D only in the closure")
        s = f"_wp_s{self._store_uid}"
        self._store_uid += 1
        return [
            f"int {s} = ({offs[0]}) + _wp_lane;",
            f"if ({s} < {meta}.shape[0] || !{bc_msl}) {{ *{self._tile_addr(spec, [s])} = {src_name}; }}",
        ]

    def _tile_map(self, target: str, rhs: str) -> list[str]:
        # `wp::tile_map(op, a[, b[, c]])` — emit_Call picks the bare tile_map
        # carrier for ternary user-op maps (mjw constraint._compute_jacp over
        # cdof/offset/affects tiles) and binary/unary_map for 1-2 operands;
        # all three lower identically as an elementwise op application
        m = re.match(r"^wp::tile_(?:(binary|unary)_map|map)\(", rhs)
        if m is None:
            raise TranslationError(f"cannot parse tile_map {rhs!r}")
        open_p = rhs.index("(")
        close = _find_balanced(rhs, open_p)
        args = split_args(rhs[open_p + 1 : close])
        arity = {"binary": 2, "unary": 1}.get(m.group(1), len(args) - 1)
        if len(args) != 1 + arity:
            raise TranslationError(
                f"tile_{m.group(1) or 'map'} with {len(args) - 1} operand(s): expected {arity} — "
                "maps outside the unary/binary/ternary forms are outside the closure"
            )
        op = args[0].strip()
        if op.startswith("wp::"):
            op_msl = _CALL_TABLE.get(op[4:])
        elif op in self.user_funcs:
            # user-function map (vendored dedup suffixes: mjw solver's
            # _state_check_0/_active_check_0 over the JTDAJ tiles) — the
            # translated unit function is directly callable with the same
            # element arity
            op_msl = op
        else:
            op_msl = _CALL_TABLE.get(op)
        if op_msl is None:
            raise TranslationError(f"tile_map operator {op!r} has no MSL translation")
        self.unit.uses_tile_ops = True
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_map target {target!r} is not a tile variable")
        # operands may be tiles (either tier) or plain scalars — upstream
        # tile_map broadcasts a scalar arg per lane (builtins.py b_is_tile)
        ops = [self.tile_vars.get(a.strip()) for a in args[1:]]
        if all(o is None for o in ops):
            raise TranslationError(f"tile_map operands are not tile variables in {rhs!r}")
        scalar_reads: dict[int, str] = {}
        for oi, a in enumerate(args[1:]):
            if ops[oi] is None:
                scalar_reads[oi] = self.expr(a.strip())
        if any(o is not None and o.kind == "shr" for o in ops):
            if tvar.kind != "shr":
                raise TranslationError(
                    f"tile_map over a shared tile requires a shared target tile ({target!r} is per-thread)"
                )
            bd = self._require_block_dim("wp::tile_map")
            self._check_uniform("wp::tile_map over a shared tile")
            arr = f"_wp_tile_shr_{tvar.slot}"
            # operand reads and the target write go through each tile's own
            # linear->storage mapping (a transposed/view operand reorders
            # coordinates); the strided pass iterates the tile's row-major
            # linear index. Coordinate temps get per-tile names so nested
            # views never collide.
            unravel: list[str] = []
            t_lines, t_addr = tvar.lin_addr("_wp_t", prefix="_wp_mt")
            unravel += t_lines
            reads = []
            for oi, (a, o) in enumerate(zip([a.strip() for a in args[1:]], ops)):
                if o is None:
                    reads.append(scalar_reads[oi])
                elif o.kind == "shr":
                    o_lines, o_addr = o.lin_addr("_wp_t", prefix=f"_wp_m{oi}")
                    unravel += o_lines
                    reads.append(f"_wp_tile_shr_{o.slot}[{o_addr}]")
                else:
                    reads.append(self._tile_scalar_read(a, o))
            return [
                f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
                *(f"    {ln};" for ln in unravel),
                f"    {arr}[{t_addr}] = {op_msl}({', '.join(reads)});",
                "}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
            ]
        # per-thread one-element form: element-wise, no barrier
        exprs = [
            scalar_reads[oi] if o is None else self._tile_scalar_read(a.strip(), o)
            for oi, (a, o) in enumerate(zip([a.strip() for a in args[1:]], ops))
        ]
        return [f"{target} = {op_msl}({', '.join(exprs)});"]

    def _tile_scalar_read(self, name: str, tvar: TileVar) -> str:
        """Value expression for a tile operand: the thread's own element."""
        return name  # scalar tiles are plain MSL lvalues

    _TILE_BIN_COPS = {"mul": "*", "add": "+", "sub": "-", "div": "/", "mod": "%"}

    def _tile_binop(self, target: str, rhs: str, op_name: str) -> list[str]:
        """``var = wp::tile_mul(a, b);`` (also add/sub/div/mod/min/max) —
        elementwise tile arithmetic. Both-tile operands are the tile_map
        dispatch's folded binary form; the adjoint codegen also folds
        tile⊗scalar broadcasts into the same call with a plain register
        operand (JTDAJ ``wp.tile_ones(...) * (nefc - k)`` → ``tile_mul(var,
        scalar)``), which lowers as an in-place elementwise map."""
        open_p = rhs.index("(")
        close = _find_balanced(rhs, open_p)
        args = [a.strip() for a in split_args(rhs[open_p + 1 : close])]
        if len(args) != 2:
            raise TranslationError(f"tile_{op_name} with {len(args)} args (needs two operands)")
        a, b = args
        ta = self.tile_vars.get(a)
        tb = self.tile_vars.get(b)
        if ta is not None and tb is not None:
            return self._tile_map(target, f"wp::tile_binary_map(wp::{op_name}, {a}, {b})")
        if ta is None and tb is None:
            raise TranslationError(f"tile_{op_name} in {rhs!r} has no tile operand")
        t_src_name = a if ta is not None else b
        t_src = self.tile_vars[t_src_name]
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_{op_name} target {target!r} is not a tile variable")
        if tvar.elem != t_src.elem:
            raise TranslationError(f"tile_{op_name} elem mismatch {tvar.elem} != {t_src.elem}")
        scalar_src = self.expr(b if ta is not None else a)
        if op_name in self._TILE_BIN_COPS:
            cop = self._TILE_BIN_COPS[op_name]
        elif op_name in ("min", "max"):
            cop = None
        else:
            raise TranslationError(f"tile_{op_name} is not an elementwise arithmetic op")
        combine = (
            (lambda x, y: f"({x} {cop} {y})")
            if cop is not None
            else (lambda x, y: f"wp_{op_name}({x}, {y})")
        )
        self.unit.uses_tile_ops = True
        if tvar.kind == "scalar":
            # per-thread one-element form: elementwise, no barrier (the tile
            # operand is the thread's own element lvalue)
            if t_src.kind != "scalar":
                raise TranslationError(
                    f"tile_{op_name}: per-thread target with shared operand {t_src_name!r} is outside the closure"
                )
            return [f"{target} = {combine(t_src_name, scalar_src)};"]
        bd = self._require_block_dim(f"wp::tile_{op_name}")
        self._check_uniform(f"wp::tile_{op_name}")
        arr = f"_wp_tile_shr_{tvar.slot}"
        lines, addr = tvar.lin_addr("_wp_t", prefix="_wp_tb")
        if t_src.kind != "shr":
            raise TranslationError(
                f"tile_{op_name}: shared target with per-thread operand {t_src_name!r} is outside the closure"
            )
        if t_src_name != target and t_src.slot != tvar.slot:
            # the operand tile has its own linear->storage layout (a
            # view/transpose source reorders coordinates) — read through it
            slines, saddr = t_src.lin_addr("_wp_t", prefix="_wp_tbs")
            lines = slines + lines
            elem_read = f"_wp_tile_shr_{t_src.slot}[{saddr}]"
        else:
            elem_read = f"{arr}[{addr}]"
        return [
            *lines,
            f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
            f"    {arr}[{addr}] = {combine(elem_read, scalar_src)};",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_add_inplace(self, stripped: str) -> list[str]:
        """``wp::tile_add_inplace(dst, src);`` — elementwise ``dst += src``
        (upstream tile_iadd; the JTDAJ accumulator adds the matmul result into
        the reshaped m_tile alias). Each tile keeps its own linear->storage
        mapping; a trailing barrier orders the writes before the next
        cross-lane consumer (the loop's scatter_add)."""
        open_p = stripped.index("(")
        close = _find_balanced(stripped, open_p)
        args = [a.strip() for a in split_args(stripped[open_p + 1 : close])]
        if len(args) != 2:
            raise TranslationError(f"tile_add_inplace with {len(args)} args (needs dst, src)")
        dst_name, src_name = args
        dst = self.tile_vars.get(dst_name)
        src = self.tile_vars.get(src_name)
        if dst is None or src is None:
            raise TranslationError(f"tile_add_inplace operand is not a tile variable in {stripped!r}")
        if dst.size != src.size:
            raise TranslationError(f"tile_add_inplace size mismatch {dst.size} != {src.size}")
        self.unit.uses_tile_ops = True
        if dst.kind == "scalar":
            return [f"{dst_name} = {dst_name} + {src_name};"]
        bd = self._require_block_dim("wp::tile_add_inplace")
        self._check_uniform("wp::tile_add_inplace")
        dlines, daddr = dst.lin_addr("_wp_t", prefix="_wp_ai_d")
        slines, saddr = src.lin_addr("_wp_t", prefix="_wp_ai_s")
        # coordinate decls reference the loop var — emit them INSIDE the
        # loop (a strided operand, e.g. a tile_broadcast alias with a 0
        # stride, unravels coordinates per iteration)
        return [
            f"for (int _wp_t = _wp_lane; _wp_t < {dst.size}; _wp_t += {bd}) {{",
            *(f"    {ln};" for ln in [*slines, *dlines]),
            f"    _wp_tile_shr_{dst.slot}[{daddr}] += _wp_tile_shr_{src.slot}[{saddr}];",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_arange(self, target: str, rhs: str) -> list[str]:
        """``var = wp::tile_arange<dtype, N>(start, stop, step);`` —
        tile.h:2887: ``out[i] = start + i*step`` (stop only sizes the tile
        upstream-side; the SSA template N is authoritative and must equal the
        target tile's element count). start/step are compile-time constants
        (the dispatch folds and dtype-casts them before codegen)."""
        open_p = rhs.index("(")
        m_t = re.match(r"^wp::tile_arange<(.+)>$", rhs[:open_p].strip())
        if m_t is None:
            raise TranslationError(f"cannot parse tile_arange template {rhs!r}")
        targs = split_args(m_t.group(1))
        if len(targs) != 2:
            raise TranslationError(f"tile_arange template {m_t.group(1)!r} needs (dtype, size)")
        elem = msl_type(targs[0])
        n = int(targs[1])
        aargs = split_args(rhs[open_p + 1 : _find_balanced(rhs, open_p)])
        if len(aargs) != 3:
            raise TranslationError(f"tile_arange with {len(aargs)} args (start, stop, step)")
        start_v = self._const_eval(aargs[0])
        step_v = self._const_eval(aargs[2])
        if start_v is None or step_v is None:
            raise TranslationError(
                "tile_arange start/step must be compile-time constants in the closure "
                "(upstream folds them through the dispatch's dtype cast)"
            )
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_arange target {target!r} is not a tile variable")
        if tvar.elem != elem:
            raise TranslationError(f"tile_arange elem {elem} does not match target tile {tvar.elem}")
        if tvar.kind != "scalar" and tvar.size != n:
            raise TranslationError(f"tile_arange template size {n} != target tile {tvar.size}")
        self.unit.uses_tile_ops = True
        lit = _tile_scalar_lit(elem, start_v)
        stepl = _tile_scalar_lit(elem, step_v)
        if tvar.kind == "scalar":
            # per-thread register form: the thread's element is index
            # _wp_lane, but the one-element tier pins the tile to element 0
            return [f"{target} = {lit};"]
        bd = self._require_block_dim("wp::tile_arange")
        self._check_uniform("wp::tile_arange")
        arr = f"_wp_tile_shr_{tvar.slot}"
        return [
            f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
            f"    {arr}[_wp_t] = {lit} + _wp_t * {stepl};",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_ones(self, target: str, rhs: str) -> list[str]:
        """``var = wp::tile_ones<dtype, D0...>();`` — tile.h:2734: every
        element ``T(1)``."""
        open_p = rhs.index("(")
        m_t = re.match(r"^wp::tile_ones<(.+)>$", rhs[:open_p].strip())
        if m_t is None:
            raise TranslationError(f"cannot parse tile_ones template {rhs!r}")
        targs = split_args(m_t.group(1))
        elem = msl_type(targs[0])
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_ones target {target!r} is not a tile variable")
        if tvar.elem != elem:
            raise TranslationError(f"tile_ones elem {elem} does not match target tile {tvar.elem}")
        dims = [int(a.strip()) for a in targs[1:]]
        n = 1
        for d in dims:
            n *= d
        if tvar.kind != "scalar" and tvar.size != n:
            raise TranslationError(f"tile_ones template size {n} != target tile {tvar.size}")
        self.unit.uses_tile_ops = True
        lit = _tile_scalar_lit(elem, 1)
        if tvar.kind == "scalar":
            return [f"{target} = {lit};"]
        bd = self._require_block_dim("wp::tile_ones")
        self._check_uniform("wp::tile_ones")
        arr = f"_wp_tile_shr_{tvar.slot}"
        return [
            f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
            f"    {arr}[_wp_t] = {lit};",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_zeros(self, target: str, rhs: str) -> list[str]:
        open_p = rhs.index("(")
        m_t = re.match(r"^wp::tile_zeros<(.+)>$", rhs[:open_p].strip())
        if m_t is None:
            raise TranslationError(f"cannot parse tile_zeros template {rhs!r}")
        zargs = split_args(m_t.group(1))
        elem = msl_type(zargs[0])
        dims = [int(a.strip()) for a in zargs[1:]]
        size = 1
        for d in dims:
            size *= d
        tvar = self.tile_vars.get(target)
        if tvar is None:
            raise TranslationError(f"tile_zeros target {target!r} is not a tile variable")
        if tvar.elem != elem:
            raise TranslationError(f"tile_zeros elem {elem} does not match target tile {tvar.elem}")
        self.unit.uses_tile_ops = True
        if tvar.kind == "scalar":
            # per-thread register form: the M1 per-thread zero stands
            return [f"{target} = {{}};"]
        bd = self._require_block_dim("wp::tile_zeros with storage=\"shared\"")
        self._check_uniform("wp::tile_zeros into a shared tile")
        arr = f"_wp_tile_shr_{tvar.slot}"
        return [
            f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
            f"    {arr}[_wp_t] = {{}};",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_broadcast(self, target: str, rhs: str) -> list[str]:
        """``var = wp::tile<T>(v);`` — broadcast a register value into a tile.

        Per-thread target: identity (each thread keeps its own copy, M1).
        Shared target (upstream spills broadcasts when any lane can read any
        element, e.g. before a dynamic extract): a strided fill of the
        threadgroup array + barrier.
        """
        self.unit.uses_tile_ops = True
        open_paren = rhs.index("(")
        close = _find_balanced(rhs, open_paren)
        val_src = self.expr(rhs[open_paren + 1 : close])
        tvar = self.tile_vars.get(target)
        if tvar is not None and tvar.kind == "shr":
            bd = self._require_block_dim("wp::tile broadcast into a shared tile")
            self._check_uniform("wp::tile broadcast into a shared tile")
            arr = f"_wp_tile_shr_{tvar.slot}"
            return [
                f"for (int _wp_t = _wp_lane; _wp_t < {tvar.size}; _wp_t += {bd}) {{",
                f"    {arr}[_wp_t] = {val_src};",
                "}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
            ]
        return [f"{target} = {val_src};"]

    def _tile_scatter(self, stripped: str) -> list[str]:
        """``wp::tile_scatter_masked(tile, idx[, idx1], v, mask);`` — each
        masked thread writes its slot of the SHARED tile (1-D linear index or
        2-D coordinate pair), then a barrier orders the write before any
        cross-lane read (single-lane writes are visible to the whole block —
        the M1 per-thread-copy defect is gone)."""
        self.unit.uses_tile_ops = True
        open_p = len("wp::tile_scatter_masked(") - 1
        close = _find_balanced(stripped, open_p)
        sargs = split_args(stripped[open_p + 1 : close])
        if len(sargs) not in (4, 5):
            raise TranslationError(f"tile_scatter_masked with {len(sargs)} args")
        nd = len(sargs) - 3  # 1-D (idx) or 2-D (idx0, idx1)
        tvar = self._tvar(sargs[0].strip(), "tile_scatter_masked")
        if tvar.kind != "shr":
            if self.unit.block_dim is None:
                # plain launch without block_dim: per-thread degenerate
                # lowering — the tile is the thread's own single element,
                # written only when the index selects element (0[,0]) and the
                # mask is set. No barrier: nothing is cross-lane visible.
                tgt = sargs[0].strip()
                guards = [f"({self.expr(s)}) == 0" for s in sargs[1 : 1 + nd]]
                return [
                    (
                        f"if ({self.expr(sargs[2 + nd])} && {' && '.join(guards)}) "
                        f"{{ {tgt} = {self.expr(sargs[1 + nd])}; }}"
                    )
                ]
            raise TranslationError(
                f"tile_scatter_masked target {sargs[0].strip()!r} is not a shared-storage tile: "
                "the per-thread-copy lowering dropped writes to non-writing lanes (M1 defect, "
                "fixed in M2.4) — declare the tile with storage=\"shared\""
            )
        self._check_uniform("wp::tile_scatter_masked")
        arr = f"_wp_tile_shr_{tvar.slot}"
        if nd == 1:
            addr = tvar.coord_addr([self.expr(sargs[1])])
        else:
            addr = tvar.coord_addr([self.expr(sargs[1]), self.expr(sargs[2])])
        value = self.expr(sargs[1 + nd])
        mask = self.expr(sargs[2 + nd])
        return [
            f"if ({mask}) {{ {arr}[{addr}] = {value}; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_scatter_add(self, stripped: str) -> list[str]:
        """``wp::tile_scatter_add(tile, idx[, idx1], value, has_value);`` —
        tile.h:4859 collective scatter-ADD: every lane calls it, lanes with
        ``has_value == false`` skip the write, and the sync at tile.h:4960 is
        unconditional (it also orders the accumulation before cross-lane
        reads).

        Lowered on the Atomic=false template branch verbatim (`data(i) += v`,
        tile.h:4868): the closure's only site (mjw solver.py:2709 JTDAJ
        densify) scatters CSR entries whose (col, row) positions are unique
        per block, so in-block addresses never collide and a serial add per
        lane is bit-exact against the CPU bd=1 golden. A future closure with
        genuine same-address collisions must switch to a CAS loop (Metal has
        no native float atomics) — fail loud is not available here, so the
        uniqueness premise lives in this comment and the corpus fixture.
        """
        self.unit.uses_tile_ops = True
        open_p = len("wp::tile_scatter_add(") - 1
        close = _find_balanced(stripped, open_p)
        sargs = split_args(stripped[open_p + 1 : close])
        if len(sargs) not in (4, 5):
            raise TranslationError(f"tile_scatter_add with {len(sargs)} args")
        nd = len(sargs) - 3  # 1-D (idx) or 2-D (idx0, idx1)
        tvar = self._tvar(sargs[0].strip(), "tile_scatter_add")
        if tvar.kind != "shr":
            if self.unit.block_dim is None:
                # plain launch without block_dim: per-thread degenerate —
                # the tile is the thread's own single element, accumulated
                # only when the index selects element (0[,0]) and has_value
                # is set (matches the scalar tier of tile_scatter_masked)
                tgt = sargs[0].strip()
                guards = [f"({self.expr(s)}) == 0" for s in sargs[1 : 1 + nd]]
                return [
                    (
                        f"if ({self.expr(sargs[2 + nd])} && {' && '.join(guards)}) "
                        f"{{ {tgt} += {self.expr(sargs[1 + nd])}; }}"
                    )
                ]
            raise TranslationError(
                f"tile_scatter_add target {sargs[0].strip()!r} is not a shared-storage tile: "
                "the per-thread-copy lowering cannot accumulate across lanes — "
                'declare the tile with storage="shared"'
            )
        self._check_uniform("wp::tile_scatter_add")
        arr = f"_wp_tile_shr_{tvar.slot}"
        if nd == 1:
            addr = tvar.coord_addr([self.expr(sargs[1])])
        else:
            addr = tvar.coord_addr([self.expr(sargs[1]), self.expr(sargs[2])])
        value = self.expr(sargs[1 + nd])
        has_value = self.expr(sargs[2 + nd])
        return [
            f"if ({has_value}) {{ {arr}[{addr}] += {value}; }}",
            # tile.h:4960: the trailing sync is UNCONDITIONAL (also reached
            # when every lane's has_value is false)
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_extract_target(self, target: str, rhs: str) -> list[str] | None:
        """Statement-position ``target = wp::tile_extract(src, idx);``.

        Returns None when the extract must stay an expression (direct forms
        are handled in _call). Scratch forms (non-zero index on a per-thread
        tile: the owning thread writes a one-element scratch slot, two
        barriers order it — upstream tile.h extract semantics) only lower
        here because they emit statements.
        """
        open_p = rhs.index("(")
        close = _find_balanced(rhs, open_p)
        args = split_args(rhs[open_p + 1 : close])
        if len(args) < 2:
            raise TranslationError(f"tile_extract with {len(args)} args")
        src = args[0].strip()
        tvar = self._tvar(src, "tile_extract")
        if len(args) > 2:
            # per-coordinate form tile_extract(t, i, j, ...): the element
            # address is coords·strides directly (tile.h:4743-4747 overloads)
            coords = [self.expr(a) for a in args[1:]]
            if tvar.kind != "shr" or len(coords) != len(tvar.dims):
                raise TranslationError(
                    f"wp::tile_extract with {len(coords)} coordinate indices needs a "
                    f"shared-storage rank-{len(tvar.dims)} tile"
                )
            return [f"{target} = _wp_tile_shr_{tvar.slot}[{tvar.coord_addr(coords)}];"]
        idx_var = args[1].strip()
        idx_const = self.const_vars.get(idx_var)
        if tvar.kind == "shr":
            arr = f"_wp_tile_shr_{tvar.slot}"
            if idx_const is not None:
                k = int(idx_const)
                if not 0 <= k < tvar.size:
                    raise TranslationError(
                        f"wp::tile_extract index {k} out of bounds for tile of {tvar.size} element(s)"
                    )
                _lines, addr = tvar.lin_addr(str(k))
                return [f"{target} = {arr}[{addr}];"]
            # dynamic index: every element lives in the shared array, so the
            # read is direct through the tile's linear->storage mapping
            # (upstream's negative-index normalization is already in the SSA
            # index expression)
            d_lines, d_addr = tvar.lin_addr(self.expr(idx_var), prefix="_wp_x")
            return [
                *(f"{ln};" for ln in d_lines),
                f"{target} = {arr}[{d_addr}];",
            ]
        if idx_const == "0":
            return [f"{target} = {src};"]  # per-thread form: identity (M1)
        if idx_const is None:
            raise TranslationError(
                f"wp::tile_extract with a runtime index on a per-thread tile ({src!r}): "
                "the degenerate tile holds a single element, so only the constant index 0 "
                "is in the closure (upstream spills to shared storage, which needs an "
                "explicit block_dim launch)"
            )
        # constant non-zero index: the owning lane writes, everyone reads
        bd = self._require_block_dim("wp::tile_extract from a per-thread tile")
        self._check_uniform("wp::tile_extract broadcast")
        self.unit.uses_tile_ops = True
        k = self._claim_shr(tvar.elem, 1)
        return [
            f"if (_wp_lane == {int(idx_const)}) {{ _wp_tile_shr_{k}[0] = {src}; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"{target} = _wp_tile_shr_{k}[0];",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    # -- tile algebra (KEP-0001 M2.4 P3+P4) -----------------------------------
    #
    # Dual-mode like every tile op: launch_tiled lowers the shared-storage
    # (threadgroup) forms below; any plain launch lowers every op to the
    # per-thread scalar degenerate (one-element tile — the CPU-golden
    # convention; serial-by-construction, so cholesky/solve degrade to the
    # same serial algorithm upstream runs at WP_TILE_BLOCK_DIM == 1).
    #
    # Float-order discipline: every loop order below matches the upstream
    # scalar fallback exactly (tile_cholesky.h / tile_solve.h / tile_matmul.h
    # partitioned_gemm scalar paths, tile_reduce.h axis fold, tile_scan.h
    # sequential CPU pass) so warp CPU goldens and Metal agree bit-for-bit up
    # to sqrt/FMA rounding — well inside the 1e-5 corpus tolerance.

    def _shrub(self, tvar: TileVar) -> str:
        if tvar.kind != "shr" or tvar.slot < 0:
            raise TranslationError(
                f"tile op requires a shared-storage tile (got "
                f"{'an unbound alias' if tvar.slot < 0 else 'a per-thread scalar tile'}); "
                "upstream forces operands to shared storage at parse time, which a "
                "plain launch (per-thread scalar tier) cannot retrofit"
            )
        return f"_wp_tile_shr_{tvar.slot}"

    def _float_tile(self, tvar: TileVar, what: str) -> None:
        if tvar.elem not in ("float", "half"):
            raise TranslationError(f"{what} over tile element {tvar.elem!r}: float/half tiles only")
        if tvar.elem == "half":
            raise TranslationError(f"{what}: half-precision tiles are outside the closure")

    def _parse_tile_call(self, rhs: str, name: str, with_template: bool) -> tuple[list[str], str]:
        """(args, template text) behind ``[wp::]name<...>(...)``."""
        open_p = rhs.index("(")
        head = rhs[:open_p].strip()
        m = re.match(r"^(?:wp::)?([A-Za-z_]\w*)(?:<(.*)>)?$", head, re.DOTALL)
        if m is None or m.group(1) != name:
            raise TranslationError(f"cannot parse {name} call {rhs!r}")
        close = _find_balanced(rhs, open_p)
        if rhs[close + 1 :].strip():
            raise TranslationError(f"trailing text after {name} call {rhs!r}")
        args = split_args(rhs[open_p + 1 : close])
        tmpl = (m.group(2) or "").strip()
        if with_template and not tmpl:
            raise TranslationError(f"{name} call {rhs!r} carries no template (outside the closure)")
        return args, tmpl

    def _tile_alias_bind(self, target: str, rhs: str, op: str) -> list[str]:
        """``var = wp::tile_transpose(src)`` / ``tile_reshape<T>(src)`` /
        ``tile_view<T>(src, offs...)`` — pure metadata alias (upstream
        tile.h:5462-5628 Owner=false shared views, zero movement, zero
        barrier). The target's pending alias decl binds to the source tile's
        threadgroup array with the view's shape/strides/base; physical
        reordering happens in the consuming ops' linear->storage mapping.
        Plain launches bind the per-thread scalar value (identity alias; a
        non-zero view offset cannot address the thread's one element and
        fails loud)."""
        self.unit.uses_tile_ops = True
        with_template = op in ("tile_reshape", "tile_view", "tile_broadcast")
        args, tmpl = self._parse_tile_call(rhs, op, with_template)
        src_name = args[0].strip()
        tgt = self.tile_vars.get(target)
        src = self.tile_vars.get(src_name)
        if tgt is None or src is None:
            raise TranslationError(f"wp::{op} operand is not a tile variable in {rhs!r}")
        if src.kind == "scalar" or self.unit.block_dim is None:
            if op == "tile_view":
                offs = [self.const_vars.get(a.strip(), a.strip()) for a in args[1:]]
                if any(o not in ("0", None) and o != 0 for o in offs):
                    raise TranslationError(
                        "wp::tile_view with a non-zero offset on a plain launch: the per-thread "
                        "scalar tier holds one element per tile — identity aliases only"
                    )
            tgt.kind = "scalar"
            return [f"{target} = {src_name};"]
        if tgt.slot >= 0:
            raise TranslationError(
                f"wp::{op} target {target!r} owns storage (its decl was not the alias '= nullptr' form)"
            )
        if op == "tile_transpose":
            if len(tgt.dims) != 2 or len(src.dims) != 2:
                raise TranslationError("wp::tile_transpose is 2-D only in the closure")
            tgt.dims = list(reversed(src.dims))
            tgt.strides = list(reversed(src.strides))
            tgt.base = src.base
        elif op == "tile_broadcast":
            # ``wp::tile_broadcast<D0, D1, ..., s0, s1, ...>(src)`` — upstream
            # tile.h:5488 forces the source to shared storage and returns a
            # pure stride-0 alias: the flat template carries the broadcast
            # dims then the strides verbatim (stride-0 dims repeat the source
            # lanes; coord_addr already skips those axes)
            targs = [int(a) for a in tmpl.replace(" ", "").split(",") if a]
            if not targs or len(targs) % 2:
                raise TranslationError(
                    f"wp::tile_broadcast template {tmpl!r} must be dims-then-strides (even count)"
                )
            half = len(targs) // 2
            tgt.dims = targs[:half]
            tgt.strides = targs[half:]
            tgt.base = src.base
        else:
            dims = _tile_shape_dims(tmpl)
            stride_m = re.search(r"wp::tile_stride_t<([\d,\s]+)>", tmpl)
            if stride_m is None:
                raise TranslationError(f"wp::{op} template carries no tile_stride_t: {rhs!r}")
            strides = [int(s) for s in stride_m.group(1).replace(" ", "").split(",") if s]
            tgt.dims = dims
            tgt.strides = strides
            if op == "tile_reshape":
                tgt.base = src.base
            else:  # tile_view: base = source base + offset·source strides
                terms = [] if src.base == "0" else [f"({src.base})"]
                offs = args[1:]
                for i, stride in enumerate(src.strides):
                    off = self.expr(offs[i]) if i < len(offs) else "0"
                    if stride == 1:
                        terms.append(f"({off})")
                    elif stride != 0:
                        terms.append(f"({off}) * {stride}")
                tgt.base = " + ".join(terms) if terms else "0"
        size = 1
        for d in tgt.dims:
            size *= d
        tgt.size = size
        tgt.slot = src.slot  # the alias borrows the source's threadgroup array
        return []

    def _tile_matmul(self, target: str | None, rhs: str, acc_name: bool) -> list[str]:
        """``c = wp::tile_matmul(a, b)`` (alpha=1, beta=0) /
        ``wp::tile_matmul_acc(a, b, c)`` (alpha=1, beta=1) — the SSA carries
        all eight args ``(fwd, bkA, bkB, A, B, C, alpha, beta)`` either way
        (the mathdx function pointers lower to 0; dormant). Simplified GEMM:
        per-thread strided output elements, k ascending (upstream's k order),
        one end barrier."""
        args, _tmpl = self._parse_tile_call(rhs, "tile_matmul_acc" if acc_name else "tile_matmul", False)
        if len(args) != 8:
            raise TranslationError(f"tile_matmul with {len(args)} args (needs fwd, bkA, bkB, A, B, C, alpha, beta)")
        a_t = self._tvar(args[3].strip(), "tile_matmul")
        b_t = self._tvar(args[4].strip(), "tile_matmul")
        if target is None:
            target = args[5].strip()
        c_t = self._tvar(target, "tile_matmul")
        alpha = self.expr(args[6])
        beta = self.expr(args[7])
        if len(a_t.dims) != 2 or len(b_t.dims) != 2 or len(c_t.dims) != 2:
            raise TranslationError("tile_matmul operands must be 2-D tiles")
        m, k = a_t.dims
        if b_t.dims[0] != k:
            raise TranslationError(f"tile_matmul shape mismatch A {a_t.dims} B {b_t.dims}")
        if c_t.dims[0] != m or c_t.dims[1] != b_t.dims[1]:
            raise TranslationError(f"tile_matmul shape mismatch A {a_t.dims} B {b_t.dims} C {c_t.dims}")
        self._float_tile(c_t, "tile_matmul")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            # plain scalar tier: one-element tiles — (1,1)@(1,1) accumulate
            if a_t.kind != "shr" and b_t.kind != "shr" and c_t.kind != "shr":
                return [
                    (
                        f"{target} = ({alpha}) * ({args[3].strip()} * {args[4].strip()}) "
                        f"+ ({beta}) * {target};"
                    )
                ]
            raise TranslationError("tile_matmul on a plain launch lowers per-thread scalar tiles only")
        bd = self._require_block_dim("wp::tile_matmul")
        self._check_uniform("wp::tile_matmul")
        aarr, barr, carr = self._shrub(a_t), self._shrub(b_t), self._shrub(c_t)
        n = b_t.dims[1]
        a_un, _a_addr = a_t.lin_addr("_wp_g", prefix="_wp_ga")
        b_un, _b_addr = b_t.lin_addr("_wp_g", prefix="_wp_gb")
        c_un, c_addr = c_t.lin_addr("_wp_g", prefix="_wp_gc")
        # k ascending — upstream's partitioned_gemm scalar accumulation order
        return [
            f"for (int _wp_g = _wp_lane; _wp_g < {m * n}; _wp_g += {bd}) {{",
            *(f"    {ln};" for ln in a_un + b_un + c_un),
            f"    int _wp_i = _wp_g / {n};",
            f"    int _wp_j = _wp_g - _wp_i * {n};",
            "    float _wp_acc = {};",
            f"    for (int _wp_k = 0; _wp_k < {k}; ++_wp_k) {{",
            f"        int _wp_ai = _wp_i * {k} + _wp_k;",
            f"        int _wp_bi = _wp_k * {n} + _wp_j;",
            *(f"        {ln};" for ln in a_t.lin_addr("_wp_ai", prefix="_wp_ka")[0]),
            *(f"        {ln};" for ln in b_t.lin_addr("_wp_bi", prefix="_wp_kb")[0]),
            (
                f"        _wp_acc = _wp_acc + {aarr}[{a_t.lin_addr('_wp_ai', prefix='_wp_ka')[1]}] "
                f"* {barr}[{b_t.lin_addr('_wp_bi', prefix='_wp_kb')[1]}];"
            ),
            "    }",
            f"    {carr}[{c_addr}] = ({alpha}) * _wp_acc + ({beta}) * {carr}[{c_addr}];",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_diag_add(self, target: str, rhs: str) -> list[str]:
        """``c = wp::tile_diag_add(a, b)`` — c = a with c[t,t] += b[t]
        (upstream tile.h:6131; the target is a fresh tile, the source stays
        intact)."""
        args, _tmpl = self._parse_tile_call(rhs, "tile_diag_add", False)
        if len(args) != 3:
            raise TranslationError(f"tile_diag_add with {len(args)} args (needs a, b, c)")
        a_t = self._tvar(args[0].strip(), "tile_diag_add")
        b_t = self._tvar(args[1].strip(), "tile_diag_add")
        if args[2].strip() != target:
            raise TranslationError(f"tile_diag_add target {target!r} does not match its SSA out arg")
        c_t = self._tvar(target, "tile_diag_add")
        if len(a_t.dims) != 2 or a_t.dims[0] != a_t.dims[1]:
            raise TranslationError(f"tile_diag_add source {a_t.dims} is not square")
        if len(c_t.dims) != 2 or c_t.dims != a_t.dims:
            raise TranslationError(f"tile_diag_add target {c_t.dims} does not match source {a_t.dims}")
        if len(b_t.dims) != 1 or b_t.dims[0] != a_t.dims[0]:
            raise TranslationError(f"tile_diag_add diagonal {b_t.dims} does not match {a_t.dims}")
        self._float_tile(c_t, "tile_diag_add")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            if a_t.kind != "shr" and b_t.kind != "shr" and c_t.kind != "shr":
                # plain scalar tier: the one-element tile IS the diagonal
                return [f"{target} = {args[0].strip()} + {args[1].strip()};"]
            raise TranslationError("tile_diag_add on a plain launch lowers per-thread scalar tiles only")
        bd = self._require_block_dim("wp::tile_diag_add")
        self._check_uniform("wp::tile_diag_add")
        n = a_t.dims[0]
        aarr, barr, carr = self._shrub(a_t), self._shrub(b_t), self._shrub(c_t)
        c_un, c_addr = c_t.lin_addr("_wp_t", prefix="_wp_ct")
        b_addr = b_t.coord_addr(["_wp_i"])
        a_un2, a_addr2 = a_t.lin_addr("_wp_q", prefix="_wp_ca")
        return [
            "{ float _wp_z = {};",
            f"for (int _wp_t = _wp_lane; _wp_t < {n * n}; _wp_t += {bd}) {{",
            *(f"    {ln};" for ln in c_un),
            f"    int _wp_i = _wp_t / {n};",
            f"    int _wp_j = _wp_t - _wp_i * {n};",
            f"    int _wp_q = _wp_i * {n} + _wp_j;",
            *(f"    {ln};" for ln in a_un2),
            f"    {carr}[{c_addr}] = {aarr}[{a_addr2}]",
            f"        + ((_wp_i == _wp_j) ? {barr}[{b_addr}] : _wp_z);",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
        ]

    def _tile_scan(self, target: str, rhs: str, inclusive: bool) -> list[str]:
        """``r = wp::tile_scan_inclusive/exclusive(t)`` — prefix sum over the
        tile's linear elements (upstream tile_scan.h: shared copy-in, block
        scan, register copy-out). Simplified block scan: Hillis-Steele over
        the scratch copy (double barrier per log2 step, read-before-write
        captured per lane); float order is NOT upstream's sequential one, so
        conformance comparisons allow 1e-4 relative."""
        args, _tmpl = self._parse_tile_call(rhs, "tile_scan_inclusive" if inclusive else "tile_scan_exclusive", False)
        if len(args) != 1:
            raise TranslationError(f"tile_scan with {len(args)} args")
        src = self._tvar(args[0].strip(), "tile_scan")
        tgt = self._tvar(target, "tile_scan")
        self._float_tile(tgt, "tile_scan")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            # plain scalar tier: the one-element prefix is the element itself;
            # the exclusive prefix is the empty-sum identity
            if src.kind != "shr" and tgt.kind != "shr":
                if inclusive:
                    return [f"{target} = {args[0].strip()};"]
                return ["{ float _wp_z = {};", f"    {target} = _wp_z;", "}"]
            raise TranslationError("tile_scan on a plain launch lowers per-thread scalar tiles only")
        bd = self._require_block_dim("wp::tile_scan")
        self._check_uniform("wp::tile_scan")
        n = src.size
        sarr = self._shrub(src)
        if tgt.kind == "shr":
            out_arr = self._shrub(tgt)
            n_scan = n
            scratch = self._claim_shr("float", n)
        else:
            out_arr = None
            n_scan = bd if src.kind != "shr" else src.size
            scratch = self._claim_shr("float", n_scan)
        # ping-pong scratch: the add step reads A[t - d] while peers write
        # A[t'], so each Hillis-Steele step computes into a second scratch,
        # then copies back — every read sees the previous step's state
        scratch_b = self._claim_shr("float", n_scan)
        if src.kind == "shr":
            s_lines, s_addr = src.lin_addr("_wp_t", prefix="_wp_ss")
            fill = [
                f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
                *(f"    {ln};" for ln in s_lines),
                f"    _wp_tile_shr_{scratch}[_wp_t] = {sarr}[{s_addr}];",
                "}",
            ]
        else:
            # scalar tile source: each lane stages its own element (its
            # linear index is the lane — the M1 convention)
            fill = [
                f"_wp_tile_shr_{scratch}[_wp_lane] = {args[0].strip()};",
            ]
        lines = [
            *fill,
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            # Hillis-Steele inclusive scan: one add pass + copy-back per
            # log2(n) step (the copy-back pass exists so the ping-pong
            # scratch never aliases the reads)
            f"for (int _wp_d = 1; _wp_d < {n_scan}; _wp_d <<= 1) {{",
            f"    for (int _wp_t = _wp_lane; _wp_t < {n_scan}; _wp_t += {bd}) {{",
            f"        _wp_tile_shr_{scratch_b}[_wp_t] = _wp_tile_shr_{scratch}[_wp_t]",
            f"            + ((_wp_t >= _wp_d) ? _wp_tile_shr_{scratch}[_wp_t - _wp_d] : 0.0f);",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"    for (int _wp_t = _wp_lane; _wp_t < {n_scan}; _wp_t += {bd}) {{",
            f"        _wp_tile_shr_{scratch}[_wp_t] = _wp_tile_shr_{scratch_b}[_wp_t];",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
        ]
        if tgt.kind == "shr":
            t_lines, t_addr = tgt.lin_addr("_wp_t", prefix="_wp_to")
            if inclusive:
                copy_out = [
                    f"for (int _wp_t = _wp_lane; _wp_t < {tgt.size}; _wp_t += {bd}) {{",
                    *(f"    {ln};" for ln in t_lines),
                    f"    {out_arr}[{t_addr}] = _wp_tile_shr_{scratch}[_wp_t];",
                    "}",
                ]
            else:
                copy_out = [
                    "{ float _wp_z = {};",
                    f"for (int _wp_t = _wp_lane; _wp_t < {tgt.size}; _wp_t += {bd}) {{",
                    *(f"    {ln};" for ln in t_lines),
                    f"    {out_arr}[{t_addr}] = (_wp_t > 0) ? _wp_tile_shr_{scratch}[_wp_t - 1] : _wp_z;",
                    "}",
                    "}",
                ]
            lines += copy_out
        else:
            if inclusive:
                lines.append(f"{target} = _wp_tile_shr_{scratch}[_wp_lane];")
            else:
                lines += [
                    "{ float _wp_z = {};",
                    f"    {target} = (_wp_lane > 0) ? _wp_tile_shr_{scratch}[_wp_lane - 1] : _wp_z;",
                    "}",
                ]
        lines.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        return lines

    def _tile_sum_axis(self, target: str, rhs: str) -> list[str]:
        """``r = wp::tile_sum_axis<K>(t)`` — reduce along axis K, result tile
        with that dimension removed (upstream tile_reduce.h axis fold: each
        output slice accumulates its line in ascending linear order; here the
        per-output accumulation is one thread's serial k loop — same order as
        upstream's CPU bd=1 path)."""
        args, tmpl = self._parse_tile_call(rhs, "tile_sum_axis", True)
        if len(args) != 1:
            raise TranslationError(f"tile_sum_axis with {len(args)} args")
        if tmpl not in ("0", "1"):
            raise TranslationError(f"tile_sum_axis axis {tmpl!r}: only 0/1 are in the closure")
        axis = int(tmpl)
        src = self._tvar(args[0].strip(), "tile_sum_axis")
        tgt = self._tvar(target, "tile_sum_axis")
        self._float_tile(tgt, "tile_sum_axis")
        if len(src.dims) != 2:
            raise TranslationError(f"tile_sum_axis over {src.dims}: 2-D tiles only in the closure")
        want = [src.dims[1 - axis]]
        if len(tgt.dims) != 1 or tgt.dims[0] != want[0]:
            raise TranslationError(f"tile_sum_axis result {tgt.dims} does not match axis-{axis} shape {want}")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            if src.kind != "shr" and tgt.kind != "shr":
                return [f"{target} = {args[0].strip()};"]
            raise TranslationError("tile_sum_axis on a plain launch lowers per-thread scalar tiles only")
        bd = self._require_block_dim("wp::tile_sum_axis")
        self._check_uniform("wp::tile_sum_axis")
        sarr = self._shrub(src)
        out_len = want[0]
        red_len = src.dims[axis]
        lines = ["{ float _wp_z = {};"]
        if tgt.kind == "shr":
            oarr = self._shrub(tgt)
            o_un, o_addr = tgt.lin_addr("_wp_o", prefix="_wp_ot")
            lines += [
                f"for (int _wp_o = _wp_lane; _wp_o < {out_len}; _wp_o += {bd}) {{",
                *(f"    {ln};" for ln in o_un),
                "    float _wp_acc = _wp_z;",
                f"    for (int _wp_k = 0; _wp_k < {red_len}; ++_wp_k) {{",
                f"        int _wp_q = {f'_wp_k * {out_len} + _wp_o' if axis == 0 else f'_wp_o * {red_len} + _wp_k'};",
                *(f"        {ln};" for ln in src.lin_addr("_wp_q", prefix="_wp_qs")[0]),
                f"        _wp_acc = _wp_acc + {sarr}[{src.lin_addr('_wp_q', prefix='_wp_qs')[1]}];",
                "    }",
                f"    {oarr}[{o_addr}] = _wp_acc;",
                "}",
            ]
        else:
            if out_len != bd:
                raise TranslationError(
                    f"tile_sum_axis result tile {tgt.dims} is per-thread scalar but its shape "
                    "does not match the block_dim lane count"
                )
            lines += [
                f"for (int _wp_o = _wp_lane; _wp_o < {out_len}; _wp_o += {bd}) {{",
                "    float _wp_acc = _wp_z;",
                f"    for (int _wp_k = 0; _wp_k < {red_len}; ++_wp_k) {{",
                f"        int _wp_q = {f'_wp_k * {out_len} + _wp_o' if axis == 0 else f'_wp_o * {red_len} + _wp_k'};",
                *(f"        {ln};" for ln in src.lin_addr("_wp_q", prefix="_wp_qs")[0]),
                f"        _wp_acc = _wp_acc + {sarr}[{src.lin_addr('_wp_q', prefix='_wp_qs')[1]}];",
                "    }",
                f"    if (_wp_o == _wp_lane) {{ {target} = _wp_acc; }}",
                "}",
            ]
        lines += [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
        ]
        return lines

    def _cholesky_lines(self, a_t: TileVar, o_t: TileVar, upper: bool) -> list[str]:
        """Cooperative column Cholesky (upstream tile_cholesky.h
        scalar_cholesky_impl verbatim): redundant diagonal compute, thread-0
        diagonal write, thread-strided row updates and opposite-triangle
        zeroing, one barrier per column. ``upper`` selects the stored
        triangle (fill_mode)."""
        n = a_t.dims[0]
        aarr, oarr = self._shrub(a_t), self._shrub(o_t)
        bd = self.unit.block_dim

        def ai(r: str, c: str) -> str:
            # A is read in ORIGINAL coordinates — upstream reads the fill-mode
            # triangle literally: A.data(tile_coord(j, i)) for upper, A.data(
            # tile_coord(i, j)) for lower (tile_cholesky.h:116); only the OUT
            # mapping (oi) is transposed. The call sites pass the upstream
            # order already, so this is the identity — transposing here (or at
            # the call site) double-swaps onto the opposite triangle and
            # silently drops the coupling entries of a triangle-filled input
            # (the mjw Hessian stores J^T D A J fill-in in the upper triangle
            # only), which perturbs the factor and NaNs the augmented border
            # block downstream.
            return a_t.coord_addr([r, c])

        def oi(r: str, c: str) -> str:
            return o_t.coord_addr([c, r] if upper else [r, c])

        return [
            f"{a_t.elem} _wp_z = {{}};",
            f"for (int _wp_j = 0; _wp_j < {n}; ++_wp_j) {{",
            f"    float _wp_s = {aarr}[{ai('_wp_j', '_wp_j')}];",
            # upstream's in-place alias guard: order the diagonal read
            # before any peer's Out writes of this column
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "    for (int _wp_k = 0; _wp_k < _wp_j; ++_wp_k) {",
            f"        float _wp_r = {oarr}[{oi('_wp_j', '_wp_k')}];",
            "        _wp_s = _wp_s - _wp_r * _wp_r;",
            "    }",
            "    _wp_s = sqrt(_wp_s);",
            "    float _wp_inv_s = 1.0f / _wp_s;",
            f"    if (_wp_lane == 0) {{ {oarr}[{oi('_wp_j', '_wp_j')}] = _wp_s; }}",
            f"    for (int _wp_i = _wp_j + 1 + _wp_lane; _wp_i < {n}; _wp_i += {bd}) {{",
            f"        float _wp_si = {aarr}[{ai('_wp_j', '_wp_i') if upper else ai('_wp_i', '_wp_j')}];",
            "        for (int _wp_k = 0; _wp_k < _wp_j; ++_wp_k) {",
            f"            _wp_si = _wp_si - {oarr}[{oi('_wp_i', '_wp_k')}] * {oarr}[{oi('_wp_j', '_wp_k')}];",
            "        }",
            f"        {oarr}[{oi('_wp_i', '_wp_j')}] = _wp_si * _wp_inv_s;",
            "    }",
            f"    for (int _wp_k = _wp_j + 1 + _wp_lane; _wp_k < {n}; _wp_k += {bd}) {{",
            f"        {oarr}[{oi('_wp_j', '_wp_k')}] = _wp_z;",
            "    }",
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "}",
        ]

    def _tile_cholesky(self, target: str | None, rhs: str) -> list[str]:
        """``l = wp::tile_cholesky<Upper>(fwd, bkA, bkB, a, out)`` (mathdx
        pointers lower to 0; dormant) / statement
        ``wp::tile_cholesky_inplace<Upper>(fwd, a)``. Out-of-place writes the
        target tile (fresh); in-place factors the source tile itself."""
        name = "tile_cholesky_inplace" if rhs.startswith(("wp::tile_cholesky_inplace", "tile_cholesky_inplace")) else "tile_cholesky"
        args, tmpl = self._parse_tile_call(rhs, name, True)
        if tmpl not in ("true", "false"):
            raise TranslationError(f"{name} fill template {tmpl!r}: only true/false are in the closure")
        upper = tmpl == "true"
        if name == "tile_cholesky_inplace":
            if len(args) != 2:
                raise TranslationError(f"{name} with {len(args)} args (needs fwd, a)")
            a_name = args[1].strip()
            a_t = self._tvar(a_name, name)
            o_t = a_t
        else:
            if len(args) != 5:
                raise TranslationError(f"{name} with {len(args)} args (needs fwd, bkA, bkB, a, out)")
            a_t = self._tvar(args[3].strip(), name)
            if target is None:
                target = args[4].strip()
            o_t = self._tvar(target, name)
        self._float_tile(o_t, name)
        if len(a_t.dims) != 2 or a_t.dims[0] != a_t.dims[1]:
            raise TranslationError(f"{name} source {a_t.dims} is not square")
        if o_t.dims != a_t.dims:
            raise TranslationError(f"{name} target {o_t.dims} does not match source {a_t.dims}")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            # plain scalar tier: the one-element factor is sqrt(a) (the n=1
            # serial case — upper/lower coincide)
            if a_t.kind != "shr":
                if name == "tile_cholesky_inplace":
                    return [f"{a_name} = sqrt({a_name});"]
                return [f"{target} = sqrt({args[3].strip()});"]
            raise TranslationError(f"{name} on a plain launch lowers per-thread scalar tiles only")
        self._require_block_dim(name)
        self._check_uniform(name)
        lines = self._cholesky_lines(a_t, o_t, upper)
        lines.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        return lines

    def _solve_phase_lines(
        self,
        a_t: TileVar,
        x_t: TileVar,
        y_t: TileVar | None,
        phase: str,
        a_swap: bool,
    ) -> list[str]:
        """One triangular-substitution phase (upstream tile_solve.h
        scalar_cholesky_forward/back_substitution): vector RHS serial on
        thread 0, matrix RHS strided over RHS columns, one barrier at the
        end. ``phase`` "fwd" (i ascending) or "bwd" (i descending);
        ``a_swap`` reads A at (j, i) instead of (i, j) (the transposed
        triangle)."""
        n = a_t.dims[0]
        aarr = self._shrub(a_t)
        xarr = self._shrub(x_t)
        matrix = len(x_t.dims) == 2
        m = x_t.dims[1] if matrix else 1
        bd = self.unit.block_dim

        def ai(i: str, j: str) -> str:
            return a_t.coord_addr([j, i] if a_swap else [i, j])

        def xi(i: str, k: str) -> str:
            return x_t.coord_addr([i] if not matrix else [i, k])

        yarr = xarr if y_t is None else self._shrub(y_t)

        def yi_expr(i: str, k: str) -> str:
            if y_t is None:
                return xi(i, k)
            return y_t.coord_addr([i] if not matrix else [i, k])

        lines: list[str] = []
        if phase == "fwd":
            i_head = f"for (int _wp_i = 0; _wp_i < {n}; ++_wp_i)"
        else:
            i_head = f"for (int _wp_i = {n} - 1; _wp_i >= 0; --_wp_i)"
        if matrix:
            lines.append(f"for (int _wp_k = _wp_lane; _wp_k < {m}; _wp_k += {bd}) {{")
            lines.append(f"    {i_head} {{")
            lines.append(f"        float _wp_s = {yarr}[{yi_expr('_wp_i', '_wp_k')}];")
        else:
            lines.append("if (_wp_lane == 0) {")
            lines.append(f"    {i_head} {{")
            lines.append(f"        float _wp_s = {yarr}[{yi_expr('_wp_i', '_wp_i')}];")
        if phase == "fwd":
            jbounds = ("0", "_wp_i")
            jcond = "_wp_j < _wp_i"
        else:
            jbounds = ("_wp_i + 1", str(n))
            jcond = "_wp_j < " + str(n)
        lines.append(f"        for (int _wp_j = {jbounds[0]}; {jcond}; ++_wp_j) {{")
        lines.append(f"            _wp_s = _wp_s - {aarr}[{ai('_wp_i', '_wp_j')}] * {xarr}[{xi('_wp_j', '_wp_k')}];")
        lines.append("        }")
        lines.append(f"        float _wp_d = {aarr}[{ai('_wp_i', '_wp_i')}];")
        lines.append(f"        {xarr}[{xi('_wp_i', '_wp_k')}] = (_wp_d != 0.0f) ? _wp_s / _wp_d : _wp_s;")
        lines.append("    }")
        lines.append("}")
        lines.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        return lines

    def _tile_solve(self, target: str | None, rhs: str) -> list[str]:
        """The triangular-solve family (upstream tile_solve.h):
        ``z = [wp::]tile_lower_solve(fwd, bkwd, L, y, z)`` /
        ``x = [wp::]tile_upper_solve(fwd, U, z, x)`` / statement in-place
        variants / ``x = wp::tile_cholesky_solve<Upper>(fwd, a, y, x)`` and
        its in-place form. Out-of-place copies the RHS into the result tile
        first (alias protection, upstream ``z = y``); the solve phases then
        run as in upstream (vector RHS serial on thread 0, matrix RHS
        thread-strided over RHS columns)."""
        m = re.match(r"^(?:wp::)?(tile_lower_solve|tile_upper_solve|tile_cholesky_solve"
                     r"|tile_lower_solve_inplace|tile_upper_solve_inplace|tile_cholesky_solve_inplace)"
                     r"(?:<([^>]*)>)?\(", rhs)
        if m is None:
            raise TranslationError(f"cannot parse solve call {rhs!r}")
        name = m.group(1)
        tmpl = (m.group(2) or "").strip()
        open_p = rhs.index("(")
        close = _find_balanced(rhs, open_p)
        args = split_args(rhs[open_p + 1 : close])
        inplace = name.endswith("_inplace")
        base = name[: -len("_inplace")] if inplace else name
        if inplace:
            if len(args) != 3:
                raise TranslationError(f"{name} with {len(args)} args (needs fwd, a, y)")
            a_name = args[1].strip()
            a_t = self._tvar(a_name, name)
            y_name = args[-1].strip()
            y_t = self._tvar(y_name, name)
            x_t, x_name = y_t, y_name
        else:
            if base == "tile_cholesky_solve":
                if len(args) != 4:
                    raise TranslationError(f"{name} with {len(args)} args (needs fwd, a, y, x)")
                a_name = args[1].strip()
                y_name = args[2].strip()
                a_t = self._tvar(a_name, name)
                y_t = self._tvar(y_name, name)
                x_name = args[3].strip() if target is None else target
            elif base == "tile_lower_solve":
                if len(args) != 5:
                    raise TranslationError(f"{name} with {len(args)} args (needs fwd, bkwd, L, y, z)")
                a_name = args[2].strip()
                y_name = args[3].strip()
                a_t = self._tvar(a_name, name)
                y_t = self._tvar(y_name, name)
                x_name = args[4].strip() if target is None else target
            else:  # tile_upper_solve
                if len(args) != 4:
                    raise TranslationError(f"{name} with {len(args)} args (needs fwd, U, z, x)")
                a_name = args[1].strip()
                y_name = args[2].strip()
                a_t = self._tvar(a_name, name)
                y_t = self._tvar(y_name, name)
                x_name = args[3].strip() if target is None else target
            x_t = self._tvar(x_name, name)
        self._float_tile(x_t, name)
        if len(a_t.dims) != 2 or a_t.dims[0] != a_t.dims[1]:
            raise TranslationError(f"{name} matrix {a_t.dims} is not square")
        if len(y_t.dims) > 2 or y_t.dims[0] != a_t.dims[0]:
            raise TranslationError(f"{name} rhs {y_t.dims} does not match the matrix rows {a_t.dims}")
        if x_t.dims != y_t.dims:
            raise TranslationError(f"{name} result {x_t.dims} does not match rhs {y_t.dims}")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            # plain scalar tier: the n=1 serial solve — one guarded division
            # per phase (vector form; matrix rhs with one column is identical)
            if a_t.kind != "shr" and y_t.kind != "shr" and x_t.kind != "shr":
                a_src = a_name  # the resolved matrix-tile SSA var for every form
                if inplace:
                    y_src = y_name
                    step = f"({y_src} = ({a_src} != 0.0f) ? {y_src} / {a_src} : {y_src})"
                    lines = [step]
                    if base == "tile_cholesky_solve":
                        lines.append(step)  # second (backward) phase
                    return lines
                # out-of-place: the result tile starts at the zeros/broadcast
                # value — copy the rhs in first (upstream `z = y`), then the
                # per-phase guarded division(s)
                phase = f"({x_name} = ({a_src} != 0.0f) ? {x_name} / {a_src} : {x_name})"
                head = f"{x_name} = {y_name};"
                if base == "tile_cholesky_solve":
                    return [head, phase, phase]
                return [head, phase]
            raise TranslationError(f"{name} on a plain launch lowers per-thread scalar tiles only")
        self._require_block_dim(name)
        self._check_uniform(name)
        lines: list[str] = []
        if not inplace and x_t.slot != y_t.slot:
            # copy y into x first (upstream `z = y`), through each tile's own
            # linear->storage mapping
            yarr = self._shrub(y_t)
            xarr = self._shrub(x_t)
            y_un, y_addr = y_t.lin_addr("_wp_t", prefix="_wp_cy")
            x_un, x_addr = x_t.lin_addr("_wp_t", prefix="_wp_cx")
            lines += [
                f"for (int _wp_t = _wp_lane; _wp_t < {x_t.size}; _wp_t += {self.unit.block_dim}) {{",
                *(f"    {ln};" for ln in y_un + x_un),
                f"    {xarr}[{x_addr}] = {yarr}[{y_addr}];",
                "}",
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
            ]
        if base == "tile_lower_solve":
            lines += self._solve_phase_lines(a_t, x_t, None, "fwd", a_swap=False)
        elif base == "tile_upper_solve":
            # upstream tile_solve.h: back substitution reads U(i, j) AS GIVEN
            # — a transposed operand carries the transpose in its own
            # strides, so no coordinate swap here
            lines += self._solve_phase_lines(a_t, x_t, None, "bwd", a_swap=False)
        else:  # tile_cholesky_solve: A = F F^T, F read from the fill_mode
            # triangle (upstream names the argument L). fwd solves F^T z = y
            # — reading F^T(i, j) = F(j, i), the transposed coords through
            # the input's own mapping — so the swap tracks fill_mode; bwd
            # solves F x = z through the un-transposed coords.
            upper = tmpl == "true"
            lines += self._solve_phase_lines(a_t, x_t, None, "fwd", a_swap=upper)
            lines += self._solve_phase_lines(a_t, x_t, None, "bwd", a_swap=not upper)
        return lines

    # -- tile_sort / tile_stack / tile_mesh_query_aabb (KEP-0001 M2.4 P5) ----
    #
    # Newton-closure cooperative ops. Single-rule split like every tile op:
    # launch_tiled lowers the shared-storage (threadgroup) forms below; any
    # plain launch lowers per-thread (one-lane block: a 1-element sort is the
    # identity, a stack is the thread's own serial LIFO, a mesh query is the
    # thread's own scalar iterator — exactly upstream's CPU block_dim=1
    # semantics, which is what the plain-launch goldens encode).

    def _stack_var(self, name: str, what: str) -> "StackVar":
        sv = self.stack_vars.get(name)
        if sv is None:
            raise TranslationError(f"{what}: source {name!r} is not a tile_stack variable")
        return sv

    def _stack_count_expr(self, sv: StackVar) -> str:
        if sv.tier == "shr":
            return f"_wp_tile_shr_{sv.count_slot}[0]"
        return f"{sv.name}_c"

    def _stack_data_expr(self, sv: StackVar) -> str:
        if sv.tier == "shr":
            return f"_wp_tile_shr_{sv.data_slot}"
        return f"{sv.name}_d"

    def _parse_stack_decl(self, line: str) -> list[str] | None:
        """``wp::tile_stack_t<T, C> var = wp::tile_stack_alloc<T, C>();`` —
        the tile_stack primal decl. Registers a StackVar; the cooperative tier
        claims (data, count, rank-stage) shared slots, the per-thread tier
        declares locals. Returns None when the line is not a stack decl."""
        m = re.match(
            r"^\s*wp::tile_stack_t<(.+),\s*(\d+)>\s+(var_\w+)\s*=\s*"
            r"wp::tile_stack_alloc<(.+),\s*(\d+)>\(\);\s*$",
            line,
        )
        if m is None:
            return None
        dtype_ctype, cap, name, alloc_dtype, alloc_cap = m.group(1).strip(), int(m.group(2)), m.group(3), m.group(4).strip(), int(m.group(5))
        if dtype_ctype != alloc_dtype or cap != alloc_cap:
            raise TranslationError(f"tile_stack decl/alloc template mismatch in {line!r}")
        if cap <= 0:
            raise TranslationError(f"tile_stack capacity must be positive, got {cap}")
        elem = msl_type(dtype_ctype)
        sv = StackVar("shr" if self.unit.block_dim is not None else "scalar", elem, cap, name)
        self.stack_vars[name] = sv
        if sv.tier == "shr":
            sv.data_slot = self._claim_shr(elem, cap)
            sv.count_slot = self._claim_shr("int", 1)
            sv.stage_slot = self._claim_shr("int", self.unit.block_dim)
            return []
        # per-thread tier: locals at function scope (SSA stack decls hoist to
        # the primal-vars section, so this is top scope even when the Python
        # source declares the stack inside a loop)
        return [
            f"{elem} {name}_d[{cap}];",
            f"int {name}_c = 0;",
            f"    (void)({name}_d); (void)({name}_c);  // keep -Wunused-variable diagnostics from clipping the error buffer",
        ]

    def _stack_init(self, target: str, rhs: str) -> list[str]:
        """``var = wp::tile_stack_init<T, C>();`` — count := 0 and (cooperative
        tier) the strided data zero-fill; upstream tile_stack_t operator=(int)
        (tile.h:6198). Emits the barrier every stack op pairs with."""
        m = re.match(r"^wp::tile_stack_init<(.+),\s*(\d+)>\(\)$", rhs)
        if m is None:
            raise TranslationError(f"cannot parse tile_stack_init {rhs!r}")
        sv = self._stack_var(target, "tile_stack_init")
        elem, cap = sv.elem, sv.cap
        if sv.tier == "scalar":
            return [
                f"{sv.name}_c = 0;",
                f"{{ {elem} _wp_z = {{}}; for (int _wp_i = 0; _wp_i < {cap}; ++_wp_i) {{ {sv.name}_d[_wp_i] = _wp_z; }} }}",
            ]
        self.unit.uses_tile_ops = True
        bd = self._require_block_dim("wp::tile_stack_init")
        self._check_uniform("wp::tile_stack_init")
        data = self._stack_data_expr(sv)
        return [
            f"{{ {elem} _wp_z = {{}}; for (int _wp_i = _wp_lane; _wp_i < {cap}; _wp_i += {bd}) {{ {data}[_wp_i] = _wp_z; }} }}",
            f"if (_wp_lane == 0) {{ _wp_tile_shr_{sv.count_slot}[0] = 0; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _stack_push(self, target: str, rhs: str) -> list[str]:
        """``slot = wp::tile_stack_push(stack, value, has_value);`` — every
        contributing lane claims one free slot (upstream tile.h:6247 hands out
        slots by atomic-increment order, which is lane order between barriers
        on upstream's CPU fiber runtime; the deterministic lane-rank refinement
        is exactly that CPU order, so goldens match bit-for-bit). The count
        saturates at capacity and surplus values drop with slot -1."""
        sargs = self._parse_tile_call(rhs, "tile_stack_push", False)[0]
        if len(sargs) != 3:
            raise TranslationError(f"tile_stack_push with {len(sargs)} args (needs stack, value, has_value)")
        sv = self._stack_var(sargs[0].strip(), "tile_stack_push")
        value = self.expr(sargs[1])
        has_value = self.expr(sargs[2])
        if sv.tier == "scalar":
            # one-lane block: the serial push (upstream CPU block_dim=1)
            return [
                f"{{ int _wp_slot = -1;",
                f"    if ({has_value}) {{ int _wp_i = {sv.name}_c; if (_wp_i < {sv.cap}) {{ {sv.name}_d[_wp_i] = {value}; {sv.name}_c = _wp_i + 1; _wp_slot = _wp_i; }} }}",
                f"    {target} = _wp_slot; }}",
            ]
        self.unit.uses_tile_ops = True
        self.unit.uses_live = True
        bd = self._require_block_dim("wp::tile_stack_push")
        self._check_uniform("wp::tile_stack_push")
        arr = f"_wp_tile_shr_{sv.data_slot}"
        stage = f"_wp_tile_shr_{sv.stage_slot}"
        cnt = f"_wp_tile_shr_{sv.count_slot}"
        # leading barrier (upstream: orders prior count reads before this
        # push's count write), then lane-rank staging and a serial rank walk
        return [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"{stage}[_wp_lane] = ({has_value}) ? 1 : 0;",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"{{ int _wp_base = {cnt}[0]; int _wp_rank = 0; int _wp_total = 0;",
            f"    for (int _wp_j = 0; _wp_j < _wp_live; ++_wp_j) {{ int _wp_v = {stage}[_wp_j]; if (_wp_j < _wp_lane) _wp_rank += _wp_v; _wp_total += _wp_v; }}",
            f"    int _wp_slot = _wp_base + _wp_rank;",
            f"    bool _wp_ok = ({has_value}) && (_wp_slot < {sv.cap});",
            f"    if (_wp_ok) {{ {arr}[_wp_slot] = {value}; }}",
            f"    if (_wp_lane == 0) {{ {cnt}[0] = min(_wp_base + _wp_total, {sv.cap}); }}",
            f"    {target} = _wp_ok ? _wp_slot : -1; }}",
            # trailing barrier: every lane sees the clamped count before any
            # subsequent count read or push (upstream's closing sync)
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _stack_pop(self, src: str) -> list[str]:
        """``wp::tile_stack_pop(stack, out_value, out_slot);`` — the block
        hands out up to one element per live lane from the TOP of the stack
        (later pushes pop first — upstream tile.h:6276 atomic decrement
        order, mirrored deterministically: lane r takes slot count-1-r).
        Popping an empty stack is harmless (slot -1, zero value)."""
        open_p = src.index("(")
        close = _find_balanced(src, open_p)
        sargs = split_args(src[open_p + 1 : close])
        if len(sargs) != 3:
            raise TranslationError(f"tile_stack_pop with {len(sargs)} args (needs stack, out_value, out_slot)")
        sv = self._stack_var(sargs[0].strip(), "tile_stack_pop")
        out_value, out_slot = sargs[1].strip(), sargs[2].strip()
        elem = sv.elem
        if sv.tier == "scalar":
            return [
                f"{{ int _wp_slot = -1; {elem} _wp_v = {{}};",
                f"    if ({sv.name}_c > 0) {{ {sv.name}_c -= 1; _wp_slot = {sv.name}_c; _wp_v = {sv.name}_d[_wp_slot]; }}",
                f"    {out_slot} = _wp_slot; {out_value} = _wp_v; }}",
            ]
        self.unit.uses_tile_ops = True
        self.unit.uses_live = True
        self._require_block_dim("wp::tile_stack_pop")
        self._check_uniform("wp::tile_stack_pop")
        arr = f"_wp_tile_shr_{sv.data_slot}"
        cnt = f"_wp_tile_shr_{sv.count_slot}"
        # leading barrier (upstream), then a top-down handout: hand =
        # min(count, _wp_live) elements go to lanes 0..hand-1, highest slots
        # first; the remainder stays for the next pop call
        return [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"{{ int _wp_count = {cnt}[0];",
            f"    int _wp_hand = min(_wp_count, _wp_live);",
            f"    int _wp_slot = _wp_count - 1 - _wp_lane;",
            f"    bool _wp_ok = (_wp_lane < _wp_hand);",
            f"    {out_slot} = _wp_ok ? _wp_slot : -1;",
            f"    {out_value} = _wp_ok ? {arr}[_wp_slot] : {elem}{{}};",
            f"    if (_wp_lane == 0) {{ {cnt}[0] = _wp_count - _wp_hand; }} }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _stack_clear(self, src: str) -> list[str]:
        """``wp::tile_stack_clear(stack);`` — count := 0 (upstream
        tile.h:6300; thread 0 writes, barrier brackets the write)."""
        open_p = src.index("(")
        close = _find_balanced(src, open_p)
        sargs = split_args(src[open_p + 1 : close])
        if len(sargs) != 1:
            raise TranslationError(f"tile_stack_clear with {len(sargs)} args")
        sv = self._stack_var(sargs[0].strip(), "tile_stack_clear")
        if sv.tier == "scalar":
            return [f"{sv.name}_c = 0;"]
        self.unit.uses_tile_ops = True
        self._require_block_dim("wp::tile_stack_clear")
        self._check_uniform("wp::tile_stack_clear")
        return [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"if (_wp_lane == 0) {{ _wp_tile_shr_{sv.count_slot}[0] = 0; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_sort(self, src: str) -> list[str]:
        """``wp::tile_sort(keys, values);`` — in-place ascending key-value
        sort (upstream tile.h radix sort; the observable contract is: keys
        ascending, values permuted together, equal-key order unspecified).
        Bitonic network over the shared arrays — every phase is a strided
        lane pass closed by a barrier, so each pass is one uniform
        compare-exchange round.

        Max supported tile size: the padded scratch must fit the 32 KB
        threadgroup budget shared with every other tile array — for float32
        keys + int32 payload that is n <= 4096 (two padded arrays of
        next_pow2(n)); the budget check fails loud beyond it (the launch then
        also needs the other tiles to fit). Requires power-of-two block_dim
        (validated at launch) and uniform control flow (checked).
        """
        open_p = src.index("(")
        close = _find_balanced(src, open_p)
        sargs = split_args(src[open_p + 1 : close])
        if len(sargs) != 2:
            raise TranslationError(f"tile_sort with {len(sargs)} args (needs keys, values)")
        k_t = self._tvar(sargs[0].strip(), "tile_sort")
        v_t = self._tvar(sargs[1].strip(), "tile_sort")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            # plain scalar tier: one-element tiles — the identity
            if k_t.kind != "scalar" or v_t.kind != "scalar":
                raise TranslationError("tile_sort on a plain launch lowers per-thread scalar tiles only")
            return []
        if k_t.dims != v_t.dims:
            raise TranslationError(f"tile_sort shape mismatch keys {k_t.dims} values {v_t.dims}")
        if len(k_t.dims) != 1:
            raise TranslationError(f"tile_sort over {k_t.dims}: 1-D tiles only in the closure")
        # upstream places both tiles in shared storage whatever they were
        # created with; the closure keeps the fail-loud form (the per-thread
        # scalar carve-out — e.g. a wp.tile(v) broadcast — has no array to
        # sort; declare the operands with storage="shared" like the Newton
        # SAP kernel does)
        if k_t.kind != "shr" or v_t.kind != "shr":
            raise TranslationError(
                "tile_sort operands must be shared-storage tiles (storage=\"shared\"): "
                f"got {'scalar' if k_t.kind != 'shr' else k_t.dims} keys / "
                f"{'scalar' if v_t.kind != 'shr' else v_t.dims} values"
            )
        # padding identity sorts LAST (ascending): the largest value of the
        # key type, as MSL literal tokens (INT64_MAX and friends are not in
        # the MSL environment — literals keep the padded scratch compiling)
        key_pad = _SORT_PAD.get(k_t.elem)
        if key_pad is None:
            raise TranslationError(
                f"tile_sort key element {k_t.elem!r} is outside the closure "
                "(float32/int32/uint32/int64/uint64 keys only, as upstream)"
            )
        bd = self._require_block_dim("wp::tile_sort")
        self._check_uniform("wp::tile_sort")
        n = k_t.size
        karr = f"_wp_tile_shr_{k_t.slot}"
        varr = f"_wp_tile_shr_{v_t.slot}"

        def _bitonic(arr_k: str, arr_v: str, size: int) -> list[str]:
            lines = [
                f"for (int _wp_k = 2; _wp_k <= {size}; _wp_k <<= 1) {{",
                f"    for (int _wp_j = _wp_k >> 1; _wp_j > 0; _wp_j >>= 1) {{",
                f"        for (int _wp_t = _wp_lane; _wp_t < {size}; _wp_t += {bd}) {{",
                "            int _wp_l = _wp_t ^ _wp_j;",
                "            if (_wp_l > _wp_t) {",
                "            bool _wp_asc = ((_wp_t & _wp_k) == 0);",
                f"            if (_wp_asc ? ({arr_k}[_wp_t] > {arr_k}[_wp_l]) : ({arr_k}[_wp_t] < {arr_k}[_wp_l])) {{",
                f"                {_swap_line(k_t.elem, arr_k, '_wp_t', '_wp_l')}",
                f"                {_swap_line(v_t.elem, arr_v, '_wp_t', '_wp_l')}",
                "            }",
                "            }",
                "        }",
                "        threadgroup_barrier(mem_flags::mem_threadgroup);",
                "    }",
                "}",
            ]
            return lines

        pow2 = n & (n - 1) == 0
        if pow2 and k_t.is_contig() and v_t.is_contig():
            # in place: no extra storage
            return _bitonic(karr, varr, n)
        # non-power-of-two (or strided view) tiles sort through a padded
        # scratch: keys pad with the type's max identity (sentinels like
        # Newton SAP's 1e30 sort past it untouched), payload pads with {}
        npow = 1
        while npow < n:
            npow *= 2
        ks = self._claim_shr(k_t.elem, npow)
        vs = self._claim_shr(v_t.elem, npow)
        sk, sv_arr = f"_wp_tile_shr_{ks}", f"_wp_tile_shr_{vs}"
        k_lin = k_t.lin_addr("_wp_t", prefix="_wp_sk")
        v_lin = v_t.lin_addr("_wp_t", prefix="_wp_sv")
        return [
            f"for (int _wp_t = _wp_lane; _wp_t < {npow}; _wp_t += {bd}) {{",
            *(f"    {ln};" for ln in k_lin[0] + v_lin[0]),
            f"    {sk}[_wp_t] = (_wp_t < {n}) ? {karr}[{k_lin[1]}] : {key_pad};",
            f"    {sv_arr}[_wp_t] = (_wp_t < {n}) ? {varr}[{v_lin[1]}] : {v_t.elem}{{}};",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            *_bitonic(sk, sv_arr, npow),
            f"for (int _wp_t = _wp_lane; _wp_t < {n}; _wp_t += {bd}) {{",
            *(f"    {ln};" for ln in k_lin[0] + v_lin[0]),
            f"    {karr}[{k_lin[1]}] = {sk}[_wp_t];",
            f"    {varr}[{v_lin[1]}] = {sv_arr}[_wp_t];",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _meshq_slots_for(self, qvar: str, what: str) -> tuple[int, int]:
        slots = self.meshq_slots.get(qvar)
        if slots is None:
            raise TranslationError(
                f"{what}: query {qvar!r} has no shared handoff slots "
                "(no wp::tile_mesh_query_aabb construction reached it)"
            )
        return slots

    def _meshq_construct(self, target: str, rhs: str) -> list[str]:
        """``query = wp::tile_mesh_query_aabb(id, lo, hi);`` — every lane
        builds its private wp_mesh_query_aabb_t (only lane 0's copy is ever
        advanced), and lane 0 pre-seeds the shared valid latch so the first
        ``while tile_query_valid`` test enters the loop (upstream tile_mesh.h
        pre-seeds last_valid_shared_mem the same way). One face is handed out
        per _meshq_next call — upstream's CUDA batches block_dim faces per
        call, the CPU golden hands out one; the visited-face SET is identical
        and Newton consumes faces independently, so the serial handoff (which
        matches the CPU golden bit-for-bit) is the closure lowering."""
        args, _ = self._parse_tile_call(rhs, "tile_mesh_query_aabb", False)
        if len(args) != 3:
            raise TranslationError(f"tile_mesh_query_aabb with {len(args)} args (needs id, lower, upper)")
        self.used_mesh = True
        self.unit.uses_tile_ops = True
        lo, hi = self.expr(args[1]), self.expr(args[2])
        call = f"wp_mesh_query_aabb(_wp_mesh_table[{self.expr(args[0])}], {lo}, {hi})"
        if self.unit.block_dim is None:
            # plain scalar tier: each thread drives its own scalar query
            return [f"{target} = {call};"]
        self._require_block_dim("wp::tile_mesh_query_aabb")
        self._check_uniform("wp::tile_mesh_query_aabb")
        res = self._claim_shr("int", 1)
        latch = self._claim_shr("int", 1)
        self.meshq_slots[target] = (res, latch)
        return [
            f"{target} = {call};",
            f"if (_wp_lane == 0) {{ _wp_tile_shr_{latch}[0] = 1; _wp_tile_shr_{res}[0] = -1; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _meshq_next(self, target: str, rhs: str) -> list[str]:
        """``tile = wp::tile_mesh_query_aabb_next(query);`` — lane 0 advances
        the private iterator one step and latches validity; lane 0 receives
        the face index, all other lanes receive -1 (upstream's CPU tile
        carries the index at linear slot 0 and -1 elsewhere — the per-thread
        untile of a block_dim-wide register tile reads exactly slot
        lane)."""
        args, _ = self._parse_tile_call(rhs, "tile_mesh_query_aabb_next", False)
        if len(args) != 1:
            raise TranslationError(f"tile_mesh_query_aabb_next with {len(args)} args")
        qvar = args[0].strip()
        if self.unit.block_dim is None:
            # plain scalar tier: the thread's own iterator, upstream CPU form
            return [
                f"{{ int _wp_mqi = -1; bool _wp_mqv = wp_mesh_query_aabb_next({qvar}, _wp_mqi);",
                f"    {qvar}.last_query_valid = _wp_mqv; {target} = _wp_mqi; }}",
            ]
        res, latch = self._meshq_slots_for(qvar, "wp::tile_mesh_query_aabb_next")
        self._require_block_dim("wp::tile_mesh_query_aabb_next")
        self._check_uniform("wp::tile_mesh_query_aabb_next")
        return [
            f"if (_wp_lane == 0) {{ int _wp_mqi = -1;",
            f"    bool _wp_mqv = wp_mesh_query_aabb_next({qvar}, _wp_mqi);",
            f"    _wp_tile_shr_{res}[0] = _wp_mqv ? _wp_mqi : -1;",
            f"    _wp_tile_shr_{latch}[0] = _wp_mqv ? 1 : 0; }}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"{target} = (_wp_lane == 0) ? _wp_tile_shr_{res}[0] : -1;",
            # trailing barrier before the next advance reuses the slots
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def _tile_assign_stmt(self, rhs: str) -> list[str]:
        """``wp::tile_assign(dst, src, offs...)`` — copy src into dst at the
        offset (upstream tile.h:5943). The source is snapshotted into scratch
        before any write (upstream stages to registers: dest may be a view
        aliasing src, NumPy assignment semantics), one barrier on each side.
        Plain launches bind the per-thread value (non-zero offsets cannot
        address the thread's one element and fail loud)."""
        args, _tmpl = self._parse_tile_call(rhs, "tile_assign", False)
        if len(args) < 3:
            if len(args) != 2:
                raise TranslationError(f"tile_assign with {len(args)} args (needs dst, src, offsets...)")
            d_t0 = self._tvar(args[0].strip(), "tile_assign")
            if d_t0.dims != self._tvar(args[1].strip(), "tile_assign").dims:
                raise TranslationError("tile_assign without offsets needs matching dst/src dims")
            args = args + ["0"] * len(d_t0.dims)
        d_t = self._tvar(args[0].strip(), "tile_assign")
        s_t = self._tvar(args[1].strip(), "tile_assign")
        if d_t.dims != s_t.dims:
            pass  # src must FIT in dst; full-tile assign is the closure form
        offs = args[2:]
        if len(offs) != len(d_t.dims):
            raise TranslationError(f"tile_assign needs one offset per dst dimension, got {len(offs)}")
        self.unit.uses_tile_ops = True
        if self.unit.block_dim is None:
            off_vals = [self.const_vars.get(o.strip(), o.strip()) for o in offs]
            if any(v != "0" for v in off_vals):
                raise TranslationError(
                    "wp::tile_assign with a non-zero offset on a plain launch: the per-thread "
                    "scalar tier holds one element per tile — identity assigns only"
                )
            return [f"{args[0].strip()} = {args[1].strip()};"]
        bd = self._require_block_dim("wp::tile_assign")
        self._check_uniform("wp::tile_assign")
        darr, sarr = self._shrub(d_t), self._shrub(s_t)
        if s_t.elem != d_t.elem:
            raise TranslationError(f"tile_assign elem mismatch {s_t.elem} -> {d_t.elem}")
        scratch = self._claim_shr(d_t.elem, s_t.size)
        s_un, s_addr = s_t.lin_addr("_wp_t", prefix="_wp_as")
        # dst coordinates: _wp_t is a SRC-linear index — unravel it through
        # the src shape, add the offset per axis, then map through the dst
        # strides (a view/alias dst reorders through its own mapping; the
        # src linear index must never be reused as a dst index, it lands at
        # the wrong element whenever the offset wraps a dst row).
        w_lines: list[str] = []
        rem = "_wp_t"
        if len(s_t.dims) > 1:
            div = 1
            for d in s_t.dims[1:]:
                div *= d
            for i in range(len(s_t.dims) - 1):
                coord = f"_wp_ad{i}"
                w_lines.append(f"int {coord} = {rem} / {div}")
                rem = f"({rem} - {coord} * {div})"
                div //= s_t.dims[i] if s_t.dims[i] else 1
        addr_terms = []
        for i, o in enumerate(offs):
            coord = rem if i == len(s_t.dims) - 1 else f"_wp_ad{i}"
            stride = d_t.strides[i]
            term = f"({self.expr(o)}) + {coord}"
            addr_terms.append(f"({term})" + (f" * {stride}" if stride != 1 else ""))
        # a view/alias dst owns no storage: its base selects the region of
        # the underlying source-tile array
        if d_t.base != "0":
            addr_terms.append(f"({d_t.base})")
        d_addr = " + ".join(addr_terms) if addr_terms else "0"
        return [
            # snapshot phase
            f"for (int _wp_t = _wp_lane; _wp_t < {s_t.size}; _wp_t += {bd}) {{",
            *(f"    {ln};" for ln in s_un),
            f"    _wp_tile_shr_{scratch}[_wp_t] = {sarr}[{s_addr}];",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            # write phase at the offset (a view dst reorders through its mapping)
            f"for (int _wp_t = _wp_lane; _wp_t < {s_t.size}; _wp_t += {bd}) {{",
            *(f"    {ln};" for ln in w_lines),
            f"    {darr}[{d_addr}] = _wp_tile_shr_{scratch}[_wp_t];",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

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
        m_ctor = re.match(r"^wp::mat_t<(\d+), (\d+), wp::(float32|int32)>$", key)
        if m_ctor:  # generic-size matrix constructor (FACE_NORMALS tables,
            # mujoco_warp's float 6x6 basis + int 6x2 flex-edge tables)
            msl_t = f"mat{m_ctor.group(1)}x{m_ctor.group(2)}{'f' if m_ctor.group(3) == 'float32' else 'i'}"
            ensure_mat_def(self.unit, msl_t)
            return f"make_{msl_t}({', '.join(self.expr(a) for a in args)})"
        m_vctor = re.match(r"^wp::vec_t<(\d+), wp::(float32|int32)>$", key)
        if m_vctor and int(m_vctor.group(1)) not in (
            _VEC_SIZES if m_vctor.group(2) == "float32" else _VECI_SIZES
        ):
            # generic-size vector constructor (mujoco_warp vec10 cinert/crb
            # locals `vec10()`, Newton mujoco-glue vec12 eq_data): zero-arg,
            # scalar-fill and element forms all route through the generated
            # make_ factory (KEP-0001 M3.2 package c)
            msl_t = f"vec{m_vctor.group(1)}{'f' if m_vctor.group(2) == 'float32' else 'i'}"
            ensure_vec_def(self.unit, msl_t)
            return f"make_{msl_t}({', '.join(self.expr(a) for a in args)})"
        if key == "wp::copy":
            return self.expr(args[0])
        if key == "wp::trace":
            # matrix trace (sum of diagonal): fixed sizes via the prelude,
            # generic square mats via ensure_mat_def's overload
            return f"wp_trace({self.expr(args[0])})"
        if key == "wp::eig3":
            # symmetric 3x3 eigendecomposition (KEP-0001 M3.2; prelude
            # _EIG3_OPS — upstream svd.h eig3 port). The vendored codegen
            # emits BOTH overloads (the (Q, d)-returning one and the
            # caller-output mutating one) identically: a statement call
            # carrying the outputs as trailing ref args
            # (`wp::eig3(A, Q, d);`), so one routing covers both and the
            # argument order pins the output contract (Q first, d second).
            # fp32 only — fp64 fails loud (B1: Metal has no fp64).
            if len(args) != 3:
                raise TranslationError(f"wp::eig3 with {len(args)} args (needs A, Q, d)")
            if any("float64" in a for a in args):
                raise TranslationError(
                    "wp::eig3: the fp64 overload has no MSL translation "
                    "(Metal has no fp64; decompose in float32 — see Newton "
                    "inertia.py B2 note)"
                )
            return f"wp_eig3({', '.join(self.expr(a) for a in args)})"
        if key == "wp::store":
            if len(args) == 2:
                # store through an element pointer (wp::indexref result or a
                # wp::address into an array): `wp::store(p, v)` → *p = v
                tgt = args[0].strip()
                # Inline component store through an array element address:
                # `wp::store(&(wp::index(arr, i[, j]).component_ref(k)), v)` —
                # upstream's lowering of a chained subscript store
                # (`cvel_out[w, 0][e] = 0.0`; mjw _comvel_root, Newton
                # update_axis_properties). Same element pointer the alias
                # path emits; the device-space wp_vec_ref overload picks the
                # component (negative wrap, spatial w/v dispatch).
                m_comp = re.match(
                    r"^&\(wp::index\((.*)\)\.component_ref\(([^()]+)\)\)$", tgt
                )
                if m_comp:
                    iargs = split_args(m_comp.group(1))
                    arr_var = iargs[0].strip()
                    spec = self.array_args.get(arr_var)
                    if spec is None:
                        raise TranslationError(
                            f"wp::store &(...component_ref) target on non-array arg {arr_var!r}"
                        )
                    idx_srcs = [self.expr(a) for a in iargs[1:]]
                    k_src = self.expr(m_comp.group(2))
                    return (
                        f"(wp_vec_ref(*{self._elem_addr(spec, idx_srcs)}, {k_src})"
                        f" = {self.expr(args[1])})"
                    )
                # Struct-field store through an array element address:
                # `wp::store(&(wp::index(arr, i).field), v)` — the lowering of
                # `arr[i].field = v` on a struct-typed array (viewer GL
                # fill_vertex_data / fill_line_vertex_data: `vertices[tid].pos =
                # points[tid]`). The MSL struct keeps the Python member names
                # (_emit_struct_def), so the element pointer dereferences and
                # projects the member in place. (The `.component_ref(k)` form
                # above ends in `)` before the closing paren, so it never
                # matches this member-name pattern.)
                m_field = re.match(r"^&\(wp::index\((.*)\)\.([A-Za-z_]\w*)\)$", tgt)
                if m_field:
                    iargs = split_args(m_field.group(1))
                    arr_var = iargs[0].strip()
                    spec = self.array_args.get(arr_var)
                    if spec is None:
                        raise TranslationError(
                            f"wp::store &(...field) target on non-array arg {arr_var!r}"
                        )
                    idx_srcs = [self.expr(a) for a in iargs[1:]]
                    return (
                        f"(((*{self._elem_addr(spec, idx_srcs)}).{m_field.group(2)}"
                        f" = {self.expr(args[1])}))"
                    )
                alias = self.aliases.get(tgt)
                if alias is None or not alias[0]:
                    raise TranslationError(f"wp::store target {tgt!r} is not a pointer alias")
                return f"(*({alias[1]}) = {self.expr(args[1])})"
            # register-vector element store: `wp::store(v, i, x);` (array
            # stores go through wp::array_store, which binds via data+meta)
            tgt = args[0].strip()
            return f"({tgt} = wp_vec_insert({tgt}, {self.expr(args[1])}, {self.expr(args[2])}))"
        if key == "wp::load":
            return self._load(args[0])
        if key == "wp::len":
            # array.h:2297 len(const A<T>& a) → a.shape[0]. Only the array
            # overload survives to codegen (vec/matrix/tile forms are folded
            # by static_len_value_func at adjoint build).
            spec = self.array_args.get(args[0].strip())
            if spec is None:
                raise TranslationError(f"wp::len on non-array arg {args[0]!r}")
            return f"{self._meta_ref(spec)}.shape[0]"
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
            # real wp.Mesh: read the mesh entry from the per-launch mesh
            # table (`device wp_mesh* _wp_mesh_table`, host-built — see
            # runtime/host.py; the kernel takes the binding when its call
            # graph reaches wp::mesh_get). ids index the table directly
            # (entry 0 is the null mesh); upstream reads mesh_array[id]
            # unguarded the same way.
            self.used_mesh = True
            return f"_wp_mesh_table[{self.expr(args[0])}]"
        if key in ("wp::mesh_get_bvh", "wp::mesh_get_group_root"):
            # id-semantics mesh helpers (mesh.h mesh_get_bvh/mesh_get_group_root):
            # both address the mesh's EMBEDDED BVH inside the bound table buffer,
            # which the M2.1 bvh query family reads through wp_bvh_get directly.
            if len(args) != (1 if key == "wp::mesh_get_bvh" else 2):
                raise TranslationError(f"{key} with {len(args)} args")
            bvh_addr = f"(uint64_t)(&_wp_mesh_table[{self.expr(args[0])}].bvh)"
            self.used_mesh = True
            if key == "wp::mesh_get_bvh":
                return bvh_addr
            return f"wp_bvh_get_group_root({bvh_addr}, {self.expr(args[1])})"
        if key.startswith("wp::mesh_"):
            # M2.2 query family: expand the id argument to the wp_mesh VALUE
            # read from the per-launch table, then call the prelude port
            # (upstream dereferences the id via mesh_get; the value copy the
            # C++ performs happens at the table read here). Unported upstream
            # variants (sign/winding/tracing machinery) fail loud below.
            route = _MESH_CALL_TABLE.get(key[4:])
            if route is None:
                raise TranslationError(
                    f"{key}: not in the KEP-0001 M2.2 MSL mesh-query closure "
                    "(sign/winding-number/tracing variants are ported fail-loud; "
                    "see prelude _MESH_QUERY_OPS)"
                )
            self.used_mesh = True
            if key[4:] not in _MESH_QUERY_FIRST_CALLS:
                # the aabb/sphere iterator family is query-first (mesh.h:2327-2346
                # take `mesh_query_aabb_t& query, int& index` — the mesh rides
                # inside the query by value), so arg0 passes through untouched.
                args = list(args)
                args[0] = f"_wp_mesh_table[{self.expr(args[0])}]"
            return f"{route}({', '.join(self.expr(a) for a in args)})"
        if key == "wp::tile_stack_count":
            # stack count read: a plain shared-memory read (no barrier —
            # upstream tile.h:6313 leans on the closing sync of every
            # count-writing op); safe from a single thread or an `if`, since
            # the value is uniform across the block
            sv = self._stack_var(args[0].strip(), "wp::tile_stack_count")
            return self._stack_count_expr(sv)
        if key == "wp::tile_query_valid":
            # latch read: shared slot (cooperative tier) or the per-thread
            # query's own flag (plain tier); uniform after the producing
            # op's barrier
            qvar = args[0].strip()
            if self.unit.block_dim is None:
                return f"({qvar}.last_query_valid)"
            _res, latch = self._meshq_slots_for(qvar, "wp::tile_query_valid")
            return f"(_wp_tile_shr_{latch}[0] != 0)"
        if key == "wp::tile_extract":
            # Expression-position extract. Shared-storage tiles (including
            # reduction results, which land in their own one-element shared
            # array): the read is direct — every element lives in the
            # threadgroup array, already ordered by the producing op's
            # barrier. Per-thread tiles: only the constant-0 identity is an
            # expression (non-zero extracts broadcast through a scratch slot
            # and lower at statement position only — see
            # _tile_extract_target).
            self.unit.uses_tile_ops = True
            idx_var = args[1].strip()
            idx_const = self.const_vars.get(idx_var)
            src = args[0].strip()
            tvar = self._tvar(src, "tile_extract")
            if len(args) > 2:
                # per-coordinate form tile_extract(t, i, j, ...): coords·strides
                # is a pure expression (strides are compile-time), unlike the
                # linear unravel in lin_addr
                coords = [self.expr(a) for a in args[1:]]
                if tvar.kind != "shr" or len(coords) != len(tvar.dims):
                    raise TranslationError(
                        f"wp::tile_extract with {len(coords)} coordinate indices needs a "
                        f"shared-storage rank-{len(tvar.dims)} tile"
                    )
                return f"_wp_tile_shr_{tvar.slot}[{tvar.coord_addr(coords)}]"
            if tvar.kind == "shr":
                arr = f"_wp_tile_shr_{tvar.slot}"
                if idx_const is not None:
                    k = int(idx_const)
                    if not 0 <= k < tvar.size:
                        raise TranslationError(
                            f"wp::tile_extract index {k} out of bounds for tile of {tvar.size} element(s)"
                        )
                    _lines, addr = tvar.lin_addr(str(k))
                    return f"{arr}[{addr}]"
                d_lines, d_addr = tvar.lin_addr(self.expr(idx_var), prefix="_wp_x")
                if d_lines:
                    raise TranslationError(
                        "wp::tile_extract with a runtime index on a strided view must be assigned "
                        "to a local first (the coordinate unravel emits statements)"
                    )
                return f"{arr}[{d_addr}]"
            if idx_const == "0":
                return src
            raise TranslationError(
                f"wp::tile_extract({src!r}, non-zero index) must be assigned to a local first: "
                "the scratch broadcast emits statements and only lowers at statement position"
            )
        if key in ("wp::tile_binary_map", "wp::tile_unary_map", "wp::tile_map"):
            raise TranslationError(
                "wp::tile_map is only supported as an assignment RHS (`m = wp.tile_map(op, a, b);`) — "
                "nested-in-expression maps are outside the closure"
            )
        if key == "wp::block_dim":
            # wp::block_dim() folds to the launch's compile-time block size;
            # plain launches run one-lane blocks (CPU-golden convention), so
            # the degenerate block size there is 1
            if self.unit.block_dim is None:
                return "1"
            return str(self.unit.block_dim)
        if key == "wp::bvh_get_group_root":
            # BVH host build lives in runtime/bvh.py (upstream bvh.cpp port,
            # same device struct layout — see prelude); group-root traversal
            # is ported to MSL there
            return "wp_bvh_get_group_root(" + ", ".join(self.expr(a) for a in args) + ")"
        if key == "wp::hash_grid_query":
            raise TranslationError(f"{key}: HashGrid device builtins are Phase 2 (KEP-0001); not available")
        if key == "wp::static":
            # wp.static(x): compile-time constant expression — emit the arg
            # (the `wp::Any` decl it feeds is deferred to the first assignment)
            if len(args) != 1:
                raise TranslationError(f"wp::static with {len(args)} args")
            return self.expr(args[0])
        # generic calls carry a template suffix (`wp::matrix_from_cols<wp::float32>`);
        # strip it for the plain-call table (cast/ctor tables keep the full key)
        base = key[4:].split("<", 1)[0]
        m_ext = re.match(r"^wp::extract<(\d+)>$", key)
        if m_ext and len(args) == 1 and self.var_types.get(args[0].strip(), "").startswith("wp_tuple_"):
            # `wp::extract<i>(tuple)` — tuple.h member read, lowered inline
            return f"({self.expr(args[0])}).v{m_ext.group(1)}"
        if base == "extract" and len(args) == 2 and args[1].strip() in self._slices:
            # `wp::extract<vecT>(q, slice)` — quat component slicing
            # (``q[:3]`` / ``q[:4]`` / ``q[1:3]``; upstream quat.h index
            # lowering). The slice is a compile-time constant range over the
            # quat's xyzw components; the result is a vecN value (or a bare
            # component for a single-element range).
            b, e, s = self._slices[args[1].strip()]
            if s != 1 or b < 0:
                raise TranslationError(
                    f"quat slice ({b}, {e}, {s}): only unit-step non-negative ranges are in the closure"
                )
            end = 4 if e in (-1, 4) else e
            src = self.expr(args[0])
            comps = ["x", "y", "z", "w"][b:end]
            if len(comps) == 1:
                return f"({src}.{comps[0]})"
            # quatf and vecNf are distinct MSL structs — always build the
            # vec value explicitly (a whole-quat slice is NOT an identity)
            return f"make_vec{len(comps)}f({', '.join(f'{src}.{c}' for c in comps)})"
        if base == "extract" and len(args) == 3 and args[1].strip() in self._slices:
            # `wp::extract<N>(m, slice, j)` — the `m[:, j]` column form
            # (rangefinder depth: cam_xmat[:, 2]); upstream vec.h mat_t
            # slice indexing. Only a full-range slice is in the closure:
            # (0, -1, 1) or (0, rows, 1) with the matrix's own row count
            # (the codegen spells `m[0:3, 2]` on a mat33 with the literal).
            b, e, s = self._slices[args[1].strip()]
            rows = None
            m_src = self.var_types.get(args[0].strip(), "")
            m_rows = re.match(r"^mat(\d+)x(\d+)f$", m_src)
            if m_rows:
                rows = int(m_rows.group(1))
            else:
                m_sq = re.match(r"^mat(\d+)f$", m_src)
                if m_sq:
                    # square mats spell one digit per dim (mat33f = 3x3)
                    rows = int(m_sq.group(1)[0])
            if not (b == 0 and s == 1 and (e == -1 or (rows is not None and e == rows))):
                raise TranslationError(
                    f"wp::extract over slice ({b}, {e}, {s}): only the full-range "
                    "[:, j] slice is in the closure"
                )
            return f"wp_matcol({self.expr(args[0])}, {self.expr(args[2])})"
        if base.startswith("tile_"):
            # tile-family calls (tile_mesh_query_aabb stub family etc.) ride
            # launch_tiled (block, lane) semantics like every tile construct
            self.unit.uses_tile_ops = True
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
        self._const_note(stripped)
        self._note_uniformity(stripped)

        # control flow: label/goto → while(true)/break/continue
        if m := _RE_LABEL.match(line):
            if m.group(1) == "start":
                self._block_stack.append("loop")
                return ["while (true) {"]
            if self._block_stack:
                self._block_stack.pop()
            return ["}"]
        if _RE_GOTO.search(line):
            def _repl(mo):
                return "continue;" if mo.group(1).startswith("start_") else "break;"

            out = _RE_GOTO.sub(_repl, stripped)
            # bare iter_cmp/iter_next on the range-loop condition line
            return [_RE_ITER.sub(lambda mo: f"wp_{mo.group(1)}(", out)]
        if m := _RE_IF_OPEN.match(line):
            cond = m.group(1).strip()
            # a compile-time-constant condition (the wp.constant fold — e.g.
            # steel's `if COMPUTE_DIAG != 0:`) reaches every lane of the block
            # with the same value: the branch is taken or skipped block-wide,
            # so barriers inside it cannot diverge. Next best: a provably
            # uniform runtime condition (loop indices over uniform ranges,
            # scalar params, arithmetic over those — mujoco's blocked
            # cholesky guards tile_assign with `end == matrix_size`); warp's
            # own uniformity rule admits tile ops there. Only genuinely
            # divergent runtime conditions mark the scope non-uniform.
            if self._const_eval(cond) is not None:
                marker = "if-const"
            elif self._is_uniform_expr(cond):
                marker = "if-uniform"
            else:
                marker = "if"
            self._block_stack.append(marker)
            # array truthiness (`if arr:` — upstream emits `if (array_var)`
            # via array_t's non-null data check; warp-metal binds a real
            # buffer even for None args, so truthiness reads the meta's
            # ndim — 0 only for a null (None) array binding; the `!` form
            # (upstream's else-branch lowering) negates it)
            bare = cond[1:].strip() if cond.startswith("!") else cond
            if bare in self.array_args:
                spec = self.array_args[bare]
                test = f"{self._meta_ref(spec)}.ndim != 0"
                if cond.startswith("!"):
                    test = f"!({test})"
                return [f"if ({test}) {{"]
            return [f"if ({self.expr(m.group(1))}) {{"]
        if _RE_ELSE.match(line):
            return ["} else {"]
        if _RE_BRACE_CLOSE.match(line):
            if self._block_stack:
                self._block_stack.pop()
            return ["}"]
        if m := _RE_RETURN.match(line):
            tv = self.tile_vars.get(m.group(1).strip())
            if tv is not None and tv.kind == "shr":
                # tile-returning @wp.func epilogue: the value lives in the
                # callee's claimed shared slot — nothing to move by MSL value
                return ["return;"]
            return [f"return {self.expr(m.group(1))};".replace("return ;", "return;")]

        # tile_stack decl (`wp::tile_stack_t<T, C> var = wp::tile_stack_alloc<
        # T, C>();`): claims the cooperative slots (launch_tiled) or declares
        # the per-thread locals (plain launch); init arrives with the
        # wp::tile_stack_init assignment
        out = self._parse_stack_decl(line)
        if out is not None:
            return out

        # tile value decls (`wp::tile_register_t<T, layout> var_N = ...{};` /
        # `wp::tile_shared_t<T, layout, B> var_N = wp::tile_alloc_empty<...>();`):
        # shared tiles and register tiles whose shape is not the per-thread
        # one-element form are threadgroup-backed (the kernel preamble owns
        # the array; the decl itself emits nothing), `wp.tile(v)` broadcasts
        # keep the M1 per-thread element value
        m = _RE_TILE_DECL.match(line)
        if m:
            kind_word, template, name = m.group(1), m.group(2), m.group(3)
            elem = msl_type(split_args(template)[0])
            dims = _tile_shape_dims(split_args(template)[1])
            self.unit.uses_tile_ops = True
            init = m.group(4)
            if init is not None and init.strip() == "nullptr":
                # non-owning alias decl (``= nullptr``): a tile_transpose /
                # tile_reshape / tile_view result — upstream Owner=false shared
                # view. No storage is claimed; strides may differ from
                # row-major (a transposed layout), and the producing
                # assignment binds the slot/base to the source tile's array.
                stride_m = re.search(r"wp::tile_stride_t<([\d,\s]+)>", template)
                strides = (
                    [int(s) for s in stride_m.group(1).replace(" ", "").split(",") if s]
                    if stride_m
                    else None
                )
                if self.unit.block_dim is None:
                    self.tile_vars[name] = TileVar("scalar", elem, dims=dims)
                else:
                    size = 1
                    for d in dims:
                        size *= d
                    self.tile_vars[name] = TileVar("shr", elem, slot=-1, size=size, dims=dims, strides=strides)
                return []
            if kind_word == "register" and (
                self.unit.block_dim is None or dims == [self.unit.block_dim]
            ):
                # per-thread 1-element register form (wp.tile(v) broadcast and
                # friends). Plain launches without block_dim keep the M1
                # per-thread degenerate lowering for every register tile —
                # plain-launch tile semantics stay untouched.
                self.tile_vars[name] = TileVar("scalar", elem, dims=dims)
                return [f"{elem} {name} = {{}};"]
            if kind_word == "shared" and self.unit.block_dim is None:
                # plain launch without block_dim: requested shared storage
                # degrades to the per-thread scalar form like every other
                # tile (one-lane blocks — the thread holds its own element).
                # Shared storage is meaningless without a defined block
                # layout, and per-thread semantics are what the plain-launch
                # CPU goldens were generated with.
                self.tile_vars[name] = TileVar("scalar", elem, dims=dims)
                return [f"{elem} {name} = {{}};"]
            size = 1
            for d in dims:
                size *= d
            slot = self._claim_shr(elem, size)
            self.tile_vars[name] = TileVar("shr", elem, slot, size, dims)
            return []

        # plain assignment `name = expr;` (targets may be struct members:
        # `var_contact_data.field = ...`) — must be tried before the decl
        # regex, whose lazy `(\S.*?)` can otherwise swallow `ret_0 =` as a
        # "type" on multi-output ref-assign lines (T & ret_0 params)
        if m := re.match(r"^\s*(\w+(?:\.\w+)*)\s*=\s*(.+?);\s*$", line):
            target, rhs = m.group(1), m.group(2)
            if rhs.startswith("wp::slice_t("):
                # `var = wp::slice_t(begin, end, step);` — record for the
                # sliced wp::extract; emits nothing (pure metadata)
                aargs = split_args(rhs[len("wp::slice_t(") : -1])
                vals = tuple(self._const_eval(a) for a in aargs)
                if len(vals) != 3 or any(v is None for v in vals):
                    raise TranslationError(f"wp::slice_t with non-constant bounds {rhs!r}")
                self._slices[target] = vals
                return []
            if rhs.startswith("wp::address(") or rhs.startswith("&("):
                self._alias(target, rhs)
                return []  # alias collapsed: no MSL line, no decl (pointer decl was dropped)
            if rhs.startswith("wp::indexref("):
                # `p = wp::indexref(base, idx);` — pointer to element idx of
                # a vector VALUE the base aliases (SSA form of `v[i] = x`
                # through a `&((struct).vecfield)` alias); consumed by the
                # 2-arg wp::store / wp::load
                iargs = split_args(rhs[len("wp::indexref(") : -1])
                base = iargs[0].strip()
                alias = self.aliases.get(base)
                if alias is None or alias[0]:
                    raise TranslationError(f"wp::indexref base {base!r} is not a value alias")
                idx_src = self.expr(iargs[1])
                self.aliases[target] = (True, f"(&wp_vec_ref({alias[1]}, {idx_src}))")
                return []
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
            m_copy = re.match(r"^wp::copy\((\w+)\)$", rhs)
            if m_copy and m_copy.group(1) in self.array_args:
                # `t = wp::copy(arr);` — array ref copy (the SSA materializes
                # struct-field arrays through &field → wp::load → wp::copy
                # chains, e.g. `edges = provider.support_neighbors`)
                self.array_args[target] = self.array_args[m_copy.group(1)]
                return []
            m_where = re.match(r"^wp::where\((.*)\)$", rhs)
            if m_where:
                wargs = split_args(m_where.group(1))
                if (
                    len(wargs) == 3
                    and wargs[1].strip() in self.array_args
                    and wargs[2].strip() in self.array_args
                ):
                    # phi select between two array bindings
                    # (`arr_a if cond else arr_b`): materialize a local
                    # (data, meta) pair the downstream wp::address reads
                    cond = self.expr(wargs[0])
                    sa = self.array_args[wargs[1].strip()]
                    sb = self.array_args[wargs[2].strip()]
                    spec = ArgSpec(target, "array", sa.ctype)
                    spec.local = True
                    self.array_args[target] = spec
                    elem = array_elem_msl(sa.ctype)
                    self.hoisted.extend([
                        f"wp_arr_meta {target}_m = {{}};",
                        f"device {elem}* {target}_data = (device {elem}*)0;",
                    ])
                    return [
                        f"{target}_m = {self._meta_ref(sa)};",
                        f"{target}_data = {sa.name}_data;",
                        f"if (!({cond})) {{ {target}_m = {self._meta_ref(sb)}; {target}_data = {sb.name}_data; }}",
                    ]
            if rhs.startswith("wp::fixedarray_t<"):
                # in-device wp.zeros(shape=(N,), dtype=T): a fixed-size stack
                # array. Lowered to a slice of the per-thread arena (each site
                # has a unique static byte offset, so nested live allocations
                # can't collide) plus a zero-fill and a local (data, meta)
                # binding — see KernelUnit.arena_next_off.
                open_p = rhs.index("(")
                m_fa = re.match(r"^wp::fixedarray_t<(\d+),\s*(.+?)>$", rhs[:open_p].strip())
                if not m_fa:
                    raise TranslationError(f"cannot parse fixedarray ctor {rhs!r}")
                n, dtype_ctype = int(m_fa.group(1)), m_fa.group(2).strip()
                close = _find_balanced(rhs, open_p)
                if rhs[close + 1 :].strip() or len(split_args(rhs[open_p + 1 : close])) > 1:
                    raise TranslationError(f"fixedarray ctor with unexpected args: {rhs!r}")
                elem = msl_type(dtype_ctype)
                ebytes = _ELEM_BYTES.get(elem)
                if ebytes is None:
                    raise TranslationError(f"no _ELEM_BYTES entry for {elem!r} ({rhs!r})")
                spec = ArgSpec(target, "array", f"wp::array_t<{dtype_ctype}>")
                spec.local = True
                self.array_args[target] = spec
                self.used_arena = True
                off = self.unit.arena_next_off
                self.unit.arena_next_off += (n * ebytes + 15) & ~15
                self.hoisted.extend([
                    f"wp_arr_meta {target}_m = {{}};",
                    f"device {elem}* {target}_data = (device {elem}*)0;",
                ])
                return [
                    f"{target}_data = (device {elem}*)(_wp_arena + {off});",
                    f"{target}_m.offset = 0; {target}_m.ndim = 1;",
                    f"{target}_m.shape[0] = {n}; {target}_m.shape[1] = 0; {target}_m.shape[2] = 0; {target}_m.shape[3] = 0;",
                    f"{target}_m.strides[0] = {ebytes}; {target}_m.strides[1] = 0; {target}_m.strides[2] = 0; {target}_m.strides[3] = 0;",
                    f"{{ {elem} _wp_z = {{}}; for (int _wp_i = 0; _wp_i < {n}; ++_wp_i) {{ {target}_data[_wp_i] = _wp_z; }} }}",
                ]
            if rhs.startswith("wp::array_t<"):
                # `v = wp.array(ptr=..., shape=..., dtype=T)`: a view over an
                # existing device address. The SSA passes the address as an
                # integer value (via .data pointer loads, see the .ptr expr
                # hook) plus shape scalars; lower to a local (data, meta)
                # pair with byte strides.
                open_p = rhs.index("(")
                m_at = re.match(r"^wp::array_t<(.+)>$", rhs[:open_p].strip())
                if not m_at:
                    raise TranslationError(f"cannot parse array_t ctor {rhs!r}")
                dtype_ctype = m_at.group(1).strip()
                close = _find_balanced(rhs, open_p)
                cargs = split_args(rhs[open_p + 1 : close])
                if len(cargs) < 2:
                    raise TranslationError(f"array_t ctor needs (ptr, shape...): {rhs!r}")
                elem = msl_type(dtype_ctype)
                ebytes = _ELEM_BYTES.get(elem)
                if ebytes is None:
                    raise TranslationError(f"no _ELEM_BYTES entry for {elem!r} ({rhs!r})")
                nd = len(cargs) - 1
                spec = ArgSpec(target, "array", f"wp::array_t<{dtype_ctype}>")
                spec.local = True
                self.array_args[target] = spec
                self.hoisted.extend([
                    f"wp_arr_meta {target}_m = {{}};",
                    f"device {elem}* {target}_data = (device {elem}*)0;",
                ])
                lines = [
                    f"{target}_data = (device {elem}*)(uintptr_t)({self.expr(cargs[0])});",
                    f"{target}_m.offset = 0; {target}_m.ndim = {nd};",
                ]
                for i in range(4):
                    if i < nd:
                        lines.append(f"{target}_m.shape[{i}] = (int)({self.expr(cargs[1 + i])});")
                    else:
                        lines.append(f"{target}_m.shape[{i}] = 0;")
                # byte strides, last dim contiguous (row-major, as upstream)
                lines.append(f"{target}_m.strides[{nd - 1}] = {ebytes};")
                for i in range(nd - 2, -1, -1):
                    lines.append(f"{target}_m.strides[{i}] = {target}_m.strides[{i + 1}] * {target}_m.shape[{i + 1}];")
                for i in range(nd, 4):
                    lines.append(f"{target}_m.strides[{i}] = 0;")
                return lines
            m_dataptr = re.match(r"^\((wp::uint64)\*\)&\((\w+)\.data\)$", rhs)
            if m_dataptr and m_dataptr.group(2) in self.array_args:
                # `(wp::uint64*)&(stack_array.data)`: SSA materialization of
                # `stack_array.ptr` — the address OF the data pointer. For an
                # arena-backed stack array that address IS the data pointer,
                # so this collapses to a value alias of the raw address (the
                # subsequent wp::load then yields the uint64 address for
                # wp.array(ptr=...) views).
                spec = self.array_args[m_dataptr.group(2)]
                self.aliases[target] = (False, f"((uint64_t)((device char*){self._data_expr(spec)}))")
                return []
            if rhs.startswith("wp::tuple("):
                # `t = wp::tuple(x);` shape-tuple temp: a value alias (only
                # fixedarray/array_t ctors consume these, which read the
                # template + shape scalars, not the tuple itself). Multi-arg
                # ctors are VALUE tuples (tuple.h tuple_t — the mjw octree
                # weight triples): a real struct with a make_ ctor
                targs = split_args(rhs[len("wp::tuple(") : -1])
                if len(targs) > 1:
                    elems = []
                    for a in targs:
                        e = self.var_types.get(a.strip())
                        if e is None:
                            raise TranslationError(f"wp::tuple member {a.strip()!r} has no tracked type")
                        elems.append(e)
                    tname = ensure_tuple_def(self.unit, elems)
                    prev = self.var_types.get(target)
                    self.var_types[target] = tname
                    # A dropped/never-emitted all-int tuple decl (shape-tuple
                    # heuristic above; mjw _dof_chain_advance's Tuple[int, int])
                    # leaves this assignment with no declaration in scope —
                    # MSL "use of undeclared identifier". Merge decl+init.
                    if target in self._tuple_decl_dropped or prev != tname:
                        self._tuple_decl_dropped.discard(target)
                        return [f"{tname} {target} = make_{tname}({', '.join(self.expr(a) for a in targs)});"]
                    return [f"{target} = make_{tname}({', '.join(self.expr(a) for a in targs)});"]
                inner = rhs[len("wp::tuple(") : -1].strip()
                self.aliases[target] = (False, self.expr(inner))
                return []
            if target in self.deferred_any:
                # first assignment to a `wp::Any` temp (wp.static lowering):
                # declare with a deduced `auto` type at the point of init
                self.deferred_any.discard(target)
                return [f"const auto {target} = {self.expr(rhs)};"]
            if rhs.startswith("wp::tile_alloc_empty<"):
                # tile reservation (`var = wp::tile_alloc_empty<...>();`): the
                # threadgroup array (or per-thread value from the decl) is
                # already allocated — the fill arrives with the producing op
                self.unit.uses_tile_ops = True
                return []
            if rhs.startswith("wp::tile_extract("):
                out = self._tile_extract_target(target, rhs)
                if out is not None:
                    return out
                # direct forms ride the expression path below
            if rhs.startswith("wp::tile_load<"):
                return self._tile_load(target, rhs)
            if rhs.startswith("wp::tile_load_indexed<"):
                return self._tile_load_indexed(target, rhs)
            if rhs.startswith(("wp::tile_binary_map(", "wp::tile_unary_map(", "wp::tile_map(")):
                return self._tile_map(target, rhs)
            if rhs.startswith(("wp::tile_reduce(", "wp::tile_sum(", "wp::tile_argmin(", "wp::tile_argmax(")):
                return self._tile_reduce(target, rhs)
            if m_mmx := re.match(r"^wp::tile_(min|max)\(", rhs):
                # two homonyms: the 1-arg convenience reduce form and the
                # adjoint's folded binary map (tile⊗tile/tile⊗scalar) —
                # dispatch on arity, the reduce branch first (it must win
                # over the binop matcher below for the 1-arg shape)
                open_p = rhs.index("(")
                close = _find_balanced(rhs, open_p)
                if len(split_args(rhs[open_p + 1 : close])) == 1:
                    return self._tile_reduce(target, rhs)
                return self._tile_binop(target, rhs, m_mmx.group(1))
            if m_tilebin := re.match(r"^wp::tile_(mul|add|sub|div|mod)\(", rhs):
                return self._tile_binop(target, rhs, m_tilebin.group(1))
            if rhs.startswith("wp::tile<"):
                return self._tile_broadcast(target, rhs)
            if rhs.startswith("wp::tile_zeros<"):
                return self._tile_zeros(target, rhs)
            if rhs.startswith("wp::tile_arange<"):
                return self._tile_arange(target, rhs)
            if rhs.startswith("wp::tile_ones<"):
                return self._tile_ones(target, rhs)
            if rhs.startswith(("wp::tile_transpose(", "wp::tile_reshape<", "wp::tile_view<", "wp::tile_broadcast<")):
                op = (
                    "tile_transpose"
                    if rhs.startswith("wp::tile_transpose(")
                    else (
                        "tile_reshape"
                        if rhs.startswith("wp::tile_reshape<")
                        else ("tile_broadcast" if rhs.startswith("wp::tile_broadcast<") else "tile_view")
                    )
                )
                return self._tile_alias_bind(target, rhs, op)
            if rhs.startswith(("wp::tile_matmul_acc(", "wp::tile_matmul(")):
                # the SSA names every tile_matmul form tile_matmul_acc — the
                # alpha/beta split travels in the arguments
                return self._tile_matmul(target, rhs, acc_name=rhs.startswith("wp::tile_matmul_acc("))
            if rhs.startswith("wp::tile_diag_add("):
                return self._tile_diag_add(target, rhs)
            if rhs.startswith(("wp::tile_scan_inclusive(", "wp::tile_scan_exclusive(")):
                return self._tile_scan(target, rhs, inclusive=rhs.startswith("wp::tile_scan_inclusive("))
            if rhs.startswith("wp::tile_sum_axis<"):
                return self._tile_sum_axis(target, rhs)
            if rhs.startswith("wp::tile_cholesky<"):
                return self._tile_cholesky(target, rhs)
            if rhs.startswith(
                (
                    "wp::tile_lower_solve(",
                    "tile_lower_solve(",
                    "wp::tile_upper_solve(",
                    "tile_upper_solve(",
                    "wp::tile_cholesky_solve<",
                    "tile_cholesky_solve<",
                )
            ):
                # lower/upper_solve reach the SSA without the wp:: namespace;
                # cholesky_solve keeps it
                return self._tile_solve(target, rhs)
            if rhs.startswith("wp::tile_stack_init<"):
                return self._stack_init(target, rhs)
            if rhs.startswith("wp::tile_stack_push("):
                return self._stack_push(target, rhs)
            if rhs.startswith("wp::tile_mesh_query_aabb("):
                return self._meshq_construct(target, rhs)
            if rhs.startswith("wp::tile_mesh_query_aabb_next("):
                return self._meshq_next(target, rhs)
            if rhs.startswith("wp::view("):
                # `v = wp::view(parent, i0[, i1...]);` — an array-prefix
                # slice (``parent[i]`` / ``parent[i, j]`` on an N-D array
                # yields an (N-k)-D view over the parent's buffer). A local
                # (data, meta) pair derived from the parent's runtime meta;
                # the data pointer advances by one full outer slice of BYTE
                # strides per index. mjw _mark_tactile_contacts views
                # weld_geom_seen_out[worldid, tactile_idx] (array3d → row)
                # for its atomic_or.
                vargs = split_args(rhs[len("wp::view(") : -1])
                if len(vargs) < 2:
                    raise TranslationError(f"wp::view with {len(vargs)} args: needs (parent, indices...)")
                if len(vargs) > 4:
                    raise TranslationError(f"wp::view with {len(vargs)} indices: exceeds the 4-slot meta")
                parent = vargs[0].strip()
                pspec = self.array_args.get(parent)
                if pspec is None:
                    raise TranslationError(f"wp::view on non-array {parent!r}")
                n = len(vargs) - 1
                elem = array_elem_msl(pspec.ctype)
                nspec = ArgSpec(target, "array", f"wp::array_t<{elem}>")
                nspec.local = True
                self.array_args[target] = nspec
                pref = self._meta_ref(pspec)
                # Upstream array.h view(): one full outer slice per index —
                # an i * strides[k] BYTE advance (meta strides mirror
                # array_t's byte strides, the same units wp_address uses),
                # with negative indices wrapped Python-style. A shape-product
                # advance is wrong for every ndim that lacks trailing dims: on
                # a 2-D parent it collapses to shape[1] * 0 == 0 and pins
                # every slice to row 0.
                idx_exprs = [self.expr(v) for v in vargs[1:]]
                advance = " + ".join(
                    f"(size_t)(((int)({ie}) < 0 ? (int)({ie}) + {pref}.shape[{k}] : (int)({ie})) * ({pref}.strides[{k}]))"
                    for k, ie in enumerate(idx_exprs)
                )
                self.hoisted.extend([
                    f"wp_arr_meta {target}_m = {{}};",
                    f"device {elem}* {target}_data = (device {elem}*)0;",
                ])
                lines = [
                    (
                        f"{target}_data = (device {elem}*)((device char*)({self._data_expr(pspec)}) + {advance});"
                    ),
                    f"{target}_m.offset = {pref}.offset; {target}_m.ndim = {pref}.ndim - {n};",
                ]
                # meta shifts left by n dims; vacated slots zero (the
                # 4-slot meta is dense from slot 0 for every ndim ≤ 4)
                for slot in range(4):
                    src_slot = slot + n
                    shape_v = f"{pref}.shape[{src_slot}]" if src_slot < 4 else "0"
                    lines.append(f"{target}_m.shape[{slot}] = {shape_v};")
                for slot in range(4):
                    src_slot = slot + n
                    stride_v = f"{pref}.strides[{src_slot}]" if src_slot < 4 else "0"
                    lines.append(f"{target}_m.strides[{slot}] = {stride_v};")
                return lines
            m_cp = re.match(r"^wp::copy\((\w+)\)$", rhs)
            base = m_cp.group(1) if m_cp else (rhs.strip() if re.match(r"^\w+$", rhs.strip()) else None)
            if rhs.startswith("wp::mesh_get(") or (base is not None and base in self.mesh_vars):
                # wp.Mesh value (mesh_get result or a value copy): members
                # bind as real arrays (see the _alias member path)
                self.mesh_vars.add(target)
            m_uc = re.match(r"^([A-Za-z_]\w*)\(", rhs)
            ufn = self.user_funcs.get(m_uc.group(1)) if m_uc else None
            if ufn is not None and getattr(ufn, "tile_return", None) is not None:
                # `var = tile_returning_fn(args);` — the value travels in the
                # callee's claimed shared slot: emit the bare call and rebind
                # the target as a shared tile view of that slot (upstream
                # tile_shared_t by-value copy collapses to slot ownership;
                # clone-per-site keeps the slot unique per instantiation)
                elem, dims, slot = ufn.tile_return
                size = 1
                for d in dims:
                    size *= d
                # owner tile: row-major element strides (upstream
                # tile_shared_t owner layout)
                strides = []
                acc = 1
                for d in reversed(dims):
                    strides.append(acc)
                    acc *= d
                strides.reverse()
                tgt = self.tile_vars.get(target)
                if tgt is None:
                    self.tile_vars[target] = TileVar(
                        "shr", elem, slot=slot, size=size, dims=dims, strides=strides
                    )
                else:
                    tgt.kind = "shr"
                    tgt.elem = elem
                    tgt.slot = slot
                    tgt.size = size
                    tgt.dims = list(dims)
                    tgt.strides = strides
                    tgt.base = "0"
                self.unit.uses_tile_ops = True
                return [f"{self._call_user(rhs)};"]
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
            if _RE_FIXEDARRAY_T.match(ctype.strip()):
                # in-device wp.zeros() decl: the fixedarray ctor ASSIGN
                # registers the (arena data, meta) pair — drop the decl
                return []
            if ctype.strip().startswith("wp::tuple_t<"):
                inner_elems = split_args(ctype.strip()[len("wp::tuple_t<") : -1])
                if all(e.strip() in ("wp::int32", "wp::int64", "wp::uint32") for e in inner_elems):
                    # shape-tuple temp decl: `t = wp::tuple(x);` registers the
                    # value alias — drop the decl. All-int tuples can ALSO be
                    # genuine value tuples (mjw _dof_chain_advance's
                    # Tuple[int, int]), and the decl line is the only way to
                    # tell: the multi-arg `wp::tuple(a, b)` ctor below checks
                    # this set and re-emits decl+init when the drop starved
                    # it of a declaration.
                    self._tuple_decl_dropped.add(name)
                    return []
                # value tuple (tuple.h tuple_t): a declared struct local
                # (mjw find_oct's vec8 weight triples)
                tname = ensure_tuple_def(self.unit, [msl_type(e.strip()) for e in inner_elems])
                self.var_types[name] = tname
                return [f"{tname} {name};"]
            if ctype.strip() == "wp::Any":
                # wp.static temps: type-erased in the SSA; declared as
                # `const auto` at the first (only) assignment instead
                self.deferred_any.add(name)
                return []
            if ctype.strip() == "wp::str":
                # string-literal temps feed only wp.print, which is elided
                return []
            if ctype.strip() == "wp::slice_t":
                # bare slice decl: the bounds arrive with the
                # wp::slice_t(begin, end, step) assignment — pure metadata
                return []
            if not ctype:
                raise TranslationError(f"EMPTY TYPE on line {line!r}")
            try:
                msl_t = msl_type(ctype)
            except TranslationError as e:
                raise TranslationError(f"decl line {line!r}: {e}")
            self.var_types[name] = msl_t
            if _RE_GEN_MAT.match(msl_t):
                ensure_mat_def(self.unit, msl_t)
            elif _RE_GEN_VEC.match(msl_t):
                ensure_vec_def(self.unit, msl_t)
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
                return [
                    f"{'const ' if const else ''}{msl_t} {name} = {{{elems}}};",
                    f"    (void)({name});  // keep -Wunused-variable diagnostics from clipping the error buffer",
                ]
            if init is not None:
                if const and re.fullmatch(r"-?\d+(\.\d+)?", init.strip()):
                    # compile-time SSA constant (e.g. the `var_8 = 0` index
                    # behind tile_extract's [0]) — recorded for constant-fold
                    self.const_vars[name] = init.strip()
                return [
                    f"{'const ' if const else ''}{msl_t} {name} = {self.expr(init)};",
                    f"    (void)({name});  // keep -Wunused-variable diagnostics from clipping the error buffer",
                ]
            # zero-init: Metal promotes -Wsometimes-uninitialized to an error
            # on phi-merged temps; SSA assigns before every reachable read,
            # so a static zero is never observable. `{}` value-initializes
            # every MSL type here (scalars, vectors, user structs — a struct
            # def may not even be registered yet when a leaf function creates
            # a local of it, so name-based struct detection is unreliable).
            return [
                f"{'const ' if const else ''}{msl_t} {name} = {{}};",
                f"    (void)({name});  // keep -Wunused-variable diagnostics from clipping the error buffer",
            ]

        if stripped.startswith("wp::tile_scatter_masked("):
            return self._tile_scatter(stripped)
        if stripped.startswith("wp::tile_scatter_add("):
            return self._tile_scatter_add(stripped)
        if stripped.startswith("wp::tile_add_inplace("):
            return self._tile_add_inplace(stripped)
        if stripped.startswith("wp::tile_sort("):
            # in-place key-value sort (the SSA strips the trailing semicolon
            # below; the parser wants the bare call)
            return self._tile_sort(stripped.removesuffix(";"))
        if stripped.startswith("wp::tile_stack_pop("):
            return self._stack_pop(stripped.removesuffix(";"))
        if stripped.startswith("wp::tile_stack_clear("):
            return self._stack_clear(stripped.removesuffix(";"))
        if stripped.startswith("wp::tile_store<"):
            return self._tile_store(stripped)
        if stripped.startswith(("wp::tile_matmul_acc(", "wp::tile_matmul(")):
            # in-place GEMM (alpha/beta in the args; the SSA names every form
            # tile_matmul_acc); strip the statement's trailing semicolon — the
            # parsers below expect a bare call
            src = stripped.removesuffix(";")
            return self._tile_matmul(None, src, acc_name=src.startswith("wp::tile_matmul_acc("))
        if stripped.startswith("wp::tile_assign("):
            return self._tile_assign_stmt(stripped.removesuffix(";"))
        if stripped.startswith("wp::tile_cholesky_inplace<"):
            return self._tile_cholesky(None, stripped.removesuffix(";"))
        if stripped.startswith(("wp::tile_cholesky_solve_inplace<", "tile_cholesky_solve_inplace<")):
            return self._tile_solve(None, stripped.removesuffix(";"))
        if stripped.startswith(
            (
                "tile_lower_solve_inplace(",
                "wp::tile_lower_solve_inplace(",
                "tile_upper_solve_inplace(",
                "wp::tile_upper_solve_inplace(",
            )
        ):
            # lower/upper in-place solves reach the SSA without the wp:: namespace
            return self._tile_solve(None, stripped.removesuffix(";"))

        m_ca = re.match(r"^\s*(\w+(?:\.\w+)*)\s*([+\-*/])=\s*(.+?);\s*$", line)
        if m_ca:
            # compound assignment (Newton: `iteration += 1` counters)
            t, op, r = m_ca.group(1), m_ca.group(2), m_ca.group(3)
            return [f"{t} = {t} {op} ({self.expr(r)});"]

        if m := re.match(r"^\s*(.+?);\s*$", line):
            body = m.group(1)
            # tile-to-tile move (loop-carried SSA phi, mjw blocked cholesky):
            # route before call_stmt — the copy needs multi-line emission
            m_ta = re.match(r"^wp::assign\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$", body)
            if m_ta and m_ta.group(1) in self.tile_vars and m_ta.group(2) in self.tile_vars:
                return self._tile_assign_stmt(f"wp::tile_assign({m_ta.group(1)}, {m_ta.group(2)})")
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
            expr = f"wp_address({self._data_expr(spec)}, {self._meta_ref(spec)}{', ' + idx if idx else ''})"
            self.aliases[var] = (True, expr)
            return
        # &(expr) reference forms
        inner = init[2:-1].strip()
        # &(wp::index(arr, i[, j...])): address of an indexed element —
        # upstream's lowering of chained subscript stores
        # (`cvel_out[w, 0][e] = x`: take the element address, store through
        # it). Same element pointer the wp::address branch emits; the
        # prelude wp_address template carries the meta byte math.
        if inner.startswith("wp::index(") and inner.endswith(")"):
            args = split_args(inner[len("wp::index(") : -1])
            arr_var = args[0].strip()
            spec = self.array_args.get(arr_var)
            if spec is None:
                raise TranslationError(f"&(...wp::index(...)) on non-array arg {arr_var!r}")
            idx = ", ".join(self.expr(a) for a in args[1:])
            expr = f"wp_address({self._data_expr(spec)}, {self._meta_ref(spec)}{', ' + idx if idx else ''})"
            self.aliases[var] = (True, expr)
            return
        shape_of = re.match(r"^\(?(var_\w+)\.shape\)?$", inner)
        if shape_of:
            spec = self.array_args.get(shape_of.group(1))
            if spec is None:
                raise TranslationError(f"&(...shape) on non-array arg {inner!r}")
            s = f"{self._meta_ref(spec)}.shape"
            expr = f"int4({s}[0], {s}[1], {s}[2], {s}[3])"
            self.aliases[var] = (False, expr)
            return
        # struct member reference &((var).field) / &(var.field)
        member = re.match(r"^(?:\()?(\w+)(?:\))?((?:\.\w+)+)$", inner)
        if not member:
            # struct-array element member address: &(((*wp::address(arr, idx))).field)
            # — e.g. in_gaussians[g].num_points; lowers through the array
            # alias machinery (element pointer, then member projection)
            elem_member = re.match(r"^\(\(\*wp::address\(([^()]+)\)\)\)((?:\.\w+)+)$", inner)
            if elem_member:
                addr_args = split_args(elem_member.group(1))
                spec = self.array_args.get(addr_args[0].strip())
                if spec is None:
                    raise TranslationError(f"&(...address(...).field) on non-array arg {addr_args[0]!r}")
                idx = ", ".join(self.expr(i) for i in addr_args[1:])
                addr = f"wp_address({self._data_expr(spec)}, {self._meta_ref(spec)}{', ' + idx if idx else ''})"
                self.aliases[var] = (True, f"(&(((*({addr}))){elem_member.group(2)}))")
                return
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
        m_mesh = re.match(r"^(var_\w+)\.(points|velocities|indices)$", key)
        if m_mesh and m_mesh.group(1) in self.mesh_vars:
            # real wp.Mesh member array (device pointers inside the mesh
            # struct — see prelude wp_mesh / runtime/host.py; velocities
            # joined with M2.2's full Mesh struct)
            field = m_mesh.group(2)
            if field == "indices":
                elem_ctype = "wp::int32"
            else:
                elem_ctype = "wp::vec_t<3, wp::float32>"
            spec = ArgSpec(var, "array", f"wp::array_t<{elem_ctype}>")
            spec.mesh_member = (m_mesh.group(1), field)
            self.array_args[var] = spec
            return
        self.aliases[var] = (False, self.expr(inner))

    def call_stmt(self, src: str) -> str:
        src = src.strip()
        m = re.match(r"^builtin_tid([1234])d\s*\((.*)\)$", src)
        if m:
            return self._builtin_tid(m.group(1), m.group(2))
        if re.match(r"^builtin_block_dim\s*\(\s*\)$", src):
            # wp.block_dim() folds to the launch's compile-time block size
            if self.unit.block_dim is None:
                raise TranslationError("wp.block_dim() requires an explicit block_dim launch")
            return str(self.unit.block_dim)
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
            if len(args) == 2 and args[0].strip() in self.tile_vars and args[1].strip() in self.tile_vars:
                raise TranslationError("tile-to-tile wp::assign reached call_stmt (stmt routing bug)")
            if len(args) >= 3 and args[0].strip() in self.tile_vars:
                # tile element store (`t[i, j] = v` / `t[i] = v`): a plain
                # store through the tile's coordinate map — every lane writes
                # the same address with the same value under uniform indices
                # (upstream warp semantics), so no barrier is needed; the
                # next collective op's barrier orders it.
                tvar = self.tile_vars[args[0].strip()]
                coords = [self.expr(a) for a in args[1:-1]]
                value = self.expr(args[-1])
                if len(coords) != len(tvar.dims):
                    raise TranslationError(
                        f"wp::assign element store needs {len(tvar.dims)} indices, got {len(coords)}"
                    )
                if tvar.kind == "scalar":
                    if len(coords) != 1:
                        raise TranslationError("per-thread tile element store is 1-D only in the closure")
                    return f"{args[0].strip()} = {value}"
                arr = self._shrub(tvar)
                return f"{arr}[{tvar.coord_addr(coords)}] = {value}"
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
            return f"*wp_address({self._data_expr(spec)}, {self._meta_ref(spec)}{', ' + idx if idx else ''}) = {value}"
        if key and key[4:].startswith("atomic_"):
            return self._atomic(key[4:], args)
        if key in _MULTI_OUT_BUILTINS:
            return self._multi_out_call(key, args)
        # value-producing builtins as statements: emit as expression statement
        return self._call(src)

    def _multi_out_call(self, key: str, args: list[str]) -> str:
        """Multi-output builtin statement (upstream codegen emits
        ``func(fwd_args + output_list)`` — the SSA output vars ride as
        trailing call arguments, cf. the native exports
        ``wp_builtin_quat_to_axis_angle_quatf(q, ret_0, ret_1)``).

        Lowered through per-output thread temps aliased to the SSA vars:
        the decls + call land on one statement line, so the temps are in
        scope exactly where upstream declares its output vars (the call
        site's block); the following SSA loads resolve through the alias
        table."""
        spec = _MULTI_OUT_BUILTINS[key]
        n_in = spec["inputs"]
        out_types = spec["outputs"]
        out_vars = [a.strip() for a in args[n_in:]]
        if len(out_vars) != len(out_types):
            raise TranslationError(
                f"{key}: expected {n_in} input + {len(out_types)} output args, got {len(args)} args"
            )
        in_exprs = [self.expr(a) for a in args[:n_in]]
        parts: list[str] = []
        call_args = list(in_exprs)
        for var, msl_t in zip(out_vars, out_types):
            temp = f"_wp_out_{var}"
            parts.append(f"{msl_t} {temp};")
            self.aliases[var] = (False, temp)
            call_args.append(temp)
        parts.append(f"{spec['msl']}({', '.join(call_args)})")
        return " ".join(parts)

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
        if getattr(fn, "is_template", False):
            # clone-per-site monomorphization (ensure_mat_def pattern one
            # form wider): bind each template tile param to the concrete
            # caller-side tile, instantiate under a per-slot suffixed name,
            # and re-target the call. Directed decision (coordinator,
            # 2026-11-15): the dtype-keyed-specialization alternative was
            # rejected as novel machinery — this follows the established
            # per-unit generic precedent.
            bindings: dict[str, tuple[int, str, list[int]]] = {}
            slots: list[int] = []
            for p in fn.params:
                if p.kind != "tile_binding":
                    continue
                idx = fn.params.index(p)
                arg = args[idx]
                tvar = self.tile_vars.get(arg.strip())
                if tvar is None or tvar.kind != "shr" or tvar.slot < 0:
                    raise TranslationError(
                        f"call {name}: tile param {p.name} got {arg!r} — template callee "
                        "monomorphization needs a shared-storage tile argument"
                    )
                bindings[p.msl] = (tvar.slot, tvar.elem, list(tvar.dims))
                slots.append(tvar.slot)
            concrete = name + "".join(f"__shr{s}" for s in slots)
            if concrete not in self.user_funcs:
                inst = _translate_function(
                    fn.template_func,
                    self.unit,
                    self.user_funcs,
                    self.codegen,
                    tile_bindings=bindings,
                    inst_name=concrete,
                )
                self.user_funcs[concrete] = inst
                self.unit.functions.append(inst)
            fn = self.user_funcs[concrete]
            name = concrete
        out: list[str] = []
        vi = 0
        for p in fn.params:
            if vi >= len(args):
                raise TranslationError(f"call {name}: missing argument for param {p.name}")
            arg = args[vi]
            vi += 1
            if p.kind == "tile_binding":
                # tile travels via the hidden shr-slot pointer; verify the
                # call-site tile matches the instantiation's concrete binding
                tvar = self.tile_vars.get(arg.strip())
                if tvar is None or tvar.slot != p.slot:
                    raise TranslationError(
                        f"call {name}: tile arg {arg!r} does not match the "
                        f"instantiated binding (slot {p.slot})"
                    )
                continue
            if p.kind == "array":
                spec = self._array_arg_spec(arg)
                if spec is None:
                    raise TranslationError(f"array param {p.name} of {name} got non-array {arg!r}")
                out.append(self._data_expr(spec))
                out.append(self._meta_ref(spec))
            elif p.kind == "struct" and p.array_fields:
                # struct value + one (data, meta) pair per split array field
                out.append(self.expr(arg))
                for field, elem_ctype in p.array_fields:
                    spec = self.array_args.get(f"{arg.strip()}.{field}")
                    if spec is None:
                        # Struct-array element passed by value (e.g.
                        # Gaussian.Data via wp.array[Gaussian.Data]): its
                        # array fields are inline runtime data with no launch
                        # bindings. Dead-by-construction branch in the M1
                        # closure (no Gaussian geometry exists in Phase 1 —
                        # the shape_type guard can never select it), so bind
                        # null data + a zero meta: an executed read would
                        # fault loudly, never silently diverge.
                        out.append(f"((device {msl_type(elem_ctype)}*)0)")
                        out.append("_wp_dead_meta_g")
                        self.dead_meta_emitted = True
                        continue
                    out.append(self._data_expr(spec))
                    out.append(self._meta_ref(spec))
            else:
                out.append(self.expr(arg))
        if vi != len(args):
            raise TranslationError(f"call {name}: {len(args)} args for {len(fn.params)} params")
        if fn.needs_lane:
            out.append("_wp_lane")
            out.append("_wp_live")
            for k, _elem in fn.shr_slots:
                out.append(f"_wp_tile_shr_{k}")
            self.callee_needs_lane = True
            # the forwarded _wp_live argument needs the kernel-preamble
            # definition (callees run tile reduces this body never sees)
            self.callee_needs_live = True
            for slot_k, slot_elem in fn.shr_slots:
                if all(slot_k != k for k, _e, _c in self.used_shr):
                    self.used_shr.append((slot_k, slot_elem, "+"))
        if fn.needs_arena:
            out.append("_wp_arena")
            self.callee_needs_arena = True
        if fn.needs_mesh:
            out.append("_wp_mesh_table")
            self.callee_needs_mesh = True
        return f"{name}({', '.join(out)})"


_RE_ITER = re.compile(r"\b(iter_cmp|iter_next)\(")
_RE_SEC = re.compile(r"^\s*//\s*(argument vars|primal vars|forward)\s*$")

# C99 suffixed math names appearing in Newton's func_native snippets
# (metal_stdlib exposes only the unprefixed overloads)
_C_MATH_REWRITES = {
    "sqrtf": "sqrt",
    "fabsf": "fabs",
    "fminf": "min",
    "fmaxf": "max",
    "ceilf": "ceil",
    "floorf": "floor",
    "roundf": "rint",
    "logf": "precise::log",
    "expf": "precise::exp",
    "sinf": "precise::sin",
    "cosf": "precise::cos",
    "tanf": "precise::tan",
    "asinf": "asin",
    "acosf": "acos",
    "atanf": "atan",
    "atan2f": "atan2",
    "powf": "pow",
    "fmodf": "fmod",
}


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
    if _RE_ARRAY_T.match(ctype) or _RE_FIXEDARRAY_T.match(ctype):
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
    # the tile mathdx builders (matmul/cholesky/solve lto dispatch) read
    # options["output_arch"] unconditionally — a caller that omits it parses
    # the vendored builtins with a KeyError
    opts.setdefault("output_arch", None)
    # The deterministic builder branches on `options.get("deterministic")`;
    # a missing key reads as None != NOT_GUARANTEED and turns ON the det
    # codegen path (WP_DET_SCATTER_OR_FALLBACK macros). Force plain mode.
    opts["deterministic"] = DeterministicMode.NOT_GUARANTEED
    plain_scalar_tiles = False
    if opts.get("block_dim") is None:
        # plain launch without an explicit block_dim: parse tile shapes under
        # the degenerate one-lane block (Newton's non-CUDA CPU convention —
        # the shape every plain-launch golden was generated with), but keep
        # unit.block_dim None so every TileVar lowers per-thread scalar (no
        # shared arrays, no barriers). The emitted MSL is then independent
        # of the dispatch threadgroup layout, so the host may hand it any
        # threadgroup size including the native auto choice.
        opts["block_dim"] = 1
        plain_scalar_tiles = True

    adj = kernel.adj
    adj.build(None, opts)

    # kernel MSL entry name (valid C identifier)
    base = kernel.key.rsplit("::", 1)[-1]
    name = re.sub(r"\W", "_", base)

    unit = KernelUnit(name)
    if "block_dim" in opts and not plain_scalar_tiles:
        # launch-supplied block size (launch_tiled always; a plain launch
        # only with an explicit block_dim): the vendored parse consumed
        # options["block_dim"] for tile shapes, and wp.block_dim() folding /
        # the threadgroup reduction lowering key off the same value
        unit.block_dim = int(opts["block_dim"])

    # ---- classify kernel args from the vendored type objects ----
    from ..frontend.codegen import Var as _Var
    from ..frontend.types import is_tile_stack as _is_tile_stack
    from ..frontend.types import type_is_struct as _type_is_struct

    # tile_stack payload structs must have their MSL def (and byte-size
    # fields record) emitted before any stack slot claims its size — the
    # shared data array stores the struct by value
    for var in adj.variables:
        if _is_tile_stack(var.type) and _type_is_struct(var.type.dtype):
            _emit_struct_def(var.type.dtype, unit, _codegen)

    array_args: dict[str, ArgSpec] = {}
    # struct-typed array element types waiting for their MSL def (see the
    # array-arg classification below and the tail emission before return)
    pending_elem_structs: list[tuple[str, object]] = []
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
            # Array element types reach the MSL text only through the
            # generic wp_address template (`device T*`) — no decl line ever
            # names them, so a generated vecN/matMN element needs its
            # struct def emitted here (Newton update_axis_properties
            # stores through `actuator_bias[w, a][k]` with no vec10
            # locals; latent missing-def until the component-ref store
            # path landed)
            try:
                elem = spec.msl_elem
            except TranslationError:
                # untyped `wp::array_t<wp::Any>` (repeat_array_kernel):
                # byte-copied buffers never name their element struct
                elem = None
            if elem is not None:
                if _RE_GEN_VEC.match(elem):
                    ensure_vec_def(unit, elem)
                elif _RE_GEN_MAT.match(elem):
                    ensure_mat_def(unit, elem)
                elif _type_is_struct(var.type.dtype):
                    # struct-typed array element (viewer GL RenderVertex /
                    # LineVertex — first Newton path with a struct array
                    # param): the MSL struct def must exist before the
                    # param binding and wp_address spell the mangled name.
                    # DEFERRED, not emitted here: a transitive @wp.func may
                    # carry the same struct as a value param and emit the
                    # def itself (Newton Gaussian.Data in compute_shape_
                    # local_bounds — the func-path emission registers
                    # struct_array_fields, which the kernel body's element
                    # reference resolution depends on; an early emission
                    # would poison that via the unit.structs dedup). The
                    # pending types are emitted after the body if nobody
                    # claimed them (see the tail of translate_kernel).
                    pending_elem_structs.append((elem, var.type.dtype))

    # ---- transitive user functions, callees first ----
    user_funcs: dict[str, MSLFunction] = {}
    visited: set[int] = set()

    def visit(f):
        # dedup on the Function OBJECT: generic (@wp.func with Any params)
        # instantiations share the parent's native_func, so the float and
        # vec3 instances of _alm_relaxed_ascent both arrive as
        # `_alm_relaxed_ascent_0`. Upstream emits both C++ definitions under
        # the shared name and lets C++ overload resolution dispatch the
        # identical call-site text (update_duals_joint) — MSL is C++14, so
        # the same works here: every distinct instance gets its definition,
        # and only _call_user's wiring needs a representative.
        if id(f) in visited:
            return
        visited.add(id(f))
        fname = f.native_func
        snippet = getattr(f, "native_snippet", None)
        if snippet:
            fn = _translate_native_function(f, snippet, unit, _codegen)
            user_funcs[fname] = fn
            unit.functions.append(fn)
            return
        for callee in f.adj.called_user_functions:
            visit(callee)
        fn = _translate_function(f, unit, user_funcs, _codegen)
        existing = user_funcs.get(fname)
        if existing is not None:
            # kind divergence would mis-wire array/struct args at call sites
            # resolved through the first-registered instance — fail loud
            kinds = [p.kind for p in fn.params]
            if kinds != [p.kind for p in existing.params]:
                raise TranslationError(
                    f"function {fname}: generic instances share this native name with "
                    f"incompatible parameter kinds {kinds} vs "
                    f"{[p.kind for p in existing.params]} — the shim cannot wire both"
                )
        else:
            user_funcs[fname] = fn
        if not getattr(fn, "is_template", False):
            unit.functions.append(fn)

    for uf in adj.called_user_functions:
        visit(uf)

    # ---- value-context struct defs from SSA locals (KEP-0001 M3.3) ----
    # A struct type can appear ONLY on locals and function signatures —
    # built by struct constructors, returned from / passed into @wp.func
    # callees — with no kernel parameter ever carrying it (mjw
    # primitive_narrowphase: geom_collision_pair returns two Geom values
    # that the writer funcs consume by reference). Nothing in the param
    # paths emits the def, so the body's `Geom_<hash> var = {}` decls name
    # an undefined type and the library fails to compile. Emit every
    # struct-typed variable's def in value context (array fields as
    # wp_val_array_* pairs). Runs AFTER callee translation so the split
    # forms win their dedup slot: a struct that reaches a callee as a
    # VALUE param needs its array fields split into per-field bindings
    # wired from the caller's backing array param (the compute_gaussian_bounds
    # machine) — the value form cannot serve that signature. _emit_struct_def
    # dedupes on native_name, so param-split and tile_stack payloads keep
    # their earlier forms and only never-emitted types land here.
    for var in adj.variables:
        if _type_is_struct(var.type):
            _emit_struct_def(var.type, unit, _codegen, value_arrays=True)

    # ---- kernel body ----
    src = _codegen.codegen_func_forward(adj, func_type="kernel", device="cpu")
    tr = _BodyTranslator(unit, array_args, user_funcs, codegen=_codegen)
    # kernel scalar args ride the constant Params buffer — identical for
    # every block: seed the provable-uniformity set with their SSA names
    for a in unit.args:
        if a.kind == "scalar":
            tr.uniform_vars.update({f"var_{a.name}", a.name})
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
    if tr.used_mesh:
        unit.needs_mesh = True
    if tr.callee_needs_live:
        # a callee forwards _wp_live into the kernel body's call text: the
        # preamble must define it (kernel_src checks this exact flag)
        unit.callee_needs_live = True
    unit.kernel.lines = bindings + tr.hoisted + body

    # struct-typed array element defs nobody claimed: a transitive @wp.func
    # carrying the same struct as a value param has emitted the def itself
    # (with split_arrays context — Gaussian.Data / compute_shape_local_
    # bounds); only still-missing types emit here (viewer GL RenderVertex /
    # LineVertex, which no function ever names). value_arrays=True: array
    # fields ride inline as wp_val_array pairs, the only legal value-form.
    for elem, dtype in pending_elem_structs:
        if elem in unit.structs or elem in unit.struct_array_fields:
            continue
        _emit_struct_def(dtype, unit, _codegen, value_arrays=True)
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
    # C99 suffixed math names have no MSL counterparts (metal_stdlib exposes
    # the unprefixed overloads); rewrite on word boundaries — `float` type
    # text never matches these keys
    for c_name, msl_name in _C_MATH_REWRITES.items():
        body = re.sub(rf"\b{c_name}\s*\(", f"{msl_name}(", body)
    fn.lines = [f"    {ln}" for ln in body.splitlines()]
    return fn


def _translate_function(
    f,
    unit: KernelUnit,
    user_funcs: dict,
    _codegen,
    tile_bindings: dict[str, tuple[int, str, list[int]]] | None = None,
    inst_name: str | None = None,
) -> MSLFunction:
    """Translate one built ``@wp.func`` (callee-first order guaranteed).

    Tile-typed params arrive from warp's C++ as ``template<typename tile_x>``:
    with ``tile_bindings=None`` the translation stops at a template STUB
    (params kept SSA-aligned, no body); ``_call_user`` instantiates it
    per concrete caller-side tile (clone-per-site monomorphization, the
    ensure_mat_def pattern one form wider) by re-entering with
    ``tile_bindings={template type name: (slot, elem, dims)}`` and
    ``inst_name=fn__shr<slot>``.
    """
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
    # template<typename A, typename B> prefix precedes `static ...` — strip
    # the whole clause (a plain `header[len("static "):]` chop would cut
    # seven letters off "template" and leave garbage in ret_cpp)
    template_names: list[str] = []
    if header.startswith("template<"):
        tclose = header.index(">")
        inner = header[len("template<") : tclose]
        template_names = [p.strip().split()[-1] for p in split_args(inner) if p.strip()]
        header = header[tclose + 1 :].lstrip()
    header = header[len("static ") :]
    open_idx = header.index("(")
    close_idx = header.rindex(")")
    ret_and_name = header[:open_idx].strip()
    ret_cpp, fn_name = ret_and_name.rsplit(None, 1)
    assert fn_name == fname, f"function name mismatch: {fn_name} vs {fname}"
    ret_msl = None
    m_tile_ret = re.match(r"^wp::tile_shared_t<(.+)>$", ret_cpp.strip()) if ret_cpp else None
    tile_ret: tuple[str, list[int]] | None = None
    if ret_cpp != "void" and m_tile_ret is None:
        if _RE_TUPLE_T.match(ret_cpp):
            ret_msl = ensure_tuple_def(unit, [msl_type(e.strip()) for e in split_args(_RE_TUPLE_T.match(ret_cpp).group(1))])
        else:
            ret_msl = msl_type(ret_cpp)
            if _classify_ctype(ret_cpp) == "struct":
                # struct-returning @wp.func: the def must exist in this unit
                # for the signature, the callee's own return-slot decl
                # (`T var_4 = {};`) and the caller's result decl. The
                # kernel-level value-context loop never sees struct types
                # confined to the function-call chain — a kernel that only
                # consumes scalar projections of the returned value keeps no
                # struct-typed SSA var (step_joint_C0_lambda_rho →
                # compute_geometric_rod_kappa_cached_z →
                # _measure_rod_bend_twist_z). Value form: the returned struct
                # is an SSA value; a split-form def from an earlier param use
                # wins the dedup slot (array-field structs then fail loud at
                # the caller's field read — same policy as the value-context
                # loop).
                from ..frontend.types import type_is_struct as _type_is_struct

                ret_var = f.adj.return_var
                ret_type = ret_var[0].type if ret_var else f.adj.arg_types.get("return")
                if ret_type is not None and _type_is_struct(ret_type):
                    _emit_struct_def(ret_type, unit, _codegen, value_arrays=True)
    else:
        if m_tile_ret is not None:
            # tile-returning @wp.func (warp_nn tile_transposed_gemm_2d):
            # upstream passes tile_shared_t by value; here the value travels
            # through the callee's own claimed shared slot — the MSL
            # signature stays void and the caller's result var rebinds to
            # that slot (the `= nullptr` alias-decl mechanism one step
            # wider). Owner=true (last template arg) is the only form the
            # corpus closure produces.
            if unit.block_dim is None:
                raise TranslationError(
                    f"function {fname}: tile_shared_t return outside a tiled launch "
                    "(plain launches lower tiles per-thread scalar)"
                )
            parts = split_args(m_tile_ret.group(1))
            tile_ret = (msl_type(parts[0].strip()), [])
            m_shape = re.search(r"wp::tile_shape_t<([\d,\s]+)>", m_tile_ret.group(1))
            if m_shape:
                tile_ret = (
                    tile_ret[0],
                    [int(d) for d in m_shape.group(1).replace(" ", "").split(",") if d],
                )
            if not tile_ret[1]:
                raise TranslationError(f"function {fname}: tile_shared_t return without tile_shape_t ({ret_cpp})")

    fn = MSLFunction(inst_name or fn_name, ret_msl)
    array_args: dict[str, ArgSpec] = {}

    for arg_src in split_args(header[open_idx + 1 : close_idx]):
        if "&" in arg_src:
            ctype, pname = arg_src.split("&")
            if ctype.strip() in template_names:
                # tile-typed param (warp's template<typename tile_x> form):
                # kept SSA-aligned as a tile_binding param — nothing emitted
                # in the signature, the tile travels through the caller's
                # `_wp_tile_shr_{slot}` hidden param. On the stub, ``msl``
                # carries the template type name (the binding key).
                fn.params.append(Param("tile_binding", ctype.strip(), pname.strip()))
                continue
            if _RE_TUPLE_T.match(ctype.strip()):
                ensure_tuple_def(unit, [msl_type(e.strip()) for e in split_args(_RE_TUPLE_T.match(ctype.strip()).group(1))])
            fn.params.append(Param("ref", msl_type(ctype), pname.strip()))
            continue
        ctype, pname = arg_src.rsplit(None, 1)
        m_ptup = _RE_TUPLE_T.match(ctype.strip())
        if m_ptup:
            # tuple-typed value param (warp_nn tile_transposed_gemm_2d's
            # `index: tuple[int, int]`): a tuple_t VALUE struct by value —
            # same def the kernel-side `wp::tuple(a, b)` ctor uses
            tname = ensure_tuple_def(
                unit, [msl_type(e.strip()) for e in split_args(m_ptup.group(1))]
            )
            fn.params.append(Param("value", tname, pname))
        elif _classify_ctype(ctype) == "array":
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

    # template stub: tile-typed params present, no concrete bindings yet —
    # defer body translation to the first call site (clone-per-site)
    if tile_bindings is None and any(p.kind == "tile_binding" for p in fn.params):
        fn.is_template = True
        fn.template_params = [p.msl for p in fn.params if p.kind == "tile_binding"]
        fn.template_func = f
        return fn

    bound_tiles: list[tuple[Param, int, str, list[int]]] = []
    if tile_bindings:
        for p in fn.params:
            if p.kind != "tile_binding":
                continue
            if p.msl not in tile_bindings:
                raise TranslationError(
                    f"function {fname}: no concrete binding for template param {p.msl!r}"
                )
            slot, elem, dims = tile_bindings[p.msl]
            p.slot = slot
            p.msl = elem
            bound_tiles.append((p, slot, elem, dims))

    tr = _BodyTranslator(unit, array_args, user_funcs, scope="func", codegen=_codegen)
    # by-value scalar params are block-identical (C++ value semantics): their
    # names seed the provable-uniformity set (mjw blocked cholesky's
    # `end == matrix_size` guards)
    for p in fn.params:
        if p.kind == "value":
            bare = p.name.removeprefix("var_")
            tr.uniform_vars.update({p.name, bare, f"var_{bare}"})
            if p.msl.startswith("wp_tuple_"):
                # tuple value param: the body's wp::extract<i> reads need the
                # param's tuple type (params have no decl line to seed it)
                tr.var_types[p.name] = p.msl
    # bind the template tile params into the body scope: the tile ops inside
    # (tile_view / tile_map / tile_reduce over the param) read the caller's
    # threadgroup array through the hidden shr-slot param
    for p, slot, elem, dims in bound_tiles:
        size = 1
        for d in dims:
            size *= d
        tvar = TileVar("shr", elem, slot=slot, size=size, dims=list(dims))
        for key in (p.name, p.name.removeprefix("var_"), f"var_{p.name.removeprefix('var_')}"):
            tr.tile_vars[key] = tvar
        tr.used_shr.append((slot, elem, "+"))
    section = None
    depth = 0  # emitted-brace depth; 0 → the C++ function's closing brace
    # store screen for the uniform-address load rule: resolve every store /
    # atomic / copy target through the kernel's address() sites and mark the
    # array as written — a written array must not feed a uniformity argument
    # (a lane could change the value between lanes' loads). array_store names
    # the array directly; store/copy/atomic go through address() vars.
    if unit.block_dim is not None and unit.block_dim > 1:
        addr_of: dict[str, str] = {}
        for ln in lines[body_start:]:
            m_a = re.search(r"\b(var_\w+)\s*=\s*wp::address\(var_(\w+)", ln)
            if m_a:
                addr_of[m_a.group(1)] = "var_" + m_a.group(2)
                continue
            m_d = re.search(r"wp::array_store\(var_(\w+)", ln)
            if m_d:
                tr._stored_arrays.add("var_" + m_d.group(1))
                continue
            if re.search(r"wp::(?:store|copy|atomic_\w+)\(", ln):
                m_s = re.search(r"wp::(?:store|copy|atomic_\w+)\(\s*(var_\w+)", ln)
                arr = None
                if m_s:
                    arr = addr_of.get(m_s.group(1))
                if arr is None:
                    m_d2 = re.search(r"wp::address\(var_(\w+)", ln)
                    arr = "var_" + m_d2.group(1) if m_d2 else None
                if arr:
                    tr._stored_arrays.add(arr)
    for line in lines[body_start:]:
        if m := _RE_SEC.match(line):
            section = m.group(1)
            continue
        if section in (None, "argument vars"):
            continue  # args are the MSL signature params already
        if line.strip() == "}" and depth == 0:
            break  # function-closing brace (early `return` blocks nest deeper)
        out = tr.stmt(line)
        # real brace delta per emitted line (NOT endswith heuristics): tile
        # ops open blocks mid-line (`{ float _wp_z = {};`), so an endswith
        # check under-counts opens, drifts negative, and the function's
        # closing brace never breaks the loop — it leaks into fn.lines as an
        # extraneous `}` and the MSL fails to compile
        for emitted in out:
            depth += emitted.count("{") - emitted.count("}")
        fn.lines.extend(out)
    if tr.used_shr:
        # tile ops inside a @wp.func (own sites or transitively through a
        # callee): the function receives the caller's lane index and the
        # kernel-scope shared arrays as hidden trailing parameters — a device
        # function has no tid, and MSL forbids function-scope threadgroup
        # storage outside kernels. Every tile call site sits in uniform
        # control flow under the tiled kernel, so the kernel preamble's
        # `_wp_lane`/arrays serve nested users too.
        if unit.block_dim is None:
            raise TranslationError(f"function {fname}: tile ops outside a tiled launch")
        fn.needs_lane = True
        fn.shr_slots = [(k, elem) for k, elem, _combine in tr.used_shr]
    if tile_ret is not None:
        # resolve the returned shared slot from the callee's `return var;`
        # line — the callee's own tile_zeros claim (clone-per-site keeps it
        # unique to this instantiation)
        m_rv = re.search(r"\breturn\s+(var_\w+)\s*;", "\n".join(lines[body_start:]))
        rv = tr.tile_vars.get(m_rv.group(1)) if m_rv else None
        if rv is None or rv.kind != "shr" or rv.slot < 0:
            raise TranslationError(
                f"function {fname}: tile_shared_t return names no shared tile var"
                + (f" (got {rv!r})" if rv is not None else "")
            )
        if list(rv.dims or []) != tile_ret[1]:
            raise TranslationError(
                f"function {fname}: tile return shape {rv.dims} != decl shape {tile_ret[1]}"
            )
        fn.tile_return = (tile_ret[0], tile_ret[1], rv.slot)
    if getattr(tr, "used_arena", False) or getattr(tr, "callee_needs_arena", False):
        fn.needs_arena = True
    if getattr(tr, "used_mesh", False) or getattr(tr, "callee_needs_mesh", False):
        fn.needs_mesh = True
        unit.needs_mesh = True
    if tr.hoisted:
        fn.lines[:0] = tr.hoisted
    return fn


def _ensure_val_array_def(elem_msl: str, unit: KernelUnit) -> str:
    """(device ptr, meta) pair struct for an array FIELD inside a
    value-context struct def (KEP-0001 M3.3) — the wp_mesh_array machine
    generalized per element type. Zero-initializable (`= {}`), which is all
    the signature-carriage case needs: mjw's ``Geom`` bundles the mesh
    topology arrays that primitive-pair closures never touch, and the struct
    rides function out-params (`thread Geom&`) by reference.
    """
    name = f"wp_val_array_{elem_msl}"
    if name in unit.structs:
        return name
    unit.structs[name] = (
        f"struct {name} {{ device {elem_msl}* data; wp_arr_meta m; }};\n"
        f"static inline {name} make_{name}() {{\n"
        f"    {name} v = {{}};\n"
        f"    return v;\n"
        f"}}"
    )
    return name


def _emit_struct_def(s, unit: KernelUnit, _codegen, split_arrays: bool = False, value_arrays: bool = False) -> list[tuple[str, str]]:
    """Emit an MSL struct definition (warp-native layout) + constructors.

    With ``split_arrays`` (parameter context), array-typed fields are excluded
    from the MSL struct — Metal cannot store device pointers inside a
    ``constant`` buffer struct — and returned as ``[(field, elem ctype)]`` for
    per-field buffer binding. With ``value_arrays`` (value context: the type
    appears only on SSA locals and function signatures, never as a kernel
    param), array fields emit inline as ``wp_val_array_*`` (device ptr +
    meta) pairs — the definition must be complete because the MSL body names
    the type, but arrays are still never MSL values, so any closure that
    actually moves an array through the field has no lowering and fails loud
    at translation. Nested struct fields always emit recursively with
    ``split_arrays=False``: a nested array field has no binding path and
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
            if value_arrays:
                fields.append((label, _ensure_val_array_def(array_elem_msl(ctype), unit)))
                continue
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
        msl_t = msl_type(ctype)
        # generic vec/mat sizes (mjw mat63 polyvert in Geom, Newton vec10)
        # get their per-unit def ensured right where the field names them
        if _RE_GEN_VEC.match(msl_t):
            ensure_vec_def(unit, msl_t)
        elif _RE_GEN_MAT.match(msl_t):
            ensure_mat_def(unit, msl_t)
        fields.append((label, msl_t))

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
        lines.append("    return v;")
        lines.append("}")
        # zero-arg form; for empty structs it would collide with the above
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
