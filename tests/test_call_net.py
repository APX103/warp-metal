# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Static call-signature net + atomic surface pins (KEP-0001 M3.2).

The M3.2 body-force battery failed at Metal compile with "no matching
function for call to 'wp_atomic_add_ret_at'": the translator emits well-
formed call text, but overload resolution happens at Metal compile, past
every host-side gate. Two defenses land here:

1. ``tools/msl_call_check.py`` — parses every ``static inline``/template
   prototype in the prelude plus the per-unit generated defs
   (``ensure_vec_def``/``ensure_mat_def``), extracts every ``wp_*``/
   ``make_*`` call site from the translated unit, and verifies name, arity
   and rough argument types (pointer base, scalar family, struct value).
   The closure sweep is driven from the CLI (``--scenario`` for the
   real-warp CPU ground-truth trace, ``--sweep`` for the shim-side module
   sweep); this module pins the checker itself and the overload surface.

2. Prelude/generated atomic overloads observed in the live mujoco+Newton
   closure (vec3f / spatial_vectorf / mat33f 2-D adds, spatial 2-D sub,
   float 2-D sub, generated-vec 2-D add, spatial component
   assign/sub-inplace, spatial scalar broadcast) — per-component semantics
   mirroring upstream vec.h:1831-1839 (old-value vector returns).

Host-only: translation and static checks only — no Metal device, no
launch. Per-component numeric correctness is main-thread (GPU battery)
work; here we pin the emitted MSL shape so regressions surface host-side.
"""

import importlib.util
import re
import unittest
from pathlib import Path

import warp_metal as wp
from warp_metal.msl.prelude import PRELUDE
from warp_metal.msl.translate import _shr_elem_bytes, translate_kernel

# tools/ is not a package; load the checker module by path
_TOOLS = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location("msl_call_check", _TOOLS / "msl_call_check.py")
_checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checker)

vec10 = wp.types.vector(length=10, dtype=float)
# mjw types.vec6: a user-declared 6-vector; msl_type canonicalizes vec_t<6>
# to spatial_vectorf, so its scalar broadcast and component ops emit the
# spatial overloads (this is exactly how _implicit_free_body reaches them)
vec6 = wp.types.vector(length=6, dtype=float)


# ---- kernels (translate-only; never launched) ----


@wp.kernel
def k_subtree_com_style(out: wp.array2d(dtype=wp.vec3)):
    # mjw _subtree_com_acc shape: captured 2-D vec3 atomic add (ret form)
    i, j = wp.tid()
    old = wp.atomic_add(out, i, j, wp.vec3(1.0, 2.0, 3.0))
    out[i, j] = old + wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_spatial_2d_add_sub(out: wp.array2d(dtype=wp.spatial_vector)):
    # mjw flex body-force + tendon-damping shape: 2-D spatial add/sub, ret
    i, j = wp.tid()
    a = wp.atomic_add(out, i, j, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b = wp.atomic_sub(out, i, j, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    out[i, j] = a + b


@wp.kernel
def k_mat33_2d_add(out: wp.array2d(dtype=wp.mat33)):
    # mjw _flex_hessian_elem shape: 2-D mat33 atomic add, ret form
    i, j = wp.tid()
    m = wp.atomic_add(out, i, j, wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    out[i, j] = m


@wp.kernel
def k_float_2d_sub(out: wp.array2d(dtype=wp.float32)):
    # mjw _qderiv_tendon_damping shape: 2-D float atomic sub, ret form
    i, j = wp.tid()
    x = wp.atomic_sub(out, i, j, 0.5)
    out[i, j] = x


@wp.kernel
def k_vec10_2d_add(out: wp.array2d(dtype=vec10)):
    # mjw crb shape: 2-D generic-vec10 atomic add (ensure_vec_def def)
    i, j = wp.tid()
    old = wp.atomic_add(out, i, j, vec10())
    out[i, j] = old


@wp.kernel
def k_spatial_component_inplace(out: wp.array(dtype=wp.float32)):
    # mjw _implicit_free_body shape: component assign/sub on a local vec6
    # (msl_type maps vec_t<6> to spatial_vectorf)
    s = vec6(0.0)
    s[0] = 1.0
    s[3] -= 2.0
    out[0] = s[0] - s[3]


@wp.kernel
def k_spatial_scalar_broadcast(out: wp.array(dtype=wp.float32)):
    # mjw _implicit_free_body shape: vec6(scalar) broadcast ctor
    s = vec6(0.5)
    out[0] = s[0] + s[5]


vec5 = wp.types.vector(length=5, dtype=float)


@wp.kernel
def k_spatial_equal_zero(out: wp.array(dtype=wp.int32)):
    # mjw _apply_ft shape: compare a spatial vector against the zero
    # broadcast (`ft_body == wp.spatial_vector()`). Upstream vec.h
    # operator== is all-components-equal, bool result; the infix form must
    # rewrite to wp_equal because MSL structs carry no operator==
    v = vec6(0.0)
    v[0] = 1.0
    if v == vec6():
        out[0] = 1
    else:
        out[0] = 0


@wp.kernel
def k_vec5_noslip_ops(out: wp.array(dtype=wp.float32)):
    # mjw _noslip_solve shape: neg/dot/add/scalar-mul on the 5-wide efc
    # force vectors (vec.h componentwise semantics)
    a = vec5(1.0)
    b = vec5(2.0)
    n = -a
    d = wp.dot(a, b)
    s = a + b
    m = 0.5 * a
    out[0] = n[0] + d + s[1] + m[2]


@wp.kernel
def k_tile_scatter_add(x: wp.array(dtype=wp.float32)):
    # mjw solver.py:2709 JTDAJ densify shape: tile_scatter_add into a shared
    # tile with has_value gating (unique per-lane addresses — the closure's
    # only site scatters CSR entries whose (col, row) positions are unique
    # per block, which is the premise for the Atomic=false lowering)
    t = wp.tile_zeros(shape=(8,), dtype=float, storage="shared")
    i = wp.tid()
    cond = i < 4
    pos = wp.where(cond, i, 0)
    wp.tile_scatter_add(t, pos, wp.where(cond, 1.0, 0.0), cond)
    wp.tile_store(x, t)


_TILE_SCATTER_ADD_BD = 4


@wp.kernel
def k_view2d_atomic(x: wp.array3d(dtype=wp.uint32), out: wp.array(dtype=wp.int32)):
    # mjw _mark_tactile_contacts shape: a 2-index view of an array3d
    # (weld_geom_seen_out[worldid, tactile_idx]) yields the 1-D row binding
    # an atomic_or writes through — array.h view(): one strides[k] byte
    # advance per index, meta shifted left by two dims
    i = wp.tid()
    row = x[i, 0]
    wp.atomic_or(row, 0, wp.uint32(1))
    if row.shape[0] > 0:
        out[0] = 1


@wp.kernel
def k_tile_jtdaj_ops(v_in: wp.array2d(dtype=wp.float32), x: wp.array2d(dtype=wp.float32)):
    # mini mjw _update_gradient_JTDAJ_dense_tiled (solver.py:2729-2739):
    # arange index tile, ones*scalar threshold fold (the adjoint codegen's
    # tile⊗scalar mul), transpose alias + stride-0 broadcast, elementwise
    # map, matmul accumulated back through the `acc +=` tile_add_inplace form
    j = wp.tile_load(v_in, shape=(4, 2))
    jt = wp.tile_transpose(j)
    tid_tile = wp.tile_arange(4, dtype=float)
    threshold = wp.tile_ones(shape=(4,), dtype=float) * 2.0
    active = wp.tile_map(wp.mul, tid_tile, threshold)
    b = wp.tile_broadcast(active, shape=(2, 4))
    prod = wp.tile_map(wp.mul, jt, b)
    acc = wp.tile_zeros(shape=(2, 2), dtype=float, storage="shared")
    acc += wp.tile_matmul(prod, j)
    wp.tile_store(x, acc)


_JTDAJ_OPS_BD = 8


@wp.kernel
def k_tile_arange_scalar(x: wp.array(dtype=wp.float32)):
    # per-thread tier: arange/ones/add collapse to the thread's own element
    # (plain launches run one-lane blocks — CPU block_dim=1 golden semantics)
    t = wp.tile_arange(4, dtype=float)
    o = wp.tile_ones(shape=(4,), dtype=float)
    s = t + o
    wp.tile_store(x, s)


@wp.kernel
def k_tile_load_indexed(
    x: wp.array(dtype=wp.float32), idx_in: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)
):
    # mjw smooth.py:1360-1361 factorize_block shape: a shared int index tile
    # gathered through tile_load_indexed (1-D source, axis 0); the
    # structurally-absent sentinel index must read as zero via the
    # both-ends bounds check (tile.h:2963 compute_index)
    idx = wp.tile_load(idx_in, shape=(4,), offset=(0,), storage="shared")
    block = wp.tile_load_indexed(x, idx, shape=(4,), storage="shared")
    wp.tile_store(out, block)


@wp.kernel
def k_tile_load_indexed_2d(
    x: wp.array2d(dtype=wp.float32), idx_in: wp.array(dtype=wp.int32), out: wp.array2d(dtype=wp.float32)
):
    # outside the closure: a multi-dim gather (no mujoco site gathers 2-D)
    idx = wp.tile_load(idx_in, shape=(4,), offset=(0,), storage="shared")
    block = wp.tile_load_indexed(x, idx, shape=(4, 4), axis=0, storage="shared")
    wp.tile_store(out, block)


@wp.func
def _tile_tern_op(a: float, s: float, b: float):
    # mjw constraint._compute_jacp shape: a user-op map whose middle operand
    # is a plain scalar (upstream tile_map broadcasts scalar args per lane)
    return a * s + b


@wp.kernel
def k_tile_map_ternary(
    x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)
):
    a = wp.tile_load(x, shape=(4,), storage="shared")
    b = wp.tile_load(y, shape=(4,), storage="shared")
    c = wp.tile_map(_tile_tern_op, a, 2.0, b)
    wp.tile_store(out, c)


# ---- checker self-tests ----


class TestCheckerCore(unittest.TestCase):
    def test_parse_prototypes_strips_param_names(self):
        db = _checker.parse_prototypes("static inline float wp_dot(vec3f a, vec3f b) { return 0.0; }\n")
        self.assertIn("wp_dot", db)
        self.assertEqual([(p, ip) for p, ip in db["wp_dot"][0].params], [("vec3f", False), ("vec3f", False)])

    def test_parse_prototypes_template_generic(self):
        db = _checker.parse_prototypes(
            "template <typename T> static inline device T* wp_address(device T* d, wp_arr_meta m, int i) { return d; }\n"
        )
        self.assertIn("wp_address", db)
        params = db["wp_address"][0].params
        self.assertEqual(params[0], ("T", True))
        self.assertEqual(params[1], ("wp_arr_meta", False))

    def test_reference_param_is_not_pointer(self):
        db = _checker.parse_prototypes("static inline void wp_add_inplace(thread vec3f& v, int i, float x) {}\n")
        self.assertEqual(db["wp_add_inplace"][0].params[0], ("vec3f", False))

    def test_bogus_call_is_error_finding(self):
        class _StubFn:
            name = "stub"

            params = []

            lines = ["wp_totally_bogus_helper(var_0);"]

        checker = _checker.UnitChecker(_checker.parse_prototypes(PRELUDE))
        findings = checker.check_function("stub", _StubFn(), _StubFn.lines)
        errs = [f for f in findings if f.severity == "ERROR"]
        self.assertEqual(len(errs), 1)
        self.assertIn("no prototype found", errs[0].message)

    def test_unit_internal_calls_skipped(self):
        # unit-internal user functions (vendored `_0`/`_1` dedup suffixes,
        # e.g. the make_frame_0/make_frame_1 pair from mjw math.py + Newton
        # kernels.py both defining make_frame) are translated callee-first
        # into the same unit — they are not prelude surface and must not
        # register as "no prototype" findings
        class _StubFn:
            name = "stub"

            params = []

            lines = ["var_2 = make_frame_0(var_1);"]

        checker = _checker.UnitChecker(_checker.parse_prototypes(PRELUDE))
        findings = checker.check_function("stub", _StubFn(), _StubFn.lines, unit_fn_names={"make_frame_0"})
        self.assertEqual([f for f in findings if f.severity == "ERROR"], [])
        findings = checker.check_function("stub", _StubFn(), _StubFn.lines)
        errs = [f for f in findings if f.severity == "ERROR"]
        self.assertEqual(len(errs), 1)
        self.assertIn("make_frame_0", errs[0].message)


class TestCheckerOnClosureKernels(unittest.TestCase):
    """Every pinned kernel must pass the static net against the real prelude."""

    def _check_clean(self, kernel):
        from warp_metal.msl.translate import translate_kernel as _tk

        unit = _tk(kernel)
        findings = _checker.check_unit(unit)
        errs = [f for f in findings if f.severity == "ERROR"]
        self.assertEqual(errs, [], msg="\n".join(repr(e) for e in errs))

    def test_subtree_com_style(self):
        self._check_clean(k_subtree_com_style)

    def test_spatial_2d_add_sub(self):
        self._check_clean(k_spatial_2d_add_sub)

    def test_mat33_2d_add(self):
        self._check_clean(k_mat33_2d_add)

    def test_float_2d_sub(self):
        self._check_clean(k_float_2d_sub)

    def test_vec10_2d_add(self):
        self._check_clean(k_vec10_2d_add)

    def test_spatial_component_inplace(self):
        self._check_clean(k_spatial_component_inplace)

    def test_spatial_scalar_broadcast(self):
        self._check_clean(k_spatial_scalar_broadcast)

    def test_spatial_equal_zero(self):
        self._check_clean(k_spatial_equal_zero)

    def test_vec5_noslip_ops(self):
        self._check_clean(k_vec5_noslip_ops)


class TestPreludeAtomicSurface(unittest.TestCase):
    """Signature pins: the exact overload surface the closure needs."""

    def test_vec3f_2d_add(self):
        self.assertIn(
            "static inline void wp_atomic_add_at(device vec3f* d, wp_arr_meta m, int i, int j, vec3f v) {",
            PRELUDE,
        )
        self.assertIn(
            "static inline vec3f wp_atomic_add_ret_at(device vec3f* d, wp_arr_meta m, int i, int j, vec3f v) {",
            PRELUDE,
        )

    def test_spatial_2d_add_sub(self):
        self.assertIn(
            "static inline void wp_atomic_add_at(device spatial_vectorf* d, wp_arr_meta m, int i, int j, spatial_vectorf v) {",
            PRELUDE,
        )
        self.assertIn(
            "static inline spatial_vectorf wp_atomic_add_ret_at(device spatial_vectorf* d, wp_arr_meta m, int i, int j, spatial_vectorf v) {",
            PRELUDE,
        )
        self.assertIn(
            "static inline spatial_vectorf wp_atomic_sub_ret_at(device spatial_vectorf* d, wp_arr_meta m, int i, int j, spatial_vectorf v) {",
            PRELUDE,
        )

    def test_mat33_2d_add(self):
        self.assertIn(
            "static inline mat33f wp_atomic_add_ret_at(device mat33f* d, wp_arr_meta m, int i, int j, mat33f v) {",
            PRELUDE,
        )

    def test_float_2d_sub(self):
        self.assertIn(
            "static inline void wp_atomic_sub_at(device float* d, wp_arr_meta m, int i, int j, float v) {",
            PRELUDE,
        )
        self.assertIn(
            "static inline float wp_atomic_sub_ret_at(device float* d, wp_arr_meta m, int i, int j, float v) {",
            PRELUDE,
        )

    def test_float_scalar_sub_cores(self):
        # the 2-D float sub family delegates here; the pair was missing at
        # M3.2 and failed every unit's Metal compile (the 22 -> 114 battery
        # regression). Scalar float sub mirrors the float add cores
        # (atomic_fetch_sub_explicit, relaxed order).
        self.assertIn("static inline void wp_atomic_sub(device atomic_float* addr, float value) {", PRELUDE)
        self.assertIn("static inline float wp_atomic_sub_ret(device atomic_float* addr, float value) {", PRELUDE)

    def test_spatial_component_inplace(self):
        self.assertIn("static inline void wp_assign_inplace(thread spatial_vectorf& s, int i, float x) {", PRELUDE)
        self.assertIn("static inline void wp_sub_inplace(thread spatial_vectorf& s, int i, float x) {", PRELUDE)

    def test_spatial_scalar_broadcast(self):
        self.assertIn("static inline spatial_vectorf make_spatial_vectorf(float s) {", PRELUDE)


class TestPreludeSelfConsistency(unittest.TestCase):
    """The prelude compiles into EVERY unit: its own internal call
    delegation must resolve. This is the net that would have caught the
    M3.2 regression (2-D float sub family delegating to a scalar float
    wp_atomic_sub/wp_atomic_sub_ret pair that never existed)."""

    BROKEN_SUB_FAMILY = """\
static inline void wp_atomic_sub_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    wp_atomic_sub((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline float wp_atomic_sub_ret_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    return wp_atomic_sub_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
"""

    FIXED_SUB_FAMILY = BROKEN_SUB_FAMILY + """\
static inline void wp_atomic_sub(device atomic_float* addr, float value) {
    atomic_fetch_sub_explicit(addr, value, memory_order_relaxed);
}
static inline float wp_atomic_sub_ret(device atomic_float* addr, float value) {
    return atomic_fetch_sub_explicit(addr, value, memory_order_relaxed);
}
"""

    def test_stock_prelude_internal_calls_resolve(self):
        findings = _checker.self_check_prelude()
        errs = [f for f in findings if f.severity == "ERROR"]
        self.assertEqual(errs, [], msg="\n".join(repr(e) for e in errs))

    def test_broken_float_sub_family_is_caught(self):
        findings = _checker.self_check_prelude(self.BROKEN_SUB_FAMILY)
        errs = [f for f in findings if f.severity == "ERROR"]
        self.assertEqual(len(errs), 2, msg="\n".join(repr(e) for e in errs))
        self.assertTrue(all("wp_atomic_sub" in e.message for e in errs))

    def test_fixed_float_sub_family_is_clean(self):
        findings = _checker.self_check_prelude(self.FIXED_SUB_FAMILY)
        errs = [f for f in findings if f.severity == "ERROR"]
        self.assertEqual(errs, [], msg="\n".join(repr(e) for e in errs))


class TestEmissionShapes(unittest.TestCase):
    """Emitted-MSL pins: the exact call text per closure atomic form."""

    def test_subtree_com_call_shape(self):
        src = translate_kernel(k_subtree_com_style).source()
        self.assertIn("wp_atomic_add_ret_at(out_data, _wp_args.out_m, ", src)

    def test_vec10_atomic_def_generated_once(self):
        unit = translate_kernel(k_vec10_2d_add)
        src = unit.source()
        self.assertIn("struct vec10f { float c[10]; };", src)
        self.assertIn("static inline vec10f wp_atomic_add_ret_at(device vec10f* d, wp_arr_meta m, int i, int j, vec10f v) {", src)
        # per-unit def: exactly one struct definition, no prelude collision
        self.assertEqual(src.count("struct vec10f {"), 1)

    def test_spatial_inplace_emission(self):
        src = translate_kernel(k_spatial_component_inplace).source()
        self.assertIn("wp_assign_inplace(", src)
        self.assertIn("wp_sub_inplace(", src)

    def test_spatial_broadcast_emission(self):
        src = translate_kernel(k_spatial_scalar_broadcast).source()
        self.assertIn("make_spatial_vectorf(", src)

    def test_mat33_2d_call_shape(self):
        src = translate_kernel(k_mat33_2d_add).source()
        self.assertIn("wp_atomic_add_ret_at(out_data, _wp_args.out_m, ", src)


class TestStructCompareRouting(unittest.TestCase):
    """Struct equality lowers through wp_equal; struct != fails loud.

    vec.h/quat.h define operator== (all components) but NO operator!= —
    the same `a != b` on struct values fails to compile upstream native,
    so mirroring upstream means failing loud here too.
    """

    def test_equal_emits_wp_equal(self):
        lines = translate_kernel(k_spatial_equal_zero).kernel.lines
        self.assertTrue(any("wp_equal(" in l for l in lines))

    def test_not_equal_fails_loud(self):
        from warp_metal.msl.translate import TranslationError

        def _build():
            @wp.kernel
            def k_neq(out: wp.array(dtype=wp.int32)):
                v = vec6(0.0)
                if v != vec6():
                    out[0] = 1
                else:
                    out[0] = 0

            return k_neq

        with self.assertRaises(TranslationError):
            translate_kernel(_build())

    def test_scalar_compare_untouched(self):
        # scalar/bool == stays infix (wp_iter_cmp(var) == 0 range idiom);
        # kernel BODY lines only — the shared prelude carries the wp_equal
        # definitions themselves
        lines = translate_kernel(k_subtree_com_style).kernel.lines
        self.assertFalse(any("wp_equal(" in l for l in lines))

    def test_prelude_equality_signatures(self):
        self.assertIn(
            "static inline bool wp_equal(vec3f a, vec3f b) { return a.x == b.x && a.y == b.y && a.z == b.z; }",
            PRELUDE,
        )
        self.assertIn("static inline bool wp_equal(spatial_vectorf a, spatial_vectorf b) {", PRELUDE)
        self.assertIn("static inline bool wp_equal(quatf a, quatf b) {", PRELUDE)


class TestViewMultiIndex(unittest.TestCase):
    """array.h view() with N indices: per-index strides[k] byte advance,
    meta shifted left by N (vacated slots zero). mjw sensor
    _mark_tactile_contacts views a 2-index row of an array3d for atomic_or."""

    def test_translate_and_checker(self):
        unit = translate_kernel(k_view2d_atomic)
        self.assertEqual([f for f in _checker.check_unit(unit) if f.severity == "ERROR"], [])

    def test_meta_shift_emission(self):
        unit = translate_kernel(k_view2d_atomic)
        src = "\n".join(unit.kernel.lines)
        self.assertIn("_m.ndim = _wp_args.x_m.ndim - 2;", src)
        self.assertIn(".strides[0]))", src)  # per-index byte-advance term
        self.assertIn("_m.strides[1] = _wp_args.x_m.strides[3];", src)  # vacated -> 0 via slot 3


class TestTileScatterAdd(unittest.TestCase):
    """tile.h:4859 collective scatter-add on the Atomic=false branch.

    Every lane calls it; has_value=false skips the write; the trailing sync
    (tile.h:4960) is unconditional. Lowered as `data(i) += v` per lane —
    valid because the closure's only site (mjw solver.py:2709 JTDAJ CSR
    densify) scatters unique in-block addresses.
    """

    def test_emission_and_barrier(self):
        unit = translate_kernel(k_tile_scatter_add, options={"block_dim": _TILE_SCATTER_ADD_BD})
        add_lines = [l for l in unit.kernel.lines if "+=" in l and "_wp_tile_shr_" in l]
        self.assertEqual(len(add_lines), 1)
        self.assertIn("if (", add_lines[0])
        # the trailing barrier follows unconditionally (not nested in the if)
        idx = unit.kernel.lines.index(add_lines[0])
        self.assertIn("threadgroup_barrier", unit.kernel.lines[idx + 1])

    def test_checker_clean(self):
        unit = translate_kernel(k_tile_scatter_add, options={"block_dim": _TILE_SCATTER_ADD_BD})
        findings = _checker.check_unit(unit)
        self.assertEqual([f for f in findings if f.severity == "ERROR"], [])

    def test_plain_launch_scalar_tier(self):
        # plain launch (no block_dim): the tile is the thread's own single
        # element — accumulate only when the index selects element 0 and
        # has_value is set; no barrier (nothing is cross-lane visible)
        unit = translate_kernel(k_tile_scatter_add)
        add_lines = [l for l in unit.kernel.lines if "+=" in l and "_wp_tile_shr_" not in l]
        self.assertEqual(len(add_lines), 1)
        self.assertIn("== 0", add_lines[0])


class TestTileArangeOnesBroadcast(unittest.TestCase):
    """tile.h:2887 arange / :2734 ones / :5488 broadcast alias — the JTDAJ
    producer chain (mjw solver.py:2729-2737): index tile, ones*scalar
    threshold fold, stride-0 broadcast over a transpose view, and the
    ``acc += tile_matmul`` tile_add_inplace accumulate. Shared tiles fill
    with strided lane loops + trailing barrier; the broadcast binds the
    target's pending ``= nullptr`` alias decl to the source's storage
    (upstream owner=False, zero-copy, zero barrier)."""

    def test_tiled_emission(self):
        unit = translate_kernel(k_tile_jtdaj_ops, options={"block_dim": _JTDAJ_OPS_BD})
        lines = unit.kernel.lines
        arange = [l for l in lines if "= 0f + _wp_t * 1f;" in l]
        self.assertEqual(len(arange), 1)
        ones_mul = [l for l in lines if re.search(r"_wp_tile_shr_\d+\[.*\] = \(.* \* .*\);", l)]
        self.assertTrue(ones_mul)
        addi = [l for l in lines if "+= _wp_tile_shr_" in l]
        self.assertEqual(len(addi), 1)
        idx = lines.index(addi[0])
        # the accumulate loop closes before the barrier orders the writes
        self.assertIn("threadgroup_barrier", lines[idx + 2])

    def test_tiled_checker_clean(self):
        unit = translate_kernel(k_tile_jtdaj_ops, options={"block_dim": _JTDAJ_OPS_BD})
        findings = _checker.check_unit(unit)
        self.assertEqual([f for f in findings if f.severity == "ERROR"], [])

    def test_broadcast_alias_borrows_storage(self):
        # the broadcast target (and the transpose view) claim no threadgroup
        # slot of their own — upstream owner=False aliases. Owners: the
        # (4,2) load, the arange tile, the ones tile, the ones*scalar result,
        # the map result, the (2,4) map result, the (2,2) zeros accumulator,
        # the matmul result.
        unit = translate_kernel(k_tile_jtdaj_ops, options={"block_dim": _JTDAJ_OPS_BD})
        self.assertEqual(len(unit.shr_arrays), 8)
        total = sum(_shr_elem_bytes(elem, unit) * size for elem, size in unit.shr_arrays)
        self.assertEqual(total, 160)

    def test_plain_launch_scalar_tier(self):
        # plain launch: arange is the thread's element (index 0), ones is 1,
        # tile+tile folds to the scalar core — per-thread scalar degradation
        unit = translate_kernel(k_tile_arange_scalar)
        lines = unit.kernel.lines
        self.assertTrue(any(re.match(r"^var_\d+ = 0f;$", l.strip()) for l in lines))
        self.assertTrue(any(re.match(r"^var_\d+ = 1f;$", l.strip()) for l in lines))
        self.assertTrue(any("wp_add(" in l for l in lines))
        findings = _checker.check_unit(unit)
        self.assertEqual([f for f in findings if f.severity == "ERROR"], [])


class TestTileLoadIndexed(unittest.TestCase):
    """tile.h:2963 gather: element ``c`` reads ``src[offset + idx[c]]`` with
    BOTH bounds checked — the mjw smooth.py:1361 factorize_block CSR
    densify, where the structurally-absent sentinel (io.py:1224
    ``elemid == nC``) must read as zero without a physical row."""

    def test_shared_gather_emission(self):
        unit = translate_kernel(k_tile_load_indexed, options={"block_dim": 8})
        lines = unit.kernel.lines
        gather = [l for l in lines if ">= 0 &&" in l]
        self.assertEqual(len(gather), 1)
        idx = lines.index(gather[0])
        # the index element comes from the shared index tile, the gather is
        # bounds-checked at both ends against the source, and a barrier
        # orders the fill before the cross-lane cholesky read
        self.assertIn("+ _wp_li;", lines[idx - 1])
        self.assertIn("_wp_tile_shr_0[_wp_t]", lines[idx - 2])
        self.assertIn("shape[0]", gather[0])
        self.assertIn("threadgroup_barrier", lines[idx + 2])

    def test_shared_checker_clean(self):
        unit = translate_kernel(k_tile_load_indexed, options={"block_dim": 8})
        findings = _checker.check_unit(unit)
        self.assertEqual([f for f in findings if f.severity == "ERROR"], [])

    def test_negative_axis_rejected(self):
        # multi-dim gathers have no closure site: fail loud rather than
        # mis-gather (the closure's only indexed loads are 1-D row views)
        from warp_metal.msl.translate import TranslationError

        with self.assertRaises(TranslationError):
            translate_kernel(k_tile_load_indexed_2d, options={"block_dim": 8})

    def test_plain_launch_scalar_tier(self):
        # plain launch: no shared arrays — the thread's element gathers
        # through its own index value with the same both-ends check
        unit = translate_kernel(k_tile_load_indexed)
        lines = unit.kernel.lines
        self.assertFalse(any("_wp_tile_shr_" in l for l in lines))
        self.assertTrue(any(">= 0" in l for l in lines))
        findings = _checker.check_unit(unit)
        self.assertEqual([f for f in findings if f.severity == "ERROR"], [])


class TestTileMapTernary(unittest.TestCase):
    """The bare ``wp::tile_map(op, a, b, c)`` carrier — emit_Call picks it
    for 3-operand maps (mjw constraint._compute_jacp over cdof/offset/
    affects tiles) where the middle operand is a plain scalar (upstream
    broadcasts scalar args per lane, builtins.py b_is_tile)."""

    def test_ternary_user_op_emission(self):
        unit = translate_kernel(k_tile_map_ternary, options={"block_dim": 8})
        lines = unit.kernel.lines
        calls = [l for l in lines if re.search(r"_tile_tern_op_\d+\(.*_wp_tile_shr_", l)]
        self.assertEqual(len(calls), 1)
        # the scalar operand reaches the call site
        self.assertRegex(calls[0], r"\b2\.0f\b|var_\d+")

    def test_ternary_checker_clean(self):
        for bd in (8, None):
            unit = translate_kernel(k_tile_map_ternary, options=None if bd is None else {"block_dim": bd})
            findings = _checker.check_unit(unit)
            self.assertEqual([f for f in findings if f.severity == "ERROR"], [])


class TestClosureVecOverloads(unittest.TestCase):
    """Signature pins: the sweep8 item-2 scalar-op vec surface.

    Upstream sources: vec.h componentwise (neg/dot/add, scalar broadcast
    mul), quat.h operator-, mat.h mul(vec, mat) = v^T M, vec.h
    add_inplace on the spatial (vec_t<6>) form.
    """

    def test_vec5_arithmetic(self):
        self.assertIn("static inline vec5f wp_neg(vec5f a) {", PRELUDE)
        self.assertIn("static inline float wp_dot(vec5f a, vec5f b) {", PRELUDE)
        self.assertIn("static inline vec5f wp_add(vec5f a, vec5f b) {", PRELUDE)
        self.assertIn("static inline vec5f wp_mul(float s, vec5f v) {", PRELUDE)
        self.assertIn("static inline vec5f wp_mul(vec5f v, float s) { return wp_mul(s, v); }", PRELUDE)

    def test_quat_neg(self):
        self.assertIn("static inline quatf wp_neg(quatf q) { return make_quatf(-q.x, -q.y, -q.z, -q.w); }", PRELUDE)

    def test_row_vector_times_mat33(self):
        self.assertIn("static inline vec3f wp_mul(vec3f b, mat33f a) {", PRELUDE)

    def test_spatial_add_inplace(self):
        self.assertIn(
            "static inline void wp_add_inplace(thread spatial_vectorf& v, int i, float x) {",
            PRELUDE,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
