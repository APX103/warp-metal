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
import unittest
from pathlib import Path

import warp_metal as wp
from warp_metal.msl.prelude import PRELUDE
from warp_metal.msl.translate import translate_kernel

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

    def test_spatial_component_inplace(self):
        self.assertIn("static inline void wp_assign_inplace(thread spatial_vectorf& s, int i, float x) {", PRELUDE)
        self.assertIn("static inline void wp_sub_inplace(thread spatial_vectorf& s, int i, float x) {", PRELUDE)

    def test_spatial_scalar_broadcast(self):
        self.assertIn("static inline spatial_vectorf make_spatial_vectorf(float s) {", PRELUDE)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
