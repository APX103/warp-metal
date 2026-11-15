# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Translate-only pins for the ``wp.eig3`` MSL lowering (KEP-0001 M3.2).

``wp.eig3`` is the symmetric 3x3 eigendecomposition Newton's mujoco glue
needs (``_eig_inertia`` -> ``update_body_inertia_kernel``). Both upstream
overloads — the ``(Q, d)``-returning one and the caller-output mutating one
— reach the SSA as the same statement call ``wp::eig3(A, Q, d);`` with the
outputs as trailing ref args, so one routing covers both and the argument
order pins the output contract: ret/out 0 is the eigenvector basis ``Q``,
ret/out 1 the eigenvalue vector ``d`` (upstream svd.h eig3: ``A = Q *
diag(d) * Q^T``; columns of Q are paired right-handed eigenvectors; no
ordering is promised and Newton sorts descending itself). The MSL is a
verbatim port of warp/native/svd.h (quaternion-Jacobi, JACOBI_ITERATIONS=4,
``recipSqrt`` CPU branch ``1/sqrt(x)``) into the shared prelude section
``_EIG3_OPS``; fp32 only — an fp64 call fails loud at translation (Metal
has no fp64; Newton's fork-side B2 decision decomposes in float32 already).

Host-only: every test translates a kernel to MSL text and pins the emitted
names / upstream arithmetic; no Metal device is created, so the module runs
under CPU-only verification. Numerical validation of the port happens in
the GPU battery (the mujoco-glue corpus rows); there is no host execution
path for MSL.

Also pinned here: the aggregate COPY ctors the same Newton chain materializes
(``_eig_inertia`` returns ``wp.mat33(vectors), wp.vec3(values)`` — SSA
``wp::mat_t<3, 3, f32>(var)`` / ``wp::vec_t<3, f32>(var)`` through the shared
``_CTOR_TABLE`` path). Those emit ``make_mat33f(var)`` / ``make_vec3f(var)``
single-arg calls, which only compile if the prelude carries copy overloads
alongside the scalar-fill / element overloads; the copy ctor is an EXACT
memberwise copy (upstream C++ ``mat_t(mat_t)`` / ``vec_t(vec_t)``), pinned as
the identity-return overload signature. The translator cannot type-check SSA
ctor args (single-arg may be scalar fill OR copy), so MSL overload resolution
mirrors upstream C++ resolution — the same coverage is asserted for the whole
emittable family (mat22/33/44, vec2/3/4/5f, spatial_vector, quat, transform,
vec2/3/4i) and for ``ensure_mat_def``'s generated generic mats.
"""

import re
import unittest

import warp_metal as wp
from warp_metal.msl.translate import TranslationError, translate_kernel


# ---- kernels (translate-only; never launched) ----


@wp.func
def _eig_inertia_shape(inertia: wp.mat33):
    # Newton newton/_src/geometry/inertia.py _eig_inertia core shape: the
    # return-value overload unpacked to (vectors, values) inside a @wp.func
    # that itself returns the pair
    vectors, values = wp.eig3(inertia)
    return wp.mat33(vectors), wp.vec3(values)


@wp.kernel
def k_eig_return(out: wp.array(dtype=wp.mat33), vals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    Q, d = _eig_inertia_shape(wp.mat33(4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0))
    out[i] = Q
    vals[i] = d


@wp.kernel
def k_eig_mutating(mats: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33), vals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    Q = wp.mat33()
    d = wp.vec3()
    wp.eig3(mats[i], Q, d)
    out[i] = Q
    vals[i] = d


@wp.kernel
def k_eig_fp64(out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    Q, d = wp.eig3(wp.mat33d(4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0))
    out[i] = wp.float32(d[0])


@wp.kernel
def k_copies(
    m3: wp.array(dtype=wp.mat33),
    v3: wp.array(dtype=wp.vec3),
    out_m: wp.array(dtype=wp.mat33),
    out_v: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    # Newton _eig_inertia return shape: aggregate-from-aggregate ctor
    a, b = wp.mat33(m3[i]), wp.vec3(v3[i])
    out_m[i] = a
    out_v[i] = b


@wp.kernel
def k_copies_more(
    m4: wp.array(dtype=wp.mat44),
    q: wp.array(dtype=wp.quat),
    t: wp.array(dtype=wp.transform),
    v2i: wp.array(dtype=wp.vec2i),
    out4: wp.array(dtype=wp.mat44),
    outq: wp.array(dtype=wp.quat),
    outt: wp.array(dtype=wp.transform),
    out2i: wp.array(dtype=wp.vec2i),
):
    i = wp.tid()
    out4[i] = wp.mat44(m4[i])
    outq[i] = wp.quat(q[i])
    outt[i] = wp.transform(t[i])
    out2i[i] = wp.vec2i(v2i[i])


@wp.kernel
def k_generic_mat_int(out: wp.array(dtype=wp.int32)):
    # mujoco_warp-style 6x2 int matrix: forces the ensure_mat_def generated def
    i = wp.tid()
    e = wp.where(
        i == 0,
        wp.matrix(0, 1, 1, 2, 2, 0, 2, 3, 0, 3, 1, 3, shape=(6, 2), dtype=int),
        wp.matrix(1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),
    )
    out[i] = e[i % 6, 0]


def _strip_msl_comments(src: str) -> str:
    """Drop // comment tails so prelude doc mentions of 'fp64'/'double'
    (upstream svd.h notes) cannot mask a code-level fp64 leak."""
    return "\n".join(line.split("//")[0] for line in src.splitlines())


class TestEig3Emission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {
            "return": translate_kernel(k_eig_return).source(),
            "mutating": translate_kernel(k_eig_mutating).source(),
        }
        cls.stripped = {k: _strip_msl_comments(v) for k, v in cls.sources.items()}

    def test_statement_call_routing(self):
        # both overloads emit exactly one wp_eig3 statement call with the
        # three SSA args (A, out-Q, out-d) passed through in order
        for name, src in self.stripped.items():
            calls = re.findall(r"^wp_eig3\(([^;]*)\);$", src, flags=re.MULTILINE)
            self.assertEqual(len(calls), 1, f"{name}: expected one wp_eig3 statement call")
            args = [a.strip() for a in calls[0].split(",")]
            self.assertEqual(len(args), 3, f"{name}: wp_eig3 call must carry A, Q, d")
            for arg in args:
                self.assertRegex(arg, r"^var_\w+$", f"{name}: arg {arg!r} must be an SSA variable")

    def test_output_contract_signature(self):
        # the prelude signature pins the upstream contract: (Q, d) with the
        # eigenvector basis FIRST, eigenvalues second
        for name, src in self.sources.items():
            self.assertIn(
                "static inline void wp_eig3(mat33f A, thread mat33f& Q, thread vec3f& d)",
                src,
                f"{name}: wp_eig3 signature changed — that is the (Q, d) output contract",
            )

    def test_upstream_port_pins(self):
        # verbatim svd.h arithmetic (warp/native/svd.h eig3 family)
        for name, src in self.sources.items():
            with self.subTest(name):
                # approximateGivensQuaternion constants (svd.h:204-206) at
                # the float values the C++ Type() casts produce
                self.assertIn("const float _gamma = 5.82842712474619f;", src)
                self.assertIn("const float _cstar = 0.9238795325112867f;", src)
                self.assertIn("const float _sstar = 0.3826834323650898f;", src)
                # conjugation S = Q'*S*Q (svd.h:238-240)
                self.assertIn("s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);", src)
                self.assertIn("s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);", src)
                self.assertIn("s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);", src)
                # cumulative quaternion update (svd.h:259-262)
                self.assertIn("qV[z] += sh;", src)
                self.assertIn("qV[3] -= tmp[z];", src)
                # quatToMat3 layout entries (svd.h:186-194)
                self.assertIn("m.c[0][0] = 1.0f - 2.0f * (qyy + qzz);", src)
                self.assertIn("m.c[2][1] = 2.0f * (qyz + qwx);", src)
                # fixed iteration count (svd.h _svd_config<float>: JACOBI_ITERATIONS = 4)
                self.assertIn("for (int i = 0; i < 4; ++i)", src)
                # d = diag(Q^T*A*Q) (svd.h:763)
                self.assertIn("d = make_vec3f(u.c[0][0], u.c[1][1], u.c[2][2]);", src)
                # recipSqrt CPU branch (svd.h:53), not the CUDA rsqrt fast path
                self.assertIn("float w = 1.0f / sqrt(ch * ch + sh * sh);", src)
                self.assertNotIn("rsqrt(", src)

    def test_no_fp64_anywhere(self):
        for name, src in self.stripped.items():
            self.assertNotIn("float64", src, f"{name}: fp64 leaked into emitted MSL")
            self.assertNotIn("double", src, f"{name}: fp64 leaked into emitted MSL")


class TestEig3FailLoud(unittest.TestCase):
    def test_fp64_overload_raises(self):
        # an fp64 eig3 call fails loud with the offending C++ type in the
        # message (fp64 SSA vars have no MSL mapping — same discipline as
        # the P1 fp64 pins); Newton's fork-side B2 decision decomposes in
        # float32, so no supported path reaches an fp64 eig3
        with self.assertRaises(TranslationError) as ctx:
            translate_kernel(k_eig_fp64)
        self.assertIn("float64", str(ctx.exception))


class TestAggregateCopyCtors(unittest.TestCase):
    """The copy-ctor layer under the eig3 chain (KEP-0001 M3.2 follow-up).

    _eig_inertia materializes its return with aggregate-from-aggregate ctors
    (wp.mat33(vectors) / wp.vec3(values)); the shared _CTOR_TABLE path emits
    make_<type>(var) single-arg calls, which resolve only against prelude
    copy overloads — the exact memberwise-copy semantics of upstream C++
    mat_t(mat_t)/vec_t(vec_t), pinned here as identity-return signatures.
    """

    @classmethod
    def setUpClass(cls):
        cls.sources = {
            "copies": translate_kernel(k_copies).source(),
            "copies_more": translate_kernel(k_copies_more).source(),
            "generic_mat": translate_kernel(k_generic_mat_int).source(),
        }

    def _assert_copy_overload(self, src: str, msl_t: str):
        # identity-return signature IS the exact memberwise-copy semantics;
        # the parameter name is semantic (v/q/t/s/m), so match its shape
        self.assertRegex(src, rf"static inline {msl_t} make_{msl_t}\({msl_t} \w+\) \{{ return \w+; \}}", msl_t)

    def test_newton_chain_copy_emissions(self):
        # the exact emissions the GPU battery tripped on: single-arg
        # make_mat33f/make_vec3f calls must now resolve against copy overloads
        src = self.sources["copies"]
        self.assertRegex(src, r"var_\d+ = make_mat33f\(var_\d+\);")
        self.assertRegex(src, r"var_\d+ = make_vec3f\(var_\d+\);")
        for msl_t in ("mat33f", "vec3f"):
            self._assert_copy_overload(src, msl_t)

    def test_ctor_table_family_copy_overloads(self):
        # the whole emittable _CTOR_TABLE family carries copy overloads
        src = self.sources["copies_more"]
        for arg in ("var_3", "var_6", "var_9", "var_12"):
            self.assertRegex(src, rf"var_\d+ = make_\w+\({arg}\);")
        for msl_t in ("mat44f", "quatf", "transformf", "vec2i"):
            self._assert_copy_overload(src, msl_t)

    def test_generic_mat_generated_copy_overload(self):
        # ensure_mat_def's generated ctor table gets the same copy overload
        self.assertIn("static inline mat6x2i make_mat6x2i(mat6x2i m) { return m; }", self.sources["generic_mat"])

    def test_scalar_fill_still_distinct(self):
        # copy overloads must not have displaced the scalar-fill / splat
        # forms (wp.mat33(0.0), wp.vec3(0.0) in the _eig_inertia zero branch)
        src = self.sources["copies"]
        self.assertIn("static inline mat33f make_mat33f(float s)", src)
        self.assertIn("static inline vec3f make_vec3f(float s)", src)


if __name__ == "__main__":
    unittest.main()
