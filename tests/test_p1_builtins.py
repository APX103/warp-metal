# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Translate-only pins for the P1 builtin gaps (KEP-0001 M3.2).

``wp.identity`` / ``wp.diag`` / ``wp.determinant`` / ``wp.matrix`` /
``wp.vector`` lower through the same dispatch tables as the fixed-size
family: identity becomes a zero-argument ``wp_identityN()`` factory in the
shared prelude, diag and determinant become ``wp_diag`` /
``wp_determinant`` helpers whose arithmetic mirrors upstream mat.h
verbatim (2x2 row-pair formula; 3x3 as ``dot(row0, cross(row1, row2))``;
no 4x4 overload — upstream accumulates in fp64 intermediates), and generic
(non-square / int) ``wp.matrix`` results get the per-unit ``matMxNi`` /
``matMxNf`` struct, row-major ``make_matMxN`` constructor, and ``wp_where``
select emitted on demand. Host-only: every test translates a kernel to MSL
text and pins the emitted names; no Metal device is created, so the module
runs under CPU-only verification. Anything outside the dialect (identity
beyond 4x4 or with a non-fp32 dtype, fp64 anywhere) fails loud as a
``TranslationError`` rather than silently mistranslating.
"""

import re
import unittest

import warp_metal as wp
from warp_metal.msl.translate import TranslationError, translate_kernel


# ---- kernels (translate-only; never launched) ----


@wp.kernel
def k_identity3(out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.identity(n=3, dtype=wp.float32)


@wp.kernel
def k_identity24(out: wp.array(dtype=wp.mat22), out2: wp.array(dtype=wp.mat44)):
    i = wp.tid()
    out[i] = wp.identity(n=2, dtype=wp.float32)
    out2[i] = wp.identity(n=4, dtype=wp.float32)


@wp.kernel
def k_diag(out: wp.array(dtype=wp.mat22), out2: wp.array(dtype=wp.mat33), out3: wp.array(dtype=wp.mat44)):
    i = wp.tid()
    out[i] = wp.diag(wp.vec2(1.0, 2.0))
    out2[i] = wp.diag(wp.vec3(1.0, 2.0, 3.0))
    out3[i] = wp.diag(wp.vec4(1.0, 2.0, 3.0, 4.0))


@wp.kernel
def k_determinant(a: wp.array(dtype=wp.mat22), b: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[2 * i] = wp.determinant(a[i])
    out[2 * i + 1] = wp.determinant(b[i])


@wp.func
def flex_endpoints(dim: int):
    # mujoco_warp _flex_rod_hessian endpoint table: an int (6, 2) wp.matrix
    # selected by wp.where — the shape that forced generic-int-mat support.
    return wp.where(
        dim == 3,
        wp.matrix(0, 1, 1, 2, 2, 0, 2, 3, 0, 3, 1, 3, shape=(6, 2), dtype=int),
        wp.matrix(1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),
    )


@wp.kernel
def k_int_matrix(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    e = flex_endpoints(wp.int32(3))
    out[i] = e[i % 6, 0] + e[(i + 3) % 6, 1]


@wp.kernel
def k_float_matrix66(out: wp.array(dtype=wp.matrix(shape=(6, 6), dtype=float))):
    i = wp.tid()
    out[i] = wp.matrix(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        shape=(6, 6), dtype=wp.float32,
    )


@wp.kernel
def k_vector(out: wp.array(dtype=wp.vec3), outi: wp.array(dtype=wp.vec2i)):
    i = wp.tid()
    out[i] = wp.vector(1.0, 2.0, 3.0)
    outi[i] = wp.vector(1, 2, dtype=wp.int32)


def _strip_msl_comments(src: str) -> str:
    """Drop // comment tails so prelude doc mentions of 'double' (upstream
    mat.h notes) cannot mask a code-level fp64 leak."""
    return "\n".join(line.split("//")[0] for line in src.splitlines())


class TestP1BuiltinEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {
            "identity3": translate_kernel(k_identity3).source(),
            "identity24": translate_kernel(k_identity24).source(),
            "diag": translate_kernel(k_diag).source(),
            "determinant": translate_kernel(k_determinant).source(),
            "int_matrix": translate_kernel(k_int_matrix).source(),
            "float_matrix66": translate_kernel(k_float_matrix66).source(),
            "vector": translate_kernel(k_vector).source(),
        }

    def test_identity_factories(self):
        src = self.sources["identity3"]
        self.assertRegex(src, r"= wp_identity3\(\);")
        self.assertRegex(src, r"static inline mat33f wp_identity3\(\)")
        src2 = self.sources["identity24"]
        self.assertRegex(src2, r"= wp_identity2\(\);")
        self.assertRegex(src2, r"= wp_identity4\(\);")

    def test_identity_factory_is_zero_arg(self):
        # exactly one def + one zero-arg call site; the unit diagonal is
        # baked into the shared prelude, not passed per call
        self.assertEqual(self.sources["identity3"].count("wp_identity3"), 2)

    def test_diag_overloads(self):
        src = self.sources["diag"]
        self.assertEqual(src.count("wp_diag(var_"), 3)
        self.assertIn("static inline mat22f wp_diag(vec2f d)", src)
        self.assertIn("static inline mat33f wp_diag(vec3f d)", src)
        self.assertIn("static inline mat44f wp_diag(vec4f d)", src)

    def test_determinant_22_verbatim(self):
        src = self.sources["determinant"]
        self.assertRegex(src, r"= wp_determinant\(var_\d+\);")
        self.assertEqual(src.count("wp_determinant(var_"), 2)
        # row-pair formula, character-for-character upstream mat.h
        self.assertIn("m.c[0][0] * m.c[1][1] - m.c[1][0] * m.c[0][1]", src)

    def test_determinant_33_verbatim(self):
        src = self.sources["determinant"]
        # dot(row0, cross(row1, row2)) with row extraction in mat.h order
        self.assertRegex(
            src,
            r"wp_dot\(\s*"
            r"make_vec3f\(m\.c\[0\]\[0\], m\.c\[0\]\[1\], m\.c\[0\]\[2\]\),\s*"
            r"wp_cross\(\s*"
            r"make_vec3f\(m\.c\[1\]\[0\], m\.c\[1\]\[1\], m\.c\[1\]\[2\]\),\s*"
            r"make_vec3f\(m\.c\[2\]\[0\], m\.c\[2\]\[1\], m\.c\[2\]\[2\]\)\)\);",
        )

    def test_no_4x4_determinant_overload(self):
        # upstream's 4x4 determinant accumulates in fp64 intermediates; the
        # prelude deliberately carries no wp_determinant(mat44f) overload
        # (the prelude's own comment explaining that mentions the name, so
        # pin against comment-stripped source)
        self.assertNotIn("wp_determinant(mat44f", _strip_msl_comments(self.sources["determinant"]))

    def test_int_matrix_6x2_machinery(self):
        src = self.sources["int_matrix"]
        self.assertIn("struct mat6x2i { int c[6][2]; };", src)
        # row-major constructor: a00 a01 | a10 a11 | ...
        self.assertIn("make_mat6x2i(int a00, int a01, int a10, int a11,", src)
        self.assertIn("v.c[0][0] = a00;", src)
        self.assertIn("v.c[0][1] = a01;", src)
        self.assertRegex(src, r"wp_where\(bool c, mat6x2i")

    def test_float_matrix_66_row_major(self):
        src = self.sources["float_matrix66"]
        self.assertRegex(src, r"mat6x6f var_\d+ = \{\};")
        self.assertRegex(src, r"make_mat6x6f\((?:var_\d+, ){35}var_\d+\);")
        self.assertIn("v.c[0][1] = a01;", src)

    def test_vector_ctors(self):
        src = self.sources["vector"]
        self.assertRegex(src, r"make_vec3f\(var_\d+, var_\d+, var_\d+\)")
        self.assertRegex(src, r"make_vec2i\(var_\d+, var_\d+\)")

    def test_no_fp64_anywhere(self):
        for name, src in self.sources.items():
            with self.subTest(kernel=name):
                stripped = _strip_msl_comments(src)
                self.assertNotRegex(stripped, r"\bdouble\b")
                self.assertNotIn("float64", stripped)


class TestP1FailLoud(unittest.TestCase):
    def test_identity_beyond_4x4(self):
        def build():
            @wp.kernel
            def k(out: wp.array(dtype=wp.float32)):
                i = wp.tid()
                m = wp.identity(n=5, dtype=wp.float32)
                out[i] = m[0, 0]

            translate_kernel(k)

        with self.assertRaisesRegex(TranslationError, r"wp::identity<5, wp::float32>"):
            build()

    def test_identity_int_dtype(self):
        def build():
            @wp.kernel
            def k(out: wp.array(dtype=wp.int32)):
                i = wp.tid()
                m = wp.identity(n=3, dtype=wp.int32)
                out[i] = m[0, 0]

            translate_kernel(k)

        with self.assertRaisesRegex(TranslationError, r"wp::identity<3, wp::int32>"):
            build()

    def test_identity_float64(self):
        def build():
            @wp.kernel
            def k(out: wp.array(dtype=wp.float32)):
                i = wp.tid()
                m = wp.identity(n=3, dtype=wp.float64)
                out[i] = wp.float32(m[0, 0])

            translate_kernel(k)

        with self.assertRaisesRegex(TranslationError, r"wp::mat_t<3, 3, wp::float64>"):
            build()

    def test_fp64_array_param(self):
        def build():
            @wp.kernel
            def k(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float32)):
                i = wp.tid()
                out[i] = wp.float32(a[i])

            translate_kernel(k)

        with self.assertRaisesRegex(TranslationError, r"wp::float64"):
            build()

    def test_fp64_matrix(self):
        def build():
            @wp.kernel
            def k(out: wp.array(dtype=wp.float32)):
                i = wp.tid()
                m = wp.matrix(1.0, 0.0, 0.0, 1.0, shape=(2, 2), dtype=wp.float64)
                out[i] = wp.float32(m[0, 0])

            translate_kernel(k)

        with self.assertRaisesRegex(TranslationError, r"wp::mat_t<2, 2, wp::float64>"):
            build()


if __name__ == "__main__":
    unittest.main()
