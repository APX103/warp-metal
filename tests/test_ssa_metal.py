# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M1.3 numeric conformance: Adjoint-SSA translated kernels vs CPU golden.

Each test launches a feature-covering kernel on the Metal device and checks
results against an independent numpy computation of the same semantics.
Skips cleanly when no Metal device is present (CI without Apple silicon).
"""

import unittest

import numpy as np

import warp_metal as wp
from warp_metal.runtime import devices as _devices


TOL = 1e-5

_SKIP = not _devices.is_metal_available()
_SKIP_MSG = "no Metal device"


class TestSSAKernels(unittest.TestCase):
    def test_tid_and_scalar_params(self):
        @wp.kernel
        def k(n: int, scale: float, out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i) * scale + wp.float(n)

        n, scale = 8, 2.5
        out = wp.empty(n, dtype=wp.float32, device="metal")
        wp.launch(k, dim=n, inputs=[n, scale, out], device="metal")
        wp.synchronize()
        expected = np.arange(n) * scale + n
        np.testing.assert_allclose(out.numpy_view(), expected, rtol=TOL)

    def test_2d_tid_and_array2d(self):
        @wp.kernel
        def k(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
            i, j = wp.tid()
            v = a[i, j]
            out[i, j] = v * wp.float(i) + wp.float(j)

        m, n = 4, 6
        a = wp.zeros((m, n), dtype=wp.float32, device="metal")
        a.numpy_view()[:] = np.arange(m * n, dtype=np.float32).reshape(m, n)
        out = wp.empty((m, n), dtype=wp.float32, device="metal")
        wp.launch(k, dim=(m, n), inputs=[a, out], device="metal")
        wp.synchronize()
        expected = np.arange(m * n, dtype=np.float32).reshape(m, n) * np.arange(m)[:, None] + np.arange(n)[None, :]
        np.testing.assert_allclose(out.numpy_view(), expected, rtol=TOL)

    def test_if_for_while_and_wp_func(self):
        @wp.func
        def scale_add(v: wp.vec3, s: float):
            return v * s

        @wp.kernel
        def k(a: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
            i = wp.tid()
            v = a[i]
            acc = wp.vec3()
            for kk in range(3):
                acc = acc + scale_add(v, wp.float(kk))
            x = acc[1]
            while x > 0.5:
                x = x - 0.5
            if x > 0.25:
                out[i] = acc
            else:
                out[i] = wp.vec3(x, -x, 0.0)

        n = 5
        rng = np.random.default_rng(7)
        data = rng.standard_normal((n, 3)).astype(np.float32)
        a = wp.zeros(n, dtype=wp.vec3, device="metal")
        a.numpy_view()[:] = data
        out = wp.empty(n, dtype=wp.vec3, device="metal")

        wp.launch(k, dim=n, inputs=[a, out], device="metal")
        wp.synchronize()

        got = out.numpy_view().reshape(n, 3)
        expected = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            v = data[i]
            acc = sum(v * float(kk) for kk in range(3))  # scale_add, 3 iterations
            x = acc[1]
            while x > 0.5:
                x -= 0.5
            if x > 0.25:
                expected[i] = acc
            else:
                expected[i] = (x, -x, 0.0)
        np.testing.assert_allclose(got, expected, rtol=TOL)

    def test_atomics_and_transform(self):
        @wp.kernel
        def k(
            xf: wp.array(dtype=wp.transform),
            pts: wp.array(dtype=wp.vec3),
            out: wp.array(dtype=wp.vec3),
            counts: wp.array(dtype=wp.int32),
            lo: wp.array(dtype=wp.float32),
        ):
            i = wp.tid()
            t = xf[i]
            p = wp.transform_point(t, pts[i])
            out[i] = p
            wp.atomic_add(counts, i, 2)
            wp.atomic_min(lo, i, p[0])

        n = 4
        rng = np.random.default_rng(11)
        q = rng.standard_normal((n, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        p0 = rng.standard_normal((n, 3)).astype(np.float32)
        trans = np.concatenate([p0, q], axis=1)  # warp transform layout: p, q(xyzw)

        xf = wp.zeros(n, dtype=wp.transform, device="metal")
        xf.numpy_view()[:] = trans
        pts = wp.zeros(n, dtype=wp.vec3, device="metal")
        pts.numpy_view()[:] = p0 * 2.0
        out = wp.empty(n, dtype=wp.vec3, device="metal")
        counts = wp.zeros(n, dtype=wp.int32, device="metal")
        lo = wp.full(n, 1e30, dtype=wp.float32, device="metal")

        wp.launch(k, dim=n, inputs=[xf, pts, out, counts, lo], device="metal")
        wp.synchronize()

        # numpy golden: rotate v by quat q (xyzw), add t (warp quat_rotate formula)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        v = p0 * 2.0
        c = 2.0 * w * w - 1.0
        d = 2.0 * (x * v[:, 0] + y * v[:, 1] + z * v[:, 2])
        rx = v[:, 0] * c + x * d + (y * v[:, 2] - z * v[:, 1]) * w * 2.0
        ry = v[:, 1] * c + y * d + (z * v[:, 0] - x * v[:, 2]) * w * 2.0
        rz = v[:, 2] * c + z * d + (x * v[:, 1] - y * v[:, 0]) * w * 2.0
        expected = p0 + np.stack([rx, ry, rz], axis=1)

        np.testing.assert_allclose(out.numpy_view(), expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(counts.numpy_view(), np.full(n, 2))
        np.testing.assert_allclose(lo.numpy_view(), expected[:, 0], rtol=1e-4, atol=1e-6)


    def test_struct_and_value_params(self):
        @wp.struct
        class Particle:
            pos: wp.vec3
            mass: wp.float32
            flags: wp.int32

        @wp.kernel
        def k(xf: wp.transform, p: Particle, out: wp.array(dtype=wp.vec3), n: int):
            i = wp.tid()
            local = wp.transform_point(xf, p.pos * wp.float(i))
            out[i] = local + wp.vec3(wp.float(p.flags), p.mass, wp.float(n))

        n = 4
        xform = wp.transform()
        for i, v in enumerate([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]):  # translate(1,2,3), quat identity
            xform[i] = v
        pt = Particle()
        pt.pos = wp.vec3(1.0, 2.0, 3.0)
        pt.mass = 4.5
        pt.flags = 10

        out = wp.zeros(n, dtype=wp.vec3, device="metal")
        wp.launch(k, dim=n, inputs=[xform, pt, out, n], device="metal")
        wp.synchronize()

        p0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            local = p0 * float(i) + np.array([1.0, 2.0, 3.0], dtype=np.float32)  # transform_point
            expected[i] = local + np.array([10.0, 4.5, float(n)], dtype=np.float32)
        np.testing.assert_allclose(out.numpy_view(), expected, rtol=TOL, atol=1e-6)

    def test_mat6x6_spatial_mul_transpose(self):
        # generic-size square mat ops (ensure_mat_def): Newton featherstone
        # spatial-inertia mat6x6f wp.mul/wp.transpose vs prelude mat22-44.
        @wp.kernel
        def k(
            a: wp.array(dtype=wp.spatial_matrix),
            x: wp.array(dtype=wp.spatial_vector),
            mm: wp.array(dtype=wp.spatial_matrix),
            mv: wp.array(dtype=wp.spatial_vector),
            mt: wp.array(dtype=wp.spatial_matrix),
        ):
            i = wp.tid()
            m1 = a[i]
            m2 = a[i + 1]
            v = x[i]
            mm[i] = wp.mul(m1, m2)
            mv[i] = wp.mul(m1, v)
            mt[i] = wp.transpose(m1)

        rng = np.random.default_rng(7)
        n = 3
        Amat = rng.standard_normal((n + 1, 6, 6)).astype(np.float32)
        Xvec = rng.standard_normal((n, 6)).astype(np.float32)
        a = wp.empty(n + 1, dtype=wp.spatial_matrix, device="metal")
        a.numpy_view()[:] = Amat.reshape(n + 1, 36)
        x = wp.empty(n, dtype=wp.spatial_vector, device="metal")
        x.numpy_view()[:] = Xvec
        mm = wp.empty(n, dtype=wp.spatial_matrix, device="metal")
        mv = wp.empty(n, dtype=wp.spatial_vector, device="metal")
        mt = wp.empty(n, dtype=wp.spatial_matrix, device="metal")
        wp.launch(k, dim=n, inputs=[a, x, mm, mv, mt], device="metal")
        wp.synchronize()
        np.testing.assert_allclose(
            mm.numpy_view().reshape(n, 6, 6), Amat[:n] @ Amat[1:], rtol=TOL, atol=TOL
        )
        np.testing.assert_allclose(
            mv.numpy_view().reshape(n, 6), (Amat[:n] @ Xvec[..., None])[..., 0], rtol=TOL, atol=TOL
        )
        np.testing.assert_allclose(
            mt.numpy_view().reshape(n, 6, 6), Amat[:n].transpose(0, 2, 1), rtol=TOL, atol=TOL
        )

    def test_step1_and_spatial_neg(self):
        # 1-arg wp.step overload (1.0 if x < 0, upstream builtins.py doc) and
        # wp.neg on spatial_vectorf (featherstone body_f_ext negation).
        @wp.kernel
        def k(
            x: wp.array(dtype=wp.float32),
            s: wp.array(dtype=wp.spatial_vector),
            o1: wp.array(dtype=wp.float32),
            o2: wp.array(dtype=wp.spatial_vector),
        ):
            i = wp.tid()
            o1[i] = wp.step(x[i])
            o2[i] = wp.neg(s[i])

        n = 4
        X = np.array([-2.0, -0.5, 0.0, 3.0], dtype=np.float32)
        S = rng = np.arange(n * 6, dtype=np.float32) - 4.0
        x = wp.empty(n, dtype=wp.float32, device="metal")
        x.numpy_view()[:] = X
        s = wp.empty(n, dtype=wp.spatial_vector, device="metal")
        s.numpy_view()[:] = S.reshape(n, 6)
        o1 = wp.empty(n, dtype=wp.float32, device="metal")
        o2 = wp.empty(n, dtype=wp.spatial_vector, device="metal")
        wp.launch(k, dim=n, inputs=[x, s, o1, o2], device="metal")
        wp.synchronize()
        np.testing.assert_allclose(o1.numpy_view(), (X < 0).astype(np.float32), rtol=TOL)
        np.testing.assert_allclose(o2.numpy_view().reshape(n, 6), -S.reshape(n, 6), rtol=TOL)

    def test_mod_i64_length_spatial_atomic_mat(self):
        # int64 wp.mod (ambiguity fix), wp.length on spatial_vectorf, and
        # atomic_add on a mat6x6f array (joint_damping body_I_c accumulation).
        @wp.kernel
        def k(
            a: wp.array(dtype=wp.int64),
            b: wp.array(dtype=wp.int64),
            om: wp.array(dtype=wp.int64),
            s: wp.array(dtype=wp.spatial_vector),
            ln: wp.array(dtype=wp.float32),
            m: wp.array(dtype=wp.spatial_matrix),
            am: wp.array(dtype=wp.spatial_matrix),
        ):
            i = wp.tid()
            om[i] = wp.mod(a[i], b[i])
            ln[i] = wp.length(s[i])
            wp.atomic_add(m, i, am[i])

        a = wp.empty(2, dtype=wp.int64, device="metal")
        a.numpy_view()[:] = np.array([17, -17], dtype=np.int64)
        b = wp.empty(2, dtype=wp.int64, device="metal")
        b.numpy_view()[:] = np.array([5, 5], dtype=np.int64)
        om = wp.empty(2, dtype=wp.int64, device="metal")
        s = wp.empty(2, dtype=wp.spatial_vector, device="metal")
        s.numpy_view()[:] = np.array([[3.0, 4.0, 0, 0, 0, 0], [1, 2, 2, 0, 0, 0]], dtype=np.float32)
        ln = wp.empty(2, dtype=wp.float32, device="metal")
        m = wp.zeros(2, dtype=wp.spatial_matrix, device="metal")
        am = wp.empty(2, dtype=wp.spatial_matrix, device="metal")
        am.numpy_view()[:] = np.tile(np.eye(6, dtype=np.float32) * 2.0, (2, 1)).reshape(2, 36)
        wp.launch(k, dim=2, inputs=[a, b, om, s, ln, m, am], device="metal")
        wp.synchronize()
        np.testing.assert_array_equal(om.numpy_view(), np.array([2, -2], dtype=np.int64))
        np.testing.assert_allclose(ln.numpy_view(), [5.0, 3.0], rtol=TOL)
        np.testing.assert_allclose(
            m.numpy_view().reshape(2, 6, 6), np.tile(np.eye(6) * 2.0, (2, 1)).reshape(2, 6, 6), rtol=TOL, atol=TOL
        )


if __name__ == "__main__":
    unittest.main()
