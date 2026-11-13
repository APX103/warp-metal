# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M0.4 acceptance: buffers, zero-copy numpy views, GPU dispatch.

Skips when no Metal device is available (CPU-only CI).
"""

import unittest

import numpy as np

import warp_metal as wp
from warp_metal import native as _mm
from warp_metal.msl.spike_kernels import k_atomic_accum, k_copy_scale, k_dot, k_trig


class TestMetalRuntime(unittest.TestCase):
    DEVICE = None

    @classmethod
    def setUpClass(cls):
        try:
            cls.DEVICE = _mm.device_name()
        except Exception:
            raise unittest.SkipTest("no Metal device available")

    def test_device_present(self):
        self.assertTrue(self.DEVICE)
        print(f"Metal device: {self.DEVICE}")

    def test_buffer_zero_copy_numpy_view(self):
        arr = wp.zeros(16, dtype=wp.float32)
        view = arr.numpy_view()
        view[:] = np.arange(16, dtype=np.float32)
        # write via the view, read via a second view over the same buffer
        view2 = arr.numpy_view()
        self.assertTrue(np.array_equal(view, view2))
        self.assertEqual(arr.buffer, arr.buffer)  # stable handle

    def test_composite_view_shape(self):
        arr = wp.zeros(4, dtype=wp.vec3f)
        v = arr.numpy_view()
        self.assertEqual(v.shape, (4, 3))
        v[:] = 1.5
        self.assertAlmostEqual(float(arr.numpy[0]), 1.5)

    def test_launch_copy_scale(self):
        n = 32
        a = wp.zeros(n, dtype=wp.float32)
        out = wp.zeros(n, dtype=wp.float32)
        a.numpy_view()[:] = np.arange(n, dtype=np.float32)
        wp.launch(k_copy_scale, dim=n, inputs=[a, out, 2.5], device="metal")
        np.testing.assert_allclose(out.numpy_view(), np.arange(n) * 2.5, rtol=1e-6)

    def test_launch_trig_identity(self):
        n = 32
        phase = wp.zeros(n, dtype=wp.float32)
        out = wp.zeros(n, dtype=wp.float32)
        phase.numpy_view()[:] = np.linspace(0.0, 6.28, n, dtype=np.float32)
        wp.launch(k_trig, dim=n, inputs=[phase, out], device="metal")
        np.testing.assert_allclose(out.numpy_view(), 1.0, atol=1e-5)

    def test_launch_dot(self):
        n = 16
        rng = np.random.default_rng(7)
        a = wp.zeros(n, dtype=wp.vec3f)
        b = wp.zeros(n, dtype=wp.vec3f)
        out = wp.zeros(n, dtype=wp.float32)
        a.numpy_view()[:] = rng.standard_normal((n, 3))
        b.numpy_view()[:] = rng.standard_normal((n, 3))
        wp.launch(k_dot, dim=n, inputs=[a, b, out], device="metal")
        expected = (a.numpy_view() * b.numpy_view()).sum(axis=1)
        np.testing.assert_allclose(out.numpy_view(), expected, atol=1e-5)

    def test_launch_atomic(self):
        n = 64
        accum = wp.zeros(8, dtype=wp.float32)
        idx = wp.zeros(n, dtype=wp.int32)
        vals = wp.zeros(n, dtype=wp.float32)
        rng = np.random.default_rng(3)
        idx.numpy_view()[:] = rng.integers(0, 8, n)
        vals.numpy_view()[:] = rng.standard_normal(n)
        wp.launch(k_atomic_accum, dim=n, inputs=[accum, idx, vals], device="metal")
        expected = np.bincount(idx.numpy_view(), weights=vals.numpy_view(), minlength=8)
        np.testing.assert_allclose(accum.numpy_view(), expected, atol=1e-4)

    def test_synchronize(self):
        wp.synchronize()  # must not raise


if __name__ == "__main__":
    unittest.main()
