# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Strided array views (Newton selection.py unblock, Phase 4 S1).

Upstream wp.array carries BYTE strides on the Python surface
(warp/_src/types.py strides_from_shape) and kernels index through
array_t strides — array.h index() multiplies a char* by the byte stride.
warp-metal's wp_arr_meta already carried that machinery; these tests pin
the host side: stride units, view construction, numpy-view correctness,
wp.array(ptr=..., strides=...) aliasing, tuple-slice views (upstream
__getitem__ semantics), and device-side addressing of a strided array
(dense golden compare).

Newton flows exercised (selection.py:1782, example_ik_cube_stacking:553):
- wp.array(ptr=<interior>, shape=..., strides=...) aliasing over a live
  allocation with non-dense byte strides,
- tuple-slice views  a[:, :k]  returning WarpArray views (not numpy),
- wp.clone of a strided view (dense copy semantics),
- requires_grad flag set (warp_nn Parameter inference path).
"""

import sys
import unittest

import numpy as np

import warp_metal as wp


@wp.kernel
def _sum_rows(a: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i][0] + a[i][1] + a[i][2]


@wp.kernel
def _sum_2d(a: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    wp.atomic_add(out, 0, a[i, j])


class TestStridedViews(unittest.TestCase):
    def test_strides_units_are_bytes(self):
        a = wp.zeros((3, 4), dtype=wp.float32)
        self.assertEqual(a.strides, (16, 4))
        v = wp.zeros(5, dtype=wp.vec3f)
        self.assertEqual(v.strides, (12,))

    def test_slice_view_keeps_sharing(self):
        a = wp.zeros(10, dtype=wp.float32)
        a.numpy_view()[:] = np.arange(10, dtype=np.float32)
        s = a[2:7]
        self.assertEqual(s.shape, (5,))
        self.assertEqual(s.offset_elems, 2)
        np.testing.assert_array_equal(s.numpy_view(), np.arange(2, 7, dtype=np.float32))

    def test_tuple_slice_view_is_array(self):
        a = wp.zeros((4, 6), dtype=wp.float32)
        a.numpy_view()[:] = np.arange(24, dtype=np.float32).reshape(4, 6)
        v = a[:, :3]
        self.assertIsInstance(v, wp.array)
        self.assertEqual(v.shape, (4, 3))
        self.assertNotEqual(v.strides, (12, 4))  # row stride stays 6*4 bytes
        self.assertEqual(v.strides, (24, 4))
        np.testing.assert_array_equal(v.numpy_view(), np.arange(24, dtype=np.float32).reshape(4, 6)[:, :3])
        # int key drops the dim (upstream semantics)
        row = a[2]
        self.assertIsInstance(row, wp.array)
        self.assertEqual(row.shape, (6,))
        np.testing.assert_array_equal(row.numpy_view(), np.arange(24, dtype=np.float32).reshape(4, 6)[2])

    def test_tuple_slice_int_mix(self):
        a = wp.zeros((4, 6), dtype=wp.float32)
        a.numpy_view()[:] = np.arange(24, dtype=np.float32).reshape(4, 6)
        v = a[1:3, 2:5]
        self.assertEqual(v.shape, (2, 3))
        np.testing.assert_array_equal(v.numpy_view(), np.arange(24, dtype=np.float32).reshape(4, 6)[1:3, 2:5])

    def test_wp_array_ptr_interior_strided(self):
        base = wp.zeros((4, 8), dtype=wp.float32)
        base.numpy_view()[:] = np.arange(32, dtype=np.float32).reshape(4, 8)
        # Newton selection.py: interior pointer + byte strides + shape
        v = wp.array(ptr=int(base.ptr) + 4, dtype=wp.float32, shape=(4, 3),
                     strides=(32, 4), device=base.device, copy=False)
        self.assertEqual(v.shape, (4, 3))
        expect = np.arange(32, dtype=np.float32).reshape(4, 8)[:, 1:4]
        np.testing.assert_array_equal(v.numpy_view(), expect)

    def test_wp_array_data_strided(self):
        src = np.arange(24, dtype=np.float32).reshape(4, 6)
        v = wp.array(data=src, shape=(4, 6), strides=(24, 4), device="metal")
        self.assertEqual(v.strides, (24, 4))
        np.testing.assert_array_equal(v.numpy_view(), src)

    def test_clone_of_strided_is_dense(self):
        a = wp.zeros((4, 6), dtype=wp.float32)
        a.numpy_view()[:] = np.arange(24, dtype=np.float32).reshape(4, 6)
        v = a[:, :3]
        c = wp.clone(v)
        self.assertTrue(c.is_contiguous)
        self.assertEqual(c.strides, (12, 4))
        np.testing.assert_array_equal(c.numpy_view(), np.arange(24, dtype=np.float32).reshape(4, 6)[:, :3])

    def test_device_kernel_reads_strided_view(self):
        # the array-addressing proof: a kernel indexing a 1-D view whose row
        # stride skips parent columns must land on the parent's elements
        # through the byte strides (dense read would sum columns 0,4,8,...)
        a = wp.zeros((5, 4), dtype=wp.vec3f)
        a.numpy_view()[:] = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
        v = wp.array(ptr=int(a.ptr), dtype=wp.vec3f, shape=(5,),
                     strides=(4 * 12,), device=a.device, copy=False)
        out = wp.zeros(5, dtype=wp.float32)
        wp.launch(_sum_rows, dim=5, inputs=[v, out], device="metal")
        got = out.numpy_view()
        want = a.numpy_view()[:, 0].sum(axis=1)  # vec3s at parent cols 0,4,8,12,16
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)

    def test_device_kernel_2d_strided_atomic(self):
        a = wp.zeros((4, 6), dtype=wp.float32)
        a.numpy_view()[:] = np.arange(24, dtype=np.float32).reshape(4, 6)
        v = a[:, 1:5]  # (4,4) view, row stride 24 bytes
        out = wp.zeros(1, dtype=wp.float32)
        wp.launch(_sum_2d, dim=(4, 4), inputs=[v, out], device="metal")
        self.assertAlmostEqual(float(out.numpy_view()[0]), float(v.numpy_view().sum()), places=4)

    def test_requires_grad_flag_false_settable(self):
        a = wp.zeros(3, dtype=wp.float32)
        a.requires_grad = False  # warp_nn Parameter inference path
        self.assertFalse(a.requires_grad)
        with self.assertRaises(RuntimeError):
            a.requires_grad = True  # Non-Goal boundary stays fail-loud

    def test_strided_empty_result(self):
        a = wp.zeros((4, 6), dtype=wp.float32)
        v = a[:, 3:3]  # empty selection (Newton early-out shape)
        self.assertEqual(v.shape, (4, 0))
        self.assertEqual(v.numpy_view().size, 0)


if __name__ == "__main__":
    unittest.main()
