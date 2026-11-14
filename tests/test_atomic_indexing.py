# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""3-D indexed atomics (KEP-0001 M2.4 steel round): ``atomic_add/min/max`` on
``array3d`` lowers to the 6-arg ``wp_atomic_*_ret_at`` prelude forms with
offset ``i*s0 + j*s1 + k*s2`` (bytes). steel ``crba_fill_par_dof`` scatters
mass into ``H_group[g, r, c]`` exactly this way.

Expectations are kept deterministic without relying on atomic order:
unique-cell adds (each cell touched at most once), same-cell adds of exact
fp32 integers (commutative), and min/max on distinct cells.
"""

import unittest

import numpy as np

import warp_metal as wp
from warp_metal.runtime import devices as _devices

_SKIP = not _devices.is_metal_available()
_SKIP_MSG = "no Metal device"


class TestAtomic3d(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_add_unique_cells(self):
        # every thread writes a distinct cell: result is exact regardless of
        # scheduling (catches wrong stride math that lands on other cells)
        @wp.kernel
        def k(out: wp.array3d(dtype=wp.float32)):
            t = wp.tid()
            i = t // 12
            j = (t // 3) % 4
            c = t % 3
            wp.atomic_add(out, i, j, c, wp.float(t) + 0.5)

        g = (2, 4, 3)  # 24 cells, 24 threads
        out = wp.zeros(g, dtype=wp.float32, device="metal")
        wp.launch(k, dim=24, inputs=[out], device="metal")
        wp.synchronize()
        expect = np.arange(24, dtype=np.float32) + np.float32(0.5)
        np.testing.assert_array_equal(out.numpy_view().reshape(g), expect.reshape(g))

    def test_add_same_cell_exact(self):
        # all threads hit one cell with exact integers: commutative in fp32
        @wp.kernel
        def k(out: wp.array3d(dtype=wp.float32)):
            t = wp.tid()
            wp.atomic_add(out, t % 3, 1, 2, 1.0)

        out = wp.zeros((3, 4, 3), dtype=wp.float32, device="metal")
        wp.launch(k, dim=33, inputs=[out], device="metal")
        wp.synchronize()
        view = out.numpy_view().reshape(3, 4, 3)
        np.testing.assert_array_equal(view[:, 1, 2], np.full(3, 11.0, dtype=np.float32))
        # untouched cells stay zero (catches offset overshoot into neighbours)
        self.assertEqual(float(view.sum()), 33.0)

    def test_min_max_int(self):
        @wp.kernel
        def k(lo: wp.array3d(dtype=wp.int32), hi: wp.array3d(dtype=wp.int32)):
            t = wp.tid()
            wp.atomic_min(lo, t // 4, 1, t % 4, t)
            wp.atomic_max(hi, t // 4, 1, t % 4, t)

        # all middle indices valid: shape (2,3,4), writes at j=1
        lo = wp.full((2, 3, 4), 10**6, dtype=wp.int32, device="metal")
        hi = wp.full((2, 3, 4), -(10**6), dtype=wp.int32, device="metal")
        wp.launch(k, dim=8, inputs=[lo, hi], device="metal")
        wp.synchronize()
        expect_lo = np.full((2, 3, 4), 10**6, dtype=np.int32)
        expect_hi = np.full((2, 3, 4), -(10**6), dtype=np.int32)
        for t in range(8):
            expect_lo[t // 4, 1, t % 4] = t
            expect_hi[t // 4, 1, t % 4] = t
        np.testing.assert_array_equal(lo.numpy_view(), expect_lo)
        np.testing.assert_array_equal(hi.numpy_view(), expect_hi)

    def test_min_max_float(self):
        @wp.kernel
        def k(lo: wp.array3d(dtype=wp.float32), hi: wp.array3d(dtype=wp.float32)):
            t = wp.tid()
            wp.atomic_min(lo, t % 2, 1, t // 2, wp.float(t) * 1.5)
            wp.atomic_max(hi, t % 2, 1, t // 2, wp.float(t) * 1.5)

        lo = wp.full((2, 2, 3), 1.0e30, dtype=wp.float32, device="metal")
        hi = wp.full((2, 2, 3), -1.0e30, dtype=wp.float32, device="metal")
        wp.launch(k, dim=6, inputs=[lo, hi], device="metal")
        wp.synchronize()
        expect_lo = np.full((2, 2, 3), 1.0e30, dtype=np.float32)
        expect_hi = np.full((2, 2, 3), -1.0e30, dtype=np.float32)
        for t in range(6):
            expect_lo[t % 2, 1, t // 2] = t * 1.5
            expect_hi[t % 2, 1, t // 2] = t * 1.5
        np.testing.assert_array_equal(lo.numpy_view(), expect_lo)
        np.testing.assert_array_equal(hi.numpy_view(), expect_hi)


if __name__ == "__main__":
    unittest.main()
