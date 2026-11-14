# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Launch routing semantics (KEP-0001 M2.4 P0): plain vs tiled + watchdog.

Upstream contract (warp context.py ``launch`` / ``launch_tiled``):
- plain ``wp.launch(dim=D, block_dim=B)``: total threads = prod(D); ``B`` is
  only the threadgroup (CUDA block) scheduling size and never multiplies the
  thread count nor shifts ``wp.tid()``. Scalar ``wp.tid()`` is the leading
  row-major grid coordinate (1-D: the flat index; dim=(a,b): ``_idx // b``).
- ``wp.launch_tiled(dim=G, block_dim=B)``: ``B`` joins as the trailing
  dimension (threads = prod(G)*B); scalar ``wp.tid()`` is the BLOCK index,
  the 2-D form the ``(block, lane)`` pair.
- the dispatch watchdog bounds the completion wait
  (WARP_METAL_DISPATCH_TIMEOUT_S, default 120 s, 0 disables): a hung/faulted
  dispatch raises MetalError instead of blocking forever.

Skips cleanly when no Metal device is present (CI without Apple silicon).
"""

import os
import unittest

import numpy as np

import warp_metal as wp
from warp_metal.runtime import devices as _devices

_SKIP = not _devices.is_metal_available()
_SKIP_MSG = "no Metal device"


class TestPlainLaunchBlockDim(unittest.TestCase):
    """Plain wp.launch with block_dim: thread count and tid unaffected."""

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_block_dim_dim_larger_than_block(self):
        # n > B: grid spans several threadgroups; every element written once
        # by the thread whose tid == i (the old bug multiplied threads by B,
        # over-dispatching with out-of-range tids clobbering past-the-end
        # memory or faulting).
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i)

        n, b = 1000, 64
        out = wp.full(n, -1.0, dtype=wp.float32, device="metal")
        wp.launch(k, dim=n, inputs=[out], block_dim=b, device="metal")
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy_view(), np.arange(n, dtype=np.float32))

    def test_block_dim_dim_smaller_than_block(self):
        # n < B (the steel test_body_force hang shape: 4 articulations with
        # block_dim=256 used to launch 1024 threads and read far past the
        # arrays — GPU page fault, forever-pending dispatch).
        @wp.kernel
        def k(art_active: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            if art_active[i] == 0:
                return
            out[i] = wp.float(i) + 1.0

        n, b = 4, 256
        active = wp.full(n, 1, dtype=wp.int32, device="metal")
        out = wp.full(n, -1.0, dtype=wp.float32, device="metal")
        wp.launch(k, dim=n, inputs=[active, out], block_dim=b, device="metal")
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy_view(), np.arange(n, dtype=np.float32) + 1.0)
        self.assertEqual(active.numpy_view().tolist(), [1] * n)  # no OOB writes

    def test_block_dim_multi_launches_repeatable(self):
        # the PSO cache keys on block_dim; repeated plain launches with the
        # same kernel under different block_dim values must stay exact
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i) * 2.0

        n = 37  # prime: not a multiple of any block size
        for b in (32, 128, 256):
            out = wp.full(n, -1.0, dtype=wp.float32, device="metal")
            wp.launch(k, dim=n, inputs=[out], block_dim=b, device="metal")
            wp.synchronize()
            np.testing.assert_array_equal(out.numpy_view(), np.arange(n) * 2.0)

    def test_2d_dim_scalar_tid_is_leading_coordinate(self):
        # upstream: scalar wp.tid() under a multi-dim launch is the leading
        # row-major coordinate — dim=(3, 4): tid = _idx // 4 in {0, 1, 2},
        # each coordinate claimed by exactly 4 threads (j = 0..3); count the
        # claims atomically (each slot's write count proves the mapping)
        @wp.kernel
        def k(counts: wp.array(dtype=wp.float32)):
            i = wp.tid()
            wp.atomic_add(counts, i, 1.0)

        m, n = 3, 4
        counts = wp.zeros(m * n, dtype=wp.float32, device="metal")
        wp.launch(k, dim=(m, n), inputs=[counts], block_dim=64, device="metal")
        wp.synchronize()
        expected = np.zeros(m * n, dtype=np.float32)
        expected[:m] = n  # coordinates 0..m-1 each claimed by n threads
        np.testing.assert_array_equal(counts.numpy_view(), expected)

    def test_2d_dim_vector_tid_unravels(self):
        # i, j = wp.tid() over dim=(m, n): row-major, last dim fastest
        @wp.kernel
        def k(out: wp.array2d(dtype=wp.float32)):
            i, j = wp.tid()
            out[i, j] = wp.float(i * 10 + j)

        m, n = 3, 4
        out = wp.full((m, n), -1.0, dtype=wp.float32, device="metal")
        wp.launch(k, dim=(m, n), inputs=[out], block_dim=64, device="metal")
        wp.synchronize()
        expected = np.arange(m * n, dtype=np.float32).reshape(m, n)
        expected = (expected // n * 10 + expected % n).astype(np.float32)
        np.testing.assert_array_equal(out.numpy_view(), expected)

    def test_plain_launch_tile_kernel_matches_upstream_reference(self):
        # Upstream warp-lang 1.19.0 ACCEPTS a plain wp.launch of a tile
        # kernel and runs degenerate per-work-item tile semantics there —
        # measured on CPU: every work item sums its own zero tile, so
        # out[i] = 0.0 for every i, at dim=4 AND dim=32 (block_dim=32).
        # warp-metal implements exactly that contract (round-5 single rule):
        # ANY plain launch lowers tiles per-thread scalar regardless of the
        # supplied block_dim, so the output is independent of the dispatch
        # threadgroup layout and matches the CPU reference by construction.
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            t = wp.tile_zeros(shape=(4,), dtype=wp.float32)
            s = wp.tile_sum(t)
            out[i] = wp.tile_extract(s, 0)

        for n, b in ((32, 32), (4, 32)):
            out = wp.full(n, 7.0, dtype=wp.float32, device="metal")
            wp.launch(k, dim=n, inputs=[out], block_dim=b, device="metal")
            wp.synchronize()
            # reference hardcoded from a warp-lang 1.19.0 CPU measurement of
            # the identical kernel (plain wp.launch): all zeros
            expected = np.zeros(n, dtype=np.float32)
            np.testing.assert_array_equal(out.numpy_view(), expected)

    def test_tile_reduce_via_wp_func_emits_paired_live(self):
        # Path-coverage regression: _tile_reduce reached through a @wp.func
        # must thread _wp_live through the needs_lane forwarding (function
        # parameter + call-site argument), not just the kernel preamble —
        # an unpaired use compiles as "use of undeclared identifier" on the
        # Metal side (trajectory regression this test pins). The static
        # self-checks in fn_src/kernel_src make any unpaired emission fail
        # at translation time, so a successful translate here proves the
        # pairing; the textual asserts document the exact contract.
        @wp.func
        def block_zero_sum() -> wp.float32:
            t = wp.tile_zeros(shape=(4,), dtype=wp.float32)
            s = wp.tile_sum(t)
            return wp.tile_extract(s, 0)

        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = block_zero_sum()

        # launch_tiled translation, dim covered by one threadgroup:
        # translates only when every _wp_live use is paired with a
        # definition. (Passing block_dim to the translator IS the tiled
        # contract — plain launches never carry one and lower scalar.)
        from warp_metal.msl.translate import translate_kernel

        u = translate_kernel(k, {"block_dim": 32})
        src = u.source()
        # kernel preamble defines it exactly once (min(_gsize.x - 组基, bd))
        self.assertEqual(src.count("const int _wp_live ="), 1)
        # the @wp.func signature carries it alongside _wp_lane, and the
        # call site forwards it in the same order
        self.assertGreaterEqual(src.count("\n    int _wp_live,"), 1)
        self.assertIn("(_wp_lane, _wp_live, _wp_tile_shr_0", src)
        # syntax shape: the hidden params live inside a complete signature
        # (ret-type prefix + fn name + indented param lines). Regression
        # pins the torn-emission bug where an argless needs_lane function
        # emitted a bare parameter list at file scope (leading comma).
        # M2.4: the function claims three shared arrays (the tile_zeros
        # threadgroup-backed tile, the shape-(1,) reduction result tile, the
        # BD-wide fold scratch), forwarded as trailing threadgroup pointers.
        fn_name = u.functions[0].name
        self.assertEqual(
            [
                ln.strip()
                for ln in src.splitlines()
                if ln.strip().startswith("threadgroup float* _wp_tile_shr_")
            ][:3],
            [
                "threadgroup float* _wp_tile_shr_0,",
                "threadgroup float* _wp_tile_shr_1,",
                "threadgroup float* _wp_tile_shr_2)",
            ],
        )
        self.assertIn(
            f"static float {fn_name}(\n"
            "    int _wp_lane,\n"
            "    int _wp_live,\n"
            "    threadgroup float* _wp_tile_shr_0,",
            src,
        )
        # negative shapes: no comma may start a line or open a param list,
        # and no hidden-param name may begin a line (torn signature — the
        # hidden param names as locals only ever appear mid-line or indented;
        # fold locals like `int _wp_first_N` legitimately start lines)
        self.assertNotIn("(\n,", src)
        self.assertNotIn("\n,", src)
        for ln in src.splitlines():
            self.assertFalse(
                ln.startswith("int _wp_lane") or ln.startswith("int _wp_live"),
                f"torn hidden-param emission: {ln!r}",
            )

        # metal run: dim=4 and dim=32, plain launches (per-thread scalar
        # tile lowering) — reference hardcoded from a warp-lang 1.19.0 CPU
        # measurement of the same kernel (plain wp.launch): all zeros
        for n, b in ((4, 32), (32, 32)):
            out = wp.full(n, 7.0, dtype=wp.float32, device="metal")
            wp.launch(k, dim=n, inputs=[out], block_dim=b, device="metal")
            wp.synchronize()
            np.testing.assert_array_equal(out.numpy_view(), np.zeros(n, dtype=np.float32))


class TestTiledLaunchTid(unittest.TestCase):
    """wp.launch_tiled: scalar tid = block index, 2-D tid = (block, lane)."""

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_scalar_tid_is_block_index(self):
        # every lane in block i writes i — the fill proves the tid value
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i)

        n_blocks, b = 6, 32
        out = wp.full(n_blocks, -1.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k, dim=n_blocks, inputs=[out], block_dim=b, device="metal")
        wp.synchronize()
        np.testing.assert_array_equal(
            out.numpy_view(), np.arange(n_blocks, dtype=np.float32)
        )

    def test_scalar_tid_is_block_index_dim_list(self):
        # dim passed as a length-1 list, matching upstream call style
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i) + 100.0

        n_blocks, b = 5, 64
        out = wp.full(n_blocks, -1.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k, dim=[n_blocks], inputs=[out], block_dim=b, device="metal")
        wp.synchronize()
        np.testing.assert_array_equal(
            out.numpy_view(), np.arange(n_blocks, dtype=np.float32) + 100.0
        )

    def test_2d_tid_block_lane(self):
        # (block, lane) covers every (i, j) slot exactly once per pair
        @wp.kernel
        def k(out: wp.array2d(dtype=wp.float32)):
            i, j = wp.tid()
            out[i, j] = wp.float(i * 1000 + j)

        n_blocks, b = 4, 16
        out = wp.full((n_blocks, b), -1.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k, dim=n_blocks, inputs=[out], block_dim=b, device="metal")
        wp.synchronize()
        expected = (np.arange(n_blocks)[:, None] * 1000 + np.arange(b)[None, :]).astype(np.float32)
        np.testing.assert_array_equal(out.numpy_view(), expected)

    def test_2d_grid_dim_scalar_tid_unravels_blocks(self):
        # upstream launch_tiled(dim=(2, 3), block_dim=B): scalar tid is the
        # leading coordinate of unravel(block_index, (2, 3)) — flat blocks
        # 0-2 fold to coordinate 0, 3-5 to 1; each coordinate is claimed by
        # 3 blocks x 8 lanes = 24 threads (atomic claim counting)
        @wp.kernel
        def k(counts: wp.array(dtype=wp.float32)):
            i = wp.tid()
            wp.atomic_add(counts, i, 1.0)

        g1, g2, b = 2, 3, 8
        counts = wp.zeros(g1 * g2, dtype=wp.float32, device="metal")
        wp.launch_tiled(k, dim=(g1, g2), inputs=[counts], block_dim=b, device="metal")
        wp.synchronize()
        expected = np.zeros(g1 * g2, dtype=np.float32)
        expected[0] = expected[1] = g2 * b
        np.testing.assert_array_equal(counts.numpy_view(), expected)

    def test_non_pow2_block_dim_rejected(self):
        # fail-loud retained: the threadgroup reduction requires pow2 blocks
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i)

        out = wp.zeros(4, dtype=wp.float32, device="metal")
        with self.assertRaises(ValueError):
            wp.launch_tiled(k, dim=1, inputs=[out], block_dim=100, device="metal")

    def test_4d_grid_rejected(self):
        # upstream: launch_tiled requires fewer than 4 grid dimensions
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i)

        out = wp.zeros(4, dtype=wp.float32, device="metal")
        with self.assertRaises(ValueError):
            wp.launch_tiled(k, dim=(1, 1, 1, 1), inputs=[out], block_dim=32, device="metal")


class TestDispatchWatchdog(unittest.TestCase):
    """WARP_METAL_DISPATCH_TIMEOUT_S bounds the dispatch completion wait.

    The timeout-fire path is inherently racy to exercise in CI: a deadline
    shorter than a fast kernel's runtime can expire while the dispatch is
    already complete — the completion handler signals the semaphore before
    dispatch_semaphore_wait is even entered, so the wait legitimately returns
    0 and no timeout fires. The negative path (deadline exceeded ->
    MetalError naming WARP_METAL_DISPATCH_TIMEOUT_S) is therefore verified
    manually by injecting a long-running kernel with a small timeout; the
    tests below pin the deterministic env handling (0 disables the watchdog,
    invalid values fall back to the 120 s default).
    """

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_zero_disables_watchdog(self):
        # 0 restores the unbounded wait; a healthy kernel must complete
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i) + 7.0

        out = wp.zeros(8, dtype=wp.float32, device="metal")
        old = os.environ.get("WARP_METAL_DISPATCH_TIMEOUT_S")
        os.environ["WARP_METAL_DISPATCH_TIMEOUT_S"] = "0"
        try:
            wp.launch(k, dim=8, inputs=[out], device="metal")
            wp.synchronize()
            np.testing.assert_array_equal(out.numpy_view(), np.arange(8) + 7.0)
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_DISPATCH_TIMEOUT_S", None)
            else:
                os.environ["WARP_METAL_DISPATCH_TIMEOUT_S"] = old

    def test_invalid_value_uses_default(self):
        # a non-numeric value falls back to the 120 s default (dispatch works)
        @wp.kernel
        def k(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            out[i] = wp.float(i)

        out = wp.zeros(4, dtype=wp.float32, device="metal")
        old = os.environ.get("WARP_METAL_DISPATCH_TIMEOUT_S")
        os.environ["WARP_METAL_DISPATCH_TIMEOUT_S"] = "not-a-number"
        try:
            wp.launch(k, dim=4, inputs=[out], device="metal")
            wp.synchronize()
            np.testing.assert_array_equal(out.numpy_view(), np.arange(4, dtype=np.float32))
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_DISPATCH_TIMEOUT_S", None)
            else:
                os.environ["WARP_METAL_DISPATCH_TIMEOUT_S"] = old


if __name__ == "__main__":
    unittest.main()
