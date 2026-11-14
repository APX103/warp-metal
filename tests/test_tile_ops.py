# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Tile op family (KEP-0001 M2.4): threadgroup-backed tile model.

Every shared-storage tile is a full-shape ``threadgroup`` array read and
written with per-thread strided passes; reductions run the guarded log2 tree
over kernel-scope scratch (argmin/argmax carry an int index tree alongside,
ties resolve to the earlier linear index — warp-metal's deterministic
refinement of upstream's unspecified tie order). tile_load zero-fills
coordinates past the source extent, tile_store silently drops them
(bounds_check=True); bounds_check=False drops the checks entirely. Shared
per-block semantics are exclusive to ``wp.launch_tiled``; any plain launch
(with or without block_dim) lowers tiles per-thread scalar — the one-lane
block convention upstream's CPU path forces, which is what the plain-launch
goldens encode. Each Metal-gated test launches on Metal and checks against
an independent numpy computation; the translate-only pins (32 KB threadgroup
budget, barrier-uniformity checkpoint, plain-launch scalar routing) run
everywhere. Skips cleanly when no Metal device is present.
"""

import unittest

import numpy as np

import warp_metal as wp
from warp_metal.runtime import devices as _devices

TOL = 1e-5

_SKIP = not _devices.is_metal_available()
_SKIP_MSG = "no Metal device"

BD = 8  # test block dim (power of two, as launch_tiled requires)


# ---- kernels (1-D/2-D load/store roundtrip through both storage modes) ----


@wp.kernel
def k_roundtrip_1d(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    off = blk * 12
    t = wp.tile_load(a, shape=(16,), offset=(off,), bounds_check=True)
    wp.tile_store(out, t, offset=(off,))


@wp.kernel
def k_roundtrip_1d_shared(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    off = blk * 12
    t = wp.tile_load(a, shape=(16,), offset=(off,), storage="shared", bounds_check=True)
    wp.tile_store(out, t, offset=(off,))


@wp.kernel
def k_roundtrip_1d_nobc(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    off = blk * 16
    t = wp.tile_load(a, shape=(16,), offset=(off,), bounds_check=False)
    wp.tile_store(out, t, offset=(off,), bounds_check=False)


@wp.kernel
def k_roundtrip_2d(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    roff = blk * 2
    t = wp.tile_load(a, shape=(4, 4), offset=(roff, 2), bounds_check=True)
    wp.tile_store(out, t, offset=(roff, 2))


@wp.kernel
def k_roundtrip_2d_shared(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    roff = blk * 2
    t = wp.tile_load(a, shape=(4, 4), offset=(roff, 2), storage="shared", bounds_check=True)
    wp.tile_store(out, t, offset=(roff, 2))


@wp.kernel
def k_map_ops(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    off = blk * 16
    t = wp.tile_load(a, shape=(16,), offset=(off,))
    u = wp.tile_load(b, shape=(16,), offset=(off,))
    m = wp.tile_map(wp.min, t, u)
    m = wp.tile_map(wp.max, m, u)
    m = wp.tile_map(wp.add, m, t)
    m = wp.tile_map(wp.sub, m, u)
    m = wp.tile_map(wp.mul, m, t)
    wp.tile_store(out, m, offset=(off,))


@wp.kernel
def k_reduce_all(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32), idx: wp.array(dtype=wp.int32)):
    blk = wp.tid()
    off = blk * 20
    t = wp.tile_load(a, shape=(20,), offset=(off,), bounds_check=True)
    s = wp.tile_sum(t)[0]
    mn = wp.tile_min(t)[0]
    mx = wp.tile_max(t)[0]
    li = wp.tile_argmin(t)[0]
    if blk == 0:
        out[0] = s
        out[1] = mn
        out[2] = mx
        idx[0] = li


@wp.kernel
def k_reduce_bcast(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    # wp.tile(v) per-thread one-element form (scalar carve-out: the tile owns
    # no shared array): block-partitioned sum via per-block atomics
    blk, lane = wp.tid()
    v = a[blk * BD + lane]
    t = wp.tile(v)
    s = wp.tile_sum(t)[0]
    if lane == 0:
        wp.atomic_add(out, blk, s)


@wp.kernel
def k_reduce_partial(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    # tile smaller than the block: lanes 5..7 own no element and must stage
    # the neutral identity, not zero (a min against 0 would pin the result)
    t = wp.tile_load(a, shape=(5,))
    mn = wp.tile_min(t)[0]
    mx = wp.tile_max(t)[0]
    if wp.tid() == 0:
        out[0] = mn
        out[1] = mx


@wp.kernel
def k_scatter_read_all(out: wp.array(dtype=wp.float32), sums: wp.array(dtype=wp.float32)):
    # single-lane write, whole-block read: the M2.4 scatter_masked contract
    block, lane = wp.tid()
    t = wp.tile_zeros(shape=(BD,), dtype=wp.float32, storage="shared")
    wp.tile_scatter_masked(t, lane, wp.float(lane * lane), lane % 2 == 1)
    v = wp.tile_extract(t, 3)
    s = wp.tile_sum(t)[0]
    flat = block * BD + lane
    out[flat] = v
    if lane == 0:
        sums[block] = s


@wp.kernel
def k_extract_shared(a: wp.array(dtype=wp.float32), picks: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
    # constant and dynamic extracts on a shared tile: every thread reads the
    # same element (count_iso_voxel_children idiom). launch_tiled tid: the
    # 2-D form is (block, lane) — the lane indexes the per-block output rows
    t = wp.tile_load(a, shape=(8,), storage="shared")
    pick = picks[0]
    x3 = wp.tile_extract(t, 3)
    xd = wp.tile_extract(t, pick)
    last = wp.tile_extract(t, wp.block_dim() - 1)
    blk, i = wp.tid()
    out[i * 3] = x3
    out[i * 3 + 1] = xd
    out[i * 3 + 2] = last


@wp.kernel
def k_zeros_shared(out: wp.array(dtype=wp.float32)):
    t = wp.tile_zeros(shape=(BD,), dtype=wp.float32, storage="shared")
    wp.tile_store(out, t, offset=(0,))


@wp.func
def block_stats(a: wp.array(dtype=wp.float32)) -> wp.float32:
    # tile ops inside a @wp.func: the kernel forwards lane/live/shared slots
    t = wp.tile_load(a, shape=(BD,), storage="shared")
    return wp.tile_sum(t)[0]


@wp.kernel
def k_via_func(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    off = blk * BD
    t = wp.tile_load(a, shape=(BD,), offset=(off,))
    s = block_stats(a)
    wp.tile_store(out, t, offset=(off,))
    if blk == 0:
        # the folded sum lands OUTSIDE the tile_store footprint: out[1] is
        # lane 1's strided store slot and no barrier orders a cross-lane
        # write against it, so the sum must not overlap the stored range
        out[2 * BD] = s


@wp.kernel
def k_plain_scalar(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    # plain launch WITHOUT block_dim: per-thread degenerate tile lowering —
    # every tile op collapses to the thread's own element (the Newton
    # non-CUDA block_dim=1 CPU convention), so the output must not depend
    # on the dispatch threadgroup size
    i = wp.tid()
    t = wp.tile_load(a, shape=(1,), offset=(i,))
    two = wp.tile(wp.float(2.0))
    m = wp.tile_map(wp.mul, t, two)
    s = wp.tile_sum(m)[0]
    mn = wp.tile_min(t)[0]
    z = wp.tile_zeros(shape=(1,), dtype=wp.float32, storage="shared")
    wp.tile_scatter_masked(z, 0, s + mn, i % 2 == 0)
    out[i] = wp.tile_extract(z, 0)


def _metal_device():
    return wp.get_device("metal")


def _ref_roundtrip_1d(src, n_blocks):
    # kernel never touches slots past n_blocks*12+4; keep the -99 sentinel there
    out = np.full(n_blocks * 12 + 16, -99.0, dtype=np.float32)
    for blk in range(n_blocks):
        off = blk * 12
        tile = np.zeros(16, dtype=np.float32)
        for t in range(16):
            s = off + t
            if s < src.shape[0]:
                tile[t] = src[s]
        out[off : off + 16] = tile
    return out


def _ref_roundtrip_2d(src, n_blocks):
    out = np.zeros_like(src)
    for blk in range(n_blocks):
        roff = blk * 2
        for i in range(4):
            for j in range(4):
                s0, s1 = roff + i, 2 + j
                if s0 < src.shape[0] and s1 < src.shape[1]:
                    out[s0, s1] = src[s0, s1]
    return out


class TestTileLoadStore(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def _run(self, kern, n_blocks, arrays):
        wp.launch_tiled(kern, dim=n_blocks, inputs=arrays, block_dim=BD, device="metal")
        wp.synchronize()

    def test_roundtrip_1d_default_storage(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(-5, 5, 14).astype(np.float32)  # tail past 14 is OOB
        n_blocks = 2
        a = wp.zeros(14, dtype=wp.float32, device="metal")
        a.numpy_view()[:] = src
        size = n_blocks * 12 + 16
        out = wp.full(size, -99.0, dtype=wp.float32, device="metal")
        self._run(k_roundtrip_1d, n_blocks, [a, out])
        want = _ref_roundtrip_1d(src, n_blocks)
        got = out.numpy_view()[:size]
        np.testing.assert_allclose(got, want, rtol=TOL, atol=1e-6)

    def test_roundtrip_1d_shared_storage(self):
        rng = np.random.default_rng(2)
        src = rng.uniform(-5, 5, 20).astype(np.float32)
        n_blocks = 2
        a = wp.zeros(20, dtype=wp.float32, device="metal")
        a.numpy_view()[:] = src
        size = n_blocks * 12 + 16
        out = wp.full(size, -99.0, dtype=wp.float32, device="metal")
        self._run(k_roundtrip_1d_shared, n_blocks, [a, out])
        want = _ref_roundtrip_1d(src, n_blocks)
        np.testing.assert_allclose(out.numpy_view()[:size], want, rtol=TOL, atol=1e-6)

    def test_roundtrip_1d_bounds_check_false(self):
        # bounds_check=False: every coordinate is in-bounds by construction
        rng = np.random.default_rng(3)
        src = rng.uniform(-5, 5, 32).astype(np.float32)
        n_blocks = 2
        a = wp.zeros(32, dtype=wp.float32, device="metal")
        a.numpy_view()[:] = src
        out = wp.full(32, -99.0, dtype=wp.float32, device="metal")
        self._run(k_roundtrip_1d_nobc, n_blocks, [a, out])
        np.testing.assert_allclose(out.numpy_view(), src, rtol=TOL, atol=1e-6)

    def test_roundtrip_2d(self):
        rng = np.random.default_rng(4)
        src = rng.uniform(-5, 5, (5, 5)).astype(np.float32)
        n_blocks = 2
        a = wp.zeros((5, 5), dtype=wp.float32, device="metal")
        a.numpy_view()[:] = src
        out = wp.zeros((5, 5), dtype=wp.float32, device="metal")
        self._run(k_roundtrip_2d, n_blocks, [a, out])
        np.testing.assert_allclose(out.numpy_view(), _ref_roundtrip_2d(src, n_blocks), rtol=TOL, atol=1e-6)

    def test_roundtrip_2d_shared_storage(self):
        rng = np.random.default_rng(5)
        src = rng.uniform(-5, 5, (5, 5)).astype(np.float32)
        n_blocks = 2
        a = wp.zeros((5, 5), dtype=wp.float32, device="metal")
        a.numpy_view()[:] = src
        out = wp.zeros((5, 5), dtype=wp.float32, device="metal")
        self._run(k_roundtrip_2d_shared, n_blocks, [a, out])
        np.testing.assert_allclose(out.numpy_view(), _ref_roundtrip_2d(src, n_blocks), rtol=TOL, atol=1e-6)


class TestTileMap(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_binary_unary_chain(self):
        rng = np.random.default_rng(6)
        a = rng.uniform(-5, 5, 32).astype(np.float32)
        b = rng.uniform(-5, 5, 32).astype(np.float32)
        n_blocks = 2
        aa = wp.zeros(32, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        bb = wp.zeros(32, dtype=wp.float32, device="metal")
        bb.numpy_view()[:] = b
        out = wp.zeros(32, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_map_ops, dim=n_blocks, inputs=[aa, bb, out], block_dim=BD, device="metal")
        wp.synchronize()
        # m = min(t,u); m = max(m,u); m = m+t; m = m-u; m = m*t — per tile
        # element, exactly the numpy chain (offsets keep the blocks disjoint)
        want = np.empty(32, dtype=np.float32)
        for blk in range(n_blocks):
            off = blk * 16
            ta, tb = a[off : off + 16], b[off : off + 16]
            m = np.minimum(ta, tb)
            m = np.maximum(m, tb)
            m = m + ta
            m = m - tb
            m = m * ta
            want[off : off + 16] = m
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-5)


class TestTileReduce(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_sum_min_max_argmin_shared_tile(self):
        # tile (20,) strided over BD=8 lanes: 4 elements per lane for lanes
        # 0..3, 3 for lanes 4..7 — exercises the strided accumulate
        rng = np.random.default_rng(7)
        a = rng.uniform(-10, 10, 40).astype(np.float32)
        aa = wp.zeros(40, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.zeros(3, dtype=wp.float32, device="metal")
        idx = wp.zeros(1, dtype=wp.int32, device="metal")
        wp.launch_tiled(k_reduce_all, dim=2, inputs=[aa, out, idx], block_dim=BD, device="metal")
        wp.synchronize()
        chunk = a[:20]
        got = out.numpy_view()
        np.testing.assert_allclose(got[0], chunk.sum(dtype=np.float32), rtol=TOL, atol=1e-4)
        np.testing.assert_allclose(got[1], chunk.min(), rtol=TOL, atol=1e-6)
        np.testing.assert_allclose(got[2], chunk.max(), rtol=TOL, atol=1e-6)
        self.assertEqual(int(idx.numpy_view()[0]), int(chunk.argmin()))

    def test_reduce_bcast_scalar_tile(self):
        # wp.tile(v) per-thread one-element form (scalar carve-out): every
        # lane of a block stages its own value into the fold scratch, so the
        # block sum is the sum over the block's a-slice — read back through
        # a per-block atomic on lane 0
        a = np.ones(16, dtype=np.float32)
        aa = wp.zeros(16, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.zeros(2, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_reduce_bcast, dim=2, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        np.testing.assert_allclose(out.numpy_view(), [8.0, 8.0], rtol=TOL)

    def test_partial_tile_min_max_identity(self):
        # 5-element tile, 8 lanes: min/max must only see the 5 loaded
        # elements (identity staging), never the zero-filled tail
        a = np.array([4.0, -7.0, 3.0, 9.0, -2.0], dtype=np.float32)
        aa = wp.zeros(5, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.zeros(2, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_reduce_partial, dim=1, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        got = out.numpy_view()
        np.testing.assert_allclose(got[0], -7.0, rtol=TOL)
        np.testing.assert_allclose(got[1], 9.0, rtol=TOL)

    def test_argmin_tie_takes_earlier_index(self):
        # warp-metal contract: ties resolve to the EARLIER linear index
        # (upstream leaves tie order unspecified); position 1 and 4 tie
        a = np.array([5.0, -3.0, 7.0, 2.0, -3.0, 1.0], dtype=np.float32)
        aa = wp.zeros(8, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = np.concatenate([a, np.zeros(2, dtype=np.float32)])
        out = wp.zeros(3, dtype=wp.float32, device="metal")
        idx = wp.zeros(1, dtype=wp.int32, device="metal")
        wp.launch_tiled(k_reduce_all, dim=1, inputs=[aa, out, idx], block_dim=BD, device="metal")
        wp.synchronize()
        self.assertEqual(int(idx.numpy_view()[0]), 1)


class TestTileScatterExtract(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_scatter_masked_single_lane_visible_to_all(self):
        # odd lanes write lane^2 (1, 9, 25, 49 — slot 3 = 9.0), even lanes
        # are masked off; afterwards EVERY lane reads slot 3 and the
        # whole-block sum — the fixed M2.4 semantics (the M1 per-thread-copy
        # lowering left non-writing lanes stale). out must be uniformly 9.0;
        # sums = sum of the odd squares.
        n_blocks = 2
        out = wp.full(n_blocks * BD, -1.0, dtype=wp.float32, device="metal")
        sums = wp.zeros(n_blocks, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_scatter_read_all, dim=n_blocks, inputs=[out, sums], block_dim=BD, device="metal")
        wp.synchronize()
        np.testing.assert_allclose(out.numpy_view(), np.full(n_blocks * BD, 9.0, dtype=np.float32), rtol=TOL)
        want = sum(float(l * l) for l in range(BD) if l % 2 == 1)
        np.testing.assert_allclose(sums.numpy_view(), [want] * n_blocks, rtol=TOL)

    def test_extract_const_dynamic_and_last(self):
        rng = np.random.default_rng(8)
        a = rng.uniform(-10, 10, 8).astype(np.float32)
        aa = wp.zeros(8, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        picks = wp.zeros(1, dtype=wp.int32, device="metal")
        picks.numpy_view()[:] = 5
        out = wp.full(3 * BD, -1.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_extract_shared, dim=1, inputs=[aa, picks, out], block_dim=BD, device="metal")
        wp.synchronize()
        got = out.numpy_view().reshape(BD, 3)
        np.testing.assert_allclose(got[:, 0], a[3], rtol=TOL)
        np.testing.assert_allclose(got[:, 1], a[5], rtol=TOL)
        np.testing.assert_allclose(got[:, 2], a[BD - 1], rtol=TOL)

    def test_zeros_shared(self):
        out = wp.full(BD, 5.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_zeros_shared, dim=1, inputs=[out], block_dim=BD, device="metal")
        wp.synchronize()
        np.testing.assert_allclose(out.numpy_view(), np.zeros(BD, dtype=np.float32))

    def test_tile_ops_via_wp_func(self):
        # needs_lane forwarding chain (P0 regression): tile ops inside a
        # @wp.func share the kernel's shared arrays through forwarded pointers
        rng = np.random.default_rng(9)
        a = rng.uniform(-5, 5, 2 * BD).astype(np.float32)
        aa = wp.zeros(2 * BD, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.full(2 * BD + 1, -1.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_via_func, dim=2, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        got = out.numpy_view()
        np.testing.assert_allclose(got[: 2 * BD], a, rtol=TOL)
        np.testing.assert_allclose(got[2 * BD], a[:BD].sum(dtype=np.float32), rtol=TOL, atol=1e-5)

    def test_plain_launch_scalar_semantics(self):
        # plain launch WITHOUT block_dim: per-thread degenerate tile lowering
        # (Newton's non-CUDA block_dim=1 convention — the semantics every
        # plain-launch CPU golden was generated with). The kernel output must
        # be independent of the dispatch threadgroup size: each thread loads
        # its own element, doubles it through a scalar tile_map, reduces
        # (identity), and conditionally scatters into a degenerate shared
        # tile — 3*a[i] on even threads, 0 on odd ones.
        rng = np.random.default_rng(10)
        a = rng.uniform(-5, 5, 16).astype(np.float32)
        aa = wp.zeros(16, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.full(16, -1.0, dtype=wp.float32, device="metal")
        wp.launch(k_plain_scalar, dim=16, inputs=[aa, out], device="metal")
        wp.synchronize()
        want = np.where(np.arange(16) % 2 == 0, 3.0 * a, np.float32(0.0)).astype(np.float32)
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL)


# ---- translate-only rejections (run on CPU; no Metal launch) ---------------


class TestTileModelPins(unittest.TestCase):
    """Fail-loud boundaries and storage-model pins of the M2.4 tile closure.

    Translation-only (CPU-safe, runs without a Metal device): the 32 KB
    threadgroup budget, the barrier-uniformity checkpoint, the block_dim
    requirement, and the scalar-carve-out / shared-backed storage split.
    """

    def _translate(self, kern, block_dim=None):
        from warp_metal.msl.translate import translate_kernel

        opts = {"block_dim": block_dim} if block_dim else {}
        return translate_kernel(kern, opts)

    def test_threadgroup_budget_rejection(self):
        # 20000 float slots = 80000 bytes > 32768 (maxThreadgroupMemoryLength):
        # the total is exact at translation time; the check runs in
        # kernel_src, so it fires when the source text is rendered
        @wp.kernel
        def k_budget(a: wp.array(dtype=wp.float32)):
            t = wp.tile_zeros(shape=(20000,), dtype=wp.float32, storage="shared")
            wp.tile_store(a, t)

        u = self._translate(k_budget, BD)
        with self.assertRaises(NotImplementedError) as ctx:
            u.source()
        self.assertIn("threadgroup tile storage budget exceeded", str(ctx.exception))

    def test_barrier_uniformity_rejection(self):
        @wp.kernel
        def k_uniform(a: wp.array(dtype=wp.float32)):
            t = wp.tile_load(a, shape=(BD,), storage="shared")
            if wp.tid() == 0:
                s = wp.tile_sum(t)[0]

        with self.assertRaises(NotImplementedError) as ctx:
            self._translate(k_uniform, BD)
        self.assertIn("non-uniform control flow", str(ctx.exception))

    def test_plain_launch_routes_to_scalar(self):
        # plain launch without block_dim: per-thread degenerate lowering (the
        # Newton non-CUDA block_dim=1 convention, the semantics every
        # plain-launch CPU golden was generated with). No shared arrays, no
        # barriers — the emitted MSL is independent of the dispatch
        # threadgroup layout, so the host's auto threadgroup size is safe.
        @wp.kernel
        def k_nobd(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            t = wp.tile_load(a, shape=(BD,))
            wp.tile_store(out, t)

        u = self._translate(k_nobd)
        src = u.source()
        self.assertEqual(u.shr_arrays, [])
        self.assertNotIn("_wp_tile_shr", src)
        self.assertNotIn("threadgroup_barrier", src)
        self.assertNotIn("_wp_live", src)
        self.assertIn("const int _wp_lane = 0;", src)

    def test_plain_shared_storage_degrades_to_scalar(self):
        # requested shared storage on a plain launch degrades to the same
        # per-thread scalar form (a threadgroup array is meaningless without
        # a defined block layout); the corpus tile kernels pin explicit
        # block_dim when they want per-block semantics
        @wp.kernel
        def k_shared_nobd(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            t = wp.tile_zeros(shape=(BD,), dtype=wp.float32, storage="shared")
            wp.tile_store(out, t)

        u = self._translate(k_shared_nobd)
        src = u.source()
        self.assertEqual(u.shr_arrays, [])
        self.assertNotIn("_wp_tile_shr", src)
        self.assertNotIn("threadgroup_barrier", src)

    def test_plain_block_dim_also_scalar(self):
        # round-5 single rule: ANY plain launch (block_dim supplied or not)
        # lowers tiles per-thread scalar — only launch_tiled gets the
        # threadgroup-backed per-block model. Upstream's CPU path forces
        # block_dim=1 on plain launches (_resolve_launch_block_dim), so the
        # goldens are per-work-item degenerate and a baked block layout
        # could never match them. The host enforces this by dropping
        # block_dim for plain launches before translation (only launch_tiled
        # reaches the translator with one) — pinned here at the routing
        # site, since invoking _unit_for directly would compile a Metal
        # library.
        import inspect

        from warp_metal.runtime import host as _host

        launch_src = inspect.getsource(_host._launch_ssa)
        self.assertIn("_unit_for(kernel, block_dim if tiled else None)", launch_src)
        # and the plain dispatch hands the native auto threadgroup size,
        # since plain MSL carries no block layout at all
        self.assertIn("_mm.dispatch(pso, handles, threads, 0)", launch_src)

    def test_plain_cooperative_ops_degenerate(self):
        # cooperative ops on a plain launch collapse to per-thread identity:
        # sum/min/max return the thread's own element, argmin/argmax resolve
        # to index 0, and a masked scatter writes element 0 under its mask —
        # no scratch, no tree, no barrier anywhere
        @wp.kernel
        def k_plain_reduce(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            t = wp.tile_load(a, shape=(1,), offset=(i,))
            s = wp.tile_sum(t)[0]
            mn = wp.tile_min(t)[0]
            z = wp.tile_zeros(shape=(1,), dtype=wp.float32, storage="shared")
            wp.tile_scatter_masked(z, 0, s + mn, i % 2 == 0)
            out[i] = wp.tile_extract(z, 0)

        u = self._translate(k_plain_reduce)
        src = u.source()
        self.assertEqual(u.shr_arrays, [])
        self.assertNotIn("threadgroup_barrier", src)
        self.assertNotIn("_wp_tile_shr", src)
        # argmin degenerate form resolves to the single index
        self.assertIn("= 0;", src)

    def test_scatter_forces_shared_storage(self):
        # upstream semantics: tile_scatter_masked forces the tile's storage to
        # "shared" at parse time, so a register-storage tile_zeros silently
        # upgrades to a threadgroup-backed tile instead of erroring
        @wp.kernel
        def k_scatter_reg(a: wp.array(dtype=wp.float32)):
            t = wp.tile_zeros(shape=(BD,), dtype=wp.float32)
            i, lane = wp.tid()
            wp.tile_scatter_masked(t, lane, 1.0, True)

        u = self._translate(k_scatter_reg, BD)
        self.assertEqual(u.shr_arrays, [("float", BD)])

    def test_register_tile_shape_bd_keeps_scalar_path(self):
        # a register-storage tile whose shape is the per-thread one-element
        # form (dims == block_dim) stays scalar: no threadgroup array is
        # claimed, yet the store lowers through the lane index — the preamble
        # defines it whenever any tile op exists
        @wp.kernel
        def k_scalar_copy(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            t = wp.tile_load(a, shape=(BD,), offset=(i,))
            wp.tile_store(out, t, offset=(i,))

        u = self._translate(k_scalar_copy, BD)
        src = u.source()
        self.assertEqual(u.shr_arrays, [])
        self.assertNotIn("_wp_tile_shr", src)
        self.assertIn(f"const int _wp_lane = _idx % {BD};", src)

    def test_tile_bcast_is_shared_backed(self):
        # wp.tile(v) materializes a shared-storage tile in vendored warp 1.19
        # (tile_shared_t with a strided layout), so under the M2.4 model the
        # broadcast fills a threadgroup array and the const-0 extract reads
        # slot 0 — per-thread values stay mutually visible block-wide
        @wp.kernel
        def k_bcast(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            v = a[i]
            t = wp.tile(v)
            out[i] = wp.tile_extract(t, 0)

        u = self._translate(k_bcast, BD)
        src = u.source()
        self.assertEqual(u.shr_arrays, [("float", BD)])
        self.assertIn(f"threadgroup float _wp_tile_shr_0[{BD}];", src)
        self.assertIn("threadgroup_barrier(mem_flags::mem_threadgroup);", src)


if __name__ == "__main__":
    unittest.main()
