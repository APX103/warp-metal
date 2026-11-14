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

import re
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

    def test_uniformity_compile_time_condition_exempt(self):
        # steel hinv_jt's `if COMPUTE_DIAG != 0:` — wp.constant folds to a
        # compile-time test (`const var = 1;` then `test = (var != 0)` in the
        # SSA), so every lane of the block reaches the same branch decision
        # and the cooperative op's barriers cannot diverge. The branch stays
        # in the emission (the fold exempts the checkpoint, it never rewrites
        # the SSA); runtime-variable conditions keep the fail-loud rejection.
        COMPUTE_DIAG = wp.constant(1)
        WRITE_GROUP = wp.constant(0)

        @wp.kernel
        def k_const_if(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            t = wp.tile_load(a, shape=(BD,), storage="shared")
            if COMPUTE_DIAG != 0:
                wp.tile_store(out, wp.tile_sum(t))
            if WRITE_GROUP != 0:
                wp.tile_store(out, t)

        u = self._translate(k_const_if, BD)
        src = u.source()
        # both branches survive emission, still guarded by their test vars
        self.assertIn("if (", src)
        self.assertIn("threadgroup_barrier", src)

        @wp.kernel
        def k_runtime_if(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            t = wp.tile_load(a, shape=(BD,), storage="shared")
            if wp.tid() == 0:
                wp.tile_store(out, wp.tile_sum(t))

        with self.assertRaises(NotImplementedError) as ctx:
            self._translate(k_runtime_if, BD)
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

    def test_row_view_advances_by_row_stride(self):
        # ``arr2d[i]`` row slices (the steel armature / Newton corpus pattern)
        # must advance the view's data pointer by i * strides[0] BYTES with a
        # Python-style negative wrap, mirroring upstream array.h view(). A
        # shape-product advance (idx * shape[1] * shape[2]) collapses to 0 on
        # 2-D parents (shape[2] == 0) and pins every slice to row 0 — the
        # corpus cholesky divergence. Pinned at the source level: both the
        # load view and the store view must carry the stride advance.
        @wp.kernel
        def k_row_view(x: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
            i = wp.tid()
            row = x[i]
            t = wp.tile_load(row, shape=(BD,), bounds_check=False)
            wp.tile_store(out[i], t)

        u = self._translate(k_row_view, BD)
        src = u.source()
        advances = re.findall(
            r"(\w+)_data = \(device \w+\*\)\(\(device char\*\)\(\w+_data\) \+ \(size_t\)\((.*?)\)\);", src
        )
        self.assertEqual(len(advances), 2)  # x view + out view
        for _var, expr in advances:
            # byte advance i * strides[0] (never a shape product, which folds
            # to zero on 2-D parents), negative indices wrapped
            self.assertIn(".strides[0]", expr)
            self.assertIn("< 0 ?", expr)
            self.assertNotIn(".shape[1]", expr)
        # the meta shift: view strides copy the parent's trailing strides
        self.assertIn(".strides[1]", src)
        self.assertIn(".ndim - 1", src)


# ---- KEP-0001 M2.4 P3+P4 kernels: matmul, alias views, diag_add, assign,
# axis reductions, scan, and the cholesky/solve family -----------------------


@wp.kernel
def k_mm(a: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    ta = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    tb = wp.tile_load(b, shape=(6, 4), offset=(0, blk * 4), storage="shared")
    c = wp.tile_zeros((4, 4), dtype=wp.float32, storage="shared")
    wp.tile_matmul(ta, tb, c)
    wp.tile_store(out, c, offset=(blk * 4, blk * 4))


@wp.kernel
def k_mm_acc(a: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), io: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    ta = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    tb = wp.tile_load(b, shape=(6, 4), offset=(0, blk * 4), storage="shared")
    tc = wp.tile_load(io, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    wp.tile_matmul(ta, tb, tc, alpha=2.0, beta=3.0)
    wp.tile_store(io, tc, offset=(blk * 4, 0))


@wp.kernel
def k_transpose(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    tt = wp.tile_transpose(t)
    wp.tile_store(out, tt, offset=(0, blk * 4))


@wp.kernel
def k_reshape(a: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    r = wp.tile_reshape(t, shape=(24,))
    wp.tile_store(out, r, offset=(blk * 24,))


@wp.kernel
def k_reshape_write_through(a: wp.array2d(dtype=wp.float32), s: wp.array(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    # the reshape alias owns no storage: assigning through it writes the
    # source tile's array (flat offset 7..10 -> t[1, 1..4]), visible in t
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    sv = wp.tile_load(s, shape=(4,), offset=(blk * 4,), storage="shared")
    r = wp.tile_reshape(t, shape=(24,))
    wp.tile_assign(r, sv, offset=(7,))
    wp.tile_store(out, t, offset=(blk * 4, 0))


@wp.kernel
def k_view_write_through(a: wp.array2d(dtype=wp.float32), s: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    # a (2,2) view at offset (1,1) aliases the source tile: the assign must
    # land in the SOURCE tile's storage at (1,1)..(2,2) through the view's
    # base+strides mapping
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    sv = wp.tile_load(s, shape=(2, 2), offset=(blk * 2, 0), storage="shared")
    v = wp.tile_view(t, offset=(1, 1), shape=(2, 2))
    wp.tile_assign(v, sv, offset=(0, 0))
    wp.tile_store(out, t, offset=(blk * 4, 0))


@wp.kernel
def k_assign_overlap(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    # src view aliases the dst columns the write shifts ONTO (read cols 0..3,
    # write cols 2..5): the snapshot staging makes the shift copy ORIGINAL
    # data (numpy assignment semantics), not half-updated columns
    blk = wp.tid()
    t = wp.tile_load(a, shape=(8, 8), offset=(blk * 8, 0), storage="shared")
    s = wp.tile_view(t, offset=(0, 0), shape=(8, 4))
    wp.tile_assign(t, s, offset=(0, 2))
    wp.tile_store(out, t, offset=(blk * 8, 0))


@wp.kernel
def k_diag_add(a: wp.array2d(dtype=wp.float32), d: wp.array(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    dv = wp.tile_load(d, shape=(4,), offset=(blk * 4,), storage="shared")
    c = wp.tile_diag_add(t, dv)
    wp.tile_store(out, c, offset=(blk * 4, 0))


@wp.kernel
def k_axis_sum(a: wp.array2d(dtype=wp.float32), out0: wp.array(dtype=wp.float32), out1: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 6), offset=(blk * 4, 0), storage="shared")
    s0 = wp.tile_sum(t, axis=0)
    s1 = wp.tile_sum(t, axis=1)
    wp.tile_store(out0, s0, offset=(blk * 6,))
    wp.tile_store(out1, s1, offset=(blk * 4,))


@wp.kernel
def k_scan(a: wp.array(dtype=wp.float32), outi: wp.array(dtype=wp.float32), oute: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(8,), offset=(blk * 8,), storage="shared")
    ri = wp.tile_scan_inclusive(t)
    re = wp.tile_scan_exclusive(t)
    wp.tile_store(outi, ri, offset=(blk * 8,))
    wp.tile_store(oute, re, offset=(blk * 8,))


@wp.kernel
def k_plain_scan(a: wp.array(dtype=wp.float32), outi: wp.array(dtype=wp.float32), oute: wp.array(dtype=wp.float32)):
    # plain launch: the per-thread degenerate tile is the one loaded element,
    # so the inclusive prefix IS the element and the exclusive prefix is the
    # empty sum
    i = wp.tid()
    t = wp.tile_load(a, shape=(1,), offset=(i,))
    ri = wp.tile_scan_inclusive(t)
    re = wp.tile_scan_exclusive(t)
    wp.tile_store(outi, ri, offset=(i,))
    wp.tile_store(oute, re, offset=(i,))


@wp.kernel
def k_chol(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    l = wp.tile_cholesky(t)
    wp.tile_store(out, l, offset=(blk * 4, 0))


@wp.kernel
def k_chol_inplace(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    t = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    wp.tile_cholesky_inplace(t)
    wp.tile_store(out, t, offset=(blk * 4, 0))


@wp.kernel
def k_lower_solve(a: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    v = wp.tile_load(y, shape=(4,), offset=(blk * 4,), storage="shared")
    z = wp.tile_lower_solve(l, v)
    wp.tile_store(out, z, offset=(blk * 4,))


@wp.kernel
def k_lower_solve_m(a: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    v = wp.tile_load(y, shape=(4, 2), offset=(blk * 4, 0), storage="shared")
    z = wp.tile_lower_solve(l, v)
    wp.tile_store(out, z, offset=(blk * 4, 0))


@wp.kernel
def k_upper_solve(a: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    # upstream idiom: U = transpose(L); upper_solve reads the transposed
    # triangle through the alias's strided mapping
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    lt = wp.tile_transpose(l)
    v = wp.tile_load(y, shape=(4, 2), offset=(blk * 4, 0), storage="shared")
    x = wp.tile_upper_solve(lt, v)
    wp.tile_store(out, x, offset=(blk * 4, 0))


@wp.kernel
def k_chol_solve(a: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    v = wp.tile_load(y, shape=(4,), offset=(blk * 4,), storage="shared")
    x = wp.tile_cholesky_solve(l, v, fill_mode="lower")
    wp.tile_store(out, x, offset=(blk * 4,))


@wp.kernel
def k_chol_solve_upper(a: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    # upper fill reads the factor from the input's upper triangle — the
    # caller passes chol(H, "upper") (= chol(H, "lower") transposed)
    blk = wp.tid()
    h = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    v = wp.tile_load(y, shape=(4,), offset=(blk * 4,), storage="shared")
    x = wp.tile_cholesky_solve(h, v, fill_mode="upper")
    wp.tile_store(out, x, offset=(blk * 4,))


@wp.kernel
def k_chol_solve_inplace(a: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    v = wp.tile_load(y, shape=(4,), offset=(blk * 4,), storage="shared")
    wp.tile_cholesky_solve_inplace(l, v, fill_mode="lower")
    wp.tile_store(out, v, offset=(blk * 4,))


@wp.kernel
def k_lower_solve_inplace(a: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    v = wp.tile_load(y, shape=(4,), offset=(blk * 4,), storage="shared")
    wp.tile_lower_solve_inplace(l, v)
    wp.tile_store(out, v, offset=(blk * 4,))


@wp.kernel
def k_upper_solve_inplace(a: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    blk = wp.tid()
    l = wp.tile_load(a, shape=(4, 4), offset=(blk * 4, 0), storage="shared")
    lt = wp.tile_transpose(l)
    v = wp.tile_load(y, shape=(4,), offset=(blk * 4,), storage="shared")
    wp.tile_upper_solve_inplace(lt, v)
    wp.tile_store(out, v, offset=(blk * 4,))


# ---- P3+P4 numpy references -----------------------------------------------


def _spd(rng, n, scale=3.0):
    a = rng.standard_normal((n, n)).astype(np.float32)
    return (a @ a.T + np.eye(n, dtype=np.float32) * scale).astype(np.float32)


def _fwd_sub(L, y):
    """forward substitution, float64 reference (vector or matrix rhs)"""
    L = L.astype(np.float64)
    y = np.atleast_2d(y.T).T.astype(np.float64) if y.ndim == 2 else y.astype(np.float64)
    x = np.zeros_like(y)
    for i in range(L.shape[0]):
        s = y[i] - L[i, :i] @ x[:i]
        x[i] = s / L[i, i]
    return x


def _bwd_sub(U, y):
    """back substitution, float64 reference (vector or matrix rhs)"""
    U = U.astype(np.float64)
    y = y.astype(np.float64)
    x = np.zeros_like(y)
    for i in range(U.shape[0] - 1, -1, -1):
        s = y[i] - U[i, i + 1 :] @ x[i + 1 :]
        x[i] = s / U[i, i]
    return x


class TestTileMatmul(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_matmul_4x6_6x4(self):
        rng = np.random.default_rng(21)
        n_blocks = 2
        a = rng.uniform(-2, 2, (n_blocks * 4, 6)).astype(np.float32)
        b = rng.uniform(-2, 2, (6, n_blocks * 4)).astype(np.float32)
        aa = wp.zeros((n_blocks * 4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        bb = wp.zeros((6, n_blocks * 4), dtype=wp.float32, device="metal")
        bb.numpy_view()[:] = b
        out = wp.full((n_blocks * 4, n_blocks * 4), -7.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_mm, dim=n_blocks, inputs=[aa, bb, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = np.full((n_blocks * 4, n_blocks * 4), -7.0, dtype=np.float32)
        for blk in range(n_blocks):
            want[blk * 4 : (blk + 1) * 4, blk * 4 : (blk + 1) * 4] = (
                a[blk * 4 : (blk + 1) * 4, :].astype(np.float64) @ b[:, blk * 4 : (blk + 1) * 4].astype(np.float64)
            )
        np.testing.assert_allclose(out.numpy_view(), want, rtol=1e-4, atol=1e-4)

    def test_matmul_accumulate_alpha_beta(self):
        # io = alpha * A @ B + beta * io, in place through the loaded tile
        rng = np.random.default_rng(22)
        n_blocks = 1
        a = rng.uniform(-2, 2, (4, 6)).astype(np.float32)
        b = rng.uniform(-2, 2, (6, 4)).astype(np.float32)
        c0 = rng.uniform(-2, 2, (4, 4)).astype(np.float32)
        aa = wp.zeros((4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        bb = wp.zeros((6, 4), dtype=wp.float32, device="metal")
        bb.numpy_view()[:] = b
        io = wp.zeros((4, 4), dtype=wp.float32, device="metal")
        io.numpy_view()[:] = c0
        wp.launch_tiled(k_mm_acc, dim=n_blocks, inputs=[aa, bb, io], block_dim=BD, device="metal")
        wp.synchronize()
        want = 2.0 * (a.astype(np.float64) @ b.astype(np.float64)) + 3.0 * c0.astype(np.float64)
        np.testing.assert_allclose(io.numpy_view(), want, rtol=1e-4, atol=1e-4)


class TestTileViewAliases(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_transpose_store_reordered(self):
        rng = np.random.default_rng(23)
        n_blocks = 2
        a = rng.uniform(-5, 5, (n_blocks * 4, 6)).astype(np.float32)
        aa = wp.zeros((n_blocks * 4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.full((6, n_blocks * 4), -7.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_transpose, dim=n_blocks, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = np.full((6, n_blocks * 4), -7.0, dtype=np.float32)
        for blk in range(n_blocks):
            want[:, blk * 4 : (blk + 1) * 4] = a[blk * 4 : (blk + 1) * 4, :].T
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-6)

    def test_reshape_flatten_row_major(self):
        rng = np.random.default_rng(24)
        n_blocks = 2
        a = rng.uniform(-5, 5, (n_blocks * 4, 6)).astype(np.float32)
        aa = wp.zeros((n_blocks * 4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.full(n_blocks * 24, -7.0, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_reshape, dim=n_blocks, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = np.full(n_blocks * 24, -7.0, dtype=np.float32)
        for blk in range(n_blocks):
            want[blk * 24 : (blk + 1) * 24] = a[blk * 4 : (blk + 1) * 4, :].reshape(24)
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-6)

    def test_reshape_write_through_visible_in_source(self):
        rng = np.random.default_rng(25)
        n_blocks = 1
        a = rng.uniform(-5, 5, (4, 6)).astype(np.float32)
        s = rng.uniform(10, 20, 4).astype(np.float32)
        aa = wp.zeros((4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        ss = wp.zeros(4, dtype=wp.float32, device="metal")
        ss.numpy_view()[:] = s
        out = wp.zeros((4, 6), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_reshape_write_through, dim=n_blocks, inputs=[aa, ss, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = a.copy()
        want[1, 1:5] = s  # flat offset 7..10 of the (4,6) reshape
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-6)

    def test_view_write_through_visible_in_source(self):
        rng = np.random.default_rng(26)
        n_blocks = 2
        a = rng.uniform(-5, 5, (n_blocks * 4, 6)).astype(np.float32)
        s = rng.uniform(10, 20, (n_blocks * 2, 2)).astype(np.float32)
        aa = wp.zeros((n_blocks * 4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        ss = wp.zeros((n_blocks * 2, 2), dtype=wp.float32, device="metal")
        ss.numpy_view()[:] = s
        out = wp.zeros((n_blocks * 4, 6), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_view_write_through, dim=n_blocks, inputs=[aa, ss, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = a.copy()
        for blk in range(n_blocks):
            want[blk * 4 + 1 : blk * 4 + 3, 1:3] = s[blk * 2 : (blk + 1) * 2, :]
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-6)

    def test_assign_overlap_snapshot_semantics(self):
        # numpy assignment semantics: the aliased source is snapshotted before
        # the shifted write, so cols 2..3 of the ORIGINAL survive in cols 4..5
        rng = np.random.default_rng(27)
        n_blocks = 2
        a = rng.uniform(-5, 5, (n_blocks * 8, 8)).astype(np.float32)
        aa = wp.zeros((n_blocks * 8, 8), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out = wp.zeros((n_blocks * 8, 8), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_assign_overlap, dim=n_blocks, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = a.copy()
        for blk in range(n_blocks):
            t = a[blk * 8 : (blk + 1) * 8, :]
            want[blk * 8 : (blk + 1) * 8, 2:6] = t[:, 0:4]
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-6)


class TestTileDiagAddAxisReduce(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_diag_add(self):
        rng = np.random.default_rng(28)
        n_blocks = 2
        a = rng.uniform(-5, 5, (n_blocks * 4, 4)).astype(np.float32)
        d = rng.uniform(0.5, 2.0, n_blocks * 4).astype(np.float32)
        aa = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        dd = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        dd.numpy_view()[:] = d
        out = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_diag_add, dim=n_blocks, inputs=[aa, dd, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = a.copy()
        for blk in range(n_blocks):
            want[blk * 4 : (blk + 1) * 4] += np.eye(4, dtype=np.float32) * d[blk * 4 : (blk + 1) * 4, None]
        np.testing.assert_allclose(out.numpy_view(), want, rtol=TOL, atol=1e-6)

    def test_axis_sum_both_axes(self):
        rng = np.random.default_rng(29)
        n_blocks = 2
        a = rng.uniform(-5, 5, (n_blocks * 4, 6)).astype(np.float32)
        aa = wp.zeros((n_blocks * 4, 6), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        out0 = wp.zeros(n_blocks * 6, dtype=wp.float32, device="metal")
        out1 = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_axis_sum, dim=n_blocks, inputs=[aa, out0, out1], block_dim=BD, device="metal")
        wp.synchronize()
        for blk in range(n_blocks):
            t = a[blk * 4 : (blk + 1) * 4, :]
            np.testing.assert_allclose(
                out0.numpy_view()[blk * 6 : (blk + 1) * 6], t.sum(axis=0, dtype=np.float64), rtol=1e-4, atol=1e-4
            )
            np.testing.assert_allclose(
                out1.numpy_view()[blk * 4 : (blk + 1) * 4], t.sum(axis=1, dtype=np.float64), rtol=1e-4, atol=1e-4
            )


class TestTileScan(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_inclusive_exclusive(self):
        # Hillis-Steele deviates from upstream's sequential float order, so
        # the comparison carries a 1e-4 relative tolerance
        rng = np.random.default_rng(30)
        n_blocks = 3
        a = rng.uniform(-5, 5, n_blocks * 8).astype(np.float32)
        aa = wp.zeros(n_blocks * 8, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        outi = wp.zeros(n_blocks * 8, dtype=wp.float32, device="metal")
        oute = wp.zeros(n_blocks * 8, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_scan, dim=n_blocks, inputs=[aa, outi, oute], block_dim=BD, device="metal")
        wp.synchronize()
        for blk in range(n_blocks):
            chunk = a[blk * 8 : (blk + 1) * 8].astype(np.float64)
            np.testing.assert_allclose(outi.numpy_view()[blk * 8 : (blk + 1) * 8], np.cumsum(chunk), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                oute.numpy_view()[blk * 8 : (blk + 1) * 8], np.concatenate([[0.0], np.cumsum(chunk)[:-1]]),
                rtol=1e-4, atol=1e-4,
            )

    def test_plain_launch_degenerate(self):
        # plain launch scalar tier: inclusive prefix = the element itself,
        # exclusive prefix = the empty sum
        rng = np.random.default_rng(31)
        a = rng.uniform(-5, 5, 16).astype(np.float32)
        aa = wp.zeros(16, dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = a
        outi = wp.full(16, -1.0, dtype=wp.float32, device="metal")
        oute = wp.full(16, -1.0, dtype=wp.float32, device="metal")
        wp.launch(k_plain_scan, dim=16, inputs=[aa, outi, oute], device="metal")
        wp.synchronize()
        np.testing.assert_allclose(outi.numpy_view(), a, rtol=TOL)
        np.testing.assert_allclose(oute.numpy_view(), np.zeros(16, dtype=np.float32), rtol=TOL)


class TestTileCholeskySolve(unittest.TestCase):
    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def _spd_blocks(self, seed, n_blocks, n=4):
        rng = np.random.default_rng(seed)
        return np.concatenate([_spd(rng, n)[None] for _ in range(n_blocks)], axis=0)

    def test_cholesky_lower(self):
        n_blocks = 3
        h = self._spd_blocks(32, n_blocks)
        aa = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = h.reshape(n_blocks * 4, 4)
        out = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_chol, dim=n_blocks, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = np.concatenate([np.linalg.cholesky(h[b].astype(np.float64))[None] for b in range(n_blocks)])
        np.testing.assert_allclose(out.numpy_view().reshape(n_blocks, 4, 4), want, rtol=1e-4, atol=1e-5)

    def test_cholesky_inplace(self):
        n_blocks = 2
        h = self._spd_blocks(33, n_blocks)
        aa = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        aa.numpy_view()[:] = h.reshape(n_blocks * 4, 4)
        out = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_chol_inplace, dim=n_blocks, inputs=[aa, out], block_dim=BD, device="metal")
        wp.synchronize()
        want = np.concatenate([np.linalg.cholesky(h[b].astype(np.float64))[None] for b in range(n_blocks)])
        np.testing.assert_allclose(out.numpy_view().reshape(n_blocks, 4, 4), want, rtol=1e-4, atol=1e-5)

    def test_lower_solve_vector_and_matrix(self):
        rng = np.random.default_rng(34)
        n_blocks = 2
        ls = np.concatenate([np.linalg.cholesky(_spd(rng, 4)).astype(np.float32)[None] for _ in range(n_blocks)])
        yv = rng.uniform(-5, 5, n_blocks * 4).astype(np.float32)
        ym = rng.uniform(-5, 5, (n_blocks * 4, 2)).astype(np.float32)
        ll = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        ll.numpy_view()[:] = ls.reshape(n_blocks * 4, 4)
        yy = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        yy.numpy_view()[:] = yv
        ymm = wp.zeros((n_blocks * 4, 2), dtype=wp.float32, device="metal")
        ymm.numpy_view()[:] = ym
        outv = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        outm = wp.zeros((n_blocks * 4, 2), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_lower_solve, dim=n_blocks, inputs=[ll, yy, outv], block_dim=BD, device="metal")
        wp.launch_tiled(k_lower_solve_m, dim=n_blocks, inputs=[ll, ymm, outm], block_dim=BD, device="metal")
        wp.synchronize()
        for b in range(n_blocks):
            np.testing.assert_allclose(
                outv.numpy_view()[b * 4 : (b + 1) * 4], _fwd_sub(ls[b], yv[b * 4 : (b + 1) * 4]), rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                outm.numpy_view()[b * 4 : (b + 1) * 4, :], _fwd_sub(ls[b], ym[b * 4 : (b + 1) * 4, :]), rtol=1e-4, atol=1e-5
            )

    def test_upper_solve_via_transpose(self):
        rng = np.random.default_rng(35)
        n_blocks = 2
        ls = np.concatenate([np.linalg.cholesky(_spd(rng, 4)).astype(np.float32)[None] for _ in range(n_blocks)])
        ym = rng.uniform(-5, 5, (n_blocks * 4, 2)).astype(np.float32)
        ll = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        ll.numpy_view()[:] = ls.reshape(n_blocks * 4, 4)
        ymm = wp.zeros((n_blocks * 4, 2), dtype=wp.float32, device="metal")
        ymm.numpy_view()[:] = ym
        outm = wp.zeros((n_blocks * 4, 2), dtype=wp.float32, device="metal")
        wp.launch_tiled(k_upper_solve, dim=n_blocks, inputs=[ll, ymm, outm], block_dim=BD, device="metal")
        wp.synchronize()
        for b in range(n_blocks):
            np.testing.assert_allclose(
                outm.numpy_view()[b * 4 : (b + 1) * 4, :], _bwd_sub(ls[b].T, ym[b * 4 : (b + 1) * 4, :]), rtol=1e-4, atol=1e-5
            )

    def test_cholesky_solve_lower_upper(self):
        rng = np.random.default_rng(36)
        n_blocks = 2
        h = self._spd_blocks(37, n_blocks)
        yv = rng.uniform(-5, 5, n_blocks * 4).astype(np.float32)
        # the solve family takes the FACTOR in the fill_mode triangle
        # (upstream names the argument L): lower fill reads the lower
        # triangle of the input, upper fill reads its upper triangle —
        # feed chol(h, "lower") for the lower-fill kernel and its transpose
        # (= chol(h, "upper")) for the upper-fill kernel
        ls = np.linalg.cholesky(h).astype(np.float32)
        hh = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        hh.numpy_view()[:] = ls.reshape(n_blocks * 4, 4)
        hu = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        hu.numpy_view()[:] = ls.transpose(0, 2, 1).reshape(n_blocks * 4, 4)
        yy = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        yy.numpy_view()[:] = yv
        outl = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        outu = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_chol_solve, dim=n_blocks, inputs=[hh, yy, outl], block_dim=BD, device="metal")
        wp.launch_tiled(k_chol_solve_upper, dim=n_blocks, inputs=[hu, yy, outu], block_dim=BD, device="metal")
        wp.synchronize()
        for b in range(n_blocks):
            want = np.linalg.solve(h[b].astype(np.float64), yv[b * 4 : (b + 1) * 4].astype(np.float64))
            np.testing.assert_allclose(outl.numpy_view()[b * 4 : (b + 1) * 4], want, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(outu.numpy_view()[b * 4 : (b + 1) * 4], want, rtol=1e-4, atol=1e-5)

    def test_solve_inplace_variants(self):
        rng = np.random.default_rng(38)
        n_blocks = 2
        ls = np.concatenate([np.linalg.cholesky(_spd(rng, 4)).astype(np.float32)[None] for _ in range(n_blocks)])
        yv = rng.uniform(-5, 5, n_blocks * 4).astype(np.float32)
        ll = wp.zeros((n_blocks * 4, 4), dtype=wp.float32, device="metal")
        ll.numpy_view()[:] = ls.reshape(n_blocks * 4, 4)
        yy = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        yy.numpy_view()[:] = yv
        yl = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        yl.numpy_view()[:] = yv
        yu = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        yu.numpy_view()[:] = yv
        yc = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        yc.numpy_view()[:] = yv
        outl = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        outu = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        outc = wp.zeros(n_blocks * 4, dtype=wp.float32, device="metal")
        wp.launch_tiled(k_lower_solve_inplace, dim=n_blocks, inputs=[ll, yl, outl], block_dim=BD, device="metal")
        wp.launch_tiled(k_upper_solve_inplace, dim=n_blocks, inputs=[ll, yu, outu], block_dim=BD, device="metal")
        wp.launch_tiled(k_chol_solve_inplace, dim=n_blocks, inputs=[ll, yc, outc], block_dim=BD, device="metal")
        wp.synchronize()
        for b in range(n_blocks):
            y = yv[b * 4 : (b + 1) * 4].astype(np.float64)
            np.testing.assert_allclose(outl.numpy_view()[b * 4 : (b + 1) * 4], _fwd_sub(ls[b], yv[b * 4 : (b + 1) * 4]), rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(outu.numpy_view()[b * 4 : (b + 1) * 4], _bwd_sub(ls[b].T, yv[b * 4 : (b + 1) * 4]), rtol=1e-4, atol=1e-5)
            want = np.linalg.solve(ls[b].astype(np.float64) @ ls[b].astype(np.float64).T, y)
            np.testing.assert_allclose(outc.numpy_view()[b * 4 : (b + 1) * 4], want, rtol=1e-4, atol=1e-5)


class TestTileP34Pins(unittest.TestCase):
    """Translate-only pins for the P3+P4 closure (CPU-safe)."""

    def _translate(self, kern, block_dim=None):
        from warp_metal.msl.translate import translate_kernel

        opts = {"block_dim": block_dim} if block_dim else {}
        return translate_kernel(kern, opts)

    def test_alias_decl_claims_no_slot(self):
        # a tile_transpose result is a metadata alias: the only threadgroup
        # array is the source tile's — the alias borrows its slot
        @wp.kernel
        def k_alias(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
            t = wp.tile_load(a, shape=(4, 6), storage="shared")
            tt = wp.tile_transpose(t)
            wp.tile_store(out, tt, offset=(0, 0))

        u = self._translate(k_alias, BD)
        self.assertEqual(u.shr_arrays, [("float", 24)])
        src = u.source()
        self.assertIn("_wp_tile_shr_0[24]", src)
        self.assertNotIn("_wp_tile_shr_1[", src)

    def test_plain_cholesky_identity(self):
        # plain scalar tier: the (1,1) factor is sqrt(a) — the n=1 serial
        # case upstream runs at WP_TILE_BLOCK_DIM == 1; no barriers, no
        # shared arrays
        @wp.kernel
        def k_plain_chol(out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            t = wp.tile_zeros((1, 1), dtype=wp.float32, storage="shared")
            wp.tile_cholesky_inplace(t)
            wp.tile_store(out, wp.tile_reshape(t, shape=(1,)), offset=(i,))

        u = self._translate(k_plain_chol)
        src = u.source()
        self.assertIn("= sqrt(", src)
        self.assertNotIn("threadgroup_barrier", src)
        self.assertEqual(u.shr_arrays, [])

    def test_plain_solve_guarded_division(self):
        # plain scalar tier: each solve phase is one diag-guarded division
        # (upstream's `(diag != 0.0f) ? s/diag : s`)
        @wp.kernel
        def k_plain_solve(a: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            i = wp.tid()
            l = wp.tile_zeros((1, 1), dtype=wp.float32, storage="shared")
            v = wp.tile_load(y, shape=(1,), offset=(i,))
            z = wp.tile_lower_solve(l, v)
            wp.tile_store(out, z, offset=(i,))

        u = self._translate(k_plain_solve)
        src = u.source()
        self.assertIn("!= 0.0f", src)
        self.assertNotIn("threadgroup_barrier", src)
        self.assertEqual(u.shr_arrays, [])

    def test_matmul_strided_gemm_loop(self):
        # the launch_tiled GEMM: per-thread strided output elements, k
        # ascending (upstream's accumulation order), one end barrier
        u = self._translate(k_mm, BD)
        src = u.source()
        self.assertIn("for (int _wp_g = _wp_lane;", src)
        self.assertIn("for (int _wp_k = 0; _wp_k < 6; ++_wp_k)", src)
        self.assertIn("* _wp_acc + ", src)
        self.assertIn("threadgroup_barrier(mem_flags::mem_threadgroup);", src)

    def test_cholesky_statement_inplace_form(self):
        # wp.tile_cholesky_inplace lowers as a statement: thread-0 diagonal
        # write, strided row updates, one barrier per column — upstream's
        # scalar_cholesky_impl verbatim
        u = self._translate(k_chol_inplace, BD)
        src = u.source()
        self.assertIn("if (_wp_lane == 0)", src)
        self.assertIn("= sqrt(_wp_s);", src)
        self.assertIn("float _wp_inv_s = 1.0f / _wp_s;", src)


if __name__ == "__main__":
    unittest.main()
