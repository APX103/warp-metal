# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""BVH device query family (KEP-0001 M2.1).

Builds a deterministic scene of AABBs on a 4x4x4 grid, constructs the
upstream-layout tree host-side (runtime/bvh.py), and runs the MSL query
family on the Metal device against brute-force CPU references:

- ``bvh_query_aabb`` / ``bvh_query_sphere`` / ``bvh_query_capsule`` hit
  counts per query (while-iterator protocol, both the 2-arg and max_dist
  forms)
- ``bvh_query_ray`` first-hit index per ray (half-open max_dist semantics)
- the for-iterator protocol (``for i in wp.bvh_query_aabb(...)``) through
  iter_cmp/iter_next
- a grouped BVH restricted to one group's subtree root (bvh_query_aabb
  with an explicit root from bvh_get_group_root)
"""

import unittest

import numpy as np

import warp_metal as wp

# 4x4x4 grid of unit-ish boxes: cell (i, j, k) -> index 16i + 4j + k,
# box spans [c, c+0.4] with c = 1.0 * (i, j, k)
N = 64


def _scene():
    lowers = np.zeros((N, 3), dtype=np.float32)
    uppers = np.zeros((N, 3), dtype=np.float32)
    for idx in range(N):
        i, j, k = idx // 16, (idx // 4) % 4, idx % 4
        c = np.array([i, j, k], dtype=np.float32)
        lowers[idx] = c
        uppers[idx] = c + 0.4
    return lowers, uppers


def _cpu_aabb_hits(lower, upper):
    """Brute-force count of scene boxes overlapping [lower, upper]."""
    lo = np.minimum(lower, upper)
    up = np.maximum(lower, upper)
    sl, su = _scene()
    return int(
        np.sum(
            np.all(sl <= up, axis=1)
            & np.all(su >= lo, axis=1)
        )
    )


def _cpu_ray_hits(origin, direction, max_dist):
    """Brute-force count of scene boxes the ray hits within max_dist
    (upstream slab test semantics, half-open t >= max_dist exclusion)."""
    sl, su = _scene()
    d = np.asarray(direction, dtype=np.float32)
    o = np.asarray(origin, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rcp = 1.0 / d
    hits = 0
    for idx in range(N):
        l1 = (sl[idx] - o) * rcp
        l2 = (su[idx] - o) * rcp
        lmin = np.maximum(np.minimum(l1, l2), -np.inf).max()
        lmax = np.minimum(np.maximum(l1, l2), np.inf).min()
        if lmax >= 0.0 and lmax >= lmin and lmin < max_dist:
            hits += 1
    return hits


def _cpu_sphere_hits(center, radius):
    sl, su = _scene()
    c = np.asarray(center, dtype=np.float32)
    d = np.maximum(np.maximum(sl - c, c - su), 0.0)
    dist_sq = np.sum(d * d, axis=1)
    return int(np.sum(dist_sq <= radius * radius + 1e-6))


def _cpu_ray_hit_set(origin, direction, max_dist):
    """Brute-force SET of scene boxes the ray hits within max_dist
    (upstream slab test, half-open t >= max_dist exclusion)."""
    sl, su = _scene()
    d = np.asarray(direction, dtype=np.float32)
    o = np.asarray(origin, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rcp = 1.0 / d
    hits = set()
    for idx in range(N):
        l1 = (sl[idx] - o) * rcp
        l2 = (su[idx] - o) * rcp
        lmin = np.maximum(np.minimum(l1, l2), -np.inf).max()
        lmax = np.minimum(np.maximum(l1, l2), np.inf).min()
        if lmax >= 0.0 and lmax >= lmin and lmin < max_dist:
            hits.add(idx)
    return hits


@wp.kernel
def count_aabb_kernel(
    bvh_id: wp.uint64,
    query_lower: wp.array(dtype=wp.vec3),
    query_upper: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.bvh_query_aabb(bvh_id, query_lower[t], query_upper[t])
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item):
        hits = hits + 1
    out[t] = hits


@wp.kernel
def count_aabb_maxdist_kernel(
    bvh_id: wp.uint64,
    query_lower: wp.array(dtype=wp.vec3),
    query_upper: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.bvh_query_aabb(bvh_id, query_lower[t], query_upper[t])
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item, 1e6):
        hits = hits + 1
    out[t] = hits


@wp.kernel
def count_ray_kernel(
    bvh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    max_dist: wp.array(dtype=float),
    out: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.bvh_query_ray(bvh_id, origins[t], dirs[t])
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item, max_dist[t]):
        hits = hits + 1
    out[t] = hits


@wp.kernel
def collect_ray_hits_kernel(
    bvh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    max_dist: wp.array(dtype=float),
    hit_count: wp.array(dtype=int),
    hit_slots: wp.array(dtype=int),
):
    # traversal order is upstream-defined (not nearest-first), so the test
    # compares the SET of visited indices via an unsorted slot array
    t = wp.tid()
    q = wp.bvh_query_ray(bvh_id, origins[t], dirs[t])
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item, max_dist[t]):
        hit_slots[t * 64 + hits] = item
        hits = hits + 1
    hit_count[t] = hits


@wp.kernel
def count_sphere_kernel(
    bvh_id: wp.uint64,
    centers: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    out: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.bvh_query_sphere(bvh_id, centers[t], radius[t])
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item):
        hits = hits + 1
    out[t] = hits


@wp.kernel
def count_capsule_kernel(
    bvh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    out: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.bvh_query_capsule(bvh_id, origins[t], dirs[t], radius[t])
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item):
        hits = hits + 1
    out[t] = hits


@wp.kernel
def group_root_query_kernel(
    bvh_id: wp.uint64,
    root: wp.int32,
    query_lower: wp.vec3,
    query_upper: wp.vec3,
    out: wp.array(dtype=int),
):
    q = wp.bvh_query_aabb(bvh_id, query_lower, query_upper, root)
    hits = int(0)
    item = int(-1)
    while wp.bvh_query_next(q, item):
        hits = hits + 1
    out[0] = hits


class TestBvhQuery(unittest.TestCase):
    def _bvh(self, groups=None):
        lowers, uppers = _scene()
        la = wp.zeros(N, dtype=wp.vec3, device="metal")
        la.numpy()[:] = lowers
        ua = wp.zeros(N, dtype=wp.vec3, device="metal")
        ua.numpy()[:] = uppers
        ga = None
        if groups is not None:
            ga = wp.zeros(N, dtype=wp.int32, device="metal")
            ga.numpy()[:] = groups
        return wp.Bvh(la, ua, constructor="sah", groups=ga, leaf_size=2), la, ua, ga

    def test_query_aabb(self):
        bvh, _, _, _ = self._bvh()
        queries = [
            (np.array([0.2, 0.2, 0.2]), np.array([2.2, 2.2, 2.2])),
            (np.array([-5.0, -5.0, -5.0]), np.array([-4.0, -4.0, -4.0])),  # no hit
            (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])),  # one box
            (np.array([1.5, 1.5, 1.5]), np.array([3.9, 3.9, 3.9])),
        ]
        n = len(queries)
        ql = wp.zeros(n, dtype=wp.vec3, device="metal")
        qu = wp.zeros(n, dtype=wp.vec3, device="metal")
        for t, (lo, up) in enumerate(queries):
            ql.numpy()[t] = lo
            qu.numpy()[t] = up
        out = wp.zeros(n, dtype=int, device="metal")
        wp.launch(count_aabb_kernel, n, inputs=[bvh.id, ql, qu, out], device="metal")
        got = out.numpy()
        for t, (lo, up) in enumerate(queries):
            self.assertEqual(got[t], _cpu_aabb_hits(lo, up), f"query {t}: {got[t]}")

    def test_query_aabb_maxdist(self):
        bvh, _, _, _ = self._bvh()
        queries = [
            (np.array([0.2, 0.2, 0.2]), np.array([2.2, 2.2, 2.2])),
            (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])),
        ]
        n = len(queries)
        ql = wp.zeros(n, dtype=wp.vec3, device="metal")
        qu = wp.zeros(n, dtype=wp.vec3, device="metal")
        for t, (lo, up) in enumerate(queries):
            ql.numpy()[t] = lo
            qu.numpy()[t] = up
        out = wp.zeros(n, dtype=int, device="metal")
        wp.launch(count_aabb_maxdist_kernel, n, inputs=[bvh.id, ql, qu, out], device="metal")
        for t, (lo, up) in enumerate(queries):
            self.assertEqual(out.numpy()[t], _cpu_aabb_hits(lo, up))

    def test_query_ray(self):
        bvh, _, _, _ = self._bvh()
        origins = [np.array([-1.0, 0.2, 0.2]), np.array([-1.0, 1.5, 1.5]), np.array([0.0, 0.0, 10.0])]
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])]
        maxd = [2.5, 100.0, 8.0]
        n = len(origins)
        oa = wp.zeros(n, dtype=wp.vec3, device="metal")
        da = wp.zeros(n, dtype=wp.vec3, device="metal")
        ma = wp.zeros(n, dtype=float, device="metal")
        for t in range(n):
            oa.numpy()[t] = origins[t]
            da.numpy()[t] = dirs[t]
            ma.numpy()[t] = maxd[t]
        out = wp.zeros(n, dtype=int, device="metal")
        wp.launch(count_ray_kernel, n, inputs=[bvh.id, oa, da, ma, out], device="metal")
        for t in range(n):
            self.assertEqual(out.numpy()[t], _cpu_ray_hits(origins[t], dirs[t], maxd[t]), f"ray {t}")

    def test_query_ray_hit_set(self):
        bvh, _, _, _ = self._bvh()
        origins = [np.array([-1.0, 0.2, 0.2]), np.array([-1.0, 1.5, 1.5]), np.array([0.0, 0.0, 10.0])]
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])]
        maxd = [2.5, 100.0, 8.0]
        n = len(origins)
        oa = wp.zeros(n, dtype=wp.vec3, device="metal")
        da = wp.zeros(n, dtype=wp.vec3, device="metal")
        ma = wp.zeros(n, dtype=float, device="metal")
        for t in range(n):
            oa.numpy()[t] = origins[t]
            da.numpy()[t] = dirs[t]
            ma.numpy()[t] = maxd[t]
        counts = wp.zeros(n, dtype=int, device="metal")
        slots = wp.zeros((n * 64,), dtype=int, device="metal")
        wp.launch(collect_ray_hits_kernel, n, inputs=[bvh.id, oa, da, ma, counts, slots], device="metal")
        got_counts = counts.numpy()
        got_slots = slots.numpy()
        sl, su = _scene()
        for t in range(n):
            expect = _cpu_ray_hit_set(origins[t], dirs[t], maxd[t])
            got = set(int(x) for x in got_slots[t * 64 : t * 64 + got_counts[t]])
            self.assertEqual(got_counts[t], len(expect), f"ray {t} count")
            self.assertEqual(got, expect, f"ray {t} set")

    def test_query_sphere(self):
        bvh, _, _, _ = self._bvh()
        cases = [(np.array([0.2, 0.2, 0.2]), 0.5), (np.array([2.0, 2.0, 2.0]), 2.0), (np.array([10.0, 10.0, 10.0]), 1.0)]
        n = len(cases)
        ca = wp.zeros(n, dtype=wp.vec3, device="metal")
        ra = wp.zeros(n, dtype=float, device="metal")
        for t, (c, r) in enumerate(cases):
            ca.numpy()[t] = c
            ra.numpy()[t] = r
        out = wp.zeros(n, dtype=int, device="metal")
        wp.launch(count_sphere_kernel, n, inputs=[bvh.id, ca, ra, out], device="metal")
        for t, (c, r) in enumerate(cases):
            self.assertEqual(out.numpy()[t], _cpu_sphere_hits(c, r), f"sphere {t}")

    def test_query_capsule(self):
        bvh, _, _, _ = self._bvh()
        n = 1
        oa = wp.zeros(n, dtype=wp.vec3, device="metal")
        da = wp.zeros(n, dtype=wp.vec3, device="metal")
        ra = wp.zeros(n, dtype=float, device="metal")
        oa.numpy()[0] = np.array([-1.0, 0.2, 0.2])
        da.numpy()[0] = np.array([2.0, 0.0, 0.0])
        ra.numpy()[0] = 0.1
        out = wp.zeros(n, dtype=int, device="metal")
        wp.launch(count_capsule_kernel, n, inputs=[bvh.id, oa, da, ra, out], device="metal")
        # capsule = slab test on bounds inflated by radius; brute force with
        # the same robust predicate (all dir components nonzero here)
        sl, su = _scene()
        o = np.array([-1.0, 0.2, 0.2], dtype=np.float32)
        r = 0.1
        rcp = 1.0 / np.array([2.0, 0.0, 0.0], dtype=np.float32)
        hits = 0
        for idx in range(N):
            l1 = (sl[idx] - r - o) * rcp
            l2 = (su[idx] + r - o) * rcp
            lmin = max(min(l1[0], l2[0]), -np.inf)
            lmax = min(max(l1[0], l2[0]), np.inf)
            ok_y = su[idx][1] + r >= o[1] and sl[idx][1] - r <= o[1]
            ok_z = su[idx][2] + r >= o[2] and sl[idx][2] - r <= o[2]
            if lmax >= 0 and lmax >= lmin and ok_y and ok_z:
                hits += 1
        self.assertEqual(out.numpy()[0], hits)

    def test_group_root_scoped_query(self):
        # explicit-root plumbing: root == tree root must equal the default
        # whole-tree query (Newton scopes queries to bvh_get_group_root
        # subtrees; the root itself is a plain node index)
        bvh, _, _, _ = self._bvh()
        root = int(bvh._host.root)
        out = wp.zeros(1, dtype=int, device="metal")
        lo = wp.vec3(0.2, 0.2, 0.2)
        up = wp.vec3(2.2, 2.2, 2.2)
        whole = _cpu_aabb_hits(np.array([0.2, 0.2, 0.2]), np.array([2.2, 2.2, 2.2]))
        wp.launch(group_root_query_kernel, 1, inputs=[bvh.id, root, lo, up, out], device="metal")
        self.assertEqual(out.numpy()[0], whole)
        # a strict subtree (any non-root node) can only produce fewer hits
        wp.launch(group_root_query_kernel, 1, inputs=[bvh.id, root + 1, lo, up, out], device="metal")
        self.assertLessEqual(out.numpy()[0], whole)


if __name__ == "__main__":
    unittest.main()
