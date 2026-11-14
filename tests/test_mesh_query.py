# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Mesh device query family (KEP-0001 M2.2).

Builds a deterministic two-part mesh (quad + tetrahedron, 8 points / 6
faces) as a warp-metal wp.Mesh (host SAH tree over per-triangle bounds,
upstream mesh.h layout) and runs the MSL query family on Metal against
numpy brute-force references over the same faces:

- ``mesh_query_point_no_sign`` (struct form): closest face + barycentrics
  vs a brute-force closest_point_to_triangle scan
- ``mesh_query_point_no_sign`` (ref-out form): face/u/v/bool outputs
- ``mesh_eval_position`` / ``mesh_eval_velocity`` / ``mesh_eval_face_normal``
  vs direct numpy interpolation (velocities array present)
- ``mesh_query_ray`` (struct + ref-out forms): closest hit vs a
  brute-force ray-triangle scan (Moeller-Trumbore reference)
- ``mesh_query_aabb`` / ``mesh_query_sphere`` iteration: hit-face SETS vs
  face-AABB / sphere-AABB overlap scans (broad-phase semantics: the AABB
  iterator reports faces whose bounds overlap, not triangle tests)
"""

import unittest

import numpy as np

import warp_metal as wp

# ---- deterministic scene: quad (z=0) + tetrahedron (offset in x) -----------

POINTS = np.array(
    [
        # quad
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        # tetrahedron
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [3.5, 1.0, 0.0],
        [3.5, 0.4, 0.9],
    ],
    dtype=np.float32,
)
INDICES = np.array(
    [
        0, 1, 2,  # quad lower
        0, 2, 3,  # quad upper
        4, 5, 6,  # tetra base
        4, 6, 7,  # tetra side a
        5, 7, 6,  # tetra side b
        4, 7, 5,  # tetra side c
    ],
    dtype=np.int32,
)
# distinct per-vertex velocities so barycentric interpolation is sensitive
VELOCITIES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, -1.0],
        [0.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)
N_TRI = len(INDICES) // 3

FACE_LOWER = POINTS[INDICES.reshape(N_TRI, 3)].min(axis=1)
FACE_UPPER = POINTS[INDICES.reshape(N_TRI, 3)].max(axis=1)


# ---- numpy references (mesh.h / intersect.h semantics) ----------------------


def _closest_point_to_triangle(a, b, c, p):
    """intersect.h closest_point_to_triangle: returns (u, v) barycentrics."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return (1.0, 0.0)
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return (0.0, 1.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return (1.0 - v, v)
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return (0.0, 0.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return (1.0 - w, 0.0)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return (0.0, 1.0 - w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return (1.0 - v - w, v)


def _closest_face(point, max_dist):
    """Brute-force mesh_query_point_no_sign (sliver skip included)."""
    best = None  # (dist_sq, face, u, v)
    tris = POINTS[INDICES.reshape(N_TRI, 3)]
    for f in range(N_TRI):
        a, b, c = tris[f]
        e0, e1, e2 = b - a, c - a, c - b
        normal = np.cross(e0, e1)
        if np.linalg.norm(normal) / (np.dot(e0, e0) + np.dot(e1, e1) + np.dot(e2, e2)) < 1.0e-6:
            continue
        u, v = _closest_point_to_triangle(a, b, c, point)
        w = 1.0 - u - v
        cp = u * a + v * b + w * c
        dist_sq = float(np.dot(cp - point, cp - point))
        if dist_sq < max_dist * max_dist and (best is None or dist_sq < best[0]):
            best = (dist_sq, f, u, v)
    if best is None:
        return None
    _, f, u, v = best
    # warp's mesh_query u/v are closest_point_to_triangle's (u, v): the
    # barycentric coefficients of (p, q, r); closest = u*p + v*q + (1-u-v)*r
    return f, u, v


def _ray_triangle(p, d, a, b, c):
    """Moeller-Trumbore reference; returns (t, u, v, sign) or None."""
    e1 = b - a
    e2 = c - a
    h = np.cross(d, e2)
    det = np.dot(e1, h)
    if abs(det) < 1.0e-20:
        return None
    sign = det
    f = 1.0 / det
    s = p - a
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(e2, q)
    if t < 0.0:
        return None
    return t, u, v, sign


def _closest_ray_hit(start, direction, max_t):
    """Brute-force mesh_query_ray: closest hit with t in [0, max_t)."""
    best = None
    tris = POINTS[INDICES.reshape(N_TRI, 3)]
    for f in range(N_TRI):
        a, b, c = tris[f]
        hit = _ray_triangle(np.asarray(start, dtype=np.float64), np.asarray(direction, dtype=np.float64), a.astype(np.float64), b.astype(np.float64), c.astype(np.float64))
        if hit is None:
            continue
        t, u, v, sign = hit
        if t < max_t and (best is None or t < best[0]):
            best = (t, f, u, v, sign)
    return best


# ---- kernels ----------------------------------------------------------------


@wp.kernel
def query_point_struct_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dists: wp.array(dtype=float),
    out_hit: wp.array(dtype=int),
    out_face: wp.array(dtype=int),
    out_u: wp.array(dtype=float),
    out_v: wp.array(dtype=float),
):
    t = wp.tid()
    res = wp.mesh_query_point_no_sign(mesh_id, points[t], max_dists[t])
    out_hit[t] = 1 if res.result else 0
    out_face[t] = res.face
    out_u[t] = res.u
    out_v[t] = res.v


@wp.kernel
def query_point_ref_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dists: wp.array(dtype=float),
    out_hit: wp.array(dtype=int),
    out_face: wp.array(dtype=int),
    out_u: wp.array(dtype=float),
    out_v: wp.array(dtype=float),
):
    t = wp.tid()
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    found = wp.mesh_query_point_no_sign(mesh_id, points[t], max_dists[t], face, u, v)
    out_hit[t] = 1 if found else 0
    out_face[t] = face
    out_u[t] = u
    out_v[t] = v


@wp.kernel
def eval_family_kernel(
    mesh_id: wp.uint64,
    faces: wp.array(dtype=int),
    us: wp.array(dtype=float),
    vs: wp.array(dtype=float),
    out_pos: wp.array(dtype=wp.vec3),
    out_vel: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_vert: wp.array(dtype=wp.vec3),
    out_index: wp.array(dtype=int),
):
    t = wp.tid()
    f = faces[t]
    out_pos[t] = wp.mesh_eval_position(mesh_id, f, us[t], vs[t])
    out_vel[t] = wp.mesh_eval_velocity(mesh_id, f, us[t], vs[t])
    out_normal[t] = wp.mesh_eval_face_normal(mesh_id, f)
    out_vert[t] = wp.mesh_get_point(mesh_id, f * 3)
    out_index[t] = wp.mesh_get_index(mesh_id, f * 3 + 1)


@wp.kernel
def query_ray_struct_kernel(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    max_t: float,
    out_hit: wp.array(dtype=int),
    out_face: wp.array(dtype=int),
    out_t: wp.array(dtype=float),
    out_normal: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    hit = wp.mesh_query_ray(mesh_id, starts[t], dirs[t], max_t)
    out_hit[t] = 1 if hit.result else 0
    out_face[t] = hit.face
    out_t[t] = hit.t
    out_normal[t] = hit.normal


@wp.kernel
def query_ray_ref_kernel(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    max_t: float,
    out_hit: wp.array(dtype=int),
    out_face: wp.array(dtype=int),
    out_t: wp.array(dtype=float),
    out_sign: wp.array(dtype=float),
):
    t = wp.tid()
    tt = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)
    found = wp.mesh_query_ray(mesh_id, starts[t], dirs[t], max_t, tt, u, v, sign, normal, face)
    out_hit[t] = 1 if found else 0
    out_face[t] = face
    out_t[t] = tt
    out_sign[t] = sign


@wp.kernel
def query_aabb_kernel(
    mesh_id: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    out_mask: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.mesh_query_aabb(mesh_id, lowers[t], uppers[t])
    face = int(-1)
    while wp.mesh_query_next(q, face):
        out_mask[t * N_TRI + face] = 1


@wp.kernel
def query_sphere_kernel(
    mesh_id: wp.uint64,
    centers: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    out_mask: wp.array(dtype=int),
):
    t = wp.tid()
    q = wp.mesh_query_sphere(mesh_id, centers[t], radii[t])
    face = int(-1)
    while wp.mesh_query_next(q, face):
        out_mask[t * N_TRI + face] = 1


class TestMeshQuery(unittest.TestCase):
    def _mesh(self):
        pa = wp.zeros(len(POINTS), dtype=wp.vec3, device="metal")
        pa.numpy()[:] = POINTS
        ia = wp.zeros(len(INDICES), dtype=wp.int32, device="metal")
        ia.numpy()[:] = INDICES
        va = wp.zeros(len(VELOCITIES), dtype=wp.vec3, device="metal")
        va.numpy()[:] = VELOCITIES
        return wp.Mesh(points=pa, indices=ia, velocities=va)

    # -- closest point ---------------------------------------------------------

    def test_query_point_no_sign_struct(self):
        mesh = self._mesh()
        # fixture points numerically verified to have a UNIQUE nearest face
        # (ties across the shared quad diagonal / tetra apex are excluded) or
        # to miss entirely within their max_dist
        queries = [
            ((0.3, 0.7, 0.3), 1.0e6),  # above quad, face 1 side
            ((0.2, 0.8, -0.4), 1.0e6),  # below quad, face 1 side
            ((3.3, 0.9, 0.15), 1.0e6),  # tetra side face 3
            ((0.5, 0.25, 0.5), 1.0e6),  # above quad, face 0
            ((3.7, 0.5, -0.3), 1.0e6),  # below tetra base, face 2
            ((5.0, 5.0, 5.0), 0.5),  # far beyond max_dist: no hit
        ]
        n = len(queries)
        qa = wp.zeros(n, dtype=wp.vec3, device="metal")
        md = wp.zeros(n, dtype=float, device="metal")
        for t, (q, m) in enumerate(queries):
            qa.numpy()[t] = q
            md.numpy()[t] = m
        hit = wp.zeros(n, dtype=int, device="metal")
        face = wp.zeros(n, dtype=int, device="metal")
        u = wp.zeros(n, dtype=float, device="metal")
        v = wp.zeros(n, dtype=float, device="metal")
        wp.launch(query_point_struct_kernel, n, inputs=[mesh.id, qa, md, hit, face, u, v], device="metal")
        got_h, got_f, got_u, got_v = hit.numpy(), face.numpy(), u.numpy(), v.numpy()
        for t, (q, m) in enumerate(queries):
            ref = _closest_face(np.asarray(q, dtype=np.float64), m)
            if ref is None:
                self.assertEqual(got_h[t], 0, f"query {t}: expected no hit")
                continue
            rf, ru, rv = ref
            self.assertEqual(got_h[t], 1, f"query {t}: expected a hit")
            self.assertEqual(int(got_f[t]), rf, f"query {t}: face {got_f[t]} != {rf}")
            self.assertAlmostEqual(float(got_u[t]), ru, places=4, msg=f"query {t}: u")
            self.assertAlmostEqual(float(got_v[t]), rv, places=4, msg=f"query {t}: v")

    def test_query_point_no_sign_ref(self):
        mesh = self._mesh()
        queries = [
            ((3.6, 0.3, 0.2), 1.0e6),  # tetra side face 4
            ((0.8, 0.3, 0.5), 1.0e6),  # quad face 0
            ((-0.4, 0.5, 0.2), 1.0e6),  # quad face 1, off-edge
            ((5.0, 5.0, 5.0), 0.5),  # miss
        ]
        n = len(queries)
        qa = wp.zeros(n, dtype=wp.vec3, device="metal")
        md = wp.zeros(n, dtype=float, device="metal")
        for t, (q, m) in enumerate(queries):
            qa.numpy()[t] = q
            md.numpy()[t] = m
        hit = wp.zeros(n, dtype=int, device="metal")
        face = wp.zeros(n, dtype=int, device="metal")
        u = wp.zeros(n, dtype=float, device="metal")
        v = wp.zeros(n, dtype=float, device="metal")
        wp.launch(query_point_ref_kernel, n, inputs=[mesh.id, qa, md, hit, face, u, v], device="metal")
        for t, (q, m) in enumerate(queries):
            ref = _closest_face(np.asarray(q, dtype=np.float64), m)
            got_h, got_f = hit.numpy()[t], int(face.numpy()[t])
            if ref is None:
                self.assertEqual(got_h, 0)
                continue
            self.assertEqual(got_h, 1)
            self.assertEqual(got_f, ref[0])
            self.assertAlmostEqual(u.numpy()[t], ref[1], places=4)
            self.assertAlmostEqual(v.numpy()[t], ref[2], places=4)

    # -- evals -----------------------------------------------------------------

    def test_eval_family(self):
        mesh = self._mesh()
        cases = [(0, 0.25, 0.5), (1, 0.5, 0.25), (2, 0.3, 0.3), (4, 0.5, 0.5), (5, 0.2, 0.6)]
        n = len(cases)
        fa = wp.zeros(n, dtype=int, device="metal")
        ua = wp.zeros(n, dtype=float, device="metal")
        va = wp.zeros(n, dtype=float, device="metal")
        for t, (f, u, v) in enumerate(cases):
            fa.numpy()[t] = f
            ua.numpy()[t] = u
            va.numpy()[t] = v
        pos = wp.zeros(n, dtype=wp.vec3, device="metal")
        vel = wp.zeros(n, dtype=wp.vec3, device="metal")
        nor = wp.zeros(n, dtype=wp.vec3, device="metal")
        vert = wp.zeros(n, dtype=wp.vec3, device="metal")
        idx = wp.zeros(n, dtype=int, device="metal")
        wp.launch(eval_family_kernel, n, inputs=[mesh.id, fa, ua, va, pos, vel, nor, vert, idx], device="metal")
        tris = POINTS[INDICES.reshape(N_TRI, 3)]
        for t, (f, u, v) in enumerate(cases):
            i, j, k = INDICES[f * 3 : f * 3 + 3]
            p, q, r = tris[f]
            w = 1.0 - u - v
            ref_pos = u * p + v * q + w * r
            ref_vel = u * VELOCITIES[i] + v * VELOCITIES[j] + w * VELOCITIES[k]
            ref_nor = np.cross(q - p, r - p)
            ref_nor = ref_nor / np.linalg.norm(ref_nor)
            np.testing.assert_allclose(pos.numpy()[t], ref_pos, atol=1.0e-5, err_msg=f"eval_position face {f}")
            np.testing.assert_allclose(vel.numpy()[t], ref_vel, atol=1.0e-5, err_msg=f"eval_velocity face {f}")
            np.testing.assert_allclose(nor.numpy()[t], ref_nor, atol=1.0e-5, err_msg=f"eval_face_normal face {f}")
            np.testing.assert_allclose(vert.numpy()[t], POINTS[i], atol=1.0e-6, err_msg=f"mesh_get_point face {f}")
            self.assertEqual(int(idx.numpy()[t]), int(j), f"mesh_get_index face {f}")

    # -- ray -------------------------------------------------------------------

    def test_query_ray_struct(self):
        mesh = self._mesh()
        rays = [
            # start off the quad diagonal (y=x): (0.5,0.5) would tie faces 0/1
            ((0.3, 0.5, 2.0), (0.0, 0.0, -1.0)),  # quad face 1 (y > x side)
            ((3.6, 0.3, 2.0), (0.0, 0.0, -1.0)),  # tetra side face 5, then base
            ((0.3, 0.5, 2.0), (0.0, -0.2, -1.0)),  # angled quad hit -> (0.3,0.1,0) face 0
            ((5.0, 5.0, 5.0), (1.0, 0.0, 0.0)),  # miss
        ]
        n = len(rays)
        sa = wp.zeros(n, dtype=wp.vec3, device="metal")
        da = wp.zeros(n, dtype=wp.vec3, device="metal")
        for t, (s, d) in enumerate(rays):
            sa.numpy()[t] = s
            da.numpy()[t] = d
        hit = wp.zeros(n, dtype=int, device="metal")
        face = wp.zeros(n, dtype=int, device="metal")
        tt = wp.zeros(n, dtype=float, device="metal")
        nor = wp.zeros(n, dtype=wp.vec3, device="metal")
        wp.launch(query_ray_struct_kernel, n, inputs=[mesh.id, sa, da, 100.0, hit, face, tt, nor], device="metal")
        for t, (s, d) in enumerate(rays):
            ref = _closest_ray_hit(s, d, 100.0)
            got_h = int(hit.numpy()[t])
            if ref is None:
                self.assertEqual(got_h, 0, f"ray {t}: expected a miss")
                continue
            rt, rf, ru, rv, rsign = ref
            self.assertEqual(got_h, 1, f"ray {t}: expected a hit")
            self.assertEqual(int(face.numpy()[t]), rf, f"ray {t}: face")
            self.assertAlmostEqual(tt.numpy()[t], rt, places=3, msg=f"ray {t}: t")
            # upstream returns the UNORIENTED geometric normal
            # normalize(cross(ab, ac)); the hit side is `sign` (the woop
            # determinant), never folded into the normal
            tri_nor = np.cross(POINTS[INDICES[rf * 3 + 1]] - POINTS[INDICES[rf * 3]], POINTS[INDICES[rf * 3 + 2]] - POINTS[INDICES[rf * 3]])
            tri_nor = tri_nor / np.linalg.norm(tri_nor)
            np.testing.assert_allclose(nor.numpy()[t], tri_nor, atol=1.0e-4, err_msg=f"ray {t}: normal")

    def test_query_ray_ref(self):
        mesh = self._mesh()
        rays = [
            ((0.3, 0.5, 2.0), (0.0, 0.0, -1.0)),  # quad face 1 (off the diagonal)
            ((3.6, 0.3, 2.0), (0.0, 0.0, -1.0)),  # tetra side face 5
            ((2.0, 0.5, 2.0), (0.0, 0.0, -1.0)),  # miss (gap between quad and tetra)
        ]
        n = len(rays)
        sa = wp.zeros(n, dtype=wp.vec3, device="metal")
        da = wp.zeros(n, dtype=wp.vec3, device="metal")
        for t, (s, d) in enumerate(rays):
            sa.numpy()[t] = s
            da.numpy()[t] = d
        hit = wp.zeros(n, dtype=int, device="metal")
        face = wp.zeros(n, dtype=int, device="metal")
        tt = wp.zeros(n, dtype=float, device="metal")
        sign = wp.zeros(n, dtype=float, device="metal")
        wp.launch(query_ray_ref_kernel, n, inputs=[mesh.id, sa, da, 100.0, hit, face, tt, sign], device="metal")
        for t, (s, d) in enumerate(rays):
            ref = _closest_ray_hit(s, d, 100.0)
            got_h = int(hit.numpy()[t])
            if ref is None:
                self.assertEqual(got_h, 0, f"ray {t}: expected a miss")
                continue
            rt, rf, _ru, _rv, _rs = ref
            self.assertEqual(got_h, 1, f"ray {t}: expected a hit")
            self.assertEqual(int(face.numpy()[t]), rf)
            self.assertAlmostEqual(tt.numpy()[t], rt, places=3)
            # upstream returns the raw determinant (det = U+V+W) as sign
            self.assertGreater(abs(float(sign.numpy()[t])), 0.0)

    # -- aabb / sphere iteration ------------------------------------------------

    def _face_mask(self, lo, up):
        mask = np.zeros(N_TRI, dtype=int)
        for f in range(N_TRI):
            if np.all(FACE_LOWER[f] <= up) and np.all(FACE_UPPER[f] >= lo):
                mask[f] = 1
        return mask

    def test_query_aabb(self):
        mesh = self._mesh()
        boxes = [
            ((-0.1, -0.1, -0.1), (1.1, 1.1, 0.1)),  # both quad faces
            ((0.2, 0.2, -1.0), (0.6, 0.6, -0.5)),  # nothing
            ((2.5, -0.5, -0.5), (4.5, 1.5, 1.5)),  # tetra faces
            ((-9.0, -9.0, -9.0), (9.0, 9.0, 9.0)),  # everything
        ]
        n = len(boxes)
        lo = wp.zeros(n, dtype=wp.vec3, device="metal")
        up = wp.zeros(n, dtype=wp.vec3, device="metal")
        for t, (a, b) in enumerate(boxes):
            lo.numpy()[t] = a
            up.numpy()[t] = b
        mask = wp.zeros(n * N_TRI, dtype=int, device="metal")
        wp.launch(query_aabb_kernel, n, inputs=[mesh.id, lo, up, mask], device="metal")
        got = mask.numpy().reshape(n, N_TRI)
        for t, (a, b) in enumerate(boxes):
            np.testing.assert_array_equal(got[t], self._face_mask(np.asarray(a), np.asarray(b)), err_msg=f"box {t}")

    def test_query_sphere(self):
        mesh = self._mesh()
        centers = [
            ((0.5, 0.5, 0.05), 0.5),  # quad faces
            ((3.5, 0.4, 0.5), 1.2),  # tetra faces
            ((9.0, 9.0, 9.0), 0.5),  # nothing
        ]
        n = len(centers)
        ca = wp.zeros(n, dtype=wp.vec3, device="metal")
        ra = wp.zeros(n, dtype=float, device="metal")
        for t, (c, r) in enumerate(centers):
            ca.numpy()[t] = c
            ra.numpy()[t] = r
        mask = wp.zeros(n * N_TRI, dtype=int, device="metal")
        wp.launch(query_sphere_kernel, n, inputs=[mesh.id, ca, ra, mask], device="metal")
        got = mask.numpy().reshape(n, N_TRI)
        for t, (c, radius) in enumerate(centers):
            ref = np.zeros(N_TRI, dtype=int)
            c = np.asarray(c, dtype=np.float64)
            for f in range(N_TRI):
                d = np.maximum(np.maximum(FACE_LOWER[f] - c, c - FACE_UPPER[f]), 0.0)
                if float(np.dot(d, d)) <= radius * radius:
                    ref[f] = 1
            np.testing.assert_array_equal(got[t], ref, err_msg=f"sphere {t}")


if __name__ == "__main__":
    unittest.main()
