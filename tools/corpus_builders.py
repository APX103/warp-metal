# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Corpus input builders (KEP-0001 M1.7(b)) — pure numpy, no warp import.

Imported by BOTH conformance sides (golden: Newton venv with real warp;
compare: warp-metal venv with the import shim), so every builder must be
deterministic (seeded rng) and free of warp/ctypes dependencies.

Each builder returns ``(args, skips)`` where ``args`` is a list of
``(name, dtype_key, value)`` triples in kernel-signature order:

- ``dtype_key`` "arr:<elem>" for arrays (``value`` a numpy array whose
  last axis packs the element; e.g. transform = (n, 7), vec3 = (n, 3))
- ``dtype_key`` "scalar:<elem>" for scalars (``value`` a python number)
- ``dtype_key`` "value:<elem>" for value-typed params (transform/vec3)

and ``skips`` a list of reasons the kernel cannot run yet (fail-loud
capability boundaries: mesh/BVH Phase 2, func_native GJK, printf).
Element keys map to warp dtypes in both venvs via conform.py.
"""

from __future__ import annotations

import numpy as np


def _quat(rng, n):
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def _xforms(rng, n):
    return np.concatenate([rng.standard_normal((n, 3)), _quat(rng, n)], axis=1).astype(np.float32)


def _pdofs(rng, n):
    p = rng.standard_normal((n, 3)).astype(np.float32)
    d = rng.standard_normal((n, 3)).astype(np.float32)
    return p, d


def _eye_inertia(rng, n):
    return (np.eye(3)[None] * rng.uniform(0.1, 1.0, (n, 1, 1))).reshape(n, 9).astype(np.float32)


def b_integrate_bodies(rng):
    """solver.integrate_bodies (all-arrays variant)."""
    n = 7
    p, d = _pdofs(rng, n)
    mass = rng.uniform(0.5, 5.0, n).astype(np.float32)
    inertia = _eye_inertia(rng, n)
    flags = np.zeros(n, dtype=np.int32)
    flags[2] = 1  # kinematic early-return branch
    return (
        [
            ("body_q", "arr:transform", _xforms(rng, n)),
            ("body_qd", "arr:spatial_vector", np.concatenate([p, d], axis=1).astype(np.float32)),
            ("body_f", "arr:spatial_vector", rng.standard_normal((n, 6)).astype(np.float32)),
            ("body_com", "arr:vec3", rng.standard_normal((n, 3)).astype(np.float32)),
            ("m", "arr:f32", mass),
            ("I", "arr:mat33", inertia),
            ("inv_m", "arr:f32", (1.0 / mass).astype(np.float32)),
            ("inv_I", "arr:mat33", (np.eye(3)[None] / rng.uniform(0.1, 1.0, (n, 1, 1))).reshape(n, 9).astype(np.float32)),
            ("body_flags", "arr:i32", flags),
            ("body_world", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("gravity", "arr:vec3", np.tile(np.float32([0.0, -9.81, 0.0]), (1, 1))),
            ("angular_damping", "scalar:f32", 0.05),
            ("dt", "scalar:f32", 1.0 / 60.0),
            ("body_q_new", "out:transform", n),
            ("body_qd_new", "out:spatial_vector", n),
        ],
        [],
    )


def b_update_effective_inv_mass_inertia(rng):
    n = 6
    flags = np.zeros(n, dtype=np.int32)
    flags[1] = flags[4] = 1
    return (
        [
            ("body_flags", "arr:i32", flags),
            ("model_inv_mass", "arr:f32", rng.uniform(0.1, 2.0, n).astype(np.float32)),
            ("model_inv_inertia", "arr:mat33", _eye_inertia(rng, n)),
            ("eff_inv_mass", "out:f32", n),
            ("eff_inv_inertia", "out:mat33", n),
        ],
        [],
    )


def b_compute_body_inverses_kernel(rng):
    """newton._src.geometry.inertia.compute_body_inverses_kernel (KEP-0001
    Phase 4 F1 flip: the stale "uses float64" skip predates B2 — fork commits
    557239bb/3da7b947 downgraded the inertia path to float32, so the kernel
    translates and runs on the shim).

    Diagonal eye-scaled inertias keep wp.inverse well-conditioned; the
    kernel's only outputs are the two inverse arrays (body_mass is read-only
    here, so it stays an ``arr:`` input)."""
    n = 5
    mass = rng.uniform(0.5, 5.0, n).astype(np.float32)
    inertia = _eye_inertia(rng, n)
    return (
        [
            ("body_mass", "arr:f32", mass),
            ("body_inertia", "arr:mat33", inertia),
            ("body_inv_mass", "out:f32", n),
            ("body_inv_inertia", "out:mat33", n),
        ],
        [],
    )


def b_validate_and_correct_inertia_kernel(rng):
    """newton._src.geometry.inertia.validate_and_correct_inertia_kernel
    (KEP-0001 Phase 4 F1 flip: same stale-fp64-skip history as
    compute_body_inverses_kernel — fp32 since B2).

    body_mass / body_inertia are IN-PLACE outputs (the kernel writes the
    bounded mass and the corrected tensor back), so they use the ``outinit:``
    key: deterministic initial content AND registered for comparison — the
    corrected inertia is the row's real output, not just the atomic counter.
    ``__dim__`` is pinned explicitly because outinit values carry fixture
    data, not launch extents (same reason estimate_world_extents pins it).
    Branch coverage: every body trips the bound_mass branch (0.5..5.0 <
    1000), body 1 carries the 1e6 inertia scale, body 3 a negative diagonal
    (eigenvalue-threshold + bound-inertia corrections); the balance-inertia
    branch stays off (diagonal fixtures satisfy I1+I2 >= I3). All eigen
    decisions sit far from their thresholds (no knife-edge branches) and the
    kernel discards the eigenvectors, so only the few-ulp eigenvalue spread
    reaches the compared outputs — inside corpus rtol."""
    n = 5
    inertia = _eye_inertia(rng, n)
    inertia[1] *= 1e6  # large-scale body: bound/threshold branches stay off
    inertia[3, 0] = -0.5  # negative diagonal → correction branch
    return (
        [
            ("__dim__", "", n),
            ("body_mass", "outinit:f32", rng.uniform(0.5, 5.0, n).astype(np.float32)),
            ("body_inertia", "outinit:mat33", inertia),
            ("balance_inertia", "scalar:bool", True),
            ("bound_mass", "scalar:f32", 1000.0),
            ("bound_inertia", "scalar:f32", 1000.0),
            ("relative_tolerance", "scalar:f32", 0.01),
            ("absolute_tolerance", "scalar:f32", 1e-6),
            ("correction", "scalar:f32", 1.0),
            ("correction_count", "out:i32", 1),
        ],
        [],
    )


def b_compute_enabled_shapes(rng):
    n = 8
    return (
        [
            ("shape_type", "arr:i32", rng.integers(0, 5, n).astype(np.int32)),
            ("shape_flags", "arr:i32", rng.integers(0, 16, n).astype(np.int32)),
            ("shape_flags_mask", "scalar:i32", 7),
            ("shape_opacity", "arr:f32", rng.uniform(0.0, 1.0, n).astype(np.float32)),
            ("out_shape_enabled", "out:u32", n),
            ("out_shape_enabled_count", "out:i32", 1),
        ],
        [],
    )


def b_compute_shape_world_transforms(rng):
    n = 5
    return (
        [
            ("in_body_transforms", "arr:transform", _xforms(rng, n)),
            ("in_shape_body", "arr:i32", np.arange(n, dtype=np.int32)),
            ("in_shape_transform", "arr:transform", _xforms(rng, n)),
            ("out_transforms", "out:transform", n),
        ],
        [],
    )


def b_compute_bvh_group_roots(rng):
    """compute_bvh_group_roots over a grouped BVH (KEP-0001 M2.1: device
    query family; the "bvh" key builds wp.Bvh per conformance phase)."""
    # two spatially separated groups of two boxes each: dense group ids
    # 0..1 (the kernel indexes the output by group id)
    lowers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float32
    )
    uppers = lowers + 0.5
    groups = np.array([0, 0, 1, 1], dtype=np.int32)
    return (
        [
            ("bvh", "bvh", {"lowers": lowers, "uppers": uppers, "groups": groups}),
            ("out_bvh_group_roots", "out:i32", 2),
            ("__dim__", "", 2),
        ],
        [],
    )


def b_vertex_triangle_collision_detection_kernel(rng):
    """geometry/kernels.vertex_triangle_collision_detection_kernel (style3d
    cloth collision path) — real Newton closure kernel exercising the BVH
    whole-tree AABB query (all-particle world -1 → query_all branch).

    Scene: three well-separated triangles; T0/T2 sit 0.3 apart in z so
    their vertices collide within max_query_radius=2.0, T1 is isolated.
    Per-vertex and per-triangle CSR buffers are sized so no resize flag
    fires; per-triangle collision buffers are in-place inputs (their slot
    ORDER is atomic-schedule dependent) while the compared outputs are the
    order-independent counts / min distances / resize flags.
    """
    tris = np.array(
        [
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (10.0, 1.0, 0.0)],
            [(0.3, 0.3, 0.3), (1.3, 0.3, 0.3), (0.3, 1.3, 0.3)],
        ],
        dtype=np.float32,
    )
    n_tri = 3
    n_vert = 3 * n_tri
    pos = tris.reshape(n_vert, 3)
    tri_indices = np.array([[t * 3, t * 3 + 1, t * 3 + 2] for t in range(n_tri)], dtype=np.int32)
    # triangle AABBs for the query BVH (single-group tree: all particles are
    # in the global world -1, so the kernel takes the query_all branch and
    # bvh_group_roots stays unread)
    lowers = tris.min(axis=1)
    uppers = tris.max(axis=1)

    k = 8  # per-vertex / per-triangle collision buffer capacity
    v_offsets = (np.arange(n_vert + 1, dtype=np.int32) * k).astype(np.int32)
    t_offsets = (np.arange(n_tri + 1, dtype=np.int32) * k).astype(np.int32)
    return (
        [
            ("max_query_radius", "scalar:f32", 2.0),
            ("min_query_radius", "scalar:f32", 0.0),
            ("bvh", "bvh", {"lowers": lowers, "uppers": uppers, "groups": None}),
            ("bvh_group_roots", "arr:i32", np.zeros(2, dtype=np.int32)),
            ("pos", "arr:vec3", pos),
            ("tri_indices", "arr2:i32", tri_indices),
            ("particle_world", "arr:i32", np.full(n_vert, -1, dtype=np.int32)),
            ("world_count", "scalar:i32", 1),
            ("vertex_colliding_triangles_offsets", "arr:i32", v_offsets),
            ("vertex_colliding_triangles_buffer_sizes", "arr:i32", np.full(n_vert, k, dtype=np.int32)),
            ("triangle_colliding_vertices_offsets", "arr:i32", t_offsets),
            ("triangle_colliding_vertices_buffer_sizes", "arr:i32", np.full(n_tri, k, dtype=np.int32)),
            ("vertex_triangle_filtering_list", "arr:i32", np.zeros(1, dtype=np.int32)),
            ("vertex_triangle_filtering_list_offsets", "arr:i32", np.zeros(n_vert + 1, dtype=np.int32)),
            ("min_distance_filtering_ref_pos", "arr:vec3", pos.copy()),
            ("vertex_colliding_triangles", "arr:i32", np.zeros(2 * k * n_vert, dtype=np.int32)),
            ("vertex_colliding_triangles_count", "out:i32", n_vert),
            ("vertex_colliding_triangles_min_dist", "out:f32", n_vert),
            ("triangle_colliding_vertices", "arr:i32", np.zeros(k * n_tri, dtype=np.int32)),
            ("triangle_colliding_vertices_count", "out:i32", n_tri),
            ("triangle_colliding_vertices_min_dist", "out:f32", n_tri),
            ("resize_flags", "out:i32", 3),
            ("__dim__", "", n_vert),
        ],
        [],
    )


def b_update_shape_xforms(rng):
    n_shapes, n_bodies = 5, 3
    world_off = np.tile(np.float32([1.0, 2.0, 3.0]), (2, 1))
    lx = np.float32([0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 1.0])
    return (
        [
            ("shape_xforms", "arr:transform", _xforms(rng, n_shapes)),
            ("shape_parents", "arr:i32", np.arange(n_shapes, dtype=np.int32) % n_bodies),
            ("body_q", "arr:transform", _xforms(rng, n_bodies)),
            ("shape_worlds", "arr:i32", np.zeros(n_shapes, dtype=np.int32)),
            ("world_offsets", "arr:vec3", world_off),
            ("layer_xform", "value:transform", lx),
            ("world_xforms", "out:transform", n_shapes),
        ],
        [],
    )


def b_repack_shape_colors(rng):
    n = 6
    slot_to_shape = np.arange(n, dtype=np.int32)
    return (
        [
            ("shape_colors", "arr:vec3", rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)),
            ("slot_to_shape", "arr:i32", slot_to_shape),
            ("packed_shape_colors", "out:vec3", n),
        ],
        [],
    )


def b_repack_shape_opacities(rng):
    n = 6
    return (
        [
            ("shape_opacities", "arr:f32", rng.uniform(0.0, 1.0, n).astype(np.float32)),
            ("slot_to_shape", "arr:i32", np.arange(n, dtype=np.int32)),
            ("packed_shape_opacities", "out:f32", n),
        ],
        [],
    )


def b_estimate_world_extents(rng):
    n = 6
    return (
        [
            # one thread per shape; outputs are (world_count, 3) and must not
            # drive the launch dim (out-shaped dim silently ran 3 threads)
            ("__dim__", "", n),
            ("shape_transform", "arr:transform", _xforms(rng, n)),
            ("shape_body", "arr:i32", np.arange(n, dtype=np.int32)),
            ("shape_collision_radius", "arr:f32", rng.uniform(0.1, 1.0, n).astype(np.float32)),
            ("shape_world", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("body_q", "arr:transform", _xforms(rng, n)),
            ("world_count", "scalar:i32", 1),
            ("world_bounds_min", "out2:f32", (1, 3)),
            ("world_bounds_max", "out2:f32", (1, 3)),
        ],
        [],
    )


def b_compute_com_positions(rng):
    n = 5
    lx = np.float32([0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 1.0])
    return (
        [
            ("body_q", "arr:transform", _xforms(rng, n)),
            ("body_com", "arr:vec3", rng.standard_normal((n, 3)).astype(np.float32)),
            ("body_world", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("world_offsets", "arr:vec3", np.tile(np.float32([1.0, 2.0, 3.0]), (1, 1))),
            ("layer_xform", "value:transform", lx),
            ("visible_worlds_mask", "arr:i32", np.ones(1, dtype=np.int32)),
            ("com_positions", "out:vec3", n),
        ],
        [],
    )


def b_compute_shape_aabbs(rng):
    n = 5
    return (
        [
            ("body_q", "arr:transform", _xforms(rng, n)),
            ("shape_transform", "arr:transform", _xforms(rng, n)),
            ("shape_body", "arr:i32", np.arange(n, dtype=np.int32)),
            ("shape_type", "arr:i32", np.full(n, 4, dtype=np.int32)),  # CAPSULE-like
            ("shape_scale", "arr:vec3", np.tile(np.float32([1.0, 1.0, 1.0]), (n, 1))),
            ("shape_collision_radius", "arr:f32", rng.uniform(0.1, 0.5, n).astype(np.float32)),
            ("shape_source_ptr", "arr:u64", np.zeros(n, dtype=np.uint64)),
            ("shape_margin", "arr:f32", np.full(n, 0.01, dtype=np.float32)),
            ("shape_gap", "arr:f32", np.zeros(n, dtype=np.float32)),
            ("shape_collision_aabb_lower", "arr:vec3", np.zeros((n, 3), dtype=np.float32)),
            ("shape_collision_aabb_upper", "arr:vec3", np.zeros((n, 3), dtype=np.float32)),
            ("shape_flags", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("shape_variant_row", "arr:bool", np.zeros(n, dtype=bool)),
            ("contact_counters", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("contact_generation", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("broad_phase_pair_count", "arr:i32", np.zeros(1, dtype=np.int32)),
            ("num_contact_counters", "scalar:i32", 0),
            ("aabb_lower", "out:vec3", n),
            ("aabb_upper", "out:vec3", n),
            ("geom_data", "io:vec4", rng.standard_normal((n, 4)).astype(np.float32)),
            ("geom_xform", "io:transform", _xforms(rng, n)),
        ],
        # M2.2: the SSA's wp::Mesh (collision_core.compute_tight_aabb_from_support,
        # reached through the support-path @wp.func) now translates against the
        # real wp_mesh struct + per-launch mesh table. The mesh branch stays
        # runtime-dead here (CONVEX_MESH shapes take the has_local_aabb branch
        # before the support path, so shape_data.shape_type can never be
        # CONVEX_MESH there) — the fixture needs no wp.Mesh, and the mesh
        # table binds empty.
        [],
    )


def b_nxn_broadphase_precomputed_pairs(rng):
    # n=3 → dim 3 (one thread per precomputed pair); larger grids SIGBUS under
    # warp's per-thread CPU stacks when the machine is under memory pressure.
    # Pairs must stay within [0, n): all i<j combos.
    n = 3
    lo = rng.uniform(-2.0, -0.2, (n, 3)).astype(np.float32)
    hi = lo + rng.uniform(0.4, 1.0, (n, 3)).astype(np.float32)
    pairs = np.array([[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    return (
        [
            ("shape_bounding_box_lower", "arr:vec3", lo),
            ("shape_bounding_box_upper", "arr:vec3", hi),
            ("shape_gap", "arr:f32", rng.uniform(0.0, 0.1, n).astype(np.float32)),
            ("shape_displacement", "arr:vec3", rng.standard_normal((n, 3)).astype(np.float32)),
            ("nxn_shape_pair", "arr:vec2i", pairs),
            ("shape_body", "arr:i32", np.arange(n, dtype=np.int32)),
            ("body_flags", "arr:i32", np.zeros(n, dtype=np.int32)),
            ("include_static_kinematic_pairs", "scalar:bool", True),
            ("candidate_pair", "out:vec2i", len(pairs)),
            ("candidate_pair_count", "out:i32", 1),
            ("max_candidate_pair", "scalar:i32", len(pairs)),
        ],
        [],
    )


def b_compute_mesh_plane_vert_counts(rng):
    """compute_mesh_plane_vert_counts over real wp.Mesh entries (KEP-0001
    M2.2: the "mesh" key builds wp.Mesh per conformance phase; shape_source
    carries the ids via the mesh_ids packing). Mesh A is the warp-doc cube
    (8 verts / 12 tris), mesh B a single triangle (3 verts / 1 tri); pairs
    reference [A, B, null, A] so vert_counts = [8, 3, 0, 8] exercises the
    live mesh read, the mesh_get on a second entry, and the id==0 guard."""
    cube_points = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    cube_indices = np.array(
        [0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        dtype=np.int32,
    )
    tri_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    tri_indices = np.array([0, 1, 2], dtype=np.int32)

    n = 4
    return (
        [
            ("mesh_a", "mesh", {"name": "mesh_a", "points": cube_points, "indices": cube_indices}),
            ("mesh_b", "mesh", {"name": "mesh_b", "points": tri_points, "indices": tri_indices}),
            (
                "shape_pairs_mesh_plane",
                "arr:vec2i",
                np.stack(
                    [np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32)],
                    axis=1,
                ),
            ),
            ("shape_pairs_mesh_plane_count", "arr:i32", np.full(1, n, dtype=np.int32)),
            ("shape_source", "arr:u64", {"mesh_ids": ["mesh_a", "mesh_b", None, "mesh_a"]}),
            ("vert_counts", "out:i32", n),
            ("total_vert_count", "out:i32", 1),
            ("total_num_threads", "scalar:i32", n),
        ],
        [],
    )


def b_verify_narrow_phase_buffers(rng):
    n = 4
    ones = np.full(1, 3, dtype=np.int32)
    args: list = []
    for base, lim in (
        ("broad_phase", "max_broad_phase"), ("split_query", "max_split_query"),
        ("gjk", "max_gjk"), ("split_gjk", "max_split_gjk"),
        ("split_manifold", "max_split_manifold"), ("mesh", "max_mesh"),
        ("triangle", "max_triangle"), ("mesh_plane", "max_mesh_plane"),
        ("mesh_mesh", "max_mesh_mesh"), ("sdf_sdf", "max_sdf_sdf"),
        ("contact", "max_contacts"),
    ):
        args.append((f"{base}_count", "arr:i32", ones))
        args.append((lim, "scalar:i32", 8))
    args += [
        ("reduction_ht_active_slots", "arr:i32", ones),
        ("reduction_ht_capacity", "scalar:i32", 16),
        ("reduction_ht_insert_failures", "arr:i32", np.zeros(1, dtype=np.int32)),
        ("reduction_buffer_overflows", "arr:i32", np.zeros(1, dtype=np.int32)),
        ("reduction_ht_warn_load_percent", "scalar:i32", 90),
    ]
    # outputs: verify writes warnings; count outputs unknown → golden drives
    return (args, [])


def b_copy_kinematic_body_state_kernel(rng):
    n = 4
    flags = np.zeros(n, dtype=np.int32)
    flags[1] = flags[3] = 1
    return (
        [
            ("body_flags", "arr:i32", flags),
            ("body_q_in", "arr:transform", _xforms(rng, n)),
            ("body_qd_in", "arr:spatial_vector", rng.standard_normal((n, 6)).astype(np.float32)),
            ("body_q_out", "out:transform", n),
            ("body_qd_out", "out:spatial_vector", n),
        ],
        [],
    )


def b_apply_body_deltas(rng):
    n = 4
    return (
        [
            ("q_in", "arr:transform", _xforms(rng, n)),
            ("qd_in", "arr:spatial_vector", rng.standard_normal((n, 6)).astype(np.float32)),
            ("body_com", "arr:vec3", rng.standard_normal((n, 3)).astype(np.float32)),
            ("body_I", "arr:mat33", _eye_inertia(rng, n)),
            ("body_inv_m", "arr:f32", rng.uniform(0.2, 2.0, n).astype(np.float32)),
            ("body_inv_I", "arr:mat33", _eye_inertia(rng, n)),
            ("deltas", "arr:spatial_vector", rng.standard_normal((n, 6)).astype(np.float32)),
            ("constraint_inv_weights", "arr:f32", rng.uniform(0.0, 1.0, n).astype(np.float32)),
            ("dt", "scalar:f32", 1.0 / 60.0),
            ("q_out", "out:transform", n),
            ("qd_out", "out:spatial_vector", n),
        ],
        [],
    )


def b_apply_joint_forces(rng):
    # kernel is joint-parallel: one thread per joint (the first output is
    # body_f over bodies, so dim must be pinned to n_joints explicitly)
    n_bodies, n_joints, n_dofs = 3, 2, 12
    return (
        [
            ("__dim__", "", n_joints),
            ("body_q", "arr:transform", _xforms(rng, n_bodies)),
            ("body_com", "arr:vec3", rng.standard_normal((n_bodies, 3)).astype(np.float32)),
            ("joint_type", "arr:i32", np.zeros(n_joints, dtype=np.int32)),  # JOINT_FREE
            ("joint_enabled", "arr:bool", np.ones(n_joints, dtype=bool)),
            ("joint_parent", "arr:i32", np.array([-1, 0], dtype=np.int32)),
            ("joint_child", "arr:i32", np.array([0, 1], dtype=np.int32)),
            ("joint_X_p", "arr:transform", _xforms(rng, n_joints)),
            ("joint_X_c", "arr:transform", _xforms(rng, n_joints)),
            ("joint_qd_start", "arr:i32", np.array([0, 6], dtype=np.int32)),
            ("joint_dof_dim", "arr2:i32", np.full((n_joints, 2), 2, dtype=np.int32)),
            ("joint_axis", "arr:vec3", np.tile(np.float32([1.0, 0.0, 0.0]), (n_dofs, 1))),
            ("joint_f", "arr:f32", rng.standard_normal(n_dofs).astype(np.float32)),
            ("dt", "scalar:f32", 1.0 / 60.0),
            ("body_f", "out:spatial_vector", n_bodies),
            ("joint_impulse", "out:spatial_vector", n_dofs),
        ],
        [],
    )


def b_solve_body_joints(rng):
    # same dof-tiling constraint as b_apply_joint_forces: FREE joints read 6
    # dofs per joint; qd_start must not run the reads past n_dofs. Joint-parallel
    # kernel: dim must be n_joints, not the first output's body count.
    n_bodies, n_joints, n_dofs = 3, 2, 12
    lo = np.full(n_dofs, -1.0, dtype=np.float32)
    hi = np.full(n_dofs, 1.0, dtype=np.float32)
    return (
        [
            ("__dim__", "", n_joints),
            ("body_q", "arr:transform", _xforms(rng, n_bodies)),
            ("body_qd", "arr:spatial_vector", rng.standard_normal((n_bodies, 6)).astype(np.float32)),
            ("body_com", "arr:vec3", rng.standard_normal((n_bodies, 3)).astype(np.float32)),
            ("body_inv_m", "arr:f32", rng.uniform(0.2, 2.0, n_bodies).astype(np.float32)),
            ("body_inv_I", "arr:mat33", _eye_inertia(rng, n_bodies)),
            ("joint_type", "arr:i32", np.zeros(n_joints, dtype=np.int32)),
            ("joint_enabled", "arr:bool", np.ones(n_joints, dtype=bool)),
            ("joint_parent", "arr:i32", np.array([-1, 0], dtype=np.int32)),
            ("joint_child", "arr:i32", np.array([0, 1], dtype=np.int32)),
            ("joint_X_p", "arr:transform", _xforms(rng, n_joints)),
            ("joint_X_c", "arr:transform", _xforms(rng, n_joints)),
            ("joint_limit_lower", "arr:f32", lo),
            ("joint_limit_upper", "arr:f32", hi),
            ("joint_qd_start", "arr:i32", np.array([0, 6], dtype=np.int32)),
            ("joint_target_q_start", "arr:i32", np.array([0, 6], dtype=np.int32)),
            ("joint_dof_dim", "arr2:i32", np.full((n_joints, 2), 2, dtype=np.int32)),
            ("joint_axis", "arr:vec3", np.tile(np.float32([1.0, 0.0, 0.0]), (n_dofs, 1))),
            ("joint_target_q", "arr:f32", np.zeros(n_dofs, dtype=np.float32)),
            ("joint_target_qd", "arr:f32", np.zeros(n_dofs, dtype=np.float32)),
            ("joint_target_ke", "arr:f32", np.full(n_dofs, 10.0, dtype=np.float32)),
            ("joint_target_kd", "arr:f32", np.full(n_dofs, 1.0, dtype=np.float32)),
            ("joint_linear_compliance", "scalar:f32", 0.0),
            ("joint_angular_compliance", "scalar:f32", 0.0),
            ("angular_relaxation", "scalar:f32", 0.7),
            ("linear_relaxation", "scalar:f32", 0.7),
            ("dt", "scalar:f32", 1.0 / 60.0),
            ("deltas", "out:spatial_vector", n_bodies),
            ("joint_impulse", "out:spatial_vector", n_dofs),
        ],
        [],
    )


def b_solve_body_contact_positions(rng):
    n_bodies, n_contacts = 3, 2
    return (
        [
            ("body_q", "arr:transform", _xforms(rng, n_bodies)),
            ("body_qd", "arr:spatial_vector", rng.standard_normal((n_bodies, 6)).astype(np.float32)),
            ("body_flags", "arr:i32", np.zeros(n_bodies, dtype=np.int32)),
            ("body_com", "arr:vec3", rng.standard_normal((n_bodies, 3)).astype(np.float32)),
            ("body_m_inv", "arr:f32", rng.uniform(0.2, 2.0, n_bodies).astype(np.float32)),
            ("body_I_inv", "arr:mat33", _eye_inertia(rng, n_bodies)),
            ("shape_body", "arr:i32", np.arange(4, dtype=np.int32) % n_bodies),
            ("contact_count", "arr:i32", np.full(1, n_contacts, dtype=np.int32)),
            ("contact_point0", "arr:vec3", rng.standard_normal((n_contacts, 3)).astype(np.float32)),
            ("contact_point1", "arr:vec3", rng.standard_normal((n_contacts, 3)).astype(np.float32)),
            ("contact_surface_velocity", "arr:vec3", np.zeros((n_contacts, 3), dtype=np.float32)),
            ("contact_offset0", "arr:vec3", rng.standard_normal((n_contacts, 3)).astype(np.float32)),
            ("contact_offset1", "arr:vec3", rng.standard_normal((n_contacts, 3)).astype(np.float32)),
            ("contact_normal", "arr:vec3", np.tile(np.float32([0.0, -1.0, 0.0]), (n_contacts, 1))),
            ("contact_thickness0", "arr:f32", np.full(n_contacts, 0.01, dtype=np.float32)),
            ("contact_thickness1", "arr:f32", np.full(n_contacts, 0.01, dtype=np.float32)),
            ("contact_shape0", "arr:i32", np.zeros(n_contacts, dtype=np.int32)),
            ("contact_shape1", "arr:i32", np.ones(n_contacts, dtype=np.int32)),
            ("shape_material_mu", "arr:f32", np.full(4, 0.5, dtype=np.float32)),
            ("shape_material_mu_torsional", "arr:f32", np.full(4, 0.01, dtype=np.float32)),
            ("shape_material_mu_rolling", "arr:f32", np.full(4, 0.01, dtype=np.float32)),
            ("relaxation", "scalar:f32", 1.0),
            ("dt", "scalar:f32", 1.0 / 60.0),
            ("deltas", "out:spatial_vector", n_bodies),
            ("contact_inv_weight", "out:f32", n_contacts),
            ("contact_impulse", "out:spatial_vector", n_contacts),
        ],
        [],
    )


def b_clear_active_kernel(rng):
    cap = 8
    n = cap  # one thread per hash-table slot
    zeros_i = np.zeros(n, dtype=np.int32)
    zeros_v = np.zeros((n, 3), dtype=np.float32)
    return (
        [
            ("ht_keys", "arr:u64", np.zeros(n, dtype=np.uint64)),
            ("ht_values", "arr:u64", np.zeros(n, dtype=np.uint64)),
            ("ht_active_slots", "out:i32", n),
            ("agg_force", "out:vec3", n),
            ("agg_depth_volume", "out:vec3", n),
            ("weighted_pos_sum", "out:vec3", n),
            ("weight_sum", "out:f32", n),
            ("total_depth_reduced", "out:f32", n),
            ("total_normal_reduced", "out:vec3", n),
            ("agg_moment_unreduced", "out:f32", n),
            ("agg_moment_reduced", "out:f32", n),
            ("agg_moment2_reduced", "out:f32", n),
            ("contact_count", "out:i32", 1),
            ("reclaimed_contact_bits", "out:u32", n),
            ("reclaimed_contact_count", "out:i32", 1),
            ("ht_insert_failures", "out:i32", 1),
            ("buffer_overflows", "out:i32", 1),
            ("ht_capacity", "scalar:i32", cap),
            ("values_per_key", "scalar:i32", 4),
            ("entry_major", "scalar:i32", 0),
            ("num_threads", "scalar:i32", n),
        ],
        [],
    )


def b_zero_active_count_kernel(rng):
    cap = 8
    return (
        [
            ("ht_active_slots", "out:i32", cap),
            ("ht_capacity", "scalar:i32", cap),
        ],
        [],
    )


def b_record_reduction_overflow(rng):
    return (
        [
            ("insert_failures", "arr:i32", np.full(1, 3, dtype=np.int32)),
            ("buffer_overflows", "arr:i32", np.full(1, 0, dtype=np.int32)),
            ("overflow", "out:i32", 1),
        ],
        [],
    )


def b_compute_block_counts_from_weights(rng):
    n = 6
    return (
        [
            ("total_weight_arr", "arr:i32", np.full(1, 10, dtype=np.int32)),
            ("weights", "arr:i32", rng.integers(1, 4, n).astype(np.int32)),
            ("pair_count_arr", "arr:i32", np.full(1, n, dtype=np.int32)),
            ("max_pairs", "scalar:i32", 64),
            ("target_blocks", "scalar:i32", 4),
            ("block_counts", "out:i32", 8),
            ("total_num_threads", "scalar:i32", n),
        ],
        [],
    )


def _encode_oct(n):
    """numpy port of contact_reduction_global.encode_oct (float32 throughout)."""
    n = np.asarray(n, dtype=np.float32)
    l1 = np.float32(np.abs(n[0]) + np.abs(n[1]) + np.abs(n[2]))
    if l1 < np.float32(1.0e-20):
        return np.zeros(2, dtype=np.float32)
    ox, oy, oz = np.float32(n[0] / l1), np.float32(n[1] / l1), np.float32(n[2] / l1)
    if oz < 0.0:
        sign_x = np.float32(-1.0) if ox < 0.0 else np.float32(1.0)
        sign_y = np.float32(-1.0) if oy < 0.0 else np.float32(1.0)
        new_x = np.float32((1.0 - abs(oy)) * sign_x)
        new_y = np.float32((1.0 - abs(ox)) * sign_y)
        ox, oy = new_x, new_y
    return np.array([ox, oy], dtype=np.float32)


# contact_reduction.py _ICOSAHEDEDRON face normals (NORMAL_BINNING_POLYHEDRON
# = "icosahedron", NUM_NORMAL_BINS = 20). Fixture normals snap to exact face
# normals so get_slot() resolves each contact to a distinct, construction-time
# known bin — no CAS contention on hashtable entries.
_ICOSA_NORMALS = np.array([
    0.80103071, 0.29640806, -0.52008851,
    0.29640808, 0.80103070, -0.52008851,
    -0.29640806, 0.52008851, -0.80103071,
    -0.15816595, -0.15816596, -0.97466254,
    0.52008851, -0.29640808, -0.80103070,
    0.80103070, -0.52008851, 0.29640808,
    0.52008851, -0.80103071, -0.29640806,
    0.57735027, 0.57735027, 0.57735027,
    0.97466254, 0.15816595, 0.15816596,
    -0.52008851, 0.80103071, 0.29640806,
    0.15816596, 0.97466254, 0.15816595,
    -0.97466254, -0.15816595, -0.15816596,
    -0.80103070, 0.52008851, -0.29640808,
    -0.15816596, -0.97466254, -0.15816595,
    -0.57735027, -0.57735027, -0.57735027,
    0.15816595, 0.15816596, 0.97466254,
    -0.52008851, 0.29640808, 0.80103070,
    -0.80103071, -0.29640806, 0.52008851,
    -0.29640808, -0.80103070, 0.52008851,
    0.29640806, -0.52008851, 0.80103071,
], dtype=np.float32).reshape(20, 3)


def b_reduce_buffered_contacts_kernel(rng):
    """reduce_buffered_contacts_kernel: 4 buffered contacts on one pair.

    Contacts use distinct icosahedron face normals (distinct normal-bin
    entries) and distinct voxel positions; dim=1 with total_num_threads=1 so
    the grid-stride loop is sequential on both sides — atomic ticket order
    and hashtable CAS outcomes are deterministic without contention.
    Contact 3 carries a large positive depth (use_beta False branch).
    Buffer index 0 is reserved (the kernel reads contacts 1..N).
    """
    cap = 16  # ht capacity (power of two)
    n = 4     # buffered contacts, ids 1..4
    slots = 7  # NUM_SPATIAL_DIRECTIONS + 1

    pd = np.zeros((n + 1, 4), dtype=np.float32)
    pd[1] = [0.5, 0.5, 0.5, -0.02]
    pd[2] = [1.5, -0.5, 2.0, -0.5]
    pd[3] = [-2.0, 1.0, 0.0, 0.5]   # separating: use_beta False
    pd[4] = [0.0, -2.5, -1.5, -0.01]

    normals = np.zeros((n + 1, 2), dtype=np.float32)
    for i, face in enumerate((0, 7, 14, 19)):
        normals[i + 1] = _encode_oct(_ICOSA_NORMALS[face])

    pairs = np.zeros((n + 1, 2), dtype=np.int32)
    pairs[1:] = (0, 1)

    reducer = {
        "position_depth": ("arr:vec4", pd),
        "normal": ("arr:vec2", normals),
        "shape_pairs": ("arr:vec2i", pairs),
        "contact_count": ("arr:i32", np.full(1, n, dtype=np.int32)),
        "exported_flags": ("arr:i32", np.zeros(n + 1, dtype=np.int32)),
        "reclaimed_contact_bits": ("arr:u32", np.zeros(n + 1, dtype=np.uint32)),
        "reclaimed_contact_count": ("arr:i32", np.zeros(1, dtype=np.int32)),
        "capacity": ("scalar:i32", n + 1),
        "contact_max": ("scalar:i32", 64),
        "shape_gap": ("arr:f32", np.zeros(n + 1, dtype=np.float32)),
        "contact_fingerprints": ("arr:i32", np.array([0, 101, 202, 303, 404], dtype=np.int32)),
        "contact_area": ("arr:f32", np.zeros(n + 1, dtype=np.float32)),
        "contact_pressure": ("arr:f32", np.zeros(n + 1, dtype=np.float32)),
        "contact_nbin_entry": ("arr:i32", np.zeros(n + 1, dtype=np.int32)),
        "agg_force": ("arr:vec3", np.zeros((cap, 3), dtype=np.float32)),
        "agg_depth_volume": ("arr:vec3", np.zeros((cap, 3), dtype=np.float32)),
        "weighted_pos_sum": ("arr:vec3", np.zeros((cap, 3), dtype=np.float32)),
        "weight_sum": ("arr:f32", np.zeros(cap, dtype=np.float32)),
        "total_depth_reduced": ("arr:f32", np.zeros(cap, dtype=np.float32)),
        "total_normal_reduced": ("arr:vec3", np.zeros((cap, 3), dtype=np.float32)),
        "ht_keys": ("out:u64", cap),
        "ht_values": ("out:u64", slots * cap),
        "ht_active_slots": ("out:i32", cap + 1),
        "ht_insert_failures": ("out:i32", 1),
        "buffer_overflows": ("arr:i32", np.zeros(1, dtype=np.int32)),
        "ht_capacity": ("scalar:i32", cap),
        "ht_values_per_key": ("scalar:i32", slots),
        "deterministic": ("scalar:i32", 1),
    }

    # shape 0 (mesh): identity transform, unit-cube-ish grid AABB;
    # shape 1 (convex): translated so voxel positions exercise transform_point
    xforms = np.zeros((2, 7), dtype=np.float32)
    xforms[0] = [0, 0, 0, 0, 0, 0, 1]
    xforms[1] = [1, 2, 3, 0, 0, 0, 1]
    aabb_lo = np.array([[-5.0, -5.0, -5.0], [-3.0, -3.0, -3.0]], dtype=np.float32)
    aabb_hi = np.array([[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    voxel_res = np.array([[4, 4, 4], [4, 4, 4]], dtype=np.int32)

    return (
        [
            ("reducer_data", "struct:GlobalContactReducerData", reducer),
            ("shape_transform", "arr:transform", xforms),
            ("shape_collision_aabb_lower", "arr:vec3", aabb_lo),
            ("shape_collision_aabb_upper", "arr:vec3", aabb_hi),
            ("shape_voxel_resolution", "arr:vec3i", voxel_res),
            ("total_num_threads", "scalar:i32", 1),
            ("__dim__", "scalar:i32", 1),
        ],
        [],
    )


def b_compute_particle_bounds(rng):
    """particle_surface_kernels.compute_particle_bounds (KEP-0001 M2.4: the
    tile_min/tile_max/tile_sum closure kernel).

    Newton's own non-CUDA launch is dim=(n_blocks, tile_size) with
    block_dim=tile_size=1 (particle_surface.py ``tile_size = ... if
    device.is_cuda else 1``) — block-partition-invariant outputs (commutative
    atomics only), so both conformance sides agree; ``__block_dim__`` pins
    that routing on both sides. Fixture: 24 particles, two exclusions —
    particle 7 carries no ACTIVE flag (use_flags=1 exercises _is_active) and
    particle 5 carries a +inf y (exercises _is_finite_position) — plus the
    ±1e30 bound sentinels via ``outinit:`` so the atomic_min/max start where
    Newton's reset_bounds leaves them. lower/upper are single-vec3 arrays
    (the kernels write slot 0); active_count is the grid-count array with
    the valid-particle total landing at index _GRID_ACTIVE_COUNT=2.
    """
    n = 24
    pos = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
    pos[5, 1] = np.float32("inf")  # non-finite position: excluded from bounds
    flags = np.ones(n, dtype=np.int32)  # ParticleFlags.ACTIVE set
    flags[7] = 0  # inactive: excluded from bounds and the valid count
    return (
        [
            ("positions", "arr:vec3", pos),
            ("flags", "arr:i32", flags),
            ("use_flags", "scalar:i32", 1),
            ("lower", "outinit:vec3", np.full((1, 3), 1.0e30, dtype=np.float32)),
            ("upper", "outinit:vec3", np.full((1, 3), -1.0e30, dtype=np.float32)),
            ("active_count", "outinit:i32", np.zeros(8, dtype=np.int32)),
            ("__dim__", "", (n, 1)),
            ("__block_dim__", "", 1),
        ],
        [],
    )


def b_narrow_phase_primitive_kernel(rng):
    """narrow_phase_primitive_kernel (write_contact_simple, non-speculative).

    Plane/sphere/sphere/capsule/box scene, 4 candidate pairs: one analytical
    contact (plane-sphere), one filtered far pair (sphere-sphere), two
    plane-capsule contacts, four plane-box contacts (mat4x3f path). dim=1 /
    total_num_threads=1 keeps atomic ticket order deterministic.
    """
    import math

    ns, npairs, cap = 5, 4, 64

    def _q_yaw_quarter():
        return [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]

    types = np.array([1, 3, 3, 4, 7], dtype=np.int32)  # PLANE SPHERE SPHERE CAPSULE BOX
    data = np.array([
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
        [0.1, 0, 0, 0],
        [0.05, 0.2, 0, 0],
        [0.1, 0.1, 0.1, 0],
    ], dtype=np.float32)
    xf = np.zeros((ns, 7), dtype=np.float32)
    xf[0] = [0, 0, 0, 0, 0, 0, 1]
    xf[1] = [0, 0, 0.05, 0, 0, 0, 1]
    xf[2] = [0, 0, 5.0, 0, 0, 0, 1]
    xf[3] = [0, 0, 0.03, *_q_yaw_quarter()]
    xf[4] = [0, 0, 0.05, 0, 0, 0, 1]

    writer = {
        "contact_max": ("scalar:i32", cap),
        "shape_index_bits": ("scalar:i32", 32),
        "sub_key_bits": ("scalar:i32", 16),
        "contact_count": ("out:i32", 1),
        "contact_pair": ("out2:vec2i", (cap,)),
        "contact_position": ("out2:vec3", (cap,)),
        "contact_normal": ("out2:vec3", (cap,)),
        "contact_penetration": ("out:f32", cap),
        "contact_tangent": ("out2:vec3", (cap,)),
        "contact_sort_key": ("out:i64", (cap,)),
        "shape_gap": ("arr:f32", np.zeros(ns, dtype=np.float32)),
        "shape_transform": ("arr:transform", xf),
        "shape_linear_velocity": ("arr:vec3", np.zeros((ns, 3), dtype=np.float32)),
        "shape_angular_velocity": ("arr:vec3", np.zeros((ns, 3), dtype=np.float32)),
        "collision_update_dt": ("scalar:f32", np.float32(1.0 / 60.0)),
        "max_speculative_extension": ("scalar:f32", np.float32(0.0)),
    }

    cand = np.array([[0, 1], [1, 2], [0, 3], [0, 4]], dtype=np.int32)
    return (
        [
            ("candidate_pair", "arr:vec2i", cand),
            ("candidate_pair_count", "arr:i32", np.full(1, npairs, dtype=np.int32)),
            ("shape_types", "arr:i32", types),
            ("shape_data", "arr:vec4", data),
            ("shape_transform", "arr:transform", xf),
            ("shape_linear_velocity", "arr:vec3", np.zeros((ns, 3), dtype=np.float32)),
            ("shape_angular_velocity", "arr:vec3", np.zeros((ns, 3), dtype=np.float32)),
            ("collision_update_dt", "scalar:f32", 1.0 / 60.0),
            ("max_speculative_extension", "scalar:f32", 0.0),
            ("shape_source", "arr:u64", np.zeros(ns, dtype=np.uint64)),
            ("shape_gap", "arr:f32", np.zeros(ns, dtype=np.float32)),
            ("shape_flags", "arr:i32", np.zeros(ns, dtype=np.int32)),
            ("shape_sdf_index", "arr:i32", np.full(ns, -1, dtype=np.int32)),
            ("shape_edge_range", "arr:vec2i", np.zeros((ns, 2), dtype=np.int32)),
            ("shape_unit_scale_sdf", "arr:i32", np.zeros(ns, dtype=np.int32)),
            ("writer_data", "struct:ContactWriterData", writer),
            ("total_num_threads", "scalar:i32", 1),
            ("gjk_candidate_pairs", "arr:vec2i", np.zeros((npairs, 2), dtype=np.int32)),
            ("gjk_candidate_pairs_count", "out:i32", 1),
            ("shape_pairs_mesh", "arr:vec2i", np.zeros((npairs, 2), dtype=np.int32)),
            ("shape_pairs_mesh_count", "out:i32", 1),
            ("shape_pairs_mesh_plane", "arr:vec2i", np.zeros((npairs, 2), dtype=np.int32)),
            ("shape_pairs_mesh_plane_cumsum", "arr:i32", np.zeros(npairs, dtype=np.int32)),
            ("shape_pairs_mesh_plane_count", "out:i32", 1),
            ("mesh_plane_vertex_total_count", "out:i32", 1),
            ("shape_pairs_mesh_mesh", "arr:vec2i", np.zeros((npairs, 2), dtype=np.int32)),
            ("shape_pairs_mesh_mesh_count", "out:i32", 1),
            ("shape_pairs_sdf_sdf", "arr:vec2i", np.zeros((npairs, 2), dtype=np.int32)),
            ("shape_pairs_sdf_sdf_count", "out:i32", 1),
            ("unit_scale_sdf_rescaled", "out:i32", ns),
            ("__dim__", "scalar:i32", 1),
        ],
        [],
    )


# ---------------------------------------------------------------------------
# KEP-0001 M2.4 P3+P4: steel solver tile factories (solver_steel.py). Each
# factory bakes one articulation per launch block over fixed-dof tile shapes,
# so the fixtures pin n_dofs=12 (3 articulations x 4 dofs) and route through
# ``__block_dim__`` (conform.py launches both sides with wp.launch_tiled:
# upstream CPU resolves bd=1 — the serial scalar-fallback discipline every
# golden is generated with; warp-metal compiles the matching threadgroup
# lowering). SPD inputs come from seeded numpy so cholesky's sqrt chain never
# touches a negative; masked articulations exercise the early return with the
# untouched outputs pre-zeroed identically on both sides.
# ---------------------------------------------------------------------------

_STEEL_ARTS = 3
_STEEL_DOFS = 12
_STEEL_JOINTS = 12  # 1-dof joints: joint j belongs to dof j within its art


def _spd_group(rng, arts, n):
    """(arts, n, n) SPD float32: A@A.T + 3I per articulation."""
    out = np.empty((arts, n, n), dtype=np.float32)
    for g in range(arts):
        a = rng.standard_normal((n, n)).astype(np.float32)
        out[g] = a @ a.T + np.eye(n, dtype=np.float32) * 3.0
    return out.astype(np.float32)


def _lower_group(rng, arts, n):
    """(arts, n, n) lower-triangular Cholesky factors of _spd_group."""
    return np.linalg.cholesky(_spd_group(rng, arts, n)).astype(np.float32)


def _steel_layout(rng):
    """(group_to_art, mass_update_mask) for the 3-articulation fixtures:
    identity art map, articulation 1 masked out (early-return branch)."""
    return (
        np.arange(_STEEL_ARTS, dtype=np.int32),
        np.array([1, 0, 1], dtype=np.int32),
    )


def b_steel_cholesky_kernel(rng):
    """steel solver_steel._get_cholesky_kernel(12, arch, 1): tile Cholesky of
    H + diag(armature) per articulation (M2.4 P4)."""
    g2a, mask = _steel_layout(rng)
    return (
        [
            ("H_group", "arr3:f32", _spd_group(rng, _STEEL_ARTS, _STEEL_DOFS)),
            ("R_group", "arr2:f32", rng.uniform(0.1, 1.0, (_STEEL_ARTS, _STEEL_DOFS)).astype(np.float32)),
            ("group_to_art", "arr:i32", g2a),
            ("mass_update_mask", "arr:i32", mask),
            ("L_group", "out3:f32", (_STEEL_ARTS, _STEEL_DOFS, _STEEL_DOFS)),
            ("__dim__", "", _STEEL_ARTS),
            ("__block_dim__", "", 1),
        ],
        [],
    )


def b_steel_triangular_solve_kernel(rng):
    """steel solver_steel._get_triangular_solve_kernel(12, arch, 1): tiled
    forward/backward substitution L L^T x = tau (M2.4 P4)."""
    g2a, _mask = _steel_layout(rng)
    return (
        [
            ("L_group", "arr3:f32", _lower_group(rng, _STEEL_ARTS, _STEEL_DOFS)),
            ("tau_group", "arr3:f32", rng.standard_normal((_STEEL_ARTS, _STEEL_DOFS, 1)).astype(np.float32)),
            ("group_to_art", "arr:i32", g2a),
            ("articulation_active", "arr:i32", np.array([1, 1, 1], dtype=np.int32)),
            ("qdd_group", "out3:f32", (_STEEL_ARTS, _STEEL_DOFS, 1)),
            ("__dim__", "", _STEEL_ARTS),
            ("__block_dim__", "", 1),
        ],
        [],
    )


def _steel_hinv_args(rng, diag_group):
    """Shared fixture for the two H^-1 J^T factories: lower factors, a fully
    initialized Jacobian (3 constraint chunks x 12 dofs, 5 active rows), and
    the world bookkeeping the kernel reads. max_constraints=8 with
    constraint_chunk_size=3 pins BOUNDS_CHECK=True (8 % 3 != 0); launch dim
    (arts, 2 chunks) drives ``idx, chunk = wp.tid()``; rows >= 5 stay at the
    pre-zeroed initial value on both sides."""
    g2a, _mask = _steel_layout(rng)
    args = [
        ("L_group", "arr3:f32", _lower_group(rng, _STEEL_ARTS, _STEEL_DOFS)),
        ("J_group", "arr3:f32", rng.standard_normal((_STEEL_ARTS, 8, _STEEL_DOFS)).astype(np.float32)),
        ("group_to_art", "arr:i32", g2a),
        ("art_to_world", "arr:i32", np.arange(_STEEL_ARTS, dtype=np.int32)),
        ("articulation_world_dof_offset", "arr:i32", np.array([0, 12, 24], dtype=np.int32)),
        ("world_constraint_count", "arr:i32", np.full(_STEEL_ARTS, 5, dtype=np.int32)),
        ("Y_group", "out3:f32", (_STEEL_ARTS, 8, _STEEL_DOFS)),
        # WRITE_WORLD=0 folds the world stores out; binding real outputs
        # verifies no stray writes land
        ("J_world", "out3:f32", (_STEEL_ARTS, 8, 36)),
        ("Y_world", "out3:f32", (_STEEL_ARTS, 8, 36)),
    ]
    if diag_group is not None:
        args.append(("diag_group", "out2:f32", (_STEEL_ARTS, 8)))
    args += [("__dim__", "", (_STEEL_ARTS, 2)), ("__block_dim__", "", 1)]
    return args


def b_steel_hinv_jt_plain_kernel(rng):
    """steel solver_steel._get_hinv_jt_plain_kernel(12, 8, arch, 1, 3):
    tile_transpose aliases + tile_lower_solve/tile_upper_solve over a (3,12)
    constraint chunk (M2.4 P3+P4)."""
    return (_steel_hinv_args(rng, None), [])


def b_steel_hinv_jt_kernel(rng):
    """steel solver_steel._get_hinv_jt_kernel(12, 8, arch, 1, 3,
    compute_diag=False): the full variant — same solve chain plus the
    (compiled-out at this fill_mode) diag output binding (M2.4 P3+P4)."""
    return (_steel_hinv_args(rng, "diag"), [])


def b_steel_crba_cholesky_kernel(rng):
    """steel solver_steel._get_crba_cholesky_kernel(12, arch, tile_threads=1):
    composite-RBA mass matrix staged via tile_scatter_masked (scalar +
    spatial_vector forms) + tile_extract + tile_diag_add + tile_cholesky
    (M2.4 P3+P4).

    Three 12-dof articulations (36 1-dof joints/bodies) — every kernel index
    domain (dof, joint, body, drive-K) is fully in-bounds, so the row is a
    deterministic test on both sides.  (The original layout staged only 4
    dofs per articulation against the kernel's 12-dof domain, leaving
    joint_S_s / drive_dof_K reads past their arrays: the CPU golden read heap
    slack, Metal read zero-filled buffer slack, and the row silently compared
    against uninitialized memory — root cause of the 2026-11-15 L_group
    review regression, whose signature was diagonal pivots collapsing to
    sqrt(armature) = sqrt(0.5).)

    The fixed source map keeps H strictly diagonally dominant so the factor
    never sqrt's a negative: S[d] = amp_d (e_{d%6} + 0.05 e_{(d+1)%6}) with
    amp_d = 0.8**d (per-art local dof), body inertias c_j I6, source_dof
    diagonal = self and subdiagonal = the dof two steps back (a weak cross
    term); the augmented drive adds positive K on the even dofs only."""
    arts, dofs = _STEEL_ARTS, _STEEL_DOFS
    joints = arts * dofs  # 1-dof joints: joint j belongs to dof j % dofs within its art
    amp = 0.8 ** np.arange(dofs, dtype=np.float32)
    s_vec = np.zeros((joints, 6), dtype=np.float32)
    for a in range(arts):
        for d in range(dofs):
            s_vec[a * dofs + d, d % 6] = amp[d]
            s_vec[a * dofs + d, (d + 1) % 6] = 0.05 * amp[d]
    body_i = np.zeros((joints + 4, 36), dtype=np.float32)
    for j in range(joints + 4):
        body_i[j] = (np.eye(6, dtype=np.float32) * (1.0 + 0.05 * j)).reshape(36)
    source_dof = np.full((dofs, dofs), -1, dtype=np.int32)
    for r in range(dofs):
        source_dof[r, r] = r
        if r >= 2:
            source_dof[r, r - 1] = r - 2
        if r + 1 < dofs:
            source_dof[r, r + 1] = r
    drive_k = np.tile(
        np.array([0.3 if d % 2 == 0 else 0.0 for d in range(dofs)], dtype=np.float32), arts
    )
    g2a, mask = _steel_layout(rng)
    return (
        [
            ("articulation_start", "arr:i32", np.arange(arts, dtype=np.int32) * dofs),
            ("articulation_dof_start", "arr:i32", np.arange(arts, dtype=np.int32) * dofs),
            ("mass_update_mask", "arr:i32", mask),
            ("joint_child", "arr:i32", np.arange(1, joints + 1, dtype=np.int32)),
            ("joint_S_s", "arr:spatial_vector", s_vec),
            ("body_I_c", "arr:spatial_matrix", body_i),
            ("group_to_art", "arr:i32", g2a),
            ("R_group", "arr2:f32", np.full((arts, dofs), 0.5, dtype=np.float32)),
            ("dof_joint_offset", "arr:i32", np.tile(np.arange(dofs, dtype=np.int32), arts)),
            ("source_dof", "arr2:i32", source_dof),
            ("fused_augmented_drive", "scalar:i32", 1),
            ("drive_dof_K", "arr:f32", drive_k),
            ("L_group", "out3:f32", (arts, dofs, dofs)),
            ("__dim__", "", arts),
            ("__block_dim__", "", 1),
        ],
        [],
    )


def b_steel_cholesky_inverse_tiled_kernel(rng):
    """steel solver_steel._get_cholesky_inverse_tiled_kernel(12, arch, 1):
    factor + tile_cholesky_solve against a shared identity tile — the
    materialized inverse (M2.4 P4)."""
    g2a, mask = _steel_layout(rng)
    return (
        [
            ("H_group", "arr3:f32", _spd_group(rng, _STEEL_ARTS, _STEEL_DOFS)),
            ("R_group", "arr2:f32", rng.uniform(0.1, 1.0, (_STEEL_ARTS, _STEEL_DOFS)).astype(np.float32)),
            ("identity", "arr2:f32", np.eye(_STEEL_DOFS, dtype=np.float32)),
            ("group_to_art", "arr:i32", g2a),
            ("mass_update_mask", "arr:i32", mask),
            ("L_group", "out3:f32", (_STEEL_ARTS, _STEEL_DOFS, _STEEL_DOFS)),
            ("Hinv_group", "out3:f32", (_STEEL_ARTS, _STEEL_DOFS, _STEEL_DOFS)),
            ("__dim__", "", _STEEL_ARTS),
            ("__block_dim__", "", 1),
        ],
        [],
    )


# ---------------------------------------------------------------------------
# KEP-0001 M3.2 P1: mujoco_warp 3.17.0 kernels whose closures use the builtin
# families wp.identity / wp.diag / wp.matrix / wp.determinant (M3.1 matrix
# §4 P1 rows). Two shared mini-scenes:
#
# _rod_layout/_rod_verts — one 1-D flex rod hinge spanning two 3-dof bodies
# (Bergou et al. rod bending): 4 flex vertices with the hinge triple in an
# off-axis knee (rod_hinge's denom = |e0||e1| + e0·e1 ~ 0.38 > MJ_MINVAL, so
# the ok gate passes and kb/A0/A1 are nonzero). The hinge MUST span bodies —
# a rod hinge is invariant under rigid motion of any single body, so a
# one-body hinge makes every dof force exactly zero and the golden degrades
# to fp32 rounding residue. Vert 1 is on body 1 (dofs 0..2), verts 2-3 on
# body 2 (dofs 3..5), each body's dof chain walked by flex_gather/scatter_vert
# through dof_parentid [-1,0,1 | -1,3,4]. flex_bending[adr..adr+3] =
# [1, 2, 3, 5.0] is the vertex-offset triple + hinge stiffness k.
#
# _ellipsoid_fixture — one ellipsoid fluid body: body 1 carries geom 1
# (GeomType.ELLIPSOID=4, semiaxes = size directly), all 12 geom_fluid
# coefficients strictly positive so the per-geom loop body always runs;
# the 3-dof tree (dof_bodyid=1, body_isdofancestor=1) keeps every early
# return gate open. M_elemid[i,j] = 3i+j makes madr = M_elemid[1,0] = 4 an
# in-bounds row of the (nworld, 9) qDeriv buffer for the single (Mi,Mj)=(1,0)
# element, so each (world, madr) slot has exactly one atomic writer.
#
# Every array is in-bounds over the kernel's full index domain (the crba
# fixture rule): dims are (2 worlds, 1 hinge/fluid body, 1 element).
# ---------------------------------------------------------------------------


def _rod_layout():
    """Model-side rod arrays shared by the rod-hinge kernel fixtures."""
    return {
        "body_rootid": np.array([0, 0], dtype=np.int32),
        "body_weldid": np.array([0, 1], dtype=np.int32),
        "body_dofnum": np.array([0, 3], dtype=np.int32),
        "body_dofadr": np.array([0, 0], dtype=np.int32),
        "body_simple": np.array([0, 0], dtype=np.int32),
        "dof_parentid": np.array([-1, 0, 1], dtype=np.int32),
        "flex_vertadr": np.array([0], dtype=np.int32),
        "flex_vertbodyid": np.array([0, 1, 1, 1], dtype=np.int32),
        "flex_bending": np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32),
        "flex_damping": np.array([0.5], dtype=np.float32),
        "flex_simple": np.array([True], dtype=bool),
        "flex_rodhinge_map": np.array([[0, 0]], dtype=np.int32),
    }


_ROD_WORLDS = 2
_ROD_VERTS = 4
_ROD_DOFS = 6  # two 3-dof bodies: dofs 0..2 on body 1, 3..5 on body 2
_ROD_BODIES = 3


def _rod_layout():
    """Model-side rod arrays shared by the rod-hinge kernel fixtures.

    The hinge spans TWO bodies (vert 1 on body 1, verts 2-3 on body 2): a
    rod hinge is invariant under any rigid motion of a single body, so an
    all-verts-on-one-body fixture makes every dof force an exactly-zero
    quantity and the golden degrades to fp32 rounding residue (the
    configuration the first draft had — caught because every output sat at
    2^-17-scale epsilon). Two 3-dof bodies chained by dof_parentid
    [-1,0,1 | -1,3,4] give the J'WJ products a live relative-motion channel."""
    return {
        "body_rootid": np.array([0, 0, 0], dtype=np.int32),
        "body_weldid": np.array([0, 1, 2], dtype=np.int32),
        "body_dofnum": np.array([0, 3, 3], dtype=np.int32),
        "body_dofadr": np.array([0, 0, 3], dtype=np.int32),
        "body_simple": np.array([0, 0, 0], dtype=np.int32),
        "dof_parentid": np.array([-1, 0, 1, -1, 3, 4], dtype=np.int32),
        "flex_vertadr": np.array([0], dtype=np.int32),
        "flex_vertbodyid": np.array([0, 1, 2, 2], dtype=np.int32),
        "flex_bending": np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32),
        "flex_damping": np.array([0.5], dtype=np.float32),
        "flex_simple": np.array([True], dtype=bool),
        "flex_rodhinge_map": np.array([[0, 0]], dtype=np.int32),
    }


def _rod_verts():
    """(nworld, 4, 3) flex-vertex positions: fixed asymmetric knee spanning
    the two bodies (vert 1 on body 1, verts 2-3 on body 2).

    rod_hinge needs denom = |e0||e1| + e0·e1 > MJ_MINVAL (here ~0.38); the
    off-axis skew keeps the hinge torque nonzero so outputs sit well above
    fp32 rounding residue."""
    verts = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.2], [1.2, 0.9, 0.1], [0.3, 1.1, 0.7]],
        dtype=np.float32,
    )
    return np.tile(verts[None], (_ROD_WORLDS, 1, 1))


def _rod_data_fields(rng):
    """Data-side rod fields: subtree_com, cdof (nworld, 6 dofs, 6), xpos."""
    return [
        ("subtree_com_in", "arr2:vec3", (rng.standard_normal((_ROD_WORLDS, _ROD_BODIES, 3)) * 0.1).astype(np.float32)),
        ("cdof_in", "arr2:spatial_vector", (rng.standard_normal((_ROD_WORLDS, _ROD_DOFS, 6)) * 0.1).astype(np.float32)),
        ("flexvert_xpos_in", "arr2:vec3", _rod_verts()),
    ]


def b_flex_rod_mul(rng):
    """mujoco_warp passive._flex_rod_mul(check_skip=False, nv=0): rod hinge
    J' W J multiply — the closure reaches wp.identity(3) through
    rod_hinge_weights (W = 2 k s1 I + sd Wd, M3.1 P1 identity row).

    check_skip=False keeps the wp.static skip gate compiled out (the bool
    array is still bound); use_timestep=True reads opt_timestep per world;
    s1_in=1.0 (>0 → s1 = t^2) with disableflags 0 keeps both the spring and
    damper terms live (t=0.4 so s1*k ~ O(1), not fp32 residue). res_out
    accumulates via flex_scatter_vert atomics — one hinge means one writer
    per dof slot."""
    rod = _rod_layout()
    args = [
        ("opt_timestep", "arr:f32", np.full(_ROD_WORLDS, 0.4, dtype=np.float32)),
        ("opt_disableflags", "scalar:i32", 0),
    ]
    for name in (
        "body_rootid", "body_weldid", "body_dofnum", "body_dofadr", "body_simple", "dof_parentid",
        "flex_vertadr", "flex_vertbodyid", "flex_bending", "flex_damping", "flex_simple",
        "flex_rodhinge_map",
    ):
        if name in ("flex_bending", "flex_damping"):
            key = "arr:f32"
        elif name == "flex_simple":
            key = "arr:bool"
        elif name == "flex_rodhinge_map":
            key = "arr:vec2i"
        else:
            key = "arr:i32"
        args.append((name, key, rod[name]))
    args += _rod_data_fields(rng)
    args += [
        ("vec_in", "arr2:f32", (rng.standard_normal((_ROD_WORLDS, _ROD_DOFS)) * 0.1).astype(np.float32)),
        ("skip", "arr:bool", np.zeros(_ROD_WORLDS, dtype=bool)),
        ("s1_in", "scalar:f32", 1.0),
        ("s2_in", "scalar:f32", 1.0),
        ("use_timestep", "scalar:bool", True),
        ("non_simple_only", "scalar:bool", False),
        ("res_out", "out2:f32", (_ROD_WORLDS, _ROD_DOFS)),
        ("__dim__", "", (_ROD_WORLDS, 1)),
    ]
    return (args, [])


def b_flex_rod_bending(rng):
    """mujoco_warp passive._flex_rod_bending(nv=0): rod hinge passive spring
    and damper forces — wp.identity(3) closure row (rod_hinge_weights), the
    qfrc_spring/qfrc_damper outputs both scatter-enabled (dsbl flags False)."""
    rod = _rod_layout()
    args = []
    for name in (
        "body_rootid", "body_weldid", "body_dofnum", "body_dofadr", "body_simple", "dof_parentid",
        "flex_vertadr", "flex_vertbodyid", "flex_bending", "flex_damping", "flex_rodhinge_map",
    ):
        if name in ("flex_bending", "flex_damping"):
            key = "arr:f32"
        elif name == "flex_rodhinge_map":
            key = "arr:vec2i"
        else:
            key = "arr:i32"
        args.append((name, key, rod[name]))
    args += [
        ("qvel_in", "arr2:f32", (rng.standard_normal((_ROD_WORLDS, _ROD_DOFS)) * 0.1).astype(np.float32)),
    ]
    args += _rod_data_fields(rng)
    args += [
        ("dsbl_spring", "scalar:bool", False),
        ("dsbl_damper", "scalar:bool", False),
        ("qfrc_spring_out", "out2:f32", (_ROD_WORLDS, _ROD_DOFS)),
        ("qfrc_damper_out", "out2:f32", (_ROD_WORLDS, _ROD_DOFS)),
        ("__dim__", "", (_ROD_WORLDS, 1)),
    ]
    return (args, [])


def b_eff_flex_rod_stiff(rng):
    """mujoco_warp derivative._eff_flex_rod_stiff(): rod hinge stiffness into
    the EFM K CSR — wp.identity(3) closure row (rod_hinge_weights), plus
    xmat frame rotations around it. Dense 6x6 CSR over the two 3-dof hinge
    bodies' dofs 0..5 (rownnz 6, sorted colind [0..5] per row —
    _find_csr_col binary-searches it) so every (dof_i, dof_j) block finds
    its column and idx + c stays in-bounds for the (nworld, 36) value
    buffer; timestep 0.4 puts s1*k at O(1). Both worlds share the layout."""
    rod = _rod_layout()
    args = [
        ("opt_timestep", "arr:f32", np.full(_ROD_WORLDS, 0.4, dtype=np.float32)),
        ("opt_disableflags", "scalar:i32", 0),
        ("body_weldid", "arr:i32", rod["body_weldid"]),
        ("body_dofnum", "arr:i32", rod["body_dofnum"]),
        ("body_dofadr", "arr:i32", rod["body_dofadr"]),
        ("flex_vertadr", "arr:i32", rod["flex_vertadr"]),
        ("flex_vertbodyid", "arr:i32", rod["flex_vertbodyid"]),
        ("flex_bending", "arr:f32", rod["flex_bending"]),
        ("flex_damping", "arr:f32", rod["flex_damping"]),
        ("efm_K_rownnz", "arr:i32", np.full(_ROD_DOFS, _ROD_DOFS, dtype=np.int32)),
        ("efm_K_rowadr", "arr:i32", np.arange(_ROD_DOFS, dtype=np.int32) * _ROD_DOFS),
        ("efm_K_colind", "arr:i32", np.tile(np.arange(_ROD_DOFS, dtype=np.int32), _ROD_DOFS)),
        ("flex_simple", "arr:bool", rod["flex_simple"]),
        ("flex_rodhinge_map", "arr:vec2i", rod["flex_rodhinge_map"]),
        ("xmat_in", "arr2:mat33", (rng.standard_normal((_ROD_WORLDS, _ROD_BODIES, 9)) * 0.1).astype(np.float32)),
        ("flexvert_xpos_in", "arr2:vec3", _rod_verts()),
        ("efm_K_val_out", "out2:f32", (_ROD_WORLDS, _ROD_DOFS * _ROD_DOFS)),
        ("__dim__", "", (_ROD_WORLDS, 1)),
    ]
    return (args, [])


def b_qderiv_ellipsoid_fluid(rng):
    """mujoco_warp derivative._qderiv_ellipsoid_fluid(): ellipsoid fluid
    qDeriv contribution — wp.diag(vec3) closure row (virtual-inertia diagonal
    in _geom_ellipsoid_fluid_B). One fluid body (body 1) with one ELLIPSOID
    geom (type 4), all 12 geom_fluid coefficients positive so the per-geom
    loop never `continue`s; IntegratorType.IMPLICITFAST=2 keeps the
    symmetrize branch on. Three (Mi,Mj) elements ((1,0),(2,0),(2,1) — Mi>=Mj
    tree order) hit three distinct madr slots of M_elemid[i,j]=3i+j in the
    (nworld, 9) buffer; water-scale density (1000) and timestep 0.4 put the
    accumulations at O(1)+ so the row is compare-meaningful rather than an
    fp32 residue; density/viscosity > 0 keep the early-return gates open."""
    args = [
        ("opt_timestep", "arr:f32", np.full(_ROD_WORLDS, 0.4, dtype=np.float32)),
        ("opt_wind", "arr:vec3", np.full((_ROD_WORLDS, 3), 0.1, dtype=np.float32)),
        ("opt_density", "arr:f32", np.full(_ROD_WORLDS, 1000.0, dtype=np.float32)),
        ("opt_viscosity", "arr:f32", np.full(_ROD_WORLDS, 1.0, dtype=np.float32)),
        ("opt_integrator", "scalar:i32", 2),
        ("body_parentid", "arr:i32", np.array([-1, 0], dtype=np.int32)),
        ("body_rootid", "arr:i32", np.array([0, 0], dtype=np.int32)),
        ("body_geomnum", "arr:i32", np.array([0, 1], dtype=np.int32)),
        ("body_geomadr", "arr:i32", np.array([0, 1], dtype=np.int32)),
        ("dof_bodyid", "arr:i32", np.full(3, 1, dtype=np.int32)),
        ("geom_type", "arr:i32", np.array([4, 4], dtype=np.int32)),
        ("geom_size", "arr2:vec3", np.tile(np.array([[0.2, 0.2, 0.2], [0.3, 0.2, 0.1]], dtype=np.float32)[None], (_ROD_WORLDS, 1, 1))),
        (
            "geom_fluid",
            "arr2:f32",
            np.array(
                [[1.0] * 12, [1.0, 0.5, 0.1, 0.2, 0.05, 0.01, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05]],
                dtype=np.float32,
            ),
        ),
        ("body_fluid_ellipsoid_adr", "arr:i32", np.array([1], dtype=np.int32)),
        ("body_isdofancestor", "arr2:i32", np.ones((2, 3), dtype=np.int32)),
        ("M_elemid", "arr2:i32", np.arange(9, dtype=np.int32).reshape(3, 3)),
        ("xipos_in", "arr2:vec3", (rng.standard_normal((_ROD_WORLDS, 2, 3)) * 0.1).astype(np.float32)),
        ("geom_xpos_in", "arr2:vec3", (rng.standard_normal((_ROD_WORLDS, 2, 3)) * 0.1).astype(np.float32)),
        ("geom_xmat_in", "arr2:mat33", (rng.standard_normal((_ROD_WORLDS, 2, 9)) * 0.1).astype(np.float32)),
        ("subtree_com_in", "arr2:vec3", (rng.standard_normal((_ROD_WORLDS, 2, 3)) * 0.1).astype(np.float32)),
        ("cdof_in", "arr2:spatial_vector", (rng.standard_normal((_ROD_WORLDS, 3, 6)) * 0.1).astype(np.float32)),
        ("cvel_in", "arr2:spatial_vector", (rng.standard_normal((_ROD_WORLDS, 2, 6)) * 0.1).astype(np.float32)),
        ("Mi", "arr:i32", np.array([1, 2, 2], dtype=np.int32)),
        ("Mj", "arr:i32", np.array([0, 0, 1], dtype=np.int32)),
        ("qDeriv_out", "out2:f32", (_ROD_WORLDS, 9)),
        ("__dim__", "", (_ROD_WORLDS, 1, 3)),
    ]
    return (args, [])


# ---------------------------------------------------------------------------
# KEP-0001 M3.2 package c: mujoco_warp kernels whose closures use the
# wp.add_inplace / wp.sub_inplace compound-element builtins (the codegen's
# `v[i] += x` / `m[i,j] -= x` lowering for LOCAL vec/mat variables) and/or a
# generic vecN dtype (vec10 cinert/crb). One shared linear-triangle flex
# element scene (dim=2 → stretch path; the snh cubic path stays
# runtime-unreachable but its closure still pins the mat6x6/vec10/vec6i
# translation surface), plus a single 6-dof free body for the implicit-mass
# multiply.
#
# Every array is in-bounds over the kernel's full index domain (the crba
# fixture rule); single atomic writer per output slot where the kernel
# scatters per body/edge.
# ---------------------------------------------------------------------------


_FLEX_WORLDS = 2


def _flex_elem_layout(rng):
    """Model+data arrays for one linear-triangle flex element (dim=2).

    3 verts / 3 edges (edges (0,1),(1,2),(2,0)), one flex, one element;
    flex_stiffness carries the 6 upper-triangle metric entries of
    stretch_metric at indices 0..5 of the 21-slot dim=2 table. Deformed
    vertex positions scale the rest triangle per-edge (1.15/0.95/1.05) so
    every elongation, tension and hessian block is nonzero at O(1).
    """
    verts0 = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.3]], dtype=np.float32
    )
    scale = np.array([1.15, 0.95, 1.05], dtype=np.float32)
    verts_in = np.tile(verts0[None] * scale[None, :, None], (_FLEX_WORLDS, 1, 1))
    e01, e12, e20 = (
        np.linalg.norm(verts0[1] - verts0[0]),
        np.linalg.norm(verts0[2] - verts0[1]),
        np.linalg.norm(verts0[0] - verts0[2]),
    )
    length0 = np.array([e01, e12, e20], dtype=np.float32)
    length_in = np.tile((length0 * scale)[None], (_FLEX_WORLDS, 1))
    stiffness = np.zeros(21, dtype=np.float32)
    # stretch_metric reads the upper triangle (e1<=e2): (0,0),(0,1),(0,2),
    # (1,1),(1,2),(2,2)
    stiffness[:6] = np.array([60.0, 12.0, 8.0, 55.0, 10.0, 50.0], dtype=np.float32)
    model = {
        "flex_dim": np.array([2], dtype=np.int32),
        "flex_interp": np.array([0], dtype=np.int32),
        "flex_vertadr": np.array([0], dtype=np.int32),
        "flex_edgeadr": np.array([0], dtype=np.int32),
        "flex_elemadr": np.array([0], dtype=np.int32),
        "flex_elemdataadr": np.array([0], dtype=np.int32),
        "flex_stiffnessadr": np.array([0], dtype=np.int32),
        "flex_elemedgeadr": np.array([0], dtype=np.int32),
        "flex_elemflexid": np.array([0], dtype=np.int32),
        "flex_edge": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        "flex_elem": np.array([0, 1, 2], dtype=np.int32),
        "flex_elemedge": np.array([0, 1, 2], dtype=np.int32),
        "flex_vert0": verts0,
        "flexedge_length0": length0,
        "flex_size": np.full((1, 3), 0.1, dtype=np.float32),
        "flex_stiffness": stiffness,
        "flex_rigid": np.array([False], dtype=bool),
    }
    data = {
        "flexvert_xpos_in": verts_in.astype(np.float32),
        "flexedge_length_in": length_in.astype(np.float32),
        "flex_hessian_valid_in": np.zeros((_FLEX_WORLDS, 1), dtype=bool),
    }
    return model, data


_FLEX_ELEM_KEYS = {
    "flex_dim": "arr:i32",
    "flex_interp": "arr:i32",
    "flex_vertadr": "arr:i32",
    "flex_edgeadr": "arr:i32",
    "flex_elemadr": "arr:i32",
    "flex_elemdataadr": "arr:i32",
    "flex_stiffnessadr": "arr:i32",
    "flex_elemedgeadr": "arr:i32",
    "flex_elemflexid": "arr:i32",
    "flex_vertbodyid": "arr:i32",
    "flex_edge": "arr:vec2i",
    "flex_elem": "arr:i32",
    "flex_elemedge": "arr:i32",
    "flex_vert0": "arr:vec3",
    "flexedge_length0": "arr:f32",
    "flex_size": "arr:vec3",
    "flex_stiffness": "arr:f32",
    "flex_damping": "arr:f32",
    "flex_rigid": "arr:bool",
}


def b_flex_hessian_elem(rng):
    """mujoco_warp passive._flex_hessian_elem(deterministic=False): per-edge
    stretch 3x3 hessian blocks — the closure carries wp.add_inplace through
    stretch_stiffness_block (`blk[r,c] += w * ...` on a local mat33) and
    _snh_cubic_metric (`metric[i,j] += value` on a local mat66) plus
    _snh_eigen3 (`B[0,0] -= mean`, sub_inplace mat33). dim=2 walks the
    stretch path at runtime (snh needs dim=3); one element, one writer per
    (world, edge) hessian slot."""
    model, data = _flex_elem_layout(rng)
    args = [
        (name, _FLEX_ELEM_KEYS[name], model[name])
        for name in (
            "flex_dim", "flex_interp", "flex_vertadr", "flex_edgeadr", "flex_elemadr",
            "flex_elemdataadr", "flex_stiffnessadr", "flex_elemedgeadr", "flex_edge",
            "flex_elem", "flex_elemedge", "flex_vert0", "flexedge_length0", "flex_size",
            "flex_stiffness", "flex_rigid", "flex_elemflexid",
        )
    ]
    args += [
        ("flexvert_xpos_in", "arr2:vec3", data["flexvert_xpos_in"]),
        ("flex_hessian_valid_in", "arr2:bool", data["flex_hessian_valid_in"]),
        ("flexedge_length_in", "arr2:f32", data["flexedge_length_in"]),
        ("flexedge_hessian_out", "out2:mat33", (_FLEX_WORLDS, 3)),
        ("__dim__", "", (_FLEX_WORLDS, 1)),
    ]
    return (args, [])


def b_flex_elasticity(rng):
    """mujoco_warp passive._flex_elasticity(deterministic=False): flex spring
    and damper body forces — wp.sub_inplace through the `force_spring[vi,x] -=
    s_val * g_val` compound assign on a local wp.matrix (6,3). Spring AND
    damper paths live (dsbl flags False, flex_damping 0.8 / timestep 0.4 →
    kD 2.0); three verts on three distinct bodies keep one atomic writer per
    (world, body) force slot.

    The xipos draw is de-conditioned (scale 0.45 + fixed per-body offset,
    grid-searched for worst-case cancellation ratio): each torque component
    is the difference of two O(|f|·|offset|) cross terms, and the seed-20261113
    draw at the original O(0.1) scale landed body-1 damper torque-y on a
    300x cancellation (r = |a-b|/(|a|+|b|) = 0.0033), amplifying the two
    fp32 evaluations' few-ulp spread (CPU golden codegen vs Metal codegen —
    neither side wrong: Metal measured CLOSER to the fp64 reference, 0.7 vs
    3.6 ulps of the pre-cancellation terms) to 4.7e-5 relative, past corpus
    rtol. min-r is now 0.22 (≤4.6x amplification).
    """
    model, data = _flex_elem_layout(rng)
    model["flex_vertbodyid"] = np.array([0, 1, 2], dtype=np.int32)
    model["flex_damping"] = np.array([0.8], dtype=np.float32)
    args = [("opt_timestep", "arr:f32", np.full(_FLEX_WORLDS, 0.4, dtype=np.float32))]
    args += [
        (name, _FLEX_ELEM_KEYS[name], model[name])
        for name in (
            "flex_dim", "flex_vertadr", "flex_edgeadr", "flex_elemadr", "flex_elemdataadr",
            "flex_stiffnessadr", "flex_elemedgeadr", "flex_vertbodyid", "flex_elem",
            "flex_elemedge", "flexedge_length0", "flex_stiffness", "flex_damping",
            "flex_elemflexid",
        )
    ]
    args += [
        ("xipos_in", "arr2:vec3",
         (rng.standard_normal((_FLEX_WORLDS, 3, 3)) * 0.45
          + np.array([[0.11, -0.09, 0.13], [0.05, 0.07, -0.11], [-0.08, 0.12, 0.06]])[None]).astype(np.float32)),
        ("flexvert_xpos_in", "arr2:vec3", data["flexvert_xpos_in"]),
        ("flexedge_length_in", "arr2:f32", data["flexedge_length_in"]),
        ("flexedge_velocity_in", "arr2:f32", (rng.standard_normal((_FLEX_WORLDS, 3)) * 0.5).astype(np.float32)),
        ("dsbl_spring", "scalar:bool", False),
        ("dsbl_damper", "scalar:bool", False),
        ("flex_spring_body_force_out", "out2:spatial_vector", (_FLEX_WORLDS, 3)),
        ("flex_damper_body_force_out", "out2:spatial_vector", (_FLEX_WORLDS, 3)),
        ("__dim__", "", (_FLEX_WORLDS, 1)),
    ]
    return (args, [])


def b_cinert(rng):
    """mujoco_warp smooth._cinert(): body cinert 10-vectors — vec10 register
    vector locals (`res = vec10()` with `res[i] += / -= / =` compound
    element assigns) and a vec10 composite array store. Three bodies with
    positive mass/inertia; ximat random rotations; body_rootid 0 keeps the
    com offset live (dif != 0)."""
    nworld, nbody = _FLEX_WORLDS, 3
    mass = np.tile(rng.uniform(0.5, 2.0, nbody)[None], (nworld, 1)).astype(np.float32)
    inertia = np.tile(rng.uniform(0.05, 0.4, (1, nbody, 3)), (nworld, 1, 1)).astype(np.float32)
    args = [
        ("body_rootid", "arr:i32", np.zeros(nbody, dtype=np.int32)),
        ("body_mass", "arr2:f32", mass),
        ("body_inertia", "arr2:vec3", inertia),
        ("xipos_in", "arr2:vec3", (rng.standard_normal((nworld, nbody, 3)) * 0.2).astype(np.float32)),
        ("ximat_in", "arr2:mat33", (rng.standard_normal((nworld, nbody, 9)) * 0.1).astype(np.float32)),
        ("subtree_com_in", "arr2:vec3", (rng.standard_normal((nworld, nbody, 3)) * 0.1).astype(np.float32)),
        ("cinert_out", "out2:vec10", (nworld, nbody)),
        ("__dim__", "", (nworld, nbody)),
    ]
    return (args, [])


def b_implicit_free_body(rng):
    """mujoco_warp forward._implicit_free_body(solve=False): the implicit
    mass matrix of one standalone 6-dof free body, multiplied by a vector —
    wp.add_inplace (damping diagonal + gyroscopic blocks) and wp.sub_inplace
    (box-fluid J'B J subtraction) on a local mat66, plus a vec10 composite
    array load (crb_in, subtree mass at index 9). Dense 6x6 M CSR
    (rownnz 6, colind [0..5] per row); DAMPER enabled, SLEEP off, density
    > 0 with body_fluid_ellipsoid False routes the fluid derivative through
    the _deriv_box_fluid branch (body 1 mass >= MJ_MINVAL)."""
    nworld, nbody, nv = _FLEX_WORLDS, 2, 6
    mass = np.zeros((nworld, nbody), dtype=np.float32)
    mass[:, 1] = 1.2
    inertia = np.zeros((nworld, nbody, 3), dtype=np.float32)
    inertia[:, 1, :] = np.array([0.08, 0.1, 0.12], dtype=np.float32)
    m_dense = rng.standard_normal((nworld, nv, nv)) * 0.1
    m_dense = (m_dense + np.transpose(m_dense, (0, 2, 1)) + 4.0 * np.eye(nv)[None]).astype(np.float32)
    crb = np.zeros((nworld, nbody, 10), dtype=np.float32)
    crb[:, 1, 0:6] = np.array([0.08, 0.1, 0.12, 0.01, 0.0, 0.02], dtype=np.float32)
    crb[:, 1, 6:9] = np.array([0.05, -0.03, 0.02], dtype=np.float32)
    crb[:, 1, 9] = 1.2
    args = [
        ("nbody", "scalar:i32", nbody),
        ("opt_timestep", "arr:f32", np.full(nworld, 0.4, dtype=np.float32)),
        ("opt_wind", "arr:vec3", np.full((nworld, 3), 0.1, dtype=np.float32)),
        ("opt_density", "arr:f32", np.full(nworld, 1000.0, dtype=np.float32)),
        ("opt_viscosity", "arr:f32", np.full(nworld, 1.0, dtype=np.float32)),
        ("opt_integrator", "scalar:i32", 2),
        ("opt_disableflags", "scalar:i32", 0),
        ("opt_enableflags", "scalar:i32", 0),
        ("body_rootid", "arr:i32", np.array([0, 1], dtype=np.int32)),
        ("body_dofadr", "arr:i32", np.array([-1, 0], dtype=np.int32)),
        ("body_geomnum", "arr:i32", np.zeros(nbody, dtype=np.int32)),
        ("body_geomadr", "arr:i32", np.zeros(nbody, dtype=np.int32)),
        ("body_mass", "arr2:f32", mass),
        ("body_inertia", "arr2:vec3", inertia),
        ("dof_treeid", "arr:i32", np.zeros(nv, dtype=np.int32)),
        ("dof_damping", "arr2:f32", np.full((nworld, nv), 0.3, dtype=np.float32)),
        ("dof_dampingpoly", "arr2:vec2", np.zeros((nworld, nv, 2), dtype=np.float32)),
        ("geom_type", "arr:i32", np.array([4], dtype=np.int32)),
        ("geom_size", "arr2:vec3", np.full((nworld, 1, 3), 0.2, dtype=np.float32)),
        ("geom_fluid", "arr2:f32", np.zeros((1, 12), dtype=np.float32)),
        ("M_rownnz", "arr:i32", np.full(nv, nv, dtype=np.int32)),
        ("M_rowadr", "arr:i32", np.arange(nv, dtype=np.int32) * nv),
        ("M_colind", "arr:i32", np.tile(np.arange(nv, dtype=np.int32), nv)),
        ("body_fluid_ellipsoid", "arr:bool", np.zeros(nbody, dtype=bool)),
        ("body_freeadr", "arr:i32", np.array([1], dtype=np.int32)),
        ("qvel_in", "arr2:f32", (rng.standard_normal((nworld, nv)) * 0.3).astype(np.float32)),
        ("xpos_in", "arr2:vec3", (rng.standard_normal((nworld, nbody, 3)) * 0.1).astype(np.float32)),
        ("xmat_in", "arr2:mat33", (rng.standard_normal((nworld, nbody, 9)) * 0.1).astype(np.float32)),
        ("xipos_in", "arr2:vec3", (rng.standard_normal((nworld, nbody, 3)) * 0.1).astype(np.float32)),
        ("ximat_in", "arr2:mat33", (rng.standard_normal((nworld, nbody, 9)) * 0.1).astype(np.float32)),
        ("geom_xpos_in", "arr2:vec3", (rng.standard_normal((nworld, 1, 3)) * 0.1).astype(np.float32)),
        ("geom_xmat_in", "arr2:mat33", (rng.standard_normal((nworld, 1, 9)) * 0.1).astype(np.float32)),
        ("subtree_com_in", "arr2:vec3", (rng.standard_normal((nworld, nbody, 3)) * 0.1).astype(np.float32)),
        ("cdof_in", "arr2:spatial_vector", (rng.standard_normal((nworld, nv, 6)) * 0.1).astype(np.float32)),
        ("crb_in", "arr2:vec10", crb),
        ("M_in", "arr2:f32", m_dense.reshape(nworld, nv * nv)),
        ("tree_awake_in", "arr2:i32", np.ones((nworld, 1), dtype=np.int32)),
        ("cvel_in", "arr2:spatial_vector", (rng.standard_normal((nworld, nbody, 6)) * 0.1).astype(np.float32)),
        ("solve", "scalar:bool", False),
        ("vec_in", "arr2:f32", (rng.standard_normal((nworld, nv)) * 0.5).astype(np.float32)),
        ("vec_out", "out2:f32", (nworld, nv)),
        ("__dim__", "", (nworld, 1)),
    ]
    return (args, [])


# ---- KEP-0001 M3.2 package b: the mujoco_warp tile closure (recon B.4) ----
# Launch discipline mirrors the real Newton-path call sites: the solver pair
# runs launch_tiled dim=(nworld,) block_dim=128 (solver.py:3466-3486), the
# smooth pair dim=(nworld, nblock) block_dim=32 (smooth.py:1388/3971). CPU
# goldens resolve the same block_dim to the serial bd=1 scalar tier, so the
# arithmetic order per element (k ascending matmul, unique-address scatter)
# is identical on both sides and the rows are bit-comparable.

_JTDAJ_NV_PAD = 16
_JTDAJ_NJMAX = 8
_JTDAJ_NWORLD = 2
_JTDAJ_NCSR = 12  # nC: CSR entry count; sentinel value == nC reads as zero


def b_jtdaj_dense_compact(rng):
    """mujoco_warp solver._update_gradient_JTDAJ_dense_tiled_compact(16, 16, 8):
    njmax=8 < tile_size=16 pins the in-kernel tile_size=8 branch. world0 has
    nefc == njmax (every active-check lane taken), world1 nefc=5 (the
    zero-branch lanes exercise _active_check's masking); efc_state mixes ±1
    (the _state_check zero branch). All domains in-bounds."""
    nefc = np.array([8, 5], dtype=np.int32)
    m = (rng.standard_normal((_JTDAJ_NWORLD, _JTDAJ_NV_PAD, _JTDAJ_NV_PAD)) * 0.5).astype(np.float32)
    jac = (rng.standard_normal((_JTDAJ_NWORLD, _JTDAJ_NJMAX, _JTDAJ_NV_PAD)) * 0.5).astype(np.float32)
    diag = (rng.standard_normal((_JTDAJ_NWORLD, _JTDAJ_NJMAX)) * 0.25 + 1.0).astype(np.float32)
    state = rng.integers(-1, 2, size=(_JTDAJ_NWORLD, _JTDAJ_NJMAX)).astype(np.int32)
    args = [
        ("nefc_in", "arr:i32", nefc),
        ("M_in", "arr3:f32", m),
        ("efc_J_in", "arr3:f32", jac),
        ("efc_D_in", "arr2:f32", diag),
        ("efc_state_in", "arr2:i32", state),
        ("ctx_done_in", "arr:bool", np.zeros(_JTDAJ_NWORLD, dtype=bool)),
        ("ctx_h_out", "out3:f32", (_JTDAJ_NWORLD, _JTDAJ_NV_PAD, _JTDAJ_NV_PAD)),
        ("__dim__", "", (_JTDAJ_NWORLD,)),
        ("__block_dim__", "", 128),
    ]
    return (args, [])


def b_jtdaj_dense(rng):
    """mujoco_warp solver._update_gradient_JTDAJ_dense_tiled(16, 16, 8, 12):
    the CSR densify variant. Scatter addresses pos = colind*nv_pad + hinit
    are pairwise unique and < nv_pad^2 (the Atomic=false tile_scatter_add
    premise); 12 CSR entries over a 256-slot tile leaves the tile_zeros
    remainder at zero (partial scatter). Same nefc mix as the compact row."""
    pos = np.arange(2, 2 + _JTDAJ_NCSR * 10, 10, dtype=np.int32)  # 2..112, unique, < 256
    colind = (pos // _JTDAJ_NV_PAD).astype(np.int32)
    hinit = (pos % _JTDAJ_NV_PAD).astype(np.int32)
    nefc = np.array([8, 5], dtype=np.int32)
    csr = (rng.standard_normal((_JTDAJ_NWORLD, _JTDAJ_NCSR)) * 0.5).astype(np.float32)
    jac = (rng.standard_normal((_JTDAJ_NWORLD, _JTDAJ_NJMAX, _JTDAJ_NV_PAD)) * 0.5).astype(np.float32)
    diag = (rng.standard_normal((_JTDAJ_NWORLD, _JTDAJ_NJMAX)) * 0.25 + 1.0).astype(np.float32)
    state = rng.integers(-1, 2, size=(_JTDAJ_NWORLD, _JTDAJ_NJMAX)).astype(np.int32)
    args = [
        ("M_colind", "arr:i32", colind),
        ("M_hinit_i", "arr:i32", hinit),
        ("nefc_in", "arr:i32", nefc),
        ("M_in", "arr2:f32", csr),
        ("efc_J_in", "arr3:f32", jac),
        ("efc_D_in", "arr2:f32", diag),
        ("efc_state_in", "arr2:i32", state),
        ("ctx_done_in", "arr:bool", np.zeros(_JTDAJ_NWORLD, dtype=bool)),
        ("ctx_h_out", "out3:f32", (_JTDAJ_NWORLD, _JTDAJ_NV_PAD, _JTDAJ_NV_PAD)),
        ("__dim__", "", (_JTDAJ_NWORLD,)),
        ("__block_dim__", "", 128),
    ]
    return (args, [])


_TILE_BLOCK_NC = 16  # CSR length; sentinel value == nC marks absent pairs
_TILE_BLOCK_ELEMID = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 16, 16, 16, 15], dtype=np.int32)


def _tile_block_m(rng):
    """CSR values for the gathered 4x4 upper block
    [[2,.2,0,.1],[_,2,.2,0],[_,_,2,0],[_,_,_,2]] — diagonally dominant SPD
    so tile_cholesky_inplace<upper> and its solve stay finite on both
    sides; the sentinel slots (elemid == nC) read T{} zero through the
    tile_load_indexed bounds check (io.py:1224 discipline)."""
    del rng
    m = np.zeros(_TILE_BLOCK_NC, dtype=np.float32)
    for slot in (0, 5, 10, 15):
        m[slot] = 2.0  # block diagonals
    m[1] = 0.2
    m[6] = 0.2
    m[3] = 0.1
    return m


def b_tile_cholesky_factorize_block(rng):
    """mujoco_warp smooth._tile_cholesky_factorize_block(TileSet(size=4)):
    one (world, block) group; block_elemid carries real CSR indices plus
    four sentinels == nC (the OOB-reads-zero discipline). Launch mirrors
    smooth.py:1388 (dim=(nworld, tile.adr.size), block_dim=32)."""
    args = [
        ("qLD_block_adr", "arr:i32", np.array([0], dtype=np.int32)),
        ("M_in", "arr2:f32", _tile_block_m(rng).reshape(1, _TILE_BLOCK_NC)),
        ("block_elemid", "arr:i32", _TILE_BLOCK_ELEMID.copy()),
        ("block_dof", "arr:i32", np.array([0], dtype=np.int32)),
        ("L_out", "out2:f32", (1, _TILE_BLOCK_NC)),
        ("__dim__", "", (1, 1)),
        ("__block_dim__", "", 32),
    ]
    return (args, [])


def b_tile_cholesky_factorize_solve_block(rng):
    """mujoco_warp smooth._tile_cholesky_factorize_solve_block(TileSet(4)):
    the fused factor+solve variant — same fixture plus a nonzero rhs y
    (x = L-upper-solve result, finite by the SPD block)."""
    y = np.zeros((1, _TILE_BLOCK_NC), dtype=np.float32)
    y[0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    args = [
        ("qLD_block_adr", "arr:i32", np.array([0], dtype=np.int32)),
        ("M_in", "arr2:f32", _tile_block_m(rng).reshape(1, _TILE_BLOCK_NC)),
        ("block_elemid", "arr:i32", _TILE_BLOCK_ELEMID.copy()),
        ("block_dof", "arr:i32", np.array([0], dtype=np.int32)),
        ("y", "arr2:f32", y),
        ("x", "out2:f32", (1, _TILE_BLOCK_NC)),
        ("L_out", "out2:f32", (1, _TILE_BLOCK_NC)),
        ("__dim__", "", (1, 1)),
        ("__block_dim__", "", 32),
    ]
    return (args, [])


BUILDERS = {
    name[2:]: fn  # strip leading "b_"
    for name, fn in list(globals().items())
    if name.startswith("b_") and callable(fn)
}
