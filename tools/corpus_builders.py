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
        # kernel internally promotes inertia math to float64 (wp.mat64);
        # Apple GPUs expose no fp64 in Metal — KEP-0001 capability boundary
        ["kernel body uses float64 (wp::mat_t<3,3,wp::float64> in SSA) — Metal has no fp64 on Apple GPUs"],
    )


def b_validate_and_correct_inertia_kernel(rng):
    n = 5
    inertia = _eye_inertia(rng, n)
    inertia[1] *= 1e6  # trigger the balance branch
    inertia[3, 0] = -0.5  # negative diagonal → correction branch
    return (
        [
            ("body_mass", "arr:f32", rng.uniform(0.5, 5.0, n).astype(np.float32)),
            ("body_inertia", "arr:mat33", inertia),
            ("balance_inertia", "scalar:bool", True),
            ("bound_mass", "scalar:f32", 1000.0),
            ("bound_inertia", "scalar:f32", 1000.0),
            ("relative_tolerance", "scalar:f32", 0.01),
            ("absolute_tolerance", "scalar:f32", 1e-6),
            ("correction", "scalar:f32", 1.0),
            ("correction_count", "out:i32", 1),
        ],
        # float64 inertia promotion inside the kernel — same boundary as above
        ["kernel body uses float64 (wp::mat_t<3,3,wp::float64> in SSA) — Metal has no fp64 on Apple GPUs"],
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


BUILDERS = {
    name[2:]: fn  # strip leading "b_"
    for name, fn in list(globals().items())
    if name.startswith("b_") and callable(fn)
}
