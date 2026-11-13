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
    return ([], ["bvh_get_group_root requires a built BVH context (Phase 2, M2.1)"])


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
            ("shape_type", "arr:i32", np.full(n, 4, dtype=np.int32)),  # GEOM_SPHERE-like
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
        # support-point path calls wp.Mesh device queries — Phase 2 (KEP-0001 M2.x)
        ["kernel body requires wp.Mesh device queries (SSA wp::Mesh) — Phase 2 (KEP-0001)"],
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
    n = 4
    return (
        [
            ("shape_pairs_mesh_plane", "arr:vec2i", np.stack([np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32)], axis=1)),
            ("shape_pairs_mesh_plane_count", "arr:i32", np.full(1, n, dtype=np.int32)),
            ("shape_source", "arr:u64", np.zeros(n, dtype=np.uint64)),
            ("vert_counts", "out:i32", n),
            ("total_vert_count", "out:i32", 1),
            ("total_num_threads", "scalar:i32", n),
        ],
        # mesh-geometry vertex layout comes from wp.Mesh device data — Phase 2
        ["kernel body requires wp.Mesh device queries (SSA wp::Mesh) — Phase 2 (KEP-0001)"],
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


# factory kernels: resolved by calling a module-level factory at collection
# time (golden/compare both call it under their own warp package)
FACTORY_BUILDS = {
    "newton._src.solvers.narrow_phase": {
        "create_narrow_phase_primitive_kernel.<locals>.narrow_phase_primitive_kernel": [
            ("primitive_kernel", ("sphere_plane",)),
            ("__builder__", "b_narrow_phase_primitive_kernel"),
        ],
    },
}


def b_narrow_phase_primitive_kernel(rng):
    # filled in after factory probe (shapes/args depend on factory closure)
    return ([], ["narrow_phase factory kernel: builder pending (M1.4 scope)"])


BUILDERS = {
    name[2:]: fn  # strip leading "b_"
    for name, fn in list(globals().items())
    if name.startswith("b_") and callable(fn)
}
