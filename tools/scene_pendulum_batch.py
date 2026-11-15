#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KEP-0001 M2.6 big-scene benchmark fixture: batched double-pendulum envs.

The M2.6 acceptance (revised, user-ratified 2026-11-15) requires the >=3x
metal-vs-single-thread-CPU gate to be judged on a **big scene**: batch>=256
parallel environments, or a cloth/particle scene with >=100k contacts.  The
three tiny stock scenes (2/7/18 bodies) are dispatch/flush-overhead bound and
structurally cannot reach the gate; they remain in the report as dispatch
overhead evidence only.

This module provides that big scene as a bench-local ``Example`` following the
Newton example protocol (``__init__(viewer, args)`` / ``step()`` /
``sim_substeps`` / ``sim_dt``), built with public Newton APIs only:

- one double-pendulum rig, byte-faithful to ``newton.examples.basic_pendulum``
  (two box links, two revolute joints, one articulation, ground plane,
  SolverXPBD, CollisionPipeline, 10 substeps at sim_dt = frame_dt/10);
- replicated ``--num-envs`` times (default 512) via
  ``ModelBuilder.replicate(..., separate_worlds=True)`` -- each copy lives in
  its own world, so copies never collide with each other and every pendulum
  simulates independently (true batch-parallel envs);
- the per-frame loop is exactly the stock example's (clear_forces; viewer
  apply_forces no-op; collide; SolverXPBD.step; state swap), recorded once at
  construction into a warp graph and replayed per step -- exactly as the stock
  example runs on each device (cpu APIC / metal record-replay).

The scene asserts structural invariants only (batch size, finite state after
warmup); it is a measurement fixture, not a registered Newton example.
"""

from __future__ import annotations

import math

import warp as wp

import newton


class Example:
    """Batched double-pendulum scene: one rig replicated ``num_envs`` times."""

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_type = "xpbd"

        self.num_envs = int(getattr(args, "num_envs", 512))
        if self.num_envs < 256:
            raise ValueError(
                f"--num-envs must be >= 256 (the M2.6 big-scene gate is defined on batch>=256); got {self.num_envs}"
            )

        # One pendulum rig, byte-faithful to newton.examples.basic_pendulum.
        rig = newton.ModelBuilder()
        hx = 1.0
        hy = 0.1
        hz = 0.1

        link_0 = rig.add_link()
        rig.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)
        link_1 = rig.add_link()
        rig.add_shape_box(link_1, hx=hx, hy=hy, hz=hz)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = rig.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        j1 = rig.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        rig.add_articulation([j0, j1], label="pendulum")
        rig.add_ground_plane()

        # Replicate into independent worlds (copies never collide with each
        # other); all copies stay at the origin per the replicate() guidance.
        builder = newton.ModelBuilder()
        builder.replicate(rig, copy_count=self.num_envs, separate_worlds=True)

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer = viewer
        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def assert_sane(self):
        """Structural sanity (used between bench runs, never inside the timed loop)."""
        host_q = self.state_0.body_q.numpy()
        if not math.isfinite(float(host_q.reshape(-1).sum())):
            raise RuntimeError("batch pendulum state went non-finite")
        if host_q.shape[0] != 2 * self.num_envs:
            raise RuntimeError(f"expected {2 * self.num_envs} bodies, got {host_q.shape[0]}")


def add_parser_arguments(parser) -> None:
    """Bench-scene CLI additions (mirrors the stock example's --solver pattern)."""
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of replicated pendulum worlds (M2.6 gate requires >=256; "
        "4096 default puts the per-frame workload far above the metal replay "
        "overhead floor so the >=3x gate is compute-bound, not dispatch-bound).",
    )
