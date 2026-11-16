# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 S1: headless smoke matrix for all registered Newton examples on metal.

Runs each registered example from the Newton repo as a strict subprocess:

    NEWTON_WARP_BACKEND=metal uv run -m newton.examples <name> \
        --device metal --viewer null --test

with a per-example watchdog (default 240 s -> rc 124). GPU dispatch has the
shim's own inner bound (WARP_METAL_DISPATCH_TIMEOUT_S default 120 s) and is
NEVER set here. Runs are strictly sequential: exactly one Metal process at a
time (solo-GPU discipline).

Pacing (mandatory, default-on — Apple Silicon shares the GPU with the
display, and sustained max-duty Metal work starves WindowServer and makes
the whole machine unusable):

- ``SMOKE_PACING_S`` — seconds to sleep after EVERY example completes
  (default 10). Set to ``0`` only when the user explicitly asks for an
  unpaced run.
- A 30 s cooldown fires after every 10 examples on top of the per-example
  pacing.
- ``--test`` runs always use the example's default frame count: the
  ``test_final`` success criteria are calibrated to those defaults, so
  shrinking the frame budget would change test semantics rather than load.
  The no-test-criteria fallback already runs a reduced plain 50-frame pass.

Results are appended to <log-dir>/results.json after every example so a
partial run is still a usable state. ``--resume`` (default) skips examples
already recorded there. Logs land at <log-dir>/<name>.log (gitignored path).

Usage (from the warp-metal checkout):

    python tools/smoke_examples.py --all
    python tools/smoke_examples.py --names basic_pendulum,robot_g1
    python tools/smoke_examples.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for paths.py
import paths  # noqa: E402

NEWTON_REPO = str(paths.newton_repo())
LOG_DIR = "/tmp/phase4_smoke_logs"
RESULTS_PATH = os.path.join(LOG_DIR, "results.json")

TIMEOUT_S = 240
TAIL_LINES = 60

# Display-shared-GPU pacing (see module docstring): sleep after every example,
# plus a longer cooldown every N examples.
PACING_S = int(os.environ.get("SMOKE_PACING_S", "10"))
COOLDOWN_EVERY = 10
COOLDOWN_S = 30

# The canonical run list: every name printed by `uv run -m newton.examples
# --list` at S1 start (124 entries). Regenerate with --refresh-list if the
# registry grows.
EXAMPLES: list[str] = [
    "basic_conveyor",
    "basic_conveyor_forces",
    "basic_dzhanibekov",
    "basic_heightfield",
    "basic_joints",
    "basic_mimic_joint",
    "basic_multi_solver_overlay",
    "basic_pendulum",
    "basic_plotting",
    "basic_shapes",
    "basic_urdf",
    "basic_viewer",
    "recording",
    "replay_viewer",
    "cable_bundle_hysteresis",
    "cable_cross_slide_table",
    "cable_initial_rest_pose",
    "cable_pile",
    "cable_plectoneme",
    "cable_twist",
    "cable_y_junction",
    "cloth_bending",
    "cloth_franka",
    "cloth_h1",
    "cloth_hanging",
    "cloth_poker_cards",
    "cloth_rollers",
    "cloth_style3d",
    "cloth_twist",
    "adaptive_collision",
    "balance_bird",
    "brick_stacking",
    "contacts_rj45_plug",
    "domino_spiral",
    "newton_cradle",
    "nut_bolt_hydro",
    "nut_bolt_sdf",
    "pyramid",
    "controller_differential_ik",
    "controller_joint_impedance_heterogeneous",
    "controller_operational_space_hybrid_force_motion",
    "diffsim_ball",
    "diffsim_bear",
    "diffsim_cloth",
    "diffsim_drone",
    "diffsim_soft_body",
    "diffsim_spring_cage",
    "ik_cube_stacking",
    "ik_custom",
    "ik_franka",
    "ik_h1",
    "kamino_basic_dr_testmech",
    "kamino_basic_fourbar",
    "kamino_basic_heterogeneous",
    "kamino_earworm_flowers",
    "kamino_robot_anymal_d",
    "kamino_robot_dr_legs",
    "mpm_anymal",
    "mpm_beam_twist",
    "mpm_grain_rendering",
    "mpm_granular",
    "mpm_multi_material",
    "mpm_pinch_carry",
    "mpm_snow_ball",
    "mpm_twoway_coupling",
    "mpm_viscous",
    "mpm_water_dam_break",
    "mujoco_sleeping",
    "admm_contact_solver",
    "franka_cable_ik_pick_place",
    "kamino_mujoco_admm_solver",
    "mujoco_franka_vbd_cable_admm_solver",
    "mujoco_mpm_coupled_solver",
    "mujoco_vbd_admm_solver",
    "mujoco_vbd_coupled_solver",
    "mujoco_xpbd_coupled_solver",
    "proxy_joint_gripper",
    "rigid_soft_contact",
    "softbody_dropping_to_cloth",
    "softbody_gift",
    "vbd_dat_rigid_soft",
    "vbd_mpm_coupled_solver",
    "xpbd_mpm_coupled_solver",
    "xpbd_vbd_coupled_solver",
    "robot_allegro_hand",
    "robot_anymal_c_walk",
    "robot_anymal_d",
    "robot_asroballet",
    "robot_cartpole",
    "robot_g1",
    "robot_h1",
    "robot_omniwheel",
    "robot_panda_hydro",
    "robot_policy",
    "robot_ur10",
    "selection_articulations",
    "selection_cartpole",
    "selection_materials",
    "selection_multiple",
    "sensor_camera",
    "sensor_contact",
    "sensor_imu",
    "softbody_franka",
    "softbody_hanging",
    "steel_determinism_arms",
    "steel_sleeping",
    "surface_gripper",
    "tendon_3d_routing",
    "tendon_capstan_friction",
    "tendon_mujoco_switch",
    "tendon_pinhole",
    "tendon_pulley",
    "cable_bend_stiffness",
    "cable_bend_twist_analytic",
    "cable_dahl_hysteresis",
    "cable_michell_threshold",
    "cable_torsion_material_mapping",
    "cable_twist_buckling",
    "cable_twist_transfer",
    "cloth_stiff_material_hanging",
    "cloth_stiff_material_stretch",
    "vbd_gripper_soft_grid",
    "vbd_gripper_soft_triangle",
    "vbd_rigid_rigid_contact",
    "vbd_soft_rigid_contact",
    "vbd_soft_rigid_mix_contact",
]


def _load_results() -> dict[str, dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_results(results: dict[str, dict]) -> None:
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    os.replace(tmp, RESULTS_PATH)


def _log_tail(path: str, n: int = TAIL_LINES) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "<no log>"


def run_one(name: str, timeout_s: int = TIMEOUT_S) -> dict:
    """Run one example strictly sequentially; returns its result record."""
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    env = os.environ.copy()
    env["NEWTON_WARP_BACKEND"] = "metal"
    env["PYTHONUNBUFFERED"] = "1"
    # Deliberately do NOT set WARP_METAL_DISPATCH_TIMEOUT_S: the shim's
    # default 120 s inner GPU-fault bound stays in force.
    env.pop("WARP_METAL_DISPATCH_TIMEOUT_S", None)

    mode = "test"
    cmd = [
        "uv", "run", "-m", "newton.examples", name,
        "--device", "metal", "--viewer", "null", "--test",
    ]

    t0 = time.monotonic()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd, cwd=NEWTON_REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = 124
            mode = "test-timeout"
    elapsed = time.monotonic() - t0

    tail = _log_tail(log_path)
    # Fallback: example rejected one of the shared flags (argparse rc 2).
    if rc == 2 and ("unrecognized arguments" in tail or "invalid choice" in tail):
        mode = "no-test-criteria"
        cmd = ["uv", "run", "-m", "newton.examples", name,
               "--device", "metal", "--viewer", "null", "--num-frames", "50"]
        t0 = time.monotonic()
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                cmd, cwd=NEWTON_REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                rc = 124
                mode = "no-test-criteria-timeout"
        elapsed = time.monotonic() - t0
        tail = _log_tail(log_path)

    return {
        "name": name,
        "rc": rc,
        "seconds": round(elapsed, 1),
        "mode": mode,
        "log": log_path,
        "tail": tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="run every registered example")
    g.add_argument("--names", type=str, default=None, help="comma-separated example names")
    parser.add_argument("--fresh", action="store_true", help="ignore and overwrite prior results.json")
    parser.add_argument("--status", action="store_true", help="print summary of recorded results and exit")
    parser.add_argument("--refresh-list", action="store_true", help="re-derive the example list from the registry and print it")
    args = parser.parse_args()

    if args.refresh_list:
        out = subprocess.run(
            ["uv", "run", "-m", "newton.examples", "--list"],
            cwd=NEWTON_REPO, capture_output=True, text=True, check=True,
        )
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith(("basic", "cable", "cloth", "adaptive", "balance", "brick", "contacts", "domino", "newton", "nut", "pyramid", "controller", "diffsim", "ik_", "kamino", "mpm", "mujoco", "admm", "franka", "proxy", "rigid", "softbody", "vbd", "xpbd", "robot", "selection", "sensor", "steel", "surface", "tendon", "recording", "replay"))]
        print(json.dumps(names, indent=0))
        return 0

    os.makedirs(LOG_DIR, exist_ok=True)
    results = {} if args.fresh else _load_results()

    if args.status:
        done = [r for r in results.values()]
        npass = sum(1 for r in done if r["rc"] == 0)
        nfail = sum(1 for r in done if r["rc"] != 0)
        print(f"recorded={len(done)} pass={npass} fail={nfail} remaining={len(EXAMPLES) - len(done)}")
        for r in sorted(done, key=lambda r: (r["rc"] != 0, r["name"])):
            mark = "PASS" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
            print(f"  {mark:14s} {r['seconds']:7.1f}s  {r['name']}")
        return 0

    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        unknown = [n for n in names if n not in EXAMPLES]
        if unknown:
            parser.error(f"unknown example name(s): {unknown}")
    elif args.all:
        names = list(EXAMPLES)
    else:
        parser.error("one of --all / --names / --status is required")

    if args.fresh:
        results = {}

    n_pass = n_fail = n_skip = 0
    ran_since_cooldown = 0
    for i, name in enumerate(names):
        if name in results and not args.fresh:
            n_skip += 1
            continue
        print(f"[{i + 1}/{len(names)}] {name} ...", flush=True)
        rec = run_one(name)
        results[name] = rec
        _save_results(results)
        mark = "PASS" if rec["rc"] == 0 else f"rc={rec['rc']}"
        print(f"    -> {mark} in {rec['seconds']}s ({rec['mode']})", flush=True)
        if rec["rc"] == 0:
            n_pass += 1
        else:
            n_fail += 1
            print("    tail:", flush=True)
            for ln in rec["tail"].splitlines()[-12:]:
                print(f"      {ln}", flush=True)
        ran_since_cooldown += 1
        # Display-shared-GPU pacing: never run examples back-to-back.
        if PACING_S > 0:
            time.sleep(PACING_S)
        if ran_since_cooldown >= COOLDOWN_EVERY:
            ran_since_cooldown = 0
            if COOLDOWN_S > 0:
                print(f"    (cooldown {COOLDOWN_S}s after {COOLDOWN_EVERY} examples)", flush=True)
                time.sleep(COOLDOWN_S)

    print(f"\ndone: pass={n_pass} fail={n_fail} skipped-already-recorded={n_skip}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
