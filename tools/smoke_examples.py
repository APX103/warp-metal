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

Feedback control (default-on — macmon must be installed, ``brew install
macmon``; without it the gate fail-opens and fixed pacing still applies):

- GPU gate — before launching each example, macmon (IOReport, sudoless,
  NOT a Metal client so the one-Metal-process discipline is untouched) is
  polled until GPU active-ratio stays below ``SMOKE_GPU_MAX_PCT`` (default
  30) for 2 consecutive samples. GUI work (video, browsers, compiles)
  automatically delays the sweep; idle machine = full speed. After
  ``SMOKE_GPU_MAX_WAIT_S`` (default 180) it proceeds anyway and says so —
  the batch budget still bounds total exposure.
- Per-example monitoring — a background sampler records each example's
  peak GPU active-ratio and peak GPU power into its results.json record.
- Batch budget — ``SMOKE_BUDGET_S`` (default 900; 0 disables) stops the
  batch cleanly before starting a new example once the wall-clock budget
  is spent, so any single invocation's GPU exposure is bounded.
- Stop file — ``touch SMOKE_STOP_FILE`` (default
  ``/tmp/phase4_smoke_STOP``) pauses the batch cleanly between examples
  (results are already saved after every example; ``--resume`` default
  picks up where it stopped).
- Scheduling — example subprocesses run under ``taskpolicy -b``
  (Darwin background priority: efficiency-core preference, yields first
  to foreground GUI). ``SMOKE_TASKPOLICY=0`` disables.

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
import shutil
import subprocess
import sys
import threading
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

# Feedback control knobs (see module docstring).
GPU_GATE = os.environ.get("SMOKE_GPU_GATE", "1") == "1"
GPU_MAX_PCT = float(os.environ.get("SMOKE_GPU_MAX_PCT", "30"))
GPU_MAX_WAIT_S = float(os.environ.get("SMOKE_GPU_MAX_WAIT_S", "180"))
BUDGET_S = int(os.environ.get("SMOKE_BUDGET_S", "900"))
STOP_FILE = os.environ.get("SMOKE_STOP_FILE", "/tmp/phase4_smoke_STOP")
TASKPOLICY = os.environ.get("SMOKE_TASKPOLICY", "1") == "1"

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


class GpuMonitor:
    """Long-lived macmon pipe reader (IOReport, sudoless, not a Metal client).

    Keeps the newest GPU sample under a lock; optionally accumulates per-run
    peaks while sampling is on. Everything fail-opens: if macmon is absent or
    dies, snapshots return None and the gate refuses to block.
    """

    def __init__(self, interval_ms: int = 500):
        self._proc = None
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._sampling = False
        self._peak: dict = {}
        if shutil.which("macmon") is None:
            return
        try:
            self._proc = subprocess.Popen(
                ["macmon", "pipe", "-i", str(interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError:
            return
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            gu = d.get("gpu_usage")
            if not (isinstance(gu, list) and len(gu) == 2):
                continue
            snap = {
                "gpu_active_ratio": float(gu[1]),
                "gpu_freq_mhz": float(gu[0]),
                "gpu_power_w": float(d.get("gpu_power") or 0.0),
                "ts": time.monotonic(),
            }
            with self._lock:
                self._latest = snap
                if self._sampling:
                    self._peak["gpu_active_ratio"] = max(
                        self._peak.get("gpu_active_ratio", 0.0),
                        snap["gpu_active_ratio"],
                    )
                    self._peak["gpu_power_w"] = max(
                        self._peak.get("gpu_power_w", 0.0),
                        snap["gpu_power_w"],
                    )

    @property
    def available(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def snapshot(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest else None

    def gate(self, max_ratio: float, max_wait_s: float, ok_streak: int = 2) -> dict:
        """Wait until GPU active-ratio stays below max_ratio for ok_streak
        consecutive samples; give up (proceed) after max_wait_s. Never raises.
        """
        started = time.monotonic()
        deadline = started + max_wait_s
        streak = 0
        gave_up = False
        while True:
            snap = self.snapshot()
            if snap is None:
                if time.monotonic() > deadline:
                    gave_up = True
                    break
                time.sleep(0.2)
                continue
            if snap["gpu_active_ratio"] < max_ratio:
                streak += 1
                if streak >= ok_streak:
                    break
            else:
                streak = 0
            if time.monotonic() > deadline:
                gave_up = True
                break
            time.sleep(0.2)
        return {
            "waited_s": round(time.monotonic() - started, 1),
            "gave_up": gave_up,
            "max_ratio": max_ratio,
        }

    def start_sampling(self) -> None:
        with self._lock:
            self._sampling = True
            self._peak = {}

    def stop_sampling(self) -> dict:
        with self._lock:
            self._sampling = False
            return dict(self._peak)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


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
    if TASKPOLICY and shutil.which("taskpolicy"):
        # Darwin background priority: efficiency-core preference, yields to
        # the foreground GUI (children inherit).
        cmd = ["taskpolicy", "-b"] + cmd

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
        if TASKPOLICY and shutil.which("taskpolicy"):
            cmd = ["taskpolicy", "-b"] + cmd
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
            peak = (r.get("gpu_peak") or {}).get("gpu_active_ratio")
            peak_s = f"  gpu-peak={peak:.0%}" if peak is not None else ""
            print(f"  {mark:14s} {r['seconds']:7.1f}s{peak_s}  {r['name']}")
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
    monitor = GpuMonitor()
    if GPU_GATE and not monitor.available:
        print("gpu gate disabled (macmon unavailable) — fixed pacing only", flush=True)
    batch_t0 = time.monotonic()
    gated_waited = 0.0
    try:
        for i, name in enumerate(names):
            if name in results and not args.fresh:
                n_skip += 1
                continue
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
                print(f"\nstop file seen ({STOP_FILE}) — pausing cleanly "
                      f"after {n_pass} pass / {n_fail} fail", flush=True)
                break
            if BUDGET_S > 0 and time.monotonic() - batch_t0 > BUDGET_S:
                print(f"\nbatch budget {BUDGET_S}s spent — stopping cleanly "
                      f"after {n_pass} pass / {n_fail} fail", flush=True)
                break
            if GPU_GATE and monitor.available:
                gate = monitor.gate(GPU_MAX_PCT / 100.0, GPU_MAX_WAIT_S)
                gated_waited += gate["waited_s"]
                if gate["waited_s"] > 1.0:
                    note = " — gave up, proceeding anyway" if gate["gave_up"] else ""
                    print(f"    (gpu gate waited {gate['waited_s']}s{note})", flush=True)
            print(f"[{i + 1}/{len(names)}] {name} ...", flush=True)
            monitor.start_sampling()
            rec = run_one(name)
            rec["gpu_peak"] = monitor.stop_sampling()
            results[name] = rec
            _save_results(results)
            mark = "PASS" if rec["rc"] == 0 else f"rc={rec['rc']}"
            peak = rec["gpu_peak"].get("gpu_active_ratio")
            peak_s = f" gpu-peak={peak:.0%}" if peak is not None else ""
            print(f"    -> {mark} in {rec['seconds']}s ({rec['mode']}){peak_s}", flush=True)
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
    finally:
        monitor.close()

    print(f"\ndone: pass={n_pass} fail={n_fail} skipped-already-recorded={n_skip} "
          f"gpu-gate-wait={gated_waited:.0f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
