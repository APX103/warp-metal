#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KEP-0001 M2.6 performance baseline harness: Newton rigid-body scenes, cpu vs metal.

Times the simulate loop of three Newton ``Example``-class scenes on the ``cpu``
and ``metal`` warp devices and writes a markdown report (scene x device x
total seconds x ms/step x speedup).  The harness measures only; it never
estimates or fabricates a timing.  Speedups are computed from measured pairs
only, and rendered as ``n/a`` when the counterpart measurement is missing.

Scenes
------
- ``basic_pendulum``  double pendulum, SolverXPBD, CollisionPipeline (box-box/plane).
- ``basic_shapes``    seven primitives + bunny **mesh** dropped onto a ground
                      plane, SolverXPBD(iterations=5), CollisionPipeline; the
                      bunny exercises the Newton narrow-phase mesh-vs-plane path.
- ``basic_conveyor``  the dedicated mesh-contact scene: a kinematic annular-prism
                      **mesh belt** plus two static rail **meshes**, with 18
                      dynamic bags (boxes/capsules/spheres) in continuous
                      mesh-vs-rigid and bag-vs-bag contact, SolverXPBD.
                      Chosen over ``contacts_rj45_plug`` (VBD + mesh SDF +
                      cable rods, picking-driven, CUDA-only in CI) and
                      ``contacts_nut_bolt_*`` (network asset download,
                      CUDA-only in CI) because it is the CPU-viable rigid-body
                      example that exercises the Newton collision pipeline's
                      mesh paths headless, and it runs in CI test mode on cpu.

GPU exclusivity law: one process = one device.  Run each measurement as its
own solo invocation, then regenerate the report::

    cd /Volumes/ssd/main_link/work/newton
    # CPU baseline (all three scenes may share one process: CPU-only)
    uv run python /Users/apx103/work/warp-metal/tools/bench_phase2.py --device cpu
    # Metal runs: one solo process per scene, backend env set by the caller
    NEWTON_WARP_BACKEND=metal uv run python \
        /Users/apx103/work/warp-metal/tools/bench_phase2.py --device metal --scene basic_pendulum
    NEWTON_WARP_BACKEND=metal uv run python \
        /Users/apx103/work/warp-metal/tools/bench_phase2.py --device metal --scene basic_shapes
    NEWTON_WARP_BACKEND=metal uv run python \
        /Users/apx103/work/warp-metal/tools/bench_phase2.py --device metal --scene basic_conveyor
    # Afterward, regenerate the markdown report from the saved measurements
    uv run python /Users/apx103/work/warp-metal/tools/bench_phase2.py --report-only

What is timed
-------------
Exactly one frame of each example's test-mode loop, i.e. ``example.step()``,
repeated ``--frames`` times (default 100, matching the CI test runner's
``num-frames=100``), after ``--warmup-frames`` untimed warmup steps (default
10; absorbs kernel compilation and first-touch allocations).  One frame =
``sim_substeps`` x (``state.clear_forces``; ``viewer.apply_forces`` (no-op
under ViewerNull); ``CollisionPipeline.collide``; ``SolverXPBD.step``; state
swap).  Per scene, faithful to the example source:

- basic_pendulum: 10 substeps/frame at sim_dt = 1/1000 s.  The example records
  the loop at construction (cpu: APIC graph; metal: record/replay graph,
  warp-metal KEP-0001 M2.5) and ``step()`` replays it -- both benchmarked
  devices replay the same recorded loop, exactly as the example runs.
- basic_shapes: 5 substeps/frame at sim_dt = 1/500 s, including the
  last-substep contact-state copy.  Same graph-replay behavior as above.
- basic_conveyor: 10 substeps/frame at sim_dt = 1/1000 s, including the belt
  joint-state kernel, kinematic ``eval_fk``, and ``advance_time`` kernel.  The
  example guards graph capture to CUDA only, so cpu and metal both step
  eagerly -- exactly as the example runs.

Model build/finalize, viewer construction, kernel compilation, and warmup are
all excluded from the timed region.  Wall-clock timing is bracketed by
``wp.synchronize()`` so async metal launches are fully accounted for.

Single-thread CPU baseline
--------------------------
Upstream Warp (1.x) has no CPU launch thread pool: ``wp.launch`` on the
``cpu`` device executes serially on the calling thread (the historical
``wp.set_max_threads`` API no longer exists in Warp 1.19).  The harness
verifies this baseline explicitly: it fails loud if
``wp.config.enable_cpu_blocks`` is missing or enabled (the experimental
multi-fiber CPU-block mode), so the cpu column is always a genuine
single-thread measurement.  Newton itself spawns no solver threads.

Determinism
-----------
The scenes use the example defaults exactly as in test mode (SolverXPBD,
default solver settings, fixed initial conditions, no RNG anywhere).  Each
measurement is persisted as JSON (one file per scene x device) so reports are
reproducible and additive; the JSON records the full environment header
(python/warp/newton versions, git shas of both repos, backend env).
"""

from __future__ import annotations

import argparse as _argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OUT = (
    "/Volumes/ssd/main_link/work/newton/keps/"
    "0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon/phase2-bench.md"
)

WARP_METAL_REPO = "/Users/apx103/work/warp-metal"

# Measured-step defaults: warmup absorbs compile/first-touch; 100 timed frames
# matches the CI example test runner's num-frames for these scenes.
DEFAULT_WARMUP_FRAMES = 10
DEFAULT_FRAMES = 100


@dataclass(frozen=True)
class SceneSpec:
    """One benchmarked Newton example scene."""

    name: str
    module: str  # importable example module path
    timed_loop: str  # human description of the timed per-frame loop
    parser_additions: str  # the example-specific CLI args its __main__ adds

    def build_parser(self):
        """Mirror the example's __main__ parser (create_parser + example args)."""
        import newton.examples  # noqa: PLC0415 - deferred: env-dependent backend

        parser = newton.examples.create_parser()
        if self.name == "basic_pendulum":
            parser.add_argument(
                "--solver",
                type=str,
                choices=["xpbd", "kamino"],
                default="xpbd",
                help="Solver backend to use.",
            )
        elif self.name == "basic_shapes":
            parser.add_argument(
                "--solver",
                type=str,
                default="xpbd",
                choices=["vbd", "xpbd", "kamino"],
                help="Solver type: xpbd (default), vbd, or kamino",
            )
        elif self.name == "basic_conveyor":
            parser.add_argument(
                "--solver",
                type=str,
                choices=["xpbd", "vbd", "kamino"],
                default="xpbd",
                help="Solver backend to use.",
            )
            parser.add_argument(
                "--belt-speed",
                type=float,
                default=0.75,  # mirrors BELT_SPEED in example_basic_conveyor.py
                help="Conveyor tangential speed [m/s].",
            )
        else:  # pragma: no cover - registry is static
            raise RuntimeError(f"SceneSpec {self.name!r} has no parser builder")
        return parser


SCENES: dict[str, SceneSpec] = {
    "basic_pendulum": SceneSpec(
        name="basic_pendulum",
        module="newton.examples.basic.example_basic_pendulum",
        timed_loop=(
            "10 substeps/frame: clear_forces; CollisionPipeline.collide; "
            "SolverXPBD.step; state swap. Graph-replayed (cpu APIC / metal "
            "record-replay), exactly as example.step() runs."
        ),
        parser_additions="--solver xpbd",
    ),
    "basic_shapes": SceneSpec(
        name="basic_shapes",
        module="newton.examples.basic.example_basic_shapes",
        timed_loop=(
            "5 substeps/frame: clear_forces; CollisionPipeline.collide "
            "(incl. bunny mesh-vs-plane narrow phase); SolverXPBD.step "
            "(iterations=5); last-substep contact-state copy + assign. "
            "Graph-replayed, exactly as example.step() runs."
        ),
        parser_additions="--solver xpbd",
    ),
    "basic_conveyor": SceneSpec(
        name="basic_conveyor",
        module="newton.examples.basic.example_basic_conveyor",
        timed_loop=(
            "10 substeps/frame: clear_forces; belt joint-state kernel; "
            "kinematic eval_fk; CollisionPipeline.collide (mesh belt + rail "
            "meshes vs 18 bags); SolverXPBD.step; advance_time kernel. "
            "Eager stepping (the example guards capture to CUDA), exactly as "
            "example.step() runs."
        ),
        parser_additions="--solver xpbd",
    ),
}

RESULT_PREFIX = "phase2-bench"


# --------------------------------------------------------------------------------------
# Environment header (fail loud; nothing is invented)
# --------------------------------------------------------------------------------------


def _git_sha(repo_hint: Path) -> str:
    """Return the HEAD sha of the git repo containing *repo_hint*."""
    probe = subprocess.run(
        ["git", "-C", str(repo_hint), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(
            f"Cannot determine git HEAD for {repo_hint}: {probe.stderr.strip() or 'empty output'}. "
            "The report header must record real commit shas."
        )
    return probe.stdout.strip()


def _assert_backend(device: str, wp) -> None:
    """Fail loud unless the active warp module matches the requested device."""
    warp_file = str(getattr(wp, "__file__", ""))
    is_shim = "warp_metal" in (getattr(wp, "__name__", ""), warp_file)
    if device == "metal":
        if not is_shim:
            raise RuntimeError(
                "--device metal requested but the imported warp module is stock warp-lang "
                f"({wp.__file__}). Set NEWTON_WARP_BACKEND=metal before starting this process "
                "(the shim must alias warp at import time)."
            )
        devices = {d.alias for d in wp.get_devices()}
        if "metal" not in devices:
            raise RuntimeError(f"warp-metal shim is active but exposes no 'metal' device; devices: {sorted(devices)}")
    else:
        if is_shim:
            raise RuntimeError(
                "--device cpu requested but the warp-metal shim is active "
                f"({warp_file}). Unset NEWTON_WARP_BACKEND for the CPU baseline."
            )


def _assert_cpu_single_thread(wp) -> None:
    """Fail loud unless CPU kernel execution is the serial (single-thread) path."""
    if not hasattr(wp.config, "enable_cpu_blocks"):
        raise RuntimeError(
            "This warp build has no wp.config.enable_cpu_blocks; cannot verify the "
            "single-thread CPU baseline. Upgrade/inspect the warp frontend."
        )
    if wp.config.enable_cpu_blocks:
        raise RuntimeError(
            "wp.config.enable_cpu_blocks is True: CPU kernels would run with multiple "
            "logical threads per block, which violates the single-thread CPU baseline. "
            "Leave it at its default False (or unset whatever enabled it)."
        )


def collect_environment(device: str, wp) -> dict:
    """Version/commit header for the report. Fails loud on any gap."""
    import newton  # noqa: PLC0415 - deferred: env-dependent backend

    newton_root = Path(newton.__file__).resolve().parents[1]
    warp_metal_root = Path(wp.__file__).resolve().parents[1] if device == "metal" else None

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "newton_version": newton.__version__,
        "newton_repo": str(newton_root),
        "newton_sha": _git_sha(newton_root),
        "warp_version": wp.__version__,
        "warp_module": str(wp.__file__),
        "warp_metal_sha": _git_sha(warp_metal_root) if warp_metal_root is not None else None,
        "warp_metal_repo": WARP_METAL_REPO,
        "newton_warp_backend_env": os.environ.get("NEWTON_WARP_BACKEND", ""),
        "device": device,
        "device_name": wp.get_device(device).name,
    }
    return env


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


@dataclass
class Measurement:
    scene: str
    device: str
    frames: int
    warmup_frames: int
    total_s: float
    ms_per_frame: float
    ms_per_substep: float
    substeps_per_frame: int
    sim_dt: float
    frame_dt: float
    solver: str
    build_s: float  # example construction (excluded from timing; informational)
    warmup_s: float  # warmup loop (excluded from timing; informational)
    environment: dict = field(default_factory=dict)
    timestamp_utc: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, data: dict) -> Measurement:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def measure_scene(spec: SceneSpec, device: str, frames: int, warmup_frames: int) -> Measurement:
    """Build the example headlessly and time frames x example.step()."""
    # Deferred imports: main() imported newton first so NEWTON_WARP_BACKEND
    # aliasing is in place; `import warp` here resolves to the aliased backend.
    import warp as wp  # noqa: PLC0415 - deferred

    from newton.viewer import ViewerNull  # noqa: PLC0415 - deferred with warp

    _assert_backend(device, wp)
    if device == "cpu":
        _assert_cpu_single_thread(wp)
    wp.set_device(device)

    module = importlib.import_module(spec.module)
    example_cls = getattr(module, "Example", None)
    if example_cls is None:
        raise RuntimeError(f"{spec.module} has no Example class (fail loud)")

    parser = spec.build_parser()
    args = parser.parse_known_args([])[0]
    args.device = device

    build_start = time.perf_counter()
    viewer = ViewerNull(num_frames=warmup_frames + frames + 1)
    example = example_cls(viewer, args)
    build_s = time.perf_counter() - build_start

    for _ in range(warmup_frames):
        example.step()
    wp.synchronize()
    warmup_s = time.perf_counter() - build_start - build_s

    timed_start = time.perf_counter()
    for _ in range(frames):
        example.step()
    wp.synchronize()
    total_s = time.perf_counter() - timed_start

    substeps = int(example.sim_substeps)
    sim_dt = float(example.sim_dt)
    frame_dt = float(getattr(example, "frame_dt", sim_dt * substeps))
    solver = str(getattr(example, "solver_type", "xpbd"))

    return Measurement(
        scene=spec.name,
        device=device,
        frames=frames,
        warmup_frames=warmup_frames,
        total_s=total_s,
        ms_per_frame=total_s / frames * 1e3,
        ms_per_substep=total_s / (frames * substeps) * 1e3,
        substeps_per_frame=substeps,
        sim_dt=sim_dt,
        frame_dt=frame_dt,
        solver=solver,
        build_s=build_s,
        warmup_s=warmup_s,
        environment=collect_environment(device, wp),
        timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------------------
# Results persistence + report
# --------------------------------------------------------------------------------------


def result_path(results_dir: Path, scene: str, device: str) -> Path:
    return results_dir / f"{RESULT_PREFIX}-{scene}-{device}.json"


def save_measurement(results_dir: Path, m: Measurement) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = result_path(results_dir, m.scene, m.device)
    path.write_text(json.dumps(m.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_measurements(results_dir: Path) -> dict[tuple[str, str], Measurement]:
    """Load every saved measurement. Fail loud on unreadable/corrupt files."""
    out: dict[tuple[str, str], Measurement] = {}
    if not results_dir.is_dir():
        return out
    for path in sorted(results_dir.glob(f"{RESULT_PREFIX}-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            m = Measurement.from_json(data)
        except (json.JSONDecodeError, TypeError, ValueError) as err:
            raise RuntimeError(f"Corrupt benchmark result {path}: {err}") from err
        if m.scene not in SCENES or m.device not in ("cpu", "metal"):
            raise RuntimeError(f"Unrecognized benchmark result {path}: scene={m.scene!r} device={m.device!r}")
        out[(m.scene, m.device)] = m
    return out


def _env_table_rows(measurements: dict[tuple[str, str], Measurement]) -> list[tuple[str, str]]:
    """Merge environment headers across measurements (they must agree)."""
    envs = [m.environment for m in measurements.values() if m.environment]
    if not envs:
        return []
    keys = [
        "python",
        "platform",
        "newton_version",
        "newton_sha",
        "warp_version",
        "warp_module",
        "warp_metal_sha",
        "newton_warp_backend_env",
    ]
    rows: list[tuple[str, str]] = []
    for key in keys:
        values = {str(e.get(key)) for e in envs}
        if len(values) > 1:
            rows.append((key, " | ".join(sorted(values)) + "  (differs between runs)"))
        else:
            rows.append((key, values.pop()))
    return rows


def format_speedup(cpu: Measurement | None, other: Measurement | None) -> str:
    """Speedup = cpu time / other time, from measured pairs only."""
    if cpu is None or other is None:
        return "n/a"
    if cpu.frames != other.frames or cpu.warmup_frames != other.warmup_frames:
        return "n/a (frame counts differ)"
    if other.total_s <= 0.0 or cpu.total_s <= 0.0:
        return "n/a (non-positive timing)"
    return f"{cpu.total_s / other.total_s:.2f}x"


def build_report(measurements: dict[tuple[str, str], Measurement]) -> str:
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# KEP-0001 M2.6 performance baseline (phase2-bench)")
    lines.append("")
    lines.append(f"Generated: {now} by `warp-metal/tools/bench_phase2.py`")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for key, value in _env_table_rows(measurements):
        lines.append(f"| {key} | `{value}` |")
    if not measurements:
        lines.append("| (none) | no measurements saved yet |")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- One process = one device (GPU exclusivity law); each row below is a separate solo run.")
    lines.append(
        "- Timed region: `--frames` repetitions of `example.step()` (one frame = `sim_substeps` solver+collision "
        "iterations) after `--warmup-frames` untimed warmup steps. Wall clock bracketed by `wp.synchronize()`."
    )
    lines.append("- Excluded from timing: model build/finalize, viewer setup, kernel compilation, warmup.")
    lines.append(
        "- CPU baseline: single-threaded. Upstream Warp 1.x executes `cpu` launches serially on the calling "
        "thread; the harness fails loud if `wp.config.enable_cpu_blocks` is enabled."
    )
    lines.append(
        "- Determinism: example defaults exactly as in test mode (SolverXPBD, fixed initial conditions, no RNG). "
        "Measurements persisted as JSON next to this report; speedups are computed from measured pairs only."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    if not measurements:
        lines.append("No measurements saved yet. Run e.g. `bench_phase2.py --device cpu` first.")
    else:
        lines.append("| scene | device | solver | frames | warmup | total s | ms/frame | ms/substep | speedup vs cpu |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for scene in SCENES:
            cpu = measurements.get((scene, "cpu"))
            for device in ("cpu", "metal"):
                m = measurements.get((scene, device))
                if m is None:
                    continue
                speedup = "1.00x (baseline)" if device == "cpu" else format_speedup(cpu, m)
                lines.append(
                    f"| {scene} | {device} | {m.solver} | {m.frames} | {m.warmup_frames} "
                    f"| {m.total_s:.3f} | {m.ms_per_frame:.3f} | {m.ms_per_substep:.4f} | {speedup} |"
                )
        lines.append("")
        lines.append("### Timed loop per scene")
        lines.append("")
        for scene, spec in SCENES.items():
            if any(k[0] == scene for k in measurements):
                lines.append(f"- **{scene}** (`newton.examples.{scene}`): {spec.timed_loop}")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /Volumes/ssd/main_link/work/newton")
    lines.append(f"uv run python {WARP_METAL_REPO}/tools/bench_phase2.py --device cpu")
    for scene in SCENES:
        lines.append(
            f"NEWTON_WARP_BACKEND=metal uv run python {WARP_METAL_REPO}/tools/bench_phase2.py "
            f"--device metal --scene {scene}"
        )
    lines.append(f"uv run python {WARP_METAL_REPO}/tools/bench_phase2.py --report-only")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def make_arg_parser() -> _argparse.ArgumentParser:
    parser = _argparse.ArgumentParser(
        prog="bench_phase2.py",
        description="KEP-0001 M2.6 performance baseline: Newton scenes on cpu vs metal (one process = one device).",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, choices=["cpu", "metal"], default=None, help="Device to measure on.")
    parser.add_argument(
        "--scene",
        type=str,
        action="append",
        choices=sorted(SCENES),
        default=None,
        help="Scene to measure (repeatable). Default: all scenes.",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Timed frames per scene.")
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES, help="Untimed warmup frames.")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="Markdown report output path.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for per-run JSON measurements (default: alongside --out).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip timing; rebuild the markdown report from saved measurements.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_arg_parser().parse_args(argv)

    if args.report_only:
        if args.device is not None:
            make_arg_parser().error("--report-only must not be combined with --device (reporting touches no GPU)")
    else:
        if args.device is None:
            make_arg_parser().error("provide --device cpu or --device metal (one process = one device)")
        if args.frames < 1:
            make_arg_parser().error("--frames must be >= 1")
        if args.warmup_frames < 0:
            make_arg_parser().error("--warmup-frames must be >= 0")

    out_path: Path = args.out.resolve()
    results_dir: Path = args.results_dir.resolve() if args.results_dir else out_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        scenes = args.scene if args.scene else sorted(SCENES)
        # Importing newton first honors NEWTON_WARP_BACKEND (the shim aliases warp
        # in sys.modules); measure_scene's `import warp` then resolves to the backend.
        import newton  # noqa: F401, PLC0415 - env-dependent warp aliasing happens here

        for scene in scenes:
            spec = SCENES[scene]
            print(
                f"[bench_phase2] measuring {scene} on {args.device} "
                f"(warmup={args.warmup_frames}, frames={args.frames}) ...",
                flush=True,
            )
            m = measure_scene(spec, args.device, args.frames, args.warmup_frames)
            path = save_measurement(results_dir, m)
            print(
                f"[bench_phase2]   {scene}/{args.device}: total {m.total_s:.3f} s, "
                f"{m.ms_per_frame:.3f} ms/frame, {m.ms_per_substep:.4f} ms/substep -> {path}",
                flush=True,
            )

    report = build_report(load_measurements(results_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"[bench_phase2] report written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
