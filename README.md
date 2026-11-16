# warp-metal

Warp-compatible `wp.*` DSL with a native Apple Metal (MSL) backend, built so
[Newton](https://github.com/newton-physics/newton) runs on Apple Silicon with
zero NVIDIA dependency. It replaces `warp-lang` (which ships CUDA-only GPU
paths and is distributed from NVIDIA's PyPI index) at the package level: the
Warp compiler frontend is vendored (Apache-2.0, frozen at warp-lang 1.19.0,
with a hard version sentinel at import), and everything behind it — MSL code
generation and the Metal runtime — is original to this repo. Warp's CPU path
is re-implemented as an LLVM-free numpy/host fallback and per-kernel
`device="cpu"` routing target.

KEP-0001 (the project this implements) is closed: Phases 0–3 complete,
plus a Phase-4 stretch (example smoke matrix, repo packaging). This README,
`docs/gotchas.md`, and the `v0.1.0` tag are the entry points.

## Install

Requirements: macOS on Apple Silicon (Metal), Python >= 3.10, numpy, and the
Xcode command-line toolchain (the native runtime dylib compiles itself with
`clang++ -framework Metal` on first import).

### warp-metal (standalone library)

```bash
git checkout <this repo>
uv venv && uv pip install --editable .
.venv/bin/python -c "import warp_metal as wp; print(wp.zeros(4, dtype=wp.float32).numpy())"
```

Plain `pip install --editable .` works identically. `import warp_metal as wp`
is the upstream-shaped surface; `warp_metal.install_as_warp()` aliases it as
`warp` in-process.

### Newton on Metal (the shim)

Newton consumes warp-metal through a fork branch and a machine-local path
dependency:

1. Check out Newton at fork branch `kep-0001-metal-shim` (the branch carries
   the `NEWTON_WARP_BACKEND=metal` switch in `newton/__init__.py`, the fp32
   inertia precision divergences, and the metal test parametrization).
2. Its `pyproject.toml` defines a `metal` extra wired to this checkout:

   ```toml
   [project.optional-dependencies]
   metal = ["warp-metal"]

   [tool.uv.sources]
   warp-metal = { path = "/Users/apx103/work/warp-metal", editable = true }
   ```

3. Sync and activate:

   ```bash
   uv sync --extra metal --extra dev
   ```

With `NEWTON_WARP_BACKEND=metal` set, every `import warp` inside Newton
resolves to warp_metal; unset (or any other value) keeps stock warp-lang.
The env var is the single switch — no code changes on the Newton side.

## Quickstart

```bash
cd <newton checkout>            # kep-0001-metal-shim branch
uv sync --extra metal --extra dev
NEWTON_WARP_BACKEND=metal uv run -m newton.examples basic_pendulum --device metal --viewer null --test
```

`--test` runs the example's embedded success criteria headlessly and exits 0
on pass. Drop `--test` (and pick a viewer) for interactive runs.

## Capability surface

| Area | Status on `device="metal"` |
|---|---|
| Kernel model (`@wp.kernel` / `@wp.func` / `@wp.struct` / `@wp.overload`, generics) | works — vendored warp 1.19.0 frontend, MSL emitter over the Adjoint SSA |
| Rigid bodies | works — XPBD and semi-implicit solvers, primitive + mesh collision, steel tiled solver, body/joint/contact kernels |
| BVH / mesh queries | works — SAH host build, whole-tree and grouped queries (aabb/ray/sphere/capsule), `mesh_query_*`, `mesh_get` |
| Tile ops | works — `tile_load/store/map/atomic_*/cholesky/solve/transpose/scatter/extract/...` via a threadgroup model (32 KB budget, uniformity-checked); `launch_tiled` |
| Capture / replay | works — `ScopedCapture` records command buffers, `capture_launch` replays (batched and per-record modes), whole-frame graph replay |
| mujoco_warp pipeline | works — full forward/constraint/solver stack; `SolverMuJoCo(device="metal")` (single robot) end-to-end |
| Atomics | works — i32/i64/u32/u64 add/min/max + vec3f add/sub, exchange/CAS |
| Arrays | works — `MTLBuffer` shared storage, zero-copy numpy views, 1–4 D, `ptr=` aliasing |
| fp64 kernel types | **fail loud** — Metal exposes no fp64; the steel friction-patch fp64 accumulation is the accepted exemption (B1) |
| `wp.Volume` / NanoVDB | not implemented (Non-Goal) |
| Adjoint / Tape (`requires_grad`, pinned allocs) | fail loud by design (Non-Goal) |
| `warp.fem` / `warp.sparse` / kamino | not ported (Non-Goal; import fails loud) |
| Texture sampling | host staging only — `wp.texture_sample` MSL lowering absent |
| HashGrid device surface | fail loud (`wp_hash_grid_create_device` path missing); blocks particle-model `builder.finalize()` |
| OptiX (RTX viewer, raytrace camera sensors) | permanently cut — no Metal equivalent |
| `wp.indexedarray` fancy indexing, `wp.array(strides=)`, `array.list()`, `wp::extract` over slices | known open gaps, each fails loud with a named error |
| Big-batch RL training throughput vs datacenter GPUs | out of scope (hardware class; run large experiments on rented CUDA) |

Design-level Non-Goals (KEP-0001): automatic differentiation/adjoint codegen,
`warp.fem`/`warp.sparse`, NanoVDB volumes, OptiX, and RTX-4090-class batch
throughput. `mujoco_warp` compatibility was a planned Phase-3 extension and
is now part of the working surface above.

## Testing

The acceptance gate is a five-piece battery; every number must come from a
**strictly serial, solo run** (see [Contributing](#contributing) for why).
From the warp-metal checkout with its venv active:

```bash
.venv/bin/python -m unittest discover -s tests         # 394 tests (dispatches Metal)
.venv/bin/python tools/phase0_align.py --compare       # 10/10 numeric alignment
.venv/bin/python -I tools/conform.py --layer surface   # 247 ok / 0 mismatch / 12 missing
```

From the Newton checkout (shim active — needed for the mujoco_warp corpus
rows and the trajectory layer):

```bash
NEWTON_WARP_BACKEND=metal uv run python ~/work/warp-metal/tools/conform.py --layer corpus      # 46 / 0 / 4
NEWTON_WARP_BACKEND=metal uv run python ~/work/warp-metal/tools/conform.py --layer trajectory  # 3/3
```

Golden references regenerate CPU-side (real warp in the Newton venv, no GPU):

```bash
env -u NEWTON_WARP_BACKEND CORPUS_FORCE_GOLDEN=1 /Volumes/ssd/main_link/work/newton/.venv/bin/python \
    -I ~/work/warp-metal/tools/conform.py --layer corpus
```

Corpus rows carry explicit skip reasons instead of silently narrowing checks;
tolerance is rtol 1e-5 / atol 1e-6 (integer arrays bitwise). `docs/gotchas.md`
documents the fixture/golden discipline these tools enforce.

## Performance

Large scenes clear the >=3x gate: `pendulum_batch` at 4096 environments runs
**6.07x** faster on metal than Newton's single-threaded CPU path. Small
scenes are dispatch-overhead dominated — a handful of tiny kernels per frame
pay more in command-buffer encoding than they win on the GPU, and can be
slower than CPU (bench evidence and the small/large scene split:
`phase2-bench.md` in the Newton working repo's
`keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon/`
directory).

## Contributing

- **Solo GPU discipline.** Apple Silicon's GPU is shared with the display:
  unified memory, and the iGPU drives WindowServer. A sustained max-duty
  Metal workload starves the UI and can make the machine unusable (this
  forced a hard shutdown once). Run GPU pieces serially, one Metal process
  at a time, never alongside another Metal run. Long batches must pace
  themselves — see `SMOKE_PACING_S` in `tools/smoke_examples.py` (default 10 s
  per example plus a 30 s cooldown every 10) as the precedent, and
  `WARP_METAL_DISPATCH_TIMEOUT_S` (default 120 s) which turns GPU faults into
  named errors instead of hangs.
- **The battery above is the acceptance gate.** A change is done when the
  five numbers hold (ratchets only go up); report what you measured, from the
  run's own stdout — not a stale report section.
- **Two-venv discipline.** The warp-metal venv deliberately has no warp-lang
  (the shim must never silently meet itself); golden references and the
  mujoco_warp-dependent rows run from the Newton venv. `docs/gotchas.md`
  has the exact seam.
- **Lint.** Match the surrounding style; `uvx ruff check` on touched files is
  the working baseline (no repo-wide formatter gate).

## Layout

- `src/warp_metal/frontend/` — vendored Warp compiler frontend (Apache-2.0,
  frozen at warp-lang 1.19.0; nvrtc/ptx/cuda paths stripped; `_vendored_version.txt`
  drives the import-time sentinel)
- `src/warp_metal/msl/` — MSL code generator (Warp AST/SSA → Metal Shading Language)
- `src/warp_metal/runtime/` — native Metal runtime (`MTLBuffer` shared storage,
  residency sets, PSO cache, command-buffer batching) + host CPU fallback
- `src/warp_metal/native/` — ObjC++ runtime (`warp_metal.mm`), self-compiling dylib
- `src/warp_metal/compat.py` — `import warp_metal as wp` entry point, version sentinel
- `src/warp_metal/shim.py` — `install_as_warp()` / `NEWTON_WARP_BACKEND` activation
- `tools/` — conformance harness (`conform.py`, `corpus_builders.py`,
  `phase0_align.py`), bench and smoke runners
- `docs/gotchas.md` — debugging field notes; read this before touching the
  runtime, the emitter, or the corpus fixtures

## License

Apache-2.0. Portions vendored from NVIDIA warp (Apache-2.0); see `NOTICE`.
