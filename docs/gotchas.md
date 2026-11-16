# warp-metal gotchas

Debugging field notes from building warp-metal (KEP-0001, Phases 0-4).
Everything here was verified empirically and each item cost at least one
debugging session. Read the sections that match the layer you are touching
before trusting a green run — several of these fail **silently**.

## Metal runtime

- **Residency for address-referenced buffers — silent zero reads.** A buffer
  referenced only through a raw device address (host-built mesh table, BVH
  nodes) never appears in an encoder binding list, so the driver owes it no
  residency and reads return ZEROS on Apple Silicon. Fix lives in
  `native/warp_metal.mm`: every allocation joins an `MTLResidencySet`, marked
  used per command buffer (`[cb useResidencySet:]`, macOS 15+). Symptom
  signature: table metadata reads correct (host-copied bytes) while
  `points.data` dereferences read zero; binding the same buffer through a
  plain kernel arg makes reads work. Toy probes with small meshes mask this
  (small buffers stay pinned) — Newton examples' `wp.Mesh` creation inside
  `finalize()` is where it bites.
- **Metal diagnostics buffer clips.** Warnings flood the buffer and clip the
  real error, so every MSL declaration needs `(void)(name);` suppression and
  hoisted declarations need `{}` init (kills `-Wsometimes-uninitialized`). A
  compile failure whose visible diagnostics are only warnings means the real
  error was clipped. Recovery recipe: set `WARP_METAL_DUMP_MSL=<dir>` to write
  each kernel's exact pre-compile MSL, then compile the dump with a small
  Objective-C `newLibraryWithSource` harness to get full diagnostics (there
  is no standalone `metal` CLI on the dev machine and no pyobjc in the venv).
- **Prelude is a shared compile unit.** One bad declaration sinks EVERY
  kernel launch. New prelude MSL must use Metal builtin names, not C names
  (`fma`, not `fmaf`), and must contain no fp64 anywhere. Upstream fp64
  robustness fallbacks (e.g. woop) degrade to fp32 with a comment.
- **Plain-launch tile semantics.** Corpus/trajectory goldens are generated
  with CPU warp, which forces `block_dim=1`; warp-metal plain launches lower
  ALL tile ops per-thread scalar regardless of the caller's `block_dim`.
  Threadgroup-backed tile lowering only exists under `launch_tiled`.
  Baked-block-dim vs auto-threadgroup mismatch shows up as
  exactly-the-first-few-slots-written.
- **Dispatch watchdog.** `WARP_METAL_DISPATCH_TIMEOUT_S` (default 120 s,
  `0` disables) bounds GPU-fault waits; GPU faults fail loud with a named
  `MetalError` instead of hanging forever in `waitUntilCompleted`.

## GPU and resource discipline

- **Why the solo law exists.** Apple Silicon's GPU is shared with the
  display: unified memory, and the iGPU drives WindowServer. A sustained
  max-duty Metal workload starves the UI — on the dev machine this once made
  macOS unusable and forced a hard shutdown (2026-11-16). Treat an unpaced
  GPU duty cycle over a few minutes as a user-facing hazard, not a
  performance opportunity.
- **Solo runs only.** Two or more processes launching Metal work concurrently
  yields `kIOGPUCommandBufferCallbackErrorInnocentVictim` discards and even
  spurious deterministic-looking import errors. All acceptance evidence must
  come from strictly serial solo runs. This includes the unittest suite:
  its deferred-micro bench DISPATCHES Metal — never run it concurrently with
  a Metal acceptance run.
- **Pacing knobs.** `tools/smoke_examples.py` paces by default:
  `SMOKE_PACING_S` (default 10 s sleep after every example) plus a 30 s
  cooldown after every 10 examples. `WARP_METAL_DISPATCH_TIMEOUT_S` bounds
  fault waits. Set pacing to 0 only when the user explicitly asks.

## Fixtures and goldens (tools/corpus_builders.py, tools/conform.py)

- **Fixture domains must be in-bounds over the kernel's FULL index domain.**
  A fixture staged 4 dofs/articulation against a kernel compiled for
  dofs=12 reads past `joint_S_s`/`drive_dof_K`. OOB reads pass on CPU (the
  allocator fills the slack) but return garbage on Metal — a row can
  "pass" while comparing against uninitialized memory (the crba L_group
  incident: diagonal pivots collapsed to `sqrt(armature)`; signature was
  only-per-articulation-diagonal-pivots-wrong, off-diagonals clean, masked
  articulation bitwise zero, two launches bitwise identical). Print the
  fixture arrays and check every array the kernel indexes against its
  domain before trusting a green run.
- **Stride-discriminant trap.** A shape with a middle dim of 1 and pinned mid
  indices (e.g. `(2,1,4)` with `j=1`) sends half the writes out of bounds
  into allocator slack — silently wrong, no crash. Unique-cell writes (each
  cell touched exactly once) are the only deterministic stride test;
  same-cell integer adds test commutativity. Newton host code can also
  create OOB-prone shapes implicitly: derive expected values from the same
  indexing math as the kernel (loop-and-assign), not hand arithmetic.
- **Symmetric fixtures hide fill-mode triangle reads.** Upstream
  `scalar_cholesky_impl` reads the fill-mode triangle of A literally; a
  helper that transposes A-reads AND call sites that pass swapped args
  double-swaps onto the opposite triangle — invisible with symmetric `_spd`
  fixtures. Signature: factor diagonal equals sqrt of the RAW input
  diagonal where the golden shows a coupled value. Regression pin =
  upper-triangle-filled fixture (numpy ref via `chol(A.T).T`).
- **Cancellation knife-edges.** A fixture output that is the difference of
  two O(10) terms canceling to near zero amplifies the two fp32 codegens'
  few-ulp spread past tolerance with NEITHER side wrong (Metal may be closer
  to fp64). Fix the fixture, not the checker: grid-search draw scale +
  fixed offsets for worst-case `r = |a-b|/(|a|+|b|)` >= 0.2, regen golden,
  npz-diff to prove only that row changed.
- **Golden provenance is not git-attestable.** `corpus_golden.npz` and
  `corpus_golden_meta.json` are gitignored; a "deterministic regression" can
  be the npz changing (allocator history, FORCE regen), not the code. Regen
  MUST use the Newton venv python — the flag only pins the phase, the venv
  decides what `warp` resolves to:

  ```bash
  env -u NEWTON_WARP_BACKEND CORPUS_FORCE_GOLDEN=1 \
    <newton-checkout>/.venv/bin/python -I \
    ~/work/warp-metal/tools/conform.py --layer corpus
  ```

  Running it under the warp-metal venv produces a self-comparing golden
  (warp_metal vs warp_metal) and silently pollutes the npz. The compare
  phase auto-spawns the Newton python only when the npz is absent; FORCE is
  for overwriting. A row failing at HEAD may have NEVER passed since the
  last regen — before blaming recent commits, run the row at the flip
  commit (git worktree) against the same npz.
- **Two-venv corpus seam.** Corpus rows that `import mujoco_warp` must run
  their compare from the **Newton venv with the shim active**
  (`NEWTON_WARP_BACKEND=metal uv run python ~/work/warp-metal/tools/conform.py
  --layer corpus` from the Newton checkout). The warp-metal standalone venv
  deliberately has no warp-lang/mujoco_warp (M0.2 discipline), so those rows
  read `ModuleNotFoundError` there — an environment seam, NOT a corpus
  regression. The surface layer runs from the standalone venv.
- **Reproduce with the compare's exact assert.** A single-row probe with a
  crude `|d|>1e-4` check can "pass" while the compare's per-element
  `atol + rtol*|want|` fails. Always re-verify with the real assertion.
- **Trust the run's stdout, not the report file.** `conform.py` rewrites
  report sections in place; a crashed compare leaves the previous phase's
  PASS numbers in the file (one false-green incident, 2026-11-14). Also:
  `DEFAULT_REPORT` is the keps-dir `conform-report.md` — do not pass
  `--report` to a scratch path for acceptance evidence.
- **`trajectory_golden.npz` is TRACKED** (unlike the corpus npz) — regen
  drift is commit-worthy. `basic_pendulum` is bitwise stable;
  `basic_shapes` shows chaotic contact-tail drift (max ~1e-3, cap 0.15)
  when solver numerics move. Refresh it as its own commit with the drift
  quantified in the message.

## Frontend and shim semantics

- **`warp_metal/__init__` self-installs the `warp` alias at import**
  (`sys.modules.setdefault`), so `install_as_warp()` takes the
  already-installed early return. Any activation side effect must run on
  BOTH paths (see `_activate_preferred_device` in `shim.py`).
- **`wp.func_grad` must wrap the grad as a Function** (same key,
  `custom_reverse_mode=True`) mirroring upstream — storing the raw function
  breaks caller-side parse (`custom_grad_func.adj` access), e.g. featherstone
  `eval_dense_cholesky_batched`.
- **Copy upstream semantics verbatim.** `wp_sign(0) = 1`; `acos`/`asin` need
  domain clamping on the GPU (FMA contraction produces NaN where the CPU is
  fine); `quat_rpy = qz*qy*qx` half-angle formulas; `wp.trace` in Newton is
  the MATRIX TRACE (sum of diagonal), not a debug print; `arr[i]` on
  ndim>1 returns an upstream array VIEW sharing the buffer (only 1-D
  indexing yields the element value) — a numpy row here is a shim divergence.
- **Prelude overloads are need-driven.** New missing overloads surface one
  class at a time; keep peeling via the probe-patch-dump loop (patch
  `native.library_create`, dump MSL, compile standalone).
- **`wp.array(ptr=...)` under the shim is a same-buffer alias** — no
  `buffer_from_ptr` needed. `WarpArray.ptr` is the unified shared-storage
  address; a base-ptr registry + a view sharing the source MTLBuffer covers
  reinterpret sites (e.g. contact.frame (n,9)f32 -> (n,3)vec3). Foreign
  pointers keep the Phase-2 fail-loud.
- **Struct types that live only on SSA locals/function signatures** (no
  kernel param carries the type) emit no MSL def — "unknown type name". Emit
  struct-typed adjoint variables' defs AFTER callee translation (emitting
  first steals the dedup slot from the param-split forms and hollows the
  callee signature). Value-context array fields emit as `wp_val_array_*`
  (ptr + meta) pairs; any closure that actually moves arrays through struct
  fields still fails loud.

## Static-analysis blind spots

- **Runtime dtype instantiation evades AST-level scans.** A kernel is fp64
  only because it is instantiated with `wp.array[wp.mat64]` at runtime, or a
  builtin is used cross-package/outside kernel sources — the AST symbol scan
  sees no fp64 and reports coverage the corpus run then refutes. Static
  scans are a prediction; the corpus run is the verdict.
- **Atomic-usage detection must scan MSL function BODIES only.**
  `ensure_vec_def` emits atomic helper DEFINITIONS into vector-dtype struct
  declarations unconditionally — whole-unit text scans false-positive on
  every float-vec kernel. The reliable criterion is `conform.py
  _kernel_atomic_sites`: scan `unit.kernel.lines` plus every transitive
  `unit.functions` body for "atomic" (struct text is separate,
  `unit.structs`).
- **Prelude `#define` mentions trip string-match test pins.** Adding an
  inert `#define WP_TILE_SYNC threadgroup_barrier(...)` made plain-launch
  tests fail `assertNotIn("threadgroup_barrier", src)` pins — an inert macro
  mention is not execution. Pins must strip `#define` lines before asserting
  (`_src_without_tile_sync_define` in `tests/test_tile_ops.py`); real
  barrier statements still fail the pin.
- **Eager mixed compute+blit fills (M2.6 disposition, D2).** Encoding eager
  zero_/fill_/copy in-stream regressed the Newton conveyor ~35x while graph
  batches stayed fine and the isolated microbench exonerated the pipeline in
  every configuration — the trigger is a conveyor-specific amplification
  (lead: flush cadence x many `.numpy()` reads). Final discipline: eager
  default = ordered HOST fills (byte-verified); GPU encodings live behind
  `WARP_METAL_EAGER_GPU_FILLS=1`; batch-replay keeps its blits (measured
  fine). If conveyor perf regresses again, check encoder mix AND
  flush-count structure first.
