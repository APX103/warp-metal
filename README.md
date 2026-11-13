# warp-metal

Warp-compatible `wp.*` DSL with a native Apple Metal (MSL) backend, built so
[Newton](https://github.com/newton-physics/newton) can run on Apple Silicon
without any NVIDIA dependency.

## Layout

- `src/warp_metal/frontend/` — vendored Warp compiler frontend (Apache-2.0,
  frozen at warp-lang 1.19.0; nvrtc/ptx/cuda paths stripped)
- `src/warp_metal/msl/` — MSL code generator (Warp AST → Metal Shading Language)
- `src/warp_metal/runtime/` — native Metal runtime (MTLBuffer with shared
  storage, PSO cache, command buffer batching) + LLVM CPU fallback
- `src/warp_metal/compat.py` — `import warp_metal as wp` entry point, with a
  vendored-version sentinel

## Status

Phase 0 spike complete (2026-11-13): vendored frontend (warp 1.19.0) with
extracted registry, hand-written MSL emitter for the 10 spike kernels,
native Metal runtime with `wp.launch(device="metal")` end-to-end, and 10/10
numeric alignment against the warp CPU(LLVM) backend (`phase0-report.md`
in the Newton KEP dir). Phase 1 (rigid-body MVP: full type system, host
runtime, per-frequency builtins, primitive collision, Newton shim) is
next — plan of record: ROADMAP.md in the Newton working repo under
`keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon/`.

## License

Apache-2.0. Portions vendored from NVIDIA warp (Apache-2.0); see `NOTICE`.
