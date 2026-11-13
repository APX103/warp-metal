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

Phase 0 spike (KEP-0001). The plan of record (ROADMAP.md) and the KEP design
live in the Newton working repo under
`keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon/`.

`import warp_metal` works; the `wp.*` API surface lands incrementally over
M0.2–M0.5. Nothing here is usable for simulation yet.

## License

Apache-2.0. Portions vendored from NVIDIA warp (Apache-2.0); see `NOTICE`.
