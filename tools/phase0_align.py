# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Phase 0 numeric-alignment harness, CPU side (KEP-0001 M0.5).

Run under the Newton venv (has warp-lang 1.19.0, LLVM CPU backend):

    uv run --directory <newton> python ~/work/warp-metal/tools/phase0_align.py --golden

Writes reference outputs of the same 10 spike-kernel computations on
device="cpu" to a .npz next to this file. The --compare phase (any venv
with warp-metal) runs the Metal side and emits the alignment report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import paths

REPO = Path(__file__).resolve().parent.parent
GOLDEN_PATH = Path(__file__).parent / "phase0_golden.npz"
# plan-of-record lives in the newton keps dir; fall back to the repo root
_KEP_DIR = paths.newton_repo() / "keps/0001-standalone-warp-metal-library-with-msl-backend-for-newton-on-apple-silicon"
REPORT_PATH = _KEP_DIR / "phase0-report.md" if _KEP_DIR.exists() else REPO / "phase0-report.md"

SEED = 0
N = 256


def _inputs():
    rng = np.random.default_rng(SEED)
    inv = lambda a: 1.0 / (a + 2.0)  # keep matrices well-conditioned
    return {
        "f32_a": rng.standard_normal(N).astype(np.float32),
        "f32_b": rng.standard_normal(N).astype(np.float32),
        "phase": np.linspace(0, 6.28, N, dtype=np.float32),
        "clip_in": np.linspace(-2, 2, N, dtype=np.float32),
        "mat33": inv(rng.standard_normal((N, 9))).astype(np.float32),
        "vec3_a": rng.standard_normal((N, 3)).astype(np.float32),
        "vec3_b": rng.standard_normal((N, 3)).astype(np.float32),
        "idx": rng.integers(0, 8, N).astype(np.int32),
    }


# ---- warp CPU kernels (mirror msl/spike_kernels.py 1:1) ----

def run_cpu_golden():
    import warp as wp  # NVIDIA warp, CPU(LLVM) backend — reference

    inp = _inputs()
    n = N
    outs = {}

    a = wp.array(inp["f32_a"], dtype=wp.float32, device="cpu")
    b = wp.array(inp["f32_b"], dtype=wp.float32, device="cpu")
    phase = wp.array(inp["phase"], dtype=wp.float32, device="cpu")
    clip_in = wp.array(inp["clip_in"], dtype=wp.float32, device="cpu")
    mat = wp.array(inp["mat33"], dtype=wp.mat33f, device="cpu")
    va = wp.array(inp["vec3_a"], dtype=wp.vec3f, device="cpu")
    vb = wp.array(inp["vec3_b"], dtype=wp.vec3f, device="cpu")
    idx = wp.array(inp["idx"], dtype=wp.int32, device="cpu")

    def out_f32():
        return wp.empty(n, dtype=wp.float32, device="cpu")

    def out_vec3():
        return wp.empty(n, dtype=wp.vec3f, device="cpu")

    # 1 copy_scale
    out = out_f32()

    @wp.kernel
    def k1(a: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32), s: wp.float32):
        i = wp.tid()
        x = a[i]
        o[i] = x * s

    wp.launch(k1, dim=n, inputs=[a, out, 2.5], device="cpu")
    outs["k_copy_scale"] = out.numpy().copy()

    # 2 saxpy3
    out = out_vec3()

    @wp.kernel
    def k2(a: wp.array(dtype=wp.vec3f), bb: wp.array(dtype=wp.vec3f), o: wp.array(dtype=wp.vec3f), s: wp.float32):
        i = wp.tid()
        o[i] = a[i] * s + bb[i]

    wp.launch(k2, dim=n, inputs=[va, vb, out, 2.0], device="cpu")
    outs["k_saxpy3"] = out.numpy().copy()

    # 3 dot
    out = out_f32()

    @wp.kernel
    def k3(a: wp.array(dtype=wp.vec3f), bb: wp.array(dtype=wp.vec3f), o: wp.array(dtype=wp.float32)):
        i = wp.tid()
        o[i] = wp.dot(a[i], bb[i])

    wp.launch(k3, dim=n, inputs=[va, vb, out], device="cpu")
    outs["k_dot"] = out.numpy().copy()

    # 4 cross_normalize (out only; l is dead)
    out = out_vec3()

    @wp.kernel
    def k4(a: wp.array(dtype=wp.vec3f), bb: wp.array(dtype=wp.vec3f), o: wp.array(dtype=wp.vec3f)):
        i = wp.tid()
        c = wp.cross(a[i], bb[i])
        o[i] = wp.normalize(c)
        l = wp.length(c)

    wp.launch(k4, dim=n, inputs=[va, vb, out], device="cpu")
    outs["k_cross_normalize"] = out.numpy().copy()

    # 5 trig
    out = out_f32()

    @wp.kernel
    def k5(p: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
        i = wp.tid()
        s = wp.sin(p[i])
        c = wp.cos(p[i])
        o[i] = s * s + c * c

    wp.launch(k5, dim=n, inputs=[phase, out], device="cpu")
    outs["k_trig"] = out.numpy().copy()

    # 6 select_clip
    out = out_f32()

    @wp.kernel
    def k6(x: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32), lo: wp.float32, hi: wp.float32):
        i = wp.tid()
        v = x[i]
        o[i] = wp.where(v < lo, lo, wp.where(v > hi, hi, v))

    wp.launch(k6, dim=n, inputs=[clip_in, out, -1.0, 1.0], device="cpu")
    outs["k_select_clip"] = out.numpy().copy()

    # 7 matvec
    out = out_vec3()

    @wp.kernel
    def k7(m: wp.array(dtype=wp.mat33f), v: wp.array(dtype=wp.vec3f), o: wp.array(dtype=wp.vec3f)):
        i = wp.tid()
        o[i] = m[i] @ v[i]

    wp.launch(k7, dim=n, inputs=[mat, va, out], device="cpu")
    outs["k_matvec"] = out.numpy().copy()

    # 8 matinv_apply
    out = out_vec3()

    @wp.kernel
    def k8(m: wp.array(dtype=wp.mat33f), v: wp.array(dtype=wp.vec3f), o: wp.array(dtype=wp.vec3f)):
        i = wp.tid()
        minv = wp.inverse(m[i])
        o[i] = minv @ v[i]

    wp.launch(k8, dim=n, inputs=[mat, va, out], device="cpu")
    outs["k_matinv_apply"] = out.numpy().copy()

    # 9 atomic_accum
    acc = wp.empty(8, dtype=wp.float32, device="cpu")
    acc.zero_()

    @wp.kernel
    def k9(ac: wp.array(dtype=wp.float32), ii: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.float32)):
        i = wp.tid()
        wp.atomic_add(ac, ii[i], v[i])

    wp.launch(k9, dim=n, inputs=[acc, idx, b], device="cpu")
    outs["k_atomic_accum"] = acc.numpy().copy()

    # 10 mixed (accum final value)
    acc2 = wp.empty(8, dtype=wp.float32, device="cpu")
    acc2.zero_()

    @wp.kernel
    def k10(a: wp.array(dtype=wp.vec3f), bb: wp.array(dtype=wp.vec3f), ac: wp.array(dtype=wp.float32), p: wp.float32):
        i = wp.tid()
        c = wp.cross(a[i], bb[i])
        l = wp.length(c)
        nn = wp.normalize(c)
        s = wp.sin(p)
        blended = wp.where(l > 0.5, nn * s, bb[i])
        wp.atomic_add(ac, 0, wp.dot(nn, blended))

    wp.launch(k10, dim=n, inputs=[va, vb, acc2, 0.7], device="cpu")
    outs["k_mixed"] = acc2.numpy().copy()

    return inp, outs


def run_metal_compare(inp):
    import warp_metal as wp
    import warp_metal.msl.spike_kernels as sk

    n = N
    outs = {}

    def arr_f32(x=None):
        a = wp.zeros(n, dtype=wp.float32)
        if x is not None:
            a.numpy_view()[:] = x
        return a

    def arr_vec3(x=None):
        a = wp.zeros(n, dtype=wp.vec3f)
        if x is not None:
            a.numpy_view()[:] = x
        return a

    a = arr_f32(inp["f32_a"])
    b = arr_f32(inp["f32_b"])
    phase = arr_f32(inp["phase"])
    clip_in = arr_f32(inp["clip_in"])
    mat = wp.zeros(n, dtype=wp.mat33f)
    # warp-metal numpy_view() now matches upstream .numpy() shapes (n,3,3);
    # golden npz stores mat33 flat (n,9) — reshape on the way in.
    mat.numpy_view()[:] = inp["mat33"].reshape(mat.numpy_view().shape)
    va = arr_vec3(inp["vec3_a"])
    vb = arr_vec3(inp["vec3_b"])
    idx = wp.zeros(n, dtype=wp.int32)
    idx.numpy_view()[:] = inp["idx"]

    out = arr_f32()
    wp.launch(sk.k_copy_scale, dim=n, inputs=[a, out, 2.5], device="metal")
    outs["k_copy_scale"] = out.to_numpy()

    out = arr_vec3()
    wp.launch(sk.k_saxpy3, dim=n, inputs=[va, vb, out, 2.0], device="metal")
    outs["k_saxpy3"] = out.to_numpy()

    out = arr_f32()
    wp.launch(sk.k_dot, dim=n, inputs=[va, vb, out], device="metal")
    outs["k_dot"] = out.to_numpy()

    out = arr_vec3()
    wp.launch(sk.k_cross_normalize, dim=n, inputs=[va, vb, out], device="metal")
    outs["k_cross_normalize"] = out.to_numpy()

    out = arr_f32()
    wp.launch(sk.k_trig, dim=n, inputs=[phase, out], device="metal")
    outs["k_trig"] = out.to_numpy()

    out = arr_f32()
    wp.launch(sk.k_select_clip, dim=n, inputs=[clip_in, out, -1.0, 1.0], device="metal")
    outs["k_select_clip"] = out.to_numpy()

    out = arr_vec3()
    wp.launch(sk.k_matvec, dim=n, inputs=[mat, va, out], device="metal")
    outs["k_matvec"] = out.to_numpy()

    out = arr_vec3()
    wp.launch(sk.k_matinv_apply, dim=n, inputs=[mat, va, out], device="metal")
    outs["k_matinv_apply"] = out.to_numpy()

    acc = wp.zeros(8, dtype=wp.float32)
    wp.launch(sk.k_atomic_accum, dim=n, inputs=[acc, idx, b], device="metal")
    outs["k_atomic_accum"] = acc.to_numpy()

    acc2 = wp.zeros(8, dtype=wp.float32)
    wp.launch(sk.k_mixed, dim=n, inputs=[va, vb, acc2, 0.7], device="metal")
    outs["k_mixed"] = acc2.to_numpy()

    return outs


TOLERANCES = {
    # kernel: (atol, rtol) — atomics allow order differences (looser)
    "k_copy_scale": (1e-5, 1e-5),
    "k_saxpy3": (1e-5, 1e-5),
    "k_dot": (1e-5, 1e-5),
    "k_cross_normalize": (1e-5, 1e-5),
    "k_trig": (1e-5, 1e-5),
    "k_select_clip": (0.0, 0.0),  # exact: pure data movement via ternaries
    "k_matvec": (1e-5, 1e-5),
    "k_matinv_apply": (1e-4, 1e-4),  # cofactor inverse, fp32 rounding
    "k_atomic_accum": (1e-4, 1e-4),  # order-dependent summation
    "k_mixed": (1e-4, 1e-4),
}


def compare(golden_path: Path, report_path: Path) -> int:
    data = np.load(golden_path)
    inp = {k.removeprefix("input_"): data[k] for k in data.files if k.startswith("input_")}
    cpu = {k: data[k] for k in data.files if not k.startswith("input_")}

    device = None
    try:
        import warp_metal as _wp

        device = _wp.device_name()
    except Exception as exc:  # pragma: no cover
        print(f"metal unavailable: {exc}")
        return 2

    metal = run_metal_compare(inp)

    lines = [
        "# Phase 0 spike — numeric alignment report (KEP-0001 M0.5)",
        "",
        f"- Date: 2026-11-13",
        f"- Reference: warp-lang 1.19.0 CPU(LLVM), N={N}, seed={SEED}",
        f"- Device under test: {device} (Metal, storageModeShared buffers)",
        f"- Tolerance policy: fp32 atol/rtol per kernel (atomics looser for order)",
        "",
        "| kernel | max abs err | max rel err | tol | verdict |",
        "|---|---|---|---|---|",
    ]
    failures = 0
    for kernel in TOLERANCES:
        ref = cpu[kernel]
        got = metal[kernel]
        atol, rtol = TOLERANCES[kernel]
        err = np.abs(got - ref)
        max_abs = float(err.max()) if err.size else 0.0
        denom = np.maximum(np.abs(ref), 1e-6)
        max_rel = float((err / denom).max()) if err.size else 0.0
        ok = np.allclose(got, ref, atol=atol, rtol=rtol)
        failures += 0 if ok else 1
        lines.append(
            f"| {kernel} | {max_abs:.3e} | {max_rel:.3e} | atol={atol} rtol={rtol} | "
            + ("PASS" if ok else "**FAIL**")
            + " |"
        )

    passed = len(TOLERANCES) - failures
    lines += [
        "",
        f"**Result: {passed}/{len(TOLERANCES)} kernels aligned.**",
        "",
        "- `wp.launch(device=\"metal\")` end-to-end: OK (emit -> MSL compile -> PSO -> dispatch -> shared-memory results)",
        "- MSL emitter maps: tid=thread_position_in_grid, arrays=device pointers + [[buffer(i)]],",
        "  scalars via constant-space buffers, sin/cos=precise::, atomics=atomic_fetch_add_explicit",
        "- Note: k_select_clip compared exactly (pure ternary routing); k_matinv_apply tolerates",
        "  cofactor-inverse fp32 rounding; atomic kernels tolerate summation order.",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"report written: {report_path}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", action="store_true", help="produce CPU golden outputs")
    ap.add_argument("--compare", action="store_true", help="run Metal side and write report")
    args = ap.parse_args()

    if args.golden:
        inp, outs = run_cpu_golden()
        payload = {f"input_{k}": v for k, v in inp.items()}
        payload.update(outs)
        np.savez(GOLDEN_PATH, **payload)
        print(f"golden written: {GOLDEN_PATH} ({len(outs)} kernels)")
        return 0

    if args.compare:
        return compare(GOLDEN_PATH, REPORT_PATH)

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
