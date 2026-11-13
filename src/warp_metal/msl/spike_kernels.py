# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""The 10 representative Phase 0 spike kernels (KEP-0001 M0.3).

Plain annotated functions (the spike emitter reads annotations directly;
the ``@wp.kernel`` decorator path lands with the module builder in M1.2).
Each kernel exercises the builtins named in the M0.3 acceptance list:

  1. k_copy_scale        — tid, array load/store (fp32)
  2. k_saxpy3            — vec3 arithmetic, array load/store (vec3)
  3. k_dot               — dot
  4. k_cross_normalize   — cross, normalize, length
  5. k_trig              — sin, cos
  6. k_select_clip       — select (ternary), scalar compare
  7. k_matvec            — mat33 @ vec3
  8. k_matinv_apply      — mat33 inverse, mat33 @ vec3
  9. k_atomic_accum      — atomic_add (fp32)
 10. k_mixed             — integration: vec math + select + trig + scalar op
"""

from __future__ import annotations

import warp_metal as wp


def k_copy_scale(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32), s: wp.float32):
    i = wp.tid()
    x = a[i]
    out[i] = x * s


def k_saxpy3(a: wp.array(dtype=wp.vec3f), b: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f), s: wp.float32):
    i = wp.tid()
    out[i] = a[i] * s + b[i]


def k_dot(a: wp.array(dtype=wp.vec3f), b: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.dot(a[i], b[i])


def k_cross_normalize(a: wp.array(dtype=wp.vec3f), b: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    c = wp.cross(a[i], b[i])
    out[i] = wp.normalize(c)
    l = wp.length(c)


def k_trig(phase: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    s = wp.sin(phase[i])
    c = wp.cos(phase[i])
    out[i] = s * s + c * c


def k_select_clip(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32), lo: wp.float32, hi: wp.float32):
    i = wp.tid()
    x = a[i]
    out[i] = wp.select(x < lo, lo, wp.select(x > hi, hi, x))


def k_matvec(m: wp.array(dtype=wp.mat33f), v: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = m[i] @ v[i]


def k_matinv_apply(m: wp.array(dtype=wp.mat33f), v: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    minv = wp.inverse(m[i])
    out[i] = minv @ v[i]


def k_atomic_accum(accum: wp.array(dtype=wp.float32), idx: wp.array(dtype=wp.int32), vals: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(accum, idx[i], vals[i])


def k_mixed(a: wp.array(dtype=wp.vec3f), b: wp.array(dtype=wp.vec3f), accum: wp.array(dtype=wp.float32), phase: wp.float32):
    i = wp.tid()
    c = wp.cross(a[i], b[i])
    l = wp.length(c)
    n = wp.normalize(c)
    s = wp.sin(phase)
    blended = wp.select(l > 0.5, n * s, b[i])
    wp.atomic_add(accum, 0, wp.dot(n, blended))


SPIKE_KERNELS = [
    k_copy_scale,
    k_saxpy3,
    k_dot,
    k_cross_normalize,
    k_trig,
    k_select_clip,
    k_matvec,
    k_matinv_apply,
    k_atomic_accum,
    k_mixed,
]
