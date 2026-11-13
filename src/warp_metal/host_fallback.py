# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""CPU fallback routing for closure-external kernels (KEP-0001 M1.3 design).

Metal has no fp64; Newton's inertia kernels deliberately use float64
intermediates (see newton/_src/geometry/inertia.py comments). They are
capability-boundary exempt in the corpus layer, so per the KEP's M1.3
design ("closure-external long tail routes to CPU fallback — unified
memory makes shared-storage arrays zero-copy") they run here as verbatim
numpy ports on the same shared-storage buffers, with float64 intermediates
preserved.

The registry is explicit: a kernel that neither translates to MSL nor is
registered here fails loud at launch. Each port is verified against the
real-warp CPU golden in the corpus layer (conform.py).
"""

from __future__ import annotations

import threading

import numpy as np

_LOCK = threading.Lock()
_FALLBACKS: dict[str, object] = {}

_FLOAT32_EPS = np.float32(np.finfo(np.float32).eps)
_INERTIA_SYMMETRY_RTOL = 1.0e-5
_INERTIA_SYMMETRY_ATOL = 1.0e-8


def register(module: str, key: str):
    """Register a CPU fallback under ``<module>.<key>`` (kernel identity)."""

    def deco(fn):
        with _LOCK:
            _FALLBACKS[f"{module}.{key}"] = fn
        return fn

    return deco


def get(kernel) -> object | None:
    ident = f"{kernel.func.__module__}.{kernel.key}"
    return _FALLBACKS.get(ident)


def _mat33_view(arr, i: int) -> np.ndarray:
    """Row-major 3x3 float32 view of element ``i`` of a mat33 array."""
    return arr.numpy.reshape(arr.n, 9)[i].reshape(3, 3)


def _eig_inertia_values(inertia_f32: np.ndarray) -> np.ndarray:
    """float32 eigenvalues of the symmetric inertia (newton _eig_inertia).

    Normalize by max |entry| (eig3's thresholds assume unit scale), decompose
    in float64, rescale — same numerics as the upstream float64 path.
    EigenVECTORS are unused by both kernels, only values matter.
    """
    scale = float(np.max(np.abs(inertia_f32)))
    if scale == 0.0:
        return np.zeros(3, dtype=np.float32)
    values = np.linalg.eigh(inertia_f32.astype(np.float64) / np.float64(scale))[0]
    return (values * np.float64(scale)).astype(np.float32)


@register("newton._src.geometry.inertia", "compute_body_inverses_kernel")
def _compute_body_inverses(kernel, dim, merged):
    body_mass, body_inertia, body_inv_mass, body_inv_inertia = merged
    for i in range(len(body_mass.numpy)):
        m = float(body_mass.numpy[i])
        if m > 0.0:
            # float64 intermediates, float32 store (upstream _inverse_inertia_wp)
            inv = np.linalg.inv(_mat33_view(body_inertia, i).astype(np.float64))
            body_inv_mass.numpy[i] = np.float32(1.0 / m)
            _mat33_view(body_inv_inertia, i)[:] = inv.astype(np.float32)
        else:
            body_inv_mass.numpy[i] = np.float32(0.0)
            _mat33_view(body_inv_inertia, i)[:] = np.float32(0.0)


@register("newton._src.geometry.inertia", "validate_and_correct_inertia_kernel")
def _validate_and_correct_inertia(kernel, dim, merged):
    (
        body_mass,
        body_inertia,
        balance_inertia,
        bound_mass,
        bound_inertia,
        relative_tolerance,
        absolute_tolerance,
        correction,
        correction_count,
    ) = merged
    bound_mass = float(getattr(bound_mass, "value", bound_mass))
    bound_inertia = float(getattr(bound_inertia, "value", bound_inertia))
    relative_tolerance = float(getattr(relative_tolerance, "value", relative_tolerance))
    absolute_tolerance = float(getattr(absolute_tolerance, "value", absolute_tolerance))
    correction = float(getattr(correction, "value", correction))

    for tid in range(dim):
        mass = np.float32(body_mass.numpy[tid])
        inertia = _mat33_view(body_inertia, tid).copy()
        original_inertia = inertia.copy()
        was_corrected = False

        if not np.all(np.isfinite(inertia)) or not np.isfinite(mass):
            mass = np.float32(0.0)
            inertia = np.zeros((3, 3), dtype=np.float32)
            was_corrected = True

        if mass < 0.0:
            mass = np.float32(0.0)
            was_corrected = True

        if bound_mass > 0.0 and 0.0 < mass < bound_mass:
            mass = np.float32(bound_mass)
            was_corrected = True

        if mass == 0.0:
            if float(np.dot(inertia.reshape(-1), inertia.reshape(-1))) > 0.0:
                was_corrected = True
            inertia = np.zeros((3, 3), dtype=np.float32)
        else:
            sym01 = np.float32((inertia[0, 1] + inertia[1, 0]) * np.float32(0.5))
            sym02 = np.float32((inertia[0, 2] + inertia[2, 0]) * np.float32(0.5))
            sym12 = np.float32((inertia[1, 2] + inertia[2, 1]) * np.float32(0.5))
            sym = np.array(
                [
                    [inertia[0, 0], sym01, sym02],
                    [sym01, inertia[1, 1], sym12],
                    [sym02, sym12, inertia[2, 2]],
                ],
                dtype=np.float32,
            )
            tol01 = np.float32(_INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * abs(sym01))
            tol02 = np.float32(_INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * abs(sym02))
            tol12 = np.float32(_INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * abs(sym12))
            if (
                abs(inertia[0, 1] - sym01) > tol01
                or abs(inertia[1, 0] - sym01) > tol01
                or abs(inertia[0, 2] - sym02) > tol02
                or abs(inertia[2, 0] - sym02) > tol02
                or abs(inertia[1, 2] - sym12) > tol12
                or abs(inertia[2, 1] - sym12) > tol12
            ):
                was_corrected = True
            inertia = sym

            eigvals = _eig_inertia_values(inertia)
            I1, I2, I3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
            if I1 > I2:
                I1, I2 = I2, I1
            if I2 > I3:
                I2, I3 = I3, I2
                if I1 > I2:
                    I1, I2 = I2, I1

            eig_threshold = max(relative_tolerance * I3, absolute_tolerance)
            if I1 < eig_threshold:
                adjustment = eig_threshold - I1 + correction
                I1 += adjustment
                I2 += adjustment
                I3 += adjustment
                inertia = inertia + np.eye(3, dtype=np.float32) * np.float32(adjustment)
                was_corrected = True

            if bound_inertia > 0.0 and I1 < bound_inertia:
                adjustment = bound_inertia - I1
                I1 += adjustment
                I2 += adjustment
                I3 += adjustment
                inertia = inertia + np.eye(3, dtype=np.float32) * np.float32(adjustment)
                was_corrected = True

            tri_tol = max(_FLOAT32_EPS * I3, absolute_tolerance)
            if balance_inertia and (I1 + I2 < I3 - tri_tol):
                deficit = I3 - I1 - I2
                adjustment = deficit + correction
                inertia = inertia + np.eye(3, dtype=np.float32) * np.float32(adjustment)
                was_corrected = True

            if was_corrected:
                eigvals = _eig_inertia_values(inertia)
                min_eig = float(min(eigvals[0], min(eigvals[1], eigvals[2])))
                if min_eig <= 0.0:
                    scale = max(
                        abs(inertia[0, 0]),
                        max(abs(inertia[1, 1]), abs(inertia[2, 2])),
                    )
                    adjustment = abs(min_eig) + max(
                        correction, max(absolute_tolerance, _FLOAT32_EPS * scale)
                    )
                    inertia = inertia + np.eye(3, dtype=np.float32) * np.float32(adjustment)

        output_inertia = inertia if was_corrected else original_inertia
        body_mass.numpy[tid] = mass
        _mat33_view(body_inertia, tid)[:] = output_inertia
        if was_corrected:
            correction_count.numpy[0] += 1
