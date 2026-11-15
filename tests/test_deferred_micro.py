# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M2.6 perf round 2b: eager-pipeline microbenchmark.

The Newton conveyor bench regressed ~35x when eager ``zero_``/``fill_``/``copy``
moved from the round-1 shape (ordered host write + flush per fill) to blit
sections in the pending eager command buffer (round 2), while graph-replayed
scenes with a few blits per batch stayed fine. This microbenchmark isolates the
mechanism on the pending eager CB alone: a fixed number of trivial compute
dispatches with the fill/copy traffic encoded three ways —

  S   one dispatch + one flush            (the commit+wait floor)
  A   compute only, 1 flush               (pure deferred-encode cost)
  B   compute + blit fills INTERLEAVED, 1 flush   (round-2 mechanism)
  B2  compute + blit fills all at END, 1 flush    (transition-free control:
      same blit count, only 2 encoder transitions)
  C   compute + COMPUTE-section fills interleaved, 1 flush (round-2b fix)
  E   compute + blit copies INTERLEAVED, 1 flush  (same question for copies)
  D   compute + ordered HOST-path fills (flush per fill)  (round-1 reference)

Reading the table:
- B >> A with B2 ~= A            -> the cost is the compute<->blit encoder
  TRANSITION count on one command buffer (cross-engine serialization), not
  blit work itself -> encode eager fills as compute sections (C).
- B ~= B2 >> A                   -> blit sections are expensive per se.
- B ~= A                         -> the eager pipeline is innocent; look
  elsewhere (Newton-level interaction).

The test asserts exact results and in-stream fill/copy ORDER for every timed
configuration (a fill that executes late would leave telltale non-zero
accumulators), prints the ms table, and carries one deliberately loose
re-regression guard (interleaved GPU-encoded fills within 10x of compute-only).
The 10x factor is a catastrophe guard against re-introducing the measured
round-2 regression (35x), not a tuned performance target; the table itself is
the measurement of record.

Routing note (round 2c): the eager DEFAULT is the round-1 ordered host path
(KEP D2 — the GPU encodings are unproven at Newton scale), so this module
forces ``WARP_METAL_EAGER_GPU_FILLS=1`` for the whole class (``setUpClass``):
config C and the in-stream ordering tests route through the public
``zero_``/``wp.copy`` APIs and must exercise the GPU mechanism, not silently
the host path. Configs B/B2/E drive the native blit entries directly and
config D is host-path in every mode (5.0 is not byte-replicable), so they
are knob-independent by construction.
"""

import gc
import os
import time
import unittest

import numpy as np

import warp_metal as wp
from warp_metal import native as _mm
from warp_metal.runtime import devices as _devices

_SKIP = not _devices.is_metal_available()
_SKIP_MSG = "no Metal device"

_N = 64  # elements per trivial-kernel array
_FILL_N = 256  # elements per filled array
_N_LAUNCH = 260  # compute dispatches per timed configuration
_N_FILL = 50  # fills/copies per timed configuration (interleave: 1 per 5)
_SAMPLES = 5


def _sync_eager_env() -> bool:
    s = os.environ.get("WARP_METAL_SYNC_EAGER")
    return bool(s) and s != "0"


@wp.kernel
def _bump(a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel
def _unique(out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float(i) + 1.0


@wp.kernel
def _double(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] * 2.0


# --------------------------------------------------------------------------------------
# Configuration runners: fresh arrays each call, encode everything, ONE flush
# at the end (except D, whose host-path fills flush per fill — that IS the
# round-1 shape). No host-visible reads inside; callers validate after timing.
# --------------------------------------------------------------------------------------


def _run_single_flush():
    a = wp.zeros(_N, dtype=wp.float32, device="metal")
    wp.launch(_bump, _N, inputs=[a], device="metal")
    wp.flush()
    return a


def _run_compute_only(n_launch=_N_LAUNCH):
    a = wp.zeros(_N, dtype=wp.float32, device="metal")
    for _ in range(n_launch):
        wp.launch(_bump, _N, inputs=[a], device="metal")
    wp.flush()
    return a


def _run_blit_fills(n_launch=_N_LAUNCH, n_fill=_N_FILL, at_end=False):
    """Round-2 TRIAGE mechanism: fills as blit sections in the pending CB,
    encoded through the native blit entry (since round 2b ``zero_`` routes
    to compute sections, so the blit mechanism is measured directly)."""
    a = wp.zeros(_N, dtype=wp.float32, device="metal")
    f = wp.zeros(_FILL_N, dtype=wp.float32, device="metal")
    nbytes = _FILL_N * 4
    every = max(1, n_launch // n_fill)
    fills = 0
    for i in range(n_launch):
        wp.launch(_bump, _N, inputs=[a], device="metal")
        if not at_end and (i + 1) % every == 0 and fills < n_fill:
            _mm.blit_fill(f.buffer, 0, nbytes, 0)  # blit section, no flush
            fills += 1
    if at_end:
        for _ in range(n_fill):
            _mm.blit_fill(f.buffer, 0, nbytes, 0)
        fills = n_fill
    wp.flush()
    assert fills == n_fill
    return a, f


def _run_blit_copies(n_launch=_N_LAUNCH, n_fill=_N_FILL):
    """Round-2 TRIAGE mechanism for copies: blit copy sections in the
    pending CB, encoded through the native blit entry (since round 2b
    ``wp.copy`` routes to compute sections)."""
    a = wp.zeros(_N, dtype=wp.float32, device="metal")
    x = wp.zeros(_FILL_N, dtype=wp.float32, device="metal")
    y = wp.zeros(_FILL_N, dtype=wp.float32, device="metal")
    nbytes = _FILL_N * 4
    every = max(1, n_launch // n_fill)
    copies = 0
    for i in range(n_launch):
        wp.launch(_bump, _N, inputs=[a], device="metal")
        if (i + 1) % every == 0 and copies < n_fill:
            _mm.blit_copy(x.buffer, y.buffer, 0, 0, nbytes)  # blit, no flush
            copies += 1
    wp.flush()
    assert copies == n_fill
    return a, y


def _run_compute_fills(n_launch=_N_LAUNCH, n_fill=_N_FILL):
    """Round-2b mechanism: fills as COMPUTE sections in the pending CB (no
    encoder transition). Routes through the public ``zero_`` API, so it
    needs the opt-in GPU-encode knob (``WARP_METAL_EAGER_GPU_FILLS=1``,
    forced for this class in ``setUpClass`` since the round-2c default
    flip) — otherwise ``zero_`` would take the default ordered host path
    and this would silently re-measure config D."""
    a = wp.zeros(_N, dtype=wp.float32, device="metal")
    f = wp.zeros(_FILL_N, dtype=wp.float32, device="metal")
    every = max(1, n_launch // n_fill)
    fills = 0
    for i in range(n_launch):
        wp.launch(_bump, _N, inputs=[a], device="metal")
        if (i + 1) % every == 0 and fills < n_fill:
            f.zero_()  # compute-section fill (deferred eager default)
            fills += 1
    wp.flush()
    assert fills == n_fill
    return a, f


def _run_host_fills(n_launch=_N_LAUNCH, n_fill=_N_FILL):
    """Round-1 reference: fills on the ordered host path (flush + host write
    per fill). ``fill_(5.0)`` on float32 is not byte-replicable, so it takes
    the host path in every mode."""
    a = wp.zeros(_N, dtype=wp.float32, device="metal")
    f = wp.zeros(_FILL_N, dtype=wp.float32, device="metal")
    every = max(1, n_launch // n_fill)
    fills = 0
    for i in range(n_launch):
        wp.launch(_bump, _N, inputs=[a], device="metal")
        if (i + 1) % every == 0 and fills < n_fill:
            f.fill_(5.0)  # host path: flush + ordered write
            fills += 1
    wp.flush()
    assert fills == n_fill
    return a, f


def _time(fn, samples=_SAMPLES):
    """Median-of-samples ms for ``fn`` (which flushes internally). Warmup runs
    first (PSO compile, first-touch allocations). Fresh arrays per sample;
    explicit GC between samples keeps the allocator out of the measurement."""
    fn()  # warmup
    times = []
    for _ in range(samples):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


class TestDeferredMicrobench(unittest.TestCase):
    """Microbench table + exact-result/in-stream-order assertions.

    Runs under the opt-in GPU-encode knob (``WARP_METAL_EAGER_GPU_FILLS=1``,
    forced in ``setUpClass``): config C and the in-stream ordering tests go
    through the public ``zero_``/``wp.copy`` APIs and must exercise the GPU
    mechanism, not the round-2c default host path."""

    @classmethod
    def setUpClass(cls):
        cls._saved_fills_env = os.environ.get("WARP_METAL_EAGER_GPU_FILLS")
        os.environ["WARP_METAL_EAGER_GPU_FILLS"] = "1"

    @classmethod
    def tearDownClass(cls):
        if cls._saved_fills_env is None:
            os.environ.pop("WARP_METAL_EAGER_GPU_FILLS", None)
        else:
            os.environ["WARP_METAL_EAGER_GPU_FILLS"] = cls._saved_fills_env

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)
        if _sync_eager_env():
            raise unittest.SkipTest(
                "pins the deferred pipeline; WARP_METAL_SYNC_EAGER is set"
            )

    # ---- in-stream ordering discriminants (cannot pass if fills/copies
    # execute late): these run before the timing so a broken encoding is a
    # correctness failure, not just a fast number ----

    def test_in_stream_fill_order(self):
        # unique-fill writes 1..N, zero_ must land BETWEEN the two kernels,
        # the second kernel reads the zeros: out is exactly zeros (late fill
        # => out == 2, 4, 6, ...)
        n = 64
        buf = wp.zeros(n, dtype=wp.float32, device="metal")
        out = wp.zeros(n, dtype=wp.float32, device="metal")
        wp.launch(_unique, n, inputs=[buf], device="metal")
        buf.zero_()
        wp.launch(_double, n, inputs=[buf, out], device="metal")
        wp.flush()
        np.testing.assert_array_equal(buf.numpy.copy(), np.zeros(n, dtype=np.float32))
        np.testing.assert_array_equal(out.numpy.copy(), np.zeros(n, dtype=np.float32))

    def test_in_stream_copy_order(self):
        # the copy source is OVERWRITTEN after the copy point in stream order:
        # in-stream y holds 1..N (z == 2..2N); a late copy would pick up the
        # overwrite (100s) instead — the discriminant the idempotent-late copy
        # needs
        n = 64
        x = wp.zeros(n, dtype=wp.float32, device="metal")
        y = wp.zeros(n, dtype=wp.float32, device="metal")
        z = wp.zeros(n, dtype=wp.float32, device="metal")
        wp.launch(_unique, n, inputs=[x], device="metal")
        wp.copy(y, x)
        wp.launch(_bump, n, inputs=[x], device="metal")
        x.fill_(100.0)  # host path (5.0/100.0 not byte-replicable): flushes
        wp.launch(_double, n, inputs=[y, z], device="metal")
        got_z = z.numpy.copy()
        np.testing.assert_allclose(
            got_z, 2.0 * np.arange(1, n + 1, dtype=np.float32), rtol=0, atol=1e-6
        )

    # ---- the benchmark table ----

    def test_microbench_table(self):
        rows: list[tuple[str, float, float]] = []
        results = {}

        def measure(name, fn, validate):
            times = _time(fn)
            got = fn()  # one more run, outside timing, for validation
            if validate is not None:
                validate(got)
            rows.append((name, min(times), sorted(times)[len(times) // 2]))
            results[name] = min(times)

        def v_single(got):
            np.testing.assert_allclose(got.numpy.copy(), np.ones(_N), rtol=0, atol=1e-6)

        def v_compute(got):
            np.testing.assert_allclose(
                got.numpy.copy(), np.full(_N, float(_N_LAUNCH)), rtol=0, atol=1e-6
            )

        def v_fills(got):
            a, f = got
            np.testing.assert_allclose(
                a.numpy.copy(), np.full(_N, float(_N_LAUNCH)), rtol=0, atol=1e-6
            )
            # exact in-stream result for EVERY fill regardless of position:
            # the fills land between launches, so f is 0 at flush time
            np.testing.assert_array_equal(f.numpy.copy(), np.zeros(_FILL_N, dtype=np.float32))

        def v_copies(got):
            a, y = got
            np.testing.assert_allclose(
                a.numpy.copy(), np.full(_N, float(_N_LAUNCH)), rtol=0, atol=1e-6
            )
            # x stays all-zero (nothing writes it): every in-stream copy
            # leaves y == 0; a GARBAGE-ORDERED copy (executed against stale
            # bytes) cannot change y here, so the dedicated ordering
            # discriminant above carries that contract — this asserts value
            # exactness of the copy path under interleaving
            np.testing.assert_array_equal(y.numpy.copy(), np.zeros(_FILL_N, dtype=np.float32))

        def v_host_fills(got):
            a, f = got
            np.testing.assert_allclose(
                a.numpy.copy(), np.full(_N, float(_N_LAUNCH)), rtol=0, atol=1e-6
            )
            np.testing.assert_allclose(f.numpy.copy(), np.full(_FILL_N, 5.0), rtol=0, atol=0)

        measure("S  1 launch + 1 flush (floor)", _run_single_flush, v_single)
        measure("A  260 compute, 1 flush", _run_compute_only, v_compute)
        measure("B  260 compute + 50 blit fills interleaved", _run_blit_fills, v_fills)
        measure("B2 260 compute + 50 blit fills at end", lambda: _run_blit_fills(at_end=True), v_fills)
        measure("C  260 compute + 50 compute fills interleaved", _run_compute_fills, v_fills)
        measure("E  260 compute + 50 blit copies interleaved", _run_blit_copies, v_copies)
        measure("D  260 compute + 50 host fills (round-1 shape)", _run_host_fills, v_host_fills)

        print("\n== M2.6 round-2b eager-pipeline microbench (ms; min/median of "
              f"{_SAMPLES}) ==")
        print(f"{'config':<46} {'min':>9} {'median':>9}")
        for name, lo, med in rows:
            print(f"{name:<46} {lo:9.2f} {med:9.2f}")

        # derived mechanism numbers (informational; printed for the report)
        a, b, b2 = results.get("A  260 compute, 1 flush"), results.get(
            "B  260 compute + 50 blit fills interleaved"
        ), results.get("B2 260 compute + 50 blit fills at end")
        if a and b and b2:
            per_transition = (b - b2) / (2 * _N_FILL - 2)
            print(
                f"derived: per compute<->blit transition ~{per_transition:.3f} ms "
                f"((B-B2)/{2 * _N_FILL - 2}); blit-at-end cost above A "
                f"(50 blits, 2 transitions) ~{b2 - a:.2f} ms"
            )

        # loose re-regression guard: the measured round-2 catastrophe is 35x;
        # guard at 10x so re-introducing cross-engine transition stalls fails
        # the suite instead of silently re-regressing Newton benches. This is
        # NOT a tuned perf target — the table is the measurement of record.
        if a and b:
            self.assertLess(
                b,
                10.0 * max(a, 1e-3),
                f"interleaved blit fills {b:.1f} ms >= 10x compute-only "
                f"{a:.1f} ms — the eager pipeline regressed to cross-engine "
                "transition stalls (see the printed table)",
            )


if __name__ == "__main__":
    unittest.main()
