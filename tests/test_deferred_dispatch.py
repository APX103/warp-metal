# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M2.6 deferred eager pipeline: launches encode into one persistent command
buffer and execute at the next flush.

Contracts under test:
- deferred dispatches produce correct results, flushed explicitly or at the
  first host-visible read (flush-on-read through ``.numpy``/``numpy_view``)
- host writes through existing arrays (``__setitem__``) land AFTER pending
  kernels, never under them (flush-before-write ordering)
- DEFAULT eager ``zero_``/``fill_``/``wp.copy``/``clone`` take the round-1
  ordered host path (flush + host write), the conveyor-safe default since
  round 2c (round 2/2b GPU encodings regressed the Newton conveyor ~35x)
- with ``WARP_METAL_EAGER_GPU_FILLS=1`` (set per test class), eager
  ``zero_``/``fill_``/``wp.copy``/``clone`` are GPU-encoded COMPUTE
  sections on the same compute encoder as the launches around them (M2.6
  round 2b; round 2 used blit sections): no host write, no flush, no
  encoder transition, in-stream order with the launches around them — a
  dependent launch observes a fill/copy without any intermediate flush,
  and host reads flush first (stale host bytes are never exposed)
- byte-exact fill eligibility under the knob (one replicable 8-bit
  pattern: 0 anywhere, -1 in int/uint, True in bool); other values segment
  to the ordered host path naming the M2.5 commit-2 rule (flush + write,
  any value kept exact)
- buffers released while a command buffer is uncommitted (params temporaries,
  GC'd arrays) outlive it (native pin until flush completes)
- ``WARP_METAL_SYNC_EAGER=1`` (per-dispatch commit+wait) agrees with the
  deferred path and keeps the ordered host write for ``zero_``/``fill_``/
  ``copy`` regardless of the fill knob
- graph replay after pre-marshaling (capture_end) matches eager execution and
  keeps the native batch-call counts the spies pin

Unique-cell write fixtures (each cell written exactly once) are the only
deterministic stride/race discriminant on Metal.
"""

import gc
import os
import unittest
from contextlib import contextmanager

import numpy as np

import warp_metal as wp
from warp_metal.runtime import devices as _devices

_SKIP = not _devices.is_metal_available()
_SKIP_MSG = "no Metal device"


def _sync_eager_env() -> bool:
    """True when WARP_METAL_SYNC_EAGER forces the per-dispatch commit+wait.

    The GPU-encoding tests pin the DEFERRED pipeline's contracts (no flush,
    compute sections); they skip under the escape hatch instead of failing —
    the parity class covers the sync-mode behavior explicitly."""
    s = os.environ.get("WARP_METAL_SYNC_EAGER")
    return bool(s) and s != "0"


@wp.kernel
def _unique_fill(out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float(i) + 1.0  # unique cell: cell i written once by thread i


@wp.kernel
def _unique_fill_i32(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = wp.int(i) + 1


@wp.kernel
def _double(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel
def _double_i32(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = a[i] * 2


@wp.kernel
def _add1(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] + 1.0


@wp.kernel
def _add_pair(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = a[i] + b[i]


def _expected_fill(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=np.float32)


class TestDeferredDispatch(unittest.TestCase):
    """Eager launches defer; results are exact at flush points."""

    N = 9

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_deferred_correct_at_explicit_flush(self):
        # two pending launches over distinct arrays; one explicit flush at the
        # end runs both, unique-cell reads are exact
        a = wp.zeros(self.N, dtype=wp.float32, device="metal")
        b = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, self.N, inputs=[a], device="metal")
        wp.launch(_unique_fill, self.N, inputs=[b], device="metal")
        wp.flush()  # explicit boundary
        wp.flush()  # idempotent: nothing pending now
        np.testing.assert_allclose(a.numpy_view(), _expected_fill(self.N), rtol=0, atol=1e-6)
        np.testing.assert_allclose(b.numpy_view(), _expected_fill(self.N), rtol=0, atol=1e-6)

    def test_flush_on_read(self):
        # no explicit flush anywhere: the first host read drains the pipeline
        a = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, self.N, inputs=[a], device="metal")
        np.testing.assert_allclose(a.numpy_view(), _expected_fill(self.N), rtol=0, atol=1e-6)

    def test_interleaved_chain_flushes_once_at_read(self):
        # a -> b (x2) -> c (+1), and a+d pair-sum, all pending at once; the
        # FIRST read flushes everything, every link is exact
        a = wp.from_numpy(np.arange(1, self.N + 1, dtype=np.float32), device="metal")
        d = wp.from_numpy(np.full(self.N, 10.0, dtype=np.float32), device="metal")
        b = wp.zeros(self.N, dtype=wp.float32, device="metal")
        c = wp.zeros(self.N, dtype=wp.float32, device="metal")
        s = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_double, self.N, inputs=[a, b], device="metal")
        wp.launch(_add1, self.N, inputs=[b, c], device="metal")
        wp.launch(_add_pair, self.N, inputs=[a, d, s], device="metal")
        # first host read = the single flush point for all three launches
        np.testing.assert_allclose(c.numpy_view(), 2.0 * a.numpy_view() + 1.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(b.numpy_view(), 2.0 * a.numpy_view(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(s.numpy_view(), a.numpy_view() + 10.0, rtol=0, atol=1e-6)

    def test_zero_ordering_vs_pending(self):
        # pending kernel writes buf (1..N), then the host zero_ must land
        # AFTER that kernel (flush-before-write), then the next pending
        # kernel reads the zeros: out is exactly zeros.
        # Without flush-before-write the zero could land under the pending
        # fill and out would read 2,4,6,... (nondeterministic).
        buf = wp.zeros(self.N, dtype=wp.float32, device="metal")
        out = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, self.N, inputs=[buf], device="metal")
        buf.zero_()
        wp.launch(_double, self.N, inputs=[buf, out], device="metal")
        wp.flush()
        np.testing.assert_array_equal(buf.numpy_view(), np.zeros(self.N, dtype=np.float32))
        np.testing.assert_array_equal(out.numpy_view(), np.zeros(self.N, dtype=np.float32))

    def test_fill_ordering_vs_pending(self):
        # same discriminant as test_zero_ordering_vs_pending with a non-zero
        # fill: the fill is an ordered HOST write AFTER the pending kernel
        # (default round-1 routing: fill_ flushes first, so the fill wins
        # over the kernel's 1..N), and the second kernel reading buf sees
        # exactly -1 everywhere (out = -2). If the fill landed under the
        # first kernel (or not at all), out would read -2 only where the
        # fill survived — never uniformly.
        # (The in-stream GPU-encoded fill ordering — fill between two
        # launches with NO flush — is pinned under the
        # WARP_METAL_EAGER_GPU_FILLS=1 knob: TestGPUEncodedFills here and
        # tests/test_deferred_micro.py test_in_stream_fill_order.)
        buf = wp.zeros(self.N, dtype=wp.int32, device="metal")
        out = wp.zeros(self.N, dtype=wp.int32, device="metal")
        wp.launch(_unique_fill_i32, self.N, inputs=[buf], device="metal")
        buf.fill_(-1)
        wp.launch(_double_i32, self.N, inputs=[buf, out], device="metal")
        wp.flush()
        np.testing.assert_array_equal(buf.numpy_view(), np.full(self.N, -1, dtype=np.int32))
        np.testing.assert_array_equal(out.numpy_view(), np.full(self.N, -2, dtype=np.int32))

    def test_setitem_ordering_vs_pending(self):
        # __setitem__ is a host write into an existing buffer: same ordering
        buf = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, self.N, inputs=[buf], device="metal")
        buf[0] = 100.0
        np.testing.assert_allclose(buf.numpy_view()[0], 100.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(buf.numpy_view()[1:], _expected_fill(self.N)[1:], rtol=0, atol=1e-6)


class TestSyncEagerEscapeHatch(unittest.TestCase):
    """WARP_METAL_SYNC_EAGER=1: the pre-M2.6 per-dispatch commit+wait."""

    N = 6

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    @staticmethod
    @contextmanager
    def _sync_eager():
        old = os.environ.get("WARP_METAL_SYNC_EAGER")
        os.environ["WARP_METAL_SYNC_EAGER"] = "1"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_SYNC_EAGER", None)
            else:
                os.environ["WARP_METAL_SYNC_EAGER"] = old

    def test_sync_eager_results_match_deferred(self):
        def run():
            a = wp.zeros(self.N, dtype=wp.float32, device="metal")
            b = wp.zeros(self.N, dtype=wp.float32, device="metal")
            out = wp.zeros(self.N, dtype=wp.float32, device="metal")
            wp.launch(_unique_fill, self.N, inputs=[a], device="metal")
            wp.launch(_double, self.N, inputs=[a, b], device="metal")
            wp.launch(_add1, self.N, inputs=[b, out], device="metal")
            return (a.numpy.copy(), b.numpy.copy(), out.numpy.copy())

        with self._sync_eager():
            synced = run()
        deferred = run()
        expected = (
            _expected_fill(self.N),
            2.0 * _expected_fill(self.N),
            2.0 * _expected_fill(self.N) + 1.0,
        )
        for got, want in zip(synced, expected):
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
        for s, d in zip(synced, deferred):
            np.testing.assert_array_equal(s, d)

    def test_sync_eager_flush_is_noop(self):
        # in per-dispatch sync mode nothing ever pends: flush stays a no-op
        with self._sync_eager():
            a = wp.zeros(4, dtype=wp.float32, device="metal")
            wp.launch(_unique_fill, 4, inputs=[a], device="metal")
            wp.flush()
            np.testing.assert_allclose(a.numpy_view(), _expected_fill(4), rtol=0, atol=1e-6)


class TestBufferLifetimeWhilePending(unittest.TestCase):
    """Buffers released while a command buffer is uncommitted outlive it.

    host.py frees its params/argbuf/arena temporaries right after every
    launch and arrays can GC mid-frame; the native pipeline pins every bound
    buffer until the flush completes (and keeps it in the residency set —
    dropping residency under pending work silently zeros reads)."""

    N = 8

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    def test_gc_input_array_while_pending(self):
        # the kernel's input array dies before the flush; its buffer must
        # still hold 1..N when the pending kernel runs
        a = wp.zeros(self.N, dtype=wp.float32, device="metal")
        out = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, self.N, inputs=[a], device="metal")
        wp.launch(_double, self.N, inputs=[a, out], device="metal")
        del a  # buffer_release while pending: native pin defers the free
        gc.collect()
        wp.flush()
        np.testing.assert_allclose(out.numpy_view(), 2.0 * _expected_fill(self.N), rtol=0, atol=1e-6)

    def test_gc_while_pending_then_reuse(self):
        # after the GC'd-buffer flush the runtime is healthy: a fresh launch
        # on (possibly reused) storage reads/writes exactly
        a = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, self.N, inputs=[a], device="metal")
        del a
        gc.collect()
        wp.flush()
        b = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_add1, self.N, inputs=[b, b], device="metal")  # in-place +1
        wp.flush()
        np.testing.assert_allclose(b.numpy_view(), np.ones(self.N, dtype=np.float32), rtol=0, atol=1e-6)

    def test_gc_same_buffer_compute_copy_while_pending(self):
        # a same-buffer (disjoint ranges) eager GPU copy pins the buffer as
        # BOTH src and dst (compute-copy section since round 2b; the blit
        # encoding had the same pin discipline); releasing it while pending
        # must consume the ownership exactly once — a double consume
        # underflows the CF refcount and SIGTRAPs in CFRelease at flush
        # teardown
        base = wp.from_numpy(np.arange(8, dtype=np.float32), device="metal")
        wp.copy(base[0:2], base[6:8])  # same buffer, disjoint: double pin
        del base
        gc.collect()
        wp.flush()
        # the runtime is healthy afterwards (the trap was in flush teardown)
        b = wp.zeros(4, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, 4, inputs=[b], device="metal")
        np.testing.assert_allclose(b.numpy_view(), _expected_fill(4), rtol=0, atol=1e-6)

    def test_gc_double_bound_kernel_array_while_pending(self):
        # an in-place launch binds the same buffer twice (input + output);
        # releasing it while pending consumes the ownership exactly once
        b = wp.zeros(4, dtype=wp.float32, device="metal")
        wp.launch(_add1, 4, inputs=[b, b], device="metal")
        del b
        gc.collect()
        wp.flush()
        c = wp.zeros(4, dtype=wp.float32, device="metal")
        wp.launch(_unique_fill, 4, inputs=[c], device="metal")
        np.testing.assert_allclose(c.numpy_view(), _expected_fill(4), rtol=0, atol=1e-6)


class TestPrebuiltReplay(unittest.TestCase):
    """Graph replay after capture_end pre-marshaling (M2.6).

    The batch-call boundary (``native.dispatch_batch``) is unchanged — the
    spies in TestCaptureReplay (test_host_runtime.py) pin its call counts
    and payloads; here the payload objects must BE the pre-built ones, and
    replay-after-premarshaling must match eager execution exactly."""

    N = 7

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    @staticmethod
    @contextmanager
    def _spied_batch():
        from warp_metal import native as mm

        calls = []
        real = mm.dispatch_batch
        old = os.environ.get("WARP_METAL_BATCH")
        os.environ["WARP_METAL_BATCH"] = "1"
        mm.dispatch_batch = lambda records: calls.append(records)
        try:
            yield calls
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_BATCH", None)
            else:
                os.environ["WARP_METAL_BATCH"] = old
            mm.dispatch_batch = real

    def test_replay_uses_prebuilt_payload(self):
        a = wp.from_numpy(np.arange(1, self.N + 1, dtype=np.float32), device="metal")
        b = wp.zeros(self.N, dtype=wp.float32, device="metal")
        out = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_double, self.N, inputs=[a, b], device="metal")
            wp.launch(_add1, self.N, inputs=[b, out], device="metal")
        # pre-marshaled at capture_end: one batch segment, one BatchPayload
        self.assertEqual(len(c.graph._segments), 1)
        kind, payload, base = c.graph._segments[0]
        self.assertEqual(kind, "batch")
        self.assertEqual(base, 0)
        from warp_metal import native as mm

        self.assertIsInstance(payload, mm.BatchPayload)
        self.assertEqual(len(payload), 2)
        # payload iterates the same record tuples dispatch_batch always took
        self.assertEqual([r[0] for r in payload], ["compute", "compute"])
        with self._spied_batch() as calls:
            wp.capture_launch(c.graph)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], payload)  # no re-marshaling on the replay path

    def test_replay_after_prebuild_matches_eager(self):
        x_np = np.arange(1, self.N + 1, dtype=np.float32)
        # eager reference
        ex = wp.from_numpy(x_np.copy(), device="metal")
        et = wp.zeros(self.N, dtype=wp.float32, device="metal")
        eo = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_double, self.N, inputs=[ex, et], device="metal")
        wp.launch(_add1, self.N, inputs=[et, eo], device="metal")
        # captured + replayed twice (idempotent, prebuilt payload reused)
        cx = wp.from_numpy(x_np.copy(), device="metal")
        ct = wp.zeros(self.N, dtype=wp.float32, device="metal")
        co = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_double, self.N, inputs=[cx, ct], device="metal")
            wp.launch(_add1, self.N, inputs=[ct, co], device="metal")
        wp.capture_launch(c.graph)
        wp.capture_launch(c.graph)
        np.testing.assert_allclose(ct.numpy_view(), et.numpy_view(), rtol=0, atol=0)
        np.testing.assert_allclose(co.numpy_view(), eo.numpy_view(), rtol=0, atol=0)

    def test_segments_track_growth_mid_capture(self):
        # replay while still recording (records-so-far), then record more and
        # replay again: the stale-segment guard rebuilds
        from warp_metal.runtime.host import active_capture

        a = wp.zeros(self.N, dtype=wp.float32, device="metal")
        b = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_unique_fill, self.N, inputs=[a], device="metal")
            wp.capture_launch(active_capture())  # records-so-far (1 op)
            self.assertEqual(len(active_capture()._records), 1)
            wp.launch(_unique_fill, self.N, inputs=[b], device="metal")
        self.assertEqual(c.graph.n_ops, 2)
        wp.capture_launch(c.graph)
        np.testing.assert_allclose(a.numpy_view(), _expected_fill(self.N), rtol=0, atol=1e-6)
        np.testing.assert_allclose(b.numpy_view(), _expected_fill(self.N), rtol=0, atol=1e-6)


class _SpiedNative:
    """Count native flush/compute-fill/copy calls while running real ones.

    ``host.py`` resolves ``_mm.flush``/``_mm.compute_fill``/``_mm.compute_copy``
    on the native module at call time, so swapping the module attributes
    spies on every eager-path call without changing behavior. (The blit
    entries are triage-only since round 2b and are NOT on the eager path —
    the microbench measures them directly.)"""

    CALLS = ("flush", "compute_fill", "compute_copy")

    def __init__(self):
        from warp_metal import native as mm

        self._mm = mm
        self._real = {name: getattr(mm, name) for name in self.CALLS}
        self.calls = {name: [] for name in self.CALLS}

    def _install(self):
        mm = self._mm

        def flush_spy():
            self.calls["flush"].append(())
            return self._real["flush"]()

        def compute_fill_spy(dst, off, nbytes, value):
            self.calls["compute_fill"].append((dst, off, nbytes, value))
            return self._real["compute_fill"](dst, off, nbytes, value)

        def compute_copy_spy(src, dst, src_off, dst_off, nbytes):
            self.calls["compute_copy"].append((src, dst, src_off, dst_off, nbytes))
            return self._real["compute_copy"](src, dst, src_off, dst_off, nbytes)

        mm.flush = flush_spy
        mm.compute_fill = compute_fill_spy
        mm.compute_copy = compute_copy_spy

    def __enter__(self):
        self._install()
        return self.calls

    def __exit__(self, *exc):
        for name, real in self._real.items():
            setattr(self._mm, name, real)
        return False


class TestGPUEncodedFills(unittest.TestCase):
    """Eager zero_/fill_ encode GPU COMPUTE sections — under the knob.

    Upstream zero_/fill_ are device memsets ON THE STREAM: ordered with the
    launches around them, no host write, no flush. That encoding is
    OPT-IN since round 2c (``WARP_METAL_EAGER_GPU_FILLS=1``, forced on for
    this class in ``setUpClass``): round 2 (blit sections) regressed the
    Newton conveyor ~35x and the 2b compute encoding is unproven at Newton
    scale, so the eager default is the round-1 ordered host write (pinned
    by ``TestDefaultEagerHostPath`` below). These tests pin the knob-on
    mechanism — GPU-encoded on the SAME compute encoder as the launches
    (the eager pipeline stays compute-only under the knob), unflushed
    across dependent launches, exact at the flush-on-read, byte-exact
    eligibility (other values segment to the ordered host path per the
    M2.5 commit-2 rule). The ``compute_fill`` call-count assertions fail
    loud if the routing ever silently falls back to the host path."""

    N = 9

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
            raise unittest.SkipTest("pins the deferred pipeline; WARP_METAL_SYNC_EAGER is set")

    def test_zero_then_launch_no_intermediate_flush(self):
        # launch writes 1..N, zero_ follows as an in-stream compute-section
        # fill, a second
        # launch reads: the whole chain stays UNFLUSHED (spy: zero native
        # flush calls) and the read at the end flushes exactly once, showing
        # the in-stream result — the kernel observed the zeros, so out2 is
        # 1.0 everywhere (2.0 would mean the fill did not precede it).
        out = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            wp.launch(_unique_fill, self.N, inputs=[out], device="metal")
            out.zero_()
            self.assertEqual(len(calls["compute_fill"]), 1)
            handle, off, nbytes, value = calls["compute_fill"][0]
            self.assertEqual(value, 0)
            self.assertEqual(off, 0)
            self.assertEqual(nbytes, self.N * 4)
            out2 = wp.zeros(self.N, dtype=wp.float32, device="metal")
            wp.launch(_add1, self.N, inputs=[out, out2], device="metal")
            self.assertEqual(calls["flush"], [])  # NO intermediate flush
            got2 = out2.numpy.copy()  # first host read = the one flush point
            self.assertEqual(len(calls["flush"]), 1)
            got = out.numpy.copy()
            self.assertEqual(len(calls["flush"]), 2)  # idle no-op flush
        np.testing.assert_allclose(got2, np.ones(self.N, dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(got, np.zeros(self.N, dtype=np.float32))

    def test_fill_negative_one_int_is_gpu_encoded(self):
        # Newton collide.py's per-frame fill_(-1) (contact match indices):
        # int32 -1 is the 0xFF pattern replicated — GPU-encoded, unflushed
        buf = wp.zeros(self.N, dtype=wp.int32, device="metal")
        with _SpiedNative() as calls:
            buf.fill_(-1)
            self.assertEqual(len(calls["compute_fill"]), 1)
            _handle, off, nbytes, value = calls["compute_fill"][0]
            self.assertEqual(value, 0xFF)
            self.assertEqual(off, 0)
            self.assertEqual(nbytes, self.N * 4)
            self.assertEqual(calls["flush"], [])  # no host flush for the fill
            got = buf.numpy.copy()  # flush-on-read
            self.assertEqual(len(calls["flush"]), 1)
        np.testing.assert_array_equal(got, np.full(self.N, -1, dtype=np.int32))

    def test_fill_byte_exact_patterns(self):
        # every dtype whose element bytes are one replicable pattern encodes
        u = wp.zeros(3, dtype=wp.uint8, device="metal")
        u.fill_(0xAB)
        np.testing.assert_array_equal(u.numpy.copy(), np.full(3, 0xAB, dtype=np.uint8))
        b = wp.zeros(4, dtype=wp.bool, device="metal")
        b.fill_(True)  # bool True is the 0x01 byte
        self.assertEqual(b.numpy.copy().tolist(), [True] * 4)
        l = wp.zeros(2, dtype=wp.int64, device="metal")
        l.fill_(-1)  # all eight bytes 0xFF: still one pattern
        np.testing.assert_array_equal(l.numpy.copy(), np.full(2, -1, dtype=np.int64))
        z = wp.zeros(3, dtype=wp.float16, device="metal")
        z.fill_(0.0)  # float16 zero is two 0x00 bytes
        np.testing.assert_array_equal(z.numpy.copy(), np.zeros(3, dtype=np.float16))

    def test_fill_nonzero_host_path(self):
        # non-byte-replicable values segment to the ordered host path (the
        # M2.5 commit-2 rule): correct values, a flush per fill, nothing
        # GPU-encoded — SolverSteel's INT32_MAX sentinels live here
        f = wp.zeros(3, dtype=wp.float32, device="metal")
        finf = wp.zeros(3, dtype=wp.float32, device="metal")
        i = wp.zeros(3, dtype=wp.int32, device="metal")
        with _SpiedNative() as calls:
            f.fill_(5.0)
            finf.fill_(float("inf"))
            i.fill_(2147483647)  # INT32_MAX sentinel (0x7fffffff, not replicable)
            self.assertEqual(calls["compute_fill"], [])
            self.assertGreaterEqual(len(calls["flush"]), 3)
        self.assertEqual(f.numpy.copy().tolist(), [5.0, 5.0, 5.0])
        self.assertEqual(finf.numpy.copy().tolist(), [np.inf, np.inf, np.inf])
        self.assertEqual(i.numpy.copy().tolist(), [2147483647] * 3)

    def test_fill_composite_value_host_path_zero_broadcast_encodes(self):
        # list/vec values broadcast per component — not byte-replicable:
        # host path with the exact broadcast result; a scalar zero broadcast
        # over a composite dtype IS byte-exact
        v = wp.zeros(2, dtype=wp.vec3, device="metal")
        with _SpiedNative() as calls:
            v.fill_([1.0, 2.0, 3.0])
            self.assertEqual(calls["compute_fill"], [])
            self.assertGreaterEqual(len(calls["flush"]), 1)
        np.testing.assert_array_equal(
            v.numpy.copy(), np.full((2, 3), [1.0, 2.0, 3.0], dtype=np.float32)
        )
        vz = wp.zeros(2, dtype=wp.vec3, device="metal")
        vz.fill_(0.0)
        np.testing.assert_array_equal(vz.numpy.copy(), np.zeros((2, 3), dtype=np.float32))

    def test_fill_negative_zero_host_path(self):
        # -0.0 compares equal to zero but its float pattern (0x80000000) is
        # not replicable: the M2.5 rule keeps it on the ordered host path
        f = wp.ones(3, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            f.fill_(-0.0)
            self.assertEqual(calls["compute_fill"], [])  # nothing GPU-encoded
            self.assertGreaterEqual(len(calls["flush"]), 1)  # ordered host write
        got = f.numpy.copy()
        self.assertTrue(np.signbit(got).all())  # really -0.0, not +0.0
        np.testing.assert_array_equal(got, np.zeros(3, dtype=np.float32))

    def test_fill_slice_view_offset_folds(self):
        # slice views share the base buffer: the fill range is the byte
        # extent of the VIEW (offset_elems folded in)
        base = wp.from_numpy(np.ones(6, dtype=np.float32), device="metal")
        view = base[2:5]
        with _SpiedNative() as calls:
            view.zero_()
            handle, off, nbytes, value = calls["compute_fill"][0]
            self.assertEqual(handle, base.buffer)
            self.assertEqual(off, 2 * 4)
            self.assertEqual(nbytes, 3 * 4)
        np.testing.assert_array_equal(
            base.numpy.copy(), np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )

    def test_zero_size_fill_encodes_nothing(self):
        e = wp.zeros(0, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            e.zero_()
            e.fill_(0.0)
            self.assertEqual(calls["compute_fill"], [])
            self.assertEqual(calls["flush"], [])

    def test_fill_fast_paths_large_arrays(self):
        # the fill kernel's host-selected paths must agree with the byte path:
        # mode 1 (4-byte words, offset and size 4-aligned), mode 0 (byte tail)
        # and odd offsets — a big grid (exercises multi-threadgroup dispatch)
        big = wp.zeros(100_003, dtype=wp.float32, device="metal")  # 400_012 B: %4==0
        big.fill_(0.0)
        np.testing.assert_array_equal(big.numpy.copy(), np.zeros(100_003, dtype=np.float32))
        odd = wp.zeros(7, dtype=wp.uint8, device="metal")  # 7 bytes: byte path
        odd.fill_(0x5A)
        np.testing.assert_array_equal(odd.numpy.copy(), np.full(7, 0x5A, dtype=np.uint8))
        view = wp.zeros(11, dtype=wp.uint8, device="metal")
        view[3:].fill_(0xFF)  # offset 3: unaligned byte path through the slice
        np.testing.assert_array_equal(
            view.numpy.copy(), np.array([0, 0, 0] + [0xFF] * 8, dtype=np.uint8)
        )


class TestGPUEncodedCopies(unittest.TestCase):
    """Eager wp.copy/clone encode GPU compute sections when eligible — under
    the knob (``WARP_METAL_EAGER_GPU_FILLS=1``, forced on in ``setUpClass``;
    the eager DEFAULT is the ordered host copy, pinned by
    ``TestDefaultEagerHostPath``).

    The eligibility matrix is the replay path's shared rule
    (``_blit_copy_payload``): equal dtype (dense row-major layouts are then
    byte-identical, composite dtypes included), same device, no same-buffer
    overlap. The GPU encoding is a compute-section copy kernel (round 2b;
    the round-2 blit encoding regressed the Newton conveyor ~35x).
    Ineligible copies keep the flush + host path — with identical results."""

    N = 7

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
            raise unittest.SkipTest("pins the deferred pipeline; WARP_METAL_SYNC_EAGER is set")

    def test_copy_then_launch_no_intermediate_flush(self):
        # copy x->y as an in-stream compute section, then a kernel reading
        # y: unflushed
        # until the single read, results exact
        x_np = np.arange(1, self.N + 1, dtype=np.float32)
        x = wp.from_numpy(x_np.copy(), device="metal")
        y = wp.zeros(self.N, dtype=wp.float32, device="metal")
        out = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            wp.copy(y, x)
            self.assertEqual(len(calls["compute_copy"]), 1)
            src, dst, src_off, dst_off, nbytes = calls["compute_copy"][0]
            self.assertEqual(src, x.buffer)
            self.assertEqual(dst, y.buffer)
            self.assertEqual((src_off, dst_off, nbytes), (0, 0, self.N * 4))
            wp.launch(_double, self.N, inputs=[y, out], device="metal")
            self.assertEqual(calls["flush"], [])  # NO intermediate flush
            got = out.numpy.copy()
            self.assertEqual(len(calls["flush"]), 1)
        np.testing.assert_allclose(got, 2.0 * x_np, rtol=0, atol=1e-6)

    def test_copy_slice_offsets_blit_matches_host(self):
        # slice views + explicit element offsets: byte offsets fold on both
        # sides (mirror of the replay-path offset test), eager reference vs
        # captured copy agree
        src_np = np.arange(10, dtype=np.float32)
        esrc = wp.from_numpy(src_np.copy(), device="metal")
        edst = wp.zeros(10, dtype=wp.float32, device="metal")
        edst.numpy[...] = 9.0  # poison: a dropped copy is caught
        with _SpiedNative() as calls:
            wp.copy(edst[3:9], esrc[2:8], dest_offset=1, src_offset=2, count=3)
            self.assertEqual(len(calls["compute_copy"]), 1)
            _src, _dst, src_off, dst_off, nbytes = calls["compute_copy"][0]
            # src base offset 2 elems + explicit 2 = 4; dst 3 + 1 = 4; 3 elems
            self.assertEqual((src_off, dst_off, nbytes), (16, 16, 12))
            self.assertEqual(calls["flush"], [])
        np.testing.assert_array_equal(
            edst.numpy.copy(),
            # poison 9.0 survives everywhere the 3-element copy didn't reach
            np.array([9.0, 9.0, 9.0, 9.0, 4.0, 5.0, 6.0, 9.0, 9.0, 9.0], dtype=np.float32),
        )

    def test_copy_composite_gpu_section(self):
        # composite dtypes (vec3/quat): equal dtype means byte-identical
        # dense elements — GPU-encoded like the replay path
        v_np = np.arange(9, dtype=np.float32).reshape(3, 3)
        ev = wp.from_numpy(v_np.copy(), dtype=wp.vec3, device="metal")
        ev_dst = wp.zeros(3, dtype=wp.vec3, device="metal")
        with _SpiedNative() as calls:
            wp.copy(ev_dst, ev)
            self.assertEqual(len(calls["compute_copy"]), 1)
            self.assertEqual(calls["flush"], [])
        np.testing.assert_allclose(ev_dst.numpy.copy(), v_np, rtol=0, atol=0)

    def test_copy_cross_dtype_same_size_host_path(self):
        # equal 4-byte elements but different dtypes: outside the GPU-copy rule
        # (upstream-legal raw memcpy) — host path, bits reinterpreted
        i = wp.from_numpy(np.array([1, 2], dtype=np.int32), device="metal")
        f = wp.zeros(2, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            wp.copy(f, i)
            self.assertEqual(calls["compute_copy"], [])
            self.assertGreaterEqual(len(calls["flush"]), 1)
        self.assertEqual(f.numpy.copy().tolist(), [1.0, 2.0])

    def test_copy_overlapping_ranges_host_path(self):
        # overlapping src/dst ranges in one buffer: undefined GPU-copy aliasing —
        # host path, same result as the replay test's overlap case
        base_np = np.arange(8, dtype=np.float32)
        ebase = wp.from_numpy(base_np.copy(), device="metal")
        with _SpiedNative() as calls:
            wp.copy(ebase[2:8], ebase[0:6])  # shifts elements 0..5 up by 2
            self.assertEqual(calls["compute_copy"], [])
            self.assertGreaterEqual(len(calls["flush"]), 1)
        np.testing.assert_array_equal(
            ebase.numpy.copy(), np.array([0, 1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
        )

    def test_copy_zero_size_noop(self):
        a = wp.zeros(4, dtype=wp.float32, device="metal")
        b = wp.zeros(4, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            wp.copy(b, a, count=0)
            self.assertEqual(calls["compute_copy"], [])
            self.assertEqual(calls["flush"], [])

    def test_copy_fast_paths_large_arrays(self):
        # mode 2 (four uint words per thread: offsets and size 16-aligned) and
        # mode 1 (one uint: 4-aligned) must agree with the byte path; a big
        # grid exercises multi-threadgroup dispatch
        n = 8192  # 32768 B: %16==0 on fresh arrays
        src_np = np.arange(1, n + 1, dtype=np.float32)
        a = wp.from_numpy(src_np.copy(), device="metal")
        b = wp.zeros(n, dtype=wp.float32, device="metal")
        wp.copy(b, a)
        np.testing.assert_array_equal(b.numpy.copy(), src_np)
        # 4-aligned but not 16-aligned: slice shifts both offsets by 4 bytes
        c = wp.zeros(n, dtype=wp.float32, device="metal")
        wp.copy(c[1:], a[0 : n - 1])
        np.testing.assert_array_equal(c.numpy.copy()[1:], src_np[:-1])
        self.assertEqual(int(c.numpy.copy()[0]), 0)  # untouched head

    def test_clone_routes_through_gpu_section(self):
        src_np = np.arange(1, self.N + 1, dtype=np.float32)
        src = wp.from_numpy(src_np.copy(), device="metal")
        with _SpiedNative() as calls:
            c = wp.clone(src)
            self.assertEqual(len(calls["compute_copy"]), 1)
            self.assertEqual(calls["flush"], [])
            got = c.numpy.copy()  # flush-on-read of the fresh dest
            self.assertEqual(len(calls["flush"]), 1)
        np.testing.assert_allclose(got, src_np, rtol=0, atol=0)

    def test_fill_and_copy_mix_in_stream(self):
        # the conveyor shape: zero_ -> fill_(-1) -> copy -> kernel, ALL in the
        # pending command buffer, one flush at the first read
        src = wp.from_numpy(np.full(4, 3.0, dtype=np.float32), device="metal")
        dst = wp.zeros(4, dtype=wp.float32, device="metal")
        marks = wp.zeros(4, dtype=wp.int32, device="metal")
        out = wp.zeros(4, dtype=wp.float32, device="metal")
        with _SpiedNative() as calls:
            dst.zero_()  # in-stream zero
            marks.fill_(-1)  # in-stream -1 pattern fill
            wp.copy(dst, src)  # in-stream copy
            wp.launch(_double, 4, inputs=[dst, out], device="metal")  # reads copy
            self.assertEqual(len(calls["compute_fill"]), 2)
            self.assertEqual(len(calls["compute_copy"]), 1)
            self.assertEqual(calls["flush"], [])
            got = out.numpy.copy()
            got_marks = marks.numpy.copy()
            self.assertEqual(len(calls["flush"]), 2)
        np.testing.assert_allclose(got, np.full(4, 6.0), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(got_marks, np.full(4, -1, dtype=np.int32))


class TestSyncEagerGPUOpsParity(unittest.TestCase):
    """WARP_METAL_SYNC_EAGER=1 keeps the OLD zero_/fill_/copy host semantics
    (ordered host write for ANY fill value) and produces the same results as
    the knob-enabled deferred GPU-encoded path (the deferred comparison leg
    forces ``WARP_METAL_EAGER_GPU_FILLS=1`` — the sync-vs-deferred parity is
    only interesting against the GPU encoding, not against the default host
    path both legs would otherwise share)."""

    N = 6

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    @staticmethod
    @contextmanager
    def _sync_eager():
        old = os.environ.get("WARP_METAL_SYNC_EAGER")
        os.environ["WARP_METAL_SYNC_EAGER"] = "1"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_SYNC_EAGER", None)
            else:
                os.environ["WARP_METAL_SYNC_EAGER"] = old

    @staticmethod
    @contextmanager
    def _deferred():
        # explicit deferred mode: the suite may itself run under
        # WARP_METAL_SYNC_EAGER=1, so the comparison leg must force it off
        old = os.environ.pop("WARP_METAL_SYNC_EAGER", None)
        try:
            yield
        finally:
            if old is not None:
                os.environ["WARP_METAL_SYNC_EAGER"] = old

    @staticmethod
    @contextmanager
    def _deferred_gpu_fills():
        # deferred + the opt-in GPU encoding (round-2b mechanism): the
        # parity target for the sync-mode host writes
        with TestSyncEagerGPUOpsParity._deferred():
            old = os.environ.get("WARP_METAL_EAGER_GPU_FILLS")
            os.environ["WARP_METAL_EAGER_GPU_FILLS"] = "1"
            try:
                yield
            finally:
                if old is None:
                    os.environ.pop("WARP_METAL_EAGER_GPU_FILLS", None)
                else:
                    os.environ["WARP_METAL_EAGER_GPU_FILLS"] = old

    def test_sync_eager_fill_any_value_host_write(self):
        # the escape hatch restores the round-1 semantics: fill_(5.0)
        # WORKS here (ordered host write), no GPU fill encoded
        with self._sync_eager():
            with _SpiedNative() as calls:
                f = wp.zeros(3, dtype=wp.float32, device="metal")
                f.fill_(5.0)
                self.assertEqual(calls["compute_fill"], [])
                got = f.numpy.copy()
        np.testing.assert_allclose(got, np.full(3, 5.0), rtol=0, atol=1e-6)
        # knob-enabled deferred mode agrees on the RESULT (host path for
        # 5.0, GPU compute-section fill for -1):
        with self._deferred_gpu_fills():
            f = wp.zeros(3, dtype=wp.float32, device="metal")
            with _SpiedNative() as calls:
                f.fill_(5.0)
                self.assertEqual(calls["compute_fill"], [])  # not byte-exact: host
            np.testing.assert_allclose(f.numpy.copy(), np.full(3, 5.0), rtol=0, atol=1e-6)
            i = wp.zeros(3, dtype=wp.int32, device="metal")
            with _SpiedNative() as calls:
                i.fill_(-1)
                self.assertEqual(len(calls["compute_fill"]), 1)  # GPU-encoded
            np.testing.assert_array_equal(i.numpy.copy(), np.full(3, -1, dtype=np.int32))

    def test_sync_eager_copy_host_path_matches_deferred(self):
        x_np = np.arange(1, self.N + 1, dtype=np.float32)

        def run():
            x = wp.from_numpy(x_np.copy(), device="metal")
            y = wp.zeros(self.N, dtype=wp.float32, device="metal")
            m = wp.zeros(self.N, dtype=wp.int32, device="metal")
            out = wp.zeros(self.N, dtype=wp.float32, device="metal")
            wp.copy(y, x)
            m.zero_()
            m.fill_(-1)
            wp.launch(_double, self.N, inputs=[y, out], device="metal")
            return y.numpy.copy(), m.numpy.copy(), out.numpy.copy()

        with self._sync_eager():
            with _SpiedNative() as calls:
                synced = run()
                self.assertEqual(calls["compute_fill"], [])
                self.assertEqual(calls["compute_copy"], [])
        with self._deferred_gpu_fills():
            deferred = run()
        for s, d in zip(synced, deferred):
            np.testing.assert_array_equal(s, d)
        np.testing.assert_allclose(synced[0], x_np, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(synced[1], np.full(self.N, -1, dtype=np.int32))
        np.testing.assert_allclose(synced[2], 2.0 * x_np, rtol=0, atol=1e-6)


class TestDefaultEagerHostPath(unittest.TestCase):
    """The eager DEFAULT routes zero_/fill_/copy/clone to the round-1 ordered
    host path (flush + host write) — the round-2c safety flip (KEP D2).

    The GPU encodings of rounds 2/2b measured healthy at micro scale but
    regressed the Newton conveyor ~35x, so they ship only behind
    ``WARP_METAL_EAGER_GPU_FILLS=1`` (pinned by TestGPUEncodedFills /
    TestGPUEncodedCopies and the micro module). These tests FORCE the knob
    off (and SYNC_EAGER off) so they pin the default routing in every suite
    configuration: no ``compute_fill``/``compute_copy`` call may appear, the
    flush fires, and the results are exact for byte-replicable AND arbitrary
    values alike."""

    N = 9

    def setUp(self):
        if _SKIP:
            raise unittest.SkipTest(_SKIP_MSG)

    @staticmethod
    @contextmanager
    def _default_deferred():
        # force the shipping default: no SYNC_EAGER, no fill knob — whatever
        # the surrounding suite run set
        saved = {}
        for name in ("WARP_METAL_SYNC_EAGER", "WARP_METAL_EAGER_GPU_FILLS"):
            saved[name] = os.environ.pop(name, None)
        try:
            yield
        finally:
            for name, old in saved.items():
                if old is not None:
                    os.environ[name] = old

    def test_default_zero_is_ordered_host_write(self):
        # pending kernel writes 1..N, zero_ flushes FIRST (host write lands
        # after the kernel), a second kernel reads the zeros: out is zeros.
        # Spy: no compute_fill call — the GPU encoding must stay opt-in.
        buf = wp.zeros(self.N, dtype=wp.float32, device="metal")
        out = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with self._default_deferred():
            with _SpiedNative() as calls:
                wp.launch(_unique_fill, self.N, inputs=[buf], device="metal")
                buf.zero_()
                self.assertEqual(calls["compute_fill"], [])  # NOT GPU-encoded
                self.assertGreaterEqual(len(calls["flush"]), 1)  # ordered host write
                wp.launch(_double, self.N, inputs=[buf, out], device="metal")
                wp.flush()
            np.testing.assert_array_equal(buf.numpy_view(), np.zeros(self.N, dtype=np.float32))
            np.testing.assert_array_equal(out.numpy_view(), np.zeros(self.N, dtype=np.float32))

    def test_default_fill_any_value_host_write(self):
        # byte-replicable (-1) AND arbitrary (5.0, INT32_MAX) values all take
        # the ordered host path by default — no eligibility segmentation
        # without the knob
        i = wp.zeros(self.N, dtype=wp.int32, device="metal")
        f = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with self._default_deferred():
            with _SpiedNative() as calls:
                i.fill_(-1)  # byte-replicable, but the knob is off: host path
                f.fill_(5.0)
                i2 = wp.zeros(2, dtype=wp.int32, device="metal")
                i2.fill_(2147483647)
                self.assertEqual(calls["compute_fill"], [])
                self.assertGreaterEqual(len(calls["flush"]), 3)
            np.testing.assert_array_equal(i.numpy.copy(), np.full(self.N, -1, dtype=np.int32))
            np.testing.assert_allclose(f.numpy.copy(), np.full(self.N, 5.0), rtol=0, atol=1e-6)
            np.testing.assert_array_equal(i2.numpy.copy(), np.full(2, 2147483647, dtype=np.int32))

    def test_default_copy_host_path(self):
        # an eligible copy (equal dtype, same device, no overlap) still takes
        # the flush + host path by default
        x_np = np.arange(1, self.N + 1, dtype=np.float32)
        x = wp.from_numpy(x_np.copy(), device="metal")
        y = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with self._default_deferred():
            with _SpiedNative() as calls:
                wp.copy(y, x)
                self.assertEqual(calls["compute_copy"], [])  # NOT GPU-encoded
                self.assertGreaterEqual(len(calls["flush"]), 1)
            np.testing.assert_allclose(y.numpy.copy(), x_np, rtol=0, atol=1e-6)

    def test_default_clone_host_path(self):
        src_np = np.arange(1, self.N + 1, dtype=np.float32)
        src = wp.from_numpy(src_np.copy(), device="metal")
        with self._default_deferred():
            with _SpiedNative() as calls:
                c = wp.clone(src)
                self.assertEqual(calls["compute_copy"], [])  # NOT GPU-encoded
                self.assertGreaterEqual(len(calls["flush"]), 1)
                got = c.numpy.copy()
        np.testing.assert_allclose(got, src_np, rtol=0, atol=1e-6)

    def test_default_in_stream_fill_ordering(self):
        # the round-1 ordering contract on the default path: the fill lands
        # BETWEEN two launches (flush-before-write, later launch observes
        # the written bytes)
        buf = wp.zeros(self.N, dtype=wp.int32, device="metal")
        out = wp.zeros(self.N, dtype=wp.int32, device="metal")
        with self._default_deferred():
            wp.launch(_unique_fill_i32, self.N, inputs=[buf], device="metal")
            buf.fill_(-1)
            wp.launch(_double_i32, self.N, inputs=[buf, out], device="metal")
            wp.flush()
            np.testing.assert_array_equal(buf.numpy_view(), np.full(self.N, -1, dtype=np.int32))
            np.testing.assert_array_equal(out.numpy_view(), np.full(self.N, -2, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
