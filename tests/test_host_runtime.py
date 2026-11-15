# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M1.2 host runtime tests: decorators, device registry, array ctors,
copy/clone, capture passthrough, host transform builtins.

NOTE: kernels/functions defined at module import time (source-file backed)
so the vendored Adjoint can extract source.
"""

import ctypes
import os
import unittest
from contextlib import contextmanager

import numpy as np

import warp_metal as wp


@wp.func
def _twice(x: wp.float32) -> wp.float32:
    return x * 2.0


@wp.kernel
def _scale_kernel(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = _twice(a[i]) * 2.0


@wp.kernel
def _chain_add1(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] + 1.0


@wp.kernel
def _chain_mul2(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel
def _chain_add2(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def _tiled_fill(out: wp.array(dtype=wp.float32)):
    # launch_tiled: scalar tid is the block index; every lane of block i
    # writes the same value to cell i (unique-cell, race-free by value)
    i = wp.tid()
    out[i] = wp.float(i) + 1.0


@wp.struct
class _Particle:
    pos: wp.vec3
    mass: wp.float32


@wp.struct
class _Bin:
    vals: wp.array(dtype=wp.float32)
    keys: wp.array(dtype=wp.int32)
    scale: wp.float32


@wp.kernel
def _struct_fill(b: _Bin, n: wp.int32):
    i = wp.tid()
    if i < n:
        b.vals[i] = b.vals[i] * b.scale
        b.keys[i] = b.keys[i] + i


class TestDecorators(unittest.TestCase):
    def test_kernel_record(self):
        self.assertIsInstance(_scale_kernel, wp.Kernel)
        self.assertEqual([a.label for a in _scale_kernel.adj.args], ["a", "out"])
        self.assertFalse(_scale_kernel.is_generic)
        self.assertIn(_scale_kernel.key, wp.get_module(_scale_kernel.func.__module__).kernels)

    def test_func_record(self):
        self.assertIsInstance(_twice, wp.Function)
        self.assertEqual(_twice.input_types.get("x"), wp.float32)

    def test_kernel_options_validation(self):
        with self.assertRaises(ValueError):
            wp.kernel(entry_point_abi="bogus")
        with self.assertRaises(ValueError):
            wp.kernel(cuda_max_registers=4, launch_bounds=8)
        # recorded but inert options
        @wp.kernel(enable_backward=False, cuda_max_registers=32, name="optcheck")
        def k(x: wp.array(dtype=wp.float32)):
            i = wp.tid()
            x[i] = 0.0
        self.assertEqual(k.options.get("cuda_max_registers"), 32)

    def test_kernel_decorator_requires_annotations(self):
        with self.assertRaises(Exception):
            @wp.kernel
            def k(x):
                pass

    def test_func_native_policy(self):
        # func_native snippets are portable C++ (upstream runs the same text
        # on the CPU backend), so the MSL emitter includes them verbatim; MSL
        # is C++14-based. Plain-C++ snippets work; CUDA-only constructs fail
        # loudly at Metal library compile — never silently.
        @wp.func_native(
            "uint32_t i = reinterpret_cast<uint32_t&>(f);\n"
            "uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000u;\n"
            "return i ^ mask;"
        )
        def float_flip(f: wp.float32) -> wp.uint32: ...

        @wp.kernel
        def k(out: wp.array(dtype=wp.uint32)):
            i = wp.tid()
            out[i] = float_flip(4.5)

        arr = wp.zeros(1, dtype=wp.uint32, device="metal")
        wp.launch(k, 1, inputs=[arr], device="metal")
        wp.synchronize()
        # stereopsis radix key for a positive float: 0x80000000 | bits
        expected = 0x80000000 | int(np.array([4.5], dtype=np.float32).view(np.uint32)[0])
        self.assertEqual(int(arr.numpy_view()[0]), expected)

        @wp.func_native("return __shfl_down_sync(0xffffffffu, value, 1);")
        def shfl_native(value: wp.float32) -> wp.float32: ...

        @wp.kernel
        def k2(out: wp.array(dtype=wp.float32)):
            out[wp.tid()] = shfl_native(1.0)

        with self.assertRaises(Exception):
            # CUDA intrinsic survives translation (verbatim inclusion) but
            # the Metal compiler rejects it — loud, at the compile boundary
            wp.launch(k2, 1, inputs=[wp.zeros(1, dtype=wp.float32, device="metal")], device="metal")
            wp.synchronize()

    def test_generic_kernel_overload(self):
        from typing import Any

        @wp.kernel
        def fill(x: wp.array(dtype=Any), v: Any):
            i = wp.tid()
            x[i] = v
        self.assertTrue(fill.is_generic)
        ovl = fill.add_overload([wp.array(dtype=wp.float32), wp.float32])
        self.assertFalse(ovl.is_generic)
        self.assertIs(fill.get_overload([wp.array(dtype=wp.float32), wp.float32]), ovl)


class TestDevices(unittest.TestCase):
    def test_device_registry(self):
        cpu = wp.get_device("cpu")
        self.assertTrue(cpu.is_cpu)
        self.assertFalse(cpu.is_cuda)
        metal = wp.get_device("metal")
        self.assertTrue(metal.is_metal)
        self.assertFalse(metal.is_cpu)
        self.assertIs(wp.get_device(metal), metal)
        with self.assertRaises(ValueError):
            wp.get_device("cuda:0")

    def test_preferred_and_cuda(self):
        self.assertFalse(wp.is_cuda_available())
        self.assertEqual(wp.get_preferred_device(), wp.get_device("metal"))

    def test_scoped_device(self):
        prev = wp.get_current_device()
        with wp.ScopedDevice("cpu"):
            self.assertEqual(wp.get_current_device().name, "cpu")
        self.assertIs(wp.get_current_device(), prev)

    def test_stream_and_event(self):
        s = wp.get_stream("metal")
        self.assertEqual(s.device.name, "metal")
        wp.synchronize_stream(s)
        e1, e2 = wp.Event("metal"), wp.Event("metal")
        e1.record()
        e2.record()
        self.assertGreaterEqual(wp.get_event_elapsed_time(e1, e2), 0.0)


class TestArrayConstructors(unittest.TestCase):
    def test_shape_and_device(self):
        a = wp.zeros((3, 4), dtype=wp.float32, device="metal")
        self.assertEqual(a.shape, (3, 4))
        self.assertEqual(a.n, 12)
        self.assertEqual(a.device, "metal")
        self.assertTrue(np.all(a.numpy == 0))

    def test_like_variants_and_copy(self):
        a = wp.zeros(4, dtype=wp.float32)
        a.numpy[:] = [1, 2, 3, 4]
        b = wp.ones_like(a)
        self.assertTrue(np.all(b.numpy == 1))
        c = wp.empty_like(a)
        wp.copy(c, a)
        self.assertEqual(c.numpy.tolist(), [1, 2, 3, 4])
        d = wp.clone(a, device="metal")
        self.assertEqual(d.device, "metal")
        self.assertEqual(d.numpy.tolist(), [1, 2, 3, 4])

    def test_grad_flags_fail_loud(self):
        with self.assertRaises(RuntimeError):
            wp.zeros(4, requires_grad=True)
        a = wp.zeros(4, dtype=wp.float32)
        with self.assertRaises(RuntimeError):
            wp.clone(a, requires_grad=True)

    def test_dtype_builtin_aliases_canonicalize(self):
        # upstream array.__init__ canonicalize block (types.py:3559-3565):
        # Python builtins map to their 32-bit warp types at construction, so
        # Newton's wp.full(n, 0.8, dtype=float) yields a float32 array
        for ctor, kwargs in (
            (wp.zeros, {"dtype": float}),
            (wp.full, {"value": 0.8, "dtype": float}),
            (wp.empty, {"dtype": float}),
        ):
            a = ctor(4, device="cpu", **kwargs)
            self.assertIs(a.dtype, wp.float32, f"{ctor.__name__}: {a.dtype}")
        self.assertIs(wp.zeros(4, dtype=int, device="cpu").dtype, wp.int32)
        self.assertIs(wp.zeros(4, dtype=bool, device="cpu").dtype, wp.bool)
        # data flows through the canonical dtype
        f = wp.full(3, 0.5, dtype=float, device="cpu")
        self.assertTrue(np.all(f.numpy == np.float32(0.5)))

    def test_copy_element_size_contract(self):
        # upstream wp.copy (context.py:15158): both sides must be arrays and
        # the copy is legal whenever element sizes match in bytes — dtype
        # identity is not required
        i = wp.from_numpy(np.array([1, 2], dtype=np.int32), device="cpu")
        f = wp.zeros(2, dtype=wp.float32, device="cpu")
        wp.copy(f, i)  # same 4-byte element: raw memcpy, upstream-legal
        self.assertEqual(f.numpy.tolist(), [1.0, 2.0])  # int32 bits reinterpreted
        with self.assertRaises(RuntimeError):
            wp.copy(f, object())
        v = wp.zeros(2, dtype=wp.vec3f, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "Incompatible array data types"):
            wp.copy(f, v)  # 12-byte vs 4-byte elements

    def test_from_numpy(self):
        n = wp.from_numpy(np.array([7.0, 8.0], dtype=np.float32))
        self.assertEqual(n.numpy.tolist(), [7.0, 8.0])
        with self.assertRaises(RuntimeError):
            wp.from_numpy(np.zeros(2, dtype=np.float32), requires_grad=True)

    def test_from_numpy_composite(self):
        # 2-D numpy + composite dtype -> 1-D array of elements (warp semantics)
        v = wp.from_numpy(np.arange(12, dtype=np.float32).reshape(4, 3), dtype=wp.vec3)
        self.assertEqual(v.shape, (4,))
        self.assertEqual(v.ndim, 1)
        np.testing.assert_allclose(v.numpy_view(), np.arange(12, dtype=np.float32).reshape(4, 3))
        t = wp.from_numpy(np.tile([1, 2, 3, 0, 0, 0, 1], (2, 1)).astype(np.float32), dtype=wp.transform)
        self.assertEqual(t.shape, (2,))
        np.testing.assert_allclose(t.numpy_view()[1], [1, 2, 3, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            wp.from_numpy(np.zeros(5, dtype=np.float32), dtype=wp.vec3)

    def test_runtime_array_is_native_family(self):
        # runtime WarpArray substitutes for the vendored wp.array family:
        # accepted by struct field setters and exposed via __ctype__
        @wp.struct
        class _Buf:
            data: wp.array(dtype=wp.float32)
            count: wp.int32

        b = _Buf()
        b.data = wp.zeros(3, dtype=wp.float32)
        b.count = 3
        self.assertEqual(b.data.ndim, 1)
        self.assertFalse(b.data.requires_grad)
        desc = b.data.__ctype__()
        self.assertEqual(int(desc.ndim), 1)
        self.assertNotEqual(int(desc.data), 0)

    def test_full(self):
        v = wp.full((2,), 3.0, dtype=wp.float32)
        self.assertTrue(np.all(v.numpy == 3.0))


class TestStructArrayFieldLaunch(unittest.TestCase):
    """M1.4: struct params with array fields split into their own bindings."""

    def test_struct_array_fields_launch(self):
        b = _Bin()
        b.vals = wp.zeros(3, dtype=wp.float32)
        b.vals.numpy[:] = [1.0, 2.0, 3.0]
        b.keys = wp.zeros(3, dtype=wp.int32)
        b.keys.numpy[:] = [10, 20, 30]
        b.scale = 2.0
        wp.launch(_struct_fill, 3, inputs=[b, 3], device="metal")
        wp.synchronize()
        np.testing.assert_allclose(b.vals.numpy_view(), [2.0, 4.0, 6.0])
        self.assertEqual(b.keys.numpy_view().tolist(), [10, 21, 32])

    def test_struct_array_field_dtype_check(self):
        b = _Bin()
        with self.assertRaises(TypeError):
            b.vals = wp.zeros(3, dtype=wp.int32)  # field expects float32


class TestLaunchAdjointFails(unittest.TestCase):
    def test_adjoint_kwargs_rejected(self):
        a = wp.zeros(4, dtype=wp.float32)
        with self.assertRaises(RuntimeError):
            wp.launch(_scale_kernel, dim=4, inputs=[a, a], adjoint=True)
        with self.assertRaises(RuntimeError):
            wp.launch(_scale_kernel, dim=4, inputs=[a, a], record_tape=True)


class TestCapturePassthrough(unittest.TestCase):
    def test_begin_end_passthrough(self):
        wp.capture_begin()
        g = wp.capture_end()
        self.assertIsInstance(g, wp.Graph)
        self.assertEqual(g.n_ops, 0)
        wp.capture_launch(g)  # empty replay: honest no-op

    def test_graph_constructs_fail_loud(self):
        # capture_if outside capture = upstream eager semantics (callable
        # dispatch on the condition element); Graph callbacks and
        # capture-INTERNAL capture_while stay the M2.5 boundary and fail
        # loud. capture_while OUTSIDE capture is upstream's no-graph eager
        # fallback (context.py "just execute the kernels directly"), so a
        # foreign condition still fails loud with the type error.
        cond = wp.zeros(1, dtype=wp.int32, device="metal")
        called = []

        wp.capture_if(cond, on_true=lambda: called.append(1))
        self.assertEqual(called, [])
        wp.capture_if(cond, on_false=lambda: called.append(2))
        self.assertEqual(called, [2])
        cond.numpy[0] = 1
        wp.capture_if(cond, on_true=lambda: called.append(3))
        self.assertEqual(called, [2, 3])
        with self.assertRaises(TypeError):
            wp.capture_if(cond, on_true=object())
        with self.assertRaises(TypeError):
            wp.capture_while(lambda: True)  # condition must be an int32 WarpArray

    def test_capture_launch_rejects_foreign(self):
        with self.assertRaises(RuntimeError):
            wp.capture_launch(object())


class TestCaptureParityAttrs(unittest.TestCase):
    """M2.5 commit 3: Device/Stream capture-parity attributes. Newton's
    capture guards read ``device.is_capturing``, ``device.captures.get(
    device.stream) is not None`` and ``wp.get_stream(d).is_capturing``;
    under warp-metal these must observe the real capture state (or be a
    documented False), not AttributeError."""

    def test_outside_capture(self):
        dev = wp.get_device("metal")
        self.assertFalse(dev.is_capturing)
        self.assertEqual(dev.captures, {})
        self.assertIsNone(dev.captures.get(dev.stream))
        self.assertFalse(dev.stream.is_capturing)
        self.assertFalse(wp.get_stream(dev).is_capturing)
        self.assertFalse(dev.is_mempool_enabled)

    def test_stream_token_is_stable_and_shared(self):
        dev = wp.get_device("metal")
        self.assertIs(dev.stream, wp.get_device("metal").stream)
        self.assertIs(dev.stream, wp.get_stream(dev))
        self.assertIs(wp.get_stream("metal").device, dev)
        self.assertIsNot(wp.get_device("cpu").stream, dev.stream)
        # hashable dict key, like a cudaStream handle upstream
        self.assertIn(dev.stream, {dev.stream: None})

    def test_inside_capture(self):
        dev = wp.get_device("metal")
        cpu = wp.get_device("cpu")
        with wp.ScopedCapture(device="metal") as c:
            self.assertTrue(dev.is_capturing)
            self.assertFalse(cpu.is_capturing)  # reports only on the target device
            capture = dev.captures.get(dev.stream)
            self.assertIsNotNone(capture)
            self.assertIsInstance(capture, wp.Graph)
            self.assertIsNone(cpu.captures.get(cpu.stream))
            self.assertEqual(list(dev.captures.keys()), [dev.stream])
        self.assertFalse(dev.is_capturing)
        self.assertEqual(dev.captures, {})
        self.assertIsNone(dev.captures.get(dev.stream))
        # the object observed during recording is the one capture_end returned
        self.assertIs(capture, c.graph)
        self.assertEqual(c.graph.n_ops, 0)  # nothing launched inside this block

    def test_capture_if_inside_capture_raises_named(self):
        cond = wp.zeros(1, dtype=wp.int32, device="metal")
        called = []
        with wp.ScopedCapture(device="metal") as c:
            with self.assertRaises(wp.CaptureStateError) as ctx:
                wp.capture_if(cond, on_true=lambda: called.append(1))
            message = str(ctx.exception)
            self.assertIn("capture_if", message)
            self.assertIn("conditional graph node", message)
            self.assertIn("outside", message)
            self.assertEqual(called, [])  # no eager execution either
        # outside capture: upstream's own eager non-CUDA semantics stay
        cond.numpy[0] = 1
        wp.capture_if(cond, on_true=lambda: called.append(2))
        self.assertEqual(called, [2])
        self.assertEqual(c.graph.n_ops, 0)  # nothing silently recorded


class TestCaptureReplay(unittest.TestCase):
    """M2.5 batched replay: a recorded graph replays equivalent to eager
    execution, ordering preserved across mixed kernel/copy/fill records
    (eligible copies and zero fills encode as blit sections on the batch's
    command buffer), and WARP_METAL_BATCH=0 (record-by-record) agrees with
    the batched path."""

    N = 6

    @staticmethod
    @contextmanager
    def _spied_batch():
        """Replace the native batch wrapper with a recorder and pin batching
        mode for the wrapped block (the spy tests pin batch-composition
        semantics, so they must not inherit the ambient WARP_METAL_BATCH
        escape-hatch setting). Yields the call list; each call is recorded as
        the list of payload records it dispatched (tuples whose [0] is the
        kind string)."""
        from warp_metal import native as mm

        calls = []
        real = mm.dispatch_batch
        old = os.environ.get("WARP_METAL_BATCH")
        os.environ["WARP_METAL_BATCH"] = "1"
        mm.dispatch_batch = lambda records: calls.append(list(records))
        try:
            yield calls
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_BATCH", None)
            else:
                os.environ["WARP_METAL_BATCH"] = old
            mm.dispatch_batch = real

    def test_dependency_chain_replay_matches_eager(self):
        # k1 -> k2 -> k3 (k3 consumes both intermediates): the replay must
        # produce the same buffers an eager execution of the same kernels does
        x_np = np.arange(1, self.N + 1, dtype=np.float32)

        # eager reference on separate buffers
        ex = wp.from_numpy(x_np.copy(), device="metal")
        et1 = wp.zeros(self.N, dtype=wp.float32, device="metal")
        et2 = wp.zeros(self.N, dtype=wp.float32, device="metal")
        eout = wp.zeros(self.N, dtype=wp.float32, device="metal")
        wp.launch(_chain_add1, self.N, inputs=[ex, et1], device="metal")
        wp.launch(_chain_mul2, self.N, inputs=[et1, et2], device="metal")
        wp.launch(_chain_add2, self.N, inputs=[et1, et2, eout], device="metal")

        # captured: same kernels, fresh buffers, deferred execution
        cx = wp.from_numpy(x_np.copy(), device="metal")
        ct1 = wp.zeros(self.N, dtype=wp.float32, device="metal")
        ct2 = wp.zeros(self.N, dtype=wp.float32, device="metal")
        cout = wp.zeros(self.N, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_chain_add1, self.N, inputs=[cx, ct1], device="metal")
            wp.launch(_chain_mul2, self.N, inputs=[ct1, ct2], device="metal")
            wp.launch(_chain_add2, self.N, inputs=[ct1, ct2, cout], device="metal")
        self.assertIsInstance(c.graph, wp.Graph)
        self.assertEqual(c.graph.n_ops, 3)
        # deferred: nothing executed at record time
        np.testing.assert_array_equal(ct1.numpy_view(), np.zeros(self.N, dtype=np.float32))
        wp.capture_launch(c.graph)
        np.testing.assert_allclose(ct1.numpy_view(), x_np + 1.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(ct2.numpy_view(), 2.0 * (x_np + 1.0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(cout.numpy_view(), 3.0 * x_np + 3.0, rtol=0, atol=1e-6)
        # replay equals eager, buffer for buffer
        np.testing.assert_allclose(ct1.numpy_view(), et1.numpy_view(), rtol=0, atol=0)
        np.testing.assert_allclose(ct2.numpy_view(), et2.numpy_view(), rtol=0, atol=0)
        np.testing.assert_allclose(cout.numpy_view(), eout.numpy_view(), rtol=0, atol=0)

    def test_mixed_graph_replay_order(self):
        # launch + copy + fill interleaved: replay order is discriminating —
        # dst reads mid BEFORE the fill (else dst would be 0) and out is
        # launched AFTER the fill (else out would be 2r+1)
        r_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        r = wp.from_numpy(r_np.copy(), device="metal")
        mid = wp.zeros(5, dtype=wp.float32, device="metal")
        dst = wp.zeros(5, dtype=wp.float32, device="metal")
        out = wp.zeros(5, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_chain_mul2, 5, inputs=[r, mid], device="metal")  # mid = 2r
            wp.copy(dst, mid)  # dst = 2r (recorded, runs at replay)
            mid.zero_()  # mid = 0 (record-time host write + replayed)
            wp.launch(_chain_add1, 5, inputs=[mid, out], device="metal")  # out = 1s
        self.assertEqual(c.graph.n_ops, 4)
        # record-time semantics: fills write immediately, copies do not
        np.testing.assert_array_equal(mid.numpy_view(), np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(dst.numpy_view(), np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(out.numpy_view(), np.zeros(5, dtype=np.float32))
        wp.capture_launch(c.graph)
        np.testing.assert_allclose(dst.numpy_view(), 2.0 * r_np, rtol=0, atol=1e-6)
        np.testing.assert_allclose(mid.numpy_view(), np.zeros(5, dtype=np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(out.numpy_view(), np.ones(5, dtype=np.float32), rtol=0, atol=1e-6)

    def test_batched_vs_unbatched_identical(self):
        # one graph mixing plain kernel records, a tiled record (exact
        # threadgroup size) and nothing else; replayed once batched (default)
        # and once with WARP_METAL_BATCH=0 — results must be identical
        def run(env_value):
            old = os.environ.get("WARP_METAL_BATCH")
            os.environ["WARP_METAL_BATCH"] = env_value
            try:
                x = wp.from_numpy(np.arange(1, self.N + 1, dtype=np.float32), device="metal")
                t1 = wp.zeros(self.N, dtype=wp.float32, device="metal")
                t2 = wp.zeros(self.N, dtype=wp.float32, device="metal")
                t3 = wp.zeros(2, dtype=wp.float32, device="metal")
                out = wp.zeros(self.N, dtype=wp.float32, device="metal")
                with wp.ScopedCapture() as c:
                    wp.launch(_chain_add1, self.N, inputs=[x, t1], device="metal")
                    wp.launch(_chain_mul2, self.N, inputs=[t1, t2], device="metal")
                    wp.launch_tiled(_tiled_fill, dim=2, inputs=[t3], block_dim=32, device="metal")
                    wp.launch(_chain_add2, self.N, inputs=[t1, t2, out], device="metal")
                self.assertEqual(c.graph.n_ops, 4)
                wp.capture_launch(c.graph)
                return (
                    t1.numpy.copy(),
                    t2.numpy.copy(),
                    t3.numpy.copy(),
                    out.numpy.copy(),
                )
            finally:
                if old is None:
                    os.environ.pop("WARP_METAL_BATCH", None)
                else:
                    os.environ["WARP_METAL_BATCH"] = old

        batched = run("1")
        unbatched = run("0")
        x_np = np.arange(1, self.N + 1, dtype=np.float32)
        expected = (
            x_np + 1.0,
            2.0 * (x_np + 1.0),
            np.array([1.0, 2.0], dtype=np.float32),
            3.0 * x_np + 3.0,
        )
        for got, want in zip(batched, expected):
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
        for b, u in zip(batched, unbatched):
            np.testing.assert_array_equal(b, u)

    def test_blit_batch_single_native_call(self):
        # launch + copy + zero_ + launch: the copy and the zero fill are
        # blit-eligible, so the whole graph replays through ONE native batch
        # call (compute | copy | fill | compute sections on one command
        # buffer) instead of the pre-blit two-batch segmentation
        r_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        r = wp.from_numpy(r_np.copy(), device="metal")
        mid = wp.zeros(5, dtype=wp.float32, device="metal")
        dst = wp.zeros(5, dtype=wp.float32, device="metal")
        out = wp.zeros(5, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_chain_mul2, 5, inputs=[r, mid], device="metal")  # mid = 2r
            wp.copy(dst, mid)  # dst = 2r (blit copy at replay)
            mid.zero_()  # mid = 0 (blit fill at replay)
            wp.launch(_chain_add1, 5, inputs=[mid, out], device="metal")  # out = 1s
        with self._spied_batch() as calls:
            wp.capture_launch(c.graph)  # spied: payload recorded, not executed
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual([r[0] for r in calls[0]], ["compute", "copy", "fill", "compute"])
        # the fill payload covers the whole mid buffer: 5 float32 = 20 bytes
        # at buffer offset 0, value 0
        fill_payload = next(r for r in calls[0] if r[0] == "fill")
        self.assertEqual(fill_payload[3], 20)
        self.assertEqual(fill_payload[2], 0)
        self.assertEqual(fill_payload[4], 0)
        # spied run did not execute: copy/fill effects must be absent
        np.testing.assert_array_equal(dst.numpy_view(), np.zeros(5, dtype=np.float32))
        # real replay (idempotent): now the values land
        wp.capture_launch(c.graph)
        np.testing.assert_allclose(dst.numpy_view(), 2.0 * r_np, rtol=0, atol=1e-6)
        np.testing.assert_allclose(mid.numpy_view(), np.zeros(5, dtype=np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(out.numpy_view(), np.ones(5, dtype=np.float32), rtol=0, atol=1e-6)

    def test_copy_offsets_blit_matches_eager(self):
        # slices/views: offset_elems folds into the blit byte offsets on both
        # sides (plus explicit element offsets and a fill on a slice view).
        # The destination is poisoned so a dropped copy or fill is caught.
        src_np = np.arange(10, dtype=np.float32)
        esrc = wp.from_numpy(src_np.copy(), device="metal")
        edst = wp.zeros(10, dtype=wp.float32, device="metal")
        edst.numpy[...] = 9.0
        wp.copy(edst[3:9], esrc[2:8], dest_offset=1, src_offset=2, count=3)  # dst[4:7] = [4,5,6]
        ef = edst[7:10]
        ef.zero_()  # buffer bytes [28:40)
        efill = wp.zeros(3, dtype=wp.float32, device="metal")
        efill.numpy[:] = [7.0, 7.0, 7.0]
        wp.copy(edst[0:3], efill)  # head = [7,7,7]

        csrc = wp.from_numpy(src_np.copy(), device="metal")
        cdst = wp.zeros(10, dtype=wp.float32, device="metal")
        cdst.numpy[...] = 9.0
        cfill = wp.zeros(3, dtype=wp.float32, device="metal")
        cfill.numpy[:] = [7.0, 7.0, 7.0]
        with wp.ScopedCapture() as c:
            wp.copy(cdst[3:9], csrc[2:8], dest_offset=1, src_offset=2, count=3)
            cf = cdst[7:10]
            cf.zero_()
            wp.copy(cdst[0:3], cfill)
        with self._spied_batch() as calls:
            wp.capture_launch(c.graph)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual([r[0] for r in calls[0]], ["copy", "fill", "copy"])
        wp.capture_launch(c.graph)  # real replay (the spied run executed nothing)
        np.testing.assert_array_equal(cdst.numpy_view(), edst.numpy_view())
        np.testing.assert_array_equal(
            cdst.numpy_view(), [7.0, 7.0, 7.0, 9.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]
        )

    def test_nonzero_fill_segments_and_replays(self):
        # fillBuffer is an 8-bit pattern: a non-zero fill is not
        # batch-eligible — it segments the batch (two native calls, each
        # compute-only) and applies on the host between them
        r_np = np.arange(1.0, 5.0, dtype=np.float32)
        r = wp.from_numpy(r_np.copy(), device="metal")
        mid = wp.zeros(4, dtype=wp.float32, device="metal")
        out = wp.zeros(4, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_chain_mul2, 4, inputs=[r, mid], device="metal")  # mid = 2r
            mid.fill_(2.5)  # mid = 2.5 (host op, splits the batch)
            wp.launch(_chain_add1, 4, inputs=[mid, out], device="metal")  # out = 3.5
        with self._spied_batch() as calls:
            wp.capture_launch(c.graph)  # spied: batch recorded, not executed
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual([[r[0] for r in c] for c in calls], [["compute"], ["compute"]])
        wp.capture_launch(c.graph)  # real replay (idempotent)
        np.testing.assert_allclose(mid.numpy_view(), np.full(4, 2.5, dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(out.numpy_view(), np.full(4, 3.5, dtype=np.float32), rtol=0, atol=1e-6)

    def test_composite_copy_blit_matches_eager(self):
        # composite dtypes (vec3 / quat arrays): same dtype means the same
        # byte-identical dense element layout, so the byte-range blit copy
        # is exact — encoded GPU-side (one batch, 'copy' section)
        v_np = np.arange(9, dtype=np.float32).reshape(3, 3)
        q_np = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        ev = wp.from_numpy(v_np.copy(), dtype=wp.vec3, device="metal")
        ev_dst = wp.zeros(3, dtype=wp.vec3, device="metal")
        eq = wp.from_numpy(q_np.copy(), dtype=wp.quat, device="metal")
        eq_dst = wp.zeros(2, dtype=wp.quat, device="metal")
        wp.copy(ev_dst, ev)
        wp.copy(eq_dst, eq)

        cv = wp.from_numpy(v_np.copy(), dtype=wp.vec3, device="metal")
        cv_dst = wp.zeros(3, dtype=wp.vec3, device="metal")
        cq = wp.from_numpy(q_np.copy(), dtype=wp.quat, device="metal")
        cq_dst = wp.zeros(2, dtype=wp.quat, device="metal")
        with wp.ScopedCapture() as c:
            wp.copy(cv_dst, cv)
            wp.copy(cq_dst, cq)
        with self._spied_batch() as calls:
            wp.capture_launch(c.graph)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual([r[0] for r in calls[0]], ["copy", "copy"])
        wp.capture_launch(c.graph)  # real replay (the spied run executed nothing)
        np.testing.assert_allclose(cv_dst.numpy_view(), ev_dst.numpy_view(), rtol=0, atol=0)
        np.testing.assert_allclose(cq_dst.numpy_view(), eq_dst.numpy_view(), rtol=0, atol=0)

    def test_overlapping_copy_segments_to_host(self):
        # overlapping src/dst ranges in one buffer have undefined blit
        # semantics: the copy segments to the host numpy copy (no 'copy'
        # section reaches the batch) and still replays correctly
        base_np = np.arange(8, dtype=np.float32)
        ebase = wp.from_numpy(base_np.copy(), device="metal")
        wp.copy(ebase[2:8], ebase[0:6])  # shifts elements 0..5 up by 2
        cbase = wp.from_numpy(base_np.copy(), device="metal")
        filler = wp.zeros(2, dtype=wp.float32, device="metal")
        with wp.ScopedCapture() as c:
            wp.launch(_chain_mul2, 2, inputs=[filler, filler], device="metal")  # opens a batch
            wp.copy(cbase[2:8], cbase[0:6])  # overlapping: host path at replay
            wp.launch(_chain_mul2, 2, inputs=[filler, filler], device="metal")
        with self._spied_batch() as calls:
            wp.capture_launch(c.graph)  # host copy applies between the spied batches
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual([[r[0] for r in c] for c in calls], [["compute"], ["compute"]])
        np.testing.assert_array_equal(cbase.numpy_view(), ebase.numpy_view())

    def test_batched_vs_unbatched_mixed_graph(self):
        # mixed kernel/copy/zero_ graph: the batched path (blit sections in
        # one command buffer) and WARP_METAL_BATCH=0 (record-by-record, host
        # copy/fill) must produce identical buffers
        def run(env_value):
            old = os.environ.get("WARP_METAL_BATCH")
            os.environ["WARP_METAL_BATCH"] = env_value
            try:
                x = wp.from_numpy(np.arange(1, self.N + 1, dtype=np.float32), device="metal")
                t1 = wp.zeros(self.N, dtype=wp.float32, device="metal")
                t2 = wp.zeros(self.N, dtype=wp.float32, device="metal")
                dst = wp.zeros(self.N, dtype=wp.float32, device="metal")
                out = wp.zeros(self.N, dtype=wp.float32, device="metal")
                with wp.ScopedCapture() as c:
                    wp.launch(_chain_add1, self.N, inputs=[x, t1], device="metal")  # t1 = x+1
                    wp.copy(dst, t1)  # dst = x+1
                    t2.zero_()  # t2 = 0 before the next kernel
                    wp.launch(_chain_mul2, self.N, inputs=[t1, t2], device="metal")  # t2 = 2(x+1)
                    wp.launch(_chain_add2, self.N, inputs=[dst, t2, out], device="metal")  # 3(x+1)
                self.assertEqual(c.graph.n_ops, 5)
                wp.capture_launch(c.graph)
                return (
                    t1.numpy.copy(),
                    t2.numpy.copy(),
                    dst.numpy.copy(),
                    out.numpy.copy(),
                )
            finally:
                if old is None:
                    os.environ.pop("WARP_METAL_BATCH", None)
                else:
                    os.environ["WARP_METAL_BATCH"] = old

        batched = run("1")
        unbatched = run("0")
        x_np = np.arange(1, self.N + 1, dtype=np.float32)
        expected = (
            x_np + 1.0,
            2.0 * (x_np + 1.0),
            x_np + 1.0,
            3.0 * (x_np + 1.0),
        )
        for got, want in zip(batched, expected):
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
        for b, u in zip(batched, unbatched):
            np.testing.assert_array_equal(b, u)

    def test_zero_record_capture_replays_as_noop(self):
        # ScopedCapture form of the empty-graph contract (the raw
        # begin/end/no-op variant lives in TestCapturePassthrough)
        with wp.ScopedCapture() as c:
            pass
        self.assertEqual(c.graph.n_ops, 0)
        wp.capture_launch(c.graph)  # honest no-op


class TestModuleHub(unittest.TestCase):
    def test_module_options(self):
        mod = wp.get_module("test_module_hub")
        wp.set_module_options({"enable_backward": False}, module=mod)
        self.assertEqual(wp.get_module_options(mod).get("enable_backward"), False)
        self.assertIs(wp.get_module("test_module_hub"), mod)

    def test_struct_registered_in_module(self):
        self.assertIsInstance(_Particle, wp.Struct)
        mod = wp.get_module(_Particle.cls.__module__)
        self.assertIn(_Particle.key, mod.structs)


class TestHostTransformBuiltins(unittest.TestCase):
    """Newton host_transform.py calls these via wp._src.context.runtime.core."""

    @classmethod
    def setUpClass(cls):
        cls.core = wp._src.context.runtime.core

    def test_transform_multiply(self):
        a = wp.transform()
        np.asarray(a)[:] = [1, 0, 0, 0, 0, 0, 1]  # translate x+1
        b = wp.transform()
        np.asarray(b)[:] = [2, 0, 0, 0, 0, 0, 1]  # translate x+2
        out = wp.transform()
        self.core.wp_builtin_mul_transformf_transformf(a, b, ctypes.byref(out))
        np.testing.assert_allclose(np.asarray(out), [3, 0, 0, 0, 0, 0, 1], atol=1e-6)

    def test_transform_point(self):
        t = wp.transform()
        np.asarray(t)[:] = [1, 0, 0, 0, 0, 0, 1]
        p = wp.vec3()
        np.asarray(p)[:] = [2, 0, 0]
        out = wp.vec3()
        self.core.wp_builtin_transform_point_transformf_vec3f(t, p, ctypes.byref(out))
        np.testing.assert_allclose(np.asarray(out), [3, 0, 0], atol=1e-6)

    def test_transform_inverse_roundtrip(self):
        # 90-degree rotation about z + translation
        q = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)
        t = wp.transform()
        np.asarray(t)[:] = [1, 2, 3, *q]
        inv = wp.transform()
        self.core.wp_builtin_transform_inverse_transformf(t, ctypes.byref(inv))
        out = wp.transform()
        self.core.wp_builtin_mul_transformf_transformf(t, inv, ctypes.byref(out))
        np.testing.assert_allclose(np.asarray(out)[:3], [0, 0, 0], atol=1e-5)
        np.testing.assert_allclose(np.asarray(out)[3:], [0, 0, 0, 1], atol=1e-5)

    def test_quat_from_matrix(self):
        # 90-degree rotation about z as row-major 4x4
        m = wp.mat44()
        np.asarray(m)[:] = [
            0, -1, 0, 0,
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
        q = wp.quat()
        self.core.wp_builtin_quat_from_matrix_mat44f(m, ctypes.byref(q))
        np.testing.assert_allclose(np.asarray(q), [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)], atol=1e-5)

    def test_transform_vector(self):
        t = wp.transform()
        np.asarray(t)[:] = [9, 9, 9, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # rotation must ignore translation
        v = wp.vec3()
        np.asarray(v)[:] = [1, 0, 0]
        out = wp.vec3()
        self.core.wp_builtin_transform_vector_transformf_vec3f(t, v, ctypes.byref(out))
        np.testing.assert_allclose(np.asarray(out), [0, 1, 0], atol=1e-5)

    def test_transform_compose_host_call(self):
        # Newton import_usd.py:687: wp.transform_compose(p, q, s) from host
        # Python. Upstream contract (math.py create_transform_compose_func):
        # M = [[sx*R00, sy*R01, sz*R02, px], [sx*R10, ...], ..., [0,0,0,1]]
        # with R = quat_to_matrix(rotation). 90-degree z rotation:
        # R = [[0,-1,0],[1,0,0],[0,0,1]]; p=(1,2,3), s=(2,3,4) ->
        # [[0,-3,0,1],[2,0,0,2],[0,0,4,3],[0,0,0,1]]
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)
        m = wp.transform_compose(wp.vec3(1.0, 2.0, 3.0), q, wp.vec3(2.0, 3.0, 4.0))
        expected = np.array(
            [
                [0.0, -3.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(m).reshape(4, 4), expected, atol=1e-5)
        # identity rotation/scale reduces to the pure translation matrix
        m2 = wp.transform_compose(wp.vec3(4.0, 5.0, 6.0), wp.quat_identity(), wp.vec3(1.0, 1.0, 1.0))
        np.testing.assert_allclose(
            np.asarray(m2).reshape(4, 4),
            np.array([[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]], dtype=np.float32),
            atol=1e-6,
        )


class TestMiscHostApi(unittest.TestCase):
    def test_init_idempotent(self):
        wp.init()
        wp.init()

    def test_transform_expand(self):
        t = wp.transform_expand([1, 2, 3, 0, 0, 0, 1])
        np.testing.assert_allclose(np.asarray(t), [1, 2, 3, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            wp.transform_expand([1, 2, 3])

    def test_utils_sorts_implemented_stubs_fail_loud(self):
        # steel-closure round: the segmented/radix sort pairs are real now;
        # out-of-closure consumers (implicit_mpm/VBD) still fail loud
        self.assertTrue(callable(wp.utils.segmented_sort_pairs))
        self.assertTrue(callable(wp.utils.radix_sort_pairs))
        with self.assertRaises(NotImplementedError):
            wp.utils.runlength_encode

    def test_constants(self):
        self.assertEqual(wp.LOG_DEBUG, 10)
        self.assertEqual(wp.LOG_ERROR, 40)


if __name__ == "__main__":
    unittest.main()
