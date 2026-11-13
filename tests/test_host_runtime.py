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
import unittest

import numpy as np

import warp_metal as wp


@wp.func
def _twice(x: wp.float32) -> wp.float32:
    return x * 2.0


@wp.kernel
def _scale_kernel(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = _twice(a[i]) * 2.0


@wp.struct
class _Particle:
    pos: wp.vec3
    mass: wp.float32


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

    def test_func_native_fails_loud(self):
        with self.assertRaises(RuntimeError):
            wp.func_native("float f(float x) { return x; }")

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

    def test_from_numpy(self):
        n = wp.from_numpy(np.array([7.0, 8.0], dtype=np.float32))
        self.assertEqual(n.numpy.tolist(), [7.0, 8.0])
        with self.assertRaises(RuntimeError):
            wp.from_numpy(np.zeros(2, dtype=np.float32), requires_grad=True)

    def test_full(self):
        v = wp.full((2,), 3.0, dtype=wp.float32)
        self.assertTrue(np.all(v.numpy == 3.0))


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
        with self.assertRaises(NotImplementedError):
            wp.capture_if(True)
        with self.assertRaises(NotImplementedError):
            wp.capture_while(lambda: True)

    def test_capture_launch_rejects_foreign(self):
        with self.assertRaises(RuntimeError):
            wp.capture_launch(object())


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


class TestMiscHostApi(unittest.TestCase):
    def test_init_idempotent(self):
        wp.init()
        wp.init()

    def test_transform_expand(self):
        t = wp.transform_expand([1, 2, 3, 0, 0, 0, 1])
        np.testing.assert_allclose(np.asarray(t), [1, 2, 3, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            wp.transform_expand([1, 2, 3])

    def test_utils_stub_fails_loud(self):
        with self.assertRaises(NotImplementedError):
            wp.utils.segmented_sort_pairs

    def test_constants(self):
        self.assertEqual(wp.LOG_DEBUG, 10)
        self.assertEqual(wp.LOG_ERROR, 40)


if __name__ == "__main__":
    unittest.main()
