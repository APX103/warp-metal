# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M0.3 acceptance: all 10 spike kernels emit structurally valid MSL."""

import unittest

from warp_metal.frontend import registry
from warp_metal.msl import MSLEmitter, SPIKE_KERNELS


class TestSpikeEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emitter = MSLEmitter(registry.builtin_functions)
        cls.sources = {}
        for kernel in SPIKE_KERNELS:
            cls.sources[kernel.__name__] = cls.emitter.emit(kernel)

    def test_all_ten_kernels_emit(self):
        self.assertEqual(len(self.sources), 10)
        self.assertEqual(set(self.sources), {k.__name__ for k in SPIKE_KERNELS})

    def _body(self, name):
        src = self.sources[name]
        body = src[src.rindex(") {") + 3:]
        return body

    def test_structural_sanity(self):
        for name, src in self.sources.items():
            with self.subTest(kernel=name):
                self.assertIn("kernel void", src)
                self.assertIn("[[thread_position_in_grid]]", src)
                self.assertIn("atomic_fetch_add_explicit", src) if "atomic" in name else None
                self.assertEqual(src.count("{"), src.count("}"))
                self.assertNotIn("warp", src.split("kernel void")[1])

    def test_copy_scale_body(self):
        body = self._body("k_copy_scale")
        self.assertIn("const int tid = int(_gid.x);", body)
        self.assertIn("float x = a[i];", body)
        self.assertIn("out[i] = (x * s[0]);", body)  # scalar params via constant space

    def test_atomic_call(self):
        body = self._body("k_atomic_accum")
        self.assertIn(
            "atomic_add_float(reinterpret_cast<device atomic_float*>(&accum[idx[i]]), vals[i]);",
            body,
        )

    def test_trig_uses_precise_intrinsics(self):
        self.assertIn("precise::sin", self._body("k_trig"))
        self.assertIn("precise::cos", self._body("k_trig"))

    def test_matvec_and_inverse(self):
        self.assertIn("mat33_mul_vec(m[i], v[i]);", self._body("k_matvec"))
        self.assertIn("mat33_inverse(m[i]);", self._body("k_matinv_apply"))

    def test_vec_arithmetic(self):
        body = self._body("k_saxpy3")
        self.assertIn("out[i] = ((a[i] * s[0]) + b[i]);", body)

    def test_select_ternary(self):
        self.assertIn("(x < lo[0]) ? lo[0] :", self._body("k_select_clip"))

    def test_registry_builtin_resolution(self):
        # every builtin named in the M0.3 acceptance list resolves
        for key in ["tid", "atomic_add", "dot", "cross", "normalize", "sin", "cos",
                    "select", "where", "length", "inverse", "sqrt"]:
            self.assertIn(key, registry.builtin_functions)


if __name__ == "__main__":
    unittest.main()
