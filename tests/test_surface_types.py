# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""M1.1 surface tests: type system (buckets A+B), @wp.struct, constants."""

import math
import unittest

import warp_metal as wp

# Bucket A: scalar / vector / matrix / quat / transform / spatial types
SCALAR_TYPES = [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "bfloat16",
]

VECTOR_MATRIX_TYPES = [
    "vec2", "vec3", "vec4", "vec2f", "vec2h", "vec3f", "vec3h", "vec3i",
    "vec4f", "vec4i", "vec4h", "vec2i",
    "mat22", "mat33", "mat44", "mat22f", "mat33f", "mat44f",
    "quat", "quath", "quatf", "quatd",
    "transform", "transformf", "transformd",
    "spatial_vector", "spatial_matrix",
]

# Bucket B: array family
ARRAY_TYPES = ["array", "array2d", "array3d", "array4d", "indexedarray"]

CONSTANTS = {
    "pi": math.pi,
    "PI": math.pi,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    "HALF_PI": math.pi / 2,
}


class TestTypeSurface(unittest.TestCase):
    def test_type_surface_complete(self):
        missing = [
            name
            for name in SCALAR_TYPES + VECTOR_MATRIX_TYPES + ARRAY_TYPES
            if getattr(wp, name, None) is None
        ]
        self.assertEqual(missing, [])

    def test_scalar_type_values(self):
        self.assertIsNot(wp.int32, wp.int64)
        self.assertIsNot(wp.float32, wp.float64)
        self.assertIsNot(wp.vec3, wp.vec4)


class TestConstants(unittest.TestCase):
    def test_constant_values(self):
        for name, value in CONSTANTS.items():
            got = getattr(wp, name)
            if math.isnan(value):
                self.assertTrue(math.isnan(got), name)
            else:
                self.assertEqual(got, value, name)


class TestStructDecorator(unittest.TestCase):
    """@wp.struct was previously shadowed by a leaked stdlib struct module."""

    def test_struct_decorator_basic(self):
        @wp.struct
        class BodyData:
            pos: wp.vec3
            rot: wp.quat
            mass: wp.float32
            index: wp.int32

        self.assertIsInstance(BodyData, wp.Struct)
        self.assertEqual(set(BodyData.vars.keys()), {"pos", "rot", "mass", "index"})
        # ctype built from annotations
        fields = [f[0] for f in BodyData.ctype._fields_]
        self.assertEqual(fields, ["pos", "rot", "mass", "index"])
        # stable hash derived from key + annotations
        self.assertEqual(BodyData.hash, wp.struct(BodyData.cls).hash)
        # instance construction returns a StructInstance subclassing the
        # user class (same as upstream warp: A itself is a Struct, not a type)
        inst = BodyData()
        self.assertEqual(inst.mass, 0.0)  # zero-initialized
        self.assertEqual(inst.pos, (0.0, 0.0, 0.0))
        mro_names = [c.__name__ for c in type(inst).__mro__]
        self.assertIn("BodyData", mro_names)
        self.assertIn("StructInstance", mro_names)

    def test_struct_decorator_with_args(self):
        @wp.struct(module="unique")
        class S:
            a: wp.int32

        self.assertIsInstance(S, wp.Struct)

    def test_struct_nested(self):
        @wp.struct
        class Inner:
            v: wp.vec3

        @wp.struct
        class Outer:
            inner: Inner
            scale: wp.float32

        self.assertIsInstance(Outer.vars["inner"].type, wp.Struct)
        # hash recursion over nested structs
        self.assertNotEqual(Outer.hash, Inner.hash)


if __name__ == "__main__":
    unittest.main()
