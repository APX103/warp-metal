# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""put_model io-shim isinstance parity (KEP-0001 M3.2; matrix §5-1).

mujoco_warp probes dataclass array specs with a two-line predicate
(``mujoco_warp/_src/warp_util.py:26``, copied verbatim below — this suite
must not import mujoco_warp)::

    isinstance(typ, wp.array) or type(typ).__name__ == "_ArrayAnnotation"

Its ``types.array()`` helper (``mujoco_warp/_src/types.py:950``) builds each
spec as ``wp.array(ndim=N, dtype=dtype)`` and then assigns ``.shape``. Under
the shim that call is the facade's annotation mode and yields the vendored
annotation-mode descriptor — unlike upstream warp, where the same call yields
a runtime array instance — so neither arm of the predicate fired and every
one of the 521 Model fields fell through to the scalar ``f.type(value)``
branch, raising ``TypeError: 'array' object is not callable`` at io.py:493.

The fix lives shim-side (black-box discipline): the public ``wp.array``
facade carries a metaclass ``__instancecheck__`` (``_ArrayMeta``) accepting
runtime WarpArrays and vendored array descriptors, restoring the predicate's
first arm — the arm real warp satisfies.

HOST-ONLY: annotation-mode ``wp.array(...)`` allocates nothing (vendored
``_init_annotation``), and the WarpArray shell below is created via
``__new__`` (bypassing the MTLBuffer-allocating ``__init__``). The full
``mujoco_warp.put_model`` repro is NOT runnable host-only: post-fix it
allocates WarpArrays (``_mm.buffer_create`` MTLBuffers — storage is
device-independent shared memory, so even ``device="cpu"`` touches Metal)
and ends in ``set_const`` kernel launches routed through the MSL/Metal
pipeline.
"""

import unittest

import warp_metal as wp


def _mujoco_warp_is_array_spec(typ) -> bool:
    """Verbatim copy of mujoco_warp._src.warp_util.is_array_spec:26."""
    return isinstance(typ, wp.array) or type(typ).__name__ == "_ArrayAnnotation"


def _mujoco_warp_types_array(shape, dtype):
    """Shape-for-shape copy of mujoco_warp._src.types.array (:950).

    Constructs the spec through the public facade (annotation mode) and
    performs the ``.shape`` assignment mujoco_warp does on the descriptor.
    """
    arr = wp.array(ndim=len(shape), dtype=dtype)
    arr.shape = shape
    return arr


class TestIsArraySpecParity(unittest.TestCase):
    """The is_array_spec-shaped probe mujoco_warp's put_model分流 relies on."""

    def test_annotation_descriptor_is_array_spec(self):
        spec = _mujoco_warp_types_array(("*",), wp.float32)
        self.assertTrue(_mujoco_warp_is_array_spec(spec))
        self.assertTrue(isinstance(spec, wp.array))

    def test_annotation_descriptor_multidim_and_composite_dtype(self):
        for shape in (("nq",), ("nworld", 6)):
            spec = _mujoco_warp_types_array(shape, wp.vec3)
            self.assertTrue(_mujoco_warp_is_array_spec(spec))
        # attributes mujoco_warp's _create_array reads off the spec
        self.assertEqual(spec.shape, ("nworld", 6))
        self.assertIsNotNone(spec.dtype)

    def test_scalar_fields_keep_the_f_type_value_branch(self):
        # non-array fields must still route to f.type(value) — the callable
        # scalar branch every field wrongly took before the fix
        self.assertFalse(_mujoco_warp_is_array_spec(wp.float32))
        self.assertFalse(_mujoco_warp_is_array_spec(float))
        self.assertFalse(_mujoco_warp_is_array_spec(1.0))

    def test_runtime_array_shell_passes_facade_check(self):
        # a bare WarpArray shell (no Metal buffer: __new__ bypasses __init__)
        # must keep passing the facade check, as upstream runtime arrays do
        arr = wp.WarpArray.__new__(wp.WarpArray)
        self.assertTrue(isinstance(arr, wp.array))
        self.assertTrue(_mujoco_warp_is_array_spec(arr))

    def test_non_arrays_stay_negative(self):
        for value in (42, "array", (1, 2), wp.float32):
            self.assertFalse(isinstance(value, wp.array), repr(value))
            self.assertFalse(_mujoco_warp_is_array_spec(value), repr(value))

    def test_struct_field_annotation_keeps_name_arm(self):
        # struct-field path annotations (wp.array[dtype]) produce
        # _ArrayAnnotation objects; mujoco_warp's second arm must keep
        # catching them, and the metaclass must not swallow the name check
        ann = wp.array[wp.float32]
        self.assertEqual(type(ann).__name__, "_ArrayAnnotation")
        self.assertTrue(_mujoco_warp_is_array_spec(ann))

    def test_facade_subclasscheck_reflexivity(self):
        # issubclass goes through _ArrayMeta too; reflexivity must hold
        self.assertTrue(issubclass(wp.array, wp.array))
        self.assertTrue(issubclass(wp.WarpArray, wp.array))
        self.assertFalse(issubclass(int, wp.array))


if __name__ == "__main__":
    unittest.main()
