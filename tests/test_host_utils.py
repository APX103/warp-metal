# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Host sort/cast utility tests: wp.utils.radix_sort_pairs /
segmented_sort_pairs / array_cast (KEP-0001 steel-closure round).

Every positive case checks against a deterministic reference permutation
computed independently here (Python's stable ``sorted`` plus struct-level
float bit patterns), never by re-calling the implementation. Upstream
contracts: warp/_src/utils.py:166-475 (sorts) and :972-1056 (array_cast);
bit semantics: warp/native/sort.cpp:97-331.
"""

import struct
import unittest

import numpy as np

import warp_metal as wp


def _stable_order(rank_keys):
    """Reference permutation: indices [0, n) stably ascending by rank_keys[i]
    (Python sorts are stable, matching upstream's LSD radix tie order)."""
    return sorted(range(len(rank_keys)), key=lambda i: rank_keys[i])


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _f32_radix_key(bits):
    """Independent stereopsis mapping (warp/native/sort.cpp:261-267).

    Takes the raw uint32 bit pattern — not a Python float: round-tripping a
    signalling NaN through ``struct`` quiets it (0x7FA00000 → 0x7FC00000),
    which would silently corrupt the reference ranking.
    """
    mask = (-(bits >> 31)) & 0xFFFFFFFF | 0x80000000
    return bits ^ mask


class TestRadixSortPairs(unittest.TestCase):
    def test_int32_int32_full_range(self):
        # 40 deterministic keys incl. negatives and duplicates; values are
        # indices so the permutation itself is verified
        keys_in = [(-7) if i % 9 == 0 else ((i * 37) % 200) - 100 for i in range(40)]
        perm = _stable_order(keys_in)
        expected_keys = [keys_in[i] for i in perm]

        keys = wp.empty(80, dtype=wp.int32, device="metal")
        values = wp.empty(80, dtype=wp.int32, device="metal")
        keys.numpy[:40] = np.array(keys_in, dtype=np.int32)
        values.numpy[:40] = np.arange(40, dtype=np.int32)

        wp.utils.radix_sort_pairs(keys, values, 40)

        self.assertEqual(keys.numpy[:40].tolist(), expected_keys)
        self.assertEqual(values.numpy[:40].tolist(), perm)

    def test_uint32_int64_payload(self):
        keys_in = [0, 4294967295, 2147483648, 7, 7, 100, 3]
        perm = _stable_order(keys_in)

        keys = wp.empty(14, dtype=wp.uint32, device="metal")
        values = wp.empty(14, dtype=wp.int64, device="metal")
        keys.numpy[:7] = np.array(keys_in, dtype=np.uint32)
        values.numpy[:7] = np.arange(7, dtype=np.int64)

        wp.utils.radix_sort_pairs(keys, values, 7)

        self.assertEqual(keys.numpy[:7].tolist(), [keys_in[i] for i in perm])
        self.assertEqual(values.numpy[:7].tolist(), perm)

    def test_float32_bit_mapping(self):
        # +0.0 deliberately precedes -0.0 in the input: the stereopsis mapping
        # orders -0.0 below +0.0 (a numeric sort would keep the input order),
        # and NaN payloads order by bit pattern — both upstream bit-exact.
        # Keys are built (and compared) as raw bit patterns so ±0.0 / NaN
        # survive round-trips exactly.
        qnan_pos, qnan_neg, snan_pos = 0x7FC00000, 0xFFC00000, 0x7FA00000
        in_bits = [
            _f32_bits(2.5), _f32_bits(-1.25), _f32_bits(0.0), _f32_bits(-0.0),
            _f32_bits(float("inf")), _f32_bits(float("-inf")),
            qnan_pos, qnan_neg, snan_pos,
            _f32_bits(7.0), _f32_bits(-3.5), _f32_bits(1.5),
        ]
        rank = [_f32_radix_key(b) for b in in_bits]
        perm = _stable_order(rank)
        expected_bits = [in_bits[i] for i in perm]

        keys = wp.empty(24, dtype=wp.float32, device="metal")
        values = wp.empty(24, dtype=wp.int32, device="metal")
        keys.numpy[:12] = np.array(in_bits, dtype=np.uint32).view(np.float32)
        values.numpy[:12] = np.arange(12, dtype=np.int32)

        wp.utils.radix_sort_pairs(keys, values, 12)

        out_bits = np.asarray(keys.numpy[:12], dtype=np.float32).view(np.uint32).tolist()
        self.assertEqual(out_bits, expected_bits)
        self.assertEqual(values.numpy[:12].tolist(), perm)
        # explicit pin: the -0.0 pattern lands before the +0.0 pattern
        self.assertIn(0x80000000, out_bits)
        self.assertIn(0, out_bits)
        self.assertLess(out_bits.index(0x80000000), out_bits.index(0))

    def test_end_bit_subset_matches_contact_sort(self):
        # contact_sort.py shape: int32 keys packed (family<<16|index), sorted
        # by the low 16 bits only (end_bit=16); full keys are preserved and
        # ties keep original order (stability)
        keys_in = [(5 << 16) | 3, (1 << 16) | 3, (9 << 16) | 1, (2 << 16) | 1, (7 << 16) | 2, (4 << 16) | 2]
        perm = _stable_order([k & 0xFFFF for k in keys_in])
        expected_keys = [keys_in[i] for i in perm]

        keys = wp.empty(12, dtype=wp.int32, device="metal")
        values = wp.empty(12, dtype=wp.int32, device="metal")
        keys.numpy[:6] = np.array(keys_in, dtype=np.int32)
        values.numpy[:6] = np.arange(6, dtype=np.int32)

        wp.utils.radix_sort_pairs(keys, values, 6, end_bit=16)

        self.assertEqual(keys.numpy[:6].tolist(), expected_keys)
        self.assertEqual(values.numpy[:6].tolist(), perm)

    def test_begin_bit_end_bit_window(self):
        # only bits [8, 16) rank: high noise (bits 16+) and low noise (bits
        # <8) must not affect the order
        keys_in = [
            (0xAB << 16) | (0x42 << 8) | 0xCD,
            (0x11 << 16) | (0x07 << 8) | 0x99,
            (0xFF << 16) | (0x42 << 8) | 0x01,
            (0x03 << 16) | (0x00 << 8) | 0xFE,
        ]
        perm = _stable_order([(k >> 8) & 0xFF for k in keys_in])

        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        keys.numpy[:4] = np.array(keys_in, dtype=np.int32)
        values.numpy[:4] = np.arange(4, dtype=np.int32)

        wp.utils.radix_sort_pairs(keys, values, 4, begin_bit=8, end_bit=16)

        self.assertEqual(values.numpy[:4].tolist(), perm)

    def test_begin_bit_equals_end_bit_noop(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        keys.numpy[:] = np.arange(8, dtype=np.int32) * 3 - 5
        values.numpy[:] = np.arange(8, dtype=np.int32)
        keys_before = keys.numpy.copy()
        values_before = values.numpy.copy()

        wp.utils.radix_sort_pairs(keys, values, 4, begin_bit=4, end_bit=4)

        self.assertEqual(keys.numpy.tolist(), keys_before.tolist())
        self.assertEqual(values.numpy.tolist(), values_before.tolist())

    def test_count_zero_noop(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        keys.numpy[:] = np.array([3, 1, 2, 0, 5, 5, 5, 5], dtype=np.int32)
        values.numpy[:] = np.arange(8, dtype=np.int32)

        wp.utils.radix_sort_pairs(keys, values, 0)

        self.assertEqual(keys.numpy.tolist(), [3, 1, 2, 0, 5, 5, 5, 5])
        self.assertEqual(values.numpy.tolist(), list(range(8)))

    def test_scratch_half_untouched(self):
        # upstream ping-pongs through [count:2*count]; our contract (documented
        # in the shim) leaves it untouched — pin that observable behavior
        count = 3
        keys = wp.empty(2 * count, dtype=wp.int32, device="metal")
        values = wp.empty(2 * count, dtype=wp.int32, device="metal")
        keys.numpy[:count] = np.array([30, 10, 20], dtype=np.int32)
        values.numpy[:count] = np.arange(count, dtype=np.int32)
        keys.numpy[count:] = 0x5A5A5A5A
        values.numpy[count:] = -123456

        wp.utils.radix_sort_pairs(keys, values, count)

        self.assertEqual(keys.numpy[:count].tolist(), [10, 20, 30])
        self.assertEqual(values.numpy[:count].tolist(), [1, 2, 0])
        self.assertEqual(keys.numpy[count:].tolist(), [0x5A5A5A5A] * count)
        self.assertEqual(values.numpy[count:].tolist(), [-123456] * count)

    def test_vec2f_payload(self):
        # upstream accepts any 4/8-byte value dtype (SortPayload<4|8>); a
        # composite payload must move element-wise
        keys = wp.empty(6, dtype=wp.int32, device="metal")
        values = wp.empty(6, dtype=wp.vec2f, device="metal")
        keys.numpy[:3] = np.array([3, 1, 2], dtype=np.int32)
        values.numpy[:3] = np.array([(1, 10), (2, 20), (3, 30)], dtype=np.float32)

        wp.utils.radix_sort_pairs(keys, values, 3)

        self.assertEqual(keys.numpy[:3].tolist(), [1, 2, 3])
        np.testing.assert_array_equal(
            np.asarray(values.numpy[:3]), np.array([(2, 20), (3, 30), (1, 10)], dtype=np.float32)
        )

    def test_device_mismatch(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="cpu")
        with self.assertRaises(RuntimeError):
            wp.utils.radix_sort_pairs(keys, values, 4)

    def test_insufficient_storage(self):
        keys = wp.empty(7, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.radix_sort_pairs(keys, values, 4)  # 7 < 2*4

    def test_unsupported_key_dtype(self):
        keys = wp.empty(8, dtype=wp.int16, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.radix_sort_pairs(keys, values, 4)

    def test_unsupported_value_dtype(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int16, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.radix_sort_pairs(keys, values, 4)

    def test_invalid_bit_range(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.radix_sort_pairs(keys, values, 4, begin_bit=20, end_bit=12)
        with self.assertRaises(RuntimeError):
            wp.utils.radix_sort_pairs(keys, values, 4, begin_bit=0, end_bit=33)


class TestSegmentedSortPairs(unittest.TestCase):
    def test_upstream_docstring_example(self):
        # warp/_src/utils.py:335-343, verbatim shapes
        keys = wp.array(np.array([3, 1, 4, 2, 0, 0, 0, 0], dtype=np.int32), dtype=wp.int32, device="metal")
        values = wp.array(np.array([30, 10, 40, 20, 0, 0, 0, 0], dtype=np.int32), dtype=wp.int32, device="metal")
        offsets = wp.array(np.array([0, 2, 4], dtype=np.int32), dtype=wp.int32, device="metal")

        wp.utils.segmented_sort_pairs(keys, values, 4, offsets)

        self.assertEqual(keys.numpy[:4].tolist(), [1, 3, 2, 4])
        self.assertEqual(values.numpy[:4].tolist(), [10, 30, 20, 40])

    def test_inferred_ends_untouched_outside_segments(self):
        # segments [0,2) and [2,5); index 5 lies inside [0, count) but outside
        # every segment -> untouched; scratch half untouched
        keys_in = [40, 10, 22, 21, 12, 77]
        seg1_perm = _stable_order(keys_in[0:2])
        seg2_perm = _stable_order(keys_in[2:5])

        keys = wp.empty(12, dtype=wp.int32, device="metal")
        values = wp.empty(12, dtype=wp.int32, device="metal")
        keys.numpy[:6] = np.array(keys_in, dtype=np.int32)
        values.numpy[:6] = np.arange(6, dtype=np.int32)
        keys.numpy[6:] = -1
        values.numpy[6:] = -7
        offsets = wp.array(np.array([0, 2, 5], dtype=np.int32), dtype=wp.int32, device="metal")

        wp.utils.segmented_sort_pairs(keys, values, 6, offsets)

        self.assertEqual(keys.numpy[0:2].tolist(), [keys_in[0:2][i] for i in seg1_perm])
        self.assertEqual(keys.numpy[2:5].tolist(), [keys_in[2:5][i] for i in seg2_perm])
        self.assertEqual(keys.numpy[5].item(), 77)  # gap element untouched
        self.assertEqual(values.numpy[5].item(), 5)
        self.assertEqual(keys.numpy[6:].tolist(), [-1] * 6)  # scratch untouched
        self.assertEqual(values.numpy[6:].tolist(), [-7] * 6)

    def test_float_keys_explicit_ends(self):
        # broad_phase_sap.py shape but with explicit end indices and gaps:
        # segments [0,2) and [3,5) of count=7, so indices 2, 5, 6 lie outside
        # every segment and must stay untouched
        keys_in = [2.5, -1.0, 9.0, 0.0, -3.5, 4.0, 1.0]
        p0 = _stable_order([_f32_radix_key(_f32_bits(v)) for v in keys_in[0:2]])
        p1 = _stable_order([_f32_radix_key(_f32_bits(v)) for v in keys_in[3:5]])
        expected_keys = (
            [keys_in[0:2][i] for i in p0] + [keys_in[2]]
            + [keys_in[3:5][i] for i in p1] + [keys_in[5], keys_in[6]]
        )
        expected_values = [p0[0], p0[1], 2, 3 + p1[0], 3 + p1[1], 5, 6]

        keys = wp.empty(14, dtype=wp.float32, device="metal")
        values = wp.empty(14, dtype=wp.int32, device="metal")
        keys.numpy[:7] = np.array(keys_in, dtype=np.float32)
        values.numpy[:7] = np.arange(7, dtype=np.int32)
        seg_start = wp.array(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device="metal")
        seg_end = wp.array(np.array([2, 5], dtype=np.int32), dtype=wp.int32, device="metal")

        wp.utils.segmented_sort_pairs(keys, values, 7, seg_start, seg_end)

        # sorting permutes, never modifies: exact equality, no tolerance
        np.testing.assert_array_equal(np.asarray(keys.numpy[:7]), np.array(expected_keys, dtype=np.float32))
        self.assertEqual(values.numpy[:7].tolist(), expected_values)

    def test_invalid_bounds_valueerror_and_no_mutation(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        keys.numpy[:] = 5
        values.numpy[:] = 9
        offsets = wp.array(np.array([0, 3, 10], dtype=np.int32), dtype=wp.int32, device="metal")

        with self.assertRaises(ValueError):  # end 10 > count 4
            wp.utils.segmented_sort_pairs(keys, values, 4, offsets)

        # upstream validates every range before sorting anything
        self.assertEqual(keys.numpy.tolist(), [5] * 8)
        self.assertEqual(values.numpy.tolist(), [9] * 8)

        offsets_neg = wp.array(np.array([0, -1, 2], dtype=np.int32), dtype=wp.int32, device="metal")
        with self.assertRaises(ValueError):
            wp.utils.segmented_sort_pairs(keys, values, 4, offsets_neg)

        offsets_rev = wp.array(np.array([2, 1], dtype=np.int32), dtype=wp.int32, device="metal")
        with self.assertRaises(ValueError):  # end < start
            wp.utils.segmented_sort_pairs(keys, values, 4, offsets_rev)

    def test_count_validation(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        offsets = wp.array(np.array([0, 2, 4], dtype=np.int32), dtype=wp.int32, device="metal")

        with self.assertRaises(TypeError):
            wp.utils.segmented_sort_pairs(keys, values, True, offsets)
        with self.assertRaises(TypeError):
            wp.utils.segmented_sort_pairs(keys, values, "4", offsets)
        with self.assertRaises(ValueError):
            wp.utils.segmented_sort_pairs(keys, values, -1, offsets)
        with self.assertRaises(ValueError):
            wp.utils.segmented_sort_pairs(keys, values, 2**31, offsets)

    def test_segment_size_mismatch(self):
        keys = wp.empty(8, dtype=wp.int32, device="metal")
        values = wp.empty(8, dtype=wp.int32, device="metal")
        seg_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=wp.int32, device="metal")
        seg_end = wp.array(np.array([2], dtype=np.int32), dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.segmented_sort_pairs(keys, values, 4, seg_start, seg_end)

    def test_unsupported_value_dtype(self):
        keys = wp.empty(8, dtype=wp.float32, device="metal")
        values = wp.empty(8, dtype=wp.int64, device="metal")
        offsets = wp.array(np.array([0, 2, 4], dtype=np.int32), dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.segmented_sort_pairs(keys, values, 4, offsets)


class TestArrayCast(unittest.TestCase):
    def test_int64_to_int32_mesh_shape(self):
        # newton/_src/utils/mesh.py:125 shape: 1-D int64 -> 1-D int32,
        # count=None; values chosen so the C-style wrap is unambiguous
        vals = [0, 1, -1, 2**31, 2**31 - 1, -2**31, 2**32 + 5, -2**32 - 5]
        expected = np.array(vals, dtype=np.int64).astype(np.int32)

        src = wp.array(np.array(vals, dtype=np.int64), dtype=wp.int64, device="metal")
        dst = wp.empty(8, dtype=wp.int32, device="metal")
        dst.numpy[:] = -777

        wp.utils.array_cast(src, dst)

        self.assertEqual(dst.numpy.tolist(), expected.tolist())

    def test_partial_count(self):
        src = wp.array(np.array([10, 20, 30, 40], dtype=np.int64), dtype=wp.int64, device="metal")
        dst = wp.full(6, 999, dtype=wp.int32, device="metal")

        wp.utils.array_cast(src, dst, count=2)

        self.assertEqual(dst.numpy.tolist(), [10, 20, 999, 999, 999, 999])

    def test_device_mismatch(self):
        src = wp.empty(4, dtype=wp.int64, device="metal")
        dst = wp.empty(4, dtype=wp.int32, device="cpu")
        with self.assertRaises(RuntimeError):
            wp.utils.array_cast(src, dst)

    def test_flatten_fallback_fails_loud(self):
        # upstream's flatten-to-scalar path (utils.py:1001-1031) is outside
        # the closure
        src = wp.empty(6, dtype=wp.int32, device="metal")
        dst = wp.empty((2, 3), dtype=wp.int32, device="metal")
        with self.assertRaises(NotImplementedError):
            wp.utils.array_cast(src, dst)

    def test_partial_multi_dim_fails(self):
        src = wp.empty((2, 3), dtype=wp.int64, device="metal")
        dst = wp.empty((2, 3), dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.array_cast(src, dst, count=3)

    def test_shape_mismatch_fails(self):
        src = wp.empty((2, 3), dtype=wp.int64, device="metal")
        dst = wp.empty((3, 2), dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.array_cast(src, dst)

    def test_count_exceeds_size_fails(self):
        src = wp.empty(6, dtype=wp.int64, device="metal")
        dst = wp.empty(6, dtype=wp.int32, device="metal")
        with self.assertRaises(RuntimeError):
            wp.utils.array_cast(src, dst, count=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
