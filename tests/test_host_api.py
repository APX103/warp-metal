# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""P2 host-surface tests (KEP-0001 M3.2): ``wp.get_suggested_block_size`` and
the ``WP_TILE_SYNC`` prelude macro.

mujoco_warp's collision_driver/collision_convex/solver launch paths call
``wp.get_suggested_block_size(kernel[, device])`` (collision_convex.py:1248,
collision_driver.py:1006, solver.py:3372); upstream (warp 1.19
``warp/_src/context.py:12543``) returns ``(1, 1)`` for every non-CUDA device,
which is the branch the shim mirrors. Upstream has no
``get_suggested_block_count`` — none is provided.

mujoco_warp embeds ``WP_TILE_SYNC()`` in ``@wp.func_native`` snippets
(smooth.py:3591, solver.py:903); M1.4 inlines those snippets verbatim into
the kernel compile unit, so the shared prelude must define the macro
(upstream CUDA: ``__syncthreads``, warp/native/tile.h:66; MSL equivalent:
``threadgroup_barrier(mem_flags::mem_threadgroup)``).

HOST-ONLY: kernel decoration and ``translate_kernel`` emit are pure-Python
(AST → MSL text); nothing here compiles a Metal library, dispatches, or
creates a WarpArray/device.
"""

import inspect
import unittest

import warp_metal as wp
from warp_metal.msl.prelude import prelude_source
from warp_metal.msl.translate import translate_kernel

_TILE_SYNC_MSL_DEF = "#define WP_TILE_SYNC threadgroup_barrier(mem_flags::mem_threadgroup)"


@wp.func_native(snippet="WP_TILE_SYNC();")
def _syncthreads():
    pass


@wp.kernel(module="unique", enable_backward=False)
def k_tile_sync(x: wp.array2d[wp.float32]):
    i, j = wp.tid()
    _syncthreads()
    x[i, j] = 0.0


def _plain_kernel():
    @wp.kernel
    def k_probe(x: wp.array[wp.float32]):
        i = wp.tid()
        x[i] = x[i] * 2.0

    return k_probe


class TestGetSuggestedBlockSize(unittest.TestCase):
    def test_signature_matches_upstream(self):
        # upstream warp 1.19 warp/_src/context.py:12543
        sig = inspect.signature(wp.get_suggested_block_size)
        self.assertEqual(list(sig.parameters), ["kernel", "device"])
        self.assertIs(sig.parameters["device"].default, None)

    def test_non_cuda_device_branch_returns_1_1(self):
        # upstream: `if not device.is_cuda: return (1, 1)` — the only branch
        # reachable on warp-metal devices (is_cuda is False by construction)
        kernel = _plain_kernel()
        self.assertEqual(wp.get_suggested_block_size(kernel), (1, 1))
        self.assertEqual(wp.get_suggested_block_size(kernel, "cpu"), (1, 1))

    def test_non_kernel_fails_loud(self):
        with self.assertRaises(TypeError):
            wp.get_suggested_block_size(lambda: None)

    def test_unknown_device_fails_loud(self):
        kernel = _plain_kernel()
        with self.assertRaises(ValueError):
            wp.get_suggested_block_size(kernel, "nosuchdevice")


class TestTileSyncPrelude(unittest.TestCase):
    def test_prelude_defines_tile_sync_before_all_sections(self):
        # the prelude is one shared compile unit (ROADMAP M2.2 discipline):
        # the macro must sit in the head so every kernel — tiled or not —
        # sees it, and it must be inert for kernels that never call it
        # (an unused preprocessor define is)
        src = prelude_source()
        self.assertIn(_TILE_SYNC_MSL_DEF, src)
        self.assertLess(src.index(_TILE_SYNC_MSL_DEF), src.index("struct vec2f"))

    def test_metal_barrier_is_the_cuda_equivalent(self):
        # expansion must stay Metal-correct: no CUDA __syncthreads, no fp64
        src = prelude_source()
        self.assertNotIn("syncthreads", src)
        self.assertNotIn("double", _TILE_SYNC_MSL_DEF)

    def test_func_native_snippet_inline_sees_the_macro(self):
        # the real mujoco_warp shape (solver.py:903: module="unique",
        # enable_backward=False, snippet "WP_TILE_SYNC();"): M1.4 inlines the
        # snippet verbatim, so the emitted unit must carry the macro before
        # the use. Emit-only — no Metal compile, no dispatch.
        src = translate_kernel(k_tile_sync).source()
        self.assertIn(_TILE_SYNC_MSL_DEF, src)
        self.assertIn("WP_TILE_SYNC();", src)
        self.assertLess(src.index(_TILE_SYNC_MSL_DEF), src.rindex("WP_TILE_SYNC();"))


if __name__ == "__main__":
    unittest.main()
