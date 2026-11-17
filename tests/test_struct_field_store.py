# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Struct-field stores through array element addresses (viewer GL unblock).

The vendored codegen lowers ``arr[i].field = v`` on a struct-typed array to
``wp::store(&(wp::index(arr, i).field), v)`` — an element pointer projected
to the member. The alias machinery only tracked ``wp::address`` results, so
the store path failed loud ("not a pointer alias") the first time a viewer
kernel compiled (Newton GL ``fill_vertex_data``: ``vertices[tid].pos =
points[tid]``, discovered via the visual demo session — headless smoke never
imports ``newton._src.viewer.gl.opengl``). ``_emit_struct_def`` keeps the
Python member names in the MSL struct, so the fix dereferences the same
``wp_address`` element pointer and projects the member in place.

Tests pin the emission and verify numerically on Metal, mirroring the two
viewer kernels (fill_vertex_data / fill_line_vertex_data, including the
optional-array truthiness paths — ``uvs`` may be ``None`` per
opengl.py:356-364) plus a field-load read-back kernel.
"""

import unittest

import numpy as np

import warp_metal as wp
from warp_metal.msl.translate import translate_kernel


@wp.struct
class Vertex:
    pos: wp.vec3f
    normal: wp.vec3f
    uv: wp.vec2f


@wp.struct
class LineVertex:
    pos: wp.vec3f
    color: wp.vec3f


@wp.kernel
def k_fill(
    points: wp.array(dtype=wp.vec3f),
    normals: wp.array(dtype=wp.vec3f),
    uvs: wp.array(dtype=wp.vec2f),
    vertices: wp.array(dtype=Vertex),
):
    tid = wp.tid()

    vertices[tid].pos = points[tid]

    if normals:
        vertices[tid].normal = normals[tid]

    if uvs:
        vertices[tid].uv = uvs[tid]


@wp.kernel
def k_lines(
    starts: wp.array(dtype=wp.vec3f),
    ends: wp.array(dtype=wp.vec3f),
    colors: wp.array(dtype=wp.vec3f),
    vertices: wp.array(dtype=LineVertex),
):
    tid = wp.tid()

    vertex_idx = tid * 2

    vertices[vertex_idx].pos = starts[tid]
    vertices[vertex_idx].color = colors[tid]

    vertices[vertex_idx + 1].pos = ends[tid]
    vertices[vertex_idx + 1].color = colors[tid]


@wp.kernel
def k_read_fields(
    vertices: wp.array(dtype=Vertex),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = vertices[i].pos[0] + vertices[i].pos[1] + vertices[i].normal[2] + vertices[i].uv[0]


def _strip_msl_comments(src: str) -> str:
    return "\n".join(line.split("//")[0] for line in src.splitlines())


class TestStructFieldStoreEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {
            name: translate_kernel(k).source()
            for name, k in (
                ("fill", k_fill),
                ("lines", k_lines),
                ("read", k_read_fields),
            )
        }

    def test_store_lowers_to_element_pointer_member_assign(self):
        src = _strip_msl_comments(self.sources["fill"])
        self.assertIn("wp_address(vertices_data", src)
        self.assertIn(").pos = ", src)
        self.assertIn(").normal = ", src)
        self.assertIn(").uv = ", src)

    def test_computed_index_store_lowers_same(self):
        src = _strip_msl_comments(self.sources["lines"])
        self.assertIn(").pos = ", src)
        self.assertIn(").color = ", src)

    def test_field_load_lowers_to_member_deref(self):
        src = _strip_msl_comments(self.sources["read"])
        self.assertIn(").pos", src)
        self.assertIn(").uv", src)


class TestStructFieldStoreDevice(unittest.TestCase):
    N = 8

    def _run_fill(self, with_uvs):
        n = self.N
        pts_np = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.25
        nrm_np = np.linspace(-1.0, 1.0, n * 3, dtype=np.float32).reshape(n, 3)
        uv_np = np.linspace(0.0, 1.0, n * 2, dtype=np.float32).reshape(n, 2)

        points = wp.array(data=pts_np, dtype=wp.vec3f, device="metal")
        normals = wp.array(data=nrm_np, dtype=wp.vec3f, device="metal")
        uvs = wp.array(data=uv_np, dtype=wp.vec2f, device="metal") if with_uvs else None
        vertices = wp.zeros(n, dtype=Vertex, device="metal")

        wp.launch(k_fill, dim=n, inputs=[points, normals, uvs], outputs=[vertices], device="metal")

        out = wp.zeros(n, dtype=wp.float32, device="metal")
        wp.launch(k_read_fields, dim=n, inputs=[vertices, out], device="metal")
        got = out.numpy()

        want = pts_np[:, 0] + pts_np[:, 1] + nrm_np[:, 2] + (uv_np[:, 0] if with_uvs else 0.0)
        np.testing.assert_array_equal(got, want.astype(np.float32))

    def test_fill_all_fields(self):
        self._run_fill(with_uvs=True)

    def test_fill_uvs_none(self):
        # viewer passes uvs=None for meshes without UVs (opengl.py:356-364)
        self._run_fill(with_uvs=False)

    def test_line_pairs(self):
        n = self.N
        starts_np = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.5
        ends_np = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * -0.5
        colors_np = np.linspace(0.0, 1.0, n * 3, dtype=np.float32).reshape(n, 3)

        starts = wp.array(data=starts_np, dtype=wp.vec3f, device="metal")
        ends = wp.array(data=ends_np, dtype=wp.vec3f, device="metal")
        colors = wp.array(data=colors_np, dtype=wp.vec3f, device="metal")
        vertices = wp.zeros(n * 2, dtype=LineVertex, device="metal")

        wp.launch(k_lines, dim=n, inputs=[starts, ends, colors], outputs=[vertices], device="metal")

        got = vertices.numpy()
        want_pos = np.empty((n * 2, 3), dtype=np.float32)
        want_pos[0::2] = starts_np
        want_pos[1::2] = ends_np
        want_color = np.repeat(colors_np, 2, axis=0).astype(np.float32)
        np.testing.assert_array_equal(got["pos"], want_pos)
        np.testing.assert_array_equal(got["color"], want_color)


if __name__ == "__main__":
    unittest.main()
