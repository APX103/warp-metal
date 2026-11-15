# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""MSL prelude for the Adjoint-SSA emitter (KEP-0001 M1.3).

Value-type structs use the warp-native scalarized layouts exactly
(quat = {x, y, z, w} imaginary-first; transform = {vec3 p, quat q};
spatial_vector = {vec3 w, vec3 v}; mat row-major ``c[r][col]``) so data
packed host-side and results are bit-comparable with the CPU(LLVM) path
within fp32 tolerance.

Every math function is a literal port of the corresponding warp native
implementation (warp 1.19 ``warp/native/{vec,mat,quat,spatial,builtin}.h``);
port notes name the source. No CUDA paths, no adjoints.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# scalar type helpers
# ---------------------------------------------------------------------------

_STRUCTS = """\
// ---- scalarized value types (warp-native layouts; quat.h/spatial.h/mat.h) ----
struct vec2f { float x, y; };
struct vec3f { float x, y, z; };
struct vec4f { float x, y, z, w; };
struct vec5f { float x, y, z, w, u; };  // vec_t<5> (Newton vec5 contact samples)
struct quatf { float x, y, z, w; };  // imaginary xyz, real w (quat.h)
struct mat22f { float c[2][2]; };    // c[row][col], row-major (mat.h data)
struct mat33f { float c[3][3]; };
struct mat44f { float c[4][4]; };
struct transformf { vec3f p; quatf q; };  // (spatial.h transform_t)
struct spatial_vectorf { vec3f w, v; };   // (spatial.h spatial_vector_t)
// integer vectors (vec.h vec_t<Type> with int; packed layout, not MSL int3)
struct vec2i { int x, y; };
struct vec3i { int x, y, z; };
struct vec4i { int x, y, z, w; };

// ---- range iterator (builtin.h range_t + warp.codegen iter protocol) ----
struct wp_range {
    int start, end, step, i;
    int count, pos;
};
static inline bool wp_iter_cmp(thread wp_range& r) { return r.pos < r.count; }
static inline int wp_iter_next(thread wp_range& r) {
    int v = r.start + r.pos * r.step;
    ++r.pos;
    return v;
}
// wp::range overloads (builtin.h range(start[, stop[, step]]))
static inline wp_range wp_range_1(int count) {
    wp_range r; r.start = 0; r.end = count; r.step = 1; r.pos = 0;
    r.count = (count > 0) ? count : 0; r.i = 0; return r;
}
static inline wp_range wp_range_2(int start, int stop) {
    wp_range r; r.start = start; r.end = stop; r.step = 1; r.pos = 0;
    r.count = (stop > start) ? (stop - start) : 0; r.i = start; return r;
}
static inline wp_range wp_range_3(int start, int stop, int step) {
    wp_range r; r.start = start; r.end = stop; r.step = step; r.pos = 0;
    r.count = 0;
    if (step > 0 && stop > start) r.count = (stop - start + step - 1) / step;
    if (step < 0 && stop < start) r.count = (start - stop - step - 1) / (-step);
    r.i = start; return r;
}

// wp.untile: each thread holds its own tile element value in this lowering —
// identity for every type
template <typename T> static inline T wp_untile(T v) { return v; }

// ---- array binding: host-side metadata + device data pointer ----
// Mirrors warp native array_t (data/offset/shape/strides); strides are in
// BYTES (array.h byte_offset), contiguous row-major only.
struct wp_arr_meta {
    int offset;
    int ndim;
    int shape[4];
    int strides[4];
};

// wp.Mesh member array payloads (wp::array_t<vec3> / wp::array_t<int>
// lowered to {device data ptr, meta}; the wp_mesh struct itself is defined
// after the BVH ops in _MESH_QUERY_OPS — it embeds wp_bvh_t). Device reads
// go through the per-launch mesh table (`device wp_mesh* _wp_mesh_table`,
// host-built; see runtime/host.py). Entry 0 is the null mesh; ids index
// the table directly.
struct wp_mesh_array { device vec3f* data; wp_arr_meta m; };
struct wp_mesh_array_i { device int* data; wp_arr_meta m; };

// tiled mesh AABB query (Newton tile_mesh_query_aabb, KEP-0001 M2.4 P5): the
// cooperative query reuses the wp_mesh_query_aabb_t iterator below as the
// lane-0-private traversal state; results hand off to the block through
// per-kernel threadgroup slots emitted by the translator (_meshq_construct /
// _meshq_next). The M1 fail-loud stub that used to live here is gone — the
// real traversal is the wp_mesh_query_aabb/wp_mesh_query_aabb_next pair.

// multi-dim tid decomposition (upstream builtin.h tid(): the launch's _idx
// is the flat THREAD index; wp_tidN yields the first N coordinates of the
// row-major unravel over the launch bounds — last dim varies fastest — after
// folding the threads BEYOND coordinate N into one coordinate, exactly like
// upstream launch_coord's coord_mult). _wp_args._dims carries the launch
// bounds: a plain launch the user dim, a tiled launch grid + [block_dim] —
// so under wp.launch_tiled scalar wp.tid() is the BLOCK index and wp_tid2
// the (block, lane) pair. Shape entries of 0 (bounds padded to 4) act as 1,
// mirroring upstream's 1-padded launch bounds.
static inline int _wp_dim_or_1(int s) { return s ? s : 1; }
static inline int wp_tid1(int idx, wp_arr_meta m) {
    int mult = _wp_dim_or_1(m.shape[1]) * _wp_dim_or_1(m.shape[2]) * _wp_dim_or_1(m.shape[3]);
    return idx / mult;
}
static inline void wp_tid2(thread int& i, thread int& j, int idx, wp_arr_meta m) {
    idx /= _wp_dim_or_1(m.shape[2]) * _wp_dim_or_1(m.shape[3]);
    int s1 = _wp_dim_or_1(m.shape[1]);
    j = idx % s1;
    i = idx / s1;
}
static inline void wp_tid3(thread int& i, thread int& j, thread int& k, int idx, wp_arr_meta m) {
    idx /= _wp_dim_or_1(m.shape[3]);
    int s2 = _wp_dim_or_1(m.shape[2]);
    k = idx % s2;
    idx /= s2;
    int s1 = _wp_dim_or_1(m.shape[1]);
    j = idx % s1;
    i = idx / s1;
}
static inline void wp_tid4(thread int& i, thread int& j, thread int& k, thread int& l, int idx, wp_arr_meta m) {
    int s3 = _wp_dim_or_1(m.shape[3]);
    l = idx % s3;
    idx /= s3;
    int s2 = _wp_dim_or_1(m.shape[2]);
    k = idx % s2;
    idx /= s2;
    int s1 = _wp_dim_or_1(m.shape[1]);
    j = idx % s1;
    i = idx / s1;
}

// ---- array addressing (warp native array.h index(): strides in BYTES,
// byte_offset arithmetic through char*, negative indices wrap Python-style) ----
template <typename T>
static inline device T* wp_address(device T* data, wp_arr_meta m, int i) {
    if (i < 0) i += m.shape[0];
    return (device T*)((device char*)data + (size_t)m.offset + (size_t)(i * m.strides[0]));
}
template <typename T>
static inline device T* wp_address(device T* data, wp_arr_meta m, int i, int j) {
    if (i < 0) i += m.shape[0];
    if (j < 0) j += m.shape[1];
    return (device T*)((device char*)data + (size_t)m.offset
        + (size_t)(i * m.strides[0]) + (size_t)(j * m.strides[1]));
}
template <typename T>
static inline device T* wp_address(device T* data, wp_arr_meta m, int i, int j, int k) {
    if (i < 0) i += m.shape[0];
    if (j < 0) j += m.shape[1];
    if (k < 0) k += m.shape[2];
    return (device T*)((device char*)data + (size_t)m.offset
        + (size_t)(i * m.strides[0]) + (size_t)(j * m.strides[1]) + (size_t)(k * m.strides[2]));
}
template <typename T>
static inline device T* wp_address(device T* data, wp_arr_meta m, int i, int j, int k, int l) {
    if (i < 0) i += m.shape[0];
    if (j < 0) j += m.shape[1];
    if (k < 0) k += m.shape[2];
    if (l < 0) l += m.shape[3];
    return (device T*)((device char*)data + (size_t)m.offset
        + (size_t)(i * m.strides[0]) + (size_t)(j * m.strides[1])
        + (size_t)(k * m.strides[2]) + (size_t)(l * m.strides[3]));
}
static inline int wp_extract_shape(wp_arr_meta m, int i) { return m.shape[i]; }
"""

# ---------------------------------------------------------------------------
# vector constructors / componentwise ops / reductions (vec.h)
# ---------------------------------------------------------------------------

_VEC_OPS = """\
// ---- constructors (zero-init like warp vec_t default: c{} aggregate zero) ----
static inline vec2f make_vec2f() { vec2f v = {0.0f, 0.0f}; return v; }
static inline vec3f make_vec3f() { vec3f v = {0.0f, 0.0f, 0.0f}; return v; }
static inline vec4f make_vec4f() { vec4f v = {0.0f, 0.0f, 0.0f, 0.0f}; return v; }
static inline quatf make_quatf() { quatf v = {0.0f, 0.0f, 0.0f, 0.0f}; return v; }
static inline mat22f make_mat22f() { mat22f m = {{{0.0f, 0.0f}, {0.0f, 0.0f}}}; return m; }
static inline mat33f make_mat33f() { mat33f m = {{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}}; return m; }
static inline mat44f make_mat44f() { mat44f m = {{{0.0f}}}; return m; }
static inline transformf make_transformf() { transformf t; t.p = make_vec3f(); t.q = make_quatf(); return t; }
static inline spatial_vectorf make_spatial_vectorf() { spatial_vectorf s; s.w = make_vec3f(); s.v = make_vec3f(); return s; }
static inline spatial_vectorf make_spatial_vectorf(vec3f w, vec3f v) { spatial_vectorf s; s.w = w; s.v = v; return s; }
static inline spatial_vectorf make_spatial_vectorf(float wx, float wy, float wz, float vx, float vy, float vz) {
    spatial_vectorf s; s.w.x = wx; s.w.y = wy; s.w.z = wz; s.v.x = vx; s.v.y = vy; s.v.z = vz; return s;
}

static inline vec2f make_vec2f(float x, float y) { vec2f v; v.x = x; v.y = y; return v; }
static inline vec3f make_vec3f(float x, float y, float z) { vec3f v; v.x = x; v.y = y; v.z = z; return v; }
static inline vec4f make_vec4f(float x, float y, float z, float w) { vec4f v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline vec5f make_vec5f(float x, float y, float z, float w, float u) { vec5f v; v.x = x; v.y = y; v.z = z; v.w = w; v.u = u; return v; }
static inline vec5f make_vec5f(float s) { return make_vec5f(s, s, s, s, s); }
static inline vec5f make_vec5f() { return make_vec5f(0.0f); }
static inline quatf make_quatf(float x, float y, float z, float w) { quatf v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline quatf make_quatf(vec3f v, float w) { quatf q; q.x = v.x; q.y = v.y; q.z = v.z; q.w = w; return q; }  // quat.h quat_t(vec_t<3>, Type)
// fill constructors: wp.vec3(s) (vec.h operator vec_t(Type s) broadcast)
static inline vec2f make_vec2f(float s) { return make_vec2f(s, s); }
static inline vec3f make_vec3f(float s) { return make_vec3f(s, s, s); }
static inline vec4f make_vec4f(float s) { return make_vec4f(s, s, s, s); }
static inline quatf make_quatf(float s) { return make_quatf(s, s, s, s); }
static inline transformf make_transformf(float s) { transformf t; t.p = make_vec3f(s); t.q = make_quatf(s); return t; }
static inline transformf make_transformf(vec3f p, quatf q) { transformf t; t.p = p; t.q = q; return t; }
// matrix fill constructors: wp.mat33(s) broadcasts (mat.h mat_t(Type s) fill)
static inline mat22f make_mat22f(float s) { mat22f m; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) m.c[i][j] = s; return m; }
static inline mat33f make_mat33f(float s) { mat33f m; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m.c[i][j] = s; return m; }
static inline mat44f make_mat44f(float s) { mat44f m; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m.c[i][j] = s; return m; }
// matrix constructors from N*N scalars, row-major linear (mat.h initializer_array)
static inline mat22f make_mat22f(float a, float b, float c, float d) {
    mat22f m; m.c[0][0]=a; m.c[0][1]=b; m.c[1][0]=c; m.c[1][1]=d; return m;
}
static inline mat33f make_mat33f(float a, float b, float c, float d, float e, float f, float g, float h, float i) {
    mat33f m; m.c[0][0]=a; m.c[0][1]=b; m.c[0][2]=c; m.c[1][0]=d; m.c[1][1]=e; m.c[1][2]=f;
    m.c[2][0]=g; m.c[2][1]=h; m.c[2][2]=i; return m;
}
// ---- integer vectors (vec.h vec_t with int) ----
static inline vec2i make_vec2i() { vec2i v = {0, 0}; return v; }
static inline vec3i make_vec3i() { vec3i v = {0, 0, 0}; return v; }
static inline vec4i make_vec4i() { vec4i v = {0, 0, 0, 0}; return v; }
static inline vec2i make_vec2i(int x, int y) { vec2i v; v.x = x; v.y = y; return v; }
static inline vec3i make_vec3i(int x, int y, int z) { vec3i v; v.x = x; v.y = y; v.z = z; return v; }
static inline vec4i make_vec4i(int x, int y, int z, int w) { vec4i v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline vec2i make_vec2i(int s) { return make_vec2i(s, s); }
static inline vec3i make_vec3i(int s) { return make_vec3i(s, s, s); }
static inline vec4i make_vec4i(int s) { return make_vec4i(s, s, s, s); }
// extract-style component read with runtime index (vec.h extract + component_ref)
static inline int wp_extract(vec2i v, int i) { return i == 0 ? v.x : v.y; }
static inline int wp_extract(vec3i v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline int wp_extract(vec4i v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
// assign_inplace (vec.h component_ref with negative-index wrap)
static inline void wp_assign_inplace(thread vec2i& v, int i, int x) {
    if (i < 0) i += 2;
    if (i == 0) { v.x = x; } else { v.y = x; }
}
static inline void wp_assign_inplace(thread vec3i& v, int i, int x) {
    if (i < 0) i += 3;
    if (i == 0) { v.x = x; } else if (i == 1) { v.y = x; } else { v.z = x; }
}
static inline void wp_assign_inplace(thread vec4i& v, int i, int x) {
    if (i < 0) i += 4;
    if (i == 0) { v.x = x; } else if (i == 1) { v.y = x; } else if (i == 2) { v.z = x; } else { v.w = x; }
}
// extract-style component read with runtime index (vec.h extract + component_ref)
static inline float wp_extract(vec2f v, int i) { return i == 0 ? v.x : v.y; }
static inline float wp_extract(vec3f v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline float wp_extract(vec4f v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
static inline float wp_extract(quatf q, int i) { return i == 0 ? q.x : (i == 1 ? q.y : (i == 2 ? q.z : q.w)); }
static inline float wp_extract(transformf t, int i) { return i < 3 ? wp_extract(t.p, i) : wp_extract(t.q, i - 3); }
static inline float wp_extract(spatial_vectorf s, int i) { return i < 3 ? wp_extract(s.w, i) : wp_extract(s.v, i - 3); }
static inline float wp_extract(vec5f v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : (i == 3 ? v.w : v.u)));
}
// register-vector element store (codegen "store" builtin: wp::store(v, i, x))
static inline vec2f wp_vec_insert(vec2f v, int i, float x) { vec2f r = v; if (i == 0) r.x = x; else r.y = x; return r; }
static inline vec3f wp_vec_insert(vec3f v, int i, float x) { vec3f r = v; if (i == 0) r.x = x; else if (i == 1) r.y = x; else r.z = x; return r; }
static inline vec4f wp_vec_insert(vec4f v, int i, float x) { vec4f r = v; if (i == 0) r.x = x; else if (i == 1) r.y = x; else if (i == 2) r.z = x; else r.w = x; return r; }
static inline vec5f wp_vec_insert(vec5f v, int i, float x) { vec5f r = v; if (i == 0) r.x = x; else if (i == 1) r.y = x; else if (i == 2) r.z = x; else if (i == 3) r.w = x; else r.u = x; return r; }
static inline void wp_assign_inplace(thread vec5f& v, int i, float x) {
    if (i < 0) i += 5;
    if (i == 0) v.x = x; else if (i == 1) v.y = x; else if (i == 2) v.z = x; else if (i == 3) v.w = x; else v.u = x;
}
static inline quatf wp_vec_insert(quatf q, int i, float x) { quatf r = q; if (i == 0) r.x = x; else if (i == 1) r.y = x; else if (i == 2) r.z = x; else r.w = x; return r; }
static inline vec2i wp_vec_insert(vec2i v, int i, int x) { vec2i r = v; if (i == 0) r.x = x; else r.y = x; return r; }
static inline vec3i wp_vec_insert(vec3i v, int i, int x) { vec3i r = v; if (i == 0) r.x = x; else if (i == 1) r.y = x; else r.z = x; return r; }
static inline vec4i wp_vec_insert(vec4i v, int i, int x) { vec4i r = v; if (i == 0) r.x = x; else if (i == 1) r.y = x; else if (i == 2) r.z = x; else r.w = x; return r; }
static inline spatial_vectorf wp_vec_insert(spatial_vectorf s, int i, float x) {
    spatial_vectorf r = s;
    if (i < 3) { r.w = wp_vec_insert(r.w, i, x); } else { r.v = wp_vec_insert(r.v, i - 3, x); }
    return r;
}
static inline transformf wp_vec_insert(transformf t, int i, float x) {
    transformf r = t;
    if (i < 3) { r.p = wp_vec_insert(r.p, i, x); } else { r.q = wp_vec_insert(r.q, i - 3, x); }
    return r;
}
// rows of matrices (mat.h extract(mat, row))
static inline vec2f wp_extract(mat22f m, int r) { return make_vec2f(m.c[r][0], m.c[r][1]); }
static inline vec3f wp_extract(mat33f m, int r) { return make_vec3f(m.c[r][0], m.c[r][1], m.c[r][2]); }
static inline vec4f wp_extract(mat44f m, int r) { return make_vec4f(m.c[r][0], m.c[r][1], m.c[r][2], m.c[r][3]); }
static inline float wp_extract(mat22f m, int r, int c) { return m.c[r][c]; }
static inline float wp_extract(mat33f m, int r, int c) { return m.c[r][c]; }
static inline float wp_extract(mat44f m, int r, int c) { return m.c[r][c]; }
// trace (mat.h wp::trace: sum of the diagonal; gen-mat sizes get their own
// overload from ensure_mat_def)
static inline float wp_trace(mat22f m) { return m.c[0][0] + m.c[1][1]; }
static inline float wp_trace(mat33f m) { return m.c[0][0] + m.c[1][1] + m.c[2][2]; }
static inline float wp_trace(mat44f m) { return m.c[0][0] + m.c[1][1] + m.c[2][2] + m.c[3][3]; }
// neg/add/sub (mat.h operator surface: elementwise, closed over the fixed
// sizes; gen mats get the same from ensure_mat_def)
static inline mat22f wp_neg(mat22f m) { return make_mat22f(-m.c[0][0], -m.c[0][1], -m.c[1][0], -m.c[1][1]); }
static inline mat33f wp_neg(mat33f m) { return make_mat33f(-m.c[0][0], -m.c[0][1], -m.c[0][2], -m.c[1][0], -m.c[1][1], -m.c[1][2], -m.c[2][0], -m.c[2][1], -m.c[2][2]); }
static inline mat44f wp_neg(mat44f m) {
    mat44f r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.c[i][j] = -m.c[i][j]; return r;
}
// div (mat.h:3148/3161: elementwise scalar division, both operand orders)
static inline mat22f wp_div(mat22f m, float s) {
    mat22f r; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.c[i][j] = m.c[i][j] / s; return r;
}
static inline mat22f wp_div(float s, mat22f m) {
    mat22f r; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.c[i][j] = s / m.c[i][j]; return r;
}
static inline mat33f wp_div(mat33f m, float s) {
    mat33f r; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.c[i][j] = m.c[i][j] / s; return r;
}
static inline mat33f wp_div(float s, mat33f m) {
    mat33f r; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.c[i][j] = s / m.c[i][j]; return r;
}
static inline mat44f wp_div(mat44f m, float s) {
    mat44f r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.c[i][j] = m.c[i][j] / s; return r;
}
static inline mat44f wp_div(float s, mat44f m) {
    mat44f r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.c[i][j] = s / m.c[i][j]; return r;
}
// assign_inplace (vec.h / quat.h: component_ref with negative-index wrap)
static inline void wp_assign_inplace(thread vec2f& v, int i, float x) {
    if (i < 0) i += 2;
    if (i == 0) { v.x = x; } else { v.y = x; }
}
static inline void wp_assign_inplace(thread vec3f& v, int i, float x) {
    if (i < 0) i += 3;
    if (i == 0) { v.x = x; } else if (i == 1) { v.y = x; } else { v.z = x; }
}
static inline void wp_assign_inplace(thread vec4f& v, int i, float x) {
    if (i < 0) i += 4;
    if (i == 0) { v.x = x; } else if (i == 1) { v.y = x; } else if (i == 2) { v.z = x; } else { v.w = x; }
}
static inline void wp_assign_inplace(thread mat22f& m, int i, float x) {
    if (i < 0) i += 4;
    m.c[i / 2][i % 2] = x;
}
static inline void wp_assign_inplace(thread mat33f& m, int i, float x) {
    if (i < 0) i += 9;
    m.c[i / 3][i % 3] = x;
}
static inline void wp_assign_inplace(thread mat44f& m, int i, float x) {
    if (i < 0) i += 16;
    m.c[i / 4][i % 4] = x;
}
// row assign (`m[i] = v` on fixed mats, matching the generic matNxM form)
static inline void wp_assign_inplace(thread mat22f& m, int i, vec2f v) {
    if (i < 0) i += 2;
    m.c[i][0] = v.x; m.c[i][1] = v.y;
}
static inline void wp_assign_inplace(thread mat33f& m, int i, vec3f v) {
    if (i < 0) i += 3;
    m.c[i][0] = v.x; m.c[i][1] = v.y; m.c[i][2] = v.z;
}
static inline void wp_assign_inplace(thread mat44f& m, int i, vec4f v) {
    if (i < 0) i += 4;
    m.c[i][0] = v.x; m.c[i][1] = v.y; m.c[i][2] = v.z; m.c[i][3] = v.w;
}
// element assign (`m[i][j] = x` on fixed mats, matching the generic matNxM form)
static inline void wp_assign_inplace(thread mat22f& m, int i, int j, float x) { m.c[i][j] = x; }
static inline void wp_assign_inplace(thread mat33f& m, int i, int j, float x) { m.c[i][j] = x; }
static inline void wp_assign_inplace(thread mat44f& m, int i, int j, float x) { m.c[i][j] = x; }

// lvalue element access (wp::indexref lowering: `&v[i]` on plain structs,// which carry no operator[]); returns a thread reference so stores/loads
// through the pointer alias hit the underlying variable
static inline thread float& wp_vec_ref(thread vec2f& v, int i) { if (i < 0) i += 2; return i == 0 ? v.x : v.y; }
static inline thread float& wp_vec_ref(thread vec3f& v, int i) { if (i < 0) i += 3; return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline thread float& wp_vec_ref(thread vec4f& v, int i) { if (i < 0) i += 4; return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
static inline thread float& wp_vec_ref(thread vec5f& v, int i) { if (i < 0) i += 5; return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : (i == 3 ? v.w : v.u))); }
static inline thread int& wp_vec_ref(thread vec2i& v, int i) { if (i < 0) i += 2; return i == 0 ? v.x : v.y; }
static inline thread int& wp_vec_ref(thread vec3i& v, int i) { if (i < 0) i += 3; return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline thread int& wp_vec_ref(thread vec4i& v, int i) { if (i < 0) i += 4; return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
static inline thread float& wp_vec_ref(thread quatf& q, int i) { if (i < 0) i += 4; return i == 0 ? q.x : (i == 1 ? q.y : (i == 2 ? q.z : q.w)); }

// ---- componentwise operators (vec.h operator overloads) ----
static inline vec2f operator+(vec2f a, vec2f b) { return make_vec2f(a.x+b.x, a.y+b.y); }
static inline vec2f operator-(vec2f a, vec2f b) { return make_vec2f(a.x-b.x, a.y-b.y); }
static inline vec2f operator*(vec2f a, vec2f b) { return make_vec2f(a.x*b.x, a.y*b.y); }
static inline vec2f operator/(vec2f a, vec2f b) { return make_vec2f(a.x/b.x, a.y/b.y); }
static inline vec2f operator*(float s, vec2f b) { return make_vec2f(s*b.x, s*b.y); }
static inline vec2f operator*(vec2f a, float s) { return make_vec2f(a.x*s, a.y*s); }
static inline vec2f operator/(vec2f a, float s) { return make_vec2f(a.x/s, a.y/s); }
static inline vec3f operator+(vec3f a, vec3f b) { return make_vec3f(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3f operator-(vec3f a, vec3f b) { return make_vec3f(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3f operator*(vec3f a, vec3f b) { return make_vec3f(a.x*b.x, a.y*b.y, a.z*b.z); }
static inline vec3f operator/(vec3f a, vec3f b) { return make_vec3f(a.x/b.x, a.y/b.y, a.z/b.z); }
static inline vec3f operator*(float s, vec3f b) { return make_vec3f(s*b.x, s*b.y, s*b.z); }
static inline vec3f operator*(vec3f a, float s) { return make_vec3f(a.x*s, a.y*s, a.z*s); }
static inline vec3f operator/(vec3f a, float s) { return make_vec3f(a.x/s, a.y/s, a.z/s); }
static inline vec4f operator+(vec4f a, vec4f b) { return make_vec4f(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
static inline vec4f operator-(vec4f a, vec4f b) { return make_vec4f(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
static inline vec4f operator*(vec4f a, vec4f b) { return make_vec4f(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
static inline vec4f operator/(vec4f a, vec4f b) { return make_vec4f(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }
static inline vec4f operator*(float s, vec4f b) { return make_vec4f(s*b.x, s*b.y, s*b.z, s*b.w); }
static inline vec4f operator*(vec4f a, float s) { return make_vec4f(a.x*s, a.y*s, a.z*s, a.w*s); }
static inline vec4f operator/(vec4f a, float s) { return make_vec4f(a.x/s, a.y/s, a.z/s, a.w/s); }
static inline vec2f operator-(vec2f a) { return make_vec2f(-a.x, -a.y); }
static inline vec3f operator-(vec3f a) { return make_vec3f(-a.x, -a.y, -a.z); }
static inline vec4f operator-(vec4f a) { return make_vec4f(-a.x, -a.y, -a.z, -a.w); }
// integer vector operators (vec.h vec_t with int)
static inline vec2i operator+(vec2i a, vec2i b) { return make_vec2i(a.x+b.x, a.y+b.y); }
static inline vec2i operator-(vec2i a, vec2i b) { return make_vec2i(a.x-b.x, a.y-b.y); }
static inline vec2i operator*(vec2i a, vec2i b) { return make_vec2i(a.x*b.x, a.y*b.y); }
static inline vec2i operator/(vec2i a, vec2i b) { return make_vec2i(a.x/b.x, a.y/b.y); }
static inline vec3i operator+(vec3i a, vec3i b) { return make_vec3i(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3i operator-(vec3i a, vec3i b) { return make_vec3i(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3i operator*(vec3i a, vec3i b) { return make_vec3i(a.x*b.x, a.y*b.y, a.z*b.z); }
static inline vec3i operator/(vec3i a, vec3i b) { return make_vec3i(a.x/b.x, a.y/b.y, a.z/b.z); }
static inline vec4i operator+(vec4i a, vec4i b) { return make_vec4i(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
static inline vec4i operator-(vec4i a, vec4i b) { return make_vec4i(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
static inline vec4i operator*(vec4i a, vec4i b) { return make_vec4i(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
static inline vec4i operator/(vec4i a, vec4i b) { return make_vec4i(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }
static inline vec2i operator-(vec2i a) { return make_vec2i(-a.x, -a.y); }
static inline vec3i operator-(vec3i a) { return make_vec3i(-a.x, -a.y, -a.z); }
static inline vec4i operator-(vec4i a) { return make_vec4i(-a.x, -a.y, -a.z, -a.w); }

// ---- reductions (vec.h dot/length/length_sq/normalize; exact formulas) ----
static inline float dot(vec2f a, vec2f b) { return a.x*b.x + a.y*b.y; }
static inline float dot(vec3f a, vec3f b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float dot(vec4f a, vec4f b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
static inline vec3f cross(vec3f a, vec3f b) {  // vec.h cross
    return make_vec3f(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float length(vec2f a) { return sqrt(dot(a, a)); }
static inline float length(vec3f a) { return sqrt(dot(a, a)); }
static inline float length(vec4f a) { return sqrt(dot(a, a)); }
static inline float length_sq(vec2f a) { return dot(a, a); }
static inline float length_sq(vec3f a) { return dot(a, a); }
static inline float length_sq(vec4f a) { return dot(a, a); }
// vec.h normalize: eps guard, fallback to unit axis for length 0
static inline vec2f normalize(vec2f a) {
    float l = length(a);
    if (l > 1e-20f) { float inv_l = 1.0f / l; return make_vec2f(a.x*inv_l, a.y*inv_l); }
    return make_vec2f(0.0f, 0.0f);
}
static inline vec3f normalize(vec3f a) {
    float l = length(a);
    if (l > 1e-20f) { float inv_l = 1.0f / l; return make_vec3f(a.x*inv_l, a.y*inv_l, a.z*inv_l); }
    return make_vec3f(0.0f, 0.0f, 0.0f);
}
static inline vec4f normalize(vec4f a) {
    float l = length(a);
    if (l > 1e-20f) { float inv_l = 1.0f / l; return make_vec4f(a.x*inv_l, a.y*inv_l, a.z*inv_l, a.w*inv_l); }
    return make_vec4f(0.0f, 0.0f, 0.0f, 0.0f);
}
static inline bool wp_isfinite(vec2f a) { return isfinite(a.x) && isfinite(a.y); }
static inline bool wp_isfinite(vec3f a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }
static inline bool wp_isfinite(vec4f a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && isfinite(a.w); }
static inline bool wp_isfinite(quatf q) { return isfinite(q.x) && isfinite(q.y) && isfinite(q.z) && isfinite(q.w); }
static inline bool wp_isfinite(transformf t) { return wp_isfinite(t.p) && wp_isfinite(t.q); }
static inline bool wp_isfinite(spatial_vectorf s) { return wp_isfinite(s.w) && wp_isfinite(s.v); }
// componentwise min/max (vec.h min/max overloads)
static inline vec2f wp_min(vec2f a, vec2f b) { return make_vec2f(min(a.x,b.x), min(a.y,b.y)); }
static inline vec2f wp_max(vec2f a, vec2f b) { return make_vec2f(max(a.x,b.x), max(a.y,b.y)); }
static inline vec3f wp_min(vec3f a, vec3f b) { return make_vec3f(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)); }
static inline vec3f wp_max(vec3f a, vec3f b) { return make_vec3f(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
static inline vec4f wp_min(vec4f a, vec4f b) { return make_vec4f(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w)); }
static inline vec4f wp_max(vec4f a, vec4f b) { return make_vec4f(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w)); }
// single-arg component reductions (wp.min(vec) / wp.max(vec))
static inline float wp_min(vec2f a) { return min(a.x, a.y); }
static inline float wp_min(vec3f a) { return min(min(a.x, a.y), a.z); }
static inline float wp_min(vec4f a) { return min(min(a.x, a.y), min(a.z, a.w)); }
static inline float wp_max(vec2f a) { return max(a.x, a.y); }
static inline float wp_max(vec3f a) { return max(max(a.x, a.y), a.z); }
static inline float wp_max(vec4f a) { return max(max(a.x, a.y), max(a.z, a.w)); }
static inline vec2f wp_abs(vec2f a) { return make_vec2f(abs(a.x), abs(a.y)); }
static inline vec3f wp_abs(vec3f a) { return make_vec3f(abs(a.x), abs(a.y), abs(a.z)); }
static inline vec4f wp_abs(vec4f a) { return make_vec4f(abs(a.x), abs(a.y), abs(a.z), abs(a.w)); }
// scalar clamp (builtin.h: min(max(a, x), b))
static inline float wp_clamp(float x, float a, float b) { return min(max(a, x), b); }
static inline int wp_clamp(int x, int a, int b) { return min(max(a, x), b); }
static inline vec3f wp_clamp(vec3f x, float a, float b) { return wp_min(wp_max(make_vec3f(a), x), make_vec3f(b)); }
// nonzero (builtin.h: x == T(0) ? T(0) : T(1))
static inline float wp_nonzero(float x) { return x == 0.0f ? 0.0f : 1.0f; }
static inline int wp_nonzero(int x) { return x == 0 ? 0 : 1; }
static inline uint wp_nonzero(uint x) { return x == 0u ? 0u : 1u; }
static inline bool wp_nonzero(bool x) { return x; }
// where/select (builtins.py where overloads)
static inline vec2f wp_where(bool c, vec2f a, vec2f b) { return c ? a : b; }
static inline vec3f wp_where(bool c, vec3f a, vec3f b) { return c ? a : b; }
static inline vec4f wp_where(bool c, vec4f a, vec4f b) { return c ? a : b; }
static inline quatf wp_where(bool c, quatf a, quatf b) { return c ? a : b; }
static inline mat33f wp_where(bool c, mat33f a, mat33f b) { return c ? a : b; }
// warp skew(v): the cross-product matrix with skew(v) * x == cross(v, x)
static inline mat33f wp_skew(vec3f v) {
    mat33f m = make_mat33f();
    m.c[0][1] = -v.z; m.c[0][2] = v.y;
    m.c[1][0] = v.z;  m.c[1][2] = -v.x;
    m.c[2][0] = -v.y; m.c[2][1] = v.x;
    return m;
}
static inline transformf wp_where(bool c, transformf a, transformf b) { return c ? a : b; }
static inline spatial_vectorf wp_where(bool c, spatial_vectorf a, spatial_vectorf b) { return c ? a : b; }
static inline float wp_where(bool c, float a, float b) { return c ? a : b; }
static inline int wp_where(bool c, int a, int b) { return c ? a : b; }
static inline vec2i wp_where(bool c, vec2i a, vec2i b) { return c ? a : b; }
static inline vec3i wp_where(bool c, vec3i a, vec3i b) { return c ? a : b; }
static inline vec4i wp_where(bool c, vec4i a, vec4i b) { return c ? a : b; }
static inline uint wp_where(bool c, uint a, uint b) { return c ? a : b; }
static inline uint64_t wp_where(bool c, uint64_t a, uint64_t b) { return c ? a : b; }
static inline int64_t wp_where(bool c, int64_t a, int64_t b) { return c ? a : b; }
// generic fallback: user structs (Vert_hash, ContactData_hash ...) phi-merge
// through wp_where too — the template covers every type at once (exact
// non-template overloads above win when present, same semantics)
template <typename T> static inline T wp_where(bool c, T a, T b) { return c ? a : b; }
"""

# quat assign_inplace needs care: quat is distinct type from vec4f.
_QUAT_ASSIGN = """\
static inline void wp_assign_inplace(thread quatf& q, int i, float x) {
    if (i < 0) i += 4;
    if (i == 0) { q.x = x; } else if (i == 1) { q.y = x; } else if (i == 2) { q.z = x; } else { q.w = x; }
}
"""

# ---------------------------------------------------------------------------
# matrix ops (mat.h): matmul, transpose, inverse (exact warp native ports)
# ---------------------------------------------------------------------------

_MAT_OPS = """\
// ---- matrix products (mat.h mul: mat*mat matmul, mat*vec matvec) ----
static inline mat22f wp_mul(mat22f a, mat22f b) {
    mat22f r;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            r.c[i][j] = a.c[i][0]*b.c[0][j] + a.c[i][1]*b.c[1][j];
    return r;
}
static inline mat33f wp_mul(mat33f a, mat33f b) {
    mat33f r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.c[i][j] = a.c[i][0]*b.c[0][j] + a.c[i][1]*b.c[1][j] + a.c[i][2]*b.c[2][j];
    return r;
}
static inline mat44f wp_mul(mat44f a, mat44f b) {
    mat44f r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r.c[i][j] = a.c[i][0]*b.c[0][j] + a.c[i][1]*b.c[1][j] + a.c[i][2]*b.c[2][j] + a.c[i][3]*b.c[3][j];
    return r;
}
static inline vec2f wp_mul(mat22f m, vec2f v) {
    return make_vec2f(m.c[0][0]*v.x + m.c[0][1]*v.y, m.c[1][0]*v.x + m.c[1][1]*v.y);
}
static inline vec3f wp_mul(mat33f m, vec3f v) {
    return make_vec3f(
        m.c[0][0]*v.x + m.c[0][1]*v.y + m.c[0][2]*v.z,
        m.c[1][0]*v.x + m.c[1][1]*v.y + m.c[1][2]*v.z,
        m.c[2][0]*v.x + m.c[2][1]*v.y + m.c[2][2]*v.z);
}
static inline vec4f wp_mul(mat44f m, vec4f v) {
    return make_vec4f(
        m.c[0][0]*v.x + m.c[0][1]*v.y + m.c[0][2]*v.z + m.c[0][3]*v.w,
        m.c[1][0]*v.x + m.c[1][1]*v.y + m.c[1][2]*v.z + m.c[1][3]*v.w,
        m.c[2][0]*v.x + m.c[2][1]*v.y + m.c[2][2]*v.z + m.c[2][3]*v.w,
        m.c[3][0]*v.x + m.c[3][1]*v.y + m.c[3][2]*v.z + m.c[3][3]*v.w);
}
static inline mat22f wp_mul(float s, mat22f m) {
    mat22f r;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            r.c[i][j] = s*m.c[i][j];
    return r;
}
static inline mat33f wp_mul(float s, mat33f m) {
    mat33f r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.c[i][j] = s*m.c[i][j];
    return r;
}
static inline mat44f wp_mul(float s, mat44f m) {
    mat44f r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r.c[i][j] = s*m.c[i][j];
    return r;
}
static inline mat22f wp_mul(mat22f m, float s) { return wp_mul(s, m); }
static inline mat33f wp_mul(mat33f m, float s) { return wp_mul(s, m); }
static inline mat44f wp_mul(mat44f m, float s) { return wp_mul(s, m); }
static inline mat22f wp_transpose(mat22f m) {
    mat22f r;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.c[i][j] = m.c[j][i];
    return r;
}
static inline mat33f wp_transpose(mat33f m) {
    mat33f r;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.c[i][j] = m.c[j][i];
    return r;
}
static inline mat44f wp_transpose(mat44f m) {
    mat44f r;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.c[i][j] = m.c[j][i];
    return r;
}
// matrix_from_rows/cols: row-major c[row][col], col j of from_cols is arg j
static inline mat22f wp_matrix_from_cols(vec2f c0, vec2f c1) {
    mat22f r;
    r.c[0][0]=c0.x; r.c[0][1]=c1.x;
    r.c[1][0]=c0.y; r.c[1][1]=c1.y;
    return r;
}
static inline mat33f wp_matrix_from_cols(vec3f c0, vec3f c1, vec3f c2) {
    mat33f r;
    r.c[0][0]=c0.x; r.c[0][1]=c1.x; r.c[0][2]=c2.x;
    r.c[1][0]=c0.y; r.c[1][1]=c1.y; r.c[1][2]=c2.y;
    r.c[2][0]=c0.z; r.c[2][1]=c1.z; r.c[2][2]=c2.z;
    return r;
}
static inline mat44f wp_matrix_from_cols(vec4f c0, vec4f c1, vec4f c2, vec4f c3) {
    mat44f r;
    r.c[0][0]=c0.x; r.c[0][1]=c1.x; r.c[0][2]=c2.x; r.c[0][3]=c3.x;
    r.c[1][0]=c0.y; r.c[1][1]=c1.y; r.c[1][2]=c2.y; r.c[1][3]=c3.y;
    r.c[2][0]=c0.z; r.c[2][1]=c1.z; r.c[2][2]=c2.z; r.c[2][3]=c3.z;
    r.c[3][0]=c0.w; r.c[3][1]=c1.w; r.c[3][2]=c2.w; r.c[3][3]=c3.w;
    return r;
}
static inline mat22f wp_matrix_from_rows(vec2f r0, vec2f r1) { return wp_transpose(wp_matrix_from_cols(r0, r1)); }
static inline mat33f wp_matrix_from_rows(vec3f r0, vec3f r1, vec3f r2) { return wp_transpose(wp_matrix_from_cols(r0, r1, r2)); }
static inline mat44f wp_matrix_from_rows(vec4f r0, vec4f r1, vec4f r2, vec4f r3) { return wp_transpose(wp_matrix_from_cols(r0, r1, r2, r3)); }
// exact port of warp native mat.h inverse_impl<3,3> (cofactor form: full
// adjugate first, then elementwise scale by 1/det) for numeric parity
static inline mat33f wp_inverse(mat33f m) {
    float det = m.c[0][0]*(m.c[1][1]*m.c[2][2] - m.c[1][2]*m.c[2][1])
              - m.c[0][1]*(m.c[1][0]*m.c[2][2] - m.c[1][2]*m.c[2][0])
              + m.c[0][2]*(m.c[1][0]*m.c[2][1] - m.c[1][1]*m.c[2][0]);
    if (det != 0.0f) {
        mat33f b;
        b.c[0][0] = m.c[1][1]*m.c[2][2] - m.c[1][2]*m.c[2][1];
        b.c[1][0] = m.c[1][2]*m.c[2][0] - m.c[1][0]*m.c[2][2];
        b.c[2][0] = m.c[1][0]*m.c[2][1] - m.c[1][1]*m.c[2][0];
        b.c[0][1] = m.c[0][2]*m.c[2][1] - m.c[0][1]*m.c[2][2];
        b.c[1][1] = m.c[0][0]*m.c[2][2] - m.c[0][2]*m.c[2][0];
        b.c[2][1] = m.c[0][1]*m.c[2][0] - m.c[0][0]*m.c[2][1];
        b.c[0][2] = m.c[0][1]*m.c[1][2] - m.c[0][2]*m.c[1][1];
        b.c[1][2] = m.c[0][2]*m.c[1][0] - m.c[0][0]*m.c[1][2];
        b.c[2][2] = m.c[0][0]*m.c[1][1] - m.c[0][1]*m.c[1][0];
        float rcp_det = 1.0f / det;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                b.c[i][j] *= rcp_det;
        return b;
    }
    return make_mat33f();
}
// ddot: double-dot / Frobenius inner product (mat.h tensordot: sum m.data[i][j]*n.data[i][j])
static inline float wp_ddot(mat33f m, mat33f n) {
    float r = 0.0f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r += m.c[i][j]*n.c[i][j];
    return r;
}
"""

# ---------------------------------------------------------------------------
# quat + transform + spatial ops (quat.h / spatial.h)
# ---------------------------------------------------------------------------

_QUAT_TRANSFORM_OPS = """\
// ---- quat ops (quat.h, exact ports) ----
static inline quatf wp_quat_identity() { return make_quatf(0.0f, 0.0f, 0.0f, 1.0f); }
static inline quatf wp_quat_inverse(quatf q) { return make_quatf(-q.x, -q.y, -q.z, q.w); }  // quat.h quat_inverse
static inline float wp_dot(quatf a, quatf b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
static inline float wp_length(quatf q) { return sqrt(wp_dot(q, q)); }
static inline float wp_length_sq(quatf q) { return wp_dot(q, q); }
static inline quatf wp_normalize(quatf q) {  // quat.h normalize (eps guard -> unit quat)
    float l = wp_length(q);
    if (l > 1e-20f) { float inv_l = 1.0f / l; return make_quatf(q.x*inv_l, q.y*inv_l, q.z*inv_l, q.w*inv_l); }
    return make_quatf(0.0f, 0.0f, 0.0f, 1.0f);
}
// quat.h quat_rotate (Rodrigues form used by warp native)
static inline vec3f wp_quat_rotate(quatf q, vec3f x) {
    float c = (2.0f * q.w * q.w - 1.0f);
    float d = 2.0f * (q.x * x.x + q.y * x.y + q.z * x.z);
    return make_vec3f(
        x.x * c + q.x * d + (q.y * x.z - q.z * x.y) * q.w * 2.0f,
        x.y * c + q.y * d + (q.z * x.x - q.x * x.z) * q.w * 2.0f,
        x.z * c + q.z * d + (q.x * x.y - q.y * x.x) * q.w * 2.0f
    );
}
static inline vec3f wp_quat_rotate_inv(quatf q, vec3f x) {
    float c = (2.0f * q.w * q.w - 1.0f);
    float d = 2.0f * (q.x * x.x + q.y * x.y + q.z * x.z);
    return make_vec3f(
        x.x * c + q.x * d - (q.y * x.z - q.z * x.y) * q.w * 2.0f,
        x.y * c + q.y * d - (q.z * x.x - q.x * x.z) * q.w * 2.0f,
        x.z * c + q.z * d - (q.x * x.y - q.y * x.x) * q.w * 2.0f
    );
}
// quat.h quat_to_matrix: columns are rotated basis vectors, matrix_from_cols
static inline mat33f wp_quat_to_matrix(quatf q) {
    vec3f c1 = wp_quat_rotate(q, make_vec3f(1.0f, 0.0f, 0.0f));
    vec3f c2 = wp_quat_rotate(q, make_vec3f(0.0f, 1.0f, 0.0f));
    vec3f c3 = wp_quat_rotate(q, make_vec3f(0.0f, 0.0f, 1.0f));
    mat33f m;
    m.c[0][0] = c1.x; m.c[0][1] = c2.x; m.c[0][2] = c3.x;
    m.c[1][0] = c1.y; m.c[1][1] = c2.y; m.c[1][2] = c3.y;
    m.c[2][0] = c1.z; m.c[2][1] = c2.z; m.c[2][2] = c3.z;
    return m;
}

// quat.h quat_from_matrix (Shepperd), exact port for numeric parity
static inline quatf wp_quat_from_matrix(mat33f m) {
    float tr = m.c[0][0] + m.c[1][1] + m.c[2][2];
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f, h = 0.0f;
    if (tr >= 0.0f) {
        h = sqrt(tr + 1.0f);
        w = 0.5f * h;
        h = 0.5f / h;
        x = (m.c[2][1] - m.c[1][2]) * h;
        y = (m.c[0][2] - m.c[2][0]) * h;
        z = (m.c[1][0] - m.c[0][1]) * h;
    } else {
        int max_diag = 0;
        if (m.c[1][1] > m.c[0][0]) max_diag = 1;
        if (m.c[2][2] > m.c[max_diag][max_diag]) max_diag = 2;
        if (max_diag == 0) {
            h = sqrt((m.c[0][0] - (m.c[1][1] + m.c[2][2])) + 1.0f);
            x = 0.5f * h;
            h = 0.5f / h;
            y = (m.c[0][1] + m.c[1][0]) * h;
            z = (m.c[2][0] + m.c[0][2]) * h;
            w = (m.c[2][1] - m.c[1][2]) * h;
        } else if (max_diag == 1) {
            h = sqrt((m.c[1][1] - (m.c[2][2] + m.c[0][0])) + 1.0f);
            y = 0.5f * h;
            h = 0.5f / h;
            z = (m.c[1][2] + m.c[2][1]) * h;
            x = (m.c[0][1] + m.c[1][0]) * h;
            w = (m.c[0][2] - m.c[2][0]) * h;
        }
        if (max_diag == 2) {
            h = sqrt((m.c[2][2] - (m.c[0][0] + m.c[1][1])) + 1.0f);
            z = 0.5f * h;
            h = 0.5f / h;
            x = (m.c[2][0] + m.c[0][2]) * h;
            y = (m.c[1][2] + m.c[2][1]) * h;
            w = (m.c[1][0] - m.c[0][1]) * h;
        }
    }
    float len = sqrt(x*x + y*y + z*z + w*w);
    return make_quatf(x/len, y/len, z/len, w/len);
}

// quat.h quat_from_axis_angle (exact port)
static inline quatf wp_quat_from_axis_angle(vec3f axis, float angle) {
    float half_angle = angle * 0.5f;
    float w = cos(half_angle);
    float s = sin(half_angle);
    return make_quatf(axis.x*s, axis.y*s, axis.z*s, w);
}

// ---- transform ops (spatial.h, exact ports) ----
static inline transformf wp_transform_identity() { transformf t; t.p = make_vec3f(); t.q = wp_quat_identity(); return t; }
static inline vec3f wp_transform_get_translation(transformf t) { return t.p; }
static inline quatf wp_transform_get_rotation(transformf t) { return t.q; }
static inline vec3f wp_transform_point(transformf t, vec3f x) { return t.p + wp_quat_rotate(t.q, x); }
static inline vec3f wp_transform_vector(transformf t, vec3f x) { return wp_quat_rotate(t.q, x); }
static inline transformf wp_transform_multiply(transformf a, transformf b) {
    transformf r;
    r.p = wp_quat_rotate(a.q, b.p) + a.p;
    // quat*quat Hamilton product, formula from quat.h mul(quat_t, quat_t)
    quatf q;
    q.w = a.q.w*b.q.w - a.q.x*b.q.x - a.q.y*b.q.y - a.q.z*b.q.z;
    q.x = a.q.w*b.q.x + a.q.x*b.q.w + a.q.y*b.q.z - a.q.z*b.q.y;
    q.y = a.q.w*b.q.y - a.q.x*b.q.z + a.q.y*b.q.w + a.q.z*b.q.x;
    q.z = a.q.w*b.q.z + a.q.x*b.q.y - a.q.y*b.q.x + a.q.z*b.q.w;
    r.q = q;
    return r;
}
static inline transformf wp_transform_inverse(transformf t) {
    quatf q_inv = wp_quat_inverse(t.q);
    transformf r;
    r.p = -(wp_quat_rotate(q_inv, t.p));
    r.q = q_inv;
    return r;
}

// ---- spatial vector ops (spatial.h) ----
static inline vec3f wp_spatial_top(spatial_vectorf a) { return a.w; }
static inline vec3f wp_spatial_bottom(spatial_vectorf a) { return a.v; }
static inline float wp_dot(spatial_vectorf a, spatial_vectorf b) { return dot(a.w, b.w) + dot(a.v, b.v); }
static inline float wp_length_sq(spatial_vectorf s) { return wp_dot(s, s); }
static inline float wp_length(spatial_vectorf s) { return sqrt(wp_length_sq(s)); }
static inline spatial_vectorf operator-(spatial_vectorf a) {
    spatial_vectorf r; r.w = -a.w; r.v = -a.v; return r;
}
"""

# ---------------------------------------------------------------------------
# scalar builtins with warp-native semantics
# ---------------------------------------------------------------------------

_SCALAR_OPS = """\
// ---- scalar semantics (builtin.h DECLARE macros; Python-mod for floats) ----
static inline int wp_mod(int a, int b) { return a % b; }            // builtin.h: a%b
static inline uint wp_mod(uint a, uint b) { return a % b; }
static inline int64_t wp_mod(int64_t a, int64_t b) { return a % b; }
static inline uint64_t wp_mod(uint64_t a, uint64_t b) { return a % b; }
static inline float wp_mod(float a, float b) {                      // builtin.h float mod: fmod + Python-sign fix
    float r = fmod(a, b);
    if (r != 0.0f && (r < 0.0f) != (b < 0.0f)) r += b;
    return r;
}
static inline float wp_asin(float x) { return asin(clamp(x, -1.0f, 1.0f)); }   // domain guard: FMA noise can push |x|>1 (CPU stays inside)
static inline float wp_acos(float x) { return acos(clamp(x, -1.0f, 1.0f)); }
static inline float wp_atan(float x) { return atan(x); }
static inline float wp_atan2(float y, float x) { return atan2(y, x); }
static inline float wp_sin(float x) { return precise::sin(x); }     // align CPU numeric path
static inline float wp_cos(float x) { return precise::cos(x); }
static inline float wp_tan(float x) { return precise::tan(x); }
static inline float wp_sqrt(float x) { return sqrt(x); }
static inline float wp_log(float x) { return precise::log(x); }     // align CPU numeric path
static inline float wp_exp(float x) { return precise::exp(x); }     // align CPU numeric path
static inline bool wp_isfinite(float x) { return isfinite(x); }
static inline bool wp_isfinite(int x) { return true; }
static inline float wp_nan() { return NAN; }
// atomic ops (Metal device atomics; float via atomic_fetch_add_explicit)
static inline void wp_atomic_add(device atomic_float* addr, float value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_add(device atomic_int* addr, int value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_add(device atomic_uint* addr, uint value) {
    atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline int wp_atomic_add_ret(device atomic_int* addr, int value) {
    return atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline uint wp_atomic_add_ret(device atomic_uint* addr, uint value) {
    return atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_sub(device atomic_int* addr, int value) {
    atomic_fetch_sub_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_sub(device atomic_uint* addr, uint value) {
    atomic_fetch_sub_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_min(device atomic_int* addr, int value) {
    int old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value < old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
}
static inline void wp_atomic_min(device atomic_uint* addr, uint value) {
    uint old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value < old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
}
static inline void wp_atomic_max(device atomic_int* addr, int value) {
    int old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value > old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
}
static inline void wp_atomic_max(device atomic_uint* addr, uint value) {
    uint old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value > old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
}
static inline void wp_atomic_min(device atomic_float* addr, float value) {
    float old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value < old) {
        if (atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) break;
    }
}
static inline void wp_atomic_max(device atomic_float* addr, float value) {
    float old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value > old) {
        if (atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) break;
    }
}
// _ret cores: upstream atomics return the OLD value (builtin.h atomicAdd /
// atomicMin return type; float CAS loop returns old), used when the SSA
// captures the result (var_x = wp::atomic_min(...)).
static inline float wp_atomic_add_ret(device atomic_float* addr, float value) {
    return atomic_fetch_add_explicit(addr, value, memory_order_relaxed);
}
static inline int wp_atomic_sub_ret(device atomic_int* addr, int value) {
    return atomic_fetch_sub_explicit(addr, value, memory_order_relaxed);
}
static inline uint wp_atomic_sub_ret(device atomic_uint* addr, uint value) {
    return atomic_fetch_sub_explicit(addr, value, memory_order_relaxed);
}
// bitwise atomics (Newton uses or/and on mask/flag words; xor for parity)
static inline void wp_atomic_or(device atomic_int* addr, int value) {
    atomic_fetch_or_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_or(device atomic_uint* addr, uint value) {
    atomic_fetch_or_explicit(addr, value, memory_order_relaxed);
}
static inline int wp_atomic_or_ret(device atomic_int* addr, int value) {
    return atomic_fetch_or_explicit(addr, value, memory_order_relaxed);
}
static inline uint wp_atomic_or_ret(device atomic_uint* addr, uint value) {
    return atomic_fetch_or_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_and(device atomic_int* addr, int value) {
    atomic_fetch_and_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_and(device atomic_uint* addr, uint value) {
    atomic_fetch_and_explicit(addr, value, memory_order_relaxed);
}
static inline int wp_atomic_and_ret(device atomic_int* addr, int value) {
    return atomic_fetch_and_explicit(addr, value, memory_order_relaxed);
}
static inline uint wp_atomic_and_ret(device atomic_uint* addr, uint value) {
    return atomic_fetch_and_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_xor(device atomic_int* addr, int value) {
    atomic_fetch_xor_explicit(addr, value, memory_order_relaxed);
}
static inline void wp_atomic_xor(device atomic_uint* addr, uint value) {
    atomic_fetch_xor_explicit(addr, value, memory_order_relaxed);
}
static inline int wp_atomic_xor_ret(device atomic_int* addr, int value) {
    return atomic_fetch_xor_explicit(addr, value, memory_order_relaxed);
}
static inline uint wp_atomic_xor_ret(device atomic_uint* addr, uint value) {
    return atomic_fetch_xor_explicit(addr, value, memory_order_relaxed);
}
static inline int wp_atomic_min_ret(device atomic_int* addr, int value) {
    int old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value < old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
    return old;
}
static inline uint wp_atomic_min_ret(device atomic_uint* addr, uint value) {
    uint old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value < old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
    return old;
}
static inline float wp_atomic_min_ret(device atomic_float* addr, float value) {
    float old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value < old) {
        if (atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) break;
    }
    return old;
}
static inline int wp_atomic_max_ret(device atomic_int* addr, int value) {
    int old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value > old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
    return old;
}
static inline uint wp_atomic_max_ret(device atomic_uint* addr, uint value) {
    uint old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value > old && !atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
    return old;
}
static inline float wp_atomic_max_ret(device atomic_float* addr, float value) {
    float old = atomic_load_explicit(addr, memory_order_relaxed);
    while (value > old) {
        if (atomic_compare_exchange_weak_explicit(addr, &old, value, memory_order_relaxed, memory_order_relaxed)) break;
    }
    return old;
}

// 64-bit atomics: this Metal runtime exposes the atomic_ulong TYPE but NO
// 64-bit atomic functions (probed on M4/Metal 4, language version pinned to
// 4.0 — atomic_fetch/cas/store with 64-bit operands all fail to resolve).
// Emulated via 32-bit CAS pairs below. Guarantees: convergent (unique-key
// inserts and monotone max loops retry on unexpected old values) — not
// linearizable under adversarial same-word contention, unlike CUDA's native
// 64-bit CAS. Any warp kernel doing arbitrary 64-bit atomics gets this
// emulation; corpus/trajectory layers validate the reducer's usage patterns.
static inline uint64_t wp_atomic_u64_load(device atomic_uint* halves) {
    for (;;) {
        uint lo = atomic_load_explicit(halves, memory_order_relaxed);
        uint hi = atomic_load_explicit(halves + 1, memory_order_relaxed);
        uint lo2 = atomic_load_explicit(halves, memory_order_relaxed);
        uint hi2 = atomic_load_explicit(halves + 1, memory_order_relaxed);
        if (lo == lo2 && hi == hi2) {
            return ((uint64_t)hi << 32) | (uint64_t)lo;
        }
    }
}
static inline uint64_t wp_atomic_u64_cas(device atomic_uint* halves, uint64_t expected, uint64_t desired) {
    for (;;) {
        uint lo = atomic_load_explicit(halves, memory_order_relaxed);
        uint hi = atomic_load_explicit(halves + 1, memory_order_relaxed);
        uint64_t old = ((uint64_t)hi << 32) | (uint64_t)lo;
        if (old != expected) {
            return old;
        }
        uint new_lo = (uint)(desired & 0xffffffffull);
        uint new_hi = (uint)(desired >> 32);
        if (atomic_compare_exchange_weak_explicit(halves, &lo, new_lo, memory_order_relaxed, memory_order_relaxed)) {
            uint hi_old = hi;
            if (atomic_compare_exchange_weak_explicit(halves + 1, &hi_old, new_hi, memory_order_relaxed, memory_order_relaxed)) {
                return expected;
            }
            // high word raced past us: restore low, retry (convergent)
            atomic_store_explicit(halves, lo, memory_order_relaxed);
        }
    }
}
static inline uint64_t wp_atomic_max_u64(device atomic_uint* halves, uint64_t value) {
    for (;;) {
        uint64_t old = wp_atomic_u64_load(halves);
        if (value <= old) {
            return old;
        }
        if (wp_atomic_u64_cas(halves, old, value) == old) {
            return old;
        }
    }
}
static inline uint64_t wp_atomic_min_u64(device atomic_uint* halves, uint64_t value) {
    for (;;) {
        uint64_t old = wp_atomic_u64_load(halves);
        if (value >= old) {
            return old;
        }
        if (wp_atomic_u64_cas(halves, old, value) == old) {
            return old;
        }
    }
}
// compare-and-swap (upstream atomic_compare_swap returns the OLD value);
static inline uint64_t wp_atomic_cas_ret(device atomic_ulong* addr, uint64_t expected, uint64_t desired) {
    return wp_atomic_u64_cas((device atomic_uint*)addr, expected, desired);
}
static inline int wp_atomic_cas_ret(device atomic_int* addr, int expected, int desired) {
    atomic_compare_exchange_weak_explicit(addr, &expected, desired, memory_order_relaxed, memory_order_relaxed);
    return expected;
}
static inline uint64_t wp_atomic_cas_ret_at(device uint64_t* d, wp_arr_meta m, int i, uint64_t expected, uint64_t desired) {
    return wp_atomic_cas_ret((device atomic_ulong*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), expected, desired);
}
static inline int wp_atomic_cas_ret_at(device int* d, wp_arr_meta m, int i, int expected, int desired) {
    return wp_atomic_cas_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), expected, desired);
}
static inline uint64_t wp_atomic_max_ret_at(device uint64_t* d, wp_arr_meta m, int i, uint64_t v) {
    device atomic_uint* halves = (device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));
    wp_atomic_max_u64(halves, v);
    return wp_atomic_u64_load(halves);
}
static inline uint64_t wp_atomic_min_ret_at(device uint64_t* d, wp_arr_meta m, int i, uint64_t v) {
    device atomic_uint* halves = (device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));
    wp_atomic_min_u64(halves, v);
    return wp_atomic_u64_load(halves);
}
// ---- wp_*_at: array-indexed atomic entry points (translator emits these) ----
// exact-match overloads only: calling on an unsupported element type is a
// compile error (fail loud), never a silent non-atomic path.
static inline void wp_atomic_add_at(device float* d, wp_arr_meta m, int i, float v) {
    wp_atomic_add((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_or_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_or((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_or_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_or((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_or_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_or_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_or_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_or_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_and_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_and((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_and_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_and((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_and_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_and_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_and_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_and_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_xor_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_xor((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_xor_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_xor((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_xor_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_xor_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_xor_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_xor_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_add_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_add((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_add_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_add((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_add_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_add_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_add_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_add_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_sub_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_sub((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_sub_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_sub((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_min_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_min((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_min_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_min((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_min_at(device float* d, wp_arr_meta m, int i, float v) {
    wp_atomic_min((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_max_at(device int* d, wp_arr_meta m, int i, int v) {
    wp_atomic_max((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_max_at(device uint* d, wp_arr_meta m, int i, uint v) {
    wp_atomic_max((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_max_at(device float* d, wp_arr_meta m, int i, float v) {
    wp_atomic_max((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
// 2D variants (array2d atomics)
static inline void wp_atomic_add_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    wp_atomic_add((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline void wp_atomic_add_at(device int* d, wp_arr_meta m, int i, int j, int v) {
    wp_atomic_add((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline void wp_atomic_min_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    wp_atomic_min((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline void wp_atomic_max_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    wp_atomic_max((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
// _ret_at: captured-result atomics (upstream returns the old value)
static inline float wp_atomic_add_ret_at(device float* d, wp_arr_meta m, int i, float v) {
    return wp_atomic_add_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_sub_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_sub_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_sub_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_sub_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_min_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_min_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_min_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_min_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline float wp_atomic_min_ret_at(device float* d, wp_arr_meta m, int i, float v) {
    return wp_atomic_min_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_max_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return wp_atomic_max_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_max_ret_at(device uint* d, wp_arr_meta m, int i, uint v) {
    return wp_atomic_max_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline float wp_atomic_max_ret_at(device float* d, wp_arr_meta m, int i, float v) {
    return wp_atomic_max_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
// vector atomics: upstream vec.h:1841-1850 does component-wise atomic_min/max
// through the scalar cores and returns the old vector (Newton
// compute_particle_bounds reduces bounds as vec3 arrays)
static inline vec3f wp_atomic_min_ret_at(device vec3f* d, wp_arr_meta m, int i, vec3f v) {
    device atomic_float* c = (device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));
    vec3f old;
    old.x = wp_atomic_min_ret(c + 0, v.x);
    old.y = wp_atomic_min_ret(c + 1, v.y);
    old.z = wp_atomic_min_ret(c + 2, v.z);
    return old;
}
static inline vec3f wp_atomic_max_ret_at(device vec3f* d, wp_arr_meta m, int i, vec3f v) {
    device atomic_float* c = (device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));
    vec3f old;
    old.x = wp_atomic_max_ret(c + 0, v.x);
    old.y = wp_atomic_max_ret(c + 1, v.y);
    old.z = wp_atomic_max_ret(c + 2, v.z);
    return old;
}
// 2D _ret_at variants (array2d atomics, e.g. world extents reduction)
static inline float wp_atomic_add_ret_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    return wp_atomic_add_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline float wp_atomic_min_ret_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    return wp_atomic_min_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline float wp_atomic_max_ret_at(device float* d, wp_arr_meta m, int i, int j, float v) {
    return wp_atomic_max_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline int wp_atomic_add_ret_at(device int* d, wp_arr_meta m, int i, int j, int v) {
    return wp_atomic_add_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline int wp_atomic_min_ret_at(device int* d, wp_arr_meta m, int i, int j, int v) {
    return wp_atomic_min_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline int wp_atomic_max_ret_at(device int* d, wp_arr_meta m, int i, int j, int v) {
    return wp_atomic_max_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
// 3D variants (array3d atomics, e.g. steel crba_fill_par_dof H_group scatter)
static inline void wp_atomic_add_at(device float* d, wp_arr_meta m, int i, int j, int k, float v) {
    wp_atomic_add((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline void wp_atomic_add_at(device int* d, wp_arr_meta m, int i, int j, int k, int v) {
    wp_atomic_add((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline float wp_atomic_add_ret_at(device float* d, wp_arr_meta m, int i, int j, int k, float v) {
    return wp_atomic_add_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline float wp_atomic_min_ret_at(device float* d, wp_arr_meta m, int i, int j, int k, float v) {
    return wp_atomic_min_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline float wp_atomic_max_ret_at(device float* d, wp_arr_meta m, int i, int j, int k, float v) {
    return wp_atomic_max_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline int wp_atomic_add_ret_at(device int* d, wp_arr_meta m, int i, int j, int k, int v) {
    return wp_atomic_add_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline int wp_atomic_min_ret_at(device int* d, wp_arr_meta m, int i, int j, int k, int v) {
    return wp_atomic_min_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
static inline int wp_atomic_max_ret_at(device int* d, wp_arr_meta m, int i, int j, int k, int v) {
    return wp_atomic_max_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2])), v);
}
// atomic_exch (warp atomic_exch always returns the old value; int/uint only —
// e.g. steel _warn_constraint_row_overflow latching warning flags)
static inline int wp_atomic_exch_ret_at(device int* d, wp_arr_meta m, int i, int v) {
    return atomic_exchange_explicit((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v, memory_order_relaxed);
}
static inline int wp_atomic_exch_ret_at(device int* d, wp_arr_meta m, int i, int j, int v) {
    return atomic_exchange_explicit((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v, memory_order_relaxed);
}

// composite (spatial_vectorf) atomics: per-component fetch on the packed
// 6-float layout (upstream lowers wp.atomic_add on vector dtypes the same way)
static inline spatial_vectorf wp_atomic_add_ret_at(device spatial_vectorf* d, wp_arr_meta m, int i, spatial_vectorf v) {
    device float* base = (device float*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));
    spatial_vectorf r;
    r.w.x = atomic_fetch_add_explicit((device atomic_float*)(base + 0), v.w.x, memory_order_relaxed);
    r.w.y = atomic_fetch_add_explicit((device atomic_float*)(base + 1), v.w.y, memory_order_relaxed);
    r.w.z = atomic_fetch_add_explicit((device atomic_float*)(base + 2), v.w.z, memory_order_relaxed);
    r.v.x = atomic_fetch_add_explicit((device atomic_float*)(base + 3), v.v.x, memory_order_relaxed);
    r.v.y = atomic_fetch_add_explicit((device atomic_float*)(base + 4), v.v.y, memory_order_relaxed);
    r.v.z = atomic_fetch_add_explicit((device atomic_float*)(base + 5), v.v.z, memory_order_relaxed);
    return r;
}
static inline spatial_vectorf wp_atomic_sub_ret_at(device spatial_vectorf* d, wp_arr_meta m, int i, spatial_vectorf v) {
    device float* base = (device float*)((device char*)d + (size_t)(m.offset + i * m.strides[0]));
    spatial_vectorf r;
    r.w.x = atomic_fetch_sub_explicit((device atomic_float*)(base + 0), v.w.x, memory_order_relaxed);
    r.w.y = atomic_fetch_sub_explicit((device atomic_float*)(base + 1), v.w.y, memory_order_relaxed);
    r.w.z = atomic_fetch_sub_explicit((device atomic_float*)(base + 2), v.w.z, memory_order_relaxed);
    r.v.x = atomic_fetch_sub_explicit((device atomic_float*)(base + 3), v.v.x, memory_order_relaxed);
    r.v.y = atomic_fetch_sub_explicit((device atomic_float*)(base + 4), v.v.y, memory_order_relaxed);
    r.v.z = atomic_fetch_sub_explicit((device atomic_float*)(base + 5), v.v.z, memory_order_relaxed);
    return r;
}
// wp.atomic_cas on int64: this Metal runtime has no native 64-bit atomic
// functions (see the 64-bit atomic policy note above); the emulated uint64
// CAS pair works on the identical two's-complement bit pattern.
static inline int64_t wp_atomic_cas_ret_at(device int64_t* d, wp_arr_meta m, int i, int64_t expected, int64_t desired) {
    return (int64_t)wp_atomic_cas_ret_at(
        (device uint64_t*)d, m, i, (uint64_t)expected, (uint64_t)desired);
}

// ---- bit ops (builtin.h bit_and/bit_or/bit_xor/lshift/rshift) ----
static inline int wp_bit_and(int a, int b) { return a & b; }
static inline uint wp_bit_and(uint a, uint b) { return a & b; }
static inline int64_t wp_bit_and(int64_t a, int64_t b) { return a & b; }
static inline uint64_t wp_bit_and(uint64_t a, uint64_t b) { return a & b; }
static inline int wp_bit_or(int a, int b) { return a | b; }
static inline uint wp_bit_or(uint a, uint b) { return a | b; }
static inline int64_t wp_bit_or(int64_t a, int64_t b) { return a | b; }
static inline uint64_t wp_bit_or(uint64_t a, uint64_t b) { return a | b; }
static inline int wp_bit_xor(int a, int b) { return a ^ b; }
static inline uint wp_bit_xor(uint a, uint b) { return a ^ b; }
static inline int64_t wp_bit_xor(int64_t a, int64_t b) { return a ^ b; }
static inline uint64_t wp_bit_xor(uint64_t a, uint64_t b) { return a ^ b; }
static inline int wp_lshift(int a, int b) { return a << b; }
static inline uint wp_lshift(uint a, int b) { return a << b; }
static inline int64_t wp_lshift(int64_t a, int b) { return a << b; }
static inline uint64_t wp_lshift(uint64_t a, int b) { return a << b; }
static inline int wp_rshift(int a, int b) { return a >> b; }
static inline uint wp_rshift(uint a, int b) { return a >> b; }
static inline int64_t wp_rshift(int64_t a, int b) { return a >> b; }
static inline uint64_t wp_rshift(uint64_t a, int b) { return a >> b; }
// invert (builtin.h: bitwise complement, Int types)
static inline int wp_invert(int a) { return ~a; }
static inline uint wp_invert(uint a) { return ~a; }
static inline int64_t wp_invert(int64_t a) { return ~a; }
static inline uint64_t wp_invert(uint64_t a) { return ~a; }
// unot (builtin.h: logical not, Bool types)
static inline bool wp_unot(bool a) { return !a; }
static inline uint64_t wp_rshift(uint64_t a, uint64_t b) { return a >> b; }

// ---- scalar sign / neg / div (builtin.h semantics) ----
static inline float wp_sign(float x) { return x < 0.0f ? -1.0f : 1.0f; }  // upstream: -1 if x<0, else 1 (sign(0)=1)
static inline int wp_sign(int x) { return x < 0 ? -1 : 1; }
// rounding (metal_stdlib floor/ceil/round/trunc/step/pow resolve by name;
// float-returning only — Newton wraps int() casts explicitly)
static inline float wp_floor(float x) { return floor(x); }
static inline float wp_ceil(float x) { return ceil(x); }
static inline float wp_round(float x) { return round(x); }
static inline float wp_trunc(float x) { return trunc(x); }
static inline float wp_step(float x, float y) { return x < y ? 0.0f : 1.0f; }  // builtin.h: y < x ? 0 : 1
static inline float wp_step(float x) { return x < 0.0f ? 1.0f : 0.0f; }  // 1-arg overload: 1.0 if x < 0
static inline float wp_pow(float x, float y) { return pow(x, y); }
static inline bool wp_isinf(float x) { return isinf(x) != 0; }
static inline bool wp_isnan(float x) { return isnan(x) != 0; }
static inline float wp_neg(float x) { return -x; }
static inline int wp_neg(int x) { return -x; }
static inline vec2f wp_neg(vec2f a) { return -a; }
static inline vec3f wp_neg(vec3f a) { return -a; }
static inline vec4f wp_neg(vec4f a) { return -a; }
static inline spatial_vectorf wp_neg(spatial_vectorf s) { return make_spatial_vectorf(-s.w, -s.v); }
// cw_mul/cw_div: component-wise vector ops (upstream sametypes constraint)
static inline vec2f wp_cw_mul(vec2f a, vec2f b) { return make_vec2f(a.x * b.x, a.y * b.y); }
static inline vec3f wp_cw_mul(vec3f a, vec3f b) { return make_vec3f(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline vec4f wp_cw_mul(vec4f a, vec4f b) { return make_vec4f(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static inline vec2i wp_cw_mul(vec2i a, vec2i b) { return make_vec2i(a.x * b.x, a.y * b.y); }
static inline vec3i wp_cw_mul(vec3i a, vec3i b) { return make_vec3i(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline vec4i wp_cw_mul(vec4i a, vec4i b) { return make_vec4i(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static inline vec2f wp_cw_div(vec2f a, vec2f b) { return make_vec2f(a.x / b.x, a.y / b.y); }
static inline vec3f wp_cw_div(vec3f a, vec3f b) { return make_vec3f(a.x / b.x, a.y / b.y, a.z / b.z); }
static inline vec4f wp_cw_div(vec4f a, vec4f b) { return make_vec4f(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
static inline vec2i wp_cw_div(vec2i a, vec2i b) { return make_vec2i(a.x / b.x, a.y / b.y); }
static inline vec3i wp_cw_div(vec3i a, vec3i b) { return make_vec3i(a.x / b.x, a.y / b.y, a.z / b.z); }
static inline vec4i wp_cw_div(vec4i a, vec4i b) { return make_vec4i(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
static inline float wp_div(float a, float b) { return a / b; }
static inline int wp_div(int a, int b) { return a / b; }
// floordiv (warp native builtin.h floordiv_signed, ported verbatim):
// C++ division truncates toward zero; warp floors — correct when signs differ
// and the remainder is nonzero.
static inline int wp_floordiv(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}
static inline int64_t wp_floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}
static inline float wp_add(float a, float b) { return a + b; }
static inline int wp_add(int a, int b) { return a + b; }
static inline float wp_sub(float a, float b) { return a - b; }
static inline int wp_sub(int a, int b) { return a - b; }
static inline float wp_mul(float a, float b) { return a * b; }
static inline int wp_mul(int a, int b) { return a * b; }
static inline int wp_min(int a, int b) { return a < b ? a : b; }
static inline int wp_max(int a, int b) { return a > b ? a : b; }
static inline float wp_min(float a, float b) { return a < b ? a : b; }
static inline float wp_max(float a, float b) { return a > b ? a : b; }
// 32-bit unsigned (Newton contact_reduction float_flip returns wp.uint32)
static inline uint wp_div(uint a, uint b) { return a / b; }
static inline uint wp_add(uint a, uint b) { return a + b; }
static inline uint wp_sub(uint a, uint b) { return a - b; }
static inline uint wp_mul(uint a, uint b) { return a * b; }
static inline uint wp_min(uint a, uint b) { return a < b ? a : b; }
static inline uint wp_max(uint a, uint b) { return a > b ? a : b; }
static inline uint wp_abs(uint a) { return a; }
// 64-bit scalar arithmetic (warp supports int64/uint64 like int)
static inline int64_t wp_add(int64_t a, int64_t b) { return a + b; }
static inline int64_t wp_sub(int64_t a, int64_t b) { return a - b; }
static inline int64_t wp_mul(int64_t a, int64_t b) { return a * b; }
static inline int64_t wp_div(int64_t a, int64_t b) { return a / b; }
static inline int64_t wp_min(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t wp_max(int64_t a, int64_t b) { return a > b ? a : b; }
static inline uint64_t wp_add(uint64_t a, uint64_t b) { return a + b; }
static inline uint64_t wp_sub(uint64_t a, uint64_t b) { return a - b; }
static inline uint64_t wp_mul(uint64_t a, uint64_t b) { return a * b; }
static inline uint64_t wp_div(uint64_t a, uint64_t b) { return a / b; }
static inline uint64_t wp_min(uint64_t a, uint64_t b) { return a < b ? a : b; }
static inline uint64_t wp_max(uint64_t a, uint64_t b) { return a > b ? a : b; }
static inline bool wp_isfinite(int64_t x) { return true; }
static inline bool wp_isfinite(uint64_t x) { return true; }

// ---- vector bridges for the wp_-prefixed call table ----
static inline float wp_dot(vec2f a, vec2f b) { return dot(a, b); }
static inline float wp_dot(vec3f a, vec3f b) { return dot(a, b); }
static inline float wp_dot(vec4f a, vec4f b) { return dot(a, b); }
// outer products (mat.h outer(v, w): r[i][j] = v[i]*w[j], c[][] is row-major)
static inline mat22f wp_outer(vec2f a, vec2f b) {
    mat22f r;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.c[i][j] = wp_extract(a, i) * wp_extract(b, j);
    return r;
}
static inline mat33f wp_outer(vec3f a, vec3f b) {
    mat33f r;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.c[i][j] = wp_extract(a, i) * wp_extract(b, j);
    return r;
}
static inline vec3f wp_cross(vec3f a, vec3f b) { return cross(a, b); }
static inline float wp_length(vec2f a) { return length(a); }
static inline float wp_length(vec3f a) { return length(a); }
static inline float wp_length(vec4f a) { return length(a); }
static inline float wp_length_sq(vec2f a) { return length_sq(a); }
static inline float wp_length_sq(vec3f a) { return length_sq(a); }
static inline float wp_length_sq(vec4f a) { return length_sq(a); }
static inline vec2f wp_normalize(vec2f a) { return normalize(a); }
static inline vec3f wp_normalize(vec3f a) { return normalize(a); }
static inline vec4f wp_normalize(vec4f a) { return normalize(a); }
static inline float wp_abs(float a) { return abs(a); }
static inline int wp_abs(int a) { return abs(a); }
static inline vec2f wp_add(vec2f a, vec2f b) { return a + b; }
static inline vec3f wp_add(vec3f a, vec3f b) { return a + b; }
static inline vec4f wp_add(vec4f a, vec4f b) { return a + b; }
// matrix element-wise add/sub (mat.h add(mat, mat) — e.g. inertia sums)
static inline mat22f wp_add(mat22f a, mat22f b) {
    mat22f r; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.c[i][j] = a.c[i][j] + b.c[i][j]; return r;
}
static inline mat33f wp_add(mat33f a, mat33f b) {
    mat33f r; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.c[i][j] = a.c[i][j] + b.c[i][j]; return r;
}
static inline mat44f wp_add(mat44f a, mat44f b) {
    mat44f r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.c[i][j] = a.c[i][j] + b.c[i][j]; return r;
}
static inline mat22f wp_sub(mat22f a, mat22f b) {
    mat22f r; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.c[i][j] = a.c[i][j] - b.c[i][j]; return r;
}
static inline mat33f wp_sub(mat33f a, mat33f b) {
    mat33f r; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.c[i][j] = a.c[i][j] - b.c[i][j]; return r;
}
static inline mat44f wp_sub(mat44f a, mat44f b) {
    mat44f r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r.c[i][j] = a.c[i][j] - b.c[i][j]; return r;
}
static inline vec2f wp_sub(vec2f a, vec2f b) { return a - b; }
static inline vec3f wp_sub(vec3f a, vec3f b) { return a - b; }
static inline vec4f wp_sub(vec4f a, vec4f b) { return a - b; }
static inline vec2f wp_mul(vec2f a, vec2f b) { return a * b; }
static inline vec3f wp_mul(vec3f a, vec3f b) { return a * b; }
static inline vec4f wp_mul(vec4f a, vec4f b) { return a * b; }
static inline vec2f wp_mul(vec2f a, float s) { return a * s; }
static inline vec3f wp_mul(vec3f a, float s) { return a * s; }
static inline vec4f wp_mul(vec4f a, float s) { return a * s; }
static inline vec2f wp_mul(float s, vec2f a) { return s * a; }
static inline vec3f wp_mul(float s, vec3f a) { return s * a; }
static inline vec4f wp_mul(float s, vec4f a) { return s * a; }
static inline vec2f wp_div(vec2f a, vec2f b) { return a / b; }
static inline vec3f wp_div(vec3f a, vec3f b) { return a / b; }
static inline vec4f wp_div(vec4f a, vec4f b) { return a / b; }
static inline vec2f wp_div(vec2f a, float s) { return a / s; }
static inline vec3f wp_div(vec3f a, float s) { return a / s; }
static inline vec4f wp_div(vec4f a, float s) { return a / s; }
// spatial vector arithmetic (spatial.h add/sub/mul overloads)
static inline spatial_vectorf operator+(spatial_vectorf a, spatial_vectorf b) {
    spatial_vectorf r; r.w = a.w + b.w; r.v = a.v + b.v; return r;
}
static inline spatial_vectorf operator-(spatial_vectorf a, spatial_vectorf b) {
    spatial_vectorf r; r.w = a.w - b.w; r.v = a.v - b.v; return r;
}
static inline spatial_vectorf wp_add(spatial_vectorf a, spatial_vectorf b) { return a + b; }
static inline spatial_vectorf wp_sub(spatial_vectorf a, spatial_vectorf b) { return a - b; }
static inline spatial_vectorf wp_mul(spatial_vectorf a, float s) {
    spatial_vectorf r; r.w = a.w * s; r.v = a.v * s; return r;
}
static inline spatial_vectorf operator*(spatial_vectorf a, float s) { return wp_mul(a, s); }
// quat arithmetic (quat.h add/sub/mul scale)
static inline quatf operator+(quatf a, quatf b) { return make_quatf(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
static inline quatf operator-(quatf a, quatf b) { return make_quatf(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
static inline quatf operator*(quatf a, float s) { return make_quatf(a.x*s, a.y*s, a.z*s, a.w*s); }
static inline quatf wp_add(quatf a, quatf b) { return a + b; }
static inline quatf wp_sub(quatf a, quatf b) { return a - b; }
// quat*quat Hamilton product (quat.h mul(quat_t, quat_t) — used by transform_multiply)
static inline quatf wp_mul(quatf a, quatf b) {
    return make_quatf(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    );
}
static inline quatf wp_mul(quatf a, float s) { return a * s; }   // quat.h mul(quat_t, Type)
static inline quatf wp_mul(float s, quatf a) { return a * s; }   // quat.h operator*(Type, quat_t)
static inline transformf wp_mul(transformf a, transformf b) { return wp_transform_multiply(a, b); }  // spatial.h mul(transform, transform)
static inline transformf wp_mul(transformf a, float s) {  // spatial.h mul(transform, s)
    transformf r; r.p = a.p * s; r.q = a.q * s; return r;
}
// int4 shape-vector extract (warp native extract(shape_t, i) → s.dims[i])
static inline int wp_extract(int4 v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}
"""

_MAT_FACTORY_OPS = """\
// ---- identity / diag / determinant factories (KEP-0001 M3.2 P1 builtins;
// exact mat.h semantics — wp.identity/wp.diag/wp.determinant closure) ----
// identity (mat.h identity<T>(): zero-init + unit diagonal)
static inline mat22f wp_identity2() {
    mat22f m = make_mat22f(); m.c[0][0] = 1.0f; m.c[1][1] = 1.0f; return m;
}
static inline mat33f wp_identity3() {
    mat33f m = make_mat33f(); m.c[0][0] = 1.0f; m.c[1][1] = 1.0f; m.c[2][2] = 1.0f; return m;
}
static inline mat44f wp_identity4() {
    mat44f m = make_mat44f(); m.c[0][0] = 1.0f; m.c[1][1] = 1.0f; m.c[2][2] = 1.0f; m.c[3][3] = 1.0f; return m;
}
// diag (mat.h diag<T>(d): zero-fill, then the vector on the diagonal)
static inline mat22f wp_diag(vec2f d) {
    mat22f m = make_mat22f(); m.c[0][0] = d.x; m.c[1][1] = d.y; return m;
}
static inline mat33f wp_diag(vec3f d) {
    mat33f m = make_mat33f(); m.c[0][0] = d.x; m.c[1][1] = d.y; m.c[2][2] = d.z; return m;
}
static inline mat44f wp_diag(vec4f d) {
    mat44f m = make_mat44f(); m.c[0][0] = d.x; m.c[1][1] = d.y; m.c[2][2] = d.z; m.c[3][3] = d.w; return m;
}
// determinant (mat.h determinant — upstream arithmetic verbatim)
static inline float wp_determinant(mat22f m) {
    return m.c[0][0] * m.c[1][1] - m.c[1][0] * m.c[0][1];
}
static inline float wp_determinant(mat33f m) {
    return wp_dot(
        make_vec3f(m.c[0][0], m.c[0][1], m.c[0][2]),
        wp_cross(
            make_vec3f(m.c[1][0], m.c[1][1], m.c[1][2]),
            make_vec3f(m.c[2][0], m.c[2][1], m.c[2][2])));
}
// No 4x4: upstream's determinant(mat_t<4,4>) accumulates in double
// intermediates (mat.h:3543 "Adapted from USD"), which has no fp32-faithful
// MSL port (prelude carries no fp64). A 4x4 determinant call fails loud at
// Metal compile (no wp_determinant(mat44f) overload); the mujoco_warp
// closure uses 3x3 only.
"""

_EIG3_OPS = """\
// ---- eig3: symmetric 3x3 eigendecomposition (KEP-0001 M3.2; verbatim
// port of upstream warp/native/svd.h eig3 — quaternion-Jacobi eigenanalysis).
// fp32 only: the fp64 overload fails loud at MSL translation (prelude
// carries no fp64). JACOBI_ITERATIONS = 4 per _svd_config<float> (svd.h:39);
// recipSqrt uses the CPU branch 1/sqrt(x) (svd.h:53), not the CUDA rsqrt
// fast approximation — CPU warp is the golden arithmetic. Newton's mujoco
// glue (update_body_inertia_kernel via _eig_inertia) consumes the contract
// wp.eig3(A) -> (Q, d) with A = Q*diag(d)*Q^T: columns of Q are paired
// right-handed eigenvectors, d the eigenvalues; upstream explicitly does
// not promise an ordering, so consumers must not rely on one (Newton sorts
// descending itself). eig3 reads only A's lower triangle — symmetric input
// assumed (svd.h:736-741).
static inline mat33f wp_eig3_mult_ab(mat33f a, mat33f b) {  // svd.h multAB
    mat33f m;
    m.c[0][0] = a.c[0][0] * b.c[0][0] + a.c[0][1] * b.c[1][0] + a.c[0][2] * b.c[2][0];
    m.c[0][1] = a.c[0][0] * b.c[0][1] + a.c[0][1] * b.c[1][1] + a.c[0][2] * b.c[2][1];
    m.c[0][2] = a.c[0][0] * b.c[0][2] + a.c[0][1] * b.c[1][2] + a.c[0][2] * b.c[2][2];
    m.c[1][0] = a.c[1][0] * b.c[0][0] + a.c[1][1] * b.c[1][0] + a.c[1][2] * b.c[2][0];
    m.c[1][1] = a.c[1][0] * b.c[0][1] + a.c[1][1] * b.c[1][1] + a.c[1][2] * b.c[2][1];
    m.c[1][2] = a.c[1][0] * b.c[0][2] + a.c[1][1] * b.c[1][2] + a.c[1][2] * b.c[2][2];
    m.c[2][0] = a.c[2][0] * b.c[0][0] + a.c[2][1] * b.c[1][0] + a.c[2][2] * b.c[2][0];
    m.c[2][1] = a.c[2][0] * b.c[0][1] + a.c[2][1] * b.c[1][1] + a.c[2][2] * b.c[2][1];
    m.c[2][2] = a.c[2][0] * b.c[0][2] + a.c[2][1] * b.c[1][2] + a.c[2][2] * b.c[2][2];
    return m;
}

static inline mat33f wp_eig3_mult_at_b(mat33f a, mat33f b) {  // svd.h multAtB
    mat33f m;
    m.c[0][0] = a.c[0][0] * b.c[0][0] + a.c[1][0] * b.c[1][0] + a.c[2][0] * b.c[2][0];
    m.c[0][1] = a.c[0][0] * b.c[0][1] + a.c[1][0] * b.c[1][1] + a.c[2][0] * b.c[2][1];
    m.c[0][2] = a.c[0][0] * b.c[0][2] + a.c[1][0] * b.c[1][2] + a.c[2][0] * b.c[2][2];
    m.c[1][0] = a.c[0][1] * b.c[0][0] + a.c[1][1] * b.c[1][0] + a.c[2][1] * b.c[2][0];
    m.c[1][1] = a.c[0][1] * b.c[0][1] + a.c[1][1] * b.c[1][1] + a.c[2][1] * b.c[2][1];
    m.c[1][2] = a.c[0][1] * b.c[0][2] + a.c[1][1] * b.c[1][2] + a.c[2][1] * b.c[2][2];
    m.c[2][0] = a.c[0][2] * b.c[0][0] + a.c[1][2] * b.c[1][0] + a.c[2][2] * b.c[2][0];
    m.c[2][1] = a.c[0][2] * b.c[0][1] + a.c[1][2] * b.c[1][1] + a.c[2][2] * b.c[2][1];
    m.c[2][2] = a.c[0][2] * b.c[0][2] + a.c[1][2] * b.c[1][2] + a.c[2][2] * b.c[2][2];
    return m;
}

static inline void wp_eig3_quat_to_mat3(const thread float* qV, thread mat33f& m) {  // svd.h quatToMat3
    float w = qV[3];
    float x = qV[0];
    float y = qV[1];
    float z = qV[2];

    float qxx = x * x;
    float qyy = y * y;
    float qzz = z * z;
    float qxz = x * z;
    float qxy = x * y;
    float qyz = y * z;
    float qwx = w * x;
    float qwy = w * y;
    float qwz = w * z;

    m.c[0][0] = 1.0f - 2.0f * (qyy + qzz);
    m.c[0][1] = 2.0f * (qxy - qwz);
    m.c[0][2] = 2.0f * (qxz + qwy);
    m.c[1][0] = 2.0f * (qxy + qwz);
    m.c[1][1] = 1.0f - 2.0f * (qxx + qzz);
    m.c[1][2] = 2.0f * (qyz - qwx);
    m.c[2][0] = 2.0f * (qxz - qwy);
    m.c[2][1] = 2.0f * (qyz + qwx);
    m.c[2][2] = 1.0f - 2.0f * (qxx + qyy);
}

static inline void wp_eig3_approx_givens(float a11, float a12, float a22, thread float& ch, thread float& sh) {
    // svd.h approximateGivensQuaternion: gamma = sqrt(8)+3, cstar = cos(pi/8),
    // sstar = sin(pi/8) — upstream constexpr fp64 literals, read here at the
    // same float values the C++ Type() casts produce
    const float _gamma = 5.82842712474619f;
    const float _cstar = 0.9238795325112867f;
    const float _sstar = 0.3826834323650898f;

    ch = 2.0f * (a11 - a22);
    sh = a12;
    bool b = _gamma * sh * sh < ch * ch;
    float w = 1.0f / sqrt(ch * ch + sh * sh);
    ch = b ? w * ch : _cstar;
    sh = b ? w * sh : _sstar;
}

static inline void wp_eig3_jacobi_conjugation(
    const int x, const int y, const int z,
    thread float& s11, thread float& s21, thread float& s22,
    thread float& s31, thread float& s32, thread float& s33,
    thread float* qV
) {  // svd.h jacobiConjugation
    float ch, sh;
    wp_eig3_approx_givens(s11, s21, s22, ch, sh);

    float scale = ch * ch + sh * sh;
    float a = (ch * ch - sh * sh) / scale;
    float b = (2.0f * sh * ch) / scale;

    // make temp copy of S
    float _s11 = s11;
    float _s21 = s21;
    float _s22 = s22;
    float _s31 = s31;
    float _s32 = s32;
    float _s33 = s33;

    // perform conjugation S = Q'*S*Q
    // Q already implicitly solved from a, b
    s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
    s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    s31 = a * _s31 + b * _s32;
    s32 = -b * _s31 + a * _s32;
    s33 = _s33;

    // update cumulative rotation qV
    float tmp[3] = {};
    tmp[0] = qV[0] * sh;
    tmp[1] = qV[1] * sh;
    tmp[2] = qV[2] * sh;
    sh *= qV[3];

    qV[0] *= ch;
    qV[1] *= ch;
    qV[2] *= ch;
    qV[3] *= ch;

    // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
    // for (p,q) = ((0,1),(1,2),(0,2))
    qV[z] += sh;
    qV[3] -= tmp[z];  // w
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    _s11 = s22;
    _s21 = s32;
    _s22 = s33;
    _s31 = s21;
    _s32 = s31;
    _s33 = s11;
    s11 = _s11;
    s21 = _s21;
    s22 = _s22;
    s31 = _s31;
    s32 = _s32;
    s33 = _s33;
}

static inline void wp_eig3_jacobi_eigenanalysis(
    thread float& s11,
    thread float& s21, thread float& s22,
    thread float& s31, thread float& s32, thread float& s33,
    // quaternion representation of V
    thread float* qV
) {  // svd.h jacobiEigenanlysis
    qV[3] = 1.0f;
    qV[0] = 0.0f;
    qV[1] = 0.0f;
    qV[2] = 0.0f;  // follow same indexing convention as GLM
    for (int i = 0; i < 4; ++i) {  // JACOBI_ITERATIONS (svd.h _svd_config<float>)
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
        //  asymptotic convergence
        wp_eig3_jacobi_conjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV);  // p,q = 0,1
        wp_eig3_jacobi_conjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV);  // p,q = 1,2
        wp_eig3_jacobi_conjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV);  // p,q = 0,2
    }
}

// svd.h eig3: Q = eigenbasis (columns), d = eigenvalues; Q built from the
// cumulative Jacobi rotation quaternion, d from diag(Q^T*A*Q)
static inline void wp_eig3(mat33f A, thread mat33f& Q, thread vec3f& d) {
    float qV[4] = {};
    float s11 = A.c[0][0];
    float s21 = A.c[1][0];
    float s22 = A.c[1][1];
    float s31 = A.c[2][0];
    float s32 = A.c[2][1];
    float s33 = A.c[2][2];

    wp_eig3_jacobi_eigenanalysis(s11, s21, s22, s31, s32, s33, qV);
    wp_eig3_quat_to_mat3(qV, Q);
    mat33f t = wp_eig3_mult_at_b(Q, A);
    mat33f u = wp_eig3_mult_ab(t, Q);
    d = make_vec3f(u.c[0][0], u.c[1][1], u.c[2][2]);
}
"""

_BVH_OPS = """
// ---- BVH (upstream warp/native/bvh.h layout; runtime/bvh.py builds it) ----
// BVHPackedNodeHalf packs child/payload in i:31 + leaf flag in b:1; clang
// lays the first declared bitfield member in the LOW bits, decoded here as
// u & 0x7fffffff / u >> 31.
struct wp_node_half_t { float x; float y; float z; uint u; };
struct wp_bvh_t {
    device wp_node_half_t* node_lowers;
    device wp_node_half_t* node_uppers;
    device int* node_parents;
    device int* node_counts;
    device int* primitive_indices;
    device int* root;
    device vec3f* item_lowers;
    device vec3f* item_uppers;
    device int* item_groups;
    int max_depth;
    int max_nodes;
    int num_nodes;
    int num_leaf_nodes;
    int num_items;
    int leaf_size;
    int constructor_type;
    int _pad0;
    uint64_t context;
};
static inline wp_bvh_t wp_bvh_get(uint64_t id) { return *(device wp_bvh_t*)(id); }
static inline int wp_bvh_get_leaf_group(const thread wp_bvh_t& bvh, int leaf) {
    if (!bvh.item_groups) return 0;
    return bvh.item_groups[bvh.primitive_indices[bvh.node_lowers[leaf].u & 0x7fffffffu]];
}
static inline int wp_bvh_lower_bound_group(const thread wp_bvh_t& bvh, int group) {
    int lo = 0;
    int hi = bvh.num_leaf_nodes;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (wp_bvh_get_leaf_group(bvh, mid) < group) lo = mid + 1; else hi = mid;
    }
    if (lo == bvh.num_leaf_nodes || wp_bvh_get_leaf_group(bvh, lo) != group) return -1;
    return lo;
}
static inline int wp_bvh_upper_bound_group(const thread wp_bvh_t& bvh, int group) {
    int lo = 0;
    int hi = bvh.num_leaf_nodes;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (wp_bvh_get_leaf_group(bvh, mid) <= group) lo = mid + 1; else hi = mid;
    }
    return lo;
}
static inline int wp_bvh_lca(device int* parent, int node_a, int node_b) {
    int da = 0, db = 0;
    for (int t = node_a; t != -1; t = parent[t]) ++da;
    for (int t = node_b; t != -1; t = parent[t]) ++db;
    if (da > db) {
        int diff = da - db;
        while (diff-- && node_a != -1) node_a = parent[node_a];
    } else if (db > da) {
        int diff = db - da;
        while (diff-- && node_b != -1) node_b = parent[node_b];
    }
    while (node_a != node_b) {
        if (node_a == -1 || node_b == -1) return -1;
        node_a = parent[node_a];
        node_b = parent[node_b];
    }
    return node_a;
}
static inline int wp_bvh_get_group_root(uint64_t id, int group_id) {
    wp_bvh_t bvh = wp_bvh_get(id);
    const int first = wp_bvh_lower_bound_group(bvh, group_id);
    if (first < 0) return -1;
    const int last = wp_bvh_upper_bound_group(bvh, group_id) - 1;
    return wp_bvh_lca(bvh.node_parents, first, last);
}
"""

# KEP-0001 M2.1: device query family, ported semantically verbatim from
# upstream warp/native/bvh.h (bvh_query_t + the bvh_query_next_impl
# traversal skeleton with the runtime-kind dispatch of bvh_query_next_dynamic —
# Newton only ever calls the kind-erased bvh_query_next) and warp/native/
# intersect.h (the four overlap tests). The tree itself is built host-side
# into the same layout (runtime/bvh.py), so query kernels dereference ids
# through wp_bvh_get. MSL has no <cfloat>: FLT_MAX as an exact literal.
_BVH_QUERY_OPS = """
constant float WP_FLT_MAX = 3.402823466e+38;

// ---- intersect.h (verbatim; vec3f struct → explicit components) ----------

static inline bool wp_intersect_ray_aabb(vec3f pos, vec3f rcp_dir, vec3f lower, vec3f upper, thread float& t) {
    float l1, l2, lmin, lmax;
    l1 = (lower.x - pos.x) * rcp_dir.x;
    l2 = (upper.x - pos.x) * rcp_dir.x;
    lmin = fmin(l1, l2);
    lmax = fmax(l1, l2);
    l1 = (lower.y - pos.y) * rcp_dir.y;
    l2 = (upper.y - pos.y) * rcp_dir.y;
    lmin = fmax(fmin(l1, l2), lmin);
    lmax = fmin(fmax(l1, l2), lmax);
    l1 = (lower.z - pos.z) * rcp_dir.z;
    l2 = (upper.z - pos.z) * rcp_dir.z;
    lmin = fmax(fmin(l1, l2), lmin);
    lmax = fmin(fmax(l1, l2), lmax);
    bool hit = (lmax >= 0.0f) && (lmax >= lmin);
    if (hit)
        t = lmin;
    return hit;
}

// robust variant: dir[i]==0 is a parallel slab (reject outside the slab,
// otherwise leave the interval unconstrained) so axis-aligned rays behave
static inline bool wp_intersect_ray_aabb_robust(
    vec3f pos, vec3f dir, vec3f rcp_dir, vec3f lower, vec3f upper, thread float& t
) {
    float lmin = -WP_FLT_MAX;
    float lmax = WP_FLT_MAX;
    if (dir.x == 0.0f) {
        if (pos.x < lower.x || pos.x > upper.x) return false;
    } else {
        float l1 = (lower.x - pos.x) * rcp_dir.x;
        float l2 = (upper.x - pos.x) * rcp_dir.x;
        lmin = fmax(fmin(l1, l2), lmin);
        lmax = fmin(fmax(l1, l2), lmax);
    }
    if (dir.y == 0.0f) {
        if (pos.y < lower.y || pos.y > upper.y) return false;
    } else {
        float l1 = (lower.y - pos.y) * rcp_dir.y;
        float l2 = (upper.y - pos.y) * rcp_dir.y;
        lmin = fmax(fmin(l1, l2), lmin);
        lmax = fmin(fmax(l1, l2), lmax);
    }
    if (dir.z == 0.0f) {
        if (pos.z < lower.z || pos.z > upper.z) return false;
    } else {
        float l1 = (lower.z - pos.z) * rcp_dir.z;
        float l2 = (upper.z - pos.z) * rcp_dir.z;
        lmin = fmax(fmin(l1, l2), lmin);
        lmax = fmin(fmax(l1, l2), lmax);
    }
    bool hit = (lmax >= 0.0f) && (lmax >= lmin);
    if (hit)
        t = lmin;
    return hit;
}

static inline bool wp_intersect_aabb_aabb(vec3f a_lower, vec3f a_upper, vec3f b_lower, vec3f b_upper) {
    if (a_lower.x > b_upper.x || a_lower.y > b_upper.y || a_lower.z > b_upper.z || a_upper.x < b_lower.x
        || a_upper.y < b_lower.y || a_upper.z < b_lower.z) {
        return false;
    }
    return true;
}

// squared distance from center to the AABB, per axis (Ericson, RTCD)
static inline bool wp_intersect_sphere_aabb(vec3f center, float radius_sq, vec3f lower, vec3f upper) {
    float dx = fmax(fmax(lower.x - center.x, center.x - upper.x), 0.0f);
    float dy = fmax(fmax(lower.y - center.y, center.y - upper.y), 0.0f);
    float dz = fmax(fmax(lower.z - center.z, center.z - upper.z), 0.0f);
    return dx * dx + dy * dy + dz * dz <= radius_sq;
}

// ---- bvh_query_t + traversal (bvh.h) -------------------------------------

// BvhQueryKind: AABB=0, RAY=1, CAPSULE=2, SPHERE=3 (upstream enum class)
struct wp_bvh_query_t {
    wp_bvh_t bvh;
    int stack[64];  // BVH_QUERY_STACK_SIZE
    int count;
    int prim_cur;
    int prim_end;
    vec3f input_lower;  // start for ray/capsule, center for sphere
    vec3f input_upper;  // rcp_dir for ray/capsule
    int bounds_nr;
    bool last_query_valid;
    int kind;
    float radius;
    float radius_sq;
};

static inline wp_bvh_query_t wp_bvh_query_init(uint64_t id, vec3f lower, vec3f upper, int root) {
    wp_bvh_query_t query;
    query.bvh = wp_bvh_get(id);
    query.bounds_nr = -1;
    query.count = 1;
    query.prim_cur = 0;
    query.prim_end = 0;
    query.stack[0] = root == -1 ? *(query.bvh.root) : root;
    query.input_lower = lower;
    query.input_upper = upper;
    query.last_query_valid = true;
    query.radius = 0.0f;
    query.radius_sq = 0.0f;
    return query;
}

static inline wp_bvh_query_t wp_bvh_query_aabb(uint64_t id, vec3f lower, vec3f upper, int root) {
    wp_bvh_query_t query = wp_bvh_query_init(id, lower, upper, root);
    query.kind = 0;
    return query;
}

static inline vec3f wp_rcp_vec3(vec3f v) {
    vec3f r;
    r.x = 1.0f / v.x;
    r.y = 1.0f / v.y;
    r.z = 1.0f / v.z;
    return r;
}

static inline wp_bvh_query_t wp_bvh_query_ray(uint64_t id, vec3f start, vec3f dir, int root) {
    wp_bvh_query_t query = wp_bvh_query_init(id, start, wp_rcp_vec3(dir), root);
    query.kind = 1;
    return query;
}

static inline wp_bvh_query_t wp_bvh_query_capsule(uint64_t id, vec3f start, vec3f dir, float radius, int root) {
    wp_bvh_query_t query = wp_bvh_query_init(id, start, wp_rcp_vec3(dir), root);
    query.kind = 2;
    query.radius = fmax(radius, 0.0f);
    query.radius_sq = query.radius * query.radius;
    return query;
}

static inline wp_bvh_query_t wp_bvh_query_sphere(uint64_t id, vec3f center, float radius, int root) {
    wp_bvh_query_t query = wp_bvh_query_init(id, center, center, root);
    query.kind = 3;
    query.radius = fmax(radius, 0.0f);
    query.radius_sq = query.radius * query.radius;
    return query;
}

// node/primitive overlap test — the runtime-kind switch of upstream's
// bvh_query_test<QUERY_KIND> (kind folded per instantiation there, dynamic here)
static inline bool wp_bvh_query_test(thread wp_bvh_query_t& query, vec3f node_lower, vec3f node_upper, float max_dist) {
    switch (query.kind) {
    case 3:  // sphere: exact sphere-AABB with pre-computed radius_sq
        return wp_intersect_sphere_aabb(query.input_lower, query.radius_sq, node_lower, node_upper);
    case 2:  // capsule: inflate by radius, robust slab test, closed endpoint
    {
        float t = WP_FLT_MAX;
        vec3f infl_l = node_lower;
        infl_l.x -= query.radius;
        infl_l.y -= query.radius;
        infl_l.z -= query.radius;
        vec3f infl_u = node_upper;
        infl_u.x += query.radius;
        infl_u.y += query.radius;
        infl_u.z += query.radius;
        bool hit = wp_intersect_ray_aabb_robust(query.input_lower, query.input_upper, query.input_upper, infl_l, infl_u, t);
        return hit && !(t > max_dist);
    }
    case 1:  // ray: plain slab test, half-open max_dist bound
    {
        float t = WP_FLT_MAX;
        bool hit = wp_intersect_ray_aabb(query.input_lower, query.input_upper, node_lower, node_upper, t);
        return hit && !(t >= max_dist);
    }
    default:  // aabb
        return wp_intersect_aabb_aabb(query.input_lower, query.input_upper, node_lower, node_upper);
    }
}

// bvh_query_next_impl traversal skeleton (shared across kinds upstream;
// the per-kind tests route through wp_bvh_query_test). Every iteration
// emits one primitive from the leaf being enumerated, or pops one node.
static inline bool wp_bvh_query_next(thread wp_bvh_query_t& query, thread int& index, float max_dist) {
    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = query.bvh.primitive_indices[query.prim_cur++];
            if (!wp_bvh_query_test(
                    query, query.bvh.item_lowers[primitive_index], query.bvh.item_uppers[primitive_index], max_dist
                )) {
                continue;
            }
            index = primitive_index;
            query.bounds_nr = primitive_index;
            return true;
        }
        if (!query.count)
            return false;
        const int node_index = query.stack[--query.count];
        wp_node_half_t node_lower = query.bvh.node_lowers[node_index];
        wp_node_half_t node_upper = query.bvh.node_uppers[node_index];
        vec3f nl, nu;
        nl.x = node_lower.x;
        nl.y = node_lower.y;
        nl.z = node_lower.z;
        nu.x = node_upper.x;
        nu.y = node_upper.y;
        nu.z = node_upper.z;
        if (!wp_bvh_query_test(query, nl, nu, max_dist)) {
            continue;
        }
        const int left_index = (int)(node_lower.u & 0x7fffffffu);
        const int right_index = (int)(node_upper.u & 0x7fffffffu);
        if ((node_lower.u >> 31u) & 1u) {
            // packed leaf: [start, end) primitive range
            const int start = left_index;
            const int end = right_index;
            if (end - start == 1) {
                int primitive_index = query.bvh.primitive_indices[start];
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }
            query.prim_cur = start;
            query.prim_end = end;
        } else {
            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
}
"""

# KEP-0001 M2.2: Mesh query family, ported semantically verbatim from
# upstream warp/native/mesh.h (warp 1.19): struct Mesh (three array_t
# members + per-triangle bounds + the EMBEDDED BVH — the same wp_bvh_t the
# M2.1 query family reads), mesh_query_point_no_sign (BVH best-fit
# traversal + closest_point_to_triangle), mesh_query_aabb/sphere + the
# templated traversal skeleton instantiated per kind, mesh_query_ray
# (packed-node closest hit) with the intersect.h woop ray-triangle test,
# and the mesh_eval_*/mesh_get_* accessors. Upstream variants that need
# sign resolution (solid-angle / parity / tracing machinery) are ported
# fail-loud: the translator rejects them before any MSL exists.
#
# The tree itself is built host-side over the per-triangle bounds
# (runtime/bvh.py HostBVH at wp.Mesh construction); runtime/host.py
# _MeshEntry is the byte-exact host mirror of wp_mesh (asserted there).
# Upstream functions take the mesh id and deref it (mesh_get); here the
# translator routes every wp::mesh_* call's id through the per-launch
# table (`_wp_mesh_table[id]` — see translate.py) so the functions below
# take the wp_mesh VALUE. mesh_get_bvh / mesh_get_group_root keep id
# semantics (an address into the bound table buffer).
_MESH_QUERY_OPS = """
// ---- struct Mesh (mesh.h verbatim layout; host.py _MeshEntry mirrors it) --

struct wp_mesh {
    wp_mesh_array points;             // 0    {device vec3f*, wp_arr_meta} = 48 bytes
    wp_mesh_array velocities;         // 48
    wp_mesh_array_i indices;          // 96
    device vec3f* lowers;             // 144  per-triangle bounds (also the BVH item arrays)
    device vec3f* uppers;             // 152
    device uchar* solid_angle_props;  // 160  winding-number family is fail-loud; always null here
    int num_points;                   // 168
    int num_tris;                     // 172
    wp_bvh_t bvh;                     // 176  embedded BVH (mesh_get_bvh = address of this member)
    uint64_t context;                 // 288
    float average_edge_length;        // 296
    int _pad0;                        // 300
};                                    // 304 bytes total

// array_t operator[] equivalents (array.h byte-stride indexing)
static inline vec3f wp_mesh_point(const thread wp_mesh& mesh, int i) {
    return *(device vec3f*)((device char*)mesh.points.data + (size_t)mesh.points.m.offset + (size_t)i * (size_t)mesh.points.m.strides[0]);
}
static inline vec3f wp_mesh_velocity(const thread wp_mesh& mesh, int i) {
    return *(device vec3f*)((device char*)mesh.velocities.data + (size_t)mesh.velocities.m.offset + (size_t)i * (size_t)mesh.velocities.m.strides[0]);
}
static inline int wp_mesh_tri(const thread wp_mesh& mesh, int i) {
    return *(device int*)((device char*)mesh.indices.data + (size_t)mesh.indices.m.offset + (size_t)i * (size_t)mesh.indices.m.strides[0]);
}

// ---- mesh.h distance_to_aabb_sq / furthest_distance_to_aabb_sq (verbatim) --

static inline float wp_distance_to_aabb_sq(vec3f p, vec3f lower, vec3f upper) {
    const float dx = fmin(upper.x, fmax(lower.x, p.x)) - p.x;
    const float dy = fmin(upper.y, fmax(lower.y, p.y)) - p.y;
    const float dz = fmin(upper.z, fmax(lower.z, p.z)) - p.z;
    return dx * dx + dy * dy + dz * dz;
}

// ---- intersect.h closest_point_to_triangle (verbatim) ----------------------

static inline vec2f wp_closest_point_to_triangle(vec3f a, vec3f b, vec3f c, vec3f p) {
    vec3f ab = b - a;
    vec3f ac = c - a;
    vec3f ap = p - a;

    float u, v, w;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        v = 0.0f;
        w = 0.0f;
        u = 1.0f - v - w;
        return make_vec2f(u, v);
    }

    vec3f bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        v = 1.0f;
        w = 0.0f;
        u = 1.0f - v - w;
        return make_vec2f(u, v);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        v = d1 / (d1 - d3);
        w = 0.0f;
        u = 1.0f - v - w;
        return make_vec2f(u, v);
    }

    vec3f cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        v = 0.0f;
        w = 1.0f;
        u = 1.0f - v - w;
        return make_vec2f(u, v);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        v = 0.0f;
        w = d2 / (d2 - d6);
        u = 1.0f - v - w;
        return make_vec2f(u, v);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        v = 1.0f - w;
        u = 1.0f - v - w;
        return make_vec2f(u, v);
    }

    float denom = 1.0f / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;
    u = 1.0f - v - w;
    return make_vec2f(u, v);
}

// ---- intersect.h woop ray-triangle (verbatim; vec subscripting explicit) ---

// longest_axis(vec.h): strict > scan, first max wins
static inline int wp_mesh_max_dim(vec3f a) {
    float lmax = fabs(a.x);
    int ret = 0;
    const float y = fabs(a.y);
    if (y > lmax) { ret = 1; lmax = y; }
    const float z = fabs(a.z);
    if (z > lmax) { ret = 2; lmax = z; }
    return ret;
}
static inline float wp_mesh_v3_at(vec3f v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline float wp_xorf(float x, int y) { return as_type<float>(as_type<int>(x) ^ y); }  // intersect.h xorf
static inline int wp_sign_mask(float x) { return as_type<int>(x) & 0x80000000; }            // intersect.h sign_mask
static inline float wp_diff_product(float a, float b, float c, float d) {
    float cd = c * d;
    float diff = fma(a, b, -cd);
    float error = fma(-c, d, cd);
    return diff + error;
}

// http://jcgt.org/published/0002/01/05/
static inline bool wp_intersect_ray_tri_woop(
    vec3f p, vec3f dir, vec3f a, vec3f b, vec3f c,
    thread float& t, thread float& u, thread float& v, thread float& sign, thread vec3f& normal
) {
    int kz = wp_mesh_max_dim(dir);
    int kx = kz + 1;
    if (kx == 3)
        kx = 0;
    int ky = kx + 1;
    if (ky == 3)
        ky = 0;

    if (wp_mesh_v3_at(dir, kz) < 0.0f) {
        int tmp = kx;
        kx = ky;
        ky = tmp;
    }

    float Sx = wp_mesh_v3_at(dir, kx) / wp_mesh_v3_at(dir, kz);
    float Sy = wp_mesh_v3_at(dir, ky) / wp_mesh_v3_at(dir, kz);
    float Sz = 1.0f / wp_mesh_v3_at(dir, kz);

    const vec3f A = a - p;
    const vec3f B = b - p;
    const vec3f C = c - p;

    const float Ax = wp_mesh_v3_at(A, kx) - Sx * wp_mesh_v3_at(A, kz);
    const float Ay = wp_mesh_v3_at(A, ky) - Sy * wp_mesh_v3_at(A, kz);
    const float Bx = wp_mesh_v3_at(B, kx) - Sx * wp_mesh_v3_at(B, kz);
    const float By = wp_mesh_v3_at(B, ky) - Sy * wp_mesh_v3_at(B, kz);
    const float Cx = wp_mesh_v3_at(C, kx) - Sx * wp_mesh_v3_at(C, kz);
    const float Cy = wp_mesh_v3_at(C, ky) - Sy * wp_mesh_v3_at(C, kz);

    float U = wp_diff_product(Cx, By, Cy, Bx);
    float V = wp_diff_product(Ax, Cy, Ay, Cx);
    float W = wp_diff_product(Bx, Ay, By, Ax);

    if (U == 0.0f || V == 0.0f || W == 0.0f) {
        // upstream recomputes the fallback in fp64; Metal has no fp64 — degrade
        // to plain fp32 products (same values wp_diff_product already yields).
        U = Cx * By - Cy * Bx;
        V = Ax * Cy - Ay * Cx;
        W = Bx * Ay - By * Ax;
    }

    if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) {
        return false;
    }

    float det = U + V + W;

    if (det == 0.0f) {
        return false;
    }

    const float Az = Sz * wp_mesh_v3_at(A, kz);
    const float Bz = Sz * wp_mesh_v3_at(B, kz);
    const float Cz = Sz * wp_mesh_v3_at(C, kz);
    const float T = U * Az + V * Bz + W * Cz;

    int det_sign = wp_sign_mask(det);
    if (wp_xorf(T, det_sign) < 0.0f) {
        return false;
    }

    const float rcpDet = 1.0f / det;
    u = U * rcpDet;
    v = V * rcpDet;
    t = T * rcpDet;
    sign = det;

    const vec3f ab = b - a;
    const vec3f ac = c - a;
    normal = cross(ab, ac);

    return true;
}

// mesh.h mesh_query_ray_intersect_aabb (verbatim fast/robust dispatch)
static inline bool wp_mesh_ray_intersect_aabb(
    vec3f start, vec3f dir, vec3f rcp_dir, bool fast_aabb, vec3f lower, vec3f upper, thread float& t
) {
    if (fast_aabb)
        return wp_intersect_ray_aabb(start, rcp_dir, lower, upper, t);
    else
        return wp_intersect_ray_aabb_robust(start, dir, rcp_dir, lower, upper, t);
}

static inline vec3f wp_mesh_query_ray_safe_dir(vec3f dir) {
    vec3f ray_dir = dir;
    if (ray_dir.x == 0.0f)
        ray_dir.x = 1.0e-20f;
    if (ray_dir.y == 0.0f)
        ray_dir.y = 1.0e-20f;
    if (ray_dir.z == 0.0f)
        ray_dir.z = 1.0e-20f;
    return ray_dir;
}

static inline bool wp_mesh_query_ray_use_fast_aabb(vec3f dir) {
    return dir.x != 0.0f && dir.y != 0.0f && dir.z != 0.0f;
}

// ---- mesh_query_point_no_sign (mesh.h verbatim traversal) ------------------

// returns true if there is a point (strictly) < distance max_dist
static inline bool wp_mesh_query_point_no_sign(
    wp_mesh mesh, vec3f point, float max_dist, thread int& face, thread float& u, thread float& v
) {
    int stack[64];  // BVH_QUERY_STACK_SIZE
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face = 0;
    float min_v = 0.0f;
    float min_w = 0.0f;

    while (count) {
        const int nodeIndex = stack[--count];

        wp_node_half_t lower = mesh.bvh.node_lowers[nodeIndex];
        wp_node_half_t upper = mesh.bvh.node_uppers[nodeIndex];
        vec3f nl, nu;
        nl.x = lower.x; nl.y = lower.y; nl.z = lower.z;
        nu.x = upper.x; nu.y = upper.y; nu.z = upper.z;

        // re-test distance
        float node_dist_sq = wp_distance_to_aabb_sq(point, nl, nu);
        if (node_dist_sq > min_dist_sq) {
            continue;
        }

        const int left_index = (int)(lower.u & 0x7fffffffu);
        const int right_index = (int)(upper.u & 0x7fffffffu);

        if ((lower.u >> 31u) & 1u) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = wp_mesh_tri(mesh, primitive_index * 3 + 0);
                int j = wp_mesh_tri(mesh, primitive_index * 3 + 1);
                int k = wp_mesh_tri(mesh, primitive_index * 3 + 2);

                vec3f p = wp_mesh_point(mesh, i);
                vec3f q = wp_mesh_point(mesh, j);
                vec3f r = wp_mesh_point(mesh, k);
                vec3f e0 = q - p;
                vec3f e1 = r - p;
                vec3f e2 = r - q;
                vec3f normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2f barycentric = wp_closest_point_to_triangle(p, q, r, point);
                float bu = barycentric.x;
                float bv = barycentric.y;
                float w = 1.f - bu - bv;
                vec3f c = bu * p + bv * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = bv;
                    min_w = w;
                    min_face = primitive_index;
                }
            }
        } else {
            wp_node_half_t left_lower = mesh.bvh.node_lowers[left_index];
            wp_node_half_t left_upper = mesh.bvh.node_uppers[left_index];

            wp_node_half_t right_lower = mesh.bvh.node_lowers[right_index];
            wp_node_half_t right_upper = mesh.bvh.node_uppers[right_index];

            vec3f ll, lu, rl, ru;
            ll.x = left_lower.x; ll.y = left_lower.y; ll.z = left_lower.z;
            lu.x = left_upper.x; lu.y = left_upper.y; lu.z = left_upper.z;
            rl.x = right_lower.x; rl.y = right_lower.y; rl.z = right_lower.z;
            ru.x = right_upper.x; ru.y = right_upper.y; ru.z = right_upper.z;

            float left_dist_sq = wp_distance_to_aabb_sq(point, ll, lu);
            float right_dist_sq = wp_distance_to_aabb_sq(point, rl, ru);

            vec2i child_indices;
            vec2f child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = make_vec2i(right_index, left_index);
                child_dist = make_vec2f(right_dist_sq, left_dist_sq);
            } else {
                child_indices = make_vec2i(left_index, right_index);
                child_dist = make_vec2f(left_dist_sq, right_dist_sq);
            }

            if (child_dist.x < min_dist_sq)
                stack[count++] = child_indices.x;

            if (child_dist.y < min_dist_sq)
                stack[count++] = child_indices.y;
        }
    }

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        return true;
    } else {
        return false;
    }
}

// struct-return wrapper (mesh.h mesh_query_point_no_sign 3-arg form)
struct wp_mesh_query_point_t {
    bool result;
    float sign;
    int face;
    float u;
    float v;
};

static inline wp_mesh_query_point_t wp_mesh_query_point_no_sign(wp_mesh mesh, vec3f point, float max_dist) {
    wp_mesh_query_point_t query;
    query.result = false;
    query.face = 0;
    query.u = 0.0f;
    query.v = 0.0f;
    query.sign = 0.0f;
    query.result = wp_mesh_query_point_no_sign(mesh, point, max_dist, query.face, query.u, query.v);
    return query;
}

// ---- mesh_query_ray (mesh.h verbatim closest-hit traversal) ----------------

// bvh.h packed-node helpers (both halves folded into one uint64)
static inline uint64_t wp_mesh_node_pack(wp_node_half_t lower, wp_node_half_t upper) {
    return ((uint64_t)(lower.u >> 31u) << 62) | ((uint64_t)(upper.u & 0x7fffffffu) << 31) | (uint64_t)(lower.u & 0x7fffffffu);
}
static inline bool wp_mesh_node_is_leaf(uint64_t node) { return (node >> 62) != 0; }
static inline int wp_mesh_node_lower_payload(uint64_t node) { return (int)(node & 0x7fffffffu); }
static inline int wp_mesh_node_upper_payload(uint64_t node) { return (int)((node >> 31) & 0x7fffffffu); }

static inline bool wp_mesh_query_ray(
    wp_mesh mesh, vec3f start, vec3f dir, float max_t,
    thread float& t, thread float& u, thread float& v, thread float& sign, thread vec3f& normal, thread int& face, int root
) {
    uint64_t stack[64];  // BVH_QUERY_STACK_SIZE
    int stack_size = 0;
    wp_node_half_t rl = mesh.bvh.node_lowers[(root == -1) ? *mesh.bvh.root : root];
    wp_node_half_t ru = mesh.bvh.node_uppers[(root == -1) ? *mesh.bvh.root : root];
    uint64_t cur_node = wp_mesh_node_pack(rl, ru);

    vec3f ray_dir = wp_mesh_query_ray_safe_dir(dir);
    vec3f rcp_dir = make_vec3f(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);
    const bool fast_aabb = wp_mesh_query_ray_use_fast_aabb(dir);

    float min_t = max_t;
    int min_face = 0;
    float min_u = 0.0f;
    float min_v = 0.0f;
    float min_sign = 1.0f;
    vec3f min_normal = make_vec3f();
    bool hit = false;

    while (true) {
        if (wp_mesh_node_is_leaf(cur_node)) {
            const int primitive_begin = wp_mesh_node_lower_payload(cur_node);
            const int primitive_end = wp_mesh_node_upper_payload(cur_node);
            // Leaf: test all primitives in the leaf.
            for (int pc = primitive_begin; pc < primitive_end; ++pc) {
                int primitive_index = mesh.bvh.primitive_indices[pc];
                int i = wp_mesh_tri(mesh, primitive_index * 3 + 0);
                int j = wp_mesh_tri(mesh, primitive_index * 3 + 1);
                int k = wp_mesh_tri(mesh, primitive_index * 3 + 2);

                vec3f p = wp_mesh_point(mesh, i);
                vec3f q = wp_mesh_point(mesh, j);
                vec3f r = wp_mesh_point(mesh, k);

                float tri_t, tri_u, tri_v, tri_sign;
                vec3f n;

                if (wp_intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, n)) {
                    if (tri_t < min_t && tri_t >= 0.0f) {
                        min_t = tri_t;
                        min_face = primitive_index;
                        min_u = tri_u;
                        min_v = tri_v;
                        min_sign = tri_sign;
                        min_normal = n;
                        hit = true;
                    }
                }
            }
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
            continue;
        }

        // Inner node: load both children so we can sort by entry distance.
        const int left_index = wp_mesh_node_lower_payload(cur_node);
        const int right_index = wp_mesh_node_upper_payload(cur_node);

        wp_node_half_t left_lower = mesh.bvh.node_lowers[left_index];
        wp_node_half_t left_upper = mesh.bvh.node_uppers[left_index];
        wp_node_half_t right_lower = mesh.bvh.node_lowers[right_index];
        wp_node_half_t right_upper = mesh.bvh.node_uppers[right_index];

        vec3f ll, lu, rl2, ru2;
        ll.x = left_lower.x; ll.y = left_lower.y; ll.z = left_lower.z;
        lu.x = left_upper.x; lu.y = left_upper.y; lu.z = left_upper.z;
        rl2.x = right_lower.x; rl2.y = right_lower.y; rl2.z = right_lower.z;
        ru2.x = right_upper.x; ru2.y = right_upper.y; ru2.z = right_upper.z;

        float t0 = WP_FLT_MAX;
        float t1 = WP_FLT_MAX;
        const bool h0 = wp_mesh_ray_intersect_aabb(start, dir, rcp_dir, fast_aabb, ll, lu, t0) && t0 < min_t;
        const bool h1 = wp_mesh_ray_intersect_aabb(start, dir, rcp_dir, fast_aabb, rl2, ru2, t1) && t1 < min_t;

        if (h0 && h1) {
            const bool near_left = (t0 < t1);
            if (stack_size >= 64)
                break;
            const uint64_t left_node = wp_mesh_node_pack(left_lower, left_upper);
            const uint64_t right_node = wp_mesh_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (h0) {
            cur_node = wp_mesh_node_pack(left_lower, left_upper);
        } else if (h1) {
            cur_node = wp_mesh_node_pack(right_lower, right_upper);
        } else {
            // Neither child reachable; pop.
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
        }
    }

    if (hit) {
        // write outputs
        u = min_u;
        v = min_v;
        sign = min_sign;
        t = min_t;
        normal = normalize(min_normal);
        face = min_face;

        return true;
    } else {
        return false;
    }
}

// struct-return wrapper (mesh.h mesh_query_ray 5-arg form)
struct wp_mesh_query_ray_t {
    float sign;
    int face;
    float t;
    float u;
    float v;
    vec3f normal;
    bool result;
};

static inline wp_mesh_query_ray_t wp_mesh_query_ray(wp_mesh mesh, vec3f start, vec3f dir, float max_t, int root) {
    wp_mesh_query_ray_t query;
    query.sign = 0.0f;
    query.face = 0;
    query.t = 0.0f;
    query.u = 0.0f;
    query.v = 0.0f;
    query.normal = make_vec3f();
    query.result = false;
    query.result = wp_mesh_query_ray(mesh, start, dir, max_t, query.t, query.u, query.v, query.sign, query.normal, query.face, root);
    return query;
}

// ---- mesh_query_aabb / sphere iterators (mesh.h verbatim skeleton) ---------

// MeshQueryKind: AABB=0, SPHERE=1 (upstream enum class; the concrete kinds
// select their iterator at warp codegen time — kind is read only on the
// kind-erased / iter-protocol paths)
struct wp_mesh_query_aabb_t {
    wp_mesh mesh;
    int stack[64];  // BVH_QUERY_STACK_SIZE
    int count;
    vec3f input_lower;
    vec3f input_upper;
    int prim_cur;
    int prim_end;
    int face;
    bool last_query_valid;
    int kind;
    float radius_sq;
};

// mesh.h mesh_query_impl<IsSphere> shared factory
static inline wp_mesh_query_aabb_t wp_mesh_query_impl(wp_mesh mesh, vec3f a, vec3f b, float radius, bool is_sphere) {
    wp_mesh_query_aabb_t query;
    query.mesh = mesh;
    query.count = 0;
    query.prim_cur = 0;
    query.prim_end = 0;
    query.last_query_valid = true;
    query.kind = is_sphere ? 1 : 0;
    const float r = fmax(radius, 0.0f);
    query.radius_sq = r * r;

    query.stack[0] = *mesh.bvh.root;
    query.count = 1;
    query.input_lower = a;
    query.input_upper = b;

    return query;
}

static inline wp_mesh_query_aabb_t wp_mesh_query_aabb(wp_mesh mesh, vec3f lower, vec3f upper) {
    return wp_mesh_query_impl(mesh, lower, upper, 0.0f, false);
}

static inline wp_mesh_query_aabb_t wp_mesh_query_sphere(wp_mesh mesh, vec3f center, float radius) {
    return wp_mesh_query_impl(mesh, center, center, radius, true);
}

// mesh.h mesh_query_prim_test: broad-phase sphere test against the cached
// triangle AABB, then an exact closest-point narrow phase (with the
// degenerate-face longest-edge fallback)
static inline bool wp_mesh_query_prim_test(const thread wp_mesh_query_aabb_t& query, const thread wp_mesh& mesh, int primitive_index) {
    if (!wp_intersect_sphere_aabb(query.input_lower, query.radius_sq, mesh.lowers[primitive_index], mesh.uppers[primitive_index]))
        return false;

    int i = wp_mesh_tri(mesh, primitive_index * 3 + 0);
    int j = wp_mesh_tri(mesh, primitive_index * 3 + 1);
    int k = wp_mesh_tri(mesh, primitive_index * 3 + 2);
    vec3f a = wp_mesh_point(mesh, i);
    vec3f b = wp_mesh_point(mesh, j);
    vec3f c = wp_mesh_point(mesh, k);

    const vec3f center = query.input_lower;
    vec3f cp;
    // Guard against degenerate (zero-area) faces to avoid NaN from closest_point_to_triangle.
    vec3f ab = b - a, ac = c - a;
    vec3f crn = cross(ab, ac);
    if (dot(crn, crn) == 0.0f) {
        // Degenerate: collapse to segment or point. Find the longest edge.
        vec3f bc = c - b;
        float lab2 = dot(ab, ab), lac2 = dot(ac, ac), lbc2 = dot(bc, bc);
        vec3f p0, q0;
        float len2;
        if (lab2 >= lac2 && lab2 >= lbc2) {
            p0 = a;
            q0 = b;
            len2 = lab2;
        } else if (lac2 >= lbc2) {
            p0 = a;
            q0 = c;
            len2 = lac2;
        } else {
            p0 = b;
            q0 = c;
            len2 = lbc2;
        }
        vec3f pq = q0 - p0;
        float t = (len2 > 0.0f) ? clamp(dot(center - p0, pq) / len2, 0.0f, 1.0f) : 0.0f;
        cp = p0 + t * pq;
    } else {
        vec2f uv = wp_closest_point_to_triangle(a, b, c, center);
        cp = a * uv.x + b * uv.y + c * (1.0f - uv.x - uv.y);
    }
    vec3f d = cp - center;
    return dot(d, d) <= query.radius_sq;
}

// mesh.h mesh_query_next_impl<AabbNodeTest, AabbPrimitiveTest, false>: the
// broad-phase AABB iterator (singleton leaves pass on the node test alone —
// the leaf AABB equals the cached primitive bounds)
static inline bool wp_mesh_query_aabb_next(thread wp_mesh_query_aabb_t& query, thread int& index) {
    wp_mesh mesh = query.mesh;

    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = mesh.bvh.primitive_indices[query.prim_cur++];

            const vec3f face_lower = mesh.lowers[primitive_index];
            const vec3f face_upper = mesh.uppers[primitive_index];
            if (!wp_intersect_aabb_aabb(query.input_lower, query.input_upper, face_lower, face_upper))
                continue;

            index = primitive_index;
            query.face = primitive_index;
            return true;
        }

        if (!query.count)
            return false;

        const int node_index = query.stack[--query.count];
        wp_node_half_t node_lower = mesh.bvh.node_lowers[node_index];
        wp_node_half_t node_upper = mesh.bvh.node_uppers[node_index];
        vec3f nl, nu;
        nl.x = node_lower.x; nl.y = node_lower.y; nl.z = node_lower.z;
        nu.x = node_upper.x; nu.y = node_upper.y; nu.z = node_upper.z;

        if (!wp_intersect_aabb_aabb(query.input_lower, query.input_upper, nl, nu))
            continue;

        const int left_index = (int)(node_lower.u & 0x7fffffffu);
        const int right_index = (int)(node_upper.u & 0x7fffffffu);

        if ((node_lower.u >> 31u) & 1u) {
            const int start = left_index;
            const int end = right_index;

            // Fast path when the leaf contains exactly one primitive (its AABB
            // is the leaf node's AABB, which just passed the node test)
            if (end - start == 1) {
                int primitive_index = mesh.bvh.primitive_indices[start];
                index = primitive_index;
                query.face = primitive_index;
                return true;
            }

            query.prim_cur = start;
            query.prim_end = end;
        } else {
            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
}

// mesh.h mesh_query_next_impl<SphereNodeTest, SpherePrimitiveTest, true>
static inline bool wp_mesh_query_sphere_next(thread wp_mesh_query_aabb_t& query, thread int& index) {
    wp_mesh mesh = query.mesh;

    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = mesh.bvh.primitive_indices[query.prim_cur++];

            if (!wp_mesh_query_prim_test(query, mesh, primitive_index))
                continue;

            index = primitive_index;
            query.face = primitive_index;
            return true;
        }

        if (!query.count)
            return false;

        const int node_index = query.stack[--query.count];
        wp_node_half_t node_lower = mesh.bvh.node_lowers[node_index];
        wp_node_half_t node_upper = mesh.bvh.node_uppers[node_index];
        vec3f nl, nu;
        nl.x = node_lower.x; nl.y = node_lower.y; nl.z = node_lower.z;
        nu.x = node_upper.x; nu.y = node_upper.y; nu.z = node_upper.z;

        if (!wp_intersect_sphere_aabb(query.input_lower, query.radius_sq, nl, nu))
            continue;

        const int left_index = (int)(node_lower.u & 0x7fffffffu);
        const int right_index = (int)(node_upper.u & 0x7fffffffu);

        if ((node_lower.u >> 31u) & 1u) {
            const int start = left_index;
            const int end = right_index;

            if (end - start == 1) {
                int primitive_index = mesh.bvh.primitive_indices[start];
                if (!wp_mesh_query_prim_test(query, mesh, primitive_index))
                    continue;
                index = primitive_index;
                query.face = primitive_index;
                return true;
            }

            query.prim_cur = start;
            query.prim_end = end;
        } else {
            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }
}

// Kind-erased iterator (mesh.h mesh_query_next_dynamic)
static inline bool wp_mesh_query_next_dynamic(thread wp_mesh_query_aabb_t& query, thread int& index) {
    if (query.kind == 1)
        return wp_mesh_query_sphere_next(query, index);
    return wp_mesh_query_aabb_next(query, index);
}

// iter protocol overloads (mesh.h iter_next/iter_cmp)
static inline int wp_iter_next(thread wp_mesh_query_aabb_t& query) { return query.face; }
static inline bool wp_iter_cmp(thread wp_mesh_query_aabb_t& query) {
    return wp_mesh_query_next_dynamic(query, query.face);
}

// ---- mesh_eval_* / mesh_get_* (mesh.h verbatim) ----------------------------

static inline vec3f wp_mesh_eval_position(wp_mesh mesh, int tri, float u, float v) {
    if (!mesh.points.data)
        return make_vec3f();

    int i = wp_mesh_tri(mesh, tri * 3 + 0);
    int j = wp_mesh_tri(mesh, tri * 3 + 1);
    int k = wp_mesh_tri(mesh, tri * 3 + 2);

    vec3f p = wp_mesh_point(mesh, i);
    vec3f q = wp_mesh_point(mesh, j);
    vec3f r = wp_mesh_point(mesh, k);

    return p * u + q * v + r * (1.0f - u - v);
}

static inline vec3f wp_mesh_eval_velocity(wp_mesh mesh, int tri, float u, float v) {
    if (!mesh.velocities.data)
        return make_vec3f();

    int i = wp_mesh_tri(mesh, tri * 3 + 0);
    int j = wp_mesh_tri(mesh, tri * 3 + 1);
    int k = wp_mesh_tri(mesh, tri * 3 + 2);

    vec3f vp = wp_mesh_velocity(mesh, i);
    vec3f vq = wp_mesh_velocity(mesh, j);
    vec3f vr = wp_mesh_velocity(mesh, k);

    return vp * u + vq * v + vr * (1.0f - u - v);
}

static inline vec3f wp_mesh_eval_face_normal(wp_mesh mesh, int tri) {
    if (!mesh.points.data)
        return make_vec3f();

    int i = wp_mesh_tri(mesh, tri * 3 + 0);
    int j = wp_mesh_tri(mesh, tri * 3 + 1);
    int k = wp_mesh_tri(mesh, tri * 3 + 2);

    vec3f p = wp_mesh_point(mesh, i);
    vec3f q = wp_mesh_point(mesh, j);
    vec3f r = wp_mesh_point(mesh, k);

    return normalize(cross(q - p, r - p));
}

static inline vec3f wp_mesh_get_point(wp_mesh mesh, int index) {
    if (!mesh.points.data)
        return make_vec3f();

    int i = wp_mesh_tri(mesh, index);
    return wp_mesh_point(mesh, i);
}

static inline vec3f wp_mesh_get_velocity(wp_mesh mesh, int index) {
    if (!mesh.velocities.data)
        return make_vec3f();

    int i = wp_mesh_tri(mesh, index);
    return wp_mesh_velocity(mesh, i);
}

static inline int wp_mesh_get_index(wp_mesh mesh, int face_vertex_index) {
    if (!mesh.indices.data)
        return -1;

    return wp_mesh_tri(mesh, face_vertex_index);
}
"""

# ---- WP_TILE_SYNC (upstream warp/native/tile.h:66) ----
# mujoco_warp embeds @wp.func_native snippets calling WP_TILE_SYNC() verbatim
# (smooth.py:3591, solver.py:903); M1.4 inlines those snippets into this
# compile unit, so the macro must exist here. Upstream defines it as
# __syncthreads on CUDA; the MSL equivalent is the threadgroup barrier.
# Object-like (matching upstream) so both WP_TILE_SYNC(); and a bare
# WP_TILE_SYNC; statement expand; emitted before every other section so all
# kernels share it (prelude discipline: one shared compile unit, so the
# definition must be inert for kernels that never call it — an unused
# preprocessor define is).
_TILE_SYNC = """\
// ---- WP_TILE_SYNC (upstream warp/native/tile.h:66) ----
#define WP_TILE_SYNC threadgroup_barrier(mem_flags::mem_threadgroup)
"""

_PRELUDE = (
    "#include <metal_stdlib>\n"
    "#include <metal_atomic>\n"
    "using namespace metal;\n"
    "\n"
    + _TILE_SYNC
    + _STRUCTS
    + "\n"
    + _VEC_OPS
    + "\n"
    + _QUAT_ASSIGN
    + "\n"
    + _MAT_OPS
    + "\n"
    + _QUAT_TRANSFORM_OPS
    + "\n"
    + _SCALAR_OPS
    + "\n"
    + _MAT_FACTORY_OPS
    + "\n"
    + _EIG3_OPS
    + "\n"
    + _BVH_OPS
    + "\n"
    + _BVH_QUERY_OPS
    + "\n"
    + _MESH_QUERY_OPS
)


PRELUDE = _PRELUDE


def prelude_source() -> str:
    return _PRELUDE
