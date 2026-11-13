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
struct quatf { float x, y, z, w; };  // imaginary xyz, real w (quat.h)
struct mat22f { float c[2][2]; };    // c[row][col], row-major (mat.h data)
struct mat33f { float c[3][3]; };
struct mat44f { float c[4][4]; };
struct transformf { vec3f p; quatf q; };  // (spatial.h transform_t)
struct spatial_vectorf { vec3f w, v; };   // (spatial.h spatial_vector_t)

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

// ---- array binding: host-side metadata + device data pointer ----
// Mirrors warp native array_t (data/offset/shape/strides); strides are in
// BYTES (array.h byte_offset), contiguous row-major only.
struct wp_arr_meta {
    int offset;
    int ndim;
    int shape[4];
    int strides[4];
};

// multi-dim tid decomposition (builtin.h launch_coord: last dim fastest)
static inline void wp_tid2(thread int& i, thread int& j, int idx, constant wp_arr_meta& m) {
    j = idx % m.shape[1];
    i = idx / m.shape[1];
}
static inline void wp_tid3(thread int& i, thread int& j, thread int& k, int idx, constant wp_arr_meta& m) {
    k = idx % m.shape[2];
    idx /= m.shape[2];
    j = idx % m.shape[1];
    i = idx / m.shape[1];
}
static inline void wp_tid4(thread int& i, thread int& j, thread int& k, thread int& l, int idx, constant wp_arr_meta& m) {
    l = idx % m.shape[3];
    idx /= m.shape[3];
    k = idx % m.shape[2];
    idx /= m.shape[2];
    j = idx % m.shape[1];
    i = idx / m.shape[1];
}

// ---- array addressing (warp native array.h index(): strides in BYTES,
// byte_offset arithmetic through char*, negative indices wrap Python-style) ----
template <typename T>
static inline device T* wp_address(device T* data, constant wp_arr_meta& m, int i) {
    if (i < 0) i += m.shape[0];
    return (device T*)((device char*)data + (size_t)m.offset + (size_t)(i * m.strides[0]));
}
template <typename T>
static inline device T* wp_address(device T* data, constant wp_arr_meta& m, int i, int j) {
    if (i < 0) i += m.shape[0];
    if (j < 0) j += m.shape[1];
    return (device T*)((device char*)data + (size_t)m.offset
        + (size_t)(i * m.strides[0]) + (size_t)(j * m.strides[1]));
}
template <typename T>
static inline device T* wp_address(device T* data, constant wp_arr_meta& m, int i, int j, int k) {
    if (i < 0) i += m.shape[0];
    if (j < 0) j += m.shape[1];
    if (k < 0) k += m.shape[2];
    return (device T*)((device char*)data + (size_t)m.offset
        + (size_t)(i * m.strides[0]) + (size_t)(j * m.strides[1]) + (size_t)(k * m.strides[2]));
}
template <typename T>
static inline device T* wp_address(device T* data, constant wp_arr_meta& m, int i, int j, int k, int l) {
    if (i < 0) i += m.shape[0];
    if (j < 0) j += m.shape[1];
    if (k < 0) k += m.shape[2];
    if (l < 0) l += m.shape[3];
    return (device T*)((device char*)data + (size_t)m.offset
        + (size_t)(i * m.strides[0]) + (size_t)(j * m.strides[1])
        + (size_t)(k * m.strides[2]) + (size_t)(l * m.strides[3]));
}
static inline int wp_extract_shape(constant wp_arr_meta& m, int i) { return m.shape[i]; }
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

static inline vec2f make_vec2f(float x, float y) { vec2f v; v.x = x; v.y = y; return v; }
static inline vec3f make_vec3f(float x, float y, float z) { vec3f v; v.x = x; v.y = y; v.z = z; return v; }
static inline vec4f make_vec4f(float x, float y, float z, float w) { vec4f v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline quatf make_quatf(float x, float y, float z, float w) { quatf v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
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
// extract-style component read with runtime index (vec.h extract + component_ref)
static inline float wp_extract(vec2f v, int i) { return i == 0 ? v.x : v.y; }
static inline float wp_extract(vec3f v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline float wp_extract(vec4f v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
static inline float wp_extract(quatf q, int i) { return i == 0 ? q.x : (i == 1 ? q.y : (i == 2 ? q.z : q.w)); }
static inline float wp_extract(transformf t, int i) { return i < 3 ? wp_extract(t.p, i) : wp_extract(t.q, i - 3); }
// rows of matrices (mat.h extract(mat, row))
static inline vec2f wp_extract(mat22f m, int r) { return make_vec2f(m.c[r][0], m.c[r][1]); }
static inline vec3f wp_extract(mat33f m, int r) { return make_vec3f(m.c[r][0], m.c[r][1], m.c[r][2]); }
static inline vec4f wp_extract(mat44f m, int r) { return make_vec4f(m.c[r][0], m.c[r][1], m.c[r][2], m.c[r][3]); }
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
static inline vec2f wp_abs(vec2f a) { return make_vec2f(abs(a.x), abs(a.y)); }
static inline vec3f wp_abs(vec3f a) { return make_vec3f(abs(a.x), abs(a.y), abs(a.z)); }
static inline vec4f wp_abs(vec4f a) { return make_vec4f(abs(a.x), abs(a.y), abs(a.z), abs(a.w)); }
// scalar clamp (builtin.h: min(max(a, x), b))
static inline float wp_clamp(float x, float a, float b) { return min(max(a, x), b); }
static inline int wp_clamp(int x, int a, int b) { return min(max(a, x), b); }
static inline vec3f wp_clamp(vec3f x, float a, float b) { return wp_min(wp_max(make_vec3f(a), x), make_vec3f(b)); }
// where/select (builtins.py where overloads)
static inline vec2f wp_where(bool c, vec2f a, vec2f b) { return c ? a : b; }
static inline vec3f wp_where(bool c, vec3f a, vec3f b) { return c ? a : b; }
static inline vec4f wp_where(bool c, vec4f a, vec4f b) { return c ? a : b; }
static inline quatf wp_where(bool c, quatf a, quatf b) { return c ? a : b; }
static inline mat33f wp_where(bool c, mat33f a, mat33f b) { return c ? a : b; }
static inline transformf wp_where(bool c, transformf a, transformf b) { return c ? a : b; }
static inline spatial_vectorf wp_where(bool c, spatial_vectorf a, spatial_vectorf b) { return c ? a : b; }
static inline float wp_where(bool c, float a, float b) { return c ? a : b; }
static inline int wp_where(bool c, int a, int b) { return c ? a : b; }
static inline uint wp_where(bool c, uint a, uint b) { return c ? a : b; }
static inline uint64_t wp_where(bool c, uint64_t a, uint64_t b) { return c ? a : b; }
static inline int64_t wp_where(bool c, int64_t a, int64_t b) { return c ? a : b; }
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
static inline float wp_mod(float a, float b) {                      // builtin.h float mod: fmod + Python-sign fix
    float r = fmod(a, b);
    if (r != 0.0f && (r < 0.0f) != (b < 0.0f)) r += b;
    return r;
}
static inline float wp_asin(float x) { return asin(x); }
static inline float wp_acos(float x) { return acos(x); }
static inline float wp_sin(float x) { return precise::sin(x); }     // align CPU numeric path
static inline float wp_cos(float x) { return precise::cos(x); }
static inline float wp_tan(float x) { return precise::tan(x); }
static inline float wp_sqrt(float x) { return sqrt(x); }
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

// ---- wp_*_at: array-indexed atomic entry points (translator emits these) ----
// exact-match overloads only: calling on an unsupported element type is a
// compile error (fail loud), never a silent non-atomic path.
static inline void wp_atomic_add_at(device float* d, constant wp_arr_meta& m, int i, float v) {
    wp_atomic_add((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_add_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    wp_atomic_add((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_add_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    wp_atomic_add((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_add_ret_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    return wp_atomic_add_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_add_ret_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    return wp_atomic_add_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_sub_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    wp_atomic_sub((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_sub_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    wp_atomic_sub((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_min_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    wp_atomic_min((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_min_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    wp_atomic_min((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_min_at(device float* d, constant wp_arr_meta& m, int i, float v) {
    wp_atomic_min((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_max_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    wp_atomic_max((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_max_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    wp_atomic_max((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline void wp_atomic_max_at(device float* d, constant wp_arr_meta& m, int i, float v) {
    wp_atomic_max((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
// 2D variants (array2d atomics)
static inline void wp_atomic_add_at(device float* d, constant wp_arr_meta& m, int i, int j, float v) {
    wp_atomic_add((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline void wp_atomic_add_at(device int* d, constant wp_arr_meta& m, int i, int j, int v) {
    wp_atomic_add((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline void wp_atomic_min_at(device float* d, constant wp_arr_meta& m, int i, int j, float v) {
    wp_atomic_min((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
static inline void wp_atomic_max_at(device float* d, constant wp_arr_meta& m, int i, int j, float v) {
    wp_atomic_max((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0] + j * m.strides[1])), v);
}
// _ret_at: captured-result atomics (upstream returns the old value)
static inline float wp_atomic_add_ret_at(device float* d, constant wp_arr_meta& m, int i, float v) {
    return wp_atomic_add_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_sub_ret_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    return wp_atomic_sub_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_sub_ret_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    return wp_atomic_sub_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_min_ret_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    return wp_atomic_min_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_min_ret_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    return wp_atomic_min_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline float wp_atomic_min_ret_at(device float* d, constant wp_arr_meta& m, int i, float v) {
    return wp_atomic_min_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline int wp_atomic_max_ret_at(device int* d, constant wp_arr_meta& m, int i, int v) {
    return wp_atomic_max_ret((device atomic_int*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline uint wp_atomic_max_ret_at(device uint* d, constant wp_arr_meta& m, int i, uint v) {
    return wp_atomic_max_ret((device atomic_uint*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}
static inline float wp_atomic_max_ret_at(device float* d, constant wp_arr_meta& m, int i, float v) {
    return wp_atomic_max_ret((device atomic_float*)((device char*)d + (size_t)(m.offset + i * m.strides[0])), v);
}

// ---- bit ops (builtin.h bit_and/bit_or/bit_xor/lshift/rshift) ----
static inline int wp_bit_and(int a, int b) { return a & b; }
static inline uint wp_bit_and(uint a, uint b) { return a & b; }
static inline uint64_t wp_bit_and(uint64_t a, uint64_t b) { return a & b; }
static inline int wp_bit_or(int a, int b) { return a | b; }
static inline uint wp_bit_or(uint a, uint b) { return a | b; }
static inline uint64_t wp_bit_or(uint64_t a, uint64_t b) { return a | b; }
static inline int wp_bit_xor(int a, int b) { return a ^ b; }
static inline uint wp_bit_xor(uint a, uint b) { return a ^ b; }
static inline int wp_lshift(int a, int b) { return a << b; }
static inline uint wp_lshift(uint a, int b) { return a << b; }
static inline int wp_rshift(int a, int b) { return a >> b; }
static inline uint wp_rshift(uint a, int b) { return a >> b; }
static inline uint64_t wp_rshift(uint64_t a, uint64_t b) { return a >> b; }

// ---- scalar sign / neg / div (builtin.h semantics) ----
static inline float wp_sign(float x) { return x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f); }
static inline int wp_sign(int x) { return (x > 0) - (x < 0); }
static inline float wp_neg(float x) { return -x; }
static inline int wp_neg(int x) { return -x; }
static inline vec2f wp_neg(vec2f a) { return -a; }
static inline vec3f wp_neg(vec3f a) { return -a; }
static inline vec4f wp_neg(vec4f a) { return -a; }
static inline float wp_div(float a, float b) { return a / b; }
static inline int wp_div(int a, int b) { return a / b; }
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

// ---- vector bridges for the wp_-prefixed call table ----
static inline float wp_dot(vec2f a, vec2f b) { return dot(a, b); }
static inline float wp_dot(vec3f a, vec3f b) { return dot(a, b); }
static inline float wp_dot(vec4f a, vec4f b) { return dot(a, b); }
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
static inline transformf wp_mul(transformf a, float s) {  // spatial.h mul(transform, s)
    transformf r; r.p = a.p * s; r.q = a.q * s; return r;
}
// int4 shape-vector extract (warp native extract(shape_t, i) → s.dims[i])
static inline int wp_extract(int4 v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}
"""

_PRELUDE = (
    "#include <metal_stdlib>\n"
    "#include <metal_atomic>\n"
    "using namespace metal;\n"
    "\n"
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
)

PRELUDE = _PRELUDE


def prelude_source() -> str:
    return _PRELUDE
