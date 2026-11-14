# Copyright 2026 Jaylin Lee
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0.
"""Host-side BVH build/refit for warp-metal (upstream warp/native/bvh.cpp port).

Warp's BVH is a pointer-linked tree over packed node halves; device kernels
(``wp.bvh_get_group_root`` here, the query family in M2.1) read the exact
upstream ``BVH`` struct layout through the tree id. This module builds that
layout on the host, over the caller's shared-storage bound arrays (unified
memory: zero-copy) — the node/parent/context buffers are Metal buffers so
device code dereferences the same bytes.

Ported semantically verbatim from bvh.cpp ``TopDownBVHBuilder``: SAH and
median partitions, grouped builds (``build_with_groups`` incl. group-leaf
expansion), the leaf-front reorder (``reorder_top_down_bvh``), and recursive
refit. Node payloads pack ``i:31 | b:1`` like upstream's ``BVHPackedNodeHalf``
bitfield — decoded on device as ``u & 0x7fffffff`` / ``u >> 31`` (clang
little-endian bitfield order: the first declared member occupies the low
bits).

Query kernels (``bvh_query_aabb/ray/sphere``) are Phase 2 / M2.1 and are NOT
part of this slice; ``bvh_get_group_root`` (Newton's model finalize needs it)
is.
"""

from __future__ import annotations

import ctypes

import numpy as np

SAH = 0
MEDIAN = 1
LBVH = 2
CUBQL = -1

_SAH_NUM_BUCKETS = 12
_BVH_QUERY_STACK_SIZE = 64

# BVHPackedNodeHalf: float x, y, z + unsigned i:31, b:1 → 16 bytes
_NODE_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("u", "<u4")])

# device wp_bvh_t (see msl/prelude.py): 9 pointers (bytes 0..71), 7 ints
# (72..99), 4 pad bytes, 1 pointer (104..111) → 112 bytes total
_CTX_BYTES = 112
_CTX_U64 = 14  # u64 words: 0..8 pointers, 9..12 unused, 13 context


def _make_payload(child: int, leaf: bool) -> int:
    return (int(child) & 0x7FFFFFFF) | ((1 << 31) if leaf else 0)


def _payload_i(node) -> int:
    return int(node["u"]) & 0x7FFFFFFF


def _is_leaf(node) -> bool:
    return (int(node["u"]) >> 31) & 1 == 1


class HostBVH:
    """A BVH in the exact upstream layout, resident in Metal buffers.

    ``id`` is the GPU address of the context buffer (device ``bvh_get(id)``
    dereferences it); the node arrays live in their own buffers referenced
    from the context. The caller's lowers/uppers/groups WarpArrays are NOT
    copied — the context points into their shared storage, so a refit after
    Newton rewrites the bounds sees the new values.
    """

    def __init__(self, lowers_arr, uppers_arr, groups_arr, constructor: int, leaf_size: int, mm):
        self._mm = mm
        self.lowers_arr = lowers_arr
        self.uppers_arr = uppers_arr
        self.groups_arr = groups_arr
        self.leaf_size = int(leaf_size)
        self.constructor_type = int(constructor)
        self.max_depth = 0
        self.max_nodes = 0
        self.num_nodes = 0
        self.num_leaf_nodes = 0
        self.num_items = 0
        self.root = 0
        self.build()

    # -- numpy views over the caller's (shared-storage) item arrays ---------

    @property
    def item_lowers(self) -> np.ndarray:
        return self.lowers_arr.numpy().reshape(-1, 3)

    @property
    def item_uppers(self) -> np.ndarray:
        return self.uppers_arr.numpy().reshape(-1, 3)

    @property
    def item_groups(self) -> np.ndarray | None:
        if self.groups_arr is None:
            return None
        g = self.groups_arr.numpy()
        return g if len(g) else None

    # -- build (TopDownBVHBuilder port) --------------------------------------

    def build(self) -> None:
        n = len(self.item_lowers)
        self.max_depth = 0
        self.max_nodes = max(2 * n - 1, 0)
        self.node_lowers = np.zeros(max(self.max_nodes, 1), dtype=_NODE_DTYPE)
        self.node_uppers = np.zeros(max(self.max_nodes, 1), dtype=_NODE_DTYPE)
        self.node_parents = np.full(max(self.max_nodes, 1), -1, dtype=np.int32)
        self.num_nodes = 0
        self.num_leaf_nodes = 0
        self.num_items = n
        self.primitive_indices = np.arange(n, dtype=np.int32)
        self.root = 0
        if n:
            if self.leaf_size < 1:
                raise ValueError(f"leaf_size must be >= 1, got {self.leaf_size}")
            if self.constructor_type not in (SAH, MEDIAN):
                raise ValueError(
                    f"Unrecognized constructor type {self.constructor_type} for the CPU BVH path "
                    "(SAH/MEDIAN; LBVH falls back to SAH per upstream CPU-tree semantics)"
                )
            if self.item_groups is not None:
                self._build_with_groups()
            else:
                self._build_recursive(0, n, 0, -1)
        self._upload()

    def _calc_bounds(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.primitive_indices[start:end]
        lo = self.item_lowers[idx]
        up = np.maximum(self.item_lowers[idx], self.item_uppers[idx])
        return lo.min(axis=0), up.max(axis=0)

    def _partition_sah(self, start: int, end: int, range_lo, range_up) -> tuple[float, int]:
        """Return (SAH split coordinate, widest centroid-bounds axis)."""
        idx = self.primitive_indices[start:end]
        centers = 0.5 * (self.item_lowers[idx] + self.item_uppers[idx])
        c_lo, c_up = centers.min(axis=0), centers.max(axis=0)
        edges = c_up - c_lo
        axis = int(np.argmax(edges))

        range_start = float(c_lo[axis])
        range_end = float(c_up[axis])
        if range_end <= range_start:
            return range_start, axis

        bucket_idx = np.clip(
            (_SAH_NUM_BUCKETS * (centers[:, axis] - range_start) / (range_end - range_start)).astype(np.int32),
            0,
            _SAH_NUM_BUCKETS - 1,
        )
        buckets_lo = np.full((_SAH_NUM_BUCKETS, 3), np.inf, dtype=np.float32)
        buckets_up = np.full((_SAH_NUM_BUCKETS, 3), -np.inf, dtype=np.float32)
        bucket_counts = np.zeros(_SAH_NUM_BUCKETS, dtype=np.int64)
        for b in range(_SAH_NUM_BUCKETS):
            sel = bucket_idx == b
            bucket_counts[b] = int(sel.sum())
            if bucket_counts[b]:
                buckets_lo[b] = self.item_lowers[idx[sel]].min(axis=0)
                buckets_up[b] = np.maximum(self.item_lowers[idx[sel]], self.item_uppers[idx[sel]]).max(axis=0)

        left_areas = np.zeros(_SAH_NUM_BUCKETS - 1, dtype=np.float32)
        right_areas = np.zeros(_SAH_NUM_BUCKETS - 1, dtype=np.float32)
        counts_l = np.zeros(_SAH_NUM_BUCKETS - 1, dtype=np.int64)
        counts_r = np.zeros(_SAH_NUM_BUCKETS - 1, dtype=np.int64)
        acc_lo = np.full(3, np.inf, dtype=np.float32)
        acc_up = np.full(3, -np.inf, dtype=np.float32)
        cl = 0
        for i in range(_SAH_NUM_BUCKETS - 1):
            acc_lo = np.minimum(acc_lo, buckets_lo[i])
            acc_up = np.maximum(acc_up, buckets_up[i])
            left_areas[i] = float(np.prod(np.maximum(acc_up - acc_lo, np.float32(0.0))))
            cl += int(bucket_counts[i])
            counts_l[i] = cl
        acc_lo = np.full(3, np.inf, dtype=np.float32)
        acc_up = np.full(3, -np.inf, dtype=np.float32)
        cr = 0
        for i in range(_SAH_NUM_BUCKETS - 1):
            j = _SAH_NUM_BUCKETS - i - 1
            acc_lo = np.minimum(acc_lo, buckets_lo[j])
            acc_up = np.maximum(acc_up, buckets_up[j])
            right_areas[_SAH_NUM_BUCKETS - i - 2] = float(np.prod(np.maximum(acc_up - acc_lo, np.float32(0.0))))
            cr += int(bucket_counts[j])
            counts_r[_SAH_NUM_BUCKETS - i - 2] = cr

        range_edges = (range_up - range_lo).astype(np.float32)
        range_area = float(np.prod(np.maximum(range_edges, np.float32(0.0))))
        inv_total = 1.0 / range_area if range_area > 0 else 0.0
        costs = left_areas * inv_total * counts_l + right_areas * inv_total * counts_r
        min_split = int(np.argmin(costs))
        return range_start + (min_split + 1) * (range_end - range_start) / _SAH_NUM_BUCKETS, axis

    def _partition_by(self, start: int, end: int, axis: int, threshold: float, equal_going_right: bool) -> int:
        """std::partition on centroid[axis] vs threshold (in-place, then split)."""
        idx = self.primitive_indices[start:end]
        centers = 0.5 * (self.item_lowers[idx] + self.item_uppers[idx])[:, axis]
        if equal_going_right:
            mask = centers <= threshold  # midpoint variant
        else:
            mask = centers < threshold  # SAH variant
        self.primitive_indices[start:end] = np.concatenate([idx[mask], idx[~mask]])
        k = start + int(mask.sum())
        if k == start or k == end:
            k = (start + end) // 2
        return k

    def _partition_median(self, start: int, end: int, range_lo, range_up) -> int:
        """nth_element around the midpoint on the widest centroid axis."""
        edges = (range_up - range_lo).astype(np.float32)
        axis = int(np.argmax(edges))
        idx = self.primitive_indices[start:end]
        centers = 0.5 * (self.item_lowers[idx] + self.item_uppers[idx])[:, axis]
        order = np.argsort(centers, kind="stable")
        self.primitive_indices[start:end] = idx[order]
        return (start + end) // 2

    def _build_recursive(self, start: int, end: int, depth: int, parent: int, assigned_node: int = -1) -> int:
        n = end - start
        node_index = assigned_node if assigned_node >= 0 else self.num_nodes
        if assigned_node < 0:
            self.num_nodes += 1
        self.max_depth = max(self.max_depth, depth)

        lo, up = self._calc_bounds(start, end)

        if n <= self.leaf_size or depth >= _BVH_QUERY_STACK_SIZE - 1:
            self.node_lowers[node_index] = (lo[0], lo[1], lo[2], _make_payload(start, True))
            self.node_uppers[node_index] = (up[0], up[1], up[2], _make_payload(end, False))
            self.node_parents[node_index] = parent
            self.num_leaf_nodes += 1
            return node_index

        if self.constructor_type == SAH:
            split_point, axis = self._partition_sah(start, end, lo, up)
            split = self._partition_by(start, end, axis, split_point, equal_going_right=False)
        else:  # MEDIAN
            split = self._partition_median(start, end, lo, up)

        if split == start or split == end:
            split = (start + end) // 2

        left_child = self._build_recursive(start, split, depth + 1, node_index)
        right_child = self._build_recursive(split, end, depth + 1, node_index)

        self.node_lowers[node_index] = (lo[0], lo[1], lo[2], _make_payload(left_child, False))
        self.node_uppers[node_index] = (up[0], up[1], up[2], _make_payload(right_child, False))
        self.node_parents[node_index] = parent
        return node_index

    def _build_with_groups(self) -> None:
        groups = self.item_groups
        n = self.num_items
        sorted_indices = np.argsort(groups, kind="stable").astype(np.int32)
        self.primitive_indices = sorted_indices

        unique_groups: list[int] = []
        group_starts: list[int] = []
        group_ends: list[int] = []
        current = int(groups[sorted_indices[0]])
        unique_groups.append(current)
        group_starts.append(0)
        for i in range(1, n):
            g = int(groups[sorted_indices[i]])
            if g != current:
                group_ends.append(i)
                current = g
                unique_groups.append(current)
                group_starts.append(i)
        group_ends.append(n)
        num_groups = len(unique_groups)
        ug = np.array(unique_groups, dtype=np.int32)

        group_lo = np.zeros((num_groups, 3), dtype=np.float32)
        group_up = np.zeros((num_groups, 3), dtype=np.float32)
        for g in range(num_groups):
            prim = self.primitive_indices[group_starts[g] : group_ends[g]]
            group_lo[g] = self.item_lowers[prim].min(axis=0)
            group_up[g] = np.maximum(self.item_lowers[prim], self.item_uppers[prim]).max(axis=0)

        group_indices = np.arange(num_groups, dtype=np.int32)
        group_leaf_node_index = np.full(num_groups, -1, dtype=np.int64)

        def build_groups(start: int, end: int, depth: int, parent: int) -> int:
            count = end - start
            node_index = self.num_nodes
            self.num_nodes += 1
            self.max_depth = max(self.max_depth, depth)
            gi = group_indices[start:end]
            b_lo = group_lo[gi].min(axis=0)
            b_up = np.maximum(group_lo[gi], group_up[gi]).max(axis=0)
            if count == 1:
                g = int(group_indices[start])
                prim_start = group_starts[g]
                prim_end = group_ends[g]
                self.node_lowers[node_index] = (b_lo[0], b_lo[1], b_lo[2], _make_payload(prim_start, True))
                self.node_uppers[node_index] = (b_up[0], b_up[1], b_up[2], _make_payload(prim_end, False))
                self.node_parents[node_index] = parent
                group_leaf_node_index[g] = node_index
                return node_index
            mid = (start + end) // 2
            sub = group_indices[start:end]
            # nth_element on group ids: a stable full sort keeps the median
            # split point identical while deterministically ordering ties
            order = np.argsort(ug[sub], kind="stable")
            group_indices[start:end] = sub[order]
            left = build_groups(start, mid, depth + 1, node_index)
            right = build_groups(mid, end, depth + 1, node_index)
            self.node_lowers[node_index] = (b_lo[0], b_lo[1], b_lo[2], _make_payload(left, False))
            self.node_uppers[node_index] = (b_up[0], b_up[1], b_up[2], _make_payload(right, False))
            self.node_parents[node_index] = parent
            return node_index

        self.root = build_groups(0, num_groups, 0, -1)

        # expand each group leaf into a per-primitive subtree (in-place)
        for g in range(num_groups):
            node = int(group_leaf_node_index[g])
            if node < 0:
                continue
            prim_start = _payload_i(self.node_lowers[node])
            prim_end = _payload_i(self.node_uppers[node])
            self.node_lowers[node]["u"] = _make_payload(prim_start, False)
            self._build_recursive(prim_start, prim_end, 0, int(self.node_parents[node]), assigned_node=node)

        self._reorder()

    def _reorder(self) -> None:
        size = max(self.max_nodes, 1)
        new_lo = np.zeros(size, dtype=_NODE_DTYPE)
        new_up = np.zeros(size, dtype=_NODE_DTYPE)
        new_par = np.full(size, -1, dtype=np.int32)
        old_to_new = np.full(size, -1, dtype=np.int64)
        groups = self.item_groups

        next_pos = 0
        root_index = self.root

        # pass 1: leaves to the front; detect out-of-order group ids
        needs_sort = False
        prev_group = -(2**31)
        for i in range(self.num_nodes):
            if _is_leaf(self.node_lowers[i]):
                if groups is not None:
                    g = int(groups[int(self.primitive_indices[_payload_i(self.node_lowers[i])])])
                    if g < prev_group:
                        needs_sort = True
                    prev_group = g
                new_lo[next_pos] = self.node_lowers[i]
                new_up[next_pos] = self.node_uppers[i]
                old_to_new[i] = next_pos
                next_pos += 1

        if needs_sort:
            leaf_indices = [i for i in range(self.num_nodes) if _is_leaf(self.node_lowers[i])]
            leaf_indices.sort(
                key=lambda a: (
                    int(groups[int(self.primitive_indices[_payload_i(self.node_lowers[a])])]),
                    a,
                )
            )
            next_pos = 0
            for old_index in leaf_indices:
                new_lo[next_pos] = self.node_lowers[old_index]
                new_up[next_pos] = self.node_uppers[old_index]
                old_to_new[old_index] = next_pos
                next_pos += 1
        self.num_leaf_nodes = next_pos

        # pass 2: non-leaf, non-root nodes; root last
        for i in range(self.num_nodes):
            if i == root_index:
                self.root = 0 if _is_leaf(self.node_lowers[i]) else next_pos
            if not _is_leaf(self.node_lowers[i]):
                new_lo[next_pos] = self.node_lowers[i]
                new_up[next_pos] = self.node_uppers[i]
                old_to_new[i] = next_pos
                next_pos += 1

        for old_index in range(self.num_nodes):
            new_index = int(old_to_new[old_index])
            old_parent = int(self.node_parents[old_index])
            new_par[new_index] = int(old_to_new[old_parent]) if old_parent != -1 else -1
            if not _is_leaf(self.node_lowers[old_index]):
                new_lo[new_index]["u"] = _make_payload(int(old_to_new[_payload_i(self.node_lowers[old_index])]), False)
                new_up[new_index]["u"] = _make_payload(int(old_to_new[_payload_i(self.node_uppers[old_index])]), False)

        self.node_lowers = new_lo
        self.node_uppers = new_up
        self.node_parents = new_par

    # -- refit (bvh_refit_recursive port) ------------------------------------

    def refit(self) -> None:
        if self.num_items == 0:
            return
        self._refit_recursive(self.root)
        self._upload_nodes()

    def _refit_recursive(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        lower = self.node_lowers[index]
        upper = self.node_uppers[index]
        if _is_leaf(lower):
            lo, up = self._calc_bounds(_payload_i(lower), _payload_i(upper))
        else:
            lo_l, up_l = self._refit_recursive(_payload_i(lower))
            lo_r, up_r = self._refit_recursive(_payload_i(upper))
            lo = np.minimum(lo_l, lo_r)
            up = np.maximum(up_l, up_r)
        self.node_lowers[index]["x"], self.node_lowers[index]["y"], self.node_lowers[index]["z"] = lo
        self.node_uppers[index]["x"], self.node_uppers[index]["y"], self.node_uppers[index]["z"] = up
        return lo, up

    # -- device upload --------------------------------------------------------

    def _write_buffer(self, handle: int, arr: np.ndarray) -> None:
        dst = (ctypes.c_char * arr.nbytes).from_address(self._mm.buffer_ptr(handle))
        ctypes.memmove(dst, arr.view(np.uint8).reshape(-1).tobytes(), arr.nbytes)

    def _upload(self) -> None:
        """Allocate node/parent/context Metal buffers (shared storage)."""
        mm = self._mm
        self._buf_node_lowers = mm.buffer_create(max(self.node_lowers.nbytes, 16))
        self._buf_node_uppers = mm.buffer_create(max(self.node_uppers.nbytes, 16))
        self._buf_parents = mm.buffer_create(max(self.node_parents.nbytes, 16))
        self._buf_prims = mm.buffer_create(max(self.primitive_indices.nbytes, 16))
        self._buf_root = mm.buffer_create(4)
        self._buf_item_groups = self.groups_arr.buffer if self.groups_arr is not None else 0
        self._upload_nodes()

        ctx = self.embedded_context(mm)
        self._buf_ctx = mm.buffer_create(_CTX_BYTES)
        self._write_buffer(self._buf_ctx, ctx)
        self.id = mm.buffer_gpu_address(self._buf_ctx)

    def embedded_context(self, mm=None) -> np.ndarray:
        """The 112-byte wp_bvh_t device-struct image.

        ``_upload`` writes this into its own context buffer; the mesh table
        (runtime/host.py) embeds the same image INSIDE the 304-byte wp_mesh
        entry (upstream Mesh embeds the BVH by value — mesh_get_bvh addresses
        the member). Node-buffer addresses are stable across refits, so the
        image only changes when node buffers are reallocated (never after
        build; refit uploads in place).
        """
        if mm is None:
            mm = self._mm
        ctx = np.zeros(_CTX_BYTES, dtype=np.uint8)
        u64 = ctx.view("<u8")
        i32 = ctx.view("<i4")
        u64[0] = mm.buffer_gpu_address(self._buf_node_lowers)
        u64[1] = mm.buffer_gpu_address(self._buf_node_uppers)
        u64[2] = mm.buffer_gpu_address(self._buf_parents)
        u64[3] = 0  # node_counts (top-down builder leaves it null)
        u64[4] = mm.buffer_gpu_address(self._buf_prims)
        u64[5] = mm.buffer_gpu_address(self._buf_root)
        u64[6] = mm.buffer_gpu_address(self.lowers_arr.buffer)
        u64[7] = mm.buffer_gpu_address(self.uppers_arr.buffer)
        u64[8] = (
            mm.buffer_gpu_address(self._buf_item_groups) if self._buf_item_groups else 0
        )
        u64[13] = 0  # context pointer (upstream CUDA context; unused here)
        i32[18] = self.max_depth    # byte 72
        i32[19] = self.max_nodes    # byte 76
        i32[20] = self.num_nodes    # byte 80
        i32[21] = self.num_leaf_nodes  # byte 84
        i32[22] = self.num_items    # byte 88
        i32[23] = self.leaf_size    # byte 92
        i32[24] = self.constructor_type  # byte 96
        return ctx

    def destroy(self) -> None:
        """Release every Metal buffer this tree owns (residency set removal
        happens in the native buffer_release). The caller's item arrays are
        NOT owned here."""
        for handle in (
            self._buf_node_lowers,
            self._buf_node_uppers,
            self._buf_parents,
            self._buf_prims,
            self._buf_root,
            getattr(self, "_buf_ctx", None),
        ):
            if handle:
                self._mm.buffer_release(handle)

    def _upload_nodes(self) -> None:
        self._write_buffer(self._buf_node_lowers, self.node_lowers)
        self._write_buffer(self._buf_node_uppers, self.node_uppers)
        self._write_buffer(self._buf_parents, self.node_parents)
        self._write_buffer(self._buf_prims, self.primitive_indices)
        self._write_buffer(self._buf_root, np.array([self.root], dtype=np.int32))
