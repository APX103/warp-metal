# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph coloring utilities for parallel constraint solving.

Upstream-verbatim port of ``warp/_src/coloring.py`` plus the native kernels
it calls (``warp/native/coloring.cpp``, Apache-2.0, © 2023 Anka He Chen —
MCS/degree-ordered greedy coloring and color balancing from the Gaia
project). Upstream delegates ``graph_coloring_assign`` /
``graph_coloring_balance`` to warp-native C; warp-metal has no warp-native
library, so the same algorithms run as pure host Python over the arrays'
numpy views. Newton's ``ModelBuilder.color`` (cloth particle constraint
grouping) calls these through ``wp.utils``; the operation is a pure host
graph algorithm on CPU arrays either way, so there is no GPU surface to
port — only the host algorithm.

Known porting note: the degree-ordered greedy's ``std::sort`` tie-break
among equal-degree nodes is unspecified in C++; this port's stable sort
preserves ascending node order among ties. MCS ordering is deterministic
and ported exactly. Both algorithms produce valid colorings either way.
"""

from __future__ import annotations

import enum
import operator
import warnings

import numpy as np


class GraphColoringAlgorithm(enum.IntEnum):
    """Graph coloring algorithm selection (upstream warp/_src/coloring.py)."""

    MCS = 0
    """Maximum Cardinality Search based coloring algorithm."""

    GREEDY = 1
    """Degree-ordered greedy coloring algorithm."""


_ERROR_STRING = ""


def _set_error_string(msg: str) -> None:
    global _ERROR_STRING
    _ERROR_STRING = msg


class _Graph:
    """CSR adjacency build (verbatim of ``wp::Graph`` in coloring.cpp)."""

    def __init__(self, num_nodes: int, edges_np: np.ndarray):
        self.num_nodes = num_nodes
        self.valid = True

        if edges_np.ndim != 2 or edges_np.shape[1] != 2:
            _set_error_string("The edges array must have shape (edge_count, 2)!")
            self.valid = False
            return

        self.node_offsets = [0] * (num_nodes + 1)
        self.node_colors = [-1] * num_nodes

        node_degrees = [0] * num_nodes

        # count degrees
        for edge_idx in range(edges_np.shape[0]):
            e0 = int(edges_np[edge_idx, 0])
            e1 = int(edges_np[edge_idx, 1])

            if e0 < 0 or e0 >= num_nodes or e1 < 0 or e1 >= num_nodes:
                _set_error_string(
                    f"Graph edge {edge_idx} has endpoints ({e0}, {e1}), but valid node indices are in the range [0, {num_nodes})."
                )
                self.valid = False
                return

            node_degrees[e0] += 1
            node_degrees[e1] += 1

        offset = 0
        for node in range(num_nodes):
            offset += node_degrees[node]
            self.node_offsets[node + 1] = offset

        # fill adjacency list
        node_adjacency_fill_count = [0] * num_nodes
        self.graph_flatten = [-1] * offset
        for edge_idx in range(edges_np.shape[0]):
            e0 = int(edges_np[edge_idx, 0])
            e1 = int(edges_np[edge_idx, 1])

            fill_count_e0 = node_adjacency_fill_count[e0]
            self.graph_flatten[self.node_offsets[e0] + fill_count_e0] = e1

            fill_count_e1 = node_adjacency_fill_count[e1]
            self.graph_flatten[self.node_offsets[e1] + fill_count_e1] = e0

            node_adjacency_fill_count[e0] = fill_count_e0 + 1
            node_adjacency_fill_count[e1] = fill_count_e1 + 1

    def get_node_neighbor(self, node: int, neighbor_index: int) -> int:
        return self.graph_flatten[self.node_offsets[node] + neighbor_index]

    def get_node_degree(self, node: int) -> int:
        return self.node_offsets[node + 1] - self.node_offsets[node]


def _type_repr(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype))


def _validate_edges_and_colors(edges, node_colors) -> None:
    """Shared validation of the two mutating entry points (verbatim of the
    checks in upstream ``graph_coloring_assign``/``graph_coloring_balance``)."""
    from .types import int32  # noqa: PLC0415 -- lazy: types imports utils at init

    if not edges.device.is_cpu:
        raise RuntimeError("edges array must be on the CPU")
    if not node_colors.device.is_cpu:
        raise RuntimeError("node_colors array must be on the CPU")

    if edges.dtype != int32:
        raise RuntimeError(f"edges array must have dtype int32, got {_type_repr(edges.dtype)}")
    if node_colors.dtype != int32:
        raise RuntimeError(f"node_colors array must have dtype int32, got {_type_repr(node_colors.dtype)}")

    if edges.ndim != 2:
        raise RuntimeError(f"edges array must be 2-dimensional, got {edges.ndim} dimensions")
    if edges.shape[1] != 2:
        raise RuntimeError(f"edges array must have shape (edge_count, 2), got shape {edges.shape}")
    if node_colors.ndim != 1:
        raise RuntimeError(f"node_colors array must be 1-dimensional, got {node_colors.ndim} dimensions")


def graph_coloring_assign(
    edges,
    node_colors,
    algorithm: GraphColoringAlgorithm = GraphColoringAlgorithm.MCS,
) -> int:
    """Assign colors to graph nodes such that no two adjacent nodes share the
    same color (upstream ``graph_coloring_assign``).

    Args:
        edges: A 2D array of shape ``(edge_count, 2)`` containing edge pairs
            (undirected). Must be a CPU array with ``int32`` dtype.
        node_colors: A 1D array of shape ``(node_count,)`` filled with the
            computed color assignments. Must be a CPU array with ``int32``
            dtype; its size determines the number of nodes.
        algorithm: The coloring algorithm to use.

    Returns:
        The number of colors used in the coloring.
    """
    # Validate device/dtype/shape
    _validate_edges_and_colors(edges, node_colors)

    node_count = node_colors.shape[0]

    if node_count == 0:
        raise RuntimeError("Cannot color an empty graph")

    edges_np = edges.numpy
    num_colors = _wp_graph_coloring(
        node_count,
        edges_np,
        int(algorithm),
        node_colors,
    )

    if num_colors < 0:
        raise RuntimeError(f"Graph coloring failed: {_ERROR_STRING}")

    return num_colors


def graph_coloring_balance(
    edges,
    node_colors,
    color_count: int,
    target_max_min_ratio: float,
    *,
    max_iterations: int | None = None,
) -> float:
    """Balance the sizes of color groups in a graph coloring (upstream
    ``graph_coloring_balance``).

    Moves nodes between color groups while maintaining coloring validity;
    stops when ``target_max_min_ratio`` is achieved, no further improvement
    is possible, or ``max_iterations`` is reached.

    Returns:
        The actual max/min ratio achieved after balancing.
    """
    _validate_edges_and_colors(edges, node_colors)

    node_count = node_colors.shape[0]

    if max_iterations is None:
        max_iterations = node_count
    elif isinstance(max_iterations, (bool, np.bool_)):
        raise TypeError(f"max_iterations must be an integer or None, got {type(max_iterations).__name__}")
    else:
        try:
            max_iterations = operator.index(max_iterations)
        except TypeError:
            raise TypeError(
                f"max_iterations must be an integer or None, got {type(max_iterations).__name__}"
            ) from None

        if max_iterations < 0:
            raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")
        if max_iterations > np.iinfo(np.int32).max:
            raise ValueError(f"max_iterations must not exceed {np.iinfo(np.int32).max}, got {max_iterations}")

    iteration_limit_reached = [0]
    max_min_ratio = _wp_balance_coloring(
        node_count,
        edges.numpy,
        color_count,
        target_max_min_ratio,
        max_iterations,
        node_colors,
        iteration_limit_reached,
    )

    if max_min_ratio < 0.0:
        raise RuntimeError(f"Graph coloring balance failed: {_ERROR_STRING}")

    if iteration_limit_reached[0]:
        warnings.warn(
            f"Graph coloring stopped because it reached the {max_iterations}-iteration limit "
            f"(achieved ratio {max_min_ratio:g}, target {target_max_min_ratio:g}).",
            stacklevel=2,
        )

    return max_min_ratio


def graph_coloring_get_groups(
    node_colors,
    color_count: int,
    return_wp_array: bool = True,
    device="cpu",
):
    """Convert node colors into per-color groups (upstream
    ``graph_coloring_get_groups``).

    Upstream drives two deliberately single-threaded kernels (sequential
    node-index loops); this port computes the identical result directly on
    the host numpy view: within each color, nodes appear in ascending node
    index order (the kernels' fill order).
    """
    from .types import int32  # noqa: PLC0415 -- lazy: types imports utils at init
    # public wp.array = the runtime WarpArray constructor (upstream returns
    # wp.array objects; the vendored frontend array class has no allocator on
    # the runtime Device — all Newton allocations go through the host runtime)
    from ..runtime.host import array as wp_array  # noqa: PLC0415

    if color_count < 0:
        raise RuntimeError("color_count must be non-negative")
    if color_count == 0:
        return ()

    if not node_colors.device.is_cpu:
        raise RuntimeError("node_colors array must be on the CPU")
    if node_colors.dtype != int32:
        raise RuntimeError(f"node_colors array must have dtype int32, got {_type_repr(node_colors.dtype)}")
    if node_colors.ndim != 1:
        raise RuntimeError(f"node_colors array must be 1-dimensional, got {node_colors.ndim} dimensions")

    colors_np = node_colors.numpy
    # the sequential kernels fill each group with nodes in ascending node
    # order — a stable argsort over the colors reproduces that exactly
    order = np.argsort(colors_np, kind="stable")
    group_sizes = np.bincount(colors_np, minlength=color_count)[:color_count]
    group_offsets = np.concatenate(
        (
            np.array([0], dtype=group_sizes.dtype),
            np.cumsum(group_sizes, dtype=group_sizes.dtype),
        )
    )

    if return_wp_array:
        return tuple(
            wp_array(
                order[group_offsets[color_idx] : group_offsets[color_idx + 1]],
                dtype=int32,
                device=device,
            )
            for color_idx in range(color_count)
        )

    return tuple(
        order[group_offsets[color_idx] : group_offsets[color_idx + 1]]
        for color_idx in range(color_count)
    )


# ---------------------------------------------------------------------------
# native port (warp/native/coloring.cpp)
# ---------------------------------------------------------------------------

_NODE_WEIGHTS_PREALLOC = 64


class _NodeWeightBuckets:
    """Weight-bucket priority structure (verbatim of ``NodeWeightBuckets``)."""

    def __init__(self, num_nodes: int):
        # node_weights: -1 marks a popped (visited) node
        self.node_weights = [0] * num_nodes
        self.node_indices_in_bucket = [-1] * num_nodes
        # C++ resizes to NODE_WEIGHTS_PREALLOC up front; Python lists grow
        # on demand with the same semantics (append preserves existing lists)
        self.weight_buckets = []
        self.max_weight = 0

    def get_node_weight(self, node_idx: int) -> int:
        return self.node_weights[node_idx]

    def _ensure_bucket(self, weight: int) -> None:
        while len(self.weight_buckets) <= weight:
            self.weight_buckets.append([])

    def add_node(self, weight: int, node_idx: int) -> None:
        self._ensure_bucket(weight)
        bucket = self.weight_buckets[weight]
        self.node_indices_in_bucket[node_idx] = len(bucket)
        self.node_weights[node_idx] = weight
        bucket.append(node_idx)

        if self.max_weight < weight:
            self.max_weight = weight

    def pop_node_with_max_weight(self) -> int:
        bucket = self.weight_buckets[self.max_weight]
        node_with_max_weight = bucket[0]
        self.node_indices_in_bucket[node_with_max_weight] = -1

        # we pop the first element so it has a breadth-first like behavior,
        # which is better than depth-first (swap back → front, then pop)
        if len(bucket) > 1:
            back = bucket[-1]
            self.node_indices_in_bucket[back] = 0
            bucket[0] = back
        bucket.pop()
        # mark node deleted
        self.node_weights[node_with_max_weight] = -1

        if len(bucket) == 0:
            # we need to update max_weight because the bucket became empty
            new_max_weight = 0
            for bucket_idx in range(self.max_weight - 1, -1, -1):
                if self.weight_buckets[bucket_idx]:
                    new_max_weight = bucket_idx
                    break

            self.max_weight = new_max_weight
        # mark deleted
        return node_with_max_weight

    def increase_node_weight(self, node_idx: int) -> None:
        weight = self.node_weights[node_idx]
        node_idx_in_bucket = self.node_indices_in_bucket[node_idx]
        bucket = self.weight_buckets[weight]

        # swap index with the last element, O(1) erase
        back = bucket[-1]
        self.node_indices_in_bucket[back] = node_idx_in_bucket
        bucket[node_idx_in_bucket] = back
        bucket.pop()

        self.add_node(weight + 1, node_idx)

    def empty(self) -> bool:
        if len(self.weight_buckets) == 0:
            return True
        return self.max_weight <= 0 and len(self.weight_buckets[0]) == 0


def _ordered_greedy(order, graph: _Graph) -> int:
    """Greedy coloring over an order (verbatim of
    ``graph_coloring_ordered_greedy``)."""
    max_color = -1
    color_used: list[bool] = []

    for node in order:
        # first one
        if max_color == -1:
            max_color += 1
            graph.node_colors[node] = max_color
        else:
            color_used = [False] * (max_color + 1)

            # see its neighbor's color
            for nei_counter in range(graph.get_node_degree(node)):
                nei_node_idx = graph.get_node_neighbor(node, nei_counter)
                if graph.node_colors[nei_node_idx] >= 0:
                    color_used[graph.node_colors[nei_node_idx]] = True

            # find the minimal usable color
            min_usable_color = -1
            for color_counter in range(len(color_used)):
                if not color_used[color_counter]:
                    min_usable_color = color_counter
                    break
            if min_usable_color == -1:
                max_color += 1
                graph.node_colors[node] = max_color
            else:
                graph.node_colors[node] = min_usable_color

    return max_color + 1


def _graph_coloring_mcs_vector(graph: _Graph) -> int:
    """MCS ordering + greedy coloring (verbatim of
    ``graph_coloring_mcs_vector``; Pereira & Palsberg 2005)."""
    ordering: list[int] = []

    weight_buckets = _NodeWeightBuckets(graph.num_nodes)
    # add the first node
    weight_buckets.add_node(0, 0)

    for _node_idx in range(graph.num_nodes):
        # this only happens once per connected component
        if weight_buckets.empty():
            non_negative_node = -1
            for i in range(graph.num_nodes):
                if weight_buckets.get_node_weight(i) >= 0:
                    non_negative_node = i
                    break
            assert weight_buckets.get_node_weight(non_negative_node) == 0
            weight_buckets.add_node(0, non_negative_node)

        max_node = weight_buckets.pop_node_with_max_weight()

        # add highest weight node to the queue and increment all of its
        # neighbors' weights by 1
        ordering.append(max_node)

        for j in range(graph.get_node_degree(max_node)):
            neighbor_node = graph.get_node_neighbor(max_node, j)
            old_weight = weight_buckets.get_node_weight(neighbor_node)

            if old_weight == 0:
                # 0-weighted node is not in buckets by default
                weight_buckets.add_node(old_weight + 1, neighbor_node)
            elif old_weight > 0:
                weight_buckets.increase_node_weight(neighbor_node)
            # skip neighbor nodes with negative weight because they are visited

    return _ordered_greedy(ordering, graph)


def _graph_coloring_degree_ordered_greedy(graph: _Graph) -> int:
    """Degree-ordered greedy (verbatim of
    ``graph_coloring_degree_ordered_greedy``; Fratarcangeli & Pellacini 2015)."""
    # initialize the degree
    degrees = [graph.get_node_degree(node_idx) for node_idx in range(graph.num_nodes)]

    # order them in a descending order (stable among ties — see module note)
    ordering = sorted(range(graph.num_nodes), key=lambda n: -degrees[n])

    return _ordered_greedy(ordering, graph)


def _wp_graph_coloring(num_nodes: int, edges_np: np.ndarray, algorithm: int, node_colors) -> int:
    """Port of the native ``wp_graph_coloring`` entry point."""
    global _ERROR_STRING

    if node_colors.shape[0] != num_nodes:
        _set_error_string("The node_colors array must have the preallocated shape of (num_nodes,)!")
        return -1

    if num_nodes == 0:
        _set_error_string("Empty graph!")
        return -1

    # convert to a format that coloring algorithm can recognize
    graph = _Graph(num_nodes, edges_np)
    if not graph.valid:
        # Color counts are non-negative, so use -1 to signal failure.
        return -1

    if algorithm == 0:
        # mcs algorithm
        num_colors = _graph_coloring_mcs_vector(graph)
    elif algorithm == 1:
        # greedy
        num_colors = _graph_coloring_degree_ordered_greedy(graph)
    else:
        _set_error_string(f"Unrecognized coloring algorithm number: {algorithm}!")
        return -1

    # copy the color info back
    node_colors.numpy[:] = np.array(graph.node_colors, dtype=np.int32)

    return num_colors


def _convert_to_color_groups(num_colors: int, node_colors: list[int]) -> list[list[int]] | None:
    color_groups: list[list[int]] = [[] for _ in range(num_colors)]

    for node_idx, color in enumerate(node_colors):
        if color < 0 or color >= num_colors:
            _set_error_string(f"Node {node_idx} has color {color}, but valid colors are in the range [0, {num_colors}).")
            return None

        color_groups[color].append(node_idx)

    return color_groups


def _find_largest_smallest_groups(color_groups: list[list[int]]) -> tuple[float, int, int]:
    if len(color_groups) == 0:
        return 1.0, -1, -1

    max_size = len(color_groups[0])
    biggest_group = 0
    min_size = len(color_groups[0])
    smallest_group = 0

    for color in range(len(color_groups)):
        if max_size < len(color_groups[color]):
            biggest_group = color
            max_size = len(color_groups[color])

        if min_size > len(color_groups[color]):
            smallest_group = color
            min_size = len(color_groups[color])

    ratio = float(len(color_groups[biggest_group])) / float(len(color_groups[smallest_group]))
    return ratio, biggest_group, smallest_group


def _color_changeable(graph: _Graph, node: int, target_color: int) -> bool:
    # loop through node and see if it has target color
    for i in range(graph.get_node_degree(node)):
        nei_node_idx = graph.get_node_neighbor(node, i)
        if graph.node_colors[nei_node_idx] == target_color:
            return False
    return True


def _find_changeable_node_in_category(
    graph: _Graph, color_groups: list[list[int]], source_color: int, target_color: int
) -> int:
    source_group = color_groups[source_color]
    for node_idx in range(len(source_group)):
        if _color_changeable(graph, source_group[node_idx], target_color):
            return node_idx
    return -1


def _change_color(
    color: int,
    node_idx_in_group: int,
    target_color: int,
    node_colors: list[int],
    color_groups: list[list[int]],
) -> None:
    node_idx = color_groups[color][node_idx_in_group]
    node_colors[node_idx] = target_color

    if color_groups:
        # O(1) erase
        color_groups[color][node_idx_in_group] = color_groups[color][-1]
        color_groups[color].pop()

        color_groups[target_color].append(node_idx)


def _balance_color_groups(
    target_max_min_ratio: float,
    max_iterations: int,
    graph: _Graph,
    color_groups: list[list[int]],
    iteration_limit_reached: list[int],
) -> float:
    max_min_ratio = -1.0
    last_changed_node = -1
    last_changed_color = -1
    iteration = 0

    while True:
        prev_max_min_ratio = max_min_ratio
        max_min_ratio, biggest_group, smallest_group = _find_largest_smallest_groups(color_groups)

        if prev_max_min_ratio > 0.0 and prev_max_min_ratio < max_min_ratio:
            # Ratios do not increase before the first regression, so undoing
            # the last move restores the best coloring.
            graph.node_colors[last_changed_node] = last_changed_color
            return prev_max_min_ratio

        # graph is not optimizable anymore or target ratio reached
        if len(color_groups[biggest_group]) - len(color_groups[smallest_group]) <= 2 or (
            max_min_ratio <= target_max_min_ratio
        ):
            return max_min_ratio

        # find an available vertex from the biggest category to move to the
        # smallest category
        changeable_color_group_idx = biggest_group
        changeable_node_idx = _find_changeable_node_in_category(graph, color_groups, biggest_group, smallest_group)
        if changeable_node_idx == -1:
            for color in range(len(color_groups)):
                if color == biggest_group or color == smallest_group:
                    continue

                changeable_node_idx = _find_changeable_node_in_category(graph, color_groups, color, smallest_group)

                if changeable_node_idx != -1:
                    changeable_color_group_idx = color
                    break

        if changeable_node_idx == -1:
            # the graph is not optimizable anymore
            return max_min_ratio

        if iteration == max_iterations:
            iteration_limit_reached[0] = 1
            return max_min_ratio

        # change the color of changeable_node in group
        # changeable_color_group_idx to the smallest group
        last_changed_node = color_groups[changeable_color_group_idx][changeable_node_idx]
        last_changed_color = changeable_color_group_idx
        _change_color(
            changeable_color_group_idx,
            changeable_node_idx,
            smallest_group,
            graph.node_colors,
            color_groups,
        )
        iteration += 1


def _wp_balance_coloring(
    num_nodes: int,
    edges_np: np.ndarray,
    num_colors: int,
    target_max_min_ratio: float,
    max_iterations: int,
    node_colors,
    iteration_limit_reached: list[int],
) -> float:
    """Port of the native ``wp_balance_coloring`` entry point."""
    global _ERROR_STRING

    iteration_limit_reached[0] = 0

    if num_nodes < 0:
        _set_error_string(f"The num_nodes value must be non-negative, got {num_nodes}!")
        return -1.0

    if max_iterations < 0:
        _set_error_string(f"The max_iterations value must be non-negative, got {max_iterations}!")
        return -1.0

    graph = _Graph(num_nodes, edges_np)
    if not graph.valid:
        # Ratios are non-negative, so use -1.0 to signal failure.
        return -1.0

    # copy the color info to graph
    graph.node_colors = [int(c) for c in node_colors.numpy]
    if num_colors > 1:
        color_groups = _convert_to_color_groups(num_colors, graph.node_colors)
        if color_groups is None:
            return -1.0

        max_min_ratio = _balance_color_groups(
            target_max_min_ratio, max_iterations, graph, color_groups, iteration_limit_reached
        )
        node_colors.numpy[:] = np.array(graph.node_colors, dtype=np.int32)

        return max_min_ratio
    else:
        return 1.0
