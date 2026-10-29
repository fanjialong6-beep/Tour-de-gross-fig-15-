from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable
import numpy as np

from .gross_code import GrossCode


Node = Hashable
MERGED: Node = ("merged",)


@dataclass(frozen=True)
class Edge:
    index: int
    u: Node
    v: Node
    z_check: int | None
    kind: str


@dataclass
class LpuGraph:
    mode: str
    vertices: list[Node]
    edges: list[Edge]
    cycles: list[list[int]]
    support: dict[Node, list[int]]
    left_vertices: set[Node]
    right_vertices: set[Node]

    # Return the indices of edges incident on one LPU vertex.
    def incident_edges(self, vertex: Node) -> list[int]:
        return [edge.index for edge in self.edges if edge.u == vertex or edge.v == vertex]

    # Find paths from the LPU root for gauge correction.
    def spanning_tree_paths(self, root: Node = MERGED) -> dict[Node, list[int]]:
        adjacency: dict[Node, list[tuple[Node, int]]] = {v: [] for v in self.vertices}
        for edge in self.edges:
            adjacency[edge.u].append((edge.v, edge.index))
            adjacency[edge.v].append((edge.u, edge.index))
        paths: dict[Node, list[int]] = {root: []}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for other, edge_index in adjacency[node]:
                if other not in paths:
                    paths[other] = paths[node] + [edge_index]
                    queue.append(other)
        if len(paths) != len(self.vertices):
            raise ValueError("LPU graph is disconnected")
        return paths


# Parse a lattice label into a physical qubit index.
def _idx(code: GrossCode, text: str) -> int:
    block = 0 if text.endswith("L") else 1
    body = text[:-1]
    if body == "1":
        return code.index(block, 0, 0)
    x = y = 0
    if "x" in body:
        after_x = body.split("x", 1)[1]
        x_text = after_x.split("y", 1)[0]
        x = int(x_text) if x_text else 1
    if "y" in body:
        y_text = body.split("y", 1)[1]
        y = int(y_text) if y_text else 1
    return code.index(block, x, y)


# Return the physical supports of the two pivot X operators.
def _support_nodes(code: GrossCode) -> tuple[list[int], list[int]]:
    x1 = np.flatnonzero(code.logical_x[0]).tolist()
    x7 = np.flatnonzero(code.logical_x[1]).tolist()
    return x1, x7


# Convert a closed vertex sequence into LPU edge indices.
def _cycle_from_vertices(sequence: list[Node], lookup: dict[frozenset[Node], int]) -> list[int]:
    edge_indices: list[int] = []
    for a, b in zip(sequence, sequence[1:] + sequence[:1]):
        key = frozenset((a, b))
        if key not in lookup:
            raise ValueError(f"cycle uses missing edge: {a!r} -- {b!r}")
        edge_indices.append(lookup[key])
    return edge_indices


# Construct the half or full LPU graph for X or XX measurement.
def build_lpu(code: GrossCode, mode: str) -> LpuGraph:
    if mode not in {"X", "XX"}:
        raise ValueError(mode)
    x1, x7 = _support_nodes(code)
    left_merge_data = _idx(code, "x4R")
    right_merge_data = _idx(code, "x9yL")
    assert left_merge_data in x1 and right_merge_data in x7

    # Map a supported data qubit to its LPU vertex.
    def node_for(data_index: int) -> Node:
        if data_index in {left_merge_data, right_merge_data}:
            return MERGED
        return data_index

    left_vertices = {node_for(q) for q in x1}
    right_vertices = {node_for(q) for q in x7}
    vertices = sorted(left_vertices, key=repr)
    if mode == "XX":
        vertices = sorted(left_vertices | right_vertices, key=repr)

    support: dict[Node, list[int]] = {v: [] for v in vertices}
    for q in x1:
        support[node_for(q)].append(q)
    if mode == "XX":
        for q in x7:
            support[node_for(q)].append(q)

    edges: list[Edge] = []

    # Add one half of the LPU and its cycle checks.
    def add_half(physical_support: list[int], kind: str) -> None:
        support_set = set(physical_support)
        for check_index, row in enumerate(code.hz):
            overlap = sorted(support_set & set(np.flatnonzero(row).tolist()))
            if not overlap:
                continue
            if len(overlap) != 2:
                raise ValueError(f"Z check {check_index} overlaps {kind} support {len(overlap)} times")
            u, v = node_for(overlap[0]), node_for(overlap[1])
            edges.append(Edge(len(edges), u, v, check_index, kind))

    add_half(x1, "left")
    if mode == "XX":
        add_half(x7, "right")
    if len([e for e in edges if e.kind == "left"]) != 18:
        raise AssertionError("paper requires 18 edges in G_l")
    if mode == "XX" and len([e for e in edges if e.kind == "right"]) != 18:
        raise AssertionError("paper requires 18 edges in G_r")

    top_labels = [
        "x6y5L", "x4y2L", "x3y5R", "x5y4L", "x6yL", "x4y2R",
        "x5L", "x4L", "x3y2R", "x3yR", "x3R",
    ]
    bottom_labels = [
        "x11y4L", "x8y5R", "x8y4R", "1L", "xR", "x4y4R",
        "x8L", "x9R", "y5R", "x3y4L", "xyL",
    ]
    top = [node_for(_idx(code, label)) for label in top_labels]
    bottom = [node_for(_idx(code, label)) for label in bottom_labels]
    if mode == "XX":
        for u, v in zip(top, bottom):
            edges.append(Edge(len(edges), u, v, None, "bridge"))

    lookup: dict[frozenset[Node], int] = {}
    for edge in edges:
        key = frozenset((edge.u, edge.v))
        if key in lookup:
            raise ValueError(f"parallel LPU edges are not expected: {edge.u!r}, {edge.v!r}")
        lookup[key] = edge.index

    left_cycle_labels = [
        ["x4L", "x3y2R", "x3yR", "x5L"],
        ["x3y5R", "x3R", "x6y5L", "x4y2L"],
        ["x3y5R", "x3R", "x6y5L", "x4R", "x5y4L"],
        ["x4L", "x3y2R", "x6yL", "x5y4L", "x4R"],
        ["x3yR", "x3y2R", "x6yL", "x4y2R", "x5L"],
    ]
    cycles = [
        _cycle_from_vertices([node_for(_idx(code, label)) for label in cycle], lookup)
        for cycle in left_cycle_labels
    ]

    if mode == "XX":
        right_cycle_labels = [
            ["x9yL", "x8y4R", "x8y5R", "x11y4L"],
            ["xyL", "xR", "x4y4R", "x3y4L"],
            ["1L", "y5R", "x9R", "x9yL", "x8y4R"],
        ]
        cycles.extend(
            _cycle_from_vertices([node_for(_idx(code, label)) for label in cycle], lookup)
            for cycle in right_cycle_labels
        )
        for i in range(10):
            cycles.append(_cycle_from_vertices([top[i], top[i + 1], bottom[i + 1], bottom[i]], lookup))
        cycles.append(_cycle_from_vertices([MERGED, top[0], bottom[0]], lookup))

    expected = (12, 18, 5) if mode == "X" else (23, 47, 19)
    actual = (len(vertices), len(edges), len(cycles))
    if actual != expected:
        raise AssertionError(f"LPU census mismatch: expected {expected}, got {actual}")
    if any(len(support[v]) not in {1, 2} for v in vertices):
        raise AssertionError("each active vertex must touch one code qubit, except the merged vertex")
    return LpuGraph(mode, vertices, edges, cycles, support, left_vertices, right_vertices)

