from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from utils.logger import Logger

LOG = Logger.get_logger("tk.graph_build")


@dataclass(frozen=True)
class Graph:
    """Minimal graph container for truss skeleton."""

    nodes: np.ndarray  # (N,3) float64
    edges: List[Tuple[int, int]]  # pairs of indices into nodes
    polylines: List[np.ndarray]  # optional, list of (K,3)

    def validate(self) -> None:
        n = int(self.nodes.shape[0])
        bad = [
            (u, v)
            for (u, v) in self.edges
            if not (0 <= u < n and 0 <= v < n and u != v)
        ]
        if bad:
            raise ValueError(f"invalid edges: {bad[:3]}...")


def _dedup_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not edges:
        return []
    e = [(min(u, v), max(u, v)) for (u, v) in edges if u != v]
    e = sorted(set(e))
    return e


def build(
    nodes_xyz: np.ndarray,
    edges: List[Tuple[int, int]],
    polylines_xyz: List[np.ndarray] | None = None,
) -> Graph:
    """Make Graph with light sanity checks and edge dedup."""
    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError("nodes must be (N,3)")
    nodes = np.asarray(nodes_xyz, dtype=float)
    e = _dedup_edges(edges)
    polys = polylines_xyz or []
    g = Graph(nodes=nodes, edges=e, polylines=polys)
    g.validate()
    LOG.info(f"graph: {len(nodes)} nodes {len(e)} edges " f"polys={len(polys)}")
    return g
