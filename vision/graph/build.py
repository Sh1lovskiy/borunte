# vision/graph/build.py
"""Graph construction utilities for skeletonised clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class Graph:
    nodes: np.ndarray
    edges: List[Tuple[int, int]]


def build_graph(points: np.ndarray, *, config: VisionConfig | None = None) -> Graph:
    cfg = config or DEFAULT_VISION_CONFIG
    edges: List[Tuple[int, int]] = []
    for idx in range(len(points) - 1):
        distance = np.linalg.norm(points[idx + 1] - points[idx])
        if distance <= cfg.graph.max_distance:
            edges.append((idx, idx + 1))
    LOGGER.debug("Built graph with {} nodes and {} edges", len(points), len(edges))
    return Graph(nodes=points, edges=edges)


__all__ = ["Graph", "build_graph"]
