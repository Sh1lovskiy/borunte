# vision/graph/traversal.py
"""Graph traversal utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

from vision.graph.build import Graph
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def connected_components(graph: Graph) -> List[List[int]]:
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for start, end in graph.edges:
        adjacency[start].append(end)
        adjacency[end].append(start)
    visited: set[int] = set()
    components: List[List[int]] = []
    for node in range(len(graph.nodes)):
        if node in visited:
            continue
        queue = deque([node])
        component: List[int] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    queue.append(neighbour)
        components.append(component)
    LOGGER.debug("Found {} connected components", len(components))
    return components


__all__ = ["connected_components"]
