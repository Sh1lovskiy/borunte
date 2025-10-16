# vision/graph/mapping.py
"""Mappings between skeleton graphs and application structures."""

from __future__ import annotations

from typing import Dict, List

from vision.graph.build import Graph
from vision.graph.traversal import connected_components
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def graph_summary(graph: Graph) -> Dict[str, List[int]]:
    components = connected_components(graph)
    summary = {
        "component_sizes": [len(component) for component in components],
        "edge_count": len(graph.edges),
    }
    LOGGER.info("Graph summary with {} components", len(components))
    return summary


__all__ = ["graph_summary"]
