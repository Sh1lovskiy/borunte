# vision/viz/draw.py
"""Drawing utilities for point clouds and graphs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from vision.graph.build import Graph
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def draw_point_cloud(cloud: np.ndarray, *, config: VisionConfig | None = None) -> plt.Figure:
    cfg = config or DEFAULT_VISION_CONFIG
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=cfg.viz.point_size)
    ax.set_facecolor(cfg.viz.background_color)
    ax.set_title("Point Cloud")
    LOGGER.info("Rendered point cloud figure with {} points", cloud.shape[0])
    return fig


def draw_graph(graph: Graph, *, config: VisionConfig | None = None) -> plt.Figure:
    cfg = config or DEFAULT_VISION_CONFIG
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    nodes = graph.nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=cfg.viz.point_size)
    for start, end in graph.edges:
        segment = np.vstack([nodes[start], nodes[end]])
        ax.plot(segment[:, 0], segment[:, 1], color="white")
    ax.set_facecolor(cfg.viz.background_color)
    ax.set_title("Graph")
    LOGGER.info("Rendered graph figure with {} edges", len(graph.edges))
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved figure to {}", path)


__all__ = ["draw_graph", "draw_point_cloud", "save_figure"]
