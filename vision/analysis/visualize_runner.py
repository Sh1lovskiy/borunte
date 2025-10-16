# vision/analysis/visualize_runner.py
"""Visualisation entry point for vision results."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from config import get_settings
from utils.error_tracker import ErrorTracker
from utils.io import ensure_directory, load_json
from utils.logger import get_logger
from vision.cloud.io import load_point_cloud
from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from vision.graph.build import Graph
from vision.viz.draw import draw_graph, draw_point_cloud, save_figure

LOGGER = get_logger(__name__)


def run_visualize(
    cloud_path: Path,
    *,
    graph_path: Path | None = None,
    output_dir: Path | None = None,
    config: VisionConfig | None = None,
) -> List[Path]:
    cfg = config or DEFAULT_VISION_CONFIG
    tracker = ErrorTracker(context="vision.visualize")
    settings = get_settings()
    destination = output_dir or (settings.paths.saves_root / "vision" / "viz")
    ensure_directory(destination)
    outputs: List[Path] = []
    try:
        cloud = load_point_cloud(cloud_path)
        cloud_fig = draw_point_cloud(cloud, config=cfg)
        cloud_path_out = destination / "cloud.png"
        save_figure(cloud_fig, cloud_path_out)
        outputs.append(cloud_path_out)
        if graph_path and graph_path.exists():
            data = load_json(graph_path)
            nodes = np.asarray(cloud[:, :2])
            edges = [tuple(int(idx) for idx in edge) for edge in data.get("edges", [])]
            graph = Graph(nodes=nodes, edges=edges)
            graph_fig = draw_graph(graph, config=cfg)
            graph_path_out = destination / "graph.png"
            save_figure(graph_fig, graph_path_out)
            outputs.append(graph_path_out)
    except Exception as exc:  # noqa: BLE001
        tracker.record("visualize", str(exc))
        raise
    tracker.summary()
    LOGGER.info("Generated {} visualisation artefacts", len(outputs))
    return outputs


__all__ = ["run_visualize"]
