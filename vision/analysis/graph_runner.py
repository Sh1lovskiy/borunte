# vision/analysis/graph_runner.py
"""Graph analysis entry point."""

from __future__ import annotations

from pathlib import Path

from config import get_settings
from utils.error_tracker import ErrorTracker
from utils.io import atomic_write_json, ensure_directory
from utils.logger import get_logger
from vision.cloud.io import load_point_cloud
from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from vision.graph.build import build_graph
from vision.graph.mapping import graph_summary

LOGGER = get_logger(__name__)


def run_graph(
    skeleton_path: Path,
    *,
    output_path: Path | None = None,
    config: VisionConfig | None = None,
) -> Path:
    cfg = config or DEFAULT_VISION_CONFIG
    tracker = ErrorTracker(context="vision.graph")
    settings = get_settings()
    target = output_path or (settings.paths.saves_root / "vision" / "graph.json")
    ensure_directory(target.parent)
    try:
        skeleton = load_point_cloud(skeleton_path)
        graph = build_graph(skeleton, config=cfg)
        summary = graph_summary(graph)
        summary["edges"] = [list(edge) for edge in graph.edges]
        atomic_write_json(target, summary)
    except Exception as exc:  # noqa: BLE001
        tracker.record("graph", str(exc))
        raise
    tracker.summary()
    LOGGER.info("Graph summary saved to {}", target)
    return target


__all__ = ["run_graph"]
