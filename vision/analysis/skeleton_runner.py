# vision/analysis/skeleton_runner.py
"""Skeleton extraction entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import get_settings
from utils.error_tracker import ErrorTracker
from utils.io import atomic_write_json, ensure_directory
from utils.logger import get_logger
from vision.cloud.io import load_point_cloud, save_point_cloud
from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from vision.skeleton.geodesic import cumulative_lengths
from vision.skeleton.project2d import project_to_plane
from vision.skeleton.thinning import skeletonize

LOGGER = get_logger(__name__)


def run_skeleton(
    cloud_path: Path,
    *,
    output_dir: Path | None = None,
    config: VisionConfig | None = None,
) -> Path:
    cfg = config or DEFAULT_VISION_CONFIG
    tracker = ErrorTracker(context="vision.skeleton")
    settings = get_settings()
    destination = output_dir or (settings.paths.saves_root / "vision" / "skeleton")
    ensure_directory(destination)
    try:
        cloud = load_point_cloud(cloud_path)
        projection = project_to_plane(cloud)
        skeleton_2d = skeletonize(projection, config=cfg)
        lengths = cumulative_lengths(skeleton_2d)
        skeleton_3d = np.hstack([skeleton_2d, np.zeros((len(skeleton_2d), 1))])
        skeleton_path = destination / "skeleton.npy"
        save_point_cloud(skeleton_path, skeleton_3d)
        atomic_write_json(
            destination / "skeleton_stats.json",
            {
                "points": len(skeleton_2d),
                "total_length": float(lengths[-1]) if len(lengths) else 0.0,
            },
        )
    except Exception as exc:  # noqa: BLE001
        tracker.record("skeleton", str(exc))
        raise
    tracker.summary()
    LOGGER.info("Skeleton saved to {}", destination)
    return skeleton_path


__all__ = ["run_skeleton"]
