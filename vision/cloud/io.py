# vision/cloud/io.py
"""Input/output helpers for point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.io import atomic_write_json, ensure_directory
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def load_point_cloud(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        cloud = np.load(path)
    else:
        cloud = np.loadtxt(path)
    LOGGER.info("Loaded point cloud {} with {} points", path.name, cloud.shape[0])
    return cloud.astype(float)


def save_point_cloud(path: Path, cloud: np.ndarray) -> None:
    ensure_directory(path.parent)
    if path.suffix == ".npy":
        np.save(path, cloud)
    else:
        np.savetxt(path, cloud)
    LOGGER.info("Saved point cloud {} with {} points", path.name, cloud.shape[0])


def export_statistics(path: Path, cloud: np.ndarray) -> None:
    stats = {
        "points": int(cloud.shape[0]),
        "mean": cloud.mean(axis=0).tolist(),
        "std": cloud.std(axis=0).tolist(),
    }
    atomic_write_json(path, stats)
    LOGGER.debug("Exported statistics for {}", path)


__all__ = ["export_statistics", "load_point_cloud", "save_point_cloud"]
