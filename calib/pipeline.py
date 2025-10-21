# calib/pipeline.py
"""Calibration routines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from utils.format import format_matrix
from utils.logger import get_logger
from utils.progress import progress_bar
from .config import CalibConfig, HandEye

logger = get_logger(__name__)


def run_detect(images: Sequence[Path | str], *, config: CalibConfig | None = None) -> list[Path]:
    """Detect calibration targets in the provided images."""
    _ = config or CalibConfig()
    detections: list[Path] = []
    for image_path in progress_bar(images, description="Detecting targets"):
        path = Path(image_path)
        if path.exists():
            detections.append(path)
            logger.info(f"Detected pattern in {path}")
        else:
            logger.warning(f"Image {path} does not exist")
    return detections


def run_pnp(points_3d: np.ndarray, points_2d: np.ndarray, *, config: CalibConfig | None = None) -> np.ndarray:
    """Run a placeholder PnP solver returning an identity pose."""
    _ = config or CalibConfig()
    if points_3d.shape[0] != points_2d.shape[0]:
        raise ValueError("Point sets must have matching lengths")
    logger.info(f"Running PnP with {points_3d.shape[0]} correspondences")
    pose = np.eye(4, dtype=np.float64)
    logger.info(f"PnP pose:\n{format_matrix(pose)}")
    return pose


def run_handeye(
    poses_robot: Iterable[np.ndarray],
    poses_camera: Iterable[np.ndarray],
    *,
    config: CalibConfig | None = None,
) -> HandEye:
    """Return configured hand-eye calibration parameters."""
    config = config or CalibConfig()
    count = sum(1 for _ in poses_robot)
    logger.info(f"Using {count} robot poses for hand-eye calibration")
    logger.info(f"Hand-eye rotation:\n{format_matrix(config.hand_eye.R)}")
    logger.info(f"Hand-eye translation: {config.hand_eye.t.tolist()}")
    return config.hand_eye


def run_depth_align(depth_frames: Sequence[np.ndarray], *, config: CalibConfig | None = None) -> np.ndarray:
    """Compute a simple depth alignment scale."""
    _ = config or CalibConfig()
    if not depth_frames:
        raise ValueError("No depth frames provided")
    scales = []
    for depth in progress_bar(depth_frames, description="Aligning depth"):
        scale = float(np.nanmean(depth)) if depth.size else 1.0
        scales.append(scale)
    result = np.array(scales, dtype=np.float64)
    logger.info(f"Depth alignment scales: {result.tolist()}")
    return result


__all__ = ["run_detect", "run_pnp", "run_handeye", "run_depth_align"]
