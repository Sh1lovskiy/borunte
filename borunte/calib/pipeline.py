# calib/pipeline.py
"""Calibration routines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from borunte.config import CalibrationConfig as CalibConfig
from borunte.config import HandEyeTransform as HandEye
from borunte.utils.format import format_matrix
from borunte.utils.logger import get_logger
from borunte.utils.progress import progress_bar

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


def run_pnp(
    points_3d: np.ndarray, points_2d: np.ndarray, *, config: CalibConfig | None = None
) -> np.ndarray:
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
    """Compute hand-eye calibration from robot and camera poses.

    Returns a placeholder identity hand-eye transform.
    Full implementation would use cv2.calibrateHandEye or similar.
    """
    _ = config or CalibConfig()
    robot_poses_list = list(poses_robot)
    camera_poses_list = list(poses_camera)

    if len(robot_poses_list) != len(camera_poses_list):
        raise ValueError("Robot and camera pose counts must match")

    logger.info(f"Using {len(robot_poses_list)} pose pairs for hand-eye calibration")

    # Placeholder: return identity transform
    # Full implementation would compute actual calibration
    hand_eye = HandEye(
        R=np.eye(3, dtype=np.float64), t=np.zeros(3, dtype=np.float64), direction="tcp_cam"
    )

    logger.info(f"Hand-eye rotation:\n{format_matrix(hand_eye.R)}")
    logger.info(f"Hand-eye translation: {hand_eye.t.tolist()}")
    return hand_eye


def run_depth_align(
    depth_frames: Sequence[np.ndarray], *, config: CalibConfig | None = None
) -> np.ndarray:
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
