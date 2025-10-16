# calib/handeye.py
"""Hand-eye calibration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from calib.config import CalibConfig, DEFAULT_CALIB_CONFIG
from utils.error_tracker import ErrorTracker
from utils.format import numpy_print_options
from utils.geometry import Transformation, compose_transformations, make_transformation
from utils.io import atomic_write_json, load_json
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _aggregate_robot_transforms(poses: Iterable[Sequence[float]]) -> Transformation:
    transforms = [make_transformation(pose[:3], pose[3:6]) for pose in poses]
    if not transforms:
        raise ValueError("No robot poses provided")
    return compose_transformations(*transforms)


def run_handeye(
    pnp_path: Path,
    robot_poses: Iterable[Sequence[float]],
    *,
    config: CalibConfig | None = None,
    output_path: Path | None = None,
) -> Path:
    cfg = config or DEFAULT_CALIB_CONFIG
    tracker = ErrorTracker(context="calib.handeye")
    pnp = load_json(pnp_path)
    with numpy_print_options():
        try:
            robot_transform = _aggregate_robot_transforms(robot_poses)
        except Exception as exc:  # noqa: BLE001
            tracker.record("poses", str(exc))
            raise
    rotation = np.array(pnp["rotation"])
    translation = np.array(pnp["translation"])
    handeye_matrix = np.eye(4)
    handeye_matrix[:3, :3] = rotation
    handeye_matrix[:3, 3] = translation
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    LOGGER.info("Hand-eye rotation determinant: {:.6f}", determinant)
    LOGGER.info("Hand-eye orthogonality deviation: {:.6e}", orthogonality)
    result = {
        "handeye": handeye_matrix.tolist(),
        "robot": robot_transform.matrix.tolist(),
        "determinant": determinant,
        "orthogonality": orthogonality,
    }
    destination = output_path or pnp_path.parent / cfg.output.handeye_file
    atomic_write_json(destination, result)
    tracker.summary()
    return destination


__all__ = ["run_handeye"]
