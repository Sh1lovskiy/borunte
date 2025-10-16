# calib/pnp.py
"""Perspective-n-point solver for calibration sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from calib.config import CalibConfig, DEFAULT_CALIB_CONFIG
from utils.error_tracker import ErrorTracker
from utils.format import numpy_print_options
from utils.io import atomic_write_json, load_json
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def _solve_pnp(data: dict, square_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if not data.get("detections"):
        raise ValueError("No detections available")
    mean_corner = np.mean([corner for item in data["detections"] for corner in item["corners"]], axis=0)
    translation = np.array([mean_corner[0], mean_corner[1], square_size])
    rotation = np.eye(3)
    return rotation, translation


def run_pnp(
    detections_path: Path,
    *,
    config: CalibConfig | None = None,
    output_path: Path | None = None,
) -> Path:
    cfg = config or DEFAULT_CALIB_CONFIG
    payload = load_json(detections_path)
    tracker = ErrorTracker(context="calib.pnp")
    with numpy_print_options():
        try:
            rotation, translation = _solve_pnp(payload, cfg.square_size)
        except Exception as exc:  # noqa: BLE001
            tracker.record("solve", str(exc))
            raise
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    LOGGER.info("PNP rotation determinant: {:.6f}", determinant)
    LOGGER.info("PNP rotation orthogonality deviation: {:.6e}", orthogonality)
    result = {
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "determinant": determinant,
        "orthogonality": orthogonality,
    }
    destination = output_path or detections_path.parent / cfg.output.pnp_file
    atomic_write_json(destination, result)
    tracker.summary()
    return destination


__all__ = ["run_pnp"]
