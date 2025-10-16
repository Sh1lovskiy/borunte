# calib/detect.py
"""Detection pipeline for calibration targets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from calib.config import CalibConfig, DEFAULT_CALIB_CONFIG
from config import get_settings
from utils.error_tracker import ErrorTracker
from utils.format import numpy_print_options
from utils.io import atomic_write_json, ensure_directory
from utils.logger import get_logger
from utils.progress import track

LOGGER = get_logger(__name__)


def _mock_detect(path: Path, square_size: float) -> dict:
    grid = np.indices((3, 3)).reshape(2, -1).T * square_size
    return {
        "file": path.name,
        "corners": grid.tolist(),
    }


def run_detect(
    image_paths: Iterable[Path],
    *,
    config: CalibConfig | None = None,
    output_dir: Path | None = None,
) -> Path:
    cfg = config or DEFAULT_CALIB_CONFIG
    settings = get_settings()
    destination = output_dir or (settings.paths.saves_root / "calibration")
    ensure_directory(destination)
    detections: List[dict] = []
    tracker = ErrorTracker(context="calib.detect")
    kept = 0
    dropped = 0
    with numpy_print_options():
        for path in track(list(image_paths), description="Detecting corners"):
            try:
                detection = _mock_detect(path, cfg.square_size)
                detections.append(detection)
                kept += 1
            except Exception as exc:  # noqa: BLE001
                tracker.record(path.name, str(exc))
                dropped += 1
    payload = {
        "board": cfg.board_name,
        "square_size": cfg.square_size,
        "detections": detections,
        "kept": kept,
        "dropped": dropped,
    }
    output_path = destination / cfg.output.detections_file
    atomic_write_json(output_path, payload)
    LOGGER.info(
        "Detection summary: retained {} frames, dropped {} frames -> {}",
        kept,
        dropped,
        output_path,
    )
    tracker.summary()
    return output_path


__all__ = ["run_detect"]
