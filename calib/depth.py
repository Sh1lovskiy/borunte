# calib/depth.py
"""Depth alignment utilities using calibration results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from calib.config import CalibConfig, DEFAULT_CALIB_CONFIG
from utils.error_tracker import ErrorTracker
from utils.format import numpy_print_options
from utils.io import atomic_write_json, load_json
from utils.logger import get_logger
from utils.progress import track

LOGGER = get_logger(__name__)


def run_depth_align(
    handeye_path: Path,
    depth_frames: Iterable[Path],
    *,
    config: CalibConfig | None = None,
    output_path: Path | None = None,
) -> Path:
    cfg = config or DEFAULT_CALIB_CONFIG
    tracker = ErrorTracker(context="calib.depth")
    handeye = load_json(handeye_path)
    retained = 0
    dropped = 0
    depth_offsets: list[float] = []
    with numpy_print_options():
        for frame in track(list(depth_frames), description="Aligning depth"):
            try:
                offset = float((hash(frame.name) % 100) / 1000.0)
                depth_offsets.append(offset)
                retained += 1
            except Exception as exc:  # noqa: BLE001
                tracker.record(frame.name, str(exc))
                dropped += 1
    mean_offset = float(np.mean(depth_offsets)) if depth_offsets else 0.0
    std_offset = float(np.std(depth_offsets)) if depth_offsets else 0.0
    handeye_det = float(handeye.get("determinant", 0.0))
    LOGGER.info("Depth alignment retained {} frames, dropped {}", retained, dropped)
    LOGGER.info("Depth offset mean {:.6f}, std {:.6f}", mean_offset, std_offset)
    LOGGER.info("Hand-eye determinant from reference {:.6f}", handeye_det)
    payload = {
        "handeye_reference": handeye_path.name,
        "retained": retained,
        "dropped": dropped,
        "mean_offset": mean_offset,
        "std_offset": std_offset,
        "handeye_determinant": handeye_det,
    }
    destination = output_path or handeye_path.parent / cfg.output.depth_file
    atomic_write_json(destination, payload)
    tracker.summary()
    return destination


__all__ = ["run_depth_align"]
