# utils/rs.py
"""Minimal Intel RealSense helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import pyrealsense2 as rs
except Exception:  # noqa: BLE001
    rs = None  # type: ignore

from borunte.config import get_settings
from utils.io import ensure_directory
from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RealSenseSession:
    pipeline: Any
    config: Any


class RealSenseController:
    """Lifecycle manager for RealSense pipelines."""

    def __init__(self, *, output_dir: Optional[Path] = None) -> None:
        self._session: Optional[RealSenseSession] = None
        self._output_dir = output_dir or get_settings().paths.saves_root

    def start(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        if rs is None:  # pragma: no cover - hardware unavailable
            LOGGER.warning("pyrealsense2 not available; RealSenseController is idle")
            return
        if self._session is not None:
            LOGGER.debug("RealSense pipeline already active")
            return
        cfg_defaults = get_settings().realsense
        pipeline = rs.pipeline()
        config = rs.config()
        width = overrides.get("width", cfg_defaults.width) if overrides else cfg_defaults.width
        height = overrides.get("height", cfg_defaults.height) if overrides else cfg_defaults.height
        fps = overrides.get("fps", cfg_defaults.fps) if overrides else cfg_defaults.fps
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        pipeline.start(config)
        self._session = RealSenseSession(pipeline=pipeline, config=config)
        ensure_directory(self._output_dir)
        LOGGER.info(
            "Started RealSense pipeline at {}x{} {}fps, saving to {}",
            width,
            height,
            fps,
            self._output_dir,
        )

    def stop(self) -> None:
        if self._session is None:
            LOGGER.debug("RealSense pipeline already stopped")
            return
        self._session.pipeline.stop()
        self._session = None
        LOGGER.info("Stopped RealSense pipeline")

    def snapshot(self) -> Dict[str, Path]:
        ensure_directory(self._output_dir)
        color = self._output_dir / "color.png"
        depth = self._output_dir / "depth.png"
        LOGGER.info("Mock snapshot to {} and {}", color, depth)
        return {"color": color, "depth": depth}


__all__ = ["RealSenseController"]
