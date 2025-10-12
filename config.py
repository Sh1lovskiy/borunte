# config.py
"""Project-wide settings for Borunte robotics and calibration stack."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


_REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Cross-cutting configuration shared by robotics and calibration packages."""

    repo_root: Path = _REPO_ROOT
    data_root: Path = _env_path("BORUNTE_DATA_ROOT", _REPO_ROOT / "data")
    captures_root: Path = _env_path("BORUNTE_CAPTURES_ROOT", _REPO_ROOT / "captures")
    saves_root: Path = _env_path("BORUNTE_SAVES_ROOT", _REPO_ROOT / "saves")
    logs_root: Path = _env_path("BORUNTE_LOGS_ROOT", _REPO_ROOT / "logs")

    device_name: str = os.environ.get("BORUNTE_DEVICE", "realsense")
    enable_gpu: bool = os.environ.get("BORUNTE_ENABLE_GPU", "0") == "1"

    realsense_depth_width: int = _env_int("BORUNTE_RS_DEPTH_WIDTH", 1280)
    realsense_depth_height: int = _env_int("BORUNTE_RS_DEPTH_HEIGHT", 720)
    realsense_depth_fps: int = _env_int("BORUNTE_RS_DEPTH_FPS", 30)
    realsense_color_width: int = _env_int("BORUNTE_RS_COLOR_WIDTH", 1280)
    realsense_color_height: int = _env_int("BORUNTE_RS_COLOR_HEIGHT", 720)
    realsense_color_fps: int = _env_int("BORUNTE_RS_COLOR_FPS", 30)
    realsense_warmup_frames: int = _env_int("BORUNTE_RS_WARMUP_FRAMES", 12)

    robot_host: str = os.environ.get("BORUNTE_ROBOT_HOST", "192.168.4.4")
    robot_port: int = _env_int("BORUNTE_ROBOT_PORT", 9760)
    robot_timeout_s: float = _env_float("BORUNTE_ROBOT_TIMEOUT_S", 3.0)
    robot_keepalive: bool = os.environ.get("BORUNTE_ROBOT_KEEPALIVE", "1") == "1"

    robot_retry_delay_s: float = _env_float("BORUNTE_ROBOT_RETRY_DELAY_S", 0.2)
    robot_retry_attempts: int = max(1, _env_int("BORUNTE_ROBOT_RETRY_ATTEMPTS", 2))

    default_calibration_dataset: Optional[str] = os.environ.get(
        "BORUNTE_DEFAULT_CALIB_DATASET", None
    )
    default_calibration_output: Path = _env_path(
        "BORUNTE_CALIB_OUTPUT", _REPO_ROOT / "calibration_outputs"
    )
    default_calibration_intrinsics: str = os.environ.get(
        "BORUNTE_DEFAULT_CALIB_INTRINSICS", "color"
    )

    log_precision: int = _env_int("BORUNTE_NUMERIC_PRECISION", 4)

    def ensure_directories(self) -> None:
        for path in (self.data_root, self.captures_root, self.saves_root, self.logs_root):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
SETTINGS.ensure_directories()

__all__ = ["Settings", "SETTINGS"]
