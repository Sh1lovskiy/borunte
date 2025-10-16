# config.py
"""Centralised configuration model for the Borunte toolkit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_path(key: str, default: Path) -> Path:
    value = os.getenv(key)
    if value:
        return Path(value).expanduser()
    return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value is not None else default


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value is not None else default


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None else default


@dataclass(slots=True)
class PathSettings:
    data_root: Path
    captures_root: Path
    saves_root: Path
    logs_root: Path


@dataclass(slots=True)
class RobotSettings:
    host: str
    port: int
    connect_timeout: float
    request_timeout: float
    handover_timeout: float


@dataclass(slots=True)
class RealSenseSettings:
    width: int
    height: int
    fps: int
    exposure: float
    gain: float


@dataclass(slots=True)
class CalibrationSettings:
    board_name: str
    square_size: float
    marker_size: float
    max_reprojection_error: float
    output_intrinsics: str
    output_extrinsics: str


@dataclass(slots=True)
class Settings:
    paths: PathSettings
    robot: RobotSettings
    realsense: RealSenseSettings
    calibration: CalibrationSettings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_root = _env_path("BORUNTE_DATA_ROOT", BASE_DIR / "data")
    captures_root = _env_path("BORUNTE_CAPTURES_ROOT", data_root / "captures")
    saves_root = _env_path("BORUNTE_SAVES_ROOT", data_root / "saves")
    logs_root = _env_path("BORUNTE_LOGS_ROOT", data_root / "logs")
    paths = PathSettings(
        data_root=data_root,
        captures_root=captures_root,
        saves_root=saves_root,
        logs_root=logs_root,
    )
    robot = RobotSettings(
        host=_env_str("BORUNTE_ROBOT_HOST", "192.168.0.10"),
        port=_env_int("BORUNTE_ROBOT_PORT", 502),
        connect_timeout=_env_float("BORUNTE_ROBOT_CONNECT_TIMEOUT", 5.0),
        request_timeout=_env_float("BORUNTE_ROBOT_REQUEST_TIMEOUT", 2.0),
        handover_timeout=_env_float("BORUNTE_ROBOT_HANDOVER_TIMEOUT", 3.0),
    )
    realsense = RealSenseSettings(
        width=_env_int("BORUNTE_RS_WIDTH", 1280),
        height=_env_int("BORUNTE_RS_HEIGHT", 720),
        fps=_env_int("BORUNTE_RS_FPS", 30),
        exposure=_env_float("BORUNTE_RS_EXPOSURE", 400.0),
        gain=_env_float("BORUNTE_RS_GAIN", 16.0),
    )
    calibration = CalibrationSettings(
        board_name=_env_str("BORUNTE_CALIB_BOARD", "charuco_6x9"),
        square_size=_env_float("BORUNTE_CALIB_SQUARE", 0.03),
        marker_size=_env_float("BORUNTE_CALIB_MARKER", 0.02),
        max_reprojection_error=_env_float("BORUNTE_CALIB_REPROJ", 0.5),
        output_intrinsics=_env_str("BORUNTE_CALIB_INTRINSICS", "intrinsics.json"),
        output_extrinsics=_env_str("BORUNTE_CALIB_EXTRINSICS", "extrinsics.json"),
    )
    return Settings(
        paths=paths,
        robot=robot,
        realsense=realsense,
        calibration=calibration,
    )


__all__ = [
    "CalibrationSettings",
    "PathSettings",
    "RealSenseSettings",
    "RobotSettings",
    "Settings",
    "get_settings",
]
