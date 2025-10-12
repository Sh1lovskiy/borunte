# borunte/config.py
"""Configuration objects for Borunte robot capture and network control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from config import SETTINGS, Settings


@dataclass(frozen=True)
class RealSenseStream:
    width: int
    height: int
    fps: int
    decimation: int = 1


@dataclass(frozen=True)
class RealSensePreviewConfig:
    view: str = "both"
    disparity_shift: Optional[int] = 10
    warmup_frames: int = SETTINGS.realsense_warmup_frames
    decimation: int = 1
    spatial_magnitude: int = 2
    spatial_smooth_alpha: float = 0.5
    spatial_smooth_delta: float = 20.0
    spatial_holes_fill: int = 0
    depth_viz_min_m: Optional[float] = None
    depth_viz_max_m: Optional[float] = None


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    depth: RealSenseStream
    color: RealSenseStream
    disparity_sweep: Tuple[int, ...] = (10, 40)


@dataclass(frozen=True)
class NetworkConfig:
    host: str
    port: int
    timeout_s: float
    keepalive: bool
    heartbeat_period_s: float = 5.0
    poll_period_s: float = 0.2
    alarm_print_period_s: float = 1.0
    wait_mode_s: float = 2.5
    wait_start_s: float = 6.0
    wait_move_s: float = 180.0
    wait_stop_s: float = 5.0
    retry_delay_s: float = SETTINGS.robot_retry_delay_s
    retry_attempts: int = SETTINGS.robot_retry_attempts


@dataclass(frozen=True)
class MotionLimits:
    speed_percent: float = 35.0
    position_tol_mm: float = 2.0
    angle_tol_deg: float = 2.0
    workspace_m: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (10.0, 370.0),
        (750.0, 1200.0),
        (400.0, 650.0),
    )
    tcp_down_uvw: Tuple[float, float, float] = (180.0, 0.0, -60.0)
    total_points: int = 20
    deviation_max_deg: float = 35.0
    grid_min_counts: Tuple[int, int, int] = (3, 3, 3)


@dataclass(frozen=True)
class WaypointConfig:
    use_waypoints: bool = True
    file: Path = SETTINGS.data_root / "waypoints.json"
    fmt: str = "json"


@dataclass(frozen=True)
class CaptureMode:
    interactive: bool = True
    auto_delay_s: float = 0.8


@dataclass(frozen=True)
class CalibrationLink:
    run_mode: str = "grid_with_calib"
    dataset_override: Optional[Path] = (
        Path(SETTINGS.default_calibration_dataset)
        if SETTINGS.default_calibration_dataset
        else None
    )
    stream: str = SETTINGS.default_calibration_intrinsics
    min_charuco_corners: int = 8
    save_overlay_images: bool = True
    reproj_rmse_min_px: float = 0.6
    reproj_rmse_max_px: float = 2.6
    reproj_rmse_step_px: float = 0.05
    min_sweep_frames: int = 10
    prior_t_cam2gripper_m: Tuple[float, float, float] = (-0.036, -0.078, 0.029)


@dataclass(frozen=True)
class CharucoConfig:
    square_m: float = 0.035
    marker_m: float = 0.026
    size: Tuple[int, int] = (8, 5)
    dictionary: str = "DICT_5X5_100"


@dataclass(frozen=True)
class RegisterMap:
    base_addr: int = 800
    length: int = 6


@dataclass(frozen=True)
class BorunteConfig:
    settings: Settings
    preview: RealSensePreviewConfig = field(
        default_factory=lambda: RealSensePreviewConfig(
            decimation=1,
        )
    )
    capture_profile: CaptureProfile = field(
        default_factory=lambda: CaptureProfile(
            name="d1280x720_dec1_c1280x720",
            depth=RealSenseStream(
                width=SETTINGS.realsense_depth_width,
                height=SETTINGS.realsense_depth_height,
                fps=SETTINGS.realsense_depth_fps,
                decimation=1,
            ),
            color=RealSenseStream(
                width=SETTINGS.realsense_color_width,
                height=SETTINGS.realsense_color_height,
                fps=SETTINGS.realsense_color_fps,
                decimation=1,
            ),
            disparity_sweep=(10, 40),
        )
    )
    network: NetworkConfig = field(
        default_factory=lambda: NetworkConfig(
            host=SETTINGS.robot_host,
            port=SETTINGS.robot_port,
            timeout_s=SETTINGS.robot_timeout_s,
            keepalive=SETTINGS.robot_keepalive,
        )
    )
    motion: MotionLimits = field(default_factory=MotionLimits)
    waypoints: WaypointConfig = field(default_factory=WaypointConfig)
    capture_mode: CaptureMode = field(default_factory=CaptureMode)
    calibration: CalibrationLink = field(default_factory=CalibrationLink)
    charuco: CharucoConfig = field(default_factory=CharucoConfig)
    register_map: RegisterMap = field(default_factory=RegisterMap)
    capture_root: Path = SETTINGS.captures_root
    use_waypoints: bool = True


def load_borunte_config(settings: Settings = SETTINGS) -> BorunteConfig:
    return BorunteConfig(settings=settings)


BORUNTE_CONFIG = load_borunte_config()

__all__ = ["BorunteConfig", "BORUNTE_CONFIG", "load_borunte_config", "RealSenseStream", "RealSensePreviewConfig", "CaptureProfile", "NetworkConfig", "MotionLimits", "WaypointConfig", "CaptureMode", "CalibrationLink", "CharucoConfig", "RegisterMap"]
