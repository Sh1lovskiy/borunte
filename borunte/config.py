# borunte/config.py
"""Centralized configuration for the Borunte robotics toolkit.

All constants, settings, and configuration dataclasses are defined here.
Modules should import from this single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Tuple

import numpy as np
import numpy.typing as npt

# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent


def _env_path(key: str, default: Path) -> Path:
    """Resolve path from environment variable with fallback."""
    value = os.getenv(key)
    return Path(value).expanduser() if value else default


def _env_str(key: str, default: str) -> str:
    """Resolve string from environment variable with fallback."""
    value = os.getenv(key)
    return value if value is not None else default


def _env_int(key: str, default: int) -> int:
    """Resolve integer from environment variable with fallback."""
    value = os.getenv(key)
    return int(value) if value is not None else default


def _env_float(key: str, default: float) -> float:
    """Resolve float from environment variable with fallback."""
    value = os.getenv(key)
    return float(value) if value is not None else default


# ============================================================================
# ROBOT COMMUNICATION CONSTANTS
# ============================================================================

ROBOT_POS_SCALE: Final[int] = 1000
ROBOT_ANG_SCALE: Final[int] = 1000
ROBOT_READ_RETRIES: Final[int] = 3
ROBOT_DEFAULT_HOST: Final[str] = "192.168.4.4"
ROBOT_DEFAULT_PORT: Final[int] = 9760
ROBOT_REGISTER_BASE_ADDR: Final[int] = 800
ROBOT_REGISTER_LENGTH: Final[int] = 6

# ============================================================================
# REALSENSE CAMERA CONSTANTS
# ============================================================================

RS_DEFAULT_WIDTH: Final[int] = 1280
RS_DEFAULT_HEIGHT: Final[int] = 720
RS_DEFAULT_FPS: Final[int] = 30
RS_DEFAULT_EXPOSURE: Final[float] = 400.0
RS_DEFAULT_GAIN: Final[float] = 16.0

# ============================================================================
# VISION PROCESSING CONSTANTS
# ============================================================================

VISION_MERGE_VOXEL_SIZE: Final[float] = 0.0025
VISION_FRAME_VOXEL_SIZE: Final[float] = 0.002
VISION_DEPTH_TRUNC: Final[float] = 2.5
VISION_NODE_RADIUS: Final[float] = 0.004
VISION_RASTER_RES_PX: Final[int] = 1024

# Default camera intrinsics (fallback)
CAMERA_DEFAULT_FX: Final[float] = 920.0
CAMERA_DEFAULT_FY: Final[float] = 920.0
CAMERA_DEFAULT_CX: Final[float] = 640.0
CAMERA_DEFAULT_CY: Final[float] = 360.0

# ============================================================================
# CALIBRATION CONSTANTS
# ============================================================================

CALIB_MIN_CHARUCO_CORNERS: Final[int] = 8
CALIB_REPROJ_RMSE_MIN: Final[float] = 0.6
CALIB_REPROJ_RMSE_MAX: Final[float] = 2.6
CALIB_REPROJ_RMSE_STEP: Final[float] = 0.05
CALIB_MIN_SWEEP_FRAMES: Final[int] = 10
CALIB_CHARUCO_SQUARE_M: Final[float] = 0.035
CALIB_CHARUCO_MARKER_M: Final[float] = 0.026
CALIB_CHARUCO_DICT: Final[str] = "DICT_5X5_100"

# ============================================================================
# MOTION & WORKSPACE CONSTANTS
# ============================================================================

MOTION_SPEED_PERCENT_DEFAULT: Final[float] = 35.0
MOTION_POSITION_TOL_MM: Final[float] = 2.0
MOTION_ANGLE_TOL_DEG: Final[float] = 2.0
MOTION_DEVIATION_MAX_DEG: Final[float] = 20.0

GRID_MIN_COUNTS: Final[Tuple[int, int, int]] = (3, 3, 3)
GRID_TOTAL_POINTS_DEFAULT: Final[int] = 20

# ============================================================================
# NETWORK TIMING CONSTANTS
# ============================================================================

NET_TIMEOUT_S: Final[float] = 5.0
NET_HEARTBEAT_PERIOD_S: Final[float] = 5.0
NET_POLL_PERIOD_S: Final[float] = 0.2
NET_ALARM_PRINT_PERIOD_S: Final[float] = 1.0
NET_WAIT_MODE_S: Final[float] = 2.5
NET_WAIT_START_S: Final[float] = 6.0
NET_WAIT_MOVE_S: Final[float] = 180.0
NET_WAIT_STOP_S: Final[float] = 5.0
NET_RETRY_DELAY_S: Final[float] = 2.0
NET_RETRY_ATTEMPTS: Final[int] = 2

# ============================================================================
# FILE NAMING CONSTANTS
# ============================================================================

FILENAME_POSES_JSON: Final[str] = "poses.json"
FILENAME_INTRINSICS_JSON: Final[str] = "rs2_params.json"
FILENAME_IMAGE_DIR: Final[str] = ""

# ============================================================================
# ENUMS
# ============================================================================


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DepthUnitMode(str, Enum):
    """Depth unit interpretation mode."""

    AUTO = "auto"
    METERS = "meters"
    MILLIMETERS = "millimeters"


class PoseUnitMode(str, Enum):
    """Pose unit interpretation mode."""

    AUTO = "auto"
    METERS = "meters"
    MILLIMETERS = "millimeters"


class HandEyeDirection(str, Enum):
    """Hand-eye calibration direction."""

    AUTO = "auto"
    TCP_CAM = "tcp_cam"
    CAM_TCP_INV = "cam_tcp_inv"


class CaptureView(str, Enum):
    """RealSense preview view mode."""

    BOTH = "both"
    COLOR = "color"
    DEPTH = "depth"


# ============================================================================
# DATACLASSES - Core Data Types
# ============================================================================


@dataclass(frozen=True)
class Pose:
    """Robot TCP pose in BASE frame."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return pose as 6-tuple."""
        return (self.x, self.y, self.z, self.rx, self.ry, self.rz)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> Tuple[int, int, float, float, float, float]:
        """Return intrinsics as 6-tuple."""
        return (self.width, self.height, self.fx, self.fy, self.cx, self.cy)


@dataclass(frozen=True)
class HandEyeTransform:
    """Hand-eye calibration transformation.

    Attributes:
        R: 3x3 rotation matrix
        t: 3-vector translation in meters
        direction: Interpretation of the transform direction
    """

    R: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    direction: Literal["tcp_cam", "cam_tcp_inv"] = "tcp_cam"

    def __post_init__(self) -> None:
        """Validate array shapes."""
        if self.R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must be length 3, got {self.t.shape}")

    @property
    def T_cam_tcp(self) -> npt.NDArray[np.float64]:
        """Return CAM <- TCP as 4x4 homogeneous matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def T_tcp_cam(self) -> npt.NDArray[np.float64]:
        """Return TCP <- CAM as 4x4, respecting direction."""
        if self.direction == "tcp_cam":
            return self.T_cam_tcp
        elif self.direction == "cam_tcp_inv":
            return np.linalg.inv(self.T_cam_tcp)
        else:
            raise ValueError(
                f"Invalid direction: {self.direction}, expected 'tcp_cam' or 'cam_tcp_inv'"
            )


# ============================================================================
# DATACLASSES - Configuration Sections
# ============================================================================


@dataclass(frozen=True)
class PathsConfig:
    """File system paths configuration."""

    data_root: Path
    captures_root: Path
    saves_root: Path
    logs_root: Path


@dataclass(frozen=True)
class RobotConfig:
    """Robot communication configuration."""

    host: str = ROBOT_DEFAULT_HOST
    port: int = ROBOT_DEFAULT_PORT
    timeout_s: float = NET_TIMEOUT_S
    connect_timeout: float = 5.0
    request_timeout: float = 2.0
    handover_timeout: float = 3.0


@dataclass(frozen=True)
class RealSenseConfig:
    """RealSense camera configuration."""

    width: int = RS_DEFAULT_WIDTH
    height: int = RS_DEFAULT_HEIGHT
    fps: int = RS_DEFAULT_FPS
    exposure: float = RS_DEFAULT_EXPOSURE
    gain: float = RS_DEFAULT_GAIN


@dataclass(frozen=True)
class NetworkConfig:
    """Network timing and retry configuration."""

    heartbeat_period_s: float = NET_HEARTBEAT_PERIOD_S
    poll_period_s: float = NET_POLL_PERIOD_S
    alarm_print_period_s: float = NET_ALARM_PRINT_PERIOD_S
    wait_mode_s: float = NET_WAIT_MODE_S
    wait_start_s: float = NET_WAIT_START_S
    wait_move_s: float = NET_WAIT_MOVE_S
    wait_stop_s: float = NET_WAIT_STOP_S
    retry_delay_s: float = NET_RETRY_DELAY_S
    retry_attempts: int = NET_RETRY_ATTEMPTS
    keepalive: bool = True


@dataclass(frozen=True)
class MotionConfig:
    """Motion limits and tolerances."""

    speed_percent: float = MOTION_SPEED_PERCENT_DEFAULT
    position_tol_mm: float = MOTION_POSITION_TOL_MM
    angle_tol_deg: float = MOTION_ANGLE_TOL_DEG
    deviation_max_deg: float = MOTION_DEVIATION_MAX_DEG
    workspace_m: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (10.0, 500.0),
        (700.0, 1000.0),
        (400.0, 700.0),
    )
    tcp_down_uvw: Tuple[float, float, float] = (180.0, 0.0, -90.0)


@dataclass(frozen=True)
class VisionConfig:
    """Vision processing configuration."""

    merge_voxel_size: float = VISION_MERGE_VOXEL_SIZE
    frame_voxel_size: float = VISION_FRAME_VOXEL_SIZE
    depth_trunc: float = VISION_DEPTH_TRUNC
    node_radius: float = VISION_NODE_RADIUS
    raster_res_px: int = VISION_RASTER_RES_PX
    fx: float = CAMERA_DEFAULT_FX
    fy: float = CAMERA_DEFAULT_FY
    cx: float = CAMERA_DEFAULT_CX
    cy: float = CAMERA_DEFAULT_CY
    visualize: bool = False
    visualize_per_frame: bool = False
    visualize_every_k: int = 10
    visualize_stages: Tuple[str, ...] = ("merged",)


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration configuration."""

    board_name: str = "charuco_6x9"
    square_size: float = CALIB_CHARUCO_SQUARE_M
    marker_size: float = CALIB_CHARUCO_MARKER_M
    max_reprojection_error: float = CALIB_REPROJ_RMSE_MAX
    min_charuco_corners: int = CALIB_MIN_CHARUCO_CORNERS
    reproj_rmse_min_px: float = CALIB_REPROJ_RMSE_MIN
    reproj_rmse_max_px: float = CALIB_REPROJ_RMSE_MAX
    reproj_rmse_step_px: float = CALIB_REPROJ_RMSE_STEP
    min_sweep_frames: int = CALIB_MIN_SWEEP_FRAMES
    output_intrinsics: str = FILENAME_INTRINSICS_JSON
    output_extrinsics: str = "extrinsics.json"
    save_overlay_images: bool = True


@dataclass(frozen=True)
class GridConfig:
    """Grid capture configuration."""

    total_points: int = GRID_TOTAL_POINTS_DEFAULT
    min_counts: Tuple[int, int, int] = GRID_MIN_COUNTS
    interactive: bool = True
    auto_delay_s: float = 0.8


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class Settings:
    """Main application configuration.

    All subsystem configurations are aggregated here.
    Instances are immutable (frozen=True) to prevent accidental mutation.
    """

    paths: PathsConfig
    robot: RobotConfig
    realsense: RealSenseConfig
    network: NetworkConfig
    motion: MotionConfig
    vision: VisionConfig
    calibration: CalibrationConfig
    grid: GridConfig

    # Default camera intrinsics (fallback when JSON not available)
    default_camera_intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(
            width=RS_DEFAULT_WIDTH,
            height=RS_DEFAULT_HEIGHT,
            fx=CAMERA_DEFAULT_FX,
            fy=CAMERA_DEFAULT_FY,
            cx=CAMERA_DEFAULT_CX,
            cy=CAMERA_DEFAULT_CY,
        )
    )

    # Default hand-eye transform
    default_hand_eye: HandEyeTransform = field(
        default_factory=lambda: HandEyeTransform(
            R=np.array(
                [
                    [0.999493516, -0.012213734, -0.029385980],
                    [-0.011574748, -0.999694969, 0.021817300],
                    [0.029643487, 0.021466114, 0.999330010],
                ],
                dtype=np.float64,
            ),
            t=np.array([-0.028112, -0.086201, 0.037936], dtype=np.float64),
            direction="tcp_cam",
        )
    )

    # Workspace bounding box points in BASE (mm)
    workspace_bbox_points: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [
                [521.0, 841.0, 504.0],
                [721.0, 841.0, 504.0],
                [721.0, 1041.0, 504.0],
                [521.0, 1041.0, 504.0],
            ],
            dtype=np.float64,
        )
    )


def get_settings() -> Settings:
    """Factory function to create Settings with environment variable overrides.

    Environment variables:
        BORUNTE_DATA_ROOT: Base data directory
        BORUNTE_CAPTURES_ROOT: Captures directory
        BORUNTE_SAVES_ROOT: Saves directory
        BORUNTE_LOGS_ROOT: Logs directory
        BORUNTE_ROBOT_HOST: Robot IP address
        BORUNTE_ROBOT_PORT: Robot TCP port
        BORUNTE_RS_WIDTH: RealSense width
        BORUNTE_RS_HEIGHT: RealSense height
        BORUNTE_RS_FPS: RealSense FPS
        BORUNTE_LOG_LEVEL: Logging level
    """
    data_root = _env_path("BORUNTE_DATA_ROOT", BASE_DIR / "data")
    captures_root = _env_path("BORUNTE_CAPTURES_ROOT", data_root / "captures")
    saves_root = _env_path("BORUNTE_SAVES_ROOT", data_root / "saves")
    logs_root = _env_path("BORUNTE_LOGS_ROOT", data_root / "logs")

    paths = PathsConfig(
        data_root=data_root,
        captures_root=captures_root,
        saves_root=saves_root,
        logs_root=logs_root,
    )

    robot = RobotConfig(
        host=_env_str("BORUNTE_ROBOT_HOST", ROBOT_DEFAULT_HOST),
        port=_env_int("BORUNTE_ROBOT_PORT", ROBOT_DEFAULT_PORT),
        timeout_s=_env_float("BORUNTE_ROBOT_TIMEOUT", NET_TIMEOUT_S),
        connect_timeout=_env_float("BORUNTE_ROBOT_CONNECT_TIMEOUT", 5.0),
        request_timeout=_env_float("BORUNTE_ROBOT_REQUEST_TIMEOUT", 2.0),
        handover_timeout=_env_float("BORUNTE_ROBOT_HANDOVER_TIMEOUT", 3.0),
    )

    realsense = RealSenseConfig(
        width=_env_int("BORUNTE_RS_WIDTH", RS_DEFAULT_WIDTH),
        height=_env_int("BORUNTE_RS_HEIGHT", RS_DEFAULT_HEIGHT),
        fps=_env_int("BORUNTE_RS_FPS", RS_DEFAULT_FPS),
        exposure=_env_float("BORUNTE_RS_EXPOSURE", RS_DEFAULT_EXPOSURE),
        gain=_env_float("BORUNTE_RS_GAIN", RS_DEFAULT_GAIN),
    )

    network = NetworkConfig()
    motion = MotionConfig()
    vision = VisionConfig()
    calibration = CalibrationConfig()
    grid = GridConfig()

    return Settings(
        paths=paths,
        robot=robot,
        realsense=realsense,
        network=network,
        motion=motion,
        vision=vision,
        calibration=calibration,
        grid=grid,
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Factory
    "get_settings",
    # Main config
    "Settings",
    # Config sections
    "PathsConfig",
    "RobotConfig",
    "RealSenseConfig",
    "NetworkConfig",
    "MotionConfig",
    "VisionConfig",
    "CalibrationConfig",
    "GridConfig",
    # Core data types
    "Pose",
    "CameraIntrinsics",
    "HandEyeTransform",
    # Enums
    "LogLevel",
    "DepthUnitMode",
    "PoseUnitMode",
    "HandEyeDirection",
    "CaptureView",
    # Constants (selected for external use)
    "ROBOT_POS_SCALE",
    "ROBOT_ANG_SCALE",
    "ROBOT_READ_RETRIES",
    "ROBOT_REGISTER_BASE_ADDR",
    "ROBOT_REGISTER_LENGTH",
    "FILENAME_POSES_JSON",
    "FILENAME_INTRINSICS_JSON",
    "BASE_DIR",
]
