# borunte/config.py
"""Global configuration constants for the Borunte RemoteMonitor runner."""

from __future__ import annotations
from typing import Tuple, List, Dict, Any


# --- Pipeline selection ---
# grid            run motion and capture only
# grid_with_calib run motion and capture, then offline calibration
# calib_only      run offline calibration only
RUN_MODE: str = "grid_with_calib"  # "grid" | "grid_with_calib" | "calib_only"

# Optional dataset path for offline calibration when RUN_MODE is calib_only.
# If None, the program will try to infer the latest session automatically.
CALIB_DATA_DIR: str | None = "captures/20251009_145701/ds_10/d1280x720_dec1_c1280x720"
# Which intrinsics to use from rs2_params.json
USE_STREAM: str = "color"  # "color" or "depth"

# ChArUco board
CHARUCO_SQUARE_M: float = 0.035
CHARUCO_MARKER_M: float = 0.026
CHARUCO_SIZE: Tuple[int, int] = (8, 5)  # squaresX, squaresY
ARUCO_DICT_NAME: str = "DICT_5X5_100"

# Detection and validation
MIN_CHARUCO_CORNERS: int = 8
SAVE_OVERLAY_IMAGES: bool = True
REPROJ_RMSE_MIN_PX: float = 0.6
REPROJ_RMSE_MAX_PX: float = 2.6
REPROJ_RMSE_STEP_PX: float = 0.05
MIN_SWEEP_FRAMES: int = 10

# Prior translation cam to gripper for method selection
PRIOR_T_CAM2GRIPPER_M: Tuple[float, float, float] = (
    -0.036,
    -0.078,
    0.029,
)

# Network
IP: str = "192.168.4.4"
PORT: int = 9760
TIMEOUT_S: float = 3.0  # per-request socket timeout
HEARTBEAT_PERIOD_S: float = (
    5.0  # manual says heartbeat period should be less than or equal to 10 s
)
SOCKET_KEEPALIVE: bool = True

# Polling and waits
POLL_S: float = 0.2
ALARM_REPRINT_S: float = 1.0
WAIT_MODE_S: float = 2.5  # wait for Single Cycle mode latch
WAIT_START_S: float = 6.0  # wait for isMoving to become 1
WAIT_MOVE_S: float = 180.0  # maximum motion time
WAIT_STOP_S: float = 5.0  # wait for Stop during release

# Motion
SPEED_PERCENT: float = 35.0  # allowed range 0..100
POS_TOL_MM: float = 2.0
ANG_TOL_DEG: float = 2.0

# Workspace in millimeters: ((X0, X1), (Y0, Y1), (Z0, Z1))
WORKSPACE_M: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (10.0, 370.0),  # X
    (750.0, 1200.0),  # Y
    (400.0, 650.0),  # Z
)

# Grid shaping
# minimum nodes per axis when building a grid
GRID_MIN_COUNTS: Tuple[int, int, int] = (
    3,
    3,
    3,
)

# Orientation base and jitter
TCP_DOWN_UVW: Tuple[float, float, float] = (180.0, 0.0, -60.0)
TOTAL_POINTS: int = 20
DEV_MAX_DEG: float = 35.0

# Registers
ADDR_BASE: int = 800
ADDR_LEN: int = 6

# RealSense preview configuration
PREVIEW_VIEW: str = "both"  # one of: "depth", "rgb", "both"
PREVIEW_DEPTH_W: int = 1280
PREVIEW_DEPTH_H: int = 720
PREVIEW_DEPTH_FPS: int = 30
PREVIEW_COLOR_W: int = 1280
PREVIEW_COLOR_H: int = 720
PREVIEW_COLOR_FPS: int = 30
PREVIEW_DECIMATION: int = 1
PREVIEW_DISPARITY_SHIFT: int = 10
WARMUP_FRAMES: int = 12

# Disparity shift sweep for interactive capture at each grid point
DISPARITY_SHIFT_VALUES: List[int] = list(range(10, 46, 30))  # example: [10, 40]

# Single capture profile
CAPTURE_CONFIG: Dict[str, Any] = {
    "name": "d1280x720_dec1_c1280x720",
    "depth": {"w": 1280, "h": 720, "fps": 30, "decimation": 1},
    "color": {"w": 1280, "h": 720, "fps": 30},
}

# Depth post-filters
SPATIAL_MAG: int = 2
SPATIAL_SMOOTH_ALPHA: float = 0.5
SPATIAL_SMOOTH_DELTA: float = 20.0
SPATIAL_HOLES_FILL: int = 0

# Depth visualization range (None means auto)
DEPTH_VIZ_MIN_M = None
DEPTH_VIZ_MAX_M = None

# Capture sessions
CAPTURE_ROOT_DIR: str = (
    "captures"  # a timestamped subfolder will be created under this root
)

# Selection between grid and waypoints
USE_WAYPOINTS: bool = True  # True to follow waypoints, False to use a generated grid
WAYPOINTS_FILE: str = "waypoints.json"  # or "waypoints.csv"
WAYPOINTS_FMT: str = "json"  # "json" or "csv"

# Capture mode
CAPTURE_INTERACTIVE: bool = (
    True  # True means preview with SPACE to capture, False means auto capture
)
AUTO_CAPTURE_DELAY_S: float = (
    0.8  # delay before auto capture when CAPTURE_INTERACTIVE is False
)
