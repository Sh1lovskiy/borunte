# utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np


# ============================== CORE DATATYPES ===============================


@dataclass(frozen=True)
class Pose:
    """Robot TCP pose in BASE (position in units of the capture, angles in degrees)."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


@dataclass(frozen=True)
class CameraDefaults:
    """Fallback pinhole intrinsics (used if no JSON/YAML is found next to capture)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> Tuple[int, int, float, float, float, float]:
        return (self.width, self.height, self.fx, self.fy, self.cx, self.cy)


@dataclass(frozen=True)
class HandEye:
    """
    Hand-eye transform parts.

    R : 3x3 rotation (numpy array, row-major)
    t : 3-vector translation in meters (numpy array)
    direction:
       - "tcp_cam"     : T_tcp_cam = [R|t] (TCP <- CAM)
       - "cam_tcp_inv" : provided (R,t) is CAM <- TCP, so T_tcp_cam = inv([R|t])
         (you can resolve the actual direction at runtime)
    """

    R: np.ndarray  # shape (3,3)
    t: np.ndarray  # shape (3,)
    direction: str = "tcp_cam"

    @property
    def T_cam_tcp(self) -> np.ndarray:
        """As [R|t] 4x4 for CAM <- TCP (useful if direction == 'cam_tcp_inv')."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.reshape(3)
        return T

    @property
    def T_tcp_cam(self) -> np.ndarray:
        """
        Always return TCP <- CAM 4x4 according to `direction`.
        If direction == 'tcp_cam'  -> [R|t]
        If direction == 'cam_tcp_inv' -> inv([R|t])
        """
        if self.direction == "tcp_cam":
            return self.T_cam_tcp
        elif self.direction == "cam_tcp_inv":
            return np.linalg.inv(self.T_cam_tcp)
        else:
            raise ValueError("HandEye.direction must be 'tcp_cam' or 'cam_tcp_inv'")


# ============================== PROJECT DEFAULTS =============================

# Where a single capture lives (code can override this)
# CAPTURE_ROOT: Path = Path(
#     "/home/sha/Documents/aitech-robotics/robohand/captures3/20250904_152945/ds_5/d1280x720_dec1_c1280x720"
# )

# CAPTURE_ROOT: Path = Path(
#     "/home/sha/Documents/aitech-robotics/robohand/captures/20250912_143541/ds_90/d1280x720_dec1_c1280x720"
# )

CAPTURE_ROOT: Path = Path(
    "/home/sha/Documents/aitech-robotics/borunte/captures/20251009_153839/ds_0/d1280x720_dec1_c1280x720"
)

# Subpaths & filenames inside a capture
IMG_DIR_NAME: str = ""  # e.g. imgs/000_rgb.png, imgs/000_depth.npy
POSES_JSON: str = "poses.json"  # robot TCP poses (BASE) per frame
INTRINSICS_JSON: str = "rs2_params.json"  # camera intrinsics (optional)

# Fallback intrinsics if JSON/YAML not present (do NOT do IO here).
CAMERA_FALLBACK = CameraDefaults(
    width=1280,
    height=720,
    fx=920.0,
    fy=920.0,
    cx=640.0,
    cy=360.0,
)

# Hand-eye: store as separate R and t (meters). Default direction is TCP <- CAM.
# HAND_EYE = HandEye(
#     R=np.array(
#         [
#             [0.999048, 0.00428, -0.00625],
#             [-0.00706, 0.99658, -0.00804],
#             [0.00423, 0.00895, 0.99629],
#         ],
#         dtype=np.float64,
#     ),
#     t=np.array([-0.036, -0.078, 0.006], dtype=np.float64),
#     direction="tcp_cam",
# )
# HAND_EYE = HandEye(
#     R=np.array(
#         [
#             [0.998429636, 0.030473142, 0.047006920],
#             [-0.029784066, 0.999439396, -0.015290601],
#             [-0.047446521, 0.013866532, 0.998777526],
#         ],
#         dtype=np.float64,
#     ),
#     t=np.array([-0.036863, -0.073258, 0.024962], dtype=np.float64),
#     direction="tcp_cam",
# )
HAND_EYE = HandEye(
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
# Work region AABB vertices in BASE.
# BBOX_POINTS = np.array(
#     [
#         [-0.57, -0.34, 0.46],
#         [-0.57, 0.20, 0.20],
#         [-0.30, 0.20, 0.20],
#         [-0.30, 0.20, 0.46],
#     ],
#     dtype=np.float64,
# )
# BBOX_POINTS = np.array(
#     [
#         [-0.1, -0.3, 0.002],
#         [-0.2, -0.5, 0.002],
#         [-0.4, 0.15, 0.013],
#         [-0.4, 0.15, 0.20],
#     ],
#     dtype=np.float64,
# )
BBOX_POINTS = np.array(
    [
        [521.0, 841.0, 504.0],
        [721.0, 841.0, 504.0],
        [721.0, 1041.0, 504.0],
        [521.0, 1041.0, 504.0],
    ],
    dtype=np.float64,
)
# Depth handling defaults that callers may use.
DEPTH_TRUNC: float = 2.5
DEPTH_UNIT_MODE: str = "auto"  # "auto" | "meters" | "millimeters"

# Pose interpretation defaults.
POSE_UNIT_MODE: str = "auto"  # "auto"|"meters"|"millimeters"
POSE_EULER_ORDER: str = "auto"  # "auto"|"XYZ"|"ZYX"
HAND_EYE_DIR: str = "auto"  # "auto"|"tcp_cam"|"cam_tcp_inv"
