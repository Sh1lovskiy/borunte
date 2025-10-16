from __future__ import annotations

import sys
from typing import Final

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as SciRot

logger.remove()
logger.add(sys.stderr, format="{message}")

HAND_EYE_T: Final[list[float]] = [-0.086201, 0.028112, 0.037936]
HAND_EYE_R: Final[list[list[float]]] = [
    [0.009220435, 0.999913033, -0.009429223],
    [-0.999917557, 0.009135385, -0.009023528],
    [-0.008936603, 0.009511646, 0.999914829],
]
Rmat = np.asarray(HAND_EYE_R, dtype=np.float64)

rot = SciRot.from_matrix(Rmat)

rotvec = rot.as_rotvec()
# the magnitude of the rotation angle
angle_deg = float(np.linalg.norm(rotvec) * 180.0 / np.pi)
# direction of the axis
axis = (rotvec / (np.linalg.norm(rotvec) + 1e-12)).tolist()

eul_zyx = rot.as_euler("zyx", degrees=True).tolist()
eul_xyz = rot.as_euler("xyz", degrees=True).tolist()

logger.info(f"[degree] axis_angle_deg={angle_deg:.6f}\naxis={axis}")
logger.info(f"[degree] euler_zyx_deg={[round(x, 6) for x in eul_zyx]}")
logger.info(f"[degree] euler_xyz_deg={[round(x, 6) for x in eul_xyz]}")
