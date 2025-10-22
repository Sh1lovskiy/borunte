# borunte/core/geometry/angles.py
"""Rotation matrix conversions: Euler angles, axis-angle, degrees/radians."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as SciRot


def axis_angle_from_matrix(R: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
    """Convert rotation matrix to axis-angle representation.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (angle_degrees, axis_unit_vector)
    """
    rot = SciRot.from_matrix(R)
    rotvec = rot.as_rotvec()

    angle_rad = float(np.linalg.norm(rotvec))
    angle_deg = angle_rad * 180.0 / np.pi

    if angle_rad > 1e-12:
        axis = rotvec / angle_rad
    else:
        axis = np.array([0.0, 0.0, 1.0])  # Arbitrary axis for zero rotation

    return angle_deg, axis


def euler_from_matrix(
    R: npt.NDArray[np.float64], seq: str = "xyz", degrees: bool = True
) -> npt.NDArray[np.float64]:
    """Convert rotation matrix to Euler angles.

    Args:
        R: 3x3 rotation matrix
        seq: Euler sequence (e.g., 'xyz', 'zyx')
        degrees: Return angles in degrees if True, radians if False

    Returns:
        Array of Euler angles in specified sequence
    """
    rot = SciRot.from_matrix(R)
    return rot.as_euler(seq, degrees=degrees)


def matrix_from_euler(
    angles: npt.NDArray[np.float64], seq: str = "xyz", degrees: bool = True
) -> npt.NDArray[np.float64]:
    """Convert Euler angles to rotation matrix.

    Args:
        angles: Array of Euler angles
        seq: Euler sequence (e.g., 'xyz', 'zyx')
        degrees: Input angles in degrees if True, radians if False

    Returns:
        3x3 rotation matrix
    """
    rot = SciRot.from_euler(seq, angles, degrees=degrees)
    return rot.as_matrix()


def print_rotation_analysis(R: npt.NDArray[np.float64]) -> None:
    """Print rotation matrix analysis in multiple representations.

    Args:
        R: 3x3 rotation matrix
    """
    angle_deg, axis = axis_angle_from_matrix(R)
    euler_zyx = euler_from_matrix(R, seq="zyx", degrees=True)
    euler_xyz = euler_from_matrix(R, seq="xyz", degrees=True)

    print(f"Axis-angle: {angle_deg:.6f} degrees")
    print(f"Axis: {axis.tolist()}")
    print(f"Euler ZYX (degrees): {[round(float(x), 6) for x in euler_zyx]}")
    print(f"Euler XYZ (degrees): {[round(float(x), 6) for x in euler_xyz]}")


__all__ = [
    "axis_angle_from_matrix",
    "euler_from_matrix",
    "matrix_from_euler",
    "print_rotation_analysis",
]
