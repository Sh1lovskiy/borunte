"""ChArUco-based hand-eye calibration for Borunte robot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from borunte.utils.logger import get_logger

_log = get_logger(__name__)

# ─────────────── CHARUCO BOARD CONFIG ───────────────


@dataclass
class CharucoConfig:
    """ChArUco board configuration."""

    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.04  # meters
    marker_length: float = 0.03  # meters
    dict_name: str = "4X4_50"
    min_corners: int = 4
    method: Literal["Tsai", "Park", "Horaud", "Andreff", "Daniilidis"] = "Tsai"


# ─────────────── CHARUCO DETECTION ───────────────


def detect_charuco(
    rgb_image: np.ndarray,
    config: CharucoConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Detect ChArUco corners in RGB image.

    Returns:
        (corners, ids, rejected) or (None, None, None) if detection fails
        - corners: (N, 1, 2) float32 array of corner pixel coordinates
        - ids: (N, 1) int32 array of corner IDs
        - rejected: list of rejected marker candidates
    """
    # Get ArUco dictionary
    dict_map = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "4X4_250": cv2.aruco.DICT_4X4_250,
        "4X4_1000": cv2.aruco.DICT_4X4_1000,
        "5X5_50": cv2.aruco.DICT_5X5_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250,
        "5X5_1000": cv2.aruco.DICT_5X5_1000,
        "6X6_50": cv2.aruco.DICT_6X6_50,
        "6X6_100": cv2.aruco.DICT_6X6_100,
        "6X6_250": cv2.aruco.DICT_6X6_250,
        "6X6_1000": cv2.aruco.DICT_6X6_1000,
        "7X7_50": cv2.aruco.DICT_7X7_50,
        "7X7_100": cv2.aruco.DICT_7X7_100,
        "7X7_250": cv2.aruco.DICT_7X7_250,
        "7X7_1000": cv2.aruco.DICT_7X7_1000,
    }

    if config.dict_name not in dict_map:
        _log.tag("CHARUCO", f"Unknown dictionary {config.dict_name}, using 4X4_50", level="warning")
        dict_id = cv2.aruco.DICT_4X4_50
    else:
        dict_id = dict_map[config.dict_name]

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length,
        config.marker_length,
        aruco_dict,
    )

    # Convert to grayscale if needed
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image

    # Detect ArUco markers
    detector_params = cv2.aruco.DetectorParameters()
    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=detector_params
    )

    if marker_ids is None or len(marker_ids) == 0:
        _log.tag("CHARUCO", "No ArUco markers detected", level="warning")
        return None, None, None

    # Interpolate ChArUco corners
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if ret < config.min_corners:
        _log.tag(
            "CHARUCO",
            f"Insufficient corners: {ret}/{config.min_corners}",
            level="warning",
        )
        return None, None, None

    _log.tag("CHARUCO", f"Detected {ret} corners")
    return charuco_corners, charuco_ids, rejected


# ─────────────── CAMERA POSE ESTIMATION ───────────────


def estimate_cam_pose(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    intrinsics: dict,
    config: CharucoConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Estimate camera pose relative to ChArUco board.

    Args:
        charuco_corners: (N, 1, 2) corner positions
        charuco_ids: (N, 1) corner IDs
        intrinsics: Camera intrinsics dict with 'color' or 'depth' sub-dict
        config: ChArUco board config

    Returns:
        (rvec, tvec, rmse) where:
        - rvec: (3, 1) rotation vector
        - tvec: (3, 1) translation vector
        - rmse: reprojection RMSE in pixels
        Returns (None, None, inf) if pose estimation fails
    """
    # Extract camera matrix and distortion
    if "color" in intrinsics:
        cam_params = intrinsics["color"]
    elif "depth" in intrinsics:
        cam_params = intrinsics["depth"]
    else:
        _log.tag("POSE", "No intrinsics found", level="error")
        return None, None, float("inf")

    K = np.array(
        [
            [cam_params["fx"], 0, cam_params["ppx"]],
            [0, cam_params["fy"], cam_params["ppy"]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # Distortion coefficients
    dist = np.array(cam_params.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)

    # Create board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length,
        config.marker_length,
        aruco_dict,
    )

    # Estimate pose
    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist, None, None
    )

    if not ret:
        _log.tag("POSE", "Failed to estimate pose", level="warning")
        return None, None, float("inf")

    # Calculate reprojection error
    obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
    img_points_reproj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
    img_points_reproj = img_points_reproj.reshape(-1, 2)
    img_points_actual = charuco_corners.reshape(-1, 2)
    errors = np.linalg.norm(img_points_reproj - img_points_actual, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    _log.tag("POSE", f"RMSE={rmse:.3f} px")
    return rvec, tvec, rmse


# ─────────────── HAND-EYE CALIBRATION ───────────────


def solve_handeye(
    tcp_poses: list[np.ndarray],
    cam_poses: list[np.ndarray],
    method: str = "Tsai",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve hand-eye calibration: TCP ↔ Camera transform.

    Args:
        tcp_poses: List of (4, 4) TCP→Base transforms
        cam_poses: List of (4, 4) Cam→Board transforms
        method: OpenCV calibrateHandEye method name

    Returns:
        (R_cam2tcp, t_cam2tcp, rmse) where:
        - R_cam2tcp: (3, 3) rotation matrix from camera to TCP frame
        - t_cam2tcp: (3,) translation vector from camera to TCP frame
        - rmse: root mean square error across all pairs
    """
    method_map = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    if method not in method_map:
        _log.tag("HANDEYE", f"Unknown method {method}, using Tsai", level="warning")
        cv_method = cv2.CALIB_HAND_EYE_TSAI
    else:
        cv_method = method_map[method]

    if len(tcp_poses) != len(cam_poses):
        raise ValueError(f"Mismatched poses: {len(tcp_poses)} TCP vs {len(cam_poses)} cam")

    if len(tcp_poses) < 3:
        raise ValueError(f"Need at least 3 pose pairs, got {len(tcp_poses)}")

    # Extract rotation and translation components
    R_gripper2base = [pose[:3, :3] for pose in tcp_poses]
    t_gripper2base = [pose[:3, 3:4] for pose in tcp_poses]
    R_target2cam = [pose[:3, :3] for pose in cam_poses]
    t_target2cam = [pose[:3, 3:4] for pose in cam_poses]

    # Solve hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv_method,
    )

    # Calculate RMSE across all pairs
    errors = []
    for i in range(len(tcp_poses)):
        # Compute reprojection error using AX=XB formulation
        # A = gripper2base[i], B = target2cam[i], X = cam2gripper
        A = tcp_poses[i]
        B = cam_poses[i]
        X = np.eye(4)
        X[:3, :3] = R_cam2gripper
        X[:3, 3] = t_cam2gripper.flatten()

        # AX should equal XB
        AX = A @ X
        XB = X @ B
        error = np.linalg.norm(AX[:3, 3] - XB[:3, 3])
        errors.append(error)

    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    _log.tag("HANDEYE", f"Solved with {len(tcp_poses)} pairs, RMSE={rmse:.6f}")

    return R_cam2gripper, t_cam2gripper.flatten(), rmse


# ─────────────── OVERLAY VISUALIZATION ───────────────


def draw_overlay(
    rgb_image: np.ndarray,
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
    intrinsics: dict | None = None,
    config: CharucoConfig | None = None,
) -> np.ndarray:
    """
    Draw detected ChArUco corners and coordinate axes on image.

    Args:
        rgb_image: Input RGB image
        charuco_corners: Detected corner positions
        charuco_ids: Detected corner IDs
        rvec: Rotation vector (for axes drawing)
        tvec: Translation vector (for axes drawing)
        intrinsics: Camera intrinsics (for axes drawing)
        config: ChArUco board config (for axes drawing)

    Returns:
        Annotated RGB image
    """
    overlay = rgb_image.copy()

    # Draw detected corners
    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids, (0, 255, 0))

    # Draw coordinate axes if pose available
    if (
        rvec is not None
        and tvec is not None
        and intrinsics is not None
        and config is not None
    ):
        if "color" in intrinsics:
            cam_params = intrinsics["color"]
        elif "depth" in intrinsics:
            cam_params = intrinsics["depth"]
        else:
            return overlay

        K = np.array(
            [
                [cam_params["fx"], 0, cam_params["ppx"]],
                [0, cam_params["fy"], cam_params["ppy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist = np.array(cam_params.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)

        # Draw axes (length = square_length)
        axis_length = config.square_length
        cv2.drawFrameAxes(overlay, K, dist, rvec, tvec, axis_length, 3)

    return overlay


# ─────────────── I/O HELPERS ───────────────


def save_handeye_result(
    output_path: Path,
    R: np.ndarray,
    t: np.ndarray,
    rmse: float,
    method: str,
    num_poses: int,
) -> None:
    """Save hand-eye calibration result to JSON."""
    result = {
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "rmse": rmse,
        "method": method,
        "num_poses": num_poses,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log.tag("SAVE", f"Hand-eye result → {output_path}")


__all__ = [
    "CharucoConfig",
    "detect_charuco",
    "estimate_cam_pose",
    "solve_handeye",
    "draw_overlay",
    "save_handeye_result",
]
