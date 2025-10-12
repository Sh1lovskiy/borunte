# calib/charuco.py
"""ChArUco detection and pose estimation utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    cv2 = None  # type: ignore

from utils.logger import Logger

from .config import CALIB_CONFIG

_log = Logger.get_logger()


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV (cv2) is required for ChArUco operations")


def get_dictionary():
    _require_cv2()
    ar = cv2.aruco
    name = CALIB_CONFIG.charuco.dictionary
    if not hasattr(ar, name):
        raise ValueError(f"Unknown ArUco dictionary {name}")
    return ar.getPredefinedDictionary(getattr(ar, name))


def make_board():
    _require_cv2()
    ar = cv2.aruco
    cfg = CALIB_CONFIG.charuco
    dic = get_dictionary()
    try:
        return ar.CharucoBoard_create(
            cfg.size[0], cfg.size[1], cfg.square_m, cfg.marker_m, dic
        )
    except Exception:
        return ar.CharucoBoard(
            (cfg.size[0], cfg.size[1]), cfg.square_m, cfg.marker_m, dic
        )


def board_object_corners(board) -> np.ndarray:
    for attr in ("chessboardCorners", "getChessboardCorners", "get_chessboard_corners"):
        if hasattr(board, attr):
            obj = getattr(board, attr)
            if callable(obj):
                obj = obj()
            return np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    raise AttributeError("CharucoBoard corners accessor not found")


def detect_markers(gray: np.ndarray):
    _require_cv2()
    ar = cv2.aruco
    try:
        params = ar.DetectorParameters()
        params.cornerRefinementMethod = getattr(ar, "CORNER_REFINE_SUBPIX", 1)
        detector = ar.ArucoDetector(get_dictionary(), params)
        return detector.detectMarkers(gray)
    except Exception:
        params = ar.DetectorParameters_create()
        return ar.detectMarkers(gray, get_dictionary(), parameters=params)


def interpolate_charuco(
    gray: np.ndarray,
    corners,
    ids,
    K: np.ndarray,
    dist: np.ndarray,
    board,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    _require_cv2()
    ar = cv2.aruco
    try:
        rc = ar.refineDetectedMarkers(
            image=gray,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=None,
            cameraMatrix=K,
            distCoeffs=dist,
        )
        corners, ids = rc[0], rc[1]
    except Exception:
        pass

    num, char_corners, char_ids = ar.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
        cameraMatrix=K,
        distCoeffs=dist,
    )
    if char_ids is None or char_corners is None:
        return 0, None, None
    return int(num), char_corners, char_ids


def refine_pnp(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    iters: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    _require_cv2()
    rvec = np.zeros((3, 1), np.float64)
    tvec = np.zeros((3, 1), np.float64)
    ok = True
    for _ in range(max(1, iters)):
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            break
    if not ok:
        raise RuntimeError("solvePnP failed")
    return rvec, tvec


def detect_pose(
    img_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    board,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    _require_cv2()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detect_markers(gray)
    if ids is None or len(ids) == 0:
        return None

    n, char_corners, char_ids = interpolate_charuco(gray, corners, ids, K, dist, board)
    if char_ids is None or char_corners is None:
        return None
    if n < CALIB_CONFIG.detection.min_charuco_corners:
        return None

    all_obj = board_object_corners(board)
    sel = np.asarray(char_ids, dtype=int).flatten()
    obj = all_obj[sel, :].astype(np.float64)
    img_pts = char_corners.reshape(-1, 2).astype(np.float64)

    rvec, tvec = refine_pnp(obj, img_pts, K, dist, iters=CALIB_CONFIG.solver.refinement_iters)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1))))

    R_tc, _ = cv2.Rodrigues(rvec)
    t_tc = tvec.reshape(3, 1).astype(np.float64)
    return R_tc.astype(np.float64), t_tc, sel.copy(), rmse, int(n)


__all__ = [
    "get_dictionary",
    "make_board",
    "board_object_corners",
    "detect_markers",
    "interpolate_charuco",
    "refine_pnp",
    "detect_pose",
]
