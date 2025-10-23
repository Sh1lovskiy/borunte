#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_handeye.py

Hand Eye calibration from an already captured dataset on disk.
No TCP. No live camera.

Input directory example
/home/sha/Documents/aitech-robotics/borunte/captures/20250922_143639/ds_0/d1280x720_dec1_c1280x720
Files inside
000_rgb.png
000_depth.npy
poses.json
rs2_params.json

What this script does
1. Reads intrinsics from rs2_params.json
2. Scans all color images like NNN_rgb.png and matches depth NNN_depth.npy if present
3. Detects ChArUco on each color image and refines pose with solvePnP
4. Loads robot poses from poses.json by index
5. Runs OpenCV calibrateHandEye with 5 methods
6. Writes reports
   - handeye_report_no_validation.txt
   - handeye_report_validated.txt
   Also writes
   - charuco_detections.json
   - handeye_sweep_results.json
   - optional overlays NNN_rgb_charuco.png if SAVE_OVERLAY_IMAGES is True
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# ========================== CONFIG ==========================
# Directory with images and json files. Can be overridden by CLI arg 1.
DATA_DIR = Path(
    "/home/sha/Documents/aitech-robotics/borunte/data/captures/20251023_133254/captures/"
)

# Use intrinsics from this stream
USE_STREAM = "color"  # color or depth

# ChArUco board params
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

# Prior translation cam to gripper for selection
PRIOR_T_CAM2GRIPPER_M: Tuple[float, float, float] = (-0.036, -0.078, 0.029)
# ===========================================================


# ---------------------- IO helpers ----------------------
def _load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_pairs(dir_path: Path) -> List[int]:
    idxs = []
    for p in sorted(dir_path.glob("*_0_rgb.png")):
        stem = p.name.replace("_0_rgb.png", "")
        try:
            idxs.append(int(stem))
        except Exception:
            continue
    return sorted(idxs)


def _load_intrinsics_from_rs2_params(
    rs_params_path: Path, stream: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    data = _load_json(rs_params_path)
    if stream.lower() == "color":
        intr = data["intrinsics"]["color"]
        stream_key = "color"
    elif stream.lower() == "depth":
        intr = data["intrinsics"]["depth"]
        stream_key = "depth"
    else:
        raise ValueError("USE_STREAM must be color or depth")
    K = np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(
        list(intr.get("coeffs", [0, 0, 0, 0, 0])[:5]), dtype=np.float64
    ).ravel()
    s = data["streams"][stream_key]
    wh = (int(s["width"]), int(s["height"]))
    return K, dist, wh


def _get_versions() -> Tuple[str, str]:
    cvv = getattr(cv2, "__version__", "unknown")
    try:
        import pyrealsense2 as rs  # only for version string if present

        rsv = getattr(rs, "__version__", "unknown")
    except Exception:
        rsv = "unknown"
    return cvv, rsv


# ---------------------- ChArUco helpers ----------------------
def _get_aruco_dictionary(name: str):
    ar = cv2.aruco
    if not hasattr(ar, name):
        raise ValueError(f"Unknown ArUco dictionary {name}")
    return ar.getPredefinedDictionary(getattr(ar, name))


def _make_charuco_board(
    size_xy: Tuple[int, int], square_m: float, marker_m: float, dictionary
):
    ar = cv2.aruco
    try:
        board = ar.CharucoBoard_create(
            size_xy[0], size_xy[1], square_m, marker_m, dictionary
        )
    except Exception:
        board = ar.CharucoBoard(
            (size_xy[0], size_xy[1]), square_m, marker_m, dictionary
        )
    return board


def _charuco_board_object_corners(board) -> np.ndarray:
    if hasattr(board, "chessboardCorners"):
        obj = board.chessboardCorners
        return np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    if hasattr(board, "getChessboardCorners"):
        obj = board.getChessboardCorners()
        return np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    if hasattr(board, "get_chessboard_corners"):
        obj = board.get_chessboard_corners()
        return np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    raise AttributeError("CharucoBoard corners accessor not found")


@dataclass
class PnPResult:
    idx: int
    R_tc: np.ndarray
    t_tc: np.ndarray
    rmse_px: float
    n_corners: int
    used_ids: List[int]
    shaky: bool
    coverage_w_frac: float


def _board_coverage_fraction_px(
    char_corners: np.ndarray, img_wh: Tuple[int, int]
) -> float:
    if char_corners is None or len(char_corners) == 0:
        return 0.0
    pts = char_corners.reshape(-1, 2)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    width = max(1.0, float(img_wh[0]))
    return float(max(0.0, min(1.0, (xmax - xmin) / width)))


def _detect_charuco_and_pose(
    img_bgr: np.ndarray,
    board,
    K: np.ndarray,
    dist: np.ndarray,
    refine_iters: int = 2,
    min_corners: int = MIN_CHARUCO_CORNERS,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ar = cv2.aruco

    try:
        params = ar.DetectorParameters()
        params.cornerRefinementMethod = getattr(ar, "CORNER_REFINE_SUBPIX", 1)
        detector = ar.ArucoDetector(_get_aruco_dictionary(ARUCO_DICT_NAME), params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except Exception:
        params = ar.DetectorParameters_create()
        corners, ids, rejected = ar.detectMarkers(
            gray, _get_aruco_dictionary(ARUCO_DICT_NAME), parameters=params
        )

    if ids is None or len(ids) == 0:
        return None

    try:
        rc = ar.refineDetectedMarkers(
            image=gray,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejected,
            cameraMatrix=K,
            distCoeffs=dist,
        )
        corners, ids, rejected = rc[0], rc[1], rc[2]
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
        return None
    n = int(char_corners.shape[0])
    if n < min_corners:
        return None

    try:
        ok, rvec, tvec = ar.estimatePoseCharucoBoard(
            char_corners, char_ids, board, K, dist
        )
    except Exception:
        rvec = np.zeros((3, 1), np.float64)
        tvec = np.zeros((3, 1), np.float64)
        ok = ar.estimatePoseCharucoBoard(
            char_corners, char_ids, board, K, dist, rvec, tvec, False
        )
    if not bool(ok):
        return None

    all_obj = _charuco_board_object_corners(board)
    sel = np.asarray(char_ids, dtype=int).flatten()
    obj = all_obj[sel, :].astype(np.float64)
    img_pts = char_corners.reshape(-1, 2).astype(np.float64)

    rvec_ref = rvec.copy()
    tvec_ref = tvec.copy()
    for _ in range(max(1, refine_iters)):
        ok2, rvec_ref, tvec_ref = cv2.solvePnP(
            objectPoints=obj,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec_ref,
            tvec=tvec_ref,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok2:
            break

    proj, _ = cv2.projectPoints(obj, rvec_ref, tvec_ref, K, dist)
    proj = proj.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1))))

    R_tc, _ = cv2.Rodrigues(rvec_ref)
    t_tc = tvec_ref.reshape(3, 1).astype(np.float64)
    return R_tc.astype(np.float64), t_tc, sel.copy(), rmse, n, char_corners, char_ids


# ---------------------- Hand Eye helpers ----------------------
def _build_lists_for_handeye(
    pnp_results: List[PnPResult],
    poses_json_path: Path,
    exclude_shaky: bool = True,
):
    with open(poses_json_path, "r", encoding="utf-8") as f:
        poses_dict = json.load(f)

    valid_idxs = set(r.idx for r in pnp_results if (not exclude_shaky or not r.shaky))
    idxs = sorted(set(int(k) for k in poses_dict.keys()) & valid_idxs)
    if not idxs:
        raise RuntimeError("No overlapping indices between poses and PnP results")

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    for idx in idxs:
        pose = poses_dict[f"{idx:03d}"]
        R_bg = R.from_euler(
            "xyz", [pose["rx"], pose["ry"], pose["rz"]], degrees=True
        ).as_matrix()
        t_bg = np.array(
            [[pose["x"] / 1000.0], [pose["y"] / 1000.0], [pose["z"] / 1000.0]],
            dtype=np.float64,
        )
        R_gripper2base.append(R_bg.astype(np.float64))
        t_gripper2base.append(t_bg.astype(np.float64))

        pr = next(r for r in pnp_results if r.idx == idx)
        R_target2cam.append(pr.R_tc)
        t_target2cam.append(pr.t_tc)

    return idxs, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def run_handeye_all_methods_given_lists(
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
) -> dict:
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    out = {"methods": {}}
    for name, mcode in methods.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=mcode
        )
        euler = R.from_matrix(R_c2g).as_euler("xyz", degrees=True).tolist()
        t_list = np.asarray(t_c2g).reshape(3).tolist()
        out["methods"][name] = {
            "R_cam2gripper": R_c2g.tolist(),
            "t_cam2gripper_m": t_list,
            "euler_xyz_deg": euler,
        }
    return out


def _choose_best_by_translation_prior(
    he_methods: dict, t_prior: np.ndarray
) -> tuple[str, dict, float]:
    best_name = None
    best_entry = None
    best_dist = float("inf")
    for name, entry in he_methods.items():
        t = np.asarray(entry["t_cam2gripper_m"], dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(t - t_prior))
        if d < best_dist:
            best_dist = d
            best_name = name
            best_entry = entry
    assert best_name is not None and best_entry is not None
    return best_name, best_entry, best_dist


def _pose_diversity_stats(
    Rg2b: List[np.ndarray], tg2b: List[np.ndarray]
) -> Dict[str, float]:
    T = np.hstack(tg2b).T
    t_std = float(np.linalg.norm(T.std(axis=0)))
    euls = np.array([R.from_matrix(Ri).as_euler("xyz", degrees=True) for Ri in Rg2b])
    e_std = float(np.linalg.norm(euls.std(axis=0)))
    rel_axes = []
    for i in range(1, len(Rg2b)):
        dR = R.from_matrix(Rg2b[i - 1]).inv() * R.from_matrix(Rg2b[i])
        aa = dR.as_rotvec()
        rel_axes.append(aa)
    if len(rel_axes) >= 2:
        A = np.array(rel_axes)
        cov = np.cov(A.T)
        svals = np.linalg.svd(cov, compute_uv=False)
        rank_like = float(np.sum(svals > 1e-6))
    else:
        rank_like = 0.0
    return {
        "trans_spread_norm": t_std,
        "euler_spread_norm_deg": e_std,
        "rel_rot_rank": rank_like,
    }


def _write_handeye_report(
    out_path: Path,
    title: str,
    frames_used: int,
    pnp_rmses: List[float],
    he_methods: dict,
    best_name: str,
    best_entry: dict,
    best_dist: float,
    indices_used: List[int],
    meta: Dict[str, Any],
    diversity: Dict[str, float] | None = None,
) -> None:
    mean_rmse = float(np.mean(pnp_rmses)) if pnp_rmses else float("nan")
    med_rmse = float(np.median(pnp_rmses)) if pnp_rmses else float("nan")
    min_rmse = float(np.min(pnp_rmses)) if pnp_rmses else float("nan")
    max_rmse = float(np.max(pnp_rmses)) if pnp_rmses else float("nan")

    cvv, rsv = _get_versions()
    lines: List[str] = []
    lines.append(title)
    lines.append(f"OpenCV={cvv}  pyrealsense2={rsv}")
    lines.append(f"rs2_params_path = {meta.get('rs2_params_path','')}")
    lines.append("")
    lines.append("Intrinsics used for PnP")
    K = meta["K"]
    dist = meta["dist"]
    wh = meta["image_size"]
    lines.append(
        f"K = [[{K[0,0]:.6f}, 0.000000, {K[0,2]:.6f}], [0.000000, {K[1,1]:.6f}, {K[1,2]:.6f}], [0, 0, 1]]"
    )
    lines.append(f"dist = [{', '.join(f'{float(x):.6f}' for x in dist)}]")
    lines.append(f"image_size = {wh[0]}x{wh[1]}")
    lines.append("")
    lines.append("ChArUco reprojection error in pixels")
    lines.append(
        f"frames_ok={frames_used}  frames_total={meta.get('frames_total',0)}  frames_rejected={meta.get('frames_rejected',0)}"
    )
    lines.append(
        f"rmse_mean={mean_rmse:.4f} rmse_median={med_rmse:.4f} min={min_rmse:.4f} max={max_rmse:.4f}"
    )
    lines.append("")
    lines.append("Hand Eye results cam to gripper")
    for name, res in he_methods.items():
        t = res["t_cam2gripper_m"]
        e = res["euler_xyz_deg"]
        lines.append(f"[{name}] t(m) = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        lines.append(f"[{name}] euler_xyz_deg = [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}]")
        lines.append("")
    lines.append(f"indices_used = {indices_used}")
    lines.append("")
    lines.append("Best by translation prior")
    t = best_entry["t_cam2gripper_m"]
    Rm = np.asarray(best_entry["R_cam2gripper"], dtype=np.float64)
    lines.append(f"best_method = {best_name}")
    lines.append(f"prior_t = {PRIOR_T_CAM2GRIPPER_M}")
    lines.append(f"distance_to_prior_m = {best_dist:.6f}")
    lines.append(f"best_t_cam2gripper_m = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
    lines.append("best_R_cam2gripper =")
    for r in Rm:
        lines.append(f"[{r[0]: .9f}, {r[1]: .9f}, {r[2]: .9f}]")
    lines.append("")
    if diversity is not None:
        lines.append("Pose diversity diagnostic")
        lines.append(f"trans_spread_norm_m = {diversity['trans_spread_norm']:.6f}")
        lines.append(
            f"euler_spread_norm_deg = {diversity['euler_spread_norm_deg']:.6f}"
        )
        lines.append(f"relative_rotation_rank {diversity['rel_rot_rank']:.1f}")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------- Main processing ----------------------
def run_offline(data_dir: Path) -> Path:
    rs_params_path = data_dir / "rs2_params.json"
    poses_json_path = data_dir / "poses.json"
    if not rs_params_path.exists():
        raise FileNotFoundError(f"Missing {rs_params_path}")
    if not poses_json_path.exists():
        raise FileNotFoundError(f"Missing {poses_json_path}")

    K, dist, expect_wh = _load_intrinsics_from_rs2_params(rs_params_path, USE_STREAM)

    dictionary = _get_aruco_dictionary(ARUCO_DICT_NAME)
    board = _make_charuco_board(
        CHARUCO_SIZE, CHARUCO_SQUARE_M, CHARUCO_MARKER_M, dictionary
    )

    indices = _find_pairs(data_dir)
    if not indices:
        raise RuntimeError("No files like NNN_rgb.png found")

    results: List[PnPResult] = []
    det_log: Dict[str, Any] = {}
    total = 0

    for idx in indices:
        total += 1
        img_path = data_dir / f"{idx:03d}_0_rgb.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            det_log[f"{idx:03d}"] = {"found": False, "reason": "read_fail"}
            continue
        if (img.shape[1], img.shape[0]) != expect_wh:
            det_log[f"{idx:03d}"] = {"found": False, "reason": "size_mismatch"}
            continue

        det = _detect_charuco_and_pose(img, board, K, dist, refine_iters=2)
        if det is None:
            det_log[f"{idx:03d}"] = {"found": False}
            continue

        R_tc, t_tc, used_ids, rmse, n_used, ch_corners, ch_ids = det
        coverage = _board_coverage_fraction_px(ch_corners, expect_wh)

        if SAVE_OVERLAY_IMAGES:
            vis = img.copy()
            try:
                params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(dictionary, params)
                mk_corners, mk_ids, _ = detector.detectMarkers(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                )
            except Exception:
                params = cv2.aruco.DetectorParameters_create()
                mk_corners, mk_ids, _ = cv2.aruco.detectMarkers(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dictionary, parameters=params
                )
            if mk_ids is not None and len(mk_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, mk_corners, mk_ids)
            try:
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
            except Exception:
                pass
            try:
                cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(R_tc)[0], t_tc, 0.05)
            except Exception:
                pass
            cv2.putText(
                vis,
                f"RMSE={rmse:.3f}px N={n_used} covW={coverage:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(str(img_path.with_name(f"{img_path.stem}_charuco.png")), vis)

        shaky = (
            bool(_load_json(poses_json_path).get(f"{idx:03d}", {}).get("shaky", False))
            if poses_json_path.exists()
            else False
        )

        results.append(
            PnPResult(
                idx=idx,
                R_tc=R_tc,
                t_tc=t_tc,
                rmse_px=rmse,
                n_corners=int(n_used),
                used_ids=[int(x) for x in used_ids.tolist()],
                shaky=shaky,
                coverage_w_frac=float(coverage),
            )
        )
        det_log[f"{idx:03d}"] = {
            "found": True,
            "rmse_px": float(rmse),
            "n_corners": int(n_used),
            "ids": [int(x) for x in used_ids.tolist()],
            "shaky": bool(shaky),
            "coverage_w_frac": float(coverage),
        }

    results.sort(key=lambda r: r.idx)

    with open(data_dir / "charuco_detections.json", "w", encoding="utf-8") as f:
        json.dump(det_log, f, indent=2)

    if not results:
        raise RuntimeError("No frames with valid ChArUco pose")

    # Exclude shaky frames if flagged
    stable = [r for r in results if not r.shaky]
    if len(stable) < 3:
        stable = results[:]  # fallback

    t_prior = np.asarray(PRIOR_T_CAM2GRIPPER_M, dtype=np.float64).reshape(3)

    idxs_all, Rg2b_all, tg2b_all, Rt2c_all, tt2c_all = _build_lists_for_handeye(
        stable, poses_json_path, exclude_shaky=True
    )
    he_all = run_handeye_all_methods_given_lists(Rg2b_all, tg2b_all, Rt2c_all, tt2c_all)
    best_name_all, best_entry_all, best_dist_all = _choose_best_by_translation_prior(
        he_all["methods"], t_prior
    )
    diversity_all = _pose_diversity_stats(Rg2b_all, tg2b_all)

    meta_common = {
        "rs2_params_path": str(rs_params_path),
        "K": K,
        "dist": dist,
        "image_size": expect_wh,
        "frames_total": total,
        "frames_rejected": total - len(idxs_all),
    }

    _write_handeye_report(
        out_path=data_dir / "handeye_report_no_validation.txt",
        title="Hand Eye report no RMSE validation shaky filtered",
        frames_used=len(idxs_all),
        pnp_rmses=[r.rmse_px for r in stable],
        he_methods=he_all["methods"],
        best_name=best_name_all,
        best_entry=best_entry_all,
        best_dist=best_dist_all,
        indices_used=idxs_all,
        meta=meta_common,
        diversity=diversity_all,
    )

    sweep_results: List[Dict[str, Any]] = []
    thr = REPROJ_RMSE_MIN_PX
    while thr <= REPROJ_RMSE_MAX_PX + 1e-9:
        valid_pnp = [r for r in stable if r.rmse_px <= thr]
        if len(valid_pnp) >= MIN_SWEEP_FRAMES:
            idxs_v, Rg2b_v, tg2b_v, Rt2c_v, tt2c_v = _build_lists_for_handeye(
                valid_pnp, poses_json_path, exclude_shaky=True
            )
            he_v = run_handeye_all_methods_given_lists(Rg2b_v, tg2b_v, Rt2c_v, tt2c_v)
            best_name_v, best_entry_v, best_dist_v = _choose_best_by_translation_prior(
                he_v["methods"], t_prior
            )
            sweep_results.append(
                {
                    "threshold": float(round(thr, 3)),
                    "indices_used": idxs_v,
                    "he": he_v,
                    "best_name": best_name_v,
                    "best_entry": best_entry_v,
                    "best_dist": best_dist_v,
                    "pnp_rmses": [r.rmse_px for r in valid_pnp],
                    "frames_used": len(idxs_v),
                    "diversity": _pose_diversity_stats(Rg2b_v, tg2b_v),
                }
            )
        thr = float(round(thr + REPROJ_RMSE_STEP_PX, 10))

    if not sweep_results:
        best_pack = {
            "threshold": None,
            "indices_used": idxs_all,
            "he": he_all,
            "best_name": best_name_all,
            "best_entry": best_entry_all,
            "best_dist": best_dist_all,
            "pnp_rmses": [r.rmse_px for r in stable],
            "frames_used": len(idxs_all),
            "diversity": diversity_all,
        }
    else:
        best_pack = min(sweep_results, key=lambda d: float(d["best_dist"]))

    with open(data_dir / "handeye_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "threshold": sr["threshold"],
                    "frames_used": sr["frames_used"],
                    "best_method": sr["best_name"],
                    "best_distance_to_prior_m": float(sr["best_dist"]),
                    "indices_used": sr["indices_used"],
                    "diversity": sr["diversity"],
                }
                for sr in sweep_results
            ],
            f,
            indent=2,
        )

    meta_valid = {
        "rs2_params_path": str(rs_params_path),
        "K": K,
        "dist": dist,
        "image_size": expect_wh,
        "frames_total": total,
        "frames_rejected": total - int(best_pack["frames_used"]),
    }

    title = "Hand Eye report RMSE validation sweep shaky filtered"
    if best_pack["threshold"] is not None:
        title += f" best_threshold={best_pack['threshold']:.3f} px"

    _write_handeye_report(
        out_path=data_dir / "handeye_report_validated.txt",
        title=title,
        frames_used=int(best_pack["frames_used"]),
        pnp_rmses=best_pack["pnp_rmses"],
        he_methods=best_pack["he"]["methods"],
        best_name=best_pack["best_name"],
        best_entry=best_pack["best_entry"],
        best_dist=float(best_pack["best_dist"]),
        indices_used=[int(i) for i in best_pack["indices_used"]],
        meta=meta_valid,
        diversity=best_pack.get("diversity"),
    )

    print("Done")
    print(f"Output dir: {data_dir}")
    return data_dir


if __name__ == "__main__":
    base = DATA_DIR
    if len(sys.argv) >= 2:
        base = Path(sys.argv[1]).expanduser().resolve()
    run_offline(base)
