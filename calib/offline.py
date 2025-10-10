# calib/offline.py
"""Offline hand-eye calibration from a saved dataset.

Features:
- Dataset discovery (images, rs2_params.json, poses.json in the same folder).
- ChArUco detection + PnP refine; per-frame overlay with:
  * ChArUco corners
  * detected ArUco markers with per-marker axes
  * global target axes
- Hand-eye with multiple methods; reports include:
  * per-axis deltas to a translation prior
  * L2 distance to prior
  * per-axis spread across methods
  * simple pose diversity diagnostics
- Writes:
  * charuco_detections.json
  * handeye_sweep_results.json
  * handeye_report_no_validation.txt
  * handeye_report_validated.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import cv2
import numpy as np

from utils.logger import Logger
from utils.error_tracker import error_scope
from borunte.config import (
    USE_STREAM,
    SAVE_OVERLAY_IMAGES,
    REPROJ_RMSE_MIN_PX,
    REPROJ_RMSE_MAX_PX,
    REPROJ_RMSE_STEP_PX,
    MIN_SWEEP_FRAMES,
    CHARUCO_SQUARE_M,
    CHARUCO_MARKER_M,
    CHARUCO_SIZE,
    ARUCO_DICT_NAME,
    PRIOR_T_CAM2GRIPPER_M,
)

_log = Logger.get_logger()

# ---------------- small IO helpers ----------------


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_indices(dir_path: Path) -> List[int]:
    idxs: List[int] = []
    for p in sorted(dir_path.glob("*_rgb.png")):
        stem = p.name[:-8]  # drop "_rgb.png"
        try:
            idxs.append(int(stem))
        except Exception:
            pass
    return sorted(idxs)


def _load_intrinsics_from_rs2(
    rs_params_path: Path, stream: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    data = _load_json(rs_params_path)
    key = "color" if stream.lower() == "color" else "depth"
    intr = data["intrinsics"][key]
    K = np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(intr.get("coeffs", [0, 0, 0, 0, 0])[:5], float).ravel()
    if "streams" in data and key in data["streams"]:
        wh = (int(data["streams"][key]["width"]), int(data["streams"][key]["height"]))
    else:
        w = int(intr.get("width", 0))
        h = int(intr.get("height", 0))
        if w <= 0 or h <= 0:
            raise ValueError("Image size missing in rs2_params.json")
        wh = (w, h)
    return K, dist, wh


def _get_versions() -> Tuple[str, str]:
    cvv = getattr(cv2, "__version__", "unknown")
    try:
        import pyrealsense2 as rs  # noqa: F401

        rsv = getattr(rs, "__version__", "unknown")  # type: ignore
    except Exception:
        rsv = "unknown"
    return cvv, rsv


# ---------------- ChArUco/ArUco utils ----------------
def _aruco_dict(name: str):
    ar = cv2.aruco
    if not hasattr(ar, name):
        raise ValueError(f"Unknown ArUco dictionary {name}")
    return ar.getPredefinedDictionary(getattr(ar, name))


def _make_board():
    ar = cv2.aruco
    dic = _aruco_dict(ARUCO_DICT_NAME)
    try:
        return ar.CharucoBoard_create(
            CHARUCO_SIZE[0], CHARUCO_SIZE[1], CHARUCO_SQUARE_M, CHARUCO_MARKER_M, dic
        )
    except Exception:
        return ar.CharucoBoard(
            (CHARUCO_SIZE[0], CHARUCO_SIZE[1]), CHARUCO_SQUARE_M, CHARUCO_MARKER_M, dic
        )


def _board_obj_corners(board) -> np.ndarray:
    for attr in ("chessboardCorners", "getChessboardCorners", "get_chessboard_corners"):
        if hasattr(board, attr):
            obj = getattr(board, attr)()
            return np.asarray(obj, np.float32).reshape(-1, 3)
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


def _coverage_w_frac(char_corners: np.ndarray, wh: Tuple[int, int]) -> float:
    if char_corners is None or len(char_corners) == 0:
        return 0.0
    pts = char_corners.reshape(-1, 2)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    width = max(1.0, float(wh[0]))
    return float(max(0.0, min(1.0, (xmax - xmin) / width)))


def _detect_charuco_pose(
    img_bgr: np.ndarray,
    board,
    K: np.ndarray,
    dist: np.ndarray,
    refine_iters: int = 2,
    min_corners: int = 8,
):
    ar = cv2.aruco
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # detect markers
    try:
        params = ar.DetectorParameters()
        params.cornerRefinementMethod = getattr(ar, "CORNER_REFINE_SUBPIX", 1)
        detector = ar.ArucoDetector(_aruco_dict(ARUCO_DICT_NAME), params)
        m_corners, m_ids, m_rej = detector.detectMarkers(gray)
    except Exception:
        params = ar.DetectorParameters_create()
        m_corners, m_ids, m_rej = ar.detectMarkers(
            gray, _aruco_dict(ARUCO_DICT_NAME), parameters=params
        )
    if m_ids is None or len(m_ids) == 0:
        return None

    # refine markers against the board
    try:
        rc = ar.refineDetectedMarkers(
            image=gray,
            board=board,
            detectedCorners=m_corners,
            detectedIds=m_ids,
            rejectedCorners=m_rej,
            cameraMatrix=K,
            distCoeffs=dist,
        )
        m_corners, m_ids, m_rej = rc[0], rc[1], rc[2]
    except Exception:
        pass

    # interpolate ChArUco corners
    num, char_corners, char_ids = ar.interpolateCornersCharuco(
        markerCorners=m_corners,
        markerIds=m_ids,
        image=gray,
        board=board,
        cameraMatrix=K,
        distCoeffs=dist,
    )
    if char_ids is None or char_corners is None:
        return None
    if int(char_corners.shape[0]) < min_corners:
        return None

    # estimate global board pose
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

    # PnP refine on used corners
    obj_all = _board_obj_corners(board)
    sel = np.asarray(char_ids, int).flatten()
    obj = obj_all[sel, :].astype(np.float64)
    img_pts = char_corners.reshape(-1, 2).astype(np.float64)

    r_ref = rvec.copy()
    t_ref = tvec.copy()
    for _ in range(max(1, refine_iters)):
        ok2, r_ref, t_ref = cv2.solvePnP(
            obj,
            img_pts,
            K,
            dist,
            rvec=r_ref,
            tvec=t_ref,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok2:
            break

    proj, _ = cv2.projectPoints(obj, r_ref, t_ref, K, dist)
    proj = proj.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1))))
    R_tc, _ = cv2.Rodrigues(r_ref)
    t_tc = t_ref.reshape(3, 1).astype(np.float64)

    # per-marker poses for overlay (axes); independent marker frames
    marker_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    try:
        rvecs, tvecs, _ = ar.estimatePoseSingleMarkers(
            m_corners, CHARUCO_MARKER_M, K, dist
        )
        if rvecs is not None and tvecs is not None:
            for r, t in zip(
                np.asarray(rvecs).reshape(-1, 3, 1), np.asarray(tvecs).reshape(-1, 3, 1)
            ):
                marker_poses.append((r.astype(np.float64), t.astype(np.float64)))
    except Exception:
        pass

    return (
        R_tc.astype(np.float64),
        t_tc,
        sel.copy(),
        rmse,
        int(char_corners.shape[0]),
        char_corners,
        char_ids,
        m_corners,
        m_ids,
        marker_poses,
    )


# ---------------- Hand-Eye prep ----------------
def _build_lists_for_handeye(
    pnp: List[PnPResult], poses_json_path: Path, exclude_shaky: bool = True
):
    poses = _load_json(poses_json_path)
    valid = set(r.idx for r in pnp if (not exclude_shaky or not r.shaky))
    # supports dict {"000":{...}}
    idxs_all = sorted(int(k) for k in poses.keys() if k.isdigit())
    idxs = [i for i in idxs_all if i in valid]
    if not idxs:
        raise RuntimeError("No overlapping indices between poses and PnP")

    Rg2b: List[np.ndarray] = []
    tg2b: List[np.ndarray] = []
    Rt2c: List[np.ndarray] = []
    tt2c: List[np.ndarray] = []

    for i in idxs:
        p = poses[f"{i:03d}"]
        ux, vy, wz = np.deg2rad([p["rx"], p["ry"], p["rz"]])
        cx, sx = np.cos(ux), np.sin(ux)
        cy, sy = np.cos(vy), np.sin(vy)
        cz, sz = np.cos(wz), np.sin(wz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], float)
        R_bg = Rz @ Ry @ Rx
        t_bg = np.array(
            [[p["x"] / 1000.0], [p["y"] / 1000.0], [p["z"] / 1000.0]], float
        )

        R_g2b = R_bg.T
        t_g2b = -R_bg.T @ t_bg

        Rg2b.append(R_g2b.astype(np.float64))
        tg2b.append(t_g2b.astype(np.float64))

        pr = next(r for r in pnp if r.idx == i)
        Rt2c.append(pr.R_tc)
        tt2c.append(pr.t_tc)

    return idxs, Rg2b, tg2b, Rt2c, tt2c


def _run_handeye_all(
    Rg2b: List[np.ndarray],
    tg2b: List[np.ndarray],
    Rt2c: List[np.ndarray],
    tt2c: List[np.ndarray],
) -> dict:
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    out = {"methods": {}}
    for name, code in methods.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c, method=code)
        t = np.asarray(t_c2g, float).reshape(3).tolist()
        out["methods"][name] = {
            "R_cam2gripper": R_c2g.tolist(),
            "t_cam2gripper_m": t,
        }
        det = float(np.linalg.det(R_c2g))
        ort = float(np.linalg.norm(R_c2g.T @ R_c2g - np.eye(3)))
        _log.info(f"[HX] {name} det(R)={det:.6f} ortho={ort:.2e} t={t_c2g.ravel()}")
    return out


def _choose_best_by_t_prior(he_methods: dict, t_prior: np.ndarray):
    best_name, best_entry, best_dist = "", {}, float("inf")
    for name, entry in he_methods.items():
        t = np.asarray(entry["t_cam2gripper_m"], float).reshape(3)
        d = float(np.linalg.norm(t - t_prior))
        if d < best_dist:
            best_name, best_entry, best_dist = name, entry, d
    if not best_name:
        raise RuntimeError("No valid hand-eye candidates")
    return best_name, best_entry, best_dist


def _pose_diversity_stats(
    Rg2b: List[np.ndarray], tg2b: List[np.ndarray]
) -> Dict[str, float]:
    T = np.hstack(tg2b).T
    t_std = float(np.linalg.norm(T.std(axis=0)))
    rel_axes = []
    for i in range(1, len(Rg2b)):
        dR = Rg2b[i - 1].T @ Rg2b[i]
        rvec, _ = cv2.Rodrigues(dR)
        rel_axes.append(rvec.ravel())
    if len(rel_axes) >= 2:
        A = np.array(rel_axes)
        cov = np.cov(A.T)
        svals = np.linalg.svd(cov, compute_uv=False)
        rank_like = float(np.sum(svals > 1e-6))
    else:
        rank_like = 0.0
    return {"trans_spread_norm_m": t_std, "relative_rotation_rank": rank_like}


# ---------------- overlay helpers ----------------
def _draw_overlay(
    img_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R_tc: np.ndarray,
    t_tc: np.ndarray,
    char_corners: np.ndarray,
    char_ids: np.ndarray,
    marker_corners: List[np.ndarray] | None,
    marker_ids: np.ndarray | None,
    marker_poses: List[Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    vis = img_bgr.copy()
    ar = cv2.aruco
    try:
        # draw ChArUco used corners
        ar.drawDetectedCornersCharuco(vis, char_corners, char_ids)
    except Exception:
        pass
    try:
        # draw all detected markers outlines/ids
        if marker_corners is not None and marker_ids is not None:
            ar.drawDetectedMarkers(vis, marker_corners, marker_ids)
    except Exception:
        pass
    try:
        # draw per-marker local axes
        for rvec, tvec in marker_poses:
            cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.03)
    except Exception:
        pass
    try:
        # draw global board axes (refined pose)
        rvec, _ = cv2.Rodrigues(R_tc)
        cv2.drawFrameAxes(vis, K, dist, rvec, t_tc, 0.05)
    except Exception:
        pass
    cv2.imwrite(str(out_path), vis)


# ---------------- reporting ----------------
def _per_axis_summary(he_methods: dict, t_prior: np.ndarray) -> List[str]:
    lines: List[str] = []
    names = list(he_methods.keys())
    Ts = np.array(
        [he_methods[n]["t_cam2gripper_m"] for n in names], dtype=float
    )  # shape Mx3
    if Ts.size == 0:
        return lines
    mins = Ts.min(axis=0)
    maxs = Ts.max(axis=0)
    means = Ts.mean(axis=0)
    spreads = maxs - mins
    lines.append("Per-axis summary across methods (meters)")
    lines.append(
        f"tx: min={mins[0]:.6f} mean={means[0]:.6f} max={maxs[0]:.6f} spread={spreads[0]:.6f}"
    )
    lines.append(
        f"ty: min={mins[1]:.6f} mean={means[1]:.6f} max={maxs[1]:.6f} spread={spreads[1]:.6f}"
    )
    lines.append(
        f"tz: min={mins[2]:.6f} mean={means[2]:.6f} max={maxs[2]:.6f} spread={spreads[2]:.6f}"
    )
    lines.append("")
    lines.append("Per-method deltas to prior (meters)")
    for n in names:
        t = np.asarray(he_methods[n]["t_cam2gripper_m"], float).reshape(3)
        d = t - t_prior
        d2 = np.linalg.norm(d)
        lines.append(f"[{n}] dt=({d[0]:+.6f}, {d[1]:+.6f}, {d[2]:+.6f})  L2={d2:.6f}")
    return lines


def _write_report(
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
    t_prior = np.asarray(PRIOR_T_CAM2GRIPPER_M, float).reshape(3)

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
        f"K = [[{K[0,0]:.6f}, 0.000000, {K[0,2]:.6f}], "
        f"[0.000000, {K[1,1]:.6f}, {K[1,2]:.6f}], [0, 0, 1]]"
    )
    lines.append(f"dist = [{', '.join(f'{float(x):.6f}' for x in dist)}]")
    lines.append(f"image_size = {wh[0]}x{wh[1]}")
    lines.append("")
    lines.append("ChArUco reprojection error in pixels")
    lines.append(
        f"frames_ok={frames_used}  frames_total={meta.get('frames_total',0)}  "
        f"frames_rejected={meta.get('frames_rejected',0)}"
    )
    lines.append(
        f"rmse_mean={mean_rmse:.4f} rmse_median={med_rmse:.4f} "
        f"min={min_rmse:.4f} max={max_rmse:.4f}"
    )
    lines.append("")
    lines.append("Hand Eye results cam to gripper")
    for name, res in he_methods.items():
        t = res["t_cam2gripper_m"]
        lines.append(f"[{name}] t(m) = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
    lines.append(f"indices_used = {indices_used}")
    lines.append("")
    lines.append("Best by translation prior")
    t = best_entry["t_cam2gripper_m"]
    Rm = np.asarray(best_entry["R_cam2gripper"], np.float64)
    lines.append(f"best_method = {best_name}")
    lines.append(f"prior_t = {tuple(PRIOR_T_CAM2GRIPPER_M)}")
    lines.append(f"distance_to_prior_m = {best_dist:.6f}")
    lines.append(f"best_t_cam2gripper_m = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
    lines.append("best_R_cam2gripper =")
    for r in Rm:
        lines.append(f"[{r[0]: .9f}, {r[1]: .9f}, {r[2]: .9f}]")
    lines.append("")
    # per-axis metrics
    lines += _per_axis_summary(he_methods, t_prior)
    # diversity
    if diversity is not None:
        lines.append("Pose diversity diagnostic")
        lines.append(f"trans_spread_norm_m = {diversity['trans_spread_norm_m']:.6f}")
        lines.append(
            f"relative_rotation_rank = {diversity['relative_rotation_rank']:.1f}"
            if "relative_rotation_rank" in diversity
            else f"rel_rot_rank = {diversity.get('rel_rot_rank', 0.0):.1f}"
        )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------- main pipeline ----------------
def run_offline(data_dir: Path) -> Path:
    """Run offline hand-eye on a dataset directory."""
    with error_scope():
        base = Path(data_dir).expanduser().resolve()
        rs2 = base / "rs2_params.json"
        poses = base / "poses.json"
        if not rs2.exists() or not poses.exists():
            raise FileNotFoundError(f"Need rs2_params.json and poses.json in {base}")

        K, dist, wh = _load_intrinsics_from_rs2(rs2, USE_STREAM)
        board = _make_board()

        indices = _find_indices(base)
        if not indices:
            raise RuntimeError("No files like NNN_rgb.png found")

        det_log: Dict[str, Any] = {}
        results: List[PnPResult] = []
        total = 0

        for idx in indices:
            total += 1
            img_path = base / f"{idx:03d}_rgb.png"
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                det_log[f"{idx:03d}"] = {"found": False, "reason": "read_fail"}
                continue
            if (img.shape[1], img.shape[0]) != wh:
                det_log[f"{idx:03d}"] = {"found": False, "reason": "size_mismatch"}
                continue

            det = _detect_charuco_pose(img, board, K, dist, refine_iters=2)
            if det is None:
                det_log[f"{idx:03d}"] = {"found": False}
                continue

            (
                R_tc,
                t_tc,
                ids,
                rmse,
                n_used,
                ch,
                ch_ids,
                m_corners,
                m_ids,
                marker_poses,
            ) = det
            cov = _coverage_w_frac(ch, wh)

            shaky = bool(_load_json(poses).get(f"{idx:03d}", {}).get("shaky", False))

            if SAVE_OVERLAY_IMAGES:
                out = img_path.with_name(f"{img_path.stem}_charuco.png")
                _draw_overlay(
                    img,
                    K,
                    dist,
                    R_tc,
                    t_tc,
                    ch,
                    ch_ids,
                    m_corners,
                    m_ids,
                    marker_poses,
                    out,
                )

            results.append(
                PnPResult(
                    idx=idx,
                    R_tc=R_tc,
                    t_tc=t_tc,
                    rmse_px=rmse,
                    n_corners=int(n_used),
                    used_ids=[int(x) for x in ids.tolist()],
                    shaky=shaky,
                    coverage_w_frac=float(cov),
                )
            )
            det_log[f"{idx:03d}"] = {
                "found": True,
                "rmse_px": float(rmse),
                "n_corners": int(n_used),
                "ids": [int(x) for x in ids.tolist()],
                "shaky": bool(shaky),
                "coverage_w_frac": float(cov),
            }

        results.sort(key=lambda r: r.idx)
        _save_json(base / "charuco_detections.json", det_log)

        if not results:
            raise RuntimeError("No frames with valid ChArUco pose")

        stable = [r for r in results if not r.shaky]
        if len(stable) < 3:
            stable = results[:]

        t_prior = np.asarray(PRIOR_T_CAM2GRIPPER_M, np.float64).reshape(3)

        idxs_all, Rg_all, tg_all, Rt_all, tt_all = _build_lists_for_handeye(
            stable, poses, exclude_shaky=True
        )
        he_all = _run_handeye_all(Rg_all, tg_all, Rt_all, tt_all)
        name_all, entry_all, dist_all = _choose_best_by_t_prior(
            he_all["methods"], t_prior
        )
        diversity_all = _pose_diversity_stats(Rg_all, tg_all)

        meta_common = {
            "rs2_params_path": str(rs2),
            "K": K,
            "dist": dist,
            "image_size": wh,
            "frames_total": total,
            "frames_rejected": total - len(idxs_all),
        }

        _write_report(
            base / "handeye_report_no_validation.txt",
            "Hand Eye report no RMSE validation shaky filtered",
            len(idxs_all),
            [r.rmse_px for r in stable],
            he_all["methods"],
            name_all,
            entry_all,
            dist_all,
            idxs_all,
            meta_common,
            diversity_all,
        )

        # RMSE sweep
        sweep: List[Dict[str, Any]] = []
        thr = float(REPROJ_RMSE_MIN_PX)
        while thr <= REPROJ_RMSE_MAX_PX + 1e-9:
            valid = [r for r in stable if r.rmse_px <= thr]
            if len(valid) >= int(MIN_SWEEP_FRAMES):
                idxs_v, Rg_v, tg_v, Rt_v, tt_v = _build_lists_for_handeye(
                    valid, poses, exclude_shaky=True
                )
                he_v = _run_handeye_all(Rg_v, tg_v, Rt_v, tt_v)
                name_v, entry_v, dist_v = _choose_best_by_t_prior(
                    he_v["methods"], t_prior
                )
                sweep.append(
                    {
                        "threshold": float(round(thr, 3)),
                        "indices_used": idxs_v,
                        "he": he_v,
                        "best_name": name_v,
                        "best_entry": entry_v,
                        "best_dist": float(dist_v),
                        "pnp_rmses": [r.rmse_px for r in valid],
                        "frames_used": int(len(idxs_v)),
                        "diversity": _pose_diversity_stats(Rg_v, tg_v),
                    }
                )
            thr = float(round(thr + REPROJ_RMSE_STEP_PX, 10))

        _save_json(
            base / "handeye_sweep_results.json",
            [
                {
                    "threshold": sr["threshold"],
                    "frames_used": sr["frames_used"],
                    "best_method": sr["best_name"],
                    "best_distance_to_prior_m": float(sr["best_dist"]),
                    "indices_used": sr["indices_used"],
                    "diversity": sr["diversity"],
                }
                for sr in sweep
            ],
        )

        if not sweep:
            best_pack = {
                "threshold": None,
                "indices_used": idxs_all,
                "he": he_all,
                "best_name": name_all,
                "best_entry": entry_all,
                "best_dist": float(dist_all),
                "pnp_rmses": [r.rmse_px for r in stable],
                "frames_used": int(len(idxs_all)),
                "diversity": diversity_all,
            }
        else:
            best_pack = min(sweep, key=lambda d: float(d["best_dist"]))

        meta_valid = dict(meta_common)
        meta_valid["frames_rejected"] = total - int(best_pack["frames_used"])

        title = "Hand Eye report RMSE validation sweep shaky filtered"
        if best_pack["threshold"] is not None:
            title += f" best_threshold={best_pack['threshold']:.3f} px"

        _write_report(
            base / "handeye_report_validated.txt",
            title,
            int(best_pack["frames_used"]),
            best_pack["pnp_rmses"],
            best_pack["he"]["methods"],
            best_pack["best_name"],
            best_pack["best_entry"],
            float(best_pack["best_dist"]),
            [int(i) for i in best_pack["indices_used"]],
            meta_valid,
            best_pack.get("diversity"),
        )

        _log.tag("CALIB", f"offline calibration done in {base}")
        return base
