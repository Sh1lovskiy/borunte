# calib/handeye.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2

from utils.logger import Logger

_log = Logger.get_logger()


def _as_rot_mats(Rs: List[np.ndarray]) -> List[np.ndarray]:
    """Accept 3x3 or Rodrigues(3x1/1x3); return list of 3x3."""
    out: List[np.ndarray] = []
    for i, R in enumerate(Rs):
        A = np.asarray(R, dtype=np.float64)
        if A.shape == (3, 3):
            out.append(A)
        elif A.size == 3:
            # Rodrigues vector -> 3x3
            Rm, _ = cv2.Rodrigues(A.reshape(3, 1))
            out.append(Rm)
        else:
            _log.warn(f"[HX] bad R shape idx={i} shape={A.shape}")
    return out


def _project_to_SO3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation via SVD; fix det<0."""
    U, S, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, 2] *= -1
        Rn = U @ Vt
    return Rn


def _sanitize_pairs(
    Rg2b, tg2b, Rt2c, tt2c
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    - Convert to 3x3 where needed
    - Drop NaN/Inf
    - Project to SO(3)
    - Keep only good pairs (det≈1, ortho error small)
    """
    Rg = _as_rot_mats(Rg2b)
    Rt = _as_rot_mats(Rt2c)
    tg = [np.asarray(t, float).reshape(3, 1) for t in tg2b]
    tt = [np.asarray(t, float).reshape(3, 1) for t in tt2c]

    n = min(len(Rg), len(Rt), len(tg), len(tt))
    good_Rg, good_tg, good_Rt, good_tt = [], [], [], []
    dets_g, dets_t = [], []
    dropped = 0

    for i in range(n):
        R1, t1, R2, t2 = Rg[i], tg[i], Rt[i], tt[i]
        if not np.isfinite(R1).all() or not np.isfinite(R2).all():
            _log.warn(f"[HX] NaN/Inf at idx={i}, drop")
            dropped += 1
            continue
        R1n = _project_to_SO3(R1)
        R2n = _project_to_SO3(R2)
        d1 = float(np.linalg.det(R1n))
        d2 = float(np.linalg.det(R2n))
        ort1 = np.linalg.norm(R1n.T @ R1n - np.eye(3))
        ort2 = np.linalg.norm(R2n.T @ R2n - np.eye(3))
        if abs(d1 - 1.0) > 5e-3 or abs(d2 - 1.0) > 5e-3 or ort1 > 5e-3 or ort2 > 5e-3:
            _log.warn(
                f"[HX] reject idx={i} det=({d1:.4f},{d2:.4f}) "
                f"ortho=({ort1:.2e},{ort2:.2e})"
            )
            dropped += 1
            continue
        good_Rg.append(R1n)
        good_tg.append(t1)
        good_Rt.append(R2n)
        good_tt.append(t2)
        dets_g.append(d1)
        dets_t.append(d2)

    _log.info(
        f"[HX] pairs in={n} kept={len(good_Rg)} dropped={dropped} "
        f"detRg=[{(min(dets_g) if dets_g else float('nan')):.4f},"
        f"{(max(dets_g) if dets_g else float('nan')):.4f}] "
        f"detRt=[{(min(dets_t) if dets_t else float('nan')):.4f},"
        f"{(max(dets_t) if dets_t else float('nan')):.4f}]"
    )
    return good_Rg, good_tg, good_Rt, good_tt


def solve_all(Rg2b, tg2b, Rt2c, tt2c):
    """
    Run multiple hand-eye methods after sanitization.
    Returns dict with solutions and diagnostics.
    """
    Rg, tg, Rt, tt = _sanitize_pairs(Rg2b, tg2b, Rt2c, tt2c)
    if len(Rg) < 3:
        raise ValueError("Not enough valid pairs after sanitize (need >=3)")

    methods = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    sols = {}
    for name, mcode in methods.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=mcode)
            sols[name] = {"R": R_c2g, "t": t_c2g}
            det = float(np.linalg.det(R_c2g))
            ort = np.linalg.norm(R_c2g.T @ R_c2g - np.eye(3))
            _log.info(
                f"[HX] {name} det(R)={det:.6f} ortho={ort:.2e} " f"t={t_c2g.ravel()}"
            )
        except cv2.error as e:
            _log.warn(f"[HX] {name} failed: {e}")
        except Exception as e:
            _log.warn(f"[HX] {name} error: {e}")

    if not sols:
        raise RuntimeError("All hand-eye methods failed after sanitize")
    return sols
