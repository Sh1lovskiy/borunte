# borunte/motion.py
"""Single-point motion: write pose, run cycle, wait, verify."""

from __future__ import annotations

import time
from typing import Sequence, Tuple

from utils.logger import get_logger
from .state import query_mode_move_alarm, query_world
from .client import RobotClient
from .control import action_single_cycle, modify_gspd, start_button

_log = get_logger("borunte.motion")


def _deg_norm(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _angles_close(
    tgt: Tuple[float, float, float], cur: Tuple[float, float, float], tol_deg: float
) -> bool:
    du = abs(_deg_norm(cur[0] - tgt[0]))
    dv = abs(_deg_norm(cur[1] - tgt[1]))
    dw = abs(_deg_norm(cur[2] - tgt[2]))
    return du <= tol_deg and dv <= tol_deg and dw <= tol_deg


def write_and_run_point(
    client: RobotClient, pose: Sequence[float], max_retries: int = 2
) -> bool:
    """
    Write pose (×1000 for all six), then SingleCycle→GSPD→start, wait, verify.
    """
    x, y, z, u, v, w = map(float, pose[:6])
    _log.tag("STEP", f"target=({round(x,3)}, {round(y,3)}, {round(z,3)})")

    # write with retries
    for attempt in range(max_retries + 1):
        ok, msg, _ = client.rewrite_pose(x, y, z, u, v, w)
        if ok:
            _log.tag("WRITE", "ok=True")
            break
        _log.tag("WRITE", f"retry {attempt}/{max_retries}: {msg}", level="warning")
        time.sleep(0.3)
    else:
        _log.tag("WRITE", "all retries exhausted", level="error")
        return False

    if not client.verify_pose_write(x, y, z, u, v, w):
        _log.tag("VERIFY", "pose mismatch", level="warning")
        return False

    if not action_single_cycle(client):
        return False
    if not modify_gspd(client, 45.0):
        return False
    if not start_button(client):
        return False

    # wait start
    t0 = time.time()
    while time.time() - t0 < 20.0:
        mode, moving, alarm = query_mode_move_alarm(client)
        if alarm != 0:
            _log.tag("ALRM", f"{alarm}", level="warning")
            return False
        if moving == 1:
            wx, wy, wz, wu, wv, ww = query_world(client)
            _log.tag("STATE", f"start world=({wx}, {wy}, {wz}, {wu}, {wv}, {ww})")
            break
        time.sleep(0.1)
    else:
        _log.tag("START", "motion did not start", level="warning")
        return False

    # wait stop with periodic state
    last = 0.0
    while True:
        time.sleep(0.2)
        mode, moving, alarm = query_mode_move_alarm(client)
        if alarm != 0:
            _log.tag("ALRM", f"{alarm}", level="warning")
            return False
        if time.time() - last >= 1.0:
            wx, wy, wz, wu, wv, ww = query_world(client)
            _log.tag(
                "MOVE",
                f"mode={mode} moving={moving} alarm={alarm} "
                f"W=({wx}, {wy}, {wz}, {wu}, {wv}, {ww})",
            )
            last = time.time()
        if moving == 0:
            break

    wx, wy, wz, wu, wv, ww = query_world(client)
    pos_tol, ang_tol = 2.0, 2.0
    dpos = (abs(wx - x), abs(wy - y), abs(wz - z))
    pos_ok = all(d <= pos_tol for d in dpos)
    ang_ok = _angles_close((u, v, w), (wu, wv, ww), ang_tol)

    _log.tag("DONE", f"WORLD=({wx}, {wy}, {wz}, {wu}, {wv}, {ww})")
    _log.tag(
        "TOL",
        f"pos={tuple(round(d,3) for d in dpos)}<= {pos_tol}  "
        f"ang=({abs(_deg_norm(wu-u)):.3f}, "
        f"{abs(_deg_norm(wv-v)):.3f}, "
        f"{abs(_deg_norm(ww-w)):.3f})<= {ang_tol}  "
        f"ok={pos_ok and ang_ok}",
    )

    return bool(pos_ok and ang_ok)
