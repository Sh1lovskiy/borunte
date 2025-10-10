# borunte/motion.py
"""Motion step implementation with logging and tolerance checks."""

from __future__ import annotations

import time
import socket
from typing import List, Sequence

from utils.error_tracker import ErrorTracker  # noqa: F401
from utils.logger import Logger
from .config import (
    ADDR_BASE,
    ADDR_LEN,
    POLL_S,
    ALARM_REPRINT_S,
    WAIT_MODE_S,
    WAIT_START_S,
    WAIT_MOVE_S,
    SPEED_PERCENT,
    POS_TOL_MM,
    ANG_TOL_DEG,
)
from .wire import rewrite_800_805, query_addrs
from .state import query_mode_move_alarm, query_world
from .control import (
    single_cycle,
    modify_gspd,
    start_button,
    clear_alarm,
    clear_alarm_continue,
    stop_button,
)

_log = Logger.get_logger()


def _read_addrs_retry(
    sock: socket.socket, base: int, length: int, tries: int = 3, delay: float = 0.15
) -> List[int]:
    vals = query_addrs(sock, base, length)
    for _ in range(tries - 1):
        if any(v != 0 for v in vals):
            break
        time.sleep(delay)
        vals = query_addrs(sock, base, length)
    return vals


def _within_tol(
    actual: Sequence[float],
    target: Sequence[float],
    pos_tol: float = POS_TOL_MM,
    ang_tol: float = ANG_TOL_DEG,
) -> bool:
    dx = [abs(actual[i] - target[i]) for i in range(6)]
    return (
        dx[0] <= pos_tol
        and dx[1] <= pos_tol
        and dx[2] <= pos_tol
        and dx[3] <= ang_tol
        and dx[4] <= ang_tol
        and dx[5] <= ang_tol
    )


def _wait_single_cycle(sock: socket.socket) -> None:
    t0 = time.time()
    while time.time() - t0 < WAIT_MODE_S:
        cm, _, _ = query_mode_move_alarm(sock)
        if cm == 9:
            return
        time.sleep(0.1)


def _wait_motion_start(sock: socket.socket) -> bool:
    t0 = time.time()
    last_state = 0.0
    last_alarm = None
    while time.time() - t0 < WAIT_START_S:
        cm, mv, al = query_mode_move_alarm(sock)
        if al and al != last_alarm:
            Logger.rm_alarm(_log, code=al)
            last_alarm = al
        if mv == 1:
            w = query_world(sock)
            Logger.rm_state(
                _log, cur_mode=cm, is_moving=mv, cur_alarm=al, world=w, tag="START"
            )
            return True
        if time.time() - last_state >= 1.0:
            w = query_world(sock)
            Logger.rm_state(
                _log, cur_mode=cm, is_moving=mv, cur_alarm=al, world=w, tag="WAIT"
            )
            last_state = time.time()
        time.sleep(POLL_S)
    return False


def _motion_loop(sock: socket.socket) -> None:
    t0 = time.time()
    last_alarm = None
    last_state = 0.0
    while True:
        time.sleep(POLL_S)
        cm, mv, al = query_mode_move_alarm(sock)
        if al and (al != last_alarm or time.time() - last_state >= ALARM_REPRINT_S):
            Logger.rm_alarm(_log, code=al)
            last_alarm = al
            last_state = time.time()
        if time.time() - last_state >= 1.0:
            w = query_world(sock)
            Logger.rm_state(
                _log, cur_mode=cm, is_moving=mv, cur_alarm=al, world=w, tag="MOVE"
            )
            last_state = time.time()
        if mv == 0:
            return
        if time.time() - t0 > WAIT_MOVE_S:
            _log.tag("MOTION", "timeout, sending stop", level="warning")
            stop_button(sock)
            return


def write_and_run_point(sock: socket.socket, pose: Sequence[float]) -> bool:
    """
    Write XYZUVW into 800..805 and execute a single-cycle motion.
    Returns True when the move completes successfully, False otherwise.
    """
    try:
        Logger.rm_step_target(_log, pose)

        before = _read_addrs_retry(sock, ADDR_BASE, ADDR_LEN)
        Logger.rm_addrs(_log, base=ADDR_BASE, values=before)

        ok, msg, _ = rewrite_800_805(sock, *pose)
        _log.tag("WRITE", f"ok={ok} message='{msg}'")

        time.sleep(0.12)
        after = _read_addrs_retry(sock, ADDR_BASE, ADDR_LEN)
        Logger.rm_addrs(_log, base=ADDR_BASE, values=after)

        if not ok and all(v == 0 for v in after):
            _log.tag(
                "MOTION",
                "rewriteDataList not confirmed, aborting step",
                level="warning",
            )
            return False

        ok, msg, _ = single_cycle(sock)
        _log.tag("MODE", f"ok={ok} message='{msg}'")

        _wait_single_cycle(sock)

        ok, msg, _ = modify_gspd(sock, SPEED_PERCENT)
        _log.tag("SPEED", f"ok={ok} message='{msg}'")

        cm, mv, al = query_mode_move_alarm(sock)
        w = query_world(sock)
        Logger.rm_state(
            _log, cur_mode=cm, is_moving=mv, cur_alarm=al, world=w, tag="PRE"
        )

        ok, msg, _ = start_button(sock)
        _log.tag("RUN", f"ok={ok} message='{msg}'")

        started = _wait_motion_start(sock)
        if not started:
            if al:
                clear_alarm(sock)
                clear_alarm_continue(sock)
            _log.tag("MOTION", "motion did not start", level="warning")
            return False

        _motion_loop(sock)

        wx, wy, wz, wu, wv, ww = query_world(sock)
        actual = (wx, wy, wz, wu, wv, ww)
        Logger.rm_state(
            _log, cur_mode=cm, is_moving=0, cur_alarm=al, world=actual, tag="DONE"
        )
        Logger.rm_check_tolerance(
            _log, actual=actual, target=pose, pos_tol=POS_TOL_MM, ang_tol=ANG_TOL_DEG
        )

        _log.tag("MOTION", "completed successfully")
        return True

    except KeyboardInterrupt:
        _log.tag("CTRL", "motion interrupted by user", level="warning")
        return False
    except Exception as e:
        _log.tag("MOTION", f"exception: {e}", level="warning")
        return False
