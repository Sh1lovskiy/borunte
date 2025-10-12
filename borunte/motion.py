# borunte/motion.py
"""Robot motion helpers for single-cycle moves."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .config import BORUNTE_CONFIG, BorunteConfig
from .control import (
    CommandResult,
    clear_alarm,
    clear_alarm_continue,
    modify_gspd,
    single_cycle,
    start_button,
    stop_button,
)
from .state import query_mode_move_alarm, query_world
from .wire import RobotClient

_log = Logger.get_logger()


def _read_registers(client: RobotClient, base: int, length: int) -> list[int]:
    return client.query_addresses(base, length)


def _retry_read(client: RobotClient, base: int, length: int, tries: int = 3) -> list[int]:
    vals = _read_registers(client, base, length)
    for _ in range(max(0, tries - 1)):
        if any(v != 0 for v in vals):
            break
        time.sleep(0.15)
        vals = _read_registers(client, base, length)
    return vals


def _within_tolerance(
    actual: Sequence[float],
    target: Sequence[float],
    pos_tol: float,
    ang_tol: float,
) -> bool:
    deltas = [abs(actual[i] - target[i]) for i in range(6)]
    return (
        deltas[0] <= pos_tol
        and deltas[1] <= pos_tol
        and deltas[2] <= pos_tol
        and deltas[3] <= ang_tol
        and deltas[4] <= ang_tol
        and deltas[5] <= ang_tol
    )


def _wait_mode(client: RobotClient, mode: int, timeout_s: float) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        cur_mode, _, _ = query_mode_move_alarm(client)
        if cur_mode == mode:
            return True
        time.sleep(0.1)
    return False


def _wait_motion_start(client: RobotClient, config: BorunteConfig) -> bool:
    start = time.time()
    last_state = 0.0
    last_alarm = None
    while time.time() - start < config.network.wait_start_s:
        mode, moving, alarm = query_mode_move_alarm(client)
        if alarm and alarm != last_alarm:
            Logger.rm_alarm(_log, code=alarm)
            last_alarm = alarm
        if moving == 1:
            world = query_world(client)
            Logger.rm_state(
                _log,
                cur_mode=mode,
                is_moving=moving,
                cur_alarm=alarm,
                world=world,
                tag="START",
            )
            return True
        if time.time() - last_state >= 1.0:
            world = query_world(client)
            Logger.rm_state(
                _log,
                cur_mode=mode,
                is_moving=moving,
                cur_alarm=alarm,
                world=world,
                tag="WAIT",
            )
            last_state = time.time()
        time.sleep(config.network.poll_period_s)
    return False


def _motion_loop(client: RobotClient, config: BorunteConfig) -> None:
    start = time.time()
    last_alarm = None
    last_state = 0.0
    while True:
        time.sleep(config.network.poll_period_s)
        mode, moving, alarm = query_mode_move_alarm(client)
        if alarm and (alarm != last_alarm or time.time() - last_state >= config.network.alarm_print_period_s):
            Logger.rm_alarm(_log, code=alarm)
            last_alarm = alarm
            last_state = time.time()
        if time.time() - last_state >= 1.0:
            world = query_world(client)
            Logger.rm_state(
                _log,
                cur_mode=mode,
                is_moving=moving,
                cur_alarm=alarm,
                world=world,
                tag="MOVE",
            )
            last_state = time.time()
        if moving == 0:
            return
        if time.time() - start > config.network.wait_move_s:
            _log.tag("MOTION", "timeout, sending stop", level="warning")
            stop_button(client)
            return


def _log_registers(client: RobotClient, base: int, length: int) -> list[int]:
    regs = _retry_read(client, base, length)
    Logger.rm_addrs(_log, base=base, values=regs)
    return regs


def write_and_run_point(
    client: RobotClient,
    pose: Sequence[float],
    config: BorunteConfig = BORUNTE_CONFIG,
) -> bool:
    """Write XYZUVW into registers and execute a move."""

    try:
        Logger.rm_step_target(_log, pose)

        _log_registers(client, config.register_map.base_addr, config.register_map.length)
        ok, msg, _ = client.rewrite_pose(*pose)
        _log.tag("WRITE", f"ok={ok} message='{msg}'")

        time.sleep(0.12)
        after = _log_registers(client, config.register_map.base_addr, config.register_map.length)

        if not ok and all(v == 0 for v in after):
            _log.tag("MOTION", "rewriteDataList not confirmed, aborting", level="warning")
            return False

        write_cycle: Iterable[tuple[str, CommandResult]] = (
            ("single_cycle", single_cycle(client)),
            ("modify_gspd", modify_gspd(client, config.motion.speed_percent)),
        )
        for name, result in write_cycle:
            _log.tag("CMD", f"{name} ok={result.ok} message='{result.message}'")

        if not _wait_mode(client, mode=9, timeout_s=config.network.wait_mode_s):
            _log.tag("MOTION", "single-cycle mode not reached", level="warning")
            return False

        mode, moving, alarm = query_mode_move_alarm(client)
        world = query_world(client)
        Logger.rm_state(
            _log,
            cur_mode=mode,
            is_moving=moving,
            cur_alarm=alarm,
            world=world,
            tag="PRE",
        )

        start_res = start_button(client)
        _log.tag("RUN", f"ok={start_res.ok} message='{start_res.message}'")

        if not _wait_motion_start(client, config):
            if alarm:
                clear_alarm(client)
                clear_alarm_continue(client)
            _log.tag("MOTION", "motion did not start", level="warning")
            return False

        _motion_loop(client, config)

        wx, wy, wz, wu, wv, ww = query_world(client)
        actual = (wx, wy, wz, wu, wv, ww)
        Logger.rm_state(
            _log,
            cur_mode=mode,
            is_moving=0,
            cur_alarm=alarm,
            world=actual,
            tag="DONE",
        )
        Logger.rm_check_tolerance(
            _log,
            actual=actual,
            target=pose,
            pos_tol=config.motion.position_tol_mm,
            ang_tol=config.motion.angle_tol_deg,
        )

        if not _within_tolerance(
            actual,
            pose,
            config.motion.position_tol_mm,
            config.motion.angle_tol_deg,
        ):
            _log.tag("MOTION", "pose outside tolerance", level="warning")
            return False

        _log.tag("MOTION", "completed successfully")
        return True
    except KeyboardInterrupt:
        _log.tag("CTRL", "motion interrupted by user", level="warning")
        return False
    except Exception as exc:
        ErrorTracker.report(exc)
        _log.tag("MOTION", f"exception: {exc}", level="warning")
        return False


__all__ = ["write_and_run_point"]
