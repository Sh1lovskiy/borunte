# borunte/control.py
"""Robot control helpers built on top of RobotClient."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .config import BORUNTE_CONFIG, BorunteConfig
from .wire import RobotClient

_log = Logger.get_logger()


@dataclass
class CommandResult:
    ok: bool
    message: str
    raw: dict


class Heartbeat:
    """Background heartbeat sender to keep the host session alive."""

    def __init__(self, client: RobotClient, period_s: float | None = None):
        self.client = client
        self.period_s = (
            period_s if period_s is not None else BORUNTE_CONFIG.network.heartbeat_period_s
        )
        self._stop = False
        self._thread = None

    def start(self) -> None:
        import threading

        if self._thread is not None:
            return

        def _loop() -> None:
            while not self._stop:
                try:
                    self.client.command("heartbreak")
                except Exception as exc:
                    ErrorTracker.report(exc)
                    _log.tag("HB", f"failed: {exc}", level="warning")
                    break
                time.sleep(self.period_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        _log.tag("HB", "started")

    def stop(self, final_ping: bool = True) -> None:
        self._stop = True
        if final_ping:
            try:
                self.client.command("heartbreak")
            except Exception as exc:
                ErrorTracker.report(exc)
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        _log.tag("HB", "stopped")


def _wrap(result: Tuple[bool, str, dict]) -> CommandResult:
    return CommandResult(ok=result[0], message=result[1], raw=result[2])


def start_button(client: RobotClient) -> CommandResult:
    return _wrap(client.command("startButton"))


def stop_button(client: RobotClient) -> CommandResult:
    return _wrap(client.command("stopButton"))


def pause_action(client: RobotClient) -> CommandResult:
    return _wrap(client.command("actionPause"))


def single_cycle(client: RobotClient) -> CommandResult:
    return _wrap(client.command("actionSingleCycle"))


def stop_action(client: RobotClient) -> CommandResult:
    return _wrap(client.command("actionStop"))


def clear_alarm(client: RobotClient) -> CommandResult:
    return _wrap(client.command("clearAlarm"))


def clear_alarm_continue(client: RobotClient) -> CommandResult:
    return _wrap(client.command("clearAlarmContinue"))


def modify_gspd(client: RobotClient, speed_percent: float) -> CommandResult:
    val = int(round(max(0.0, min(100.0, float(speed_percent))) * 10))
    return _wrap(client.command("modifyGSPD", str(val)))


def _best_effort_disconnect(client: RobotClient) -> None:
    sequence = [
        ("setRemoteMode", "0"),
        ("connectHost", "0"),
        ("endHeartbreak",),
        ("exitRemoteMonitor",),
        ("disconnectHost",),
        ("closeConnectHost",),
        ("hostExit",),
        ("logout",),
        ("disconnectRM",),
        ("rmDisconnect",),
    ]
    for cmd_tuple in sequence:
        try:
            ok, msg, _ = client.command(*cmd_tuple)
            if not ok and msg:
                _log.tag("DISC", f"{cmd_tuple[0]} message={msg}")
        except Exception as exc:
            ErrorTracker.report(exc)


def graceful_release(
    client: RobotClient,
    hb: Optional[Heartbeat],
    config: BorunteConfig = BORUNTE_CONFIG,
) -> None:
    try:
        try:
            _, moving, _ = client.query(["curMode", "isMoving", "curAlarm"])
            moving_flag = int(moving)
        except Exception as exc:
            ErrorTracker.report(exc)
            moving_flag = 1

        if moving_flag:
            pause_action(client)
            time.sleep(0.1)
        stop_button(client)
        stop_action(client)

        start_time = time.time()
        while time.time() - start_time < config.network.wait_stop_s:
            try:
                mode, mv, _ = client.query(["curMode", "isMoving", "curAlarm"])
                if int(mode) == 3 and int(mv) == 0:
                    break
            except Exception as exc:
                ErrorTracker.report(exc)
                break
            time.sleep(0.2)

        if hb:
            hb.stop(final_ping=True)

        _best_effort_disconnect(client)
        time.sleep(0.5)
        _log.tag("CTRL", "graceful release complete")
    except Exception as exc:
        ErrorTracker.report(exc)
        _log.tag("CTRL", f"graceful release exception: {exc}", level="warning")


def emergency_halt(client: RobotClient) -> None:
    try:
        pause_action(client)
        time.sleep(0.05)
        stop_button(client)
        stop_action(client)
        _log.tag("CTRL", "emergency halt issued")
    except Exception as exc:
        ErrorTracker.report(exc)
        _log.tag("CTRL", f"emergency halt exception: {exc}", level="warning")
