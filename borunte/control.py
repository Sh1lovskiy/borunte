# borunte/control.py
"""Robot control helpers built on top of RobotClient."""

from __future__ import annotations

import time
import threading
from typing import Optional

from utils.error_tracker import ErrorTracker
from utils.logger import get_logger
from .client import RobotClient  # всегда локальный клиент

_log = get_logger("borunte.control")


class Heartbeat:
    """
    Background heartbeat + periodic host session assertion to avoid ERR9.
    Sends 'heartbreak' every `period_s`.
    Every `reassert_s` also does: setRemoteMode(1), connectHost(1).
    """

    def __init__(
        self, client: RobotClient, period_s: float = 5.0, reassert_s: float = 9.0
    ) -> None:
        self.client = client
        self.period_s = float(period_s)
        self.reassert_s = float(reassert_s)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, name="borunte-hb", daemon=True)
        self._th.start()
        _log.tag("HB", "started")

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        _log.tag("HB", "stopped")

    def _run(self) -> None:
        last_host = 0.0
        while not self._stop.is_set():
            try:
                ok, msg, _ = self.client.command("heartbreak", timeout=2.0)
                if not ok:
                    _log.tag("HB", f"failed: {msg}", level="warning")
            except Exception as exc:
                ErrorTracker().record("hb", str(exc))
                _log.tag("HB", f"failed: {exc!r}", level="warning")

            now = time.time()
            if now - last_host > self.reassert_s:
                try:
                    self.client.assert_host_session()
                except Exception as exc:
                    _log.tag("HOST", f"assert failed: {exc!r}", level="warning")
                last_host = now

            self._stop.wait(self.period_s)


def start_button(client: RobotClient) -> bool:
    ok, msg, _ = client.command("startButton")
    _log.tag("CTRL", f"startButton ok={ok} msg='{msg}'")
    return ok


def stop_button(client: RobotClient) -> bool:
    ok, msg, _ = client.command("stopButton")
    _log.tag("CTRL", f"stopButton ok={ok} msg='{msg}'")
    return ok


def action_stop(client: RobotClient) -> bool:
    ok, msg, _ = client.command("actionStop")
    _log.tag("CTRL", f"actionStop ok={ok} msg='{msg}'")
    return ok


def action_single_cycle(client: RobotClient) -> bool:
    ok, msg, _ = client.command("actionSingleCycle")
    _log.tag("CTRL", f"actionSingleCycle ok={ok} msg='{msg}'")
    return ok


def modify_gspd(client: RobotClient, speed_percent: float) -> bool:
    val = int(round(max(0.0, min(100.0, speed_percent)) * 10))
    ok, msg, _ = client.command("modifyGSPD", str(val))
    _log.tag("CTRL", f"modifyGSPD {val} ok={ok} msg='{msg}'")
    return ok


def clear_alarm(client: RobotClient) -> bool:
    ok, msg, _ = client.command("clearAlarm", "0")
    if not ok:
        _log.tag("CTRL", f"clearAlarm -> {msg}", level="warning")
    return ok


def clear_alarm_continue(client: RobotClient) -> bool:
    ok, msg, _ = client.command("clearAlarmContinue")
    if not ok:
        _log.tag("CTRL", f"clearAlarmContinue -> {msg}", level="warning")
    return ok


def set_remote_mode(client: RobotClient, on: bool) -> bool:
    ok, msg, _ = client.command("setRemoteMode", "1" if on else "0")
    _log.tag("CTRL", f"setRemoteMode({on}) ok={ok} msg='{msg}'")
    return ok


def graceful_release(client: RobotClient, hb: Optional[Heartbeat]) -> None:
    """Gracefully release robot control with short timeouts."""
    try:
        if hb:
            hb.stop()
    except Exception as exc:
        ErrorTracker().record("hb_stop", str(exc))

    cmds = [
        ("actionPause", []),
        ("stopButton", []),
        ("actionStop", []),
        ("setRemoteMode", ["0"]),
        ("connectHost", ["0"]),
        ("endHeartbreak", []),
        ("exitRemoteMonitor", []),
        ("disconnectHost", []),
        ("closeConnectHost", []),
        ("hostExit", []),
        ("logout", []),
    ]
    for name, args in cmds:
        try:
            ok, msg, _ = client.command(name, *args, timeout=0.8)
            if not ok and msg:
                _log.tag("CTRL", f"{name} -> {msg}", level="warning")
        except Exception as exc:
            _log.tag("CTRL", f"{name} timeout/error: {exc}", level="warning")
            break
    _log.tag("CTRL", "graceful release complete")
