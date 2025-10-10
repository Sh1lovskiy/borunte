# borunte/control.py
"""Control helpers for heartbeat, start and stop actions, and safe release."""

from __future__ import annotations

import socket
import time
from typing import Optional


from .wire import cmd
from .state import query_mode_move_alarm
from .config import WAIT_STOP_S, HEARTBEAT_PERIOD_S
from utils.logger import Logger

_log = Logger.get_logger()


class Heartbeat:
    """Background heartbeat sender to keep the host session alive."""

    def __init__(self, sock: socket.socket, period_s: float = HEARTBEAT_PERIOD_S):
        self.sock = sock
        self.period_s = float(period_s)
        self._stop = False
        self._t = None

    def start(self) -> None:
        import threading

        def _loop():
            while not self._stop:
                try:
                    cmd(self.sock, "heartbreak")
                except Exception as e:
                    _log.tag("HB", f"failed: {e}", level="warning")
                    break
                time.sleep(self.period_s)

        self._t = threading.Thread(target=_loop, daemon=True)
        self._t.start()
        _log.tag("HB", "started")

    def stop(self, final_ping: bool = True) -> None:
        self._stop = True
        try:
            if final_ping:
                cmd(self.sock, "heartbreak")
        except Exception:
            pass
        if self._t:
            self._t.join(timeout=1.0)
        _log.tag("HB", "stopped")


# Basic commands


def start_button(sock: socket.socket):
    return cmd(sock, "startButton")


def stop_button(sock: socket.socket):
    return cmd(sock, "stopButton")


def pause_action(sock: socket.socket):
    return cmd(sock, "actionPause")


def single_cycle(sock: socket.socket):
    return cmd(sock, "actionSingleCycle")


def stop_action(sock: socket.socket):
    return cmd(sock, "actionStop")


def clear_alarm(sock: socket.socket):
    return cmd(sock, "clearAlarm")


def clear_alarm_continue(sock: socket.socket):
    return cmd(sock, "clearAlarmContinue")


def modify_gspd(sock: socket.socket, p: float):
    val = int(round(max(0.0, min(100.0, float(p))) * 10))
    return cmd(sock, "modifyGSPD", str(val))


def _best_effort_host_disconnect(sock: socket.socket) -> None:
    """Attempt a sequence of disconnect commands; ignore all errors."""
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
    for c in sequence:
        try:
            ok, msg, _ = cmd(sock, *c)
            if not ok and msg:
                _log.tag("DISC", f"{c[0]} message={msg}")
        except Exception:
            pass


def graceful_release(sock: socket.socket, hb: Optional[Heartbeat]) -> None:
    """Stop motion, stop heartbeat, and disconnect host safely."""
    try:
        try:
            _, moving, _ = query_mode_move_alarm(sock)
        except Exception:
            moving = 1

        if moving:
            pause_action(sock)
            time.sleep(0.1)
        stop_button(sock)
        stop_action(sock)

        start_time = time.time()
        while time.time() - start_time < WAIT_STOP_S:
            try:
                mode, mv, _ = query_mode_move_alarm(sock)
                if mode == 3 and mv == 0:
                    break
            except Exception:
                break
            time.sleep(0.2)

        if hb:
            hb.stop(final_ping=True)

        _best_effort_host_disconnect(sock)
        time.sleep(0.5)
        _log.tag("CTRL", "graceful release complete")
    except Exception as e:
        _log.tag("CTRL", f"graceful release exception: {e}", level="warning")


def emergency_halt(sock: socket.socket) -> None:
    """Emergency halt without relying on state queries."""
    try:
        pause_action(sock)
        time.sleep(0.05)
        stop_button(sock)
        stop_action(sock)
        _log.tag("CTRL", "emergency halt issued")
    except Exception as e:
        _log.tag("CTRL", f"emergency halt exception: {e}", level="warning")


def graceful_socket_close(sock: socket.socket) -> None:
    """Perform TCP close without sending RST."""
    try:
        sock.shutdown(socket.SHUT_WR)
        sock.settimeout(0.5)
        try:
            while True:
                if not sock.recv(4096):
                    break
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
        _log.tag("NET", "socket closed gracefully")
