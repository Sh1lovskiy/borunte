# borunte/state.py
"""State query helpers and formatting utilities."""

from __future__ import annotations

from typing import Tuple, Sequence
import socket

from .wire import q
from utils.logger import Logger

_log = Logger.get_logger()


def query_mode_move_alarm(sock: socket.socket) -> Tuple[int, int, int]:
    """Query current robot mode, motion state, and alarm flag."""
    cm, mv, al = q(sock, ["curMode", "isMoving", "curAlarm"])
    return int(cm), int(mv), int(al)


def query_world(sock: socket.socket) -> Tuple[float, float, float, float, float, float]:
    """Query current world pose (X, Y, Z, U, V, W)."""
    vals = q(sock, [f"world-{i}" for i in range(6)])
    return tuple(float(v) for v in vals[:6])  # type: ignore[return-value]


def fmt_xyzuvw_thousand(raw: Sequence[int]) -> str:
    """Format XYZUVW values stored in thousandths to readable units."""
    x, y, z, u, v, w = [r / 1000.0 for r in raw[:6]]
    return f"X={x:.3f} Y={y:.3f} Z={z:.3f} U={u:.3f} V={v:.3f} W={w:.3f}"
