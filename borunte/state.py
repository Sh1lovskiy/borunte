# borunte/state.py
"""State query helpers built on RobotClient."""

from __future__ import annotations

from typing import Sequence, Tuple

from utils.logger import Logger

from .wire import RobotClient

_log = Logger.get_logger()


def query_mode_move_alarm(client: RobotClient) -> Tuple[int, int, int]:
    values = client.query(["curMode", "isMoving", "curAlarm"])
    mode, moving, alarm = (int(values[0]), int(values[1]), int(values[2]))
    _log.tag("STATE", f"mode={mode} moving={moving} alarm={alarm}")
    return mode, moving, alarm


def query_world(client: RobotClient) -> Tuple[float, float, float, float, float, float]:
    values = client.query([f"world-{i}" for i in range(6)])
    pose = tuple(float(v) for v in values[:6])
    _log.tag("STATE", f"world={pose}")
    return pose  # type: ignore[return-value]


def fmt_xyzuvw_thousand(raw: Sequence[int]) -> str:
    x, y, z, u, v, w = [r / 1000.0 for r in raw[:6]]
    return f"X={x:.3f} Y={y:.3f} Z={z:.3f} U={u:.3f} V={v:.3f} W={w:.3f}"


__all__ = ["query_mode_move_alarm", "query_world", "fmt_xyzuvw_thousand"]
