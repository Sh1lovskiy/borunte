# borunte/state.py
"""State query helpers built on RobotClient."""

from __future__ import annotations

from borunte.utils.logger import get_logger

from .client import RobotClient

_log = get_logger("borunte.state")


def _to_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


def get_state_snapshot(client: RobotClient) -> tuple[int, int, int]:
    """
    Return tuple (mode, moving, alarm).
    mode:
      0=None, 1=Manual, 2=Automatic, 3=Stop,
      7=Auto-running, 8=Step-by-Step, 9=Single Loop
    """
    values = client.query(["curMode", "isMoving", "curAlarm"])
    mode = _to_int(values[0]) if len(values) > 0 else 0
    moving = _to_int(values[1]) if len(values) > 1 else 0
    alarm = _to_int(values[2]) if len(values) > 2 else 0
    _log.tag("STATE", f"mode={mode} moving={moving} alarm={alarm}")
    return mode, moving, alarm


def query_mode_move_alarm(client: RobotClient) -> tuple[int, int, int]:
    """Backward compatibility alias."""
    return get_state_snapshot(client)


def query_world(client: RobotClient) -> tuple[float, float, float, float, float, float]:
    """Query world coordinates (X, Y, Z, U, V, W)."""
    values = client.query([f"world-{i}" for i in range(6)])
    if len(values) >= 6:
        pose = tuple(float(v) for v in values[:6])
        _log.tag("STATE", f"world={pose}")
        return pose  # type: ignore[return-value]
    else:
        _log.tag("STATE", f"world query incomplete: {values}", level="warning")
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


__all__ = [
    "get_state_snapshot",
    "query_mode_move_alarm",
    "query_world",
]
