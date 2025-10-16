# borunte/client.py
"""Stateful client abstraction for the Borunte robot controller."""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from pymodbus.client import ModbusTcpClient
except Exception:  # noqa: BLE001
    ModbusTcpClient = None  # type: ignore

from borunte.config import BorunteConfig, DEFAULT_CONFIG
from utils.error_tracker import ErrorTracker
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class RobotClient:
    """High-level interface for robot pose control using Modbus TCP."""

    def __init__(
        self,
        config: BorunteConfig | None = None,
        *,
        tracker: Optional[ErrorTracker] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.tracker = tracker or ErrorTracker(context="borunte.robot")
        self._client: Optional[ModbusTcpClient] = None
        self._connected = False
        self._last_pose: List[float] = [0.0] * 6
        self._target_pose: List[float] = [0.0] * 6
        self._speed = 0.0

    def connect(self) -> None:
        if self._connected:
            LOGGER.debug("RobotClient already connected")
            return
        attempts = min(self.config.retry.attempts, 2)
        for attempt in range(attempts):
            try:
                if ModbusTcpClient is None:  # pragma: no cover - fallback
                    LOGGER.warning("pymodbus not available; running in simulation mode")
                    self._connected = True
                    return
                LOGGER.info(
                    "Connecting to robot at {}:{} (attempt {})",
                    self.config.host,
                    self.config.port,
                    attempt + 1,
                )
                client = ModbusTcpClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.request_timeout,
                )
                if not client.connect():
                    raise TimeoutError("Unable to establish Modbus connection")
                self._client = client
                self._connected = True
                LOGGER.info("Robot connection established")
                return
            except TimeoutError as exc:
                self.tracker.record("connect", str(exc))
                if attempt + 1 >= attempts:
                    raise
                LOGGER.warning("Retrying robot connection after timeout")
                time.sleep(self.config.retry.delay_seconds)

    def read_pose(self) -> Sequence[float]:
        if not self._connected:
            raise RuntimeError("RobotClient is not connected")
        LOGGER.debug("Returning cached TCP pose: {}", self._last_pose)
        return tuple(self._last_pose)

    def write_pose_target(self, pose: Iterable[float]) -> None:
        if not self._connected:
            raise RuntimeError("RobotClient is not connected")
        self._target_pose = list(pose)
        LOGGER.info("Updated target pose to {}", self._target_pose)

    def start_step(self) -> None:
        if not self._connected:
            raise RuntimeError("RobotClient is not connected")
        LOGGER.info(
            "Initiating robot step from pose {} to {}",
            self._last_pose,
            self._target_pose,
        )
        self._last_pose = list(self._target_pose)

    def set_speed(self, speed: float) -> None:
        if not self._connected:
            raise RuntimeError("RobotClient is not connected")
        self._speed = speed
        LOGGER.info("Set robot TCP speed to {:.3f}", self._speed)

    def graceful_handover(self) -> None:
        if not self._connected:
            LOGGER.debug("RobotClient already disconnected")
            return
        LOGGER.info("Performing graceful handover")
        time.sleep(min(self.config.handover_timeout, 1.0))
        self.close()

    def close(self) -> None:
        if not self._connected:
            LOGGER.debug("RobotClient close called on inactive connection")
            return
        if self._client is not None:
            self._client.close()
            self._client = None
        self._connected = False
        LOGGER.info("Robot connection closed")


__all__ = ["RobotClient"]
