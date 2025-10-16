# borunte/config.py
"""Configuration model specific to Borunte robot integrations."""

from __future__ import annotations

from dataclasses import dataclass

from config import get_settings


@dataclass(slots=True)
class RegisterMap:
    tcp_pose: int
    tcp_target: int
    start_step: int
    speed: int


@dataclass(slots=True)
class RetryPolicy:
    attempts: int
    delay_seconds: float


@dataclass(slots=True)
class BorunteConfig:
    host: str
    port: int
    connect_timeout: float
    request_timeout: float
    handover_timeout: float
    grid_spacing: float
    capture_speed: float
    registers: RegisterMap
    retry: RetryPolicy

    @staticmethod
    def from_settings() -> "BorunteConfig":
        root = get_settings()
        registers = RegisterMap(
            tcp_pose=0x1000,
            tcp_target=0x1010,
            start_step=0x1020,
            speed=0x1030,
        )
        retry = RetryPolicy(attempts=2, delay_seconds=0.5)
        return BorunteConfig(
            host=root.robot.host,
            port=root.robot.port,
            connect_timeout=root.robot.connect_timeout,
            request_timeout=root.robot.request_timeout,
            handover_timeout=root.robot.handover_timeout,
            grid_spacing=0.05,
            capture_speed=0.1,
            registers=registers,
            retry=retry,
        )


DEFAULT_CONFIG = BorunteConfig.from_settings()

__all__ = ["BorunteConfig", "RegisterMap", "RetryPolicy", "DEFAULT_CONFIG"]
