# borunte/__init__.py
"""Borunte robot control and capture toolkit."""

from __future__ import annotations

from .config import (
    BORUNTE_CONFIG,
    BorunteConfig,
    RealSensePreviewConfig,
    RealSenseStream,
    load_borunte_config,
)
from .wire import RobotClient


def run_capture_pipeline(*args, **kwargs):
    from .runner import run_capture_pipeline as _run

    return _run(*args, **kwargs)


__all__ = [
    "BORUNTE_CONFIG",
    "BorunteConfig",
    "RealSensePreviewConfig",
    "RealSenseStream",
    "load_borunte_config",
    "RobotClient",
    "run_capture_pipeline",
]
