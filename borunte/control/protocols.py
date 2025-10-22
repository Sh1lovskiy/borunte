# borunte/protocols.py
"""Protocol interfaces for dependency inversion.

Defines abstract interfaces for external systems (robot, camera, filesystem)
to enable testing and loose coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy.typing as npt


class RobotProtocol(Protocol):
    """Protocol for robot communication interface."""

    def connect(self) -> bool:
        """Connect to robot controller."""
        ...

    def disconnect(self) -> None:
        """Disconnect from robot controller."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to robot."""
        ...

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        u: float,
        v: float,
        w: float,
        verify: bool = True,
    ) -> bool:
        """Move robot to specified pose."""
        ...

    def get_world_pose(self) -> tuple[float, ...] | None:
        """Get current robot world pose."""
        ...

    def is_moving(self) -> bool | None:
        """Check if robot is currently moving."""
        ...

    def get_alarm(self) -> str | None:
        """Get current alarm status."""
        ...


class CameraProtocol(Protocol):
    """Protocol for camera interface."""

    def start(self, overrides: dict[str, Any] | None = None) -> None:
        """Start camera pipeline."""
        ...

    def stop(self) -> None:
        """Stop camera pipeline."""
        ...

    def snapshot(self) -> dict[str, Path]:
        """Capture a single frame."""
        ...


class FileSystemProtocol(Protocol):
    """Protocol for filesystem operations."""

    def ensure_directory(self, path: Path) -> None:
        """Ensure directory exists, creating if necessary."""
        ...

    def read_json(self, path: Path) -> dict[str, Any]:
        """Read JSON file."""
        ...

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file."""
        ...

    def read_image(self, path: Path) -> npt.NDArray[Any]:
        """Read image file."""
        ...

    def write_image(self, path: Path, image: npt.NDArray[Any]) -> None:
        """Write image file."""
        ...


class LoggerProtocol(Protocol):
    """Protocol for logging interface."""

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        ...


__all__ = [
    "RobotProtocol",
    "CameraProtocol",
    "FileSystemProtocol",
    "LoggerProtocol",
]
