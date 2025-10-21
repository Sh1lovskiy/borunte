# tests/conftest.py
"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from borunte.config import CameraIntrinsics, HandEyeTransform, Pose, Settings


@pytest.fixture
def mock_socket() -> MagicMock:
    """Mock socket for robot client testing."""
    mock = MagicMock()
    mock.recv.return_value = b'{"cmdReply": ["ok"]}\n'
    return mock


@pytest.fixture
def temp_captures_dir(tmp_path: Path) -> Path:
    """Temporary directory for test captures."""
    captures = tmp_path / "captures"
    captures.mkdir()
    return captures


@pytest.fixture
def sample_pose() -> Pose:
    """Sample robot pose."""
    from borunte.config import Pose

    return Pose(x=100.0, y=200.0, z=300.0, rx=180.0, ry=0.0, rz=-90.0)


@pytest.fixture
def sample_intrinsics() -> CameraIntrinsics:
    """Sample camera intrinsics."""
    from borunte.config import CameraIntrinsics

    return CameraIntrinsics(width=1280, height=720, fx=920.0, fy=920.0, cx=640.0, cy=360.0)


@pytest.fixture
def sample_hand_eye() -> HandEyeTransform:
    """Sample hand-eye transform."""
    from borunte.config import HandEyeTransform

    return HandEyeTransform(
        R=np.eye(3, dtype=np.float64),
        t=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        direction="tcp_cam",
    )


@pytest.fixture
def test_settings(temp_captures_dir: Path) -> Settings:
    """Test settings with temporary directories."""
    from borunte.config import (
        CalibrationConfig,
        GridConfig,
        MotionConfig,
        NetworkConfig,
        PathsConfig,
        RealSenseConfig,
        RobotConfig,
        Settings,
        VisionConfig,
    )

    paths = PathsConfig(
        data_root=temp_captures_dir,
        captures_root=temp_captures_dir / "captures",
        saves_root=temp_captures_dir / "saves",
        logs_root=temp_captures_dir / "logs",
    )

    return Settings(
        paths=paths,
        robot=RobotConfig(),
        realsense=RealSenseConfig(),
        network=NetworkConfig(),
        motion=MotionConfig(),
        vision=VisionConfig(),
        calibration=CalibrationConfig(),
        grid=GridConfig(),
    )
