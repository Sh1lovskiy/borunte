# tests/test_config.py
"""Tests for centralized configuration module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from borunte.config import (
    BASE_DIR,
    FILENAME_INTRINSICS_JSON,
    FILENAME_POSES_JSON,
    ROBOT_ANG_SCALE,
    ROBOT_POS_SCALE,
    ROBOT_READ_RETRIES,
    ROBOT_REGISTER_BASE_ADDR,
    CameraIntrinsics,
    HandEyeTransform,
    Pose,
    get_settings,
)


def test_constants_are_immutable() -> None:
    """Verify that constants are defined and have expected types."""
    assert isinstance(ROBOT_POS_SCALE, int)
    assert isinstance(ROBOT_ANG_SCALE, int)
    assert isinstance(ROBOT_READ_RETRIES, int)
    assert isinstance(ROBOT_REGISTER_BASE_ADDR, int)
    assert isinstance(FILENAME_POSES_JSON, str)
    assert isinstance(FILENAME_INTRINSICS_JSON, str)
    assert isinstance(BASE_DIR, Path)


def test_pose_dataclass(sample_pose: Pose) -> None:
    """Test Pose dataclass."""
    assert sample_pose.x == 100.0
    assert sample_pose.y == 200.0
    assert sample_pose.z == 300.0
    assert sample_pose.as_tuple() == (100.0, 200.0, 300.0, 180.0, 0.0, -90.0)


def test_camera_intrinsics(sample_intrinsics: CameraIntrinsics) -> None:
    """Test CameraIntrinsics dataclass."""
    assert sample_intrinsics.width == 1280
    assert sample_intrinsics.height == 720
    assert sample_intrinsics.fx == 920.0
    assert sample_intrinsics.as_tuple() == (1280, 720, 920.0, 920.0, 640.0, 360.0)


def test_hand_eye_transform(sample_hand_eye: HandEyeTransform) -> None:
    """Test HandEyeTransform dataclass."""
    assert sample_hand_eye.R.shape == (3, 3)
    assert sample_hand_eye.t.shape == (3,)
    assert sample_hand_eye.direction == "tcp_cam"

    T = sample_hand_eye.T_cam_tcp
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])


def test_hand_eye_invalid_shapes() -> None:
    """Test HandEyeTransform rejects invalid array shapes."""
    with pytest.raises(ValueError, match="R must be 3x3"):
        HandEyeTransform(
            R=np.eye(2, dtype=np.float64),
            t=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="t must be length 3"):
        HandEyeTransform(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.1, 0.2], dtype=np.float64),
        )


def test_get_settings_factory() -> None:
    """Test get_settings factory function."""
    settings = get_settings()

    assert settings.paths is not None
    assert settings.robot is not None
    assert settings.realsense is not None
    assert settings.network is not None
    assert settings.motion is not None
    assert settings.vision is not None
    assert settings.calibration is not None
    assert settings.grid is not None


def test_settings_immutability() -> None:
    """Test that Settings is frozen and immutable."""
    settings = get_settings()

    with pytest.raises(Exception):  # FrozenInstanceError
        settings.paths = None  # type: ignore


def test_robot_config_defaults() -> None:
    """Test RobotConfig has sensible defaults."""
    settings = get_settings()

    assert isinstance(settings.robot.host, str)
    assert isinstance(settings.robot.port, int)
    assert settings.robot.port > 0
    assert settings.robot.timeout_s > 0


def test_paths_config() -> None:
    """Test PathsConfig structure."""
    settings = get_settings()

    assert isinstance(settings.paths.data_root, Path)
    assert isinstance(settings.paths.captures_root, Path)
    assert isinstance(settings.paths.saves_root, Path)
    assert isinstance(settings.paths.logs_root, Path)


def test_vision_config_defaults() -> None:
    """Test VisionConfig has valid defaults."""
    settings = get_settings()

    assert settings.vision.merge_voxel_size > 0
    assert settings.vision.frame_voxel_size > 0
    assert settings.vision.depth_trunc > 0
    assert settings.vision.fx > 0
    assert settings.vision.fy > 0


def test_default_camera_intrinsics() -> None:
    """Test default camera intrinsics are valid."""
    settings = get_settings()
    intrinsics = settings.default_camera_intrinsics

    assert intrinsics.width > 0
    assert intrinsics.height > 0
    assert intrinsics.fx > 0
    assert intrinsics.fy > 0


def test_default_hand_eye() -> None:
    """Test default hand-eye transform is valid."""
    settings = get_settings()
    hand_eye = settings.default_hand_eye

    assert hand_eye.R.shape == (3, 3)
    assert hand_eye.t.shape == (3,)
    assert np.abs(np.linalg.det(hand_eye.R)) > 0.9  # Valid transformation matrix


def test_workspace_bbox_points() -> None:
    """Test workspace bounding box points."""
    settings = get_settings()
    bbox = settings.workspace_bbox_points

    assert bbox.shape == (4, 3)
    assert bbox.dtype == np.float64
