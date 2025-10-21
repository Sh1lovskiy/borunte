# tests/test_integration.py
"""Integration smoke tests."""

from __future__ import annotations

import pytest

from borunte.config import Settings


def test_config_import_smoke() -> None:
    """Smoke test: import and instantiate main config."""
    from borunte.config import get_settings

    settings = get_settings()
    assert isinstance(settings, Settings)


def test_client_import_smoke() -> None:
    """Smoke test: import robot client."""
    from borunte.client import RobotClient

    client = RobotClient(host="localhost", port=9760)
    assert client is not None


@pytest.mark.slow
def test_full_config_structure() -> None:
    """Comprehensive test of config structure."""
    from borunte.config import get_settings

    settings = get_settings()

    # Verify all config sections are present
    assert hasattr(settings, "paths")
    assert hasattr(settings, "robot")
    assert hasattr(settings, "realsense")
    assert hasattr(settings, "network")
    assert hasattr(settings, "motion")
    assert hasattr(settings, "vision")
    assert hasattr(settings, "calibration")
    assert hasattr(settings, "grid")

    # Verify defaults
    assert hasattr(settings, "default_camera_intrinsics")
    assert hasattr(settings, "default_hand_eye")
    assert hasattr(settings, "workspace_bbox_points")
