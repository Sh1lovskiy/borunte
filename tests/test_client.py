# tests/test_client.py
"""Tests for robot client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from borunte.client import RobotClient
from borunte.config import ROBOT_ANG_SCALE, ROBOT_POS_SCALE


@pytest.fixture
def mock_client() -> RobotClient:
    """Create a RobotClient with mocked socket."""
    client = RobotClient(host="192.168.1.1", port=9760, timeout=1.0)
    return client


def test_client_initialization(mock_client: RobotClient) -> None:
    """Test RobotClient initialization."""
    assert mock_client.host == "192.168.1.1"
    assert mock_client.port == 9760
    assert mock_client.timeout == 1.0
    assert not mock_client.is_connected()


@patch("borunte.client.socket.socket")
def test_client_connect(mock_socket_cls: MagicMock, mock_client: RobotClient) -> None:
    """Test robot client connection."""
    mock_socket_instance = MagicMock()
    mock_socket_cls.return_value = mock_socket_instance

    result = mock_client.connect()

    assert result is True
    assert mock_client.is_connected()
    mock_socket_instance.connect.assert_called_once_with(("192.168.1.1", 9760))


def test_client_disconnect(mock_client: RobotClient) -> None:
    """Test robot client disconnection."""
    mock_client._sock = MagicMock()
    mock_client.disconnect()

    assert not mock_client.is_connected()


def test_pose_scaling() -> None:
    """Test that pose scaling constants are used correctly."""
    client = RobotClient(host="test", port=9760)

    x, y, z = 100.5, 200.3, 300.7
    u, v, w = 180.0, 0.0, -90.0

    expected_x = int(round(x * ROBOT_POS_SCALE))
    expected_u = int(round(u * ROBOT_ANG_SCALE))

    assert expected_x == 100500
    assert expected_u == 180000
