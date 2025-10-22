# utils/geometry.py
"""Lightweight transformation utilities shared across packages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from borunte.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class Pose:
    position: np.ndarray
    orientation: np.ndarray  # roll, pitch, yaw in radians


@dataclass(slots=True)
class Transformation:
    matrix: np.ndarray  # 4x4 homogeneous matrix

    def inverse(self) -> Transformation:
        inv = np.linalg.inv(self.matrix)
        LOGGER.debug("Computed inverse transformation")
        return Transformation(matrix=inv)


def make_transformation(translation: Iterable[float], rpy: Iterable[float]) -> Transformation:
    tx, ty, tz = translation
    roll, pitch, yaw = rpy
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    rotation = np.array(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, cx * cz - sx * sy * sz, -sx * cy],
            [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([tx, ty, tz])
    LOGGER.debug("Created transformation from translation {} and rpy {}", translation, rpy)
    return Transformation(matrix=transform)


def compose_transformations(*transforms: Transformation) -> Transformation:
    result = np.eye(4)
    for transform in transforms:
        result = result @ transform.matrix
    LOGGER.debug("Composed {} transformations", len(transforms))
    return Transformation(matrix=result)


__all__ = [
    "Pose",
    "Transformation",
    "compose_transformations",
    "make_transformation",
]
