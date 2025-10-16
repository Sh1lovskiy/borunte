# vision/skeleton/project2d.py
"""Projection helpers for skeleton extraction."""

from __future__ import annotations

import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def project_to_plane(cloud: np.ndarray) -> np.ndarray:
    projection = cloud[:, :2]
    LOGGER.debug("Projected cloud to 2D with shape {}", projection.shape)
    return projection


__all__ = ["project_to_plane"]
