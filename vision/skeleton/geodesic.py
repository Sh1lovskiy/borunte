# vision/skeleton/geodesic.py
"""Simple geodesic approximations for skeleton paths."""

from __future__ import annotations

import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    LOGGER.debug("Computed cumulative geodesic lengths with total {:.3f}", cumulative[-1])
    return cumulative


__all__ = ["cumulative_lengths"]
