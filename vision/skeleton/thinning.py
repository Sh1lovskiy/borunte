# vision/skeleton/thinning.py
"""Skeleton thinning operations."""

from __future__ import annotations

import numpy as np

from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def skeletonize(points: np.ndarray, *, config: VisionConfig | None = None) -> np.ndarray:
    cfg = config or DEFAULT_VISION_CONFIG
    step = max(cfg.skeleton.step_size, 1e-3)
    stride = max(int(round(1 / step)), 1)
    indices = np.arange(0, len(points), stride)
    if indices.size == 0 or indices[-1] != len(points) - 1:
        indices = np.append(indices, len(points) - 1)
    skeleton = points[indices]
    LOGGER.debug("Thinned skeleton from {} to {} points", len(points), len(skeleton))
    return skeleton


__all__ = ["skeletonize"]
