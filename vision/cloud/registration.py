# vision/cloud/registration.py
"""Simple registration helpers for aligning point clouds."""

from __future__ import annotations

import numpy as np

from vision.config import VisionConfig, DEFAULT_VISION_CONFIG
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def register_cloud(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    config: VisionConfig | None = None,
) -> np.ndarray:
    _ = config or DEFAULT_VISION_CONFIG
    ref_center = reference.mean(axis=0)
    mov_center = moving.mean(axis=0)
    translation = ref_center - mov_center
    LOGGER.debug("Registration translation {}", translation.tolist())
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform


__all__ = ["register_cloud"]
