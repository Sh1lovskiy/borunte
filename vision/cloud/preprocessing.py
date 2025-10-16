# vision/cloud/preprocessing.py
"""Preprocessing utilities for point clouds."""

from __future__ import annotations

import numpy as np

from vision.config import VisionConfig, DEFAULT_VISION_CONFIG
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def voxel_downsample(cloud: np.ndarray, *, config: VisionConfig | None = None) -> np.ndarray:
    cfg = config or DEFAULT_VISION_CONFIG
    size = cfg.cloud.voxel_size
    quantised = np.floor(cloud / size)
    _, unique_idx = np.unique(quantised, axis=0, return_index=True)
    downsampled = cloud[np.sort(unique_idx)]
    LOGGER.debug("Downsampled cloud from {} to {} points", cloud.shape[0], downsampled.shape[0])
    return downsampled


def center_cloud(cloud: np.ndarray) -> np.ndarray:
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    LOGGER.debug("Centered cloud around centroid {}", centroid.tolist())
    return centered


__all__ = ["center_cloud", "voxel_downsample"]
