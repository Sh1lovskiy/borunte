# vision/cloud/generator.py
"""Synthetic point cloud generator for testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CloudGeneratorConfig:
    noise_sigma: float = 0.002


class PointCloudGenerator:
    def __init__(self, config: CloudGeneratorConfig | None = None) -> None:
        self.config = config or CloudGeneratorConfig()

    def plane(self, *, size: float = 1.0, samples: int = 1000) -> np.ndarray:
        grid = np.random.rand(samples, 2) * size - size / 2
        z = np.zeros((samples, 1))
        noise = np.random.normal(scale=self.config.noise_sigma, size=(samples, 3))
        cloud = np.hstack([grid, z]) + noise
        LOGGER.debug("Generated plane cloud with {} samples", samples)
        return cloud

    def circle(self, *, radius: float = 0.5, samples: int = 500) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, samples)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.zeros_like(x)
        cloud = np.stack([x, y, z], axis=1)
        LOGGER.debug("Generated circle cloud with radius {:.3f}", radius)
        return cloud


__all__ = ["CloudGeneratorConfig", "PointCloudGenerator"]
