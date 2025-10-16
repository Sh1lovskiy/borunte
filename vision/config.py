# vision/config.py
"""Configuration values for vision processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from config import get_settings


@dataclass(slots=True)
class CloudConfig:
    voxel_size: float
    merge_radius: float
    normal_radius: float


@dataclass(slots=True)
class SkeletonConfig:
    smoothing_lambda: float
    step_size: float


@dataclass(slots=True)
class GraphConfig:
    max_distance: float
    branch_threshold: float


@dataclass(slots=True)
class VizConfig:
    background_color: tuple[float, float, float]
    point_size: float


@dataclass(slots=True)
class VisionConfig:
    cloud: CloudConfig
    skeleton: SkeletonConfig
    graph: GraphConfig
    viz: VizConfig

    @staticmethod
    def create() -> "VisionConfig":
        _ = get_settings()  # currently unused but reserved for future integration
        cloud = CloudConfig(voxel_size=0.005, merge_radius=0.02, normal_radius=0.03)
        skeleton = SkeletonConfig(smoothing_lambda=0.1, step_size=0.05)
        graph = GraphConfig(max_distance=0.04, branch_threshold=0.02)
        viz = VizConfig(background_color=(0.1, 0.1, 0.1), point_size=2.0)
        return VisionConfig(cloud=cloud, skeleton=skeleton, graph=graph, viz=viz)


DEFAULT_VISION_CONFIG = VisionConfig.create()

__all__ = [
    "CloudConfig",
    "DEFAULT_VISION_CONFIG",
    "GraphConfig",
    "SkeletonConfig",
    "VisionConfig",
    "VizConfig",
]
