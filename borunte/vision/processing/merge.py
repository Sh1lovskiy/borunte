# vision/processing/merge.py
"""Point cloud merging utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from borunte.utils.logger import get_logger
from borunte.utils.progress import progress_bar
from borunte.vision.config import VisionConfig

from .depth import create_cloud

logger = get_logger(__name__)


@dataclass
class MergeResult:
    """Result of a merge operation."""

    cloud: o3d.geometry.PointCloud
    transforms: list[np.ndarray]


@dataclass
class CloudMerger:
    """Merges point clouds from multiple frames."""

    config: VisionConfig
    visualize: bool = False
    visualize_per_frame: bool = False
    visualize_every_k: int = 10
    visualize_stages: tuple[str, ...] = ("merged",)
    viewer_factory: Callable[[], any] | None = None
    _viewer: any = field(init=False, default=None)

    def _get_viewer(self):
        if self.viewer_factory is None or not (self.visualize or self.visualize_per_frame):
            return None
        if self._viewer is None:
            self._viewer = self.viewer_factory()
        return self._viewer

    def _downsample(self, cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
        if voxel <= 0:
            return cloud
        if len(cloud.points) < 10:
            return cloud
        down = cloud.voxel_down_sample(voxel)
        if len(down.points) < 3:
            return cloud
        return down

    def _register(
        self, base: o3d.geometry.PointCloud, cloud: o3d.geometry.PointCloud
    ) -> np.ndarray:
        if len(base.points) == 0 or len(cloud.points) == 0:
            return np.eye(4)
        base_center = np.mean(np.asarray(base.points), axis=0)
        cloud_center = np.mean(np.asarray(cloud.points), axis=0)
        transform = np.eye(4)
        transform[:3, 3] = base_center - cloud_center
        return transform

    def merge(self, clouds: Sequence[o3d.geometry.PointCloud]) -> MergeResult:
        """Merge a sequence of clouds."""
        transforms: list[np.ndarray] = []
        accumulated_points: list[np.ndarray] = []
        accumulated_colors: list[np.ndarray] = []
        base_cloud: o3d.geometry.PointCloud | None = None
        viewer = self._get_viewer()

        for index, cloud in enumerate(
            progress_bar(range(len(clouds)), description="Merging frames")
        ):
            current = clouds[index]
            prepared = self._downsample(current, self.config.frame_voxel_size)
            transform = np.eye(4)
            if base_cloud is None:
                base_cloud = prepared
            else:
                transform = self._register(base_cloud, prepared)
                prepared = prepared.clone()
                prepared.transform(transform)
                if (
                    self.visualize_per_frame
                    and viewer is not None
                    and index % self.visualize_every_k == 0
                ):
                    viewer.show_clouds([base_cloud, prepared])
            transforms.append(transform)
            accumulated_points.append(np.asarray(prepared.points))
            if len(prepared.colors) > 0:
                accumulated_colors.append(np.asarray(prepared.colors))

        merged_points = (
            np.concatenate(accumulated_points) if accumulated_points else np.empty((0, 3))
        )
        merged_cloud = create_cloud(merged_points)
        if accumulated_colors:
            merged_colors = np.concatenate(accumulated_colors)
            merged_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
        merged_cloud = self._downsample(merged_cloud, self.config.merge_voxel_size)

        if self.visualize and "merged" in self.visualize_stages and viewer is not None:
            viewer.show_cloud(merged_cloud)
        if viewer is not None:
            viewer.close()
        return MergeResult(cloud=merged_cloud, transforms=transforms)


PointCloudAggregator = CloudMerger

__all__ = ["CloudMerger", "PointCloudAggregator", "MergeResult"]
