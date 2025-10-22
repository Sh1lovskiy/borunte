# vision/processing/depth.py
"""Depth processing utilities."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from borunte.vision.config import VisionConfig


def depth_to_xyz(depth: np.ndarray, config: VisionConfig) -> np.ndarray:
    """Convert a depth image to XYZ coordinates."""
    if depth.ndim == 2:
        depth_values = depth.astype(np.float32) * config.depth_scale
        mask = (depth_values > 0) & (depth_values < config.depth_trunc)
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32)
        indices_y, indices_x = np.nonzero(mask)
        z = depth_values[indices_y, indices_x]
        x = (indices_x - config.cx) * z / config.fx
        y = (indices_y - config.cy) * z / config.fy
        return np.stack((x, y, z), axis=1)
    if depth.ndim == 3 and depth.shape[1] >= 3:
        return depth[:, :3].astype(np.float32)
    raise ValueError("Depth input has unsupported shape")


def depth_to_xyz_with_mask(
    depth: np.ndarray, config: VisionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return XYZ coordinates and boolean mask for valid pixels."""
    if depth.ndim != 2:
        points = depth_to_xyz(depth, config)
        mask = np.ones((points.shape[0],), dtype=bool)
        return points, mask
    depth_values = depth.astype(np.float32) * config.depth_scale
    mask = (depth_values > 0) & (depth_values < config.depth_trunc)
    indices_y, indices_x = np.nonzero(mask)
    if indices_y.size == 0:
        return np.empty((0, 3), dtype=np.float32), mask
    z = depth_values[indices_y, indices_x]
    x = (indices_x - config.cx) * z / config.fx
    y = (indices_y - config.cy) * z / config.fy
    points = np.stack((x, y, z), axis=1)
    return points, mask


def depth_to_cloud(
    depth: np.ndarray, color: np.ndarray | None, config: VisionConfig
) -> o3d.geometry.PointCloud:
    """Convert depth and optional color into a point cloud."""
    points, mask = depth_to_xyz_with_mask(depth, config)
    cloud = create_cloud(points)
    if color is not None and color.size and mask.ndim == 2:
        valid_colors = color[mask]
        if valid_colors.size:
            colors = valid_colors.reshape((-1, color.shape[-1]))
            cloud.colors = o3d.utility.Vector3dVector(_normalize_colors(colors))
    return cloud


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    norm = colors.astype(np.float32)
    if norm.max() > 1.0:
        norm /= 255.0
    return norm[:, :3]


def create_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from arrays."""
    cloud = o3d.geometry.PointCloud()
    if points.size:
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and colors.size:
        norm_colors = _normalize_colors(colors)
        cloud.colors = o3d.utility.Vector3dVector(norm_colors)
    return cloud


__all__ = ["depth_to_xyz", "depth_to_cloud", "create_cloud", "depth_to_xyz_with_mask"]
