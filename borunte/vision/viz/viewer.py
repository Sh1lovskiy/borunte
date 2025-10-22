# vision/viz/viewer.py
"""Open3D viewer utilities."""

from __future__ import annotations

from collections.abc import Sequence

import open3d as o3d

from borunte.utils.logger import get_logger

logger = get_logger(__name__)


class Viewer:
    """Open3D visualization wrapper."""

    def __init__(self) -> None:
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(visible=True)

    def _clear(self) -> None:
        self._vis.clear_geometries()

    def _update(self) -> None:
        self._vis.poll_events()
        self._vis.update_renderer()

    def show_cloud(self, cloud: o3d.geometry.PointCloud) -> None:
        """Display a single point cloud."""
        self._clear()
        self._vis.add_geometry(cloud)
        self._update()

    def show_clouds(self, clouds: Sequence[o3d.geometry.PointCloud]) -> None:
        """Display multiple point clouds."""
        self._clear()
        for cloud in clouds:
            self._vis.add_geometry(cloud)
        self._update()

    def show_axes(self, size: float = 0.1) -> None:
        """Display coordinate axes."""
        self._clear()
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._vis.add_geometry(axes)
        self._update()

    def close(self) -> None:
        """Close the visualizer."""
        self._vis.destroy_window()


def show_cloud(cloud: o3d.geometry.PointCloud) -> None:
    """Convenience function to show a cloud."""
    viewer = Viewer()
    viewer.show_cloud(cloud)
    viewer.close()


def show_clouds(clouds: Sequence[o3d.geometry.PointCloud]) -> None:
    """Convenience function to show multiple clouds."""
    viewer = Viewer()
    viewer.show_clouds(clouds)
    viewer.close()


def show_axes(size: float = 0.1) -> None:
    """Convenience function to show axes."""
    viewer = Viewer()
    viewer.show_axes(size=size)
    viewer.close()


__all__ = ["Viewer", "show_cloud", "show_clouds", "show_axes"]
