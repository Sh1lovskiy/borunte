from __future__ import annotations

from typing import List, Tuple
import numpy as np
import open3d as o3d


def make_lineset_from_polylines(
    polys_xyz: List[np.ndarray], color: Tuple[float, float, float]
) -> o3d.geometry.LineSet:
    pts, lines = [], []
    base = 0
    for poly in polys_xyz:
        if len(poly) < 2:
            continue
        pts.extend(poly.tolist())
        lines.extend([[base + i, base + i + 1] for i in range(len(poly) - 1)])
        base += len(poly)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, float))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, int))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def make_nodes_mesh(
    nodes_xyz: np.ndarray, color: Tuple[float, float, float]
) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(nodes_xyz, float))
    pc.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, float), (len(nodes_xyz), 1))
    )
    return pc
