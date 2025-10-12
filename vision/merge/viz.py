from __future__ import annotations
from typing import Iterable
import numpy as np
import cv2
import open3d as o3d


def axes(size: float = 0.2) -> o3d.geometry.TriangleMesh:
    ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    ax.compute_vertex_normals()
    return ax


def aabb_lines(min_pt, max_pt, color=(0, 1, 0)) -> o3d.geometry.LineSet:
    min_pt = np.asarray(min_pt, float).reshape(3)
    max_pt = np.asarray(max_pt, float).reshape(3)
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt
    pts = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def viz_open3d(
    title: str,
    geoms: Iterable[o3d.geometry.Geometry],
    *,
    T_cam_to_base: np.ndarray | None = None,
    bbox_base: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None,
    base_axes_size: float = 0.12,
) -> None:
    gs: list[o3d.geometry.Geometry] = [axes(base_axes_size)]
    if T_cam_to_base is not None:
        cam_axes = axes(base_axes_size * 0.8)
        cam_axes.paint_uniform_color([0.85, 0.25, 0.25])
        cam_axes.transform(T_cam_to_base)
        gs.append(cam_axes)
    if bbox_base is not None:
        mn, mx = bbox_base
        gs.append(aabb_lines(mn, mx, color=(0, 1, 0)))
    for g in geoms:
        gs.append(g)
    o3d.visualization.draw_geometries(gs, window_name=title)


def show_depth_cv(
    depth_m: np.ndarray,
    title: str,
    depth_trunc: float,
    window_name: str,
    wait_ms: int = 1,
) -> bool:
    d = depth_m.astype(np.float32)
    if np.isfinite(depth_trunc):
        d = np.clip(d, 0, depth_trunc)
        maxv = max(1e-6, depth_trunc)
    else:
        finite = d[np.isfinite(d)]
        maxv = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        d = np.clip(d, 0, maxv)
    d8 = np.uint8(255.0 * (d / maxv))
    cm = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    color = cv2.applyColorMap(d8, cm)
    cv2.imshow(window_name, color)
    cv2.setWindowTitle(window_name, f"{title} (max={maxv:.2f} m)")
    key = cv2.waitKey(wait_ms) & 0xFF
    return key in (27, ord("q"))
