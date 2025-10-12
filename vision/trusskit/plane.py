from __future__ import annotations

import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("tk.plane")


def _basis_from_coeffs(
    abcd: np.ndarray, pts_on_plane: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build orthonormal (u,v,n) and p0 for plane a*x+b*y+c*z+d=0."""
    a, b, c, d = abcd.astype(float)
    n = np.array([a, b, c], float)
    n /= np.linalg.norm(n) + 1e-12
    p0 = pts_on_plane.mean(axis=0)

    # Orthonormal u,v via PCA in the plane
    d = pts_on_plane - p0[None, :]
    d -= (d @ n)[:, None] * n[None, :]
    C = (d.T @ d) / max(len(d), 1)
    w, V = np.linalg.eigh(C)
    u = V[:, np.argmax(w)]
    v = np.cross(n, u)
    u /= np.linalg.norm(u) + 1e-12
    v /= np.linalg.norm(v) + 1e-12
    return u, v, n, p0


def find_dominant_plane(
    cloud: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC plane with local basis (u,v,n,p0)."""
    if len(cloud.points) == 0:
        raise RuntimeError("empty cloud")
    model, inliers = cloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    pts = np.asarray(cloud.points)[inliers]
    u, v, n, p0 = _basis_from_coeffs(np.asarray(model, float), pts)
    LOG.info(f"plane inliers={len(inliers)}")
    return np.asarray(model, float), np.asarray(inliers, int), u, v, n, p0
