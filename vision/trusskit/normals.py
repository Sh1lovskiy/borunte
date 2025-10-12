from __future__ import annotations

import numpy as np
import open3d as o3d


def ensure_normals(geom) -> np.ndarray:
    """Return Nx3 normals for cloud or mesh; compute if absent."""
    if isinstance(geom, o3d.geometry.PointCloud):
        if not geom.has_normals():
            geom.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.02, max_nn=30
                )
            )
        return np.asarray(geom.normals)
    if isinstance(geom, o3d.geometry.TriangleMesh):
        if not geom.has_vertex_normals():
            geom.compute_vertex_normals()
        return np.asarray(geom.vertex_normals)
    raise TypeError("Unsupported geometry for normals")
