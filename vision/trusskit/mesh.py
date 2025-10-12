from __future__ import annotations

import open3d as o3d
from utils.logger import Logger

LOG = Logger.get_logger("tk.mesh")


def build_mesh(cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """Try Poisson first; fallback to alpha shapes."""
    pc = o3d.geometry.PointCloud(cloud)
    if not pc.has_normals():
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        LOG.info(f"poisson mesh: {len(mesh.vertices)}V {len(mesh.triangles)}T")
        return mesh
    except Exception as e:
        LOG.warning(f"poisson failed: {e}; fallback alpha")
        alpha = 0.01
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha)
        mesh.compute_vertex_normals()
        LOG.info(f"alpha mesh: {len(mesh.vertices)}V {len(mesh.triangles)}T")
        return mesh
