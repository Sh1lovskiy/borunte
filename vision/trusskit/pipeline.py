"""TrussKit pipeline with pre-plane BBox crop."""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d
from utils.logger import Logger

from . import (
    config as cfgmod,
    io,
    plane,
    project2d,
    skeleton2d,
    nodes_edges,
    graph_build,
    graph_traverse,
    mesh as meshmod,
    viewer,
)
from .transforms import PlaneFrame

LOG = Logger.get_logger("tk.pipeline")


def _apply_bbox(
    cloud: o3d.geometry.PointCloud,
    bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
) -> o3d.geometry.PointCloud:
    if not bbox:
        return cloud
    mn = np.asarray(bbox[0], dtype=float)
    mx = np.asarray(bbox[1], dtype=float)
    aabb = o3d.geometry.AxisAlignedBoundingBox(mn, mx)
    out = cloud.crop(aabb)
    LOG.info(f"crop: {len(cloud.points)} → {len(out.points)} points")
    return out


def _sphere_cloud(nodes: np.ndarray, r: float) -> o3d.geometry.TriangleMesh:
    base = o3d.geometry.TriangleMesh.create_sphere(radius=max(r, 1e-4))
    base.compute_vertex_normals()
    v, t = np.asarray(base.vertices), np.asarray(base.triangles, int)
    all_v, all_t, off = [], [], 0
    for p in nodes:
        all_v.append(v + p[None, :])
        all_t.append(t + off)
        off += len(v)
        if off > 200_000:
            break
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_v))
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_t))
    mesh.compute_vertex_normals()
    return mesh


def _lines_from_edges(
    nodes: np.ndarray, edges: List[tuple[int, int]]
) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(nodes)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(edges, int))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.6, 0.6, 0.6]]), (len(edges), 1))
    )
    return ls


def _poly_len(poly: np.ndarray) -> float:
    if len(poly) < 2:
        return 0.0
    d = np.diff(poly, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def run(
    *,
    cloud_path: str,
    merge_node_radius: float,
    raster_res_px: int,
    save_tag: str,
    normal_bin_edges_deg: Tuple[float, ...],
    orient_sectors: int,
    cache_normals: bool,
    bbox_points: Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = None,
) -> None:
    cfg = cfgmod.RunConfig(
        cloud_path,
        merge_node_radius,
        raster_res_px,
        save_tag,
        normal_bin_edges_deg,
        orient_sectors,
        cache_normals,
        bbox_points,
    )

    cloud = io.load_cloud(cfg.cloud_path)
    cloud = _apply_bbox(cloud, cfg.bbox_points)
    pts = np.asarray(cloud.points)

    model, inl, u, v, n, p0 = plane.find_dominant_plane(cloud)
    frame = PlaneFrame(u=u, v=v, n=n, p0=p0)
    LOG.info(f"plane: inliers={len(inl)} coeffs={model}")

    uv = np.stack([(pts - p0) @ u, (pts - p0) @ v], axis=1)
    span = float(np.max(np.ptp(uv, axis=0)))
    res = span / float(max(cfg.raster_res_px, 1))
    img, uv_min = project2d.rasterize(pts, u, v, p0, res)
    LOG.info(f"raster: shape={img.shape} res={res:.6f} m/px")

    sk = skeleton2d.skeletonize(img)
    nodes_rc, edges, polys_px = nodes_edges.detect(sk)
    nodes_xyz = project2d.pixels_to_world(nodes_rc, u, v, p0, res, uv_min)
    polys_xyz = [project2d.pixels_to_world(p, u, v, p0, res, uv_min) for p in polys_px]
    nodes_xyz, edges = nodes_edges.merge_world(nodes_xyz, edges, cfg.merge_node_radius)
    graph = graph_build.build(nodes_xyz, edges, polys_xyz)
    nav = graph_traverse.EdgeNavigator(graph.edges)

    skel_ls = o3d.geometry.LineSet()
    pts_all, lines, base = [], [], 0
    for poly in polys_xyz:
        if len(poly) < 2:
            continue
        pts_all.extend(poly.tolist())
        lines.extend([[base + i, base + i + 1] for i in range(len(poly) - 1)])
        base += len(poly)
    if pts_all:
        skel_ls.points = o3d.utility.Vector3dVector(np.asarray(pts_all))
        skel_ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, int))
        skel_ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.0, 1.0]]), (len(lines), 1))
        )

    graph_nodes = _sphere_cloud(graph.nodes, r=cfg.merge_node_radius * 0.6)
    graph_edges = _lines_from_edges(graph.nodes, graph.edges)

    def _save() -> None:
        from . import io as _io

        _io.save_graph(graph.nodes, graph.edges, Path(f"{save_tag}_graph.json"))

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    viewer.open_view(
        cloud,
        lambda: meshmod.build_mesh(cloud),
        skel_ls,
        graph_nodes,
        graph_edges,
        graph.nodes,
        nav,
        _save,
        frame,
        {
            "polylines": len(polys_xyz),
            "total_len": sum(_poly_len(p) for p in polys_xyz),
            "avg_width": 0.0,
        },
        {
            "nodes": int(graph.nodes.shape[0]),
            "edges": int(len(graph.edges)),
            "traversal": int(len(graph.edges)),
        },
        cfg,
    )
