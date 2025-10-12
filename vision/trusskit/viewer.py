"""Open3D interactive viewer for the Truss pipeline."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import open3d as o3d
import time

from utils.logger import Logger
from utils.keyboard import KeyHandler
from . import normals, regions

try:  # make RunConfig optional to avoid hard import failures
    from .config import RunConfig  # type: ignore
except Exception:  # fallback for older configs without RunConfig
    from typing import Any as RunConfig  # type: ignore
from .transforms import PlaneFrame

LOG = Logger.get_logger("tk.viewer")


def _toggle_group(
    vis: o3d.visualization.Visualizer,
    geoms: Dict[str, List],
    visible: Dict[str, bool],
    name: str,
) -> None:
    cur = visible.get(name, False)
    visible[name] = not cur
    for g in geoms.get(name, []):
        try:
            if visible[name]:
                vis.add_geometry(g, reset_bounding_box=False)
            else:
                vis.remove_geometry(g, reset_bounding_box=False)
        except Exception:
            pass
    vis.update_renderer()
    LOG.info(f"toggle {name} -> {visible[name]}")


def open_view(
    cloud: o3d.geometry.PointCloud,
    build_mesh: Callable[[], o3d.geometry.TriangleMesh],
    skeleton_ls: o3d.geometry.LineSet,
    graph_nodes: o3d.geometry.TriangleMesh,
    graph_edges: o3d.geometry.LineSet,
    nodes_xyz: np.ndarray,
    navigator,
    save_cb: Callable[[], None],
    frame: PlaneFrame,
    skel_stats: dict,
    graph_stats: dict,
    cfg: RunConfig,
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TrussKit", width=1024, height=768)
    vis.add_geometry(cloud)
    vis.get_render_option().line_width = 2.0
    geoms = {
        "cloud": [cloud],
        "mesh": [],
        "skeleton": [skeleton_ls],
        "graph": [graph_nodes, graph_edges],
        "highlight": [],
        "side": [],
    }
    visible = {
        "cloud": True,
        "mesh": False,
        "skeleton": True,
        "graph": True,
        "side": False,
    }
    highlight = o3d.geometry.LineSet()
    geoms["highlight"].append(highlight)

    mesh_cache: List[o3d.geometry.TriangleMesh] = []
    norm_cache: Dict[str, np.ndarray] = {}
    color_cache: Dict[Tuple[str, str], Tuple[o3d.geometry.Geometry, dict]] = {}
    side_mode = 0  # 0 off, 1 deviation, 2 orientation

    def _ensure_mesh() -> None:
        if mesh_cache:
            return
        t0 = time.time()
        m = build_mesh()
        dt = time.time() - t0
        mesh_cache.append(m)
        geoms["mesh"] = [m]
        LOG.info(
            f"mesh built: {len(m.vertices)}V {len(m.triangles)}T " f"time={dt:.3f}s"
        )

    def toggle_mesh() -> None:
        _ensure_mesh()
        if visible["mesh"]:
            _toggle_group(vis, geoms, visible, "mesh")
            if not visible["cloud"]:
                _toggle_group(vis, geoms, visible, "cloud")
        else:
            if visible["cloud"]:
                _toggle_group(vis, geoms, visible, "cloud")
            _toggle_group(vis, geoms, visible, "mesh")

    def toggle_skeleton() -> None:
        _toggle_group(vis, geoms, visible, "skeleton")
        if visible["skeleton"]:
            LOG.info(
                f"skeleton: {skel_stats['polylines']} polys "
                f"len={skel_stats['total_len']:.3f}m "
                f"avg_width={skel_stats['avg_width']:.3f}m"
            )

    def toggle_graph() -> None:
        _toggle_group(vis, geoms, visible, "graph")
        if visible["graph"]:
            if highlight not in geoms["graph"]:
                geoms["graph"].append(highlight)
                vis.add_geometry(highlight, reset_bounding_box=False)
            LOG.info(
                f"graph: {graph_stats['nodes']} nodes " f"{graph_stats['edges']} edges"
            )
        else:
            if highlight in geoms["graph"]:
                vis.remove_geometry(highlight, reset_bounding_box=False)
                geoms["graph"].remove(highlight)

    def save() -> None:
        save_cb()
        LOG.info("saved outputs")

    def color_by_deviation() -> None:
        if not mesh_cache:
            _ensure_mesh()
        m = mesh_cache[0]
        N = normals.ensure_normals(m)
        cols, _ = regions.colors_from_deviation(
            N, frame.n, tuple(cfg.normal_bin_edges_deg)
        )
        m.vertex_colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(m)

    def color_by_orientation() -> None:
        if not mesh_cache:
            _ensure_mesh()
        m = mesh_cache[0]
        N = normals.ensure_normals(m)
        cols, _ = regions.colors_from_orientation(
            N, frame.u, frame.v, frame.n, cfg.orient_sectors
        )
        m.vertex_colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(m)

    KeyHandler(vis).register(
        {
            "2": toggle_mesh,
            "3": toggle_skeleton,
            "4": toggle_graph,
            "S": save,
            "1": color_by_deviation,
            "0": color_by_orientation,
        }
    )
    vis.run()
    vis.destroy_window()
