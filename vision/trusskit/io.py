from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List
import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("tk.io")


def load_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from PLY/PCD; ensure non-empty."""
    p = Path(path)
    LOG.info(f"load cloud: {p}")
    if not p.exists():
        raise FileNotFoundError(f"cloud not found: {p}")
    cloud = o3d.io.read_point_cloud(str(p))
    if len(cloud.points) == 0:
        raise RuntimeError(f"empty cloud: {p}")
    LOG.info(f"cloud points={len(cloud.points)}")
    return cloud


def save_skeleton(polys: List[np.ndarray], path: Path) -> None:
    """Save polylines (list of Nx3) to JSON."""
    LOG.info(f"save skeleton polylines: {path}")
    data = [[p.tolist() for p in polys]]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_graph(nodes: np.ndarray, edges: List[tuple[int, int]], path: Path) -> None:
    """Save graph nodes/edges to JSON."""
    LOG.info(f"save graph: {path}")
    out = {
        "nodes": nodes.tolist(),
        "edges": [[int(a), int(b)] for a, b in edges],
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
