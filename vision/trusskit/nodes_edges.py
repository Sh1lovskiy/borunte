from __future__ import annotations

from typing import List, Tuple

import numpy as np
import cv2

from utils.logger import Logger

LOG = Logger.get_logger("tk.nodes")


def _node_mask(sk: np.ndarray) -> np.ndarray:
    """Nodes where degree != 2 (endpoints or junctions)."""
    k = np.ones((3, 3), np.uint8)
    deg = cv2.filter2D(sk, -1, k, borderType=cv2.BORDER_CONSTANT)
    deg -= sk  # exclude center
    nodes = (sk > 0) & (deg != 2)
    return nodes.astype(np.uint8)


def detect(
    sk: np.ndarray,
) -> tuple[np.ndarray, List[Tuple[int, int]], List[np.ndarray]]:
    """Extract nodes(rc), edges(u,v), polylines(rc) from 1px skeleton."""
    sk = (sk > 0).astype(np.uint8)
    LOG.info(f"skeleton pixels={int(sk.sum())}")
    nm = _node_mask(sk)
    n_rc = np.column_stack(np.nonzero(nm))

    # Cut nodes out to split edges into components
    edge_img = (sk & (1 - nm)).astype(np.uint8)
    nlab, lab = cv2.connectedComponents(edge_img, connectivity=8)
    edges, polys = [], []
    for i in range(1, nlab):
        ys, xs = np.nonzero(lab == i)
        comp = np.column_stack([ys, xs])
        if comp.size == 0:
            continue
        # Find adjacent nodes (pixels around the component)
        pad = cv2.dilate((lab == i).astype(np.uint8), np.ones((3, 3), np.uint8), 1)
        touching = np.column_stack(np.nonzero(pad & nm))
        # Map touching pixels to nearest node index
        if len(touching) == 0:
            continue
        d = ((touching[:, None, :] - n_rc[None, :, :]) ** 2).sum(axis=2)
        near = np.argmin(d, axis=1)
        uniq = np.unique(near)
        if len(uniq) < 2:
            continue
        u, v = uniq[:2]
        edges.append((int(u), int(v)))
        polys.append(comp.astype(int))
    LOG.info(f"raw nodes={len(n_rc)} edges={len(edges)}")
    return n_rc, edges, polys


def merge_world(
    nodes_xyz: np.ndarray, edges: List[Tuple[int, int]], radius: float
) -> tuple[np.ndarray, List[Tuple[int, int]]]:
    """Greedy merge nodes closer than radius (meters) in 3D."""
    if len(nodes_xyz) == 0:
        return nodes_xyz, edges
    pts = nodes_xyz.copy()
    parent = np.arange(len(pts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Union-close pairs
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    close = np.argwhere((d > 0) & (d <= radius))
    for i, j in close:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # New nodes: average per set
    sets = {}
    for i in range(len(pts)):
        r = find(i)
        sets.setdefault(r, []).append(i)
    new_idx = {}
    new_pts = []
    for k, idxs in sets.items():
        new_idx.update({i: len(new_pts) for i in idxs})
        new_pts.append(pts[idxs].mean(axis=0))
    new_pts = np.asarray(new_pts, float)

    # Rewire edges
    new_edges = []
    for u, v in edges:
        a = new_idx.get(int(u), None)
        b = new_idx.get(int(v), None)
        if a is None or b is None or a == b:
            continue
        e = (min(a, b), max(a, b))
        if e not in new_edges:
            new_edges.append(e)
    LOG.info(f"merge: {len(pts)}→{len(new_pts)} nodes; edges={len(new_edges)}")
    return new_pts, new_edges
