from __future__ import annotations

import numpy as np
from typing import Tuple


def rasterize(
    points: np.ndarray, u: np.ndarray, v: np.ndarray, p0: np.ndarray, res: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Rasterize points onto (u,v) plane with resolution 'res' meters/pixel."""
    d = points - p0[None, :]
    uv = np.stack([d @ u, d @ v], axis=1)
    uv_min = uv.min(axis=0)
    uv0 = (uv - uv_min) / max(res, 1e-9)
    rc = np.floor(uv0[:, ::-1]).astype(int)  # (r,c) = (v,u)
    H = int(rc[:, 0].max() + 1)
    W = int(rc[:, 1].max() + 1)
    img = np.zeros((H, W), np.uint8)
    img[rc[:, 0], rc[:, 1]] = 1
    return img, (float(uv_min[0]), float(uv_min[1]))


def pixels_to_world(
    path_rc: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p0: np.ndarray,
    res: float,
    uv_min: tuple[float, float],
) -> np.ndarray:
    """Map polyline pixels (r,c) back to 3D world coords."""
    u0, v0 = uv_min
    uv = np.empty((len(path_rc), 2), float)
    uv[:, 0] = (path_rc[:, 1].astype(float) * res) + u0
    uv[:, 1] = (path_rc[:, 0].astype(float) * res) + v0
    return p0[None, :] + uv[:, 0:1] * u + uv[:, 1:2] * v
