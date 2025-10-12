from __future__ import annotations

import numpy as np
from typing import Tuple


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def colors_from_deviation(N: np.ndarray, n: np.ndarray, bins_deg: Tuple[float, ...]):
    """Color by deviation from plane normal n (degrees)."""
    Nn = _normalize(N)
    n = n / (np.linalg.norm(n) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(Nn @ n, -1.0, 1.0)))
    bins = np.asarray(bins_deg, float)
    idx = np.clip(np.digitize(ang, bins) - 1, 0, len(bins) - 2)
    t = (ang - bins[idx]) / np.maximum(bins[idx + 1] - bins[idx], 1e-9)
    cols = np.stack([t, 1.0 - t, 0.2 + 0 * t], axis=1)
    counts = [(idx == i).sum() for i in range(len(bins) - 1)]
    stats = {"counts": counts, "total": len(N), "bins_deg": list(bins)}
    return cols, stats


def colors_from_orientation(
    N: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray, sectors: int
):
    """Color by azimuth angle in (u,v) plane."""
    Np = N - (N @ n)[:, None] * n[None, :]
    Np = _normalize(Np)
    ang = np.degrees(np.arctan2(Np @ v, Np @ u)) % 360.0
    width = 360.0 / max(sectors, 1)
    idx = (ang // width).astype(int)
    hue = (idx + 0.5) / float(sectors)
    cols = np.stack(
        [
            np.sin(2 * np.pi * hue) * 0.5 + 0.5,
            np.cos(2 * np.pi * hue) * 0.5 + 0.5,
            0.3 + 0 * hue,
        ],
        axis=1,
    )
    counts = [(idx == i).sum() for i in range(sectors)]
    stats = {"counts": counts, "total": len(N), "sectors": sectors}
    return cols, stats
