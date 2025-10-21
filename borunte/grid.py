# borunte/grid.py
"""Workspace grid generation utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.config import (
    DEV_MAX_DEG,
    GRID_MIN_COUNTS,
    TCP_DOWN_UVW,
)
from utils.logger import get_logger

from .config import BorunteConfig

_log = get_logger()


def _axis_linspace(a0: float, a1: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([float(a0)], dtype=np.float64)
    return np.linspace(a0, a1, num=int(n), dtype=np.float64)


def _counts_from_total(
    ws: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total: int,
    min_counts: Tuple[int, int, int],
    product_granularity: int,
    overshoot_limit: float,
) -> Tuple[int, int, int]:
    assert total >= 1
    (x0, x1), (y0, y1), (z0, z1) = ws
    lx, ly, lz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    eps = 1e-9
    lx, ly, lz = max(lx, eps), max(ly, eps), max(lz, eps)

    step = (lx * ly * lz / float(total)) ** (1.0 / 3.0)

    def n_for(length: float, minimum: int) -> int:
        pts = int(np.floor(length / step)) + 1
        return max(pts, int(minimum), 1)

    nx, ny, nz = (
        n_for(lx, min_counts[0]),
        n_for(ly, min_counts[1]),
        n_for(lz, min_counts[2]),
    )

    def spacing(count: int, length: float) -> float:
        return length / max(count - 1, 1)

    def product(a: int, b: int, c: int) -> int:
        return int(a) * int(b) * int(c)

    for _ in range(256):
        if product(nx, ny, nz) >= total:
            break
        sx, sy, sz = spacing(nx, lx), spacing(ny, ly), spacing(nz, lz)
        axis = int(np.argmax([sx, sy, sz]))
        if axis == 0:
            nx += 1
        elif axis == 1:
            ny += 1
        else:
            nz += 1

    if product_granularity >= 2:
        target = int(np.ceil(total / float(product_granularity)) * product_granularity)
        for _ in range(256):
            if product(nx, ny, nz) >= target:
                break
            sx, sy, sz = spacing(nx, lx), spacing(ny, ly), spacing(nz, lz)
            axis = int(np.argmax([sx, sy, sz]))
            if axis == 0:
                nx += 1
            elif axis == 1:
                ny += 1
            else:
                nz += 1

        for _ in range(512):
            prod_val = product(nx, ny, nz)
            if prod_val <= target:
                break
            candidates = []
            for idx, (count, length, minimum) in enumerate(
                (
                    (nx, lx, min_counts[0]),
                    (ny, ly, min_counts[1]),
                    (nz, lz, min_counts[2]),
                )
            ):
                if count - 1 >= max(1, int(minimum)):
                    new_counts = [nx, ny, nz]
                    new_counts[idx] = count - 1
                    if product(*new_counts) >= target:
                        harm = spacing(count - 1, length) - spacing(count, length)
                        candidates.append((harm, idx))
            if not candidates:
                break
            _, axis = min(candidates, key=lambda t: t[0])
            if axis == 0:
                nx -= 1
            elif axis == 1:
                ny -= 1
            else:
                nz -= 1

    prod_val = product(nx, ny, nz)
    if prod_val > int(np.ceil((1.0 + overshoot_limit) * total)):
        _log.tag(
            "GRID",
            f"product overshoot {prod_val} for total={total}"
            " consider adjusting product_granularity",
            level="warning",
        )
    return int(nx), int(ny), int(nz)


def _per_point_jitter(n: int, dev: float, rng: np.random.Generator):
    du = rng.uniform(-dev, dev, size=n)
    dv = rng.uniform(-dev, dev, size=n)
    dw = rng.uniform(-dev, dev, size=n)
    mask = rng.random((n, 3)) < 0.5
    none = ~(mask.any(axis=1))
    if np.any(none):
        pick = rng.integers(0, 3, size=int(none.sum()))
        mask[none, pick] = True
    du[~mask[:, 0]] = 0.0
    dv[~mask[:, 1]] = 0.0
    dw[~mask[:, 2]] = 0.0
    return du, dv, dw


def build_grid_for_count(
    ws: (
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None
    ) = None,
    total: int | None = None,
) -> List[List[float]]:
    """Return an ordered XYZUVW lattice covering the workspace."""

    rng = np.random.default_rng(42)
    nx, ny, nz = _counts_from_total(
        ws,
        total,
        GRID_MIN_COUNTS,
        product_granularity=5,
        overshoot_limit=0.2,
    )

    xs = _axis_linspace(ws[0][0], ws[0][1], nx)
    ys = _axis_linspace(ws[1][0], ws[1][1], ny)
    zs = _axis_linspace(ws[2][0], ws[2][1], nz)

    poses: List[List[float]] = []
    for x in xs:
        for y in ys:
            count = int(zs.shape[0])
            du, dv, dw = _per_point_jitter(count, DEV_MAX_DEG, rng)
            block = np.stack(
                [
                    np.full(count, float(x)),
                    np.full(count, float(y)),
                    zs,
                    np.full(count, float(TCP_DOWN_UVW[0])) + du,
                    np.full(count, float(TCP_DOWN_UVW[1])) + dv,
                    np.full(count, float(TCP_DOWN_UVW[2])) + dw,
                ],
                axis=1,
            )
            poses.extend(block.tolist())

    _log.tag("GRID", f"counts=({nx},{ny},{nz}) total={len(poses)} target={total}")
    return poses


__all__ = ["build_grid"]
