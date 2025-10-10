# borunte/grid.py
"""Uniform 3D grid builder with ordered scan (X -> Y -> Z).

- build_grid_for_count(...): pick near-cubic (nx, ny, nz) so that nx*ny*nz >= total.
- build_grid_by_counts(...): exact nx*ny*nz lattice.
Both return [X, Y, Z, U, V, W] and apply per-point UVW jitter.

Key guarantees:
- Full workspace coverage along X, Y, Z (inclusive endpoints).
- Strict traversal order: X -> Y -> Z (no serpentine).
- No tail trimming by total: the whole lattice is returned.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .config import TCP_DOWN_UVW, DEV_MAX_DEG, GRID_MIN_COUNTS
from utils.logger import Logger

_log = Logger.get_logger()


# ---------- helpers ----------


def _axis_linspace(a0: float, a1: float, n: int) -> np.ndarray:
    """Inclusive linspace that respects ascending/descending ranges."""
    if n <= 1:
        return np.array([float(a0)], dtype=np.float64)
    return np.linspace(a0, a1, num=int(n), dtype=np.float64)


def _counts_from_total_simple(
    ws: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total: int,
    min_counts: Tuple[int, int, int] = GRID_MIN_COUNTS,
    product_granularity: int | None = None,
    overshoot_limit: float = 0.35,
) -> Tuple[int, int, int]:
    """
    Pick (nx,ny,nz) with near-uniform spacing and inclusive endpoints.
    Guarantee nx*ny*nz >= total. If product_granularity is set (e.g. 5
    or 10), try to snap product to ceil(total / g) * g, while keeping
    spacing balanced and overshoot within overshoot_limit.
    """
    assert total >= 1
    (x0, x1), (y0, y1), (z0, z1) = ws
    lx, ly, lz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    eps = 1e-9
    lx, ly, lz = max(lx, eps), max(ly, eps), max(lz, eps)

    vol = lx * ly * lz
    if vol <= eps:
        nx = max(int(min_counts[0]), 1)
        ny = max(int(min_counts[1]), 1)
        nz = max(int(np.ceil(total / max(nx * ny, 1))), int(min_counts[2]), 1)
        return nx, ny, nz

    step = (vol / float(total)) ** (1.0 / 3.0)

    def n_for(L: float, mn: int) -> int:
        pts = int(np.floor(L / step)) + 1
        return max(pts, int(mn), 1)

    nx, ny, nz = (
        n_for(lx, min_counts[0]),
        n_for(ly, min_counts[1]),
        n_for(lz, min_counts[2]),
    )

    def spacing(n: int, L: float) -> float:
        return L / max(n - 1, 1)

    def prod(a: int, b: int, c: int) -> int:
        return int(a) * int(b) * int(c)

    for _ in range(256):
        if prod(nx, ny, nz) >= total:
            break
        sx, sy, sz = spacing(nx, lx), spacing(ny, ly), spacing(nz, lz)
        k = int(np.argmax([sx, sy, sz]))
        if k == 0:
            nx += 1
        elif k == 1:
            ny += 1
        else:
            nz += 1

    if product_granularity and product_granularity >= 2:
        target = int(np.ceil(total / float(product_granularity)) * product_granularity)
        for _ in range(256):
            if prod(nx, ny, nz) >= target:
                break
            sx, sy, sz = spacing(nx, lx), spacing(ny, ly), spacing(nz, lz)
            k = int(np.argmax([sx, sy, sz]))
            if k == 0:
                nx += 1
            elif k == 1:
                ny += 1
            else:
                nz += 1

        for _ in range(512):
            p = prod(nx, ny, nz)
            if p <= target:
                break
            candidates = []
            for i, (n, L, mn) in enumerate(
                (
                    (nx, lx, min_counts[0]),
                    (ny, ly, min_counts[1]),
                    (nz, lz, min_counts[2]),
                )
            ):
                if n - 1 >= max(1, int(mn)):
                    # проверим, что продукт не упадет ниже таргета
                    new = [nx, ny, nz]
                    new[i] = n - 1
                    if prod(*new) >= target:
                        harm = spacing(n - 1, L) - spacing(n, L)
                        candidates.append((harm, i))
            if not candidates:
                break
            _, axis = min(candidates, key=lambda t: t[0])
            if axis == 0:
                nx -= 1
            elif axis == 1:
                ny -= 1
            else:
                nz -= 1

    if prod(nx, ny, nz) > int(np.ceil((1.0 + overshoot_limit) * total)):
        _log.warn(
            f"[GRID] product overshoot {prod(nx, ny, nz)} for total={total}; "
            f"consider smaller product_granularity or higher total"
        )
    return int(nx), int(ny), int(nz)


def _per_point_jitter(n: int, dev: float, rng: np.random.Generator):
    """Random UVW jitter per point: choose 1..3 axes independently."""
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


# ---------- public API ----------


def build_grid_for_count(
    ws: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total: int,
    rx_base: float = TCP_DOWN_UVW[0],
    ry_base: float = TCP_DOWN_UVW[1],
    rz_base: float = TCP_DOWN_UVW[2],
    seed: int = 42,
    dev_max_deg: float = DEV_MAX_DEG,
    product_granularity: int = 5,
    overshoot_limit: float = 0.2,
) -> List[List[float]]:
    """
    Ordered 3D grid [X, Y, Z, U, V, W] with near-uniform spacing.
    - Computes (nx, ny, nz) so that nx*ny*nz >= total.
    - Returns the FULL lattice (no trimming or padding).
    - Traversal order: X -> Y -> Z.
    """
    rng = np.random.default_rng(int(seed))
    nx, ny, nz = _counts_from_total_simple(
        ws, total, GRID_MIN_COUNTS, product_granularity, overshoot_limit
    )

    Xs = _axis_linspace(ws[0][0], ws[0][1], nx)
    Ys = _axis_linspace(ws[1][0], ws[1][1], ny)
    Zs = _axis_linspace(ws[2][0], ws[2][1], nz)

    poses: List[List[float]] = []
    for x in Xs:
        for y in Ys:
            n_line = int(Zs.shape[0])
            du, dv, dw = _per_point_jitter(n_line, dev_max_deg, rng)
            xs = np.full(n_line, float(x))
            ys = np.full(n_line, float(y))
            us = np.full(n_line, float(rx_base)) + du
            vs = np.full(n_line, float(ry_base)) + dv
            ws_ = np.full(n_line, float(rz_base)) + dw
            block = np.stack([xs, ys, Zs, us, vs, ws_], axis=1)
            poses.extend(block.tolist())

    _log.tag(
        "GRID",
        (
            f"nx={nx} ny={ny} nz={nz} total={len(poses)} "
            f"box=(({ws[0][0]:.1f},{ws[0][1]:.1f}),"
            f"({ws[1][0]:.1f},{ws[1][1]:.1f}),"
            f"({ws[2][0]:.1f},{ws[2][1]:.1f})) order=X->Y->Z"
        ),
    )
    return poses


def build_grid_by_counts(
    ws: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    nx: int,
    ny: int,
    nz: int,
    rx_base: float = TCP_DOWN_UVW[0],
    ry_base: float = TCP_DOWN_UVW[1],
    rz_base: float = TCP_DOWN_UVW[2],
    seed: int = 42,
    dev_max_deg: float = DEV_MAX_DEG,
) -> List[List[float]]:
    """
    Build ordered 3D grid with exact counts; per-point UVW jitter.
    Traversal order: X -> Y -> Z. Inclusive endpoints on each axis.
    """
    assert nx >= 1 and ny >= 1 and nz >= 1, "counts must be >= 1"
    rng = np.random.default_rng(int(seed))

    Xs = _axis_linspace(ws[0][0], ws[0][1], nx)
    Ys = _axis_linspace(ws[1][0], ws[1][1], ny)
    Zs = _axis_linspace(ws[2][0], ws[2][1], nz)

    poses: List[List[float]] = []
    for x in Xs:
        for y in Ys:
            n_line = int(Zs.shape[0])
            du, dv, dw = _per_point_jitter(n_line, dev_max_deg, rng)
            xs = np.full(n_line, float(x))
            ys = np.full(n_line, float(y))
            us = np.full(n_line, float(rx_base)) + du
            vs = np.full(n_line, float(ry_base)) + dv
            ws_ = np.full(n_line, float(rz_base)) + dw
            block = np.stack([xs, ys, Zs, us, vs, ws_], axis=1)
            poses.extend(block.tolist())

    _log.tag("GRID", f"counts nx={nx} ny={ny} nz={nz} total={len(poses)} order=X->Y->Z")
    return poses
