# borunte/grid.py
"""Grid utilities for robot capture paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from borunte.config import BorunteConfig, DEFAULT_CONFIG


@dataclass(slots=True)
class GridPlan:
    origin: np.ndarray
    points: List[np.ndarray]

    def as_tuples(self) -> List[Tuple[float, float, float]]:
        return [tuple(point.tolist()) for point in self.points]


def build_grid(
    *,
    config: BorunteConfig | None = None,
    columns: int,
    rows: int,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> GridPlan:
    cfg = config or DEFAULT_CONFIG
    base = np.array(origin, dtype=float)
    points: List[np.ndarray] = []
    for row in range(rows):
        for col in range(columns):
            offset = np.array([
                col * cfg.grid_spacing,
                row * cfg.grid_spacing,
                0.0,
            ])
            points.append(base + offset)
    return GridPlan(origin=base, points=points)


__all__ = ["GridPlan", "build_grid"]
