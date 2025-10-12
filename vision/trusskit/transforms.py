from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PlaneFrame:
    """Orthonormal (u,v,n) basis anchored at p0."""

    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    p0: np.ndarray

    def world_to_plane(self, pts: np.ndarray) -> np.ndarray:
        """Project 3D points to (u,v) plane coords."""
        d = pts - self.p0[None, :]
        return np.stack([d @ self.u, d @ self.v], axis=1)

    def plane_to_world(self, uv: np.ndarray) -> np.ndarray:
        """Lift (u,v) plane coords back to 3D."""
        return self.p0[None, :] + uv[:, 0:1] * self.u + uv[:, 1:2] * self.v
