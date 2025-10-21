# utils/view.py
"""Visualize 6D poses as points with local axes using Plotly.

Data sources:
- If USE_GENERATED_GRID is True, poses are generated from borunte.grid.build_grid_for_count.
- Otherwise, poses are loaded from a JSON file {name: {x, y, z, rx, ry, rz}}.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

from .config import WORKSPACE_M, TOTAL_POINTS, TCP_DOWN_UVW, DEV_MAX_DEG

# Grid generator
from borunte.grid import build_grid_for_count

USE_GENERATED_GRID = True


@dataclass(frozen=True)
class Pose:
    """One 6D pose with position (x, y, z) and Euler angles (rx, ry, rz) in degrees."""

    name: str
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def euler_rad(self) -> np.ndarray:
        return np.deg2rad([self.rx, self.ry, self.rz], dtype=np.float64)

    def rotation_matrix(self) -> np.ndarray:
        return R.from_euler("xyz", self.euler_rad()).as_matrix()


class PoseLoader:
    """Load poses from a JSON mapping {name: {x, y, z, rx, ry, rz}}."""

    REQUIRED = ("x", "y", "z", "rx", "ry", "rz")

    @staticmethod
    def from_json_obj(data: Mapping[str, Mapping[str, float]]) -> List[Pose]:
        out: List[Pose] = []
        for name, item in data.items():
            vals = {k: float(item[k]) for k in PoseLoader.REQUIRED}
            out.append(Pose(name=name, **vals))  # type: ignore[arg-type]
        out.sort(key=lambda p: p.name)
        return out

    @staticmethod
    def from_json_file(path: Path) -> List[Pose]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Input JSON must be an object {name: pose}")
        return PoseLoader.from_json_obj(raw)


def poses_from_generated_grid() -> List[Pose]:
    """Generate poses using build_grid_for_count and config bounds."""
    pts = build_grid_for_count(
        ws=WORKSPACE_M,
        total=int(TOTAL_POINTS),
    )
    poses: List[Pose] = []
    for i, p in enumerate(pts, 1):
        x, y, z, u, v, w = map(float, p[:6])
        poses.append(Pose(name=f"{i:03d}", x=x, y=y, z=z, rx=u, ry=v, rz=w))
    return poses


class AxisBuilder:
    """Build 3D axis segments for a pose."""

    def __init__(self, scale: float = 30.0, line_width: int = 4):
        self.scale = float(scale)
        self.line_width = int(line_width)

    def axes_for_pose(self, pose: Pose) -> List[go.Scatter3d]:
        Rm = pose.rotation_matrix()
        origin = pose.pos()
        colors = ("red", "green", "blue")
        traces: List[go.Scatter3d] = []
        for i, color in enumerate(colors):
            tip = origin + Rm[:, i] * self.scale
            traces.append(
                go.Scatter3d(
                    x=[origin[0], tip[0]],
                    y=[origin[1], tip[1]],
                    z=[origin[2], tip[2]],
                    mode="lines",
                    line=dict(color=color, width=self.line_width),
                    name=f"{pose.name}_{color}",
                    showlegend=False,
                )
            )
        return traces


def _workspace_box_lines(
    workspace: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> List[go.Scatter3d]:
    """Create line traces for the 12 edges of an axis-aligned box."""
    (x0, x1), (y0, y1), (z0, z1) = workspace
    xs = [x0, x1]
    ys = [y0, y1]
    zs = [z0, z1]

    edges = [
        ((xs[0], ys[0], zs[0]), (xs[1], ys[0], zs[0])),
        ((xs[0], ys[1], zs[0]), (xs[1], ys[1], zs[0])),
        ((xs[0], ys[0], zs[1]), (xs[1], ys[0], zs[1])),
        ((xs[0], ys[1], zs[1]), (xs[1], ys[1], zs[1])),
        ((xs[0], ys[0], zs[0]), (xs[0], ys[1], zs[0])),
        ((xs[1], ys[0], zs[0]), (xs[1], ys[1], zs[0])),
        ((xs[0], ys[0], zs[1]), (xs[0], ys[1], zs[1])),
        ((xs[1], ys[0], zs[1]), (xs[1], ys[1], zs[1])),
        ((xs[0], ys[0], zs[0]), (xs[0], ys[0], zs[1])),
        ((xs[1], ys[0], zs[0]), (xs[1], ys[0], zs[1])),
        ((xs[0], ys[1], zs[0]), (xs[0], ys[1], zs[1])),
        ((xs[1], ys[1], zs[0]), (xs[1], ys[1], zs[1])),
    ]

    traces: List[go.Scatter3d] = []
    for a, b in edges:
        traces.append(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line=dict(color="gray", width=2),
                name="workspace",
                showlegend=False,
            )
        )
    return traces


class PlotBuilder:
    """Create a Plotly figure with pose points, local axes, and an optional workspace box."""

    def __init__(
        self,
        poses: Sequence[Pose],
        axis_scale: float = 30.0,
        marker_size: int = 4,
        show_text: bool = True,
        workspace: Optional[
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        ] = None,
    ):
        self.poses = list(poses)
        self.axis_builder = AxisBuilder(scale=axis_scale)
        self.marker_size = int(marker_size)
        self.show_text = bool(show_text)
        self.workspace = workspace

    def build(self) -> go.Figure:
        if len(self.poses) == 0:
            return go.Figure()

        points = np.array([p.pos() for p in self.poses], dtype=np.float64)
        labels = [p.name for p in self.poses]

        scat = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers+text" if self.show_text else "markers",
            text=labels if self.show_text else None,
            textposition="top center",
            marker=dict(size=self.marker_size, color="black"),
            name="poses",
            showlegend=False,
        )

        fig = go.Figure([scat])

        for p in self.poses:
            for axis in self.axis_builder.axes_for_pose(p):
                fig.add_trace(axis)

        if self.workspace is not None:
            for edge in _workspace_box_lines(self.workspace):
                fig.add_trace(edge)

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )
        return fig


def _env_path(key: str, default: str) -> Path:
    val = os.environ.get(key, default)
    return Path(val).expanduser().resolve()


if __name__ == "__main__":
    # Defaults can be overridden with environment variables:
    #   VIEW_INPUT=captures/20251009_150337/poses.json
    #   VIEW_OUTPUT=poses_plot.html
    inp = _env_path("VIEW_INPUT", "captures/20251009_150337/poses.json")
    outp = _env_path("VIEW_OUTPUT", "poses_plot.html")

    if USE_GENERATED_GRID:
        poses = poses_from_generated_grid()
        workspace = WORKSPACE_M
    else:
        data = json.loads(Path(inp).read_text(encoding="utf-8"))
        poses = PoseLoader.from_json_obj(data)  # will raise if not an object
        workspace = WORKSPACE_M  # still show box for context

    fig = PlotBuilder(
        poses,
        axis_scale=30.0,
        marker_size=5,
        show_text=True,
        workspace=workspace,
    ).build()
    fig.write_html(str(outp))
