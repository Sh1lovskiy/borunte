# vision/io/series.py
"""Input series handling for RGB-D sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import imageio.v2 as imageio
import numpy as np

from utils.io import resolve_paths


@dataclass
class FramePaths:
    """Container for paired depth and color paths."""

    stem: str
    depth: Path
    rgb: Path


@dataclass
class FrameData:
    """Loaded RGB-D frame."""

    stem: str
    depth: np.ndarray
    color: np.ndarray


def _expand_inputs(inputs: Sequence[Path | str]) -> List[Path]:
    if not inputs:
        raise ValueError("No inputs supplied")
    resolved = resolve_paths(inputs)
    files = [path for path in resolved if path.is_file()]
    if not files:
        raise FileNotFoundError("No files found for provided inputs")
    return files


def _stem_and_role(path: Path) -> tuple[str, str] | None:
    name = path.stem
    if "_" not in name:
        return None
    base, role = name.rsplit("_", 1)
    if role not in {"depth", "rgb"}:
        return None
    return base, role


def collect_series(inputs: Sequence[Path | str]) -> List[FramePaths]:
    """Collect paired depth and RGB image paths."""
    files = _expand_inputs(inputs)
    groups: dict[str, dict[str, Path]] = {}
    for file in files:
        parsed = _stem_and_role(file)
        if parsed is None:
            continue
        base, role = parsed
        groups.setdefault(base, {})[role] = file
    frames: List[FramePaths] = []
    for base, mapping in sorted(groups.items()):
        if {"depth", "rgb"}.issubset(mapping):
            frames.append(FramePaths(stem=base, depth=mapping["depth"], rgb=mapping["rgb"]))
    if not frames:
        raise ValueError("No paired RGB-D frames found")
    return frames


def load_frame(frame: FramePaths) -> FrameData:
    """Load RGB-D frame data."""
    depth = imageio.imread(frame.depth)
    color = imageio.imread(frame.rgb)
    return FrameData(stem=frame.stem, depth=np.asarray(depth), color=np.asarray(color))


def load_series(inputs: Sequence[Path | str]) -> List[FrameData]:
    """Load a sequence of RGB-D frames."""
    frames = collect_series(inputs)
    return [load_frame(frame) for frame in frames]


__all__ = ["FramePaths", "FrameData", "collect_series", "load_series", "load_frame"]
