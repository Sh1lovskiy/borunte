# calib/io_utils.py
"""File and JSON utilities for offline calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from utils.logger import Logger

_log = Logger.get_logger()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_rgb_indices(img_dir: Path) -> List[int]:
    idxs: List[int] = []
    for p in sorted(img_dir.glob("*_rgb.png")):
        stem = p.name.replace("_rgb.png", "")
        try:
            idxs.append(int(stem))
        except Exception:
            continue
    return sorted(idxs)


def resolve_dataset_paths(base: Path) -> Tuple[Path, Path, Path]:
    """
    Resolve image directory and paths to rs2_params.json and poses.json.
    Accept either:
    - a directory that contains NNN_rgb.png files, or
    - a session root that contains preview_single with images.
    """
    img_dir = base
    rs2 = base / "rs2_params.json"
    poses = base / "poses.json"

    if not any(img_dir.glob("*_rgb.png")):
        cand = base / "preview_single"
        if cand.exists() and any(cand.glob("*_rgb.png")):
            img_dir = cand

    if not rs2.exists():
        rs2 = base.parent / "rs2_params.json"
    if not poses.exists():
        poses = base.parent / "poses.json"

    if not rs2.exists() or not poses.exists():
        _log.tag("CALIB", f"dataset layout not found under {base}", "error")
        raise FileNotFoundError("rs2_params.json or poses.json not found")

    return img_dir, rs2, poses
