# borunte/cap_session.py
"""Filesystem helpers for capture sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs

from utils.logger import Logger

from .config import BORUNTE_CONFIG, BorunteConfig

_log = Logger.get_logger()


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class CaptureSession:
    """Create timestamped directories and save images, depth, and poses."""

    def __init__(
        self,
        config: BorunteConfig = BORUNTE_CONFIG,
        root_dir: Optional[Path] = None,
    ) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(root_dir) if root_dir is not None else config.capture_root
        self.root = base / ts
        self.root.mkdir(parents=True, exist_ok=True)
        self.poses_path = self.root / "poses.json"
        self.poses: Dict[str, Dict[str, float]] = {}
        self.idx = 0
        _log.tag("SESS", f"dir={self.root}")

    def index_name(self) -> str:
        return f"{self.idx:03d}"

    def next(self) -> None:
        self.idx += 1

    def save_params_json(
        self,
        profile: rs.pipeline_profile,
        out_json: Path,
        applied_disparity: int,
        decimation_mag: int,
    ) -> None:
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

        def intr_to_dict(intr: rs.intrinsics) -> Dict[str, float]:
            return {
                "width": int(intr.width),
                "height": int(intr.height),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "model": str(intr.model),
                "coeffs": [float(c) for c in intr.coeffs[:5]],
            }

        def extr_to_dict(extr: rs.extrinsics) -> Dict[str, list[float]]:
            rot = [float(x) for x in extr.rotation]
            trans = [float(x) for x in extr.translation]
            return {
                "rotation": [rot[0:3], rot[3:6], rot[6:9]],
                "translation": trans,
            }

        device = profile.get_device()
        depth_scale = float(device.first_depth_sensor().get_depth_scale())

        data = {
            "depth_scale": depth_scale,
            "intrinsics": {
                "depth": intr_to_dict(depth_profile.get_intrinsics()),
                "color": intr_to_dict(color_profile.get_intrinsics()),
            },
            "extrinsics": {
                "depth_to_color": extr_to_dict(depth_profile.get_extrinsics_to(color_profile)),
                "color_to_depth": extr_to_dict(color_profile.get_extrinsics_to(depth_profile)),
            },
            "processing": {
                "applied_disparity_shift": int(applied_disparity),
                "decimation_magnitude": int(decimation_mag),
            },
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out_json, data)
        _log.tag("RS2", f"params -> {out_json}")

    def save_rgb_depth(self, subdir: Path, name: str, rgb: np.ndarray, depth_m: np.ndarray) -> None:
        subdir.mkdir(parents=True, exist_ok=True)
        rgb_path = subdir / f"{name}_rgb.png"
        depth_path = subdir / f"{name}_depth.npy"
        cv2.imwrite(str(rgb_path), rgb)
        np.save(str(depth_path), depth_m.astype(np.float32))
        _log.tag("SAVE", f"{subdir.name}/{rgb_path.name}, {depth_path.name}")

    def update_pose(self, name: str, pose6: Dict[str, float]) -> None:
        self.poses[name] = pose6
        _atomic_write_json(self.poses_path, self.poses)
        _log.tag("POSE", f"updated index {name}")

    def save_preview_snapshot(
        self,
        subdir: Path,
        name: str,
        rgb: Optional[np.ndarray],
        depth_m: Optional[np.ndarray],
    ) -> None:
        subdir.mkdir(parents=True, exist_ok=True)
        if rgb is not None:
            cv2.imwrite(str(subdir / f"{name}_rgb.png"), rgb)
        if depth_m is not None:
            np.save(str(subdir / f"{name}_depth.npy"), depth_m.astype(np.float32))
        _log.tag("PREVIEW", f"{subdir.name}/{name}_*.{{png,npy}}")
