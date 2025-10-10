# borunte/cap_session.py
"""CaptureSession: filesystem layout and saving helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs

from utils.logger import Logger
from .config import CAPTURE_ROOT_DIR

_log = Logger.get_logger()


class CaptureSession:
    """Create timestamped dir, manage index, save images and poses."""

    def __init__(self, root_dir: Optional[str] = None):
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = root_dir or CAPTURE_ROOT_DIR
        self.root = Path(base) / ts
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
        prof: rs.pipeline_profile,
        out_json: Path,
        applied_disparity: int,
        decimation_mag: int,
    ) -> None:
        d_prof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        c_prof = prof.get_stream(rs.stream.color).as_video_stream_profile()

        intr_d = d_prof.get_intrinsics()
        intr_c = c_prof.get_intrinsics()
        extr_d2c = d_prof.get_extrinsics_to(c_prof)
        extr_c2d = c_prof.get_extrinsics_to(d_prof)

        dev = prof.get_device()
        depth_scale = float(dev.first_depth_sensor().get_depth_scale())

        def intr_to_dict(intr: Any) -> Dict[str, Any]:
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

        def extr_to_dict(extr: Any) -> Dict[str, Any]:
            rot = [float(x) for x in extr.rotation]
            t = [float(x) for x in extr.translation]
            Rm = [rot[0:3], rot[3:6], rot[6:9]]
            return {"rotation": Rm, "translation": t}

        data = {
            "depth_scale": depth_scale,
            "intrinsics": {
                "depth": intr_to_dict(intr_d),
                "color": intr_to_dict(intr_c),
            },
            "extrinsics": {
                "depth_to_color": extr_to_dict(extr_d2c),
                "color_to_depth": extr_to_dict(extr_c2d),
            },
            "processing": {
                "applied_disparity_shift": int(applied_disparity),
                "decimation_magnitude": int(decimation_mag),
            },
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _log.tag("RS2", f"params -> {out_json}")

    def save_rgb_depth(
        self, subdir: Path, name: str, rgb: np.ndarray, depth_m: np.ndarray
    ) -> None:
        subdir.mkdir(parents=True, exist_ok=True)
        rgb_path = subdir / f"{name}_rgb.png"
        dpt_path = subdir / f"{name}_depth.npy"
        cv2.imwrite(str(rgb_path), rgb)
        np.save(str(dpt_path), depth_m.astype(np.float32))
        _log.tag("SAVE", f"{subdir.name}/{rgb_path.name}, {dpt_path.name}")

    def update_pose(self, name: str, pose6: Dict[str, float]) -> None:
        self.poses[name] = pose6
        # атомарная запись poses.json
        tmp = self.poses_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.poses, f, indent=2, ensure_ascii=False)
        tmp.replace(self.poses_path)
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
