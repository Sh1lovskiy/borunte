"""Filesystem helpers for capture sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# безопасная запись PNG без GUI-опencv
try:
    import imageio.v3 as iio
except Exception:  # fallback на старый imageio
    import imageio as iio  # type: ignore

try:
    import cv2  # опционально, если есть
except Exception:  # cv2 может отсутствовать в headless окружении
    cv2 = None  # type: ignore

from utils.logger import get_logger

from .config import BORUNTE_CONFIG, BorunteConfig

_log = get_logger()


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
        # базой считаем config.paths.captures_root, если не передан root_dir
        base = (
            Path(root_dir) if root_dir is not None else Path(config.paths.captures_root)
        )
        self.root = base / ts  # строго captures/<timestamp>
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
        profile,
        out_json: Path,
        applied_disparity: int,
        decimation_mag: int,
    ) -> None:
        # Импорты pyrealsense2 могут отсутствовать в рантайме — пишем через getattr
        rs = profile.__class__.__module__.startswith("pyrealsense2")

        def intr_to_dict(intr) -> Dict[str, float]:
            return {
                "width": int(getattr(intr, "width", 0)),
                "height": int(getattr(intr, "height", 0)),
                "ppx": float(getattr(intr, "ppx", 0.0)),
                "ppy": float(getattr(intr, "ppy", 0.0)),
                "fx": float(getattr(intr, "fx", 0.0)),
                "fy": float(getattr(intr, "fy", 0.0)),
                "model": str(getattr(intr, "model", "")),
                "coeffs": [
                    float(c) for c in list(getattr(intr, "coeffs", [0, 0, 0, 0, 0]))[:5]
                ],
            }

        def extr_to_dict(extr) -> Dict[str, list[float]]:
            rot = [
                float(x) for x in getattr(extr, "rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
            ]
            trans = [float(x) for x in getattr(extr, "translation", [0, 0, 0])]
            return {
                "rotation": [rot[0:3], rot[3:6], rot[6:9]],
                "translation": trans,
            }

        try:
            depth_profile = profile.get_stream(
                2
            ).as_video_stream_profile()  # rs.stream.depth
            color_profile = profile.get_stream(
                1
            ).as_video_stream_profile()  # rs.stream.color
            device = profile.get_device()
            depth_scale = float(device.first_depth_sensor().get_depth_scale())
            data = {
                "depth_scale": depth_scale,
                "intrinsics": {
                    "depth": intr_to_dict(depth_profile.get_intrinsics()),
                    "color": intr_to_dict(color_profile.get_intrinsics()),
                },
                "extrinsics": {
                    "depth_to_color": extr_to_dict(
                        depth_profile.get_extrinsics_to(color_profile)
                    ),
                    "color_to_depth": extr_to_dict(
                        color_profile.get_extrinsics_to(depth_profile)
                    ),
                },
                "processing": {
                    "applied_disparity_shift": int(applied_disparity),
                    "decimation_magnitude": int(decimation_mag),
                },
            }
        except Exception:
            # если что-то пошло не так — записываем только минимум
            data = {
                "processing": {
                    "applied_disparity_shift": int(applied_disparity),
                    "decimation_magnitude": int(decimation_mag),
                }
            }

        out_json.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(out_json, data)
        _log.tag("RS2", f"params -> {out_json}")

    def _save_png(self, path: Path, img: np.ndarray) -> None:
        # imageio работает и в headless
        try:
            iio.imwrite(str(path), img)
        except Exception:
            # запасной вариант — если есть cv2
            if cv2 is not None and hasattr(cv2, "imwrite"):
                cv2.imwrite(
                    str(path), img[:, :, ::-1] if img.ndim == 3 else img
                )  # BGR/RGB
            else:
                raise

    def save_rgb_depth(
        self, subdir: Path, name: str, rgb: np.ndarray, depth_m: np.ndarray
    ) -> None:
        subdir.mkdir(parents=True, exist_ok=True)
        rgb_path = subdir / f"{name}_rgb.png"
        depth_path = subdir / f"{name}_depth.npy"
        self._save_png(rgb_path, rgb)
        # .npy — всегда float32 метры
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
            self._save_png(subdir / f"{name}_rgb.png", rgb)
        if depth_m is not None:
            np.save(str(subdir / f"{name}_depth.npy"), depth_m.astype(np.float32))
        _log.tag("PREVIEW", f"{subdir.name}/{name}_*.{{png,npy}}")
