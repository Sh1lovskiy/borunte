# calib/offline.py
"""Offline ChArUco detection pipeline with optional hand-eye summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    cv2 = None  # type: ignore

from utils.error_tracker import error_scope
from utils.logger import Logger

from .charuco import detect_pose, make_board
from .config import CALIB_CONFIG, CalibConfig

_log = Logger.get_logger()


def _atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_indices(root: Path) -> List[int]:
    idxs: List[int] = []
    for file in sorted(root.glob("*_rgb.png")):
        try:
            idxs.append(int(file.stem.split("_")[0]))
        except Exception:
            continue
    return sorted(idxs)


def _load_intrinsics(root: Path, stream: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    params = root / "rs2_params.json"
    if not params.exists():
        raise FileNotFoundError(f"rs2_params.json not found in {root}")
    data = _load_json(params)
    key = "color" if stream.lower() == "color" else "depth"
    intr = data["intrinsics"][key]
    K = np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(intr.get("coeffs", [0, 0, 0, 0, 0])[:5], float).ravel()
    width = int(intr.get("width", 0))
    height = int(intr.get("height", 0))
    return K, dist, (width, height)


@dataclass
class Detection:
    idx: int
    rmse_px: float
    n_corners: int
    used_ids: List[int]
    shaky: bool
    coverage_w_frac: float
    R_tc: List[List[float]]
    t_tc: List[float]


def _detect_dataset(dataset: Path, config: CalibConfig) -> List[Detection]:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV (cv2) is required for calibration")
    board = make_board()
    K, dist, _ = _load_intrinsics(dataset, config.use_stream)
    detections: List[Detection] = []
    for idx in tqdm(_find_indices(dataset), desc="Charuco", leave=False):
        rgb_path = dataset / f"{idx:03d}_rgb.png"
        if not rgb_path.exists():
            continue
        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            _log.tag("CALIB", f"skip #{idx}: image load failed", level="warning")
            continue
        pose = detect_pose(img, K, dist, board)
        if pose is None:
            continue
        R_tc, t_tc, ids, rmse, count = pose
        shaky = rmse > config.detection.reproj_rmse_max_px
        coverage = 0.0
        detections.append(
            Detection(
                idx=idx,
                rmse_px=float(rmse),
                n_corners=int(count),
                used_ids=[int(v) for v in ids],
                shaky=bool(shaky),
                coverage_w_frac=float(coverage),
                R_tc=R_tc.tolist(),
                t_tc=t_tc.reshape(-1).tolist(),
            )
        )
    return detections


def _write_outputs(dataset: Path, detections: List[Detection], config: CalibConfig) -> Path:
    out_root = config.output.root / dataset.name
    out_root.mkdir(parents=True, exist_ok=True)
    det_path = out_root / config.output.detections_file
    _atomic_write(det_path, [d.__dict__ for d in detections])
    summary_path = out_root / config.output.report_raw
    summary = {
        "total": len(detections),
        "mean_rmse_px": float(np.mean([d.rmse_px for d in detections])) if detections else 0.0,
        "median_rmse_px": float(np.median([d.rmse_px for d in detections])) if detections else 0.0,
    }
    _atomic_write(summary_path, summary)
    return out_root


def run_offline(dataset: Path | str, config: CalibConfig = CALIB_CONFIG) -> Path:
    dataset_path = Path(dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    with error_scope():
        detections = _detect_dataset(dataset_path, config)
        if not detections:
            raise RuntimeError("no ChArUco detections found")
        out_root = _write_outputs(dataset_path, detections, config)
        _log.tag("CALIB", f"detections saved -> {out_root}")
        return out_root


__all__ = ["run_offline"]
