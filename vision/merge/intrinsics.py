# merge/intrinsics.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import open3d as o3d

from utils.logger import Logger

LOG = Logger.get_logger("intrinsics")


# ---------- file pairing ----------


def _rx_pair() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """
    Filename patterns:
      000_rgb.(png|jpg|jpeg)
      000_depth.(npy|png)
    """
    rx = re.compile(r"(\d+)_rgb\.(?:png|jpg|jpeg)$", re.IGNORECASE)
    dx = re.compile(r"(\d+)_depth\.(?:npy|png)$", re.IGNORECASE)
    return rx, dx


def first_pair_paths(img_dir: Path) -> tuple[Path, Path]:
    """Find the first rgb/depth pair by frame index."""
    rx, dx = _rx_pair()
    rgbs = {
        m.group(1): p
        for p in img_dir.iterdir()
        if p.is_file() and (m := rx.search(p.name))
    }
    deps = {
        m.group(1): p
        for p in img_dir.iterdir()
        if p.is_file() and (m := dx.search(p.name))
    }
    common = sorted(set(rgbs) & set(deps), key=lambda s: int(s))
    if not common:
        LOG.error(f"No pairs in {img_dir}; rgb={len(rgbs)} depth={len(deps)}")
        raise FileNotFoundError("No *_rgb.* / *_depth.* pairs found")
    k0 = common[0]
    return rgbs[k0], deps[k0]


def iter_pairs(img_dir: Path) -> Iterator[tuple[int, Path, Path]]:
    """Yield (index, rgb_path, depth_path) in numeric order."""
    rx, dx = _rx_pair()
    rgbs = {
        m.group(1): p
        for p in img_dir.iterdir()
        if p.is_file() and (m := rx.search(p.name))
    }
    deps = {
        m.group(1): p
        for p in img_dir.iterdir()
        if p.is_file() and (m := dx.search(p.name))
    }
    ids = sorted(set(rgbs) & set(deps), key=lambda s: int(s))
    for k in ids:
        yield int(k), rgbs[k], deps[k]


# ---------- rs2 intrinsics fix/IO ----------


def _scale_intr_block(block: dict, sx: float, sy: float) -> bool:
    need = {"width", "height", "ppx", "ppy", "fx", "fy"}
    if not need.issubset(block):
        return False
    block["fx"] = float(block["fx"]) * sx
    block["fy"] = float(block["fy"]) * sy
    block["ppx"] = float(block["ppx"]) * sx
    block["ppy"] = float(block["ppy"]) * sy
    block["width"] = int(round(float(block["width"]) * sx))
    block["height"] = int(round(float(block["height"]) * sy))
    return True


def _scale_stream(block: dict, sx: float, sy: float) -> bool:
    if "width" in block and "height" in block:
        block["width"] = int(round(float(block["width"]) * sx))
        block["height"] = int(round(float(block["height"]) * sy))
        return True
    return False


def fix_rs2_intrinsics(root: Path, img_dir_name: str) -> None:
    """
    Rescale RS2 intrinsics/streams to match actual depth frame size.

    Reads {root}/{img_dir_name}/rs2_params.json and updates:
      intrinsics.depth/color and streams.depth/color if sizes differ.
    """
    img_dir = Path(root) / img_dir_name
    rs2_path = img_dir / "rs2_params.json"
    if not rs2_path.exists():
        LOG.warning(f"rs2_params.json not found: {rs2_path}")
        return

    _, depth_p = first_pair_paths(img_dir)
    depth = np.load(depth_p)
    Hn, Wn = depth.shape

    data = json.loads(rs2_path.read_text(encoding="utf-8"))
    intr = data.get("intrinsics", {})
    idepth = intr.get("depth", {})
    icolor = intr.get("color", {})

    W0 = int(idepth.get("width", Wn)) or Wn
    H0 = int(idepth.get("height", Hn)) or Hn
    sx, sy = Wn / float(W0), Hn / float(H0)

    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        LOG.info(f"Intrinsics match frames: ({W0}x{H0}) == ({Wn}x{Hn})")
        return

    LOG.info(f"[RS2] scale ({W0}x{H0}) -> ({Wn}x{Hn}); " f"sx={sx:.6f} sy={sy:.6f}")
    bak = rs2_path.with_suffix(".json.bak")
    if not bak.exists():
        bak.write_text(json.dumps(data, indent=2), encoding="utf-8")

    updates = 0
    if _scale_intr_block(idepth, sx, sy):
        data.setdefault("intrinsics", {})["depth"] = idepth
        updates += 1
    if _scale_intr_block(icolor, sx, sy):
        data.setdefault("intrinsics", {})["color"] = icolor
        updates += 1

    streams = data.get("streams", {})
    sdepth = streams.get("depth", {})
    scolor = streams.get("color", {})
    if _scale_stream(sdepth, sx, sy):
        data.setdefault("streams", {})["depth"] = sdepth
        updates += 1
    if _scale_stream(scolor, sx, sy):
        data.setdefault("streams", {})["color"] = scolor
        updates += 1

    rs2_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOG.info(f"[RS2] updated {updates} block(s) in {rs2_path}")


def o3d_intrinsics_from_rs2(
    rs2_path: Path, depth_size: tuple[int, int]
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Build Open3D intrinsics from RS2 JSON (depth), rescaled to depth_size.

    If JSON is incomplete, fall back to unit focal with center principal
    point for the provided depth_size.
    """
    Wd, Hd = int(depth_size[0]), int(depth_size[1])
    try:
        data = json.loads(Path(rs2_path).read_text(encoding="utf-8"))
        idepth = data.get("intrinsics", {}).get("depth", {})
        W0 = int(idepth.get("width", Wd)) or Wd
        H0 = int(idepth.get("height", Hd)) or Hd
        sx, sy = Wd / float(W0), Hd / float(H0)
        fx = float(idepth.get("fx", max(Wd, Hd))) * sx
        fy = float(idepth.get("fy", max(Wd, Hd))) * sy
        cx = float(idepth.get("ppx", Wd * 0.5)) * sx
        cy = float(idepth.get("ppy", Hd * 0.5)) * sy
        return o3d.camera.PinholeCameraIntrinsic(Wd, Hd, fx, fy, cx, cy)
    except Exception as e:
        LOG.warning(f"[INTR] fallback due to JSON issue: {e}")
        fx = fy = float(max(Wd, Hd))
        cx, cy = float(Wd) * 0.5, float(Hd) * 0.5
        return o3d.camera.PinholeCameraIntrinsic(Wd, Hd, fx, fy, cx, cy)


def build_intrinsics_for_capture(
    root: Path, img_dir_name: str
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Convenience: read rs2_params.json inside capture folder and
    build rescaled intrinsics matching the first depth frame.
    """
    img_dir = Path(root) / img_dir_name
    _, depth_p = first_pair_paths(img_dir)
    depth = np.load(depth_p)
    H, W = depth.shape
    rs2_path = img_dir / "rs2_params.json"
    K = o3d_intrinsics_from_rs2(rs2_path, depth_size=(W, H))
    LOG.info(f"[INTR] built Open3D intrinsics for {W}x{H}")
    return K
