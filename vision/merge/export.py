# merge/export.py
from __future__ import annotations

from pathlib import Path
import json
import open3d as o3d

from utils.logger import Logger
from .config import DEBUG_DIR_NAME

LOG = Logger.get_logger("merge.export")


def _debug_dir(root: Path) -> Path:
    d = Path(root) / DEBUG_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_cloud(
    cloud: o3d.geometry.PointCloud, root: Path, name: str = "final_merged.ply"
) -> Path:
    """Save PLY to <root>/<debug>/name."""
    path = _debug_dir(root) / name
    ok = o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to save cloud: {path}")
    LOG.info(f"[SAVE] {path} (points={len(cloud.points)})")
    return path


def interactive_picks_and_save(
    cloud: o3d.geometry.PointCloud, root: Path, prefix: str = "final"
) -> Path:
    """
    Open viewer with editing; after close, try to read default pick file
    and save it into <root>/<debug>/<prefix>_picks.json.
    """
    LOG.info("[PICKS] open viewer; press 'F12' to dump pick file in O3D")
    o3d.visualization.draw_geometries_with_editing([cloud])

    # Open3D обычно кладёт 'PickedPoints.json' в CWD; соберём, если есть.
    src = Path("PickedPoints.json")
    out = _debug_dir(root) / f"{prefix}_picks.json"
    if src.exists():
        try:
            out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            LOG.info(f"[PICKS] saved to {out}")
        except Exception as e:
            LOG.warning(f"[PICKS] failed to move picks file: {e}")
    else:
        # не критично, просто информируем
        out.write_text(json.dumps({"info": "no picks exported"}), "utf-8")
        LOG.warning("[PICKS] no PickedPoints.json found")
    return out
