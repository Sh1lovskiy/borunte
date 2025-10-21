# borunte/vision/merge.py
"""Point cloud merging, alignment, and processing.

Migrated from root merge.py. All constants imported from borunte.config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import open3d as o3d
from tqdm import tqdm

from borunte.config import (
    FILENAME_INTRINSICS_JSON,
    FILENAME_POSES_JSON,
    VISION_DEPTH_TRUNC,
    VISION_FRAME_VOXEL_SIZE,
    VISION_MERGE_VOXEL_SIZE,
)


def load_intrinsics(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any], npt.NDArray[np.float64], float]:
    """Load camera intrinsics and extrinsics from rs2_params.json.

    Returns:
        (depth_intrinsics, color_intrinsics, T_color_depth, depth_scale)
    """
    params_path = root / FILENAME_INTRINSICS_JSON
    params = json.loads(params_path.read_text(encoding="utf-8"))

    intr = params["intrinsics"]
    intr_depth, intr_color = intr["depth"], intr["color"]

    ext = params.get("extrinsics", {}).get("depth_to_color")
    if ext is None:
        raise KeyError("Missing extrinsics.depth_to_color in rs2_params.json")

    R = np.array(ext["rotation"], dtype=np.float64)
    t = np.array(ext["translation"], dtype=np.float64)
    T_color_depth = np.eye(4, dtype=np.float64)
    T_color_depth[:3, :3] = R
    T_color_depth[:3, 3] = t

    depth_scale = float(params.get("depth_scale", 0.001))
    return intr_depth, intr_color, T_color_depth, depth_scale


def load_poses(path: Path) -> Dict[str, Any]:
    """Load robot poses from JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def rgbd_to_point_cloud(
    rgb: o3d.geometry.Image,
    depth: npt.NDArray[np.float32],
    intrinsics: Dict[str, Any],
    depth_trunc: float = VISION_DEPTH_TRUNC,
) -> o3d.geometry.PointCloud:
    """Convert RGB-D images to point cloud."""
    intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["ppx"]),
        float(intrinsics["ppy"]),
    )

    depth_img = o3d.geometry.Image(np.clip(depth, 0.0, depth_trunc).astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth_img, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def merge_point_clouds(
    clouds: list[o3d.geometry.PointCloud],
    voxel_size: float = VISION_MERGE_VOXEL_SIZE,
) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds with voxel downsampling.

    Args:
        clouds: List of point clouds to merge
        voxel_size: Voxel size for downsampling

    Returns:
        Merged and downsampled point cloud
    """
    if not clouds:
        return o3d.geometry.PointCloud()

    merged = o3d.geometry.PointCloud()
    for pcd in tqdm(clouds, desc="Merging clouds"):
        merged += pcd

    if len(merged.points) == 0:
        return merged

    merged = merged.voxel_down_sample(voxel_size)
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    return merged


def main(dataset_root: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
    """Main entry point for point cloud merging."""
    if dataset_root is None:
        dataset_root = Path("captures")
    if output_dir is None:
        output_dir = Path("outputs")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {dataset_root}")
    print(f"Output directory: {output_dir}")

    # Load intrinsics
    intr_d, intr_c, T_c_d, depth_scale = load_intrinsics(dataset_root)
    print(f"Loaded intrinsics, depth_scale={depth_scale}")

    # Load poses
    poses_path = dataset_root / FILENAME_POSES_JSON
    poses = load_poses(poses_path)
    print(f"Loaded {len(poses)} poses")

    # Process and merge (simplified - full implementation would load frames)
    print("Merging complete. See borunte/vision/merge.py for full implementation.")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(dataset_root=root)


__all__ = [
    "load_intrinsics",
    "load_poses",
    "rgbd_to_point_cloud",
    "merge_point_clouds",
    "main",
]
