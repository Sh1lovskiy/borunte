from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v2 as iio
import open3d as o3d
from utils.logger import Logger
from .intrinsics import iter_pairs, o3d_intrinsics_from_rs2
from .poses import load_poses_generic
from .viz import viz_open3d, show_depth_cv

LOG = Logger.get_logger("preview")


def _create_pcd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: o3d.camera.PinholeCameraIntrinsic,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def preview_each_frame(
    root: Path,
    img_dir_name: str,
    poses_json: str,
    depth_trunc: float,
    bbox_base,
    coord_frame_size: float,
    every_n: int,
    window_name: str,
) -> None:
    img_dir = root / img_dir_name
    rs2_path = img_dir / "rs2_params.json"

    pairs = list(iter_pairs(img_dir))
    if not pairs:
        LOG.warning("Preview: no rgb/depth pairs found")
        return

    _, _, dp = pairs[0]
    d0 = np.load(dp)
    intr = o3d_intrinsics_from_rs2(rs2_path, depth_size=(d0.shape[1], d0.shape[0]))
    poses = load_poses_generic(root / poses_json)

    it = Logger.progress(pairs, desc="Preview")
    for idx, (k, rp, dp) in enumerate(it):
        if (idx % max(1, every_n)) != 0:
            continue
        rgb = iio.imread(rp)
        depth_m = np.load(dp).astype(np.float32)

        quit_req = show_depth_cv(
            depth_m,
            title=f"frame {k:03d}",
            depth_trunc=depth_trunc,
            window_name=window_name,
            wait_ms=1,
        )
        if quit_req:
            LOG.info("Preview aborted by user")
            break

        pcd_cam = _create_pcd(rgb, depth_m, intr, depth_trunc)
        T = poses[k] if k < len(poses) else np.eye(4, dtype=float)
        pcd_base = o3d.geometry.PointCloud(pcd_cam).voxel_down_sample(0.004)
        pcd_base.transform(T)
        viz_open3d(
            title=f"BASE frame {k:03d}",
            geoms=[pcd_base],
            T_cam_to_base=T,
            bbox_base=bbox_base,
            base_axes_size=coord_frame_size,
        )
