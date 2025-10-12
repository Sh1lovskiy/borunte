from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v2 as iio
import open3d as o3d
from utils.logger import Logger
from .intrinsics import iter_pairs, o3d_intrinsics_from_rs2
from .poses import load_poses_generic
from .viz import show_depth_cv

LOG = Logger.get_logger("tsdf")


def _mk_volume(preset: str) -> o3d.pipelines.integration.ScalableTSDFVolume:
    voxel = 0.004 if preset == "fast" else 0.0025
    trunc = voxel * (4.0 if preset == "fast" else 5.0)
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _camera_extrinsic(
    k: int,
    poses: list[np.ndarray],
    pose_kind: str,
    R_cam2gripper: np.ndarray,
    t_cam2gripper_m: np.ndarray,
) -> np.ndarray:
    T_cam2gr = _T_from_Rt(R_cam2gripper, t_cam2gripper_m)
    if pose_kind == "cam":
        return poses[k] if k < len(poses) else np.eye(4)
    T_base_gr = poses[k] if k < len(poses) else np.eye(4)
    return T_base_gr @ T_cam2gr


def _integrate(vol, rgb, depth_m, intr, extr, depth_trunc: float) -> None:
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    vol.integrate(rgbd, intr, extr)


def fallback_merge_tsdf(
    root: Path,
    img_dir_name: str,
    poses_json: str,
    preset: str,
    depth_trunc: float,
    bbox_base,
    coord_frame_size: float,
    viz_every_n: int,
    window_name: str,
    *,
    pose_kind: str,
    R_cam2gripper: np.ndarray,
    t_cam2gripper_m: np.ndarray,
    preview_frames: bool,
) -> o3d.geometry.PointCloud:
    img_dir = root / img_dir_name
    rs2_path = img_dir / "rs2_params.json"
    pairs = list(iter_pairs(img_dir))
    if not pairs:
        raise RuntimeError("No rgb/depth pairs for TSDF fallback")

    _, _, dp = pairs[0]
    d0 = np.load(dp)
    intr = o3d_intrinsics_from_rs2(rs2_path, depth_size=(d0.shape[1], d0.shape[0]))
    poses = load_poses_generic(root / poses_json)
    vol = _mk_volume(preset)

    it = Logger.progress(pairs, desc="TSDF")
    for idx, (k, rp, dp) in enumerate(it):
        rgb = iio.imread(rp)
        depth_m = np.load(dp).astype(np.float32)

        if preview_frames and (idx % max(1, viz_every_n)) == 0:
            quit_req = show_depth_cv(
                depth_m,
                title=f"[FB] frame {k:03d}",
                depth_trunc=depth_trunc,
                window_name=window_name,
                wait_ms=1,
            )
            if quit_req:
                LOG.info("TSDF aborted by user")
                break

        extr = _camera_extrinsic(k, poses, pose_kind, R_cam2gripper, t_cam2gripper_m)
        _integrate(vol, rgb, depth_m, intr, extr, depth_trunc)

        if (idx % 5) == 0:
            LOG.info(f"[FB] integrated k={k:03d} / total={len(pairs):03d}")

    pcd = vol.extract_point_cloud()
    if len(pcd.points) == 0:
        mesh = vol.extract_triangle_mesh()
        if len(mesh.vertices) > 0:
            pcd = mesh.sample_points_poisson_disk(number_of_points=200_000)
    return pcd
