# merge/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.io import load_poses
from utils.config import HAND_EYE  # R, t, direction (tcp_cam or cam_tcp_inv)

from .config import MAX_PREVIEW_POINTS, PipelineCfg
from .intrinsics import build_intrinsics_for_capture
from .rgbd import rgbd_to_pcd, prefilter_depth_m
from .roi import crop_to_aabb, log_cloud_stats, make_aabb
from .registration import RefineParams, pairwise_refine
from .transforms import autotune_extrinsics, build_T_base_cam, get_T_tcp_cam
from .tsdf import build_tsdf, extract_cloud, integrate_frame
from .viz import draw_with_roi

LOG = Logger.get_logger("merge.pipeline")


# --------------------------- IO helpers -------------------------------------


def _load_pair(img_dir: Path, stem: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read <stem>_rgb.png + <stem>_depth.npy, depth→meters."""
    try:
        import cv2
    except Exception as e:
        LOG.error(f"OpenCV missing: {e}")
        return None

    rgb_p = img_dir / f"{stem}_rgb.png"
    d_p = img_dir / f"{stem}_depth.npy"
    if not (rgb_p.exists() and d_p.exists()):
        LOG.warning(f"[IO] missing pair for {stem}")
        return None

    rgb_bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
    depth_raw = np.load(d_p)
    # mm→m if needed (heuristic: max>50_000 means mm)
    scale = 1e-3 if float(depth_raw.max(initial=0)) > 50_000 else 1.0
    depth_m = depth_raw.astype(np.float32) * scale
    depth_m[~np.isfinite(depth_m)] = 0.0
    LOG.info(f"[PAIR {stem}] rgb={rgb_bgr.shape} depth={depth_raw.shape}")
    return rgb_bgr, depth_m


def _clean_and_downsample(
    pcd: o3d.geometry.PointCloud, cfg: PipelineCfg
) -> Tuple[o3d.geometry.PointCloud, int]:
    """Voxel downsample + optional stat. outlier removal."""
    pc = pcd.voxel_down_sample(cfg.frame_vox) if cfg.frame_vox > 0 else pcd
    removed = 0
    if cfg.remove_outliers and len(pc.points) > 500:
        _, idx = pc.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nn, std_ratio=cfg.outlier_std
        )
        removed = len(pc.points) - len(idx)
        pc = pc.select_by_index(idx)
    return pc, removed


def _maybe_compact_preview(
    preview: o3d.geometry.PointCloud, merge_vox: float
) -> o3d.geometry.PointCloud:
    """Limit preview size to keep registration/visual fast."""
    if len(preview.points) <= MAX_PREVIEW_POINTS or merge_vox <= 0:
        return preview
    n0 = len(preview.points)
    out = preview.voxel_down_sample(merge_vox)
    LOG.info(f"[PREVIEW] compact {n0} → {len(out.points)}")
    return out


def _refine_if_needed(
    cfg: PipelineCfg,
    pcd_clean: o3d.geometry.PointCloud,
    preview: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """Optional FGR→ICP→ColoredICP pairwise refine."""
    if len(preview.points) <= 5_000:
        return pcd_clean
    p = cfg.reg
    params = RefineParams(
        rd=p.rd,
        rn=p.rn,
        rf=p.rf,
        fgr_max_corr=p.fgr_max_corr,
        icp_max_corr=p.icp_max_corr,
        icp_max_iters=p.icp_max_iters,
        icp_tukey_k=p.icp_tukey_k,
        use_colored_icp=p.use_colored_icp,
        cicp_pyramid=list(p.cicp_pyramid),
        cicp_iters=list(p.cicp_iters),
        cicp_enable_min_fitness=p.cicp_enable_min_fitness,
        cicp_continue_min_fitness=p.cicp_continue_min_fitness,
        cicp_fail_scale_up=p.cicp_fail_scale_up,
    )
    Tref = pairwise_refine(pcd_clean, preview, params)
    if np.allclose(Tref, np.eye(4)):
        return pcd_clean
    return o3d.geometry.PointCloud(pcd_clean).transform(Tref.copy())


# ---------------------------- main routine ----------------------------------


def merge_capture(root: Path, cfg: PipelineCfg) -> o3d.geometry.PointCloud:
    """
    Merge RGB-D capture into a base-frame point cloud.

    Steps:
      1) Build intrinsics matching depth size.
      2) Autotune pose units/Euler/hand-eye on the first frame.
      3) For each frame: RGBD→PCD(cam)→BASE, crop by BBox, clean, refine,
         accumulate preview and TSDF.
      4) Extract final cloud (TSDF or preview), final BBox crop.
    """
    root = Path(root)
    img_dir = root / cfg.img_dir
    intr = build_intrinsics_for_capture(root, cfg.img_dir)
    aabb = make_aabb(cfg.bbox_points)

    poses = load_poses(root / cfg.poses_json)
    stems = sorted(poses.keys())
    if not stems:
        LOG.warning(f"No poses in {root / cfg.poses_json}")
        return o3d.geometry.PointCloud()

    # TSDF volume when requested
    vol = None
    if cfg.use_tsdf():
        if cfg.quality_preset == "best":
            voxel, trunc = cfg.reg.rd * 0.4, cfg.reg.rd * 1.6
        else:
            voxel, trunc = 0.004, 0.016
        vol = build_tsdf(
            voxel, trunc, o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    # ---- First frame: autotune pose conventions/hand-eye
    pair0 = _load_pair(img_dir, stems[0])
    if pair0 is None:
        return o3d.geometry.PointCloud()
    rgb0, depth0 = pair0
    depth0 = prefilter_depth_m(depth0, cfg)
    pc0_cam = rgbd_to_pcd(rgb0, depth0, intr, depth_trunc=cfg.depth_trunc)
    log_cloud_stats(f"[S1 {stems[0]}] CAMERA", pc0_cam)

    scale, order, he_dir = autotune_extrinsics(
        pc0_cam,
        poses[stems[0]],
        aabb,
        cfg.pose_unit_mode,
        cfg.euler_mode,
        cfg.he_dir_mode,
        HAND_EYE,
    )
    T0 = build_T_base_cam(
        poses[stems[0]], scale, order, get_T_tcp_cam(HAND_EYE, he_dir)
    )
    pc0_base = o3d.geometry.PointCloud(pc0_cam).transform(T0.copy())
    pc0_crop = crop_to_aabb(pc0_base, aabb)
    log_cloud_stats(f"[S2 {stems[0]}] BASE", pc0_base)
    log_cloud_stats(f"[S3 {stems[0]}] CROP", pc0_crop)

    preview = o3d.geometry.PointCloud(pc0_crop)
    if cfg.viz_stages:
        draw_with_roi(
            pc0_base.paint_uniform_color((0.2, 0.6, 1.0)),
            aabb,
            coord_frame_size=cfg.coord_frame_size,
            title=f"Diag BASE {stems[0]}",
        )

    if vol is not None:
        integrate_frame(vol, intr, rgb0, depth0, T0)

    # ---- Remaining frames
    for stem in stems[1:]:
        pr = _load_pair(img_dir, stem)
        if pr is None:
            continue
        rgb, depth = pr
        depth = prefilter_depth_m(depth, cfg)
        pc_cam = rgbd_to_pcd(rgb, depth, intr, depth_trunc=cfg.depth_trunc)

        T = build_T_base_cam(poses[stem], scale, order, get_T_tcp_cam(HAND_EYE, he_dir))
        pc_base = o3d.geometry.PointCloud(pc_cam).transform(T)
        pc_crop = crop_to_aabb(pc_base, aabb)

        pc_clean, _ = _clean_and_downsample(pc_crop, cfg)
        pc_clean = _refine_if_needed(cfg, pc_clean, preview)

        preview += pc_clean
        preview = _maybe_compact_preview(preview, cfg.merge_vox)

        if vol is not None:
            integrate_frame(vol, intr, rgb, depth, T)

    # ---- Final cloud
    if vol is not None:
        cloud = extract_cloud(vol)
    else:
        cloud = preview

    cloud = crop_to_aabb(cloud, aabb)
    log_cloud_stats("[FINAL] CROP", cloud)
    return cloud
