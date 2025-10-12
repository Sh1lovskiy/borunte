from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class HandEye:
    """Hand-eye transform and pose type."""

    R_cam2gripper: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    t_cam2gripper_m: Tuple[float, float, float]
    pose_kind: str = "tcp"  # 'tcp' (base->gripper) or 'cam' (base->cam)


@dataclass(frozen=True)
class VizCfg:
    """Viewer/preview toggles."""

    enabled: bool = True
    preview_frames: bool = False  # per-frame depth preview
    preview_cloud: bool = True  # show merged cloud
    preview_sequence: bool = True  # frames→merge→cloud→truss
    every_n: int = 1
    coord_frame_size: float = 0.12
    preview_before_merge: bool = True
    cv_window: str = "Depth (m)"


@dataclass(frozen=True)
class MergeQuality:
    depth_trunc: float = 3.5
    frame_vox: float = 0.002
    remove_outliers: bool = True
    outlier_nn: int = 18
    outlier_std: float = 1.5
    merge_vox: float = 0.0025
    quality_preset: str = "best"  # "fast" | "best"


@dataclass(frozen=True)
class RegCfg:
    rd: float = 0.003
    rn: int = 20
    rf: int = 15
    fgr_max_corr: float = 0.02
    icp_max_corr: float = 0.02
    icp_max_iters: int = 50
    icp_tukey_k: float = 4.685
    use_colored_icp: bool = True
    cicp_pyramid: tuple[int, ...] = (10, 5, 3)
    cicp_iters: tuple[int, ...] = (20, 10, 5)
    cicp_enable_min_fitness: float = 0.2
    cicp_continue_min_fitness: float = 0.4
    cicp_fail_scale_up: float = 1.5


@dataclass(frozen=True)
class TrussCfg:
    merge_node_radius: float = 0.004
    raster_res_px: int = 1024
    save_tag: str = "first"
    normal_bin_edges_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    orient_sectors: int = 4
    cache_normals: bool = True


@dataclass(frozen=True)
class AppCfg:
    capture_root: Path
    img_dir_name: str
    bbox_points: tuple[tuple[float, float, float], tuple[float, float, float]]
    debug_dir: Path
    merged_ply: Path
    viz: VizCfg = field(default_factory=VizCfg)
    merge: MergeQuality = field(default_factory=MergeQuality)
    reg: RegCfg = field(default_factory=RegCfg)
    truss: TrussCfg = field(default_factory=TrussCfg)
    handeye: HandEye = field(
        default_factory=lambda: HandEye(
            R_cam2gripper=(
                (0.999493516, -0.012213734, -0.029385980),
                (-0.011574748, -0.999694969, 0.021817300),
                (0.029643487, 0.021466114, 0.999330010),
            ),
            t_cam2gripper_m=(0.028112, -0.086201, 0.037936),
            pose_kind="tcp",
        )
    )


def build_defaults() -> AppCfg:
    root = Path(
        "/home/sha/Documents/aitech-robotics/borunte/captures/20251009_153839/ds_0"
    )
    img_dir = "d1280x720_dec1_c1280x720"
    debug = root / "debug"
    merged = debug / "final_merged.ply"
    # axis-aligned bbox in BASE coords (min, max)
    bbox = ((-0.5, -0.3, 0.8), (0.3, 0.3, 0.3))
    return AppCfg(
        capture_root=root,
        img_dir_name=img_dir,
        bbox_points=bbox,
        debug_dir=debug,
        merged_ply=merged,
    )
