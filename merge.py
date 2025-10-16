"""
RGB-D -> COLOR cam (align depth via depth_to_color) -> TCP (hand-eye, meters)
-> BASE (poses in mm -> meters) -> shared AABB crop (meters)
-> FPFH (FGR/RANSAC) + ICP refine/accumulate.
Two tqdm bars (Preprocess, Merge). Final Open3D visualization.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import open3d as o3d
from loguru import logger
from tqdm import tqdm

# Optional SciPy for Euler; else fallback to Open3D helpers
try:
    from scipy.spatial.transform import Rotation as _SciRot  # type: ignore

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Optional SIFT (not required; main path uses FPFH+ICP)
try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ==============================
#         COEFFICIENTS
# ==============================

# --- I/O & UI ---
DATASET_ROOT: Final[Path] = Path(
    "captures/20251009_153839/ds_0/d1280x720_dec1_c1280x720"
)
POSES_JSON_NAME: Final[str] = "poses.json"
RS2_PARAMS_NAME: Final[str] = "rs2_params.json"
IMG_DIR_NAME: Final[str] = "."
OUTPUT_DIR: Final[Path] = Path("outputs")

OPEN_VIEWER: Final[bool] = True
SAVE_SCREENSHOT: Final[bool] = True
VIEW_W: Final[int] = 1280
VIEW_H: Final[int] = 720

# --- Units & truncation ---
POSES_UNIT_MODE: Final[str] = "mm"  # Robot logs are in millimeters; convert to meters
DEPTH_MODE: Final[str] = "auto"  # Depth arrays can be meters (float) or mm (uint16)
DEPTH_TRUNC: Final[float] = 1

# --- ROI (AABB in BASE frame) ---
# Purpose: crop to a consistent workspace region (meters) for speed and robustness.
BBOX_POINTS: Final[list[list[float]]] = [
    [0.021, 0.841, 0.020],
    [0.621, 0.101, 0.094],
    [0.621, 1.041, 0.040],
    [0.621, 1.041, 0.040],
]
ROI_MIN_THICKNESS: Final[float] = 1e-5  # Ensure non-zero thickness for Open3D crop()


# --- Hand–Eye (COLOR camera) ---
# Purpose: map from COLOR camera to TCP. These params are in meters.
# direction="tcp_cam" means the matrix maps CAM -> TCP directly.
@dataclass(frozen=True)
class HandEye:
    R: np.ndarray
    t: np.ndarray
    direction: str  # "tcp_cam" or "cam_tcp"


HAND_EYE_R: Final[list[list[float]]] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
HAND_EYE_R: Final[list[list[float]]] = [
    [-0.012213734, 0.999493516, -0.029385980],
    [-0.999694969, -0.011574748, 0.021817300],
    [0.021466114, 0.029643487, 0.999330010],
]

HAND_EYE_T: Final[list[float]] = [-0.086201, 0.028112, 0.037936]
# HAND_EYE_T: Final[list[float]] = [-0.036, -0.078, 0.029]
HAND_EYE: Final[HandEye] = HandEye(
    R=np.array(HAND_EYE_R, dtype=np.float64),
    t=np.array(HAND_EYE_T, dtype=np.float64),
    direction="tcp_cam",
)

# --- Registration (FPFH + ICP) ---
# Purpose: global init (FGR -> RANSAC fallback) + local refinement (ICP point-to-plane).
# Voxel sizes derived adaptively; these are minimum caps for stability.
VOX_FPFH_MIN: Final[float] = 0.003
VOX_ICP_MIN: Final[float] = 0.002
ICP_ITERS: Final[int] = 30

# --- Cleaning / Preview ---
# Purpose: keep cloud compact to reduce runtime and avoid ICP degeneracy.
MIN_CROP_PTS: Final[int] = 200
PREVIEW_VOX_TARGET: Final[float] = 0.0001
OUTLIER_NN: Final[int] = 30
OUTLIER_STD: Final[float] = 2


# ==============================
#           LOGGING
# ==============================

logger.remove()
logger.add(sys.stderr, format="{message}")


def _log(tag: str, msg: str, level: str = "info") -> None:
    getattr(logger, level.lower(), logger.info)(f"[{tag}] {msg}")


# ==============================
#          SMALL UTILS
# ==============================


def _clone_pcd(p: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if len(p.points) == 0:
        return o3d.geometry.PointCloud()
    idx = np.arange(len(p.points))
    return p.select_by_index(idx.tolist(), invert=False)


def _log_cloud_stats(tag: str, pcd: o3d.geometry.PointCloud) -> None:
    n = len(pcd.points)
    if n == 0:
        _log(tag, "0 pts", level="warning")
        return
    P = np.asarray(pcd.points)
    mn, mx = P.min(0), P.max(0)
    _log(tag, f"{n} pts | min={mn} | max={mx} | extent={mx - mn}")


def _pose_scale(mode: str) -> float:
    if mode == "mm":
        return 0.001
    if mode == "m":
        return 1.0
    raise ValueError("POSES_UNIT_MODE must be 'mm' or 'm'")


def _handeye_matrix(he: HandEye) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = he.R
    T[:3, 3] = he.t
    return T if he.direction == "tcp_cam" else np.linalg.inv(T)


def _o3d_intr_from_dict(d: dict) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        int(d["width"]),
        int(d["height"]),
        float(d["fx"]),
        float(d["fy"]),
        float(d["ppx"]),
        float(d["ppy"]),
    )


def _make_aabb(points: list[list[float]]) -> o3d.geometry.AxisAlignedBoundingBox:
    pts = np.asarray(points, dtype=np.float64)
    mn = pts.min(0)
    mx = pts.max(0)
    ext = mx - mn
    for i in range(3):
        if ext[i] <= 0.0:
            mn[i] -= ROI_MIN_THICKNESS * 0.5
            mx[i] += ROI_MIN_THICKNESS * 0.5
    aabb = o3d.geometry.AxisAlignedBoundingBox(mn, mx)
    _log(
        "CROP",
        f"Using AABB min={aabb.get_min_bound()} max={aabb.get_max_bound()} extents={aabb.get_extent()}",
    )
    return aabb


# ==============================
#       INTRINSICS / IO
# ==============================


def build_intrinsics(root: Path) -> tuple[dict, dict, np.ndarray, float]:
    """Return depth_intr, color_intr, T_color_depth, rs_depth_scale."""
    params = json.loads((root / RS2_PARAMS_NAME).read_text(encoding="utf-8"))
    intr = params["intrinsics"]
    intr_d, intr_c = intr["depth"], intr["color"]
    ext = params.get("extrinsics", {}).get("depth_to_color", None)
    if ext is None:
        raise KeyError("extrinsics.depth_to_color missing in rs2_params.json")
    R = np.array(ext["rotation"], dtype=np.float64)
    t = np.array(ext["translation"], dtype=np.float64)
    T_color_depth = np.eye(4, dtype=np.float64)
    T_color_depth[:3, :3] = R
    T_color_depth[:3, 3] = t
    rs_scale = float(params.get("depth_scale", 0.001))
    _log("MERGE", f"Loaded intrinsics (1280x720), rs_depth_scale={rs_scale}")
    return intr_d, intr_c, T_color_depth, rs_scale


def load_poses(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _guess_depth_scale(
    arr: np.ndarray, mode: str, rs_scale: float
) -> tuple[float, str]:
    if mode == "mm":
        return rs_scale, "mm->m"
    if mode == "m":
        return 1.0, "m"
    med = float(np.nanmedian(arr))
    if np.issubdtype(arr.dtype, np.floating) and med < 10.0:
        return 1.0, "m(auto)"
    return rs_scale, "mm->m(auto)"


# ==============================
#    DEPTH->COLOR ALIGNMENT
# ==============================


def _depth_to_color_align(
    depth_m: np.ndarray, intr_d: dict, intr_c: dict, T_c_d: np.ndarray
) -> np.ndarray:
    """Project depth pixels to color camera plane (z-buffered)."""
    Hd, Wd = int(intr_d["height"]), int(intr_d["width"])
    Hc, Wc = int(intr_c["height"]), int(intr_c["width"])
    fx_d, fy_d, cx_d, cy_d = (
        float(intr_d["fx"]),
        float(intr_d["fy"]),
        float(intr_d["ppx"]),
        float(intr_d["ppy"]),
    )
    fx_c, fy_c, cx_c, cy_c = (
        float(intr_c["fx"]),
        float(intr_c["fy"]),
        float(intr_c["ppx"]),
        float(intr_c["ppy"]),
    )

    u = np.arange(Wd, dtype=np.float32)
    v = np.arange(Hd, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth_m
    valid = z > 0
    if not np.any(valid):
        return np.zeros((Hc, Wc), dtype=np.float32)

    uu = uu[valid]
    vv = vv[valid]
    z = z[valid].astype(np.float32)
    Xd = (uu - cx_d) * z / fx_d
    Yd = (vv - cy_d) * z / fy_d
    Zd = z
    P_d = np.stack([Xd, Yd, Zd], axis=0)  # 3xN

    Rc = T_c_d[:3, :3]
    tc = T_c_d[:3, 3:4]
    P_c = Rc @ P_d + tc
    Xc, Yc, Zc = P_c[0], P_c[1], P_c[2]

    uc = (fx_c * (Xc / Zc) + cx_c).round().astype(np.int32)
    vc = (fy_c * (Yc / Zc) + cy_c).round().astype(np.int32)
    m = (uc >= 0) & (uc < Wc) & (vc >= 0) & (vc < Hc) & (Zc > 0)
    uc, vc, Zc = uc[m], vc[m], Zc[m].astype(np.float32)

    flat = np.full(Hc * Wc, np.inf, dtype=np.float32)
    idx = vc * Wc + uc
    np.minimum.at(flat, idx, Zc)  # z-buffer: keep nearest
    flat[np.isinf(flat)] = 0.0
    return flat.reshape(Hc, Wc)


def _load_pair(
    img_dir: Path, stem: str
) -> tuple[o3d.geometry.Image, np.ndarray] | None:
    rgb_path = img_dir / f"{stem}_rgb.png"
    d_path = img_dir / f"{stem}_depth.npy"
    if not (rgb_path.exists() and d_path.exists()):
        _log("IO", f"Missing pair for {stem}", level="warning")
        return None
    rgb = o3d.io.read_image(str(rgb_path))
    depth_raw = np.load(d_path)
    return rgb, depth_raw


def rgbd_to_pcd(
    rgb: o3d.geometry.Image,
    depth_m_color: np.ndarray,
    intr_color: dict,
    depth_trunc: float,
) -> o3d.geometry.PointCloud:
    intr = _o3d_intr_from_dict(intr_color)
    depth_img = o3d.geometry.Image(
        np.clip(depth_m_color, 0.0, depth_trunc).astype(np.float32)
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb,
        depth_img,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


# ==============================
#        TRANSFORMS (BASE)
# ==============================


def _euler_to_matrix(rx: float, ry: float, rz: float, order: str) -> np.ndarray:
    """Use a library method for Euler to rotation matrix."""
    if _HAS_SCIPY:
        # SciPy expects radians; order like 'xyz' or 'zyx'
        return _SciRot.from_euler(order, [rx, ry, rz]).as_matrix()
    # Fallback to Open3D helpers (xyz/zyx)
    if order == "xyz":
        return o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
    if order == "zyx":
        return o3d.geometry.get_rotation_matrix_from_zyx([rz, ry, rx])
    raise ValueError("order must be 'xyz' or 'zyx'")


def build_T_base_cam(
    pose: dict, unit_scale: float, order: str, he: HandEye
) -> np.ndarray:
    """BASE <- TCP <- CAM (COLOR). Poses scaled mm->m; hand-eye already in meters."""
    rx, ry, rz = [np.deg2rad(float(pose[a])) for a in ("rx", "ry", "rz")]
    R = _euler_to_matrix(rx, ry, rz, order)
    t = (
        np.array([float(pose[a]) for a in ("x", "y", "z")], dtype=np.float64)
        * unit_scale
    )
    T_base_tcp = np.eye(4, dtype=np.float64)
    T_base_tcp[:3, :3] = R
    T_base_tcp[:3, 3] = t
    T_tcp_cam = _handeye_matrix(he)  # CAM is RGB
    return T_base_tcp @ T_tcp_cam


# ==============================
#       REGISTRATION STEPS
# ==============================


def _ensure_normals(p: o3d.geometry.PointCloud, radius: float) -> None:
    if len(p.points) == 0:
        return
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
    p.orient_normals_consistent_tangent_plane(50)


def _compute_fpfh(p: o3d.geometry.PointCloud, radius: float):
    return o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )


def _fgr(
    src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, sf, tf, max_corr: float
) -> np.ndarray:
    rk = o3d.pipelines.registration
    opt = rk.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_corr, iteration_number=32
    )
    res = rk.registration_fgr_based_on_feature_matching(src, tgt, sf, tf, opt)
    return res.transformation


def _ransac(
    src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, sf, tf, dist: float
) -> np.ndarray:
    rk = o3d.pipelines.registration
    chk1 = rk.CorrespondenceCheckerBasedOnEdgeLength(0.05)
    chk2 = rk.CorrespondenceCheckerBasedOnDistance(dist)
    res = rk.registration_ransac_based_on_feature_matching(
        src,
        tgt,
        sf,
        tf,
        True,
        dist,
        rk.TransformationEstimationPointToPoint(False),
        4,
        [chk1, chk2],
        rk.RANSACConvergenceCriteria(60000, 0.999),
    )
    return res.transformation


def _icp_ptp(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init: np.ndarray,
    max_corr: float,
    iters: int,
) -> np.ndarray:
    rk = o3d.pipelines.registration
    res = rk.registration_icp(
        src,
        tgt,
        max_corr,
        init,
        rk.TransformationEstimationPointToPlane(),
        rk.ICPConvergenceCriteria(max_iteration=iters),
    )
    _log("ICP", f"fitness={res.fitness:.4f} rmse={res.inlier_rmse:.6f}")
    return res.transformation


# ==============================
#   CLEANING & PREVIEW HELPERS
# ==============================


def _clean_and_downsample(
    pcd: o3d.geometry.PointCloud, vox: float
) -> tuple[o3d.geometry.PointCloud, int]:
    before = len(pcd.points)
    q = _clone_pcd(pcd).voxel_down_sample(vox)
    if len(q.points) > 1000:
        _, idx = q.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NN, std_ratio=OUTLIER_STD
        )
        q = q.select_by_index(idx)
    removed = before - len(q.points)
    return q, removed


def _refine_to_target(
    src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, base_vox: float
) -> o3d.geometry.PointCloud:
    """FPFH global init (FGR/RANSAC) + ICP point-to-plane. Ensure normals on BOTH clouds."""
    if len(src.points) == 0 or len(tgt.points) == 0:
        return src
    vox_fpfh = max(base_vox, VOX_FPFH_MIN)
    vox_icp = max(base_vox * 0.6, VOX_ICP_MIN)
    rad_norm = vox_fpfh * 2.0
    rad_fpfh = vox_fpfh * 5.0
    dist_fgr = vox_fpfh * 4.0
    dist_ran = vox_fpfh * 3.0
    dist_icp = vox_icp * 3.0

    s_ds = _clone_pcd(src).voxel_down_sample(vox_fpfh)
    t_ds = _clone_pcd(tgt).voxel_down_sample(vox_fpfh)
    _ensure_normals(s_ds, rad_norm)
    _ensure_normals(t_ds, rad_norm)
    sf = _compute_fpfh(s_ds, rad_fpfh)
    tf = _compute_fpfh(t_ds, rad_fpfh)

    T = None
    try:
        T = _fgr(s_ds, t_ds, sf, tf, dist_fgr)
        used = "FGR"
    except Exception:
        try:
            T = _ransac(s_ds, t_ds, sf, tf, dist_ran)
            used = "RANSAC"
        except Exception:
            used = "none"
            T = np.eye(4)
    _log("GLOBAL", f"refine global={used}")

    s_icp = _clone_pcd(src).voxel_down_sample(vox_icp)
    t_icp = _clone_pcd(tgt).voxel_down_sample(vox_icp)
    _ensure_normals(s_icp, vox_icp * 2.0)
    _ensure_normals(t_icp, vox_icp * 2.0)

    s_icp.transform(T)
    Ticp = _icp_ptp(s_icp, t_icp, np.eye(4), dist_icp, ICP_ITERS)
    Tfin = Ticp @ T

    out = _clone_pcd(src)
    out.transform(Tfin)
    return out


def _maybe_compact_preview(
    preview: o3d.geometry.PointCloud, target_vox: float
) -> o3d.geometry.PointCloud:
    if len(preview.points) < 10000:
        return preview
    q = preview.voxel_down_sample(target_vox)
    _ensure_normals(q, target_vox * 2.0)
    return q


# ==============================
#         PIPELINE CFG
# ==============================


@dataclass
class PipelineCfg:
    bbox_points: list[list[float]] = None
    poses_json: str = POSES_JSON_NAME
    img_dir: str = IMG_DIR_NAME
    pose_unit_mode: str = "mm"  # robot logs are in mm; convert to meters
    euler_mode: str = "auto"  # try 'xyz' and 'zyx' for best ROI fit on frame0
    he_dir_mode: str = "tcp_cam"  # hand-eye given as CAM->TCP
    depth_trunc: float = DEPTH_TRUNC
    quality_preset: str = "best"  # TSDF only, if enabled
    remove_outliers: bool = True
    merge_vox: float = 0.0  # optional final downsample
    viz_stages: bool = False
    viz_every_n: int = 3
    coord_frame_size: float = 0.001
    use_tsdf_flag: bool = False

    @property
    def reg_rd(self) -> float:
        return 0.01

    def use_tsdf(self) -> bool:
        return self.use_tsdf_flag


DEFAULT_CFG = PipelineCfg(
    bbox_points=BBOX_POINTS,
    pose_unit_mode=POSES_UNIT_MODE,
    euler_mode="auto",
    he_dir_mode="tcp_cam",
    depth_trunc=DEPTH_TRUNC,
    quality_preset="fast",
    remove_outliers=True,
    merge_vox=0.0,
    viz_stages=False,
    use_tsdf_flag=False,
)


# ==============================
#         MAIN MERGE API
# ==============================


def autotune_extrinsics(
    pcd0_cam: o3d.geometry.PointCloud,
    pose0: dict,
    aabb: o3d.geometry.AxisAlignedBoundingBox,
    pose_unit_mode: str,
    euler_mode: str,
    he_dir_mode: str,
    he: HandEye,
) -> tuple[float, str, str]:
    """Pick unit scale, Euler order, and HE direction that maximize fraction inside AABB."""
    scales = [(pose_unit_mode, _pose_scale(pose_unit_mode))]
    orders = ["xyz"]
    dirs = [he_dir_mode]

    best = (0.0, scales[0][1], orders[0], dirs[0])
    for _, s in scales:
        for order in orders:
            for hdir in dirs:
                he_try = HandEye(R=he.R, t=he.t, direction=hdir)
                T = build_T_base_cam(pose0, s, order, he_try)
                p = _clone_pcd(pcd0_cam)
                p.transform(T)
                P = np.asarray(p.points)
                if len(P) == 0:
                    frac = 0.0
                else:
                    inside = (
                        (P >= aabb.get_min_bound()) & (P <= aabb.get_max_bound())
                    ).all(axis=1)
                    frac = float(inside.sum()) / max(1, len(P))
                if frac > best[0]:
                    best = (frac, s, order, hdir)
    _log(
        "MERGE",
        f"Autotune -> unit_scale={best[1]} order='{best[2]}' he_dir='{best[3]}' (inside {best[0]*100:.2f}%)",
    )
    return best[1], best[2], best[3]


def crop_to_aabb(
    pcd: o3d.geometry.PointCloud, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> o3d.geometry.PointCloud:
    return pcd.crop(aabb)


def merge_capture(root: Path, cfg: PipelineCfg) -> o3d.geometry.PointCloud:
    """Main merge routine: RGB-D -> BASE -> crop -> refine -> accumulate."""
    root = Path(root)
    intr_d, intr_c, T_c_d, rs_scale = build_intrinsics(root)
    aabb = _make_aabb(cfg.bbox_points)

    poses = load_poses(root / cfg.poses_json)
    stems = sorted(poses.keys())
    if not stems:
        _log("MERGE", f"No poses found at {root / cfg.poses_json}", level="warning")
        return o3d.geometry.PointCloud()

    img_dir = root / cfg.img_dir
    he = HAND_EYE

    preview = o3d.geometry.PointCloud()

    pair0 = _load_pair(img_dir, stems[0])
    if pair0 is None:
        return o3d.geometry.PointCloud()
    rgb0, depth_raw0 = pair0
    s0, label0 = _guess_depth_scale(depth_raw0, DEPTH_MODE, rs_scale)
    _log(
        "DEPTH",
        f"{stems[0]}_depth.npy: dtype={depth_raw0.dtype} med={float(np.nanmedian(depth_raw0)):.4f} scale={s0} [{label0}]",
    )
    depth0_m = depth_raw0.astype(np.float32) * s0
    depth0_c = _depth_to_color_align(depth0_m, intr_d, intr_c, T_c_d)

    pcd0_cam = rgbd_to_pcd(rgb0, depth0_c, intr_c, depth_trunc=cfg.depth_trunc)
    _log_cloud_stats(f"[S1 {stems[0]}] CAMERA", pcd0_cam)

    scale, order, hdir = autotune_extrinsics(
        pcd0_cam,
        poses[stems[0]],
        aabb,
        cfg.pose_unit_mode,
        cfg.euler_mode,
        cfg.he_dir_mode,
        he,
    )
    T0 = build_T_base_cam(
        poses[stems[0]], scale, order, HandEye(R=he.R, t=he.t, direction=hdir)
    )
    pcd0_base = _clone_pcd(pcd0_cam)
    pcd0_base.transform(T0)
    _log_cloud_stats(f"[S2 {stems[0]}] BASE (autotune)", pcd0_base)

    pcd0_crop = crop_to_aabb(pcd0_base, aabb)
    _log(
        "INFO",
        f"[S3 {stems[0]}] Crop -> {len(pcd0_crop.points)} pts (was {len(pcd0_base.points)})",
    )
    _log_cloud_stats(f"[S3 {stems[0]}] CROP", pcd0_crop)

    if len(pcd0_crop.points) > 0:
        P0 = np.asarray(pcd0_crop.points)
        ext0 = P0.max(0) - P0.min(0)
        base_vox = max(float(np.linalg.norm(ext0)) / 300.0, VOX_ICP_MIN)
        pcd0_clean, _ = _clean_and_downsample(
            pcd0_crop, max(base_vox, PREVIEW_VOX_TARGET)
        )
        preview += pcd0_clean

    prepared: list[tuple[str, o3d.geometry.Image, np.ndarray, np.ndarray]] = []
    stems_rest = stems[1:]
    for stem in tqdm(stems_rest, desc="Preprocess", leave=False):
        pair = _load_pair(img_dir, stem)
        if pair is None:
            continue
        rgb, depth_raw = pair
        s, label = _guess_depth_scale(depth_raw, DEPTH_MODE, rs_scale)
        _log(
            "DEPTH",
            f"{stem}_depth.npy: dtype={depth_raw.dtype} med={float(np.nanmedian(depth_raw)):.4f} scale={s} [{label}]",
        )
        depth_m = depth_raw.astype(np.float32) * s
        depth_c = _depth_to_color_align(depth_m, intr_d, intr_c, T_c_d)
        T = build_T_base_cam(
            poses[stem], scale, order, HandEye(R=he.R, t=he.t, direction=hdir)
        )
        prepared.append((stem, rgb, depth_c, T))

    # --- Merge bar: crop, clean, register (FPFH+ICP), accumulate
    for stem, rgb, depth_c, T in tqdm(prepared, desc="Merge", leave=False):
        pcd_cam = rgbd_to_pcd(rgb, depth_c, intr_c, depth_trunc=cfg.depth_trunc)
        pcd_base = _clone_pcd(pcd_cam)
        pcd_base.transform(T)
        pcd_crop = crop_to_aabb(pcd_base, aabb)
        _log(
            "INFO",
            f"[S3 {stem}] Crop -> {len(pcd_crop.points)} pts (was {len(pcd_base.points)})",
        )

        if len(pcd_crop.points) < MIN_CROP_PTS:
            _log("WARN", f"[{stem}] skipped (tiny after crop)", level="warning")
            continue

        P = np.asarray(pcd_crop.points)
        ext = P.max(0) - P.min(0)
        base_vox = max(float(np.linalg.norm(ext)) / 500.0, VOX_ICP_MIN)

        pcd_clean, removed = _clean_and_downsample(
            pcd_crop, max(base_vox, PREVIEW_VOX_TARGET)
        )
        _log(
            "INFO",
            f"[S4 {stem}] Clean {len(pcd_crop.points)} -> {len(pcd_clean.points)} (removed {removed})",
        )

        if len(preview.points) > 5000:
            refined = _refine_to_target(pcd_clean, preview, base_vox)
        else:
            refined = pcd_clean

        preview += refined
        preview = _maybe_compact_preview(preview, PREVIEW_VOX_TARGET)

    return preview


# ==============================
#        RUNNABLE ENTRY
# ==============================


def _visualize_and_save(pcd: o3d.geometry.PointCloud) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pcd = OUTPUT_DIR / "merged_final.pcd"
    o3d.io.write_point_cloud(str(out_pcd), pcd)
    _log("SAVE", f"Saved merged cloud to {out_pcd}")
    if OPEN_VIEWER:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Final Merged Cloud", width=VIEW_W, height=VIEW_H)
        vis.add_geometry(pcd)
        vis.run()
        if SAVE_SCREENSHOT:
            png = OUTPUT_DIR / "merged_final.png"
            vis.capture_screen_image(str(png), do_render=True)
            _log("VIZ", f"Saved screenshot to {png}")
        vis.destroy_window()


def main() -> int:
    try:
        root = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_ROOT
        if not root.exists():
            raise FileNotFoundError(root)
        cfg = DEFAULT_CFG
        merged = merge_capture(root, cfg)

        if len(merged.points) == 0:
            _log("FINAL", "[FINAL] Empty cloud.", level="warning")
        else:
            n0 = len(merged.points)
            if cfg.merge_vox and cfg.merge_vox > 0:
                merged = merged.voxel_down_sample(cfg.merge_vox)
                _log(
                    "FINAL",
                    f"[FINAL] Downsample {n0} -> {len(merged.points)} @ {cfg.merge_vox:.4f}",
                )
            else:
                _log(
                    "FINAL",
                    f"[FINAL] skip voxel_down_sample (merge_vox={cfg.merge_vox})",
                )

            if cfg.remove_outliers and len(merged.points) > 2000:
                _, idx = merged.remove_statistical_outlier(
                    nb_neighbors=OUTLIER_NN, std_ratio=OUTLIER_STD
                )
                removed = len(merged.points) - len(idx)
                merged = merged.select_by_index(idx)
                _log("FINAL", f"[FINAL] Outliers removed: {removed}")

        _visualize_and_save(merged)
        return 0
    except Exception as e:
        _log("MERGE", f"Failed: {e}", level="error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
