# borunte/cam_rs.py
"""Intel RealSense preview and capture utilities (RGB, depth, both).

PreviewStreamer runs in a background thread and shows a window.
At a grid point, the main thread waits for an action: SPACE -> capture,
ESC or Q -> skip. During capture we briefly stop preview to free the device.
"""

from __future__ import annotations

import time
from queue import SimpleQueue, Empty
from typing import Any, Dict, Optional, Tuple
from threading import Lock

import cv2
import numpy as np
import pyrealsense2 as rs

from utils.logger import Logger
from .config import (
    PREVIEW_VIEW,
    PREVIEW_DEPTH_W,
    PREVIEW_DEPTH_H,
    PREVIEW_DEPTH_FPS,
    PREVIEW_COLOR_W,
    PREVIEW_COLOR_H,
    PREVIEW_COLOR_FPS,
    PREVIEW_DECIMATION,
    PREVIEW_DISPARITY_SHIFT,
    WARMUP_FRAMES,
    SPATIAL_MAG,
    SPATIAL_SMOOTH_ALPHA,
    SPATIAL_SMOOTH_DELTA,
    SPATIAL_HOLES_FILL,
    DEPTH_VIZ_MIN_M,
    DEPTH_VIZ_MAX_M,
)

_log = Logger.get_logger()
_WIN = "Preview"


def _first_device() -> rs.device:
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def _set_and_verify_disparity_shift(dev: rs.device, value: int) -> int:
    time.sleep(0.2)
    adv = rs.rs400_advanced_mode(dev)
    if not adv.is_enabled():
        adv.toggle_advanced_mode(True)
        time.sleep(1.0)
    adv2 = rs.rs400_advanced_mode(dev)
    tbl = adv2.get_depth_table()
    tbl.disparityShift = int(value)
    adv2.set_depth_table(tbl)
    time.sleep(0.2)
    return int(adv2.get_depth_table().disparityShift)


def _depth_to_viz(depth_m: np.ndarray) -> np.ndarray:
    d = depth_m
    mask = d > 0
    if DEPTH_VIZ_MIN_M is None or DEPTH_VIZ_MAX_M is None:
        if np.any(mask):
            vmin = float(np.percentile(d[mask], 2.0))
            vmax = float(np.percentile(d[mask], 98.0))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(DEPTH_VIZ_MIN_M), float(DEPTH_VIZ_MAX_M)
    vmax = max(vmax, vmin + 1e-3)
    d8 = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    d8 = (d8 * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    color[~mask] = (0, 0, 0)
    return color


class PreviewStreamer:
    """Persistent preview window running in a background thread."""

    def __init__(
        self,
        view: str = PREVIEW_VIEW,
        disparity_shift: Optional[int] = PREVIEW_DISPARITY_SHIFT,
    ):
        self.view = view
        self.disp = disparity_shift
        self.pipe: Optional[rs.pipeline] = None
        self.prof: Optional[rs.pipeline_profile] = None
        self.depth_scale: Optional[float] = None
        self.decimate: Optional[rs.decimation_filter] = None
        self.to_disp: Optional[rs.disparity_transform] = None
        self.from_disp: Optional[rs.disparity_transform] = None
        self.spatial: Optional[rs.spatial_filter] = None
        self.actions: SimpleQueue[str] = SimpleQueue()
        self._stop = False
        self._started = False
        self.lock = Lock()
        self.last_rgb: Optional[np.ndarray] = None
        self.last_depth_m: Optional[np.ndarray] = None

    def start(self) -> None:
        import threading

        if self._started:
            return
        self._started = True
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        _log.tag("PREVIEW", "streamer started")

    def stop(self) -> None:
        if not self._started:
            return
        self._stop = True
        try:
            self._t.join(timeout=2.0)
        except Exception:
            pass
        self._started = False
        _log.tag("PREVIEW", "streamer stopped")

    def snapshot(
        self,
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[rs.pipeline_profile]
    ]:
        """Return copies of (rgb, depth_m, profile) from the preview."""
        with self.lock:
            rgb = None if self.last_rgb is None else self.last_rgb.copy()
            dpt = None if self.last_depth_m is None else self.last_depth_m.copy()
            prof = self.prof
        return rgb, dpt, prof

    def _open(self) -> None:
        dev = _first_device()
        want_depth = self.view in ("depth", "both")
        want_color = self.view in ("rgb", "both")
        if want_depth and self.disp is not None:
            try:
                applied = _set_and_verify_disparity_shift(dev, int(self.disp))
                _log.tag("RS2", f"preview disparityShift={applied}")
            except Exception as e:
                _log.tag(
                    "RS2", f"preview disparityShift not applied: {e}", level="warning"
                )

        self.pipe = rs.pipeline()
        cfg = rs.config()
        if want_depth:
            cfg.enable_stream(
                rs.stream.depth,
                PREVIEW_DEPTH_W,
                PREVIEW_DEPTH_H,
                rs.format.z16,
                PREVIEW_DEPTH_FPS,
            )
        if want_color:
            cfg.enable_stream(
                rs.stream.color,
                PREVIEW_COLOR_W,
                PREVIEW_COLOR_H,
                rs.format.bgr8,
                PREVIEW_COLOR_FPS,
            )
        self.prof = self.pipe.start(cfg)
        self.depth_scale = (
            self.prof.get_device().first_depth_sensor().get_depth_scale()
            if want_depth
            else None
        )
        if want_depth:
            self.decimate = rs.decimation_filter()
            self.decimate.set_option(
                rs.option.filter_magnitude, float(PREVIEW_DECIMATION)
            )
            self.to_disp = rs.disparity_transform(True)
            self.from_disp = rs.disparity_transform(False)
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, float(SPATIAL_MAG))
            self.spatial.set_option(
                rs.option.filter_smooth_alpha, float(SPATIAL_SMOOTH_ALPHA)
            )
            self.spatial.set_option(
                rs.option.filter_smooth_delta, float(SPATIAL_SMOOTH_DELTA)
            )
            self.spatial.set_option(rs.option.holes_fill, float(SPATIAL_HOLES_FILL))
        for _ in range(max(0, WARMUP_FRAMES)):
            self.pipe.wait_for_frames()

    def _close(self) -> None:
        try:
            if self.pipe:
                self.pipe.stop()
        except Exception:
            pass
        self.pipe = None
        self.prof = None

    def _loop(self) -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        if self.view == "both":
            w, h = PREVIEW_COLOR_W * 2, PREVIEW_COLOR_H
        elif self.view == "rgb":
            w, h = PREVIEW_COLOR_W, PREVIEW_COLOR_H
        else:
            w, h = PREVIEW_DEPTH_W, PREVIEW_DEPTH_H
        cv2.resizeWindow(_WIN, w, h)
        cv2.moveWindow(_WIN, 100, 50)

        self._open()
        try:
            while not self._stop:
                frames = self.pipe.wait_for_frames()
                rgb_img, depth_viz, depth_m = None, None, None

                if self.view in ("rgb", "both"):
                    cf = frames.get_color_frame()
                    if cf:
                        rgb_img = np.asanyarray(cf.get_data()).copy()

                if self.view in ("depth", "both"):
                    df = frames.get_depth_frame()
                    if df:
                        dfp = self.decimate.process(df)
                        dfp = self.to_disp.process(dfp)
                        dfp = self.spatial.process(dfp)
                        dfp = self.from_disp.process(dfp)
                        depth_m = np.asanyarray(dfp.get_data()).astype(
                            np.float32
                        ) * float(self.depth_scale)
                        depth_viz = _depth_to_viz(depth_m)

                with self.lock:
                    if rgb_img is not None:
                        self.last_rgb = rgb_img
                    if depth_m is not None:
                        self.last_depth_m = depth_m

                if (
                    self.view == "both"
                    and rgb_img is not None
                    and depth_viz is not None
                ):
                    hmin = min(depth_viz.shape[0], rgb_img.shape[0])
                    d_res = cv2.resize(
                        depth_viz,
                        (int(depth_viz.shape[1] * hmin / depth_viz.shape[0]), hmin),
                    )
                    c_res = cv2.resize(
                        rgb_img,
                        (int(rgb_img.shape[1] * hmin / rgb_img.shape[0]), hmin),
                    )
                    viz = np.hstack([c_res, d_res])
                else:
                    viz = rgb_img if self.view == "rgb" else depth_viz

                if viz is not None:
                    cv2.putText(
                        viz,
                        "SPACE capture | Q or ESC skip",
                        (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(_WIN, viz)

                key = cv2.waitKey(1) & 0xFFFF
                if key in (13, 10, 32):
                    self.actions.put("capture")
                elif key in (27, ord("q"), ord("Q")):
                    self.actions.put("skip")
        finally:
            try:
                self._close()
            except Exception:
                pass

    def poll_action(self, timeout_s: float | None) -> Optional[str]:
        try:
            return self.actions.get(timeout=timeout_s)
        except Empty:
            return None


def capture_one_pair(
    ds_val: int, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Capture one RGB and depth pair with filtering; returns (rgb, depth, meta)."""
    dev = _first_device()
    applied = _set_and_verify_disparity_shift(dev, int(ds_val))
    _log.tag("CAPTURE", f"disparityShift={applied}")

    depth, color = cfg["depth"], cfg["color"]
    dec_mag = int(depth["decimation"])

    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(
        rs.stream.depth,
        int(depth["w"]),
        int(depth["h"]),
        rs.format.z16,
        int(depth["fps"]),
    )
    rs_cfg.enable_stream(
        rs.stream.color,
        int(color["w"]),
        int(color["h"]),
        rs.format.bgr8,
        int(color["fps"]),
    )
    prof = pipe.start(rs_cfg)

    depth_scale = float(prof.get_device().first_depth_sensor().get_depth_scale())

    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, float(dec_mag))
    to_disp = rs.disparity_transform(True)
    from_disp = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, float(SPATIAL_MAG))
    spatial.set_option(rs.option.filter_smooth_alpha, float(SPATIAL_SMOOTH_ALPHA))
    spatial.set_option(rs.option.filter_smooth_delta, float(SPATIAL_SMOOTH_DELTA))
    spatial.set_option(rs.option.holes_fill, float(SPATIAL_HOLES_FILL))

    for _ in range(max(0, WARMUP_FRAMES)):
        pipe.wait_for_frames()

    frames = pipe.wait_for_frames()
    df, cf = frames.get_depth_frame(), frames.get_color_frame()
    if not df or not cf:
        pipe.stop()
        raise RuntimeError("No frames in capture pipeline")

    dfp = decimate.process(df)
    dfp = to_disp.process(dfp)
    dfp = spatial.process(dfp)
    dfp = from_disp.process(dfp)

    color_np = np.asanyarray(cf.get_data()).copy()
    depth_m = np.asanyarray(dfp.get_data()).astype(np.float32) * float(depth_scale)

    meta = {"profile": prof, "applied_disparity": applied, "decimation": dec_mag}
    pipe.stop()
    return color_np, depth_m, meta
