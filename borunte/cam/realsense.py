# borunte/cam_rs.py
"""Intel RealSense preview and capture helpers."""

from __future__ import annotations

import threading
import time
from queue import Empty, SimpleQueue
from threading import Lock
from typing import Any

import numpy as np
import pyrealsense2 as rs

from borunte.utils.error_tracker import ErrorTracker
from borunte.utils.logger import get_logger

from ..config import BORUNTE_CONFIG, BorunteConfig

# Проверка наличия GUI-версии OpenCV
try:
    import cv2

    HAS_CV2_GUI = hasattr(cv2, "namedWindow")
except Exception:
    cv2 = None
    HAS_CV2_GUI = False

_log = get_logger()


def _first_device() -> rs.device:
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        raise RuntimeError("No RealSense devices found")
    return devices[0]


def _set_and_verify_disparity_shift(device: rs.device, value: int) -> int:
    time.sleep(0.2)
    adv = rs.rs400_advanced_mode(device)
    if not adv.is_enabled():
        adv.toggle_advanced_mode(True)
        time.sleep(1.0)
    adv2 = rs.rs400_advanced_mode(device)
    table = adv2.get_depth_table()
    table.disparityShift = int(value)
    adv2.set_depth_table(table)
    time.sleep(0.2)
    return int(adv2.get_depth_table().disparityShift)


def _depth_to_viz(depth_m: np.ndarray, config: BorunteConfig) -> np.ndarray:
    if not HAS_CV2_GUI:
        return np.zeros_like(depth_m, dtype=np.uint8)

    preview = config.preview
    d = depth_m
    mask = d > 0
    if preview.depth_viz_min_m is None or preview.depth_viz_max_m is None:
        if np.any(mask):
            vmin = float(np.percentile(d[mask], 2.0))
            vmax = float(np.percentile(d[mask], 98.0))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin = float(preview.depth_viz_min_m)
        vmax = float(preview.depth_viz_max_m)
    vmax = max(vmax, vmin + 1e-3)
    norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~mask] = (0, 0, 0)
    return color


class PreviewStreamer:
    """Persistent preview window running in a background thread."""

    def __init__(
        self,
        config: BorunteConfig = BORUNTE_CONFIG,
        view: str | None = None,
        disparity_shift: int | None = None,
    ) -> None:
        self.config = config
        self.view = view or config.preview.view
        self.disparity_shift = (
            config.preview.disparity_shift if disparity_shift is None else disparity_shift
        )
        self.pipe: rs.pipeline | None = None
        self.profile: rs.pipeline_profile | None = None
        self.depth_scale: float | None = None
        self.decimate: rs.decimation_filter | None = None
        self.to_disp: rs.disparity_transform | None = None
        self.from_disp: rs.disparity_transform | None = None
        self.spatial: rs.spatial_filter | None = None
        self.actions: SimpleQueue[str] = SimpleQueue()
        self._stop = False
        self._started = False
        self._lock = Lock()
        self._logged_intrinsics = False
        self.last_rgb: np.ndarray | None = None
        self.last_depth_m: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._available: bool | None = None

    def _probe_available(self) -> bool:
        """Return True if a RealSense device is present."""
        if self._available is None:
            try:
                _ = _first_device()
                self._available = True
            except Exception as e:
                _log.tag("PREVIEW", f"no RealSense device: {e}", level="warning")
                self._available = False
        return bool(self._available)

    def snapshot(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, rs.pipeline_profile | None]:
        with self._lock:
            rgb = None if self.last_rgb is None else self.last_rgb.copy()
            depth = None if self.last_depth_m is None else self.last_depth_m.copy()
            profile = self.profile
        return rgb, depth, profile

    def poll_action(self, timeout_s: float | None) -> str | None:
        try:
            return self.actions.get(timeout=timeout_s)
        except Empty:
            return None

    def _open(self) -> None:
        device = _first_device()
        want_depth = self.view in ("depth", "both")
        want_color = self.view in ("rgb", "both")
        if want_depth and self.disparity_shift is not None:
            try:
                applied = _set_and_verify_disparity_shift(device, int(self.disparity_shift))
                _log.tag("RS2", f"preview disparityShift={applied}")
            except Exception as exc:
                ErrorTracker.report(exc)
                _log.tag("RS2", f"preview disparityShift not applied: {exc}", level="warning")

        self.pipe = rs.pipeline()
        cfg = rs.config()
        streams = self.config.capture_profile
        if want_depth:
            depth = streams.depth
            cfg.enable_stream(
                rs.stream.depth,
                depth.width,
                depth.height,
                rs.format.z16,
                depth.fps,
            )
        if want_color:
            color = streams.color
            cfg.enable_stream(
                rs.stream.color,
                color.width,
                color.height,
                rs.format.bgr8,
                color.fps,
            )
        self.profile = self.pipe.start(cfg)
        if self.profile and not self._logged_intrinsics:
            try:
                dev = self.profile.get_device()
                d_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
                c_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
                _log.tag(
                    "RS2",
                    f"depth fx={d_stream.get_intrinsics().fx:.2f} fy={d_stream.get_intrinsics().fy:.2f}",
                )
                _log.tag(
                    "RS2",
                    f"color fx={c_stream.get_intrinsics().fx:.2f} fy={c_stream.get_intrinsics().fy:.2f}",
                )
                _log.tag(
                    "RS2",
                    f"depth_scale={dev.first_depth_sensor().get_depth_scale():.6f}",
                )
                self._logged_intrinsics = True
            except Exception as exc:
                ErrorTracker.report(exc)
        if want_depth:
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            self.decimate = rs.decimation_filter()
            self.decimate.set_option(
                rs.option.filter_magnitude, float(self.config.preview.decimation)
            )
            self.to_disp = rs.disparity_transform(True)
            self.from_disp = rs.disparity_transform(False)
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(
                rs.option.filter_magnitude, float(self.config.preview.spatial_magnitude)
            )
            self.spatial.set_option(
                rs.option.filter_smooth_alpha,
                float(self.config.preview.spatial_smooth_alpha),
            )
            self.spatial.set_option(
                rs.option.filter_smooth_delta,
                float(self.config.preview.spatial_smooth_delta),
            )
            self.spatial.set_option(
                rs.option.holes_fill, float(self.config.preview.spatial_holes_fill)
            )
        for _ in range(max(0, self.config.preview.warmup_frames)):
            self.pipe.wait_for_frames()

    def _close(self) -> None:
        try:
            if self.pipe:
                self.pipe.stop()
        except Exception as exc:
            ErrorTracker.report(exc)
        self.pipe = None
        self.profile = None

    def start(self) -> None:
        if self._started:
            return
        if not self._probe_available():
            _log.tag("PREVIEW", "skip start: device not available", level="warning")
            return

        # Проверка GUI возможностей
        if not HAS_CV2_GUI:
            _log.tag(
                "PREVIEW",
                "GUI disabled (opencv-python-headless), using headless mode",
                level="warning",
            )

        self._stop = False
        self._started = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        _log.tag("PREVIEW", "streamer started")

    def stop(self) -> None:
        if not self._started and not self._thread:
            return
        self._stop = True
        t = self._thread
        self._thread = None
        try:
            if t:
                t.join(timeout=2.0)
        except Exception:
            pass
        self._started = False
        _log.tag("PREVIEW", "streamer stopped")

    def is_running(self) -> bool:
        """True if the preview thread is alive and stop flag is not set."""
        t = self._thread
        return bool(t and t.is_alive() and not self._stop)

    def is_alive(self) -> bool:
        return self.is_running()

    def _loop(self) -> None:
        # Инициализация окна только если есть GUI
        if HAS_CV2_GUI:
            try:
                cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
                streams = self.config.capture_profile
                if self.view == "both":
                    width, height = streams.color.width * 2, streams.color.height
                elif self.view == "rgb":
                    width, height = streams.color.width, streams.color.height
                else:
                    width, height = streams.depth.width, streams.depth.height
                cv2.resizeWindow("Preview", width, height)
                cv2.moveWindow("Preview", 100, 50)
            except Exception as exc:
                _log.tag("PREVIEW", f"window init failed: {exc}", level="warning")

        self._open()
        try:
            while not self._stop and self.pipe:
                frames = self.pipe.wait_for_frames()
                rgb_img, depth_viz, depth_m = None, None, None

                if self.view in ("rgb", "both"):
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        rgb_img = np.asanyarray(color_frame.get_data()).copy()

                if self.view in ("depth", "both") and self.depth_scale is not None:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_proc = depth_frame
                        if self.decimate is not None:
                            depth_proc = self.decimate.process(depth_proc)
                        if self.to_disp is not None:
                            depth_proc = self.to_disp.process(depth_proc)
                        if self.spatial is not None:
                            depth_proc = self.spatial.process(depth_proc)
                        if self.from_disp is not None:
                            depth_proc = self.from_disp.process(depth_proc)
                        depth_m = np.asanyarray(depth_proc.get_data()).astype(np.float32) * float(
                            self.depth_scale
                        )
                        if HAS_CV2_GUI:
                            depth_viz = _depth_to_viz(depth_m, self.config)

                with self._lock:
                    if rgb_img is not None:
                        self.last_rgb = rgb_img
                    if depth_m is not None:
                        self.last_depth_m = depth_m

                # Визуализация только если есть GUI
                if HAS_CV2_GUI:
                    viz = None
                    if self.view == "both" and rgb_img is not None and depth_viz is not None:
                        hmin = min(depth_viz.shape[0], rgb_img.shape[0])
                        depth_res = cv2.resize(
                            depth_viz,
                            (int(depth_viz.shape[1] * hmin / depth_viz.shape[0]), hmin),
                        )
                        color_res = cv2.resize(
                            rgb_img,
                            (int(rgb_img.shape[1] * hmin / rgb_img.shape[0]), hmin),
                        )
                        viz = np.hstack([color_res, depth_res])
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
                        cv2.imshow("Preview", viz)

                    key = cv2.waitKey(1) & 0xFFFF
                    if key in (13, 10, 32):
                        self.actions.put("capture")
                    elif key in (27, ord("q"), ord("Q")):
                        self.actions.put("skip")
                else:
                    # Headless режим: небольшая задержка
                    time.sleep(0.033)
        finally:
            self._close()
            if HAS_CV2_GUI:
                try:
                    cv2.destroyWindow("Preview")
                except Exception:
                    pass


def capture_one_pair(
    disparity_shift: int,
    config: BorunteConfig = BORUNTE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], rs.pipeline_profile]:
    device = _first_device()
    applied = _set_and_verify_disparity_shift(device, int(disparity_shift))
    _log.tag("CAPTURE", f"disparityShift={applied}")

    profile_cfg = config.capture_profile
    depth_cfg = profile_cfg.depth
    color_cfg = profile_cfg.color

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(
        rs.stream.depth,
        depth_cfg.width,
        depth_cfg.height,
        rs.format.z16,
        depth_cfg.fps,
    )
    rs_cfg.enable_stream(
        rs.stream.color,
        color_cfg.width,
        color_cfg.height,
        rs.format.bgr8,
        color_cfg.fps,
    )
    pipeline_profile = pipeline.start(rs_cfg)
    device = pipeline_profile.get_device()
    depth_scale = float(device.first_depth_sensor().get_depth_scale())

    try:
        for _ in range(max(0, config.preview.warmup_frames)):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Frames missing from pipeline")
        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = depth_raw * depth_scale
        meta = {
            "applied_disparity_shift": int(applied),
            "decimation": profile_cfg.depth.decimation,
            "depth_scale": depth_scale,
        }
        return rgb, depth_m, meta, pipeline_profile
    finally:
        try:
            pipeline.stop()
        except Exception as exc:
            ErrorTracker.report(exc)


__all__ = ["PreviewStreamer", "capture_one_pair"]
