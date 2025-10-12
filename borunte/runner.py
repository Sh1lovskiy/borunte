# borunte/runner.py
"""Capture pipeline orchestrating robot motion and RealSense capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from calib.config import CALIB_CONFIG, CalibConfig
from calib.offline import run_offline
from utils.error_tracker import ErrorTracker, error_scope
from utils.logger import Logger

from .cap_session import CaptureSession
from .cam_rs import PreviewStreamer, capture_one_pair
from .config import BORUNTE_CONFIG, BorunteConfig
from .control import (
    Heartbeat,
    clear_alarm,
    clear_alarm_continue,
    graceful_release,
)
from .grid import build_grid_for_count
from .motion import write_and_run_point
from .state import query_world
from .waypoints import load_default_waypoints
from .wire import RobotClient

_log = Logger.get_logger()


def _log_launch_banner(config: BorunteConfig) -> None:
    ws = config.motion.workspace_m
    tcp = config.motion.tcp_down_uvw
    net = config.network
    _log.tag(
        "START",
        "capture starting "
        f"mode={config.calibration.run_mode} interactive={config.capture_mode.interactive} "
        f"waypoints={config.use_waypoints} view={config.preview.view}",
    )
    _log.tag(
        "CFG",
        f"net={net.host}:{net.port} timeout={net.timeout_s}s keepalive={net.keepalive} "
        f"root={config.capture_root}",
    )
    _log.tag(
        "WS",
        f"X=({ws[0][0]:.1f},{ws[0][1]:.1f}) Y=({ws[1][0]:.1f},{ws[1][1]:.1f}) "
        f"Z=({ws[2][0]:.1f},{ws[2][1]:.1f}) baseUVW=({tcp[0]:.1f},{tcp[1]:.1f},{tcp[2]:.1f})",
    )


def _connect_client(config: BorunteConfig) -> RobotClient:
    client = RobotClient(config=config)
    client.connect()
    return client


def _plan(config: BorunteConfig) -> List[List[float]]:
    if config.use_waypoints and config.waypoints.use_waypoints:
        poses = load_default_waypoints(config)
        _log.tag("PLAN", f"waypoints loaded: {len(poses)} from {config.waypoints.file}")
    else:
        poses = build_grid_for_count(
            ws=config.motion.workspace_m,
            total=config.motion.total_points,
            config=config,
        )
        _log.tag("PLAN", f"grid built: {len(poses)} points target={config.motion.total_points}")
    if poses:
        first, last = poses[0], poses[-1]
        _log.tag(
            "PLAN",
            f"first={tuple(round(v, 3) for v in first[:3])} last={tuple(round(v, 3) for v in last[:3])}",
        )
    return poses


def _cleanup_preview(streamer: Optional[PreviewStreamer]) -> None:
    if streamer:
        try:
            streamer.stop()
        except Exception as exc:
            ErrorTracker.report(exc)


def _cleanup_robot(client: Optional[RobotClient], hb: Optional[Heartbeat]) -> None:
    if not client:
        return
    try:
        graceful_release(client, hb)
    except Exception as exc:
        ErrorTracker.report(exc)
    try:
        client.close()
    except Exception as exc:
        ErrorTracker.report(exc)



def _prepare_preview(config: BorunteConfig) -> Optional[PreviewStreamer]:
    use_preview = bool(config.capture_mode.interactive)
    if not use_preview:
        return None
    try:
        streamer = PreviewStreamer(config=config)
        streamer.start()
        _log.tag("PREVIEW", "ready")
        return streamer
    except Exception as exc:
        ErrorTracker.report(exc)
        _log.tag("PREVIEW", f"disabled: {exc}", level="warning")
        return None


def _capture_preview(
    session: CaptureSession,
    idx_name: str,
    pose: Sequence[float],
    streamer: PreviewStreamer,
) -> None:
    _log.tag("PREVIEW", "SPACE=capture, Q/ESC=skip (waiting)")
    while True:
        action = streamer.poll_action(timeout_s=0.1)
        if action in ("capture", "skip"):
            _log.tag("PREVIEW", f"user={action}")
            break
    if action == "capture":
        rgb, depth, profile = streamer.snapshot()
        if profile is not None:
            session.save_params_json(profile, session.root / "rs2_params.json", 0, 1)
        session.save_preview_snapshot(
            session.root / "preview_single",
            idx_name,
            rgb,
            depth,
        )
        session.update_pose(
            idx_name,
            dict(x=pose[0], y=pose[1], z=pose[2], rx=pose[3], ry=pose[4], rz=pose[5]),
        )
        _log.tag("CAPTURE", f"saved {idx_name} -> preview_single")
    else:
        _log.tag("CAPTURE", "skipped by user")


def _capture_auto(
    session: CaptureSession,
    idx_name: str,
    pose: Sequence[float],
    config: BorunteConfig,
) -> None:
    sweep = config.capture_profile.disparity_sweep or (config.preview.disparity_shift,)
    sweep = tuple(v for v in sweep if v is not None)
    sweep_dir = session.root / "auto_single"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    params_written = False
    for value in sweep:
        rgb, depth, meta, profile = capture_one_pair(value, config=config)
        session.save_rgb_depth(sweep_dir, f"{idx_name}_{value}", rgb, depth)
        if not params_written:
            session.save_params_json(profile, session.root / "rs2_params.json", meta["applied_disparity_shift"], meta["decimation"])
            params_written = True
    session.update_pose(
        idx_name,
        dict(x=pose[0], y=pose[1], z=pose[2], rx=pose[3], ry=pose[4], rz=pose[5]),
    )
    _log.tag("CAPTURE", f"auto record {idx_name} -> {sweep_dir.name}")


def _summary(session: CaptureSession, total: int) -> None:
    root = session.root
    poses_path = root / "poses.json"
    try:
        poses = json.loads(poses_path.read_text("utf-8"))
        count = len(poses)
    except Exception:
        count = 0
    _log.tag("FINAL", f"session={root.name} dir={root} saved={count}/{total}")


def run_capture_pipeline(
    config: BorunteConfig = BORUNTE_CONFIG,
    calib_config: CalibConfig = CALIB_CONFIG,
) -> Path:
    if config.calibration.run_mode == "calib_only":
        if not config.calibration.dataset_override:
            raise FileNotFoundError("calib_only requires dataset_override")
        out = run_offline(config.calibration.dataset_override, calib_config)
        _log.tag("CALIB", f"finished -> {out}")
        return config.calibration.dataset_override

    _log_launch_banner(config)
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener("esc")

    plan = _plan(config)
    session = CaptureSession(config=config)
    streamer: Optional[PreviewStreamer] = None
    client: Optional[RobotClient] = None
    hb: Optional[Heartbeat] = None

    def _cleanup() -> None:
        _cleanup_preview(streamer)
        _cleanup_robot(client, hb)
        _log.tag("FINAL", "cleanup complete")

    ErrorTracker.register_cleanup(_cleanup)

    with error_scope():
        if config.capture_mode.interactive:
            streamer_local = _prepare_preview(config)
            streamer = streamer_local
        client = _connect_client(config)
        hb = Heartbeat(client, period_s=config.network.heartbeat_period_s)
        hb.start()
        _log.tag("HB", "started")

        for idx, pose in enumerate(Logger.progress(plan, desc="Plan", total=len(plan)), 1):
            _log.tag("PT", f"{idx} target={tuple(round(v, 3) for v in pose[:3])}")
            ok = write_and_run_point(client, pose, config=config)
            if not ok:
                _log.tag("PT", "move failed; clearing alarm", level="warning")
                clear_alarm(client)
                clear_alarm_continue(client)
                continue

            idx_name = session.index_name()
            wx, wy, wz, wu, wv, ww = query_world(client)
            reached = (wx, wy, wz, wu, wv, ww)
            _log.tag(
                "PT",
                f"reached={tuple(round(v, 3) for v in reached[:3])} idx={idx_name}",
            )

            if streamer:
                _capture_preview(session, idx_name, reached, streamer)
            else:
                _capture_auto(session, idx_name, reached, config)
            session.next()

        graceful_release(client, hb)
        _log.tag("FINAL", "released to pendant")

    _summary(session, len(plan))

    if config.calibration.run_mode in ("grid_with_calib", "calib_only"):
        ds_path = _resolve_dataset_for_calibration(session.root, config)
        _log.tag("CALIB", f"running offline on {ds_path}")
        out_dir = run_offline(ds_path, calib_config)
        _log.tag("CALIB", f"finished -> {out_dir}")

    return session.root


def _resolve_dataset_for_calibration(root: Path, config: BorunteConfig) -> Path:
    if config.calibration.run_mode == "calib_only" and config.calibration.dataset_override:
        return config.calibration.dataset_override
    probe = root / "preview_single"
    if probe.exists():
        return probe
    return root


__all__ = ["run_capture_pipeline"]
