# borunte/runner.py
"""Capture pipeline orchestrating robot motion and RealSense capture."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

from utils.error_tracker import ErrorTracker, error_scope
from utils.logger import get_logger, iter_progress

from .client import RobotClient
from .cap_session import CaptureSession
from .cam_rs import PreviewStreamer, capture_one_pair, HAS_CV2_GUI
from .config import BORUNTE_CONFIG, BorunteConfig
from .control import Heartbeat, clear_alarm, clear_alarm_continue
from .control import graceful_release
from .grid import build_grid_for_count
from .waypoints import load_default_waypoints

_log = get_logger("borunte.runner")


def _log_launch_banner(config: BorunteConfig) -> None:
    ws = config.limits.workspace_m
    tcp = config.limits.tcp_down_uvw
    net = config.net
    _log.tag(
        "START",
        f"capture starting "
        f"mode={config.calib.run_mode} "
        f"interactive={config.capture_mode.interactive} "
        f"waypoints={config.waypoints.use_waypoints} "
        f"view={config.preview.view}",
    )
    _log.tag(
        "CFG",
        f"net={net.host}:{net.port} timeout={net.timeout_s}s "
        f"keepalive={net.keepalive} root={config.paths.captures_root}",
    )
    _log.tag(
        "WS",
        f"X=({ws[0][0]:.1f},{ws[0][1]:.1f}) "
        f"Y=({ws[1][0]:.1f},{ws[1][1]:.1f}) "
        f"Z=({ws[2][0]:.1f},{ws[2][1]:.1f}) "
        f"baseUVW=({tcp[0]:.1f},{tcp[1]:.1f},{tcp[2]:.1f})",
    )


def _connect_client(config: BorunteConfig) -> RobotClient:
    client = RobotClient(
        host=config.net.host,
        port=config.net.port,
        timeout=config.net.timeout_s,
    )
    if not client.connect():
        raise RuntimeError(f"failed to connect to {config.net.host}:{config.net.port}")
    return client


def _setup_robot_mode(client: RobotClient) -> bool:
    """
    Prepare controller for remote 'Single Cycle' execution.
    We DON'T require hard Stop anymore. We:
      - try to clear any pending state by 'stopButton'
      - assert Single Cycle via 'actionSingleCycle'
      - accept modes Auto/Stop/Single Loop as 'ready'
    Note: Each move uses RobotClient.move_to_pose(), which re-asserts
          Single Cycle and presses START for the step.
    """
    _log.tag("SETUP", "preparing for Single Cycle control...")

    # Мягко сбросим состояние (не критично, ошибки не фатальны)
    ok, msg, _ = client.command("stopButton", timeout=3.0)
    if ok:
        _log.tag("SETUP", "stop button pressed")
    else:
        _log.tag("SETUP", f"stop button: {msg}", level="warning")

    # Попробуем включить Single Cycle несколько раз (контроллеры иногда «сонные»)
    names = {
        "0": "None",
        "1": "Manual",
        "2": "Automatic",
        "3": "Stop",
        "7": "Auto-running",
        "8": "Step-by-Step",
        "9": "Single Loop",
    }
    for i in range(1, 4):
        ok, msg, _ = client.command("actionSingleCycle", timeout=3.0)
        if not ok:
            _log.tag("SETUP", f"actionSingleCycle: {msg}", level="warning")
        time.sleep(0.3)

        mode = client.get_mode() or ""
        _log.tag("SETUP", f"robot mode: {names.get(mode, mode)}")
        # Принимаем Stop/Auto/Single Loop как «готово» — дальше каждый шаг сам жмет START
        if mode in ("2", "3", "9"):
            _log.tag("SETUP", "remote ready (Single Cycle will be asserted per step)")
            return True

        _log.tag("SETUP", f"not ready yet; retry {i}/3", level="warning")

    _log.tag(
        "SETUP", "proceeding: Single Cycle will be asserted per step", level="warning"
    )
    return True


def _plan(config: BorunteConfig) -> List[List[float]]:
    if config.waypoints.use_waypoints:
        poses = load_default_waypoints(config)
        _log.tag(
            "PLAN",
            f"waypoints loaded: {len(poses)} from {config.waypoints.file}",
        )
    else:
        poses = build_grid_for_count(
            ws=config.limits.workspace_m,
            total=config.limits.total_points,
        )
        _log.tag(
            "PLAN",
            f"grid built: {len(poses)} points " f"target={config.limits.total_points}",
        )
    if poses:
        first, last = poses[0], poses[-1]
        _log.tag(
            "PLAN",
            f"first={tuple(round(v, 3) for v in first[:3])} "
            f"last={tuple(round(v, 3) for v in last[:3])}",
        )
    return poses


def _cleanup_preview(streamer: Optional[PreviewStreamer]) -> None:
    if streamer:
        try:
            streamer.stop()
        except Exception as exc:
            ErrorTracker().record("preview_cleanup", str(exc))


def _cleanup_robot(client: Optional[RobotClient], hb: Optional[Heartbeat]) -> None:
    if not client:
        return
    try:
        graceful_release(client, hb)
    except Exception as exc:
        ErrorTracker().record("robot_release", str(exc))
    try:
        client.disconnect()
    except Exception as exc:
        ErrorTracker().record("robot_close", str(exc))


def _prepare_preview(config: BorunteConfig) -> Optional[PreviewStreamer]:
    if not config.capture_mode.interactive or not HAS_CV2_GUI:
        if config.capture_mode.interactive and not HAS_CV2_GUI:
            _log.tag(
                "PREVIEW",
                "interactive disabled: no GUI (opencv-python-headless)",
                level="warning",
            )
        return None

    try:
        streamer = PreviewStreamer(config=config)
        streamer.start()
        _log.tag("PREVIEW", "ready")
        return streamer
    except Exception as exc:
        ErrorTracker().record("preview_start", str(exc))
        _log.tag("PREVIEW", f"disabled: {exc}")
        return None


def _capture_preview(
    session: CaptureSession,
    idx_name: str,
    pose: Sequence[float],
    streamer: PreviewStreamer,
) -> bool:
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
            session.root / "preview_single", idx_name, rgb, depth
        )
        session.update_pose(
            idx_name,
            dict(
                x=pose[0],
                y=pose[1],
                z=pose[2],
                rx=pose[3],
                ry=pose[4],
                rz=pose[5],
            ),
        )
        _log.tag("CAPTURE", f"saved {idx_name} -> preview_single")
        return True
    else:
        _log.tag("CAPTURE", "skipped by user")
        return False


def _capture_auto(
    session: CaptureSession,
    idx_name: str,
    pose: Sequence[float],
    config: BorunteConfig,
) -> bool:
    if config.capture_mode.auto_delay_s > 0:
        time.sleep(config.capture_mode.auto_delay_s)

    sweep = config.profile.disparity_sweep or ((config.preview.disparity_shift or 0),)
    sweep = tuple(v for v in sweep if v is not None)
    sweep_dir = session.root / "auto_single"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    params_written = False
    frames_written = 0
    for value in sweep:
        rgb, depth, meta, profile = capture_one_pair(value, config=config)
        session.save_rgb_depth(sweep_dir, f"{idx_name}_{value}", rgb, depth)
        frames_written += 1
        if not params_written:
            session.save_params_json(
                profile,
                session.root / "rs2_params.json",
                meta.get("applied_disparity_shift", value),
                meta.get("decimation", 1),
            )
            params_written = True
    session.update_pose(
        idx_name,
        dict(
            x=pose[0],
            y=pose[1],
            z=pose[2],
            rx=pose[3],
            ry=pose[4],
            rz=pose[5],
        ),
    )
    _log.tag(
        "CAPTURE",
        f"auto record {idx_name} -> {sweep_dir.name} " f"({frames_written} files)",
    )
    return frames_written > 0


def _summary(session: CaptureSession, captured_count: int) -> None:
    root = session.root
    _log.tag(
        "FINAL",
        f"session={root.name} dir={root} captured={captured_count}",
    )
    if captured_count == 0:
        try:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
            _log.tag("FINAL", f"no captures -> deleted empty folder {root}")
        except Exception as exc:
            ErrorTracker().record("cleanup", str(exc))


def run_capture_pipeline(config: BorunteConfig = BORUNTE_CONFIG) -> Path:
    _log_launch_banner(config)
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener("esc")

    plan = _plan(config)
    session = CaptureSession(config=config)
    streamer: Optional[PreviewStreamer] = None
    client: Optional[RobotClient] = None
    hb: Optional[Heartbeat] = None
    captured_count = 0

    def _cleanup() -> None:
        _cleanup_preview(streamer)
        _cleanup_robot(client, hb)
        _log.tag("FINAL", "cleanup complete")

    ErrorTracker.register_cleanup(_cleanup)

    with error_scope("run_capture_pipeline"):
        if config.capture_mode.interactive and HAS_CV2_GUI:
            streamer = _prepare_preview(config)
        else:
            if config.capture_mode.interactive:
                _log.tag("MODE", "auto forced (no GUI)", level="warning")

        client = _connect_client(config)

        if not _setup_robot_mode(client):
            _log.tag("SETUP", "robot not ready - aborting", level="error")
            return session.root

        hb = Heartbeat(client, period_s=config.net.heartbeat_period_s)
        hb.start()

        for idx, pose in enumerate(
            iter_progress(plan, description="Plan", total=len(plan)), 1
        ):
            x, y, z, u, v, w = map(float, pose[:6])
            _log.tag(
                "PT",
                f"{idx} target=({x:.1f}, {y:.1f}, {z:.1f})",
            )

            ok = client.move_to_pose(x, y, z, u, v, w, verify=True)
            if not ok:
                _log.tag("PT", "move failed; clearing alarm")
                clear_alarm(client)
                clear_alarm_continue(client)
                continue

            time.sleep(0.5)

            idx_name = session.index_name()
            world_pose = client.get_world_pose()
            if world_pose:
                wx, wy, wz, wu, wv, ww = world_pose
                reached = (wx, wy, wz, wu, wv, ww)
                _log.tag(
                    "PT",
                    f"reached=({wx:.1f}, {wy:.1f}, {wz:.1f}) " f"idx={idx_name}",
                )
            else:
                _log.tag("PT", "failed to query world pose", level="warning")
                reached = pose

            did = (
                _capture_preview(session, idx_name, reached, streamer)
                if streamer
                else _capture_auto(session, idx_name, reached, config)
            )
            if did:
                captured_count += 1

            session.next()

        graceful_release(client, hb)
        _log.tag("FINAL", "released to pendant")

    _summary(session, captured_count)
    return session.root


if __name__ == "__main__":
    run_capture_pipeline()
