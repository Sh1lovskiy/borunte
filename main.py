# main.py
"""Program entry: grid/waypoints capture and optional offline calibration."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Optional, List, Tuple

from borunte.cam_rs import PreviewStreamer
from borunte.cap_session import CaptureSession
from borunte.config import (
    IP,
    PORT,
    TIMEOUT_S,
    SOCKET_KEEPALIVE,
    USE_WAYPOINTS,
    WAYPOINTS_FILE,
    WAYPOINTS_FMT,
    TOTAL_POINTS,
    WORKSPACE_M,
    TCP_DOWN_UVW,
    DEV_MAX_DEG,
    CAPTURE_ROOT_DIR,
    PREVIEW_VIEW,
    PREVIEW_DISPARITY_SHIFT,
    CAPTURE_INTERACTIVE,
    AUTO_CAPTURE_DELAY_S,
    RUN_MODE,
    CALIB_DATA_DIR,
)
from borunte.control import (
    Heartbeat,
    graceful_release,
    clear_alarm,
    clear_alarm_continue,
)
from borunte.grid import build_grid_for_count
from borunte.motion import write_and_run_point
from borunte.state import query_mode_move_alarm, query_world
from borunte.waypoints import load_waypoints
from calib.offline import run_offline as run_calibration
from utils.error_tracker import ErrorTracker, error_scope
from utils.logger import Logger

_log = Logger.get_logger()


# -------- logging helpers --------


def _log_launch_banner() -> None:
    x = WORKSPACE_M[0]
    y = WORKSPACE_M[1]
    z = WORKSPACE_M[2]
    _log.tag(
        "START",
        "borunte main starting: "
        f"RUN_MODE={RUN_MODE} CAPTURE_INTERACTIVE={CAPTURE_INTERACTIVE} "
        f"USE_WAYPOINTS={USE_WAYPOINTS} VIEW={PREVIEW_VIEW} "
        f"DISP={PREVIEW_DISPARITY_SHIFT}",
    )
    _log.tag(
        "CFG",
        f"net={IP}:{PORT} to={TIMEOUT_S}s keepalive={SOCKET_KEEPALIVE} "
        f"root={Path(CAPTURE_ROOT_DIR).resolve()}",
    )
    _log.tag(
        "WS",
        f"X=({x[0]:.1f},{x[1]:.1f}) Y=({y[0]:.1f},{y[1]:.1f}) "
        f"Z=({z[0]:.1f},{z[1]:.1f}) baseUVW="
        f"({TCP_DOWN_UVW[0]:.1f},{TCP_DOWN_UVW[1]:.1f},{TCP_DOWN_UVW[2]:.1f})",
    )


def _connect() -> socket.socket:
    _log.tag("NET", f"connecting {IP}:{PORT}")
    s = socket.create_connection((IP, PORT), timeout=TIMEOUT_S)
    if SOCKET_KEEPALIVE:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    _log.tag("NET", "connected")
    return s


def _init_state(sock: socket.socket) -> None:
    try:
        cm, mv, al = query_mode_move_alarm(sock)
    except TimeoutError:
        _log.tag("NET", "first query timeout; retry once", "warning")
        time.sleep(0.2)
        cm, mv, al = query_mode_move_alarm(sock)
    _log.tag("INIT", f"curMode={cm} isMoving={mv} curAlarm={al}")
    wx, wy, wz, wu, wv, ww = query_world(sock)
    _log.tag(
        "INIT",
        f"WORLD X={wx:.3f} Y={wy:.3f} Z={wz:.3f} " f"U={wu:.3f} V={wv:.3f} W={ww:.3f}",
    )


def _plan() -> List[List[float]]:
    if USE_WAYPOINTS:
        poses = load_waypoints(WAYPOINTS_FILE, WAYPOINTS_FMT)
        _log.tag("PLAN", f"waypoints loaded: {len(poses)} " f"from {WAYPOINTS_FILE}")
    else:
        poses = build_grid_for_count(
            WORKSPACE_M,
            total=TOTAL_POINTS,
            rx_base=TCP_DOWN_UVW[0],
            ry_base=TCP_DOWN_UVW[1],
            rz_base=TCP_DOWN_UVW[2],
            seed=42,
            dev_max_deg=DEV_MAX_DEG,
        )
        _log.tag("PLAN", f"grid built: {len(poses)} points " f"(target={TOTAL_POINTS})")
    if poses:
        a, b = poses[0], poses[-1]
        _log.tag(
            "PLAN",
            f"first={tuple(round(v, 3) for v in a[:3])} "
            f"last={tuple(round(v, 3) for v in b[:3])}",
        )
    return poses


def _save_params_once(session: CaptureSession, prof, applied: int, decim: int) -> None:
    try:
        path = session.root / "rs2_params.json"
        session.save_params_json(prof, path, int(applied), int(decim))
        _log.tag("SAVE", f"rs2_params.json -> {path.name}")
    except Exception as e:
        _log.tag("SAVE", f"rs2_params.json failed: {e}", "warning")


def _wait_user_action(streamer: PreviewStreamer) -> str:
    _log.tag("PREVIEW", "SPACE=capture, Q/ESC=skip (waiting)")
    while True:
        act = streamer.poll_action(timeout_s=0.1)
        if act in ("capture", "skip"):
            _log.tag("PREVIEW", f"user={act}")
            return act


def _capture_from_preview(
    session: CaptureSession,
    idx_name: str,
    pose6: Tuple[float, float, float, float, float, float],
    streamer: PreviewStreamer,
) -> None:
    rgb, dpt, prof = streamer.snapshot()
    if rgb is None and dpt is None:
        _log.tag("CAPTURE", "snapshot empty; skip", "warning")
        return
    if prof is not None:
        _save_params_once(session, prof, applied=0, decim=1)
    subdir = session.root / "preview_single"
    session.save_preview_snapshot(subdir, idx_name, rgb, dpt)
    session.update_pose(
        idx_name,
        dict(x=pose6[0], y=pose6[1], z=pose6[2], rx=pose6[3], ry=pose6[4], rz=pose6[5]),
    )
    _log.tag("CAPTURE", f"saved {idx_name} -> {subdir.name}")


def _capture_auto(
    session: CaptureSession,
    idx_name: str,
    pose6: Tuple[float, float, float, float, float, float],
) -> None:
    _log.tag("CAPTURE", f"auto wait {AUTO_CAPTURE_DELAY_S:.1f}s")
    time.sleep(max(0.0, float(AUTO_CAPTURE_DELAY_S)))
    subdir = session.root / "auto_single"
    session.save_preview_snapshot(subdir, idx_name, None, None)
    session.update_pose(
        idx_name,
        dict(x=pose6[0], y=pose6[1], z=pose6[2], rx=pose6[3], ry=pose6[4], rz=pose6[5]),
    )
    _log.tag("CAPTURE", f"auto record {idx_name} -> {subdir.name}")


def _summary(session: CaptureSession, total_pts: int) -> None:
    root = session.root
    n_poses = 0
    try:
        import json

        poses = json.loads((root / "poses.json").read_text("utf-8"))
        n_poses = len(poses)
    except Exception:
        pass
    _log.tag(
        "FINAL",
        f"session={root.name} dir={root} saved={n_poses}/{total_pts}",
    )


def _cleanup_preview(streamer: Optional[PreviewStreamer]) -> None:
    try:
        if streamer:
            streamer.stop()
            try:
                import cv2

                cv2.destroyWindow("Preview")
            except Exception:
                pass
    except Exception:
        pass


def _cleanup_robot(sock, hb) -> None:
    if not sock:
        return
    try:
        graceful_release(sock, hb)
    except Exception:
        pass
    try:
        from borunte.control import graceful_socket_close

        graceful_socket_close(sock)
    except Exception:
        pass


def _realsense_available() -> bool:
    try:
        import pyrealsense2 as rs  # type: ignore

        ctx = rs.context()
        devs = ctx.query_devices()
        return len(devs) > 0
    except Exception:
        return False


def _run_capture_loop(plan: List[List[float]]) -> Path:
    streamer: Optional[PreviewStreamer] = None
    hb: Optional[Heartbeat] = None
    sock: Optional[socket.socket] = None
    session = CaptureSession(CAPTURE_ROOT_DIR)

    def _cleanup() -> None:
        _cleanup_preview(streamer)
        _cleanup_robot(sock, hb)
        _log.tag("FINAL", "cleanup complete")

    ErrorTracker.register_cleanup(_cleanup)

    with error_scope():
        use_preview = bool(CAPTURE_INTERACTIVE and _realsense_available())
        if use_preview:
            _log.tag(
                "PREVIEW", f"start view={PREVIEW_VIEW} disp={PREVIEW_DISPARITY_SHIFT}"
            )
            try:
                streamer = PreviewStreamer(PREVIEW_VIEW, PREVIEW_DISPARITY_SHIFT)
                streamer.start()
                _log.tag("PREVIEW", "ready")
            except Exception as e:
                _log.tag("PREVIEW", f"disabled: {e}", "warning")
                streamer = None
                use_preview = False
        else:
            _log.tag("PREVIEW", "disabled: no RealSense detected", "warning")

        sock = _connect()
        hb = Heartbeat(sock)
        hb.start()
        _log.tag("HB", "started")
        _init_state(sock)

        for idx, pose in enumerate(
            Logger.progress(plan, desc="Plan", total=len(plan)), 1
        ):
            _log.tag("PT", f"{idx} target=" f"{tuple(round(v, 3) for v in pose[:3])}")
            ok = write_and_run_point(sock, pose)
            if not ok:
                _log.tag("PT", "move failed; clearing alarm", "warning")
                clear_alarm(sock)
                clear_alarm_continue(sock)
                continue

            idx_name = session.index_name()
            wx, wy, wz, wu, wv, ww = query_world(sock)
            reached = (wx, wy, wz, wu, wv, ww)
            _log.tag(
                "PT",
                "reached="
                f"{tuple(round(v, 3) for v in reached[:3])} "
                f"idx={idx_name}",
            )

            if use_preview and streamer is not None:
                if not streamer.is_alive():
                    _log.tag("PREVIEW", "streamer died; switching to auto", "warning")
                    use_preview = False
                else:
                    act = _wait_user_action(streamer)
                    if act == "capture":
                        _capture_from_preview(session, idx_name, reached, streamer)
                    else:
                        _log.tag("CAPTURE", "skipped by user")
            if not use_preview:
                _capture_auto(session, idx_name, reached)
            session.next()

        graceful_release(sock, hb)
        _log.tag("FINAL", "released to pendant")
    _summary(session, len(plan))
    return session.root


def _resolve_calib_dataset(hint: str | Path | None) -> Path:
    def has_imgs(d: Path) -> bool:
        return any(d.glob("*_rgb.png"))

    if hint is not None:
        d = Path(hint).expanduser().resolve()
        if has_imgs(d):
            return d
        for c in d.iterdir():
            if c.is_dir() and has_imgs(c):
                return c
        if (d / "preview_single").exists():
            return d / "preview_single"
        raise FileNotFoundError(f"Dataset not found under {d}")

    root = Path(CAPTURE_ROOT_DIR).expanduser().resolve()
    for session in sorted(root.glob("*"), reverse=True):
        if not session.is_dir():
            continue
        for probe in (session / "preview_single", session):
            if has_imgs(probe):
                return probe
    raise FileNotFoundError("No calibration dataset found")


def run() -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener("esc")

    _log_launch_banner()

    if RUN_MODE == "grid":
        plan = _plan()
        _ = _run_capture_loop(plan)
    elif RUN_MODE == "grid_with_calib":
        plan = _plan()
        root = _run_capture_loop(plan)
        ds = _resolve_calib_dataset(root)
        _log.tag("CALIB", f"running offline on {ds}")
        out = run_calibration(ds)
        _log.tag("CALIB", f"finished -> {out}")
    elif RUN_MODE == "calib_only":
        ds = _resolve_calib_dataset(CALIB_DATA_DIR)
        _log.tag("CALIB", f"running offline on {ds}")
        out = run_calibration(ds)
        _log.tag("CALIB", f"finished -> {out}")
    else:
        raise ValueError("Unknown RUN_MODE")


if __name__ == "__main__":
    run()
