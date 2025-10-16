# borunte/capture.py
"""Grid capture workflow combining robot control and sensing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from borunte.client import RobotClient
from borunte.config import BorunteConfig, DEFAULT_CONFIG
from borunte.grid import build_grid
from config import get_settings
from utils.error_tracker import ErrorTracker
from utils.io import atomic_write_json, ensure_directory
from utils.logger import get_logger
from utils.progress import track
from utils.rs import RealSenseController

LOGGER = get_logger(__name__)


def _pose_payload(position: Sequence[float]) -> dict[str, Iterable[float]]:
    return {
        "position": [float(value) for value in position],
        "orientation_rpy": [0.0, 0.0, 0.0],
    }


def _prepare_session_directory(session_root: Path, name: str | None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    folder = session_root / (name or f"capture_{timestamp}")
    ensure_directory(folder)
    return folder


def run_grid_capture(
    *,
    config: BorunteConfig | None = None,
    columns: int = 3,
    rows: int = 3,
    session_name: str | None = None,
) -> Path:
    cfg = config or DEFAULT_CONFIG
    settings = get_settings()
    session_dir = _prepare_session_directory(settings.paths.captures_root, session_name)
    tracker = ErrorTracker(context="borunte.capture")
    controller = RealSenseController(output_dir=session_dir)
    robot = RobotClient(config=cfg, tracker=tracker)
    LOGGER.info("Starting grid capture into {}", session_dir)
    intrinsics_path = session_dir / settings.calibration.output_intrinsics
    extrinsics_path = session_dir / settings.calibration.output_extrinsics
    try:
        controller.start()
        robot.connect()
        robot.set_speed(cfg.capture_speed)
        plan = build_grid(config=cfg, columns=columns, rows=rows)
        intrinsics = {
            "resolution": [settings.realsense.width, settings.realsense.height],
            "fps": settings.realsense.fps,
            "exposure": settings.realsense.exposure,
            "gain": settings.realsense.gain,
        }
        extrinsics = {
            "reference_pose": _pose_payload(plan.origin),
            "frame": "world",
        }
        atomic_write_json(intrinsics_path, intrinsics)
        atomic_write_json(extrinsics_path, extrinsics)
        LOGGER.info("Logged session intrinsics to {}", intrinsics_path)
        LOGGER.info("Logged session extrinsics to {}", extrinsics_path)
        for index, point in enumerate(track(plan.points, description="Grid capture")):
            robot.write_pose_target([*point.tolist(), 0.0, 0.0, 0.0])
            robot.start_step()
            pose_path = session_dir / f"pose_{index:03d}.json"
            atomic_write_json(pose_path, _pose_payload(point))
            controller.snapshot()
            LOGGER.debug("Captured point {} -> {}", index, pose_path)
    except Exception as exc:  # noqa: BLE001
        tracker.record("runtime", str(exc))
        raise
    finally:
        controller.stop()
        robot.graceful_handover()
        tracker.summary()
    LOGGER.info("Grid capture complete for session {}", session_dir.name)
    return session_dir


__all__ = ["run_grid_capture"]
