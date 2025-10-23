#!/usr/bin/env python3
"""Test grid-based capture with updated motion sequence."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from borunte.cam import CaptureSession, capture_one_pair
from borunte.config import BORUNTE_CONFIG
from borunte.control.client import RobotClient
from borunte.control.controller import Heartbeat, graceful_release
from borunte.grid.generator import build_grid_for_count

_log = logger.bind(module="test_grid")


def main() -> int:
    """Run grid capture test with 5 points."""
    config = BORUNTE_CONFIG

    _log.info("=" * 80)
    _log.info("Grid Capture Test - 5 Points")
    _log.info("=" * 80)

    # Generate small grid
    poses = build_grid_for_count(
        ws=config.motion.workspace_m,
        total=3,  # Just 3 points for quick testing
    )

    _log.info(f"Grid generated: {len(poses)} points")
    _log.info(f"First: {tuple(round(v, 1) for v in poses[0][:3])}")
    _log.info(f"Last: {tuple(round(v, 1) for v in poses[-1][:3])}")

    # Connect to robot
    client = RobotClient(
        host=config.robot.host,
        port=config.robot.port,
        timeout=config.robot.timeout_s,
    )

    if not client.connect():
        _log.error("Failed to connect to robot")
        return 1

    _log.info("Connected to robot")

    # Create capture session
    session = CaptureSession(config=config)
    _log.info(f"Session dir: {session.root}")

    # Start heartbeat
    hb = Heartbeat(client, period_s=5.0, reassert_s=9.0)
    hb.start()
    _log.info("Heartbeat started")

    try:
        captured = 0

        for idx, pose in enumerate(poses, 1):
            x, y, z, u, v, w = map(float, pose[:6])

            _log.info(f"[{idx}/{len(poses)}] Moving to ({x:.1f}, {y:.1f}, {z:.1f})...")

            # Move to pose using verified sequence with retry
            ok = False
            for attempt in range(3):
                ok = client.move_to_pose(
                    x,
                    y,
                    z,
                    u,
                    v,
                    w,
                    verify=True,
                    speed_percent=85.0,
                    wait_complete=True,
                    timeout_s=60.0,
                )
                if ok:
                    break
                if attempt < 2:
                    _log.warning(f"Retry {attempt + 1}/2 after 2s...")
                    time.sleep(2.0)

            if not ok:
                _log.error(f"Point {idx} failed after 3 attempts")
                continue

            # Wait for stability
            time.sleep(0.5)

            # Get actual position
            world_pose = client.get_world_pose()
            if not world_pose:
                _log.warning("Could not query world pose")
                session.next()
                continue

            wx, wy, wz, wu, wv, ww = world_pose
            _log.info(f"Reached: ({wx:.1f}, {wy:.1f}, {wz:.1f})")

            idx_name = f"{idx:03d}"

            # Capture RGB/depth with retry logic
            capture_ok = False
            for cap_attempt in range(3):
                try:
                    _log.info(f"Capturing frame {idx_name} (attempt {cap_attempt + 1}/3)...")
                    rgb, depth, meta, profile = capture_one_pair(0, config=config)

                    # Save RGB/depth pair to frames/ subdirectory
                    session.save_rgb_depth(session.frames_dir, idx_name, rgb, depth)
                    _log.info(f"Saved frames/{idx_name}_rgb.png and {idx_name}_depth.npy")

                    # Save camera intrinsics once on first successful capture
                    intrinsics_path = session.root / "intrinsics.json"
                    if not intrinsics_path.exists():
                        session.save_params_json(
                            profile,
                            intrinsics_path,
                            meta.get("applied_disparity_shift", 0),
                            meta.get("decimation", 1),
                        )
                        _log.info("Saved camera intrinsics to intrinsics.json")

                    capture_ok = True
                    break

                except Exception as exc:
                    _log.warning(f"Capture attempt {cap_attempt + 1}/3 failed: {exc}")
                    if cap_attempt < 2:
                        time.sleep(1.0)

            if not capture_ok:
                _log.error(f"Capture failed for frame {idx_name} after 3 attempts")
                continue

            # Save TCP pose to JSON (TODO: should save to {idx:03d}_tcp.json file)
            session.update_pose(
                idx_name,
                {
                    "x": wx,
                    "y": wy,
                    "z": wz,
                    "rx": wu,
                    "ry": wv,
                    "rz": ww,
                },
            )
            captured += 1
            _log.info(f"Point {idx} complete: {idx_name}_rgb.png, {idx_name}_depth.npy saved")

            session.next()

        _log.info("=" * 80)
        _log.info(f"Capture complete: {captured}/{len(poses)} points")
        _log.info(f"Session saved to: {session.root}")
        _log.info("=" * 80)

    finally:
        _log.info("Cleaning up...")
        graceful_release(client, hb)
        client.disconnect()
        _log.info("Done")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
