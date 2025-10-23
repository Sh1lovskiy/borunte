#!/usr/bin/env python3
"""Move robot TCP 10mm down along Z-axis with full verification.

This module demonstrates safe robot motion control by:
1. Querying current world coordinates from the robot
2. Computing new target with Z decreased by 10mm
3. Writing target coordinates to registers 800-805 (scaled ×1000)
4. Executing motion via actionSingleCycle + startButton
5. Maintaining heartbeat during motion to prevent alarms
6. Verifying final position matches expected Z delta (~-10mm)
7. Gracefully shutting down on errors or interrupts
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Any

from loguru import logger

from borunte.config import ROBOT_ANG_SCALE, ROBOT_POS_SCALE
from borunte.control.client import RobotClient
from borunte.control.controller import Heartbeat, graceful_release

# ────────────── MODULE LOGGER ──────────────

_log = logger.bind(module="borunte.move_down_10mm")


# ────────────── GLOBAL STATE ──────────────

_shutdown_requested = False


# ────────────── SIGNAL HANDLERS ──────────────


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C and other termination signals gracefully."""
    global _shutdown_requested

    signal_name = signal.Signals(signum).name
    _log.warning(f"received signal {signal_name}, shutting down gracefully...")
    _shutdown_requested = True


def _install_signal_handlers() -> None:
    """Install handlers for graceful shutdown on interrupt."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ────────────── COORDINATE QUERIES ──────────────


def query_world_coordinates(client: RobotClient) -> tuple[float, ...] | None:
    """Query current world coordinates X, Y, Z, U, V, W from robot.

    Returns:
        Tuple of 6 floats (X, Y, Z, U, V, W) in mm and degrees, or None on failure
    """
    _log.info("querying current world coordinates...")

    try:
        pose = client.get_world_pose()
        if pose and len(pose) >= 6:
            x, y, z, u, v, w = pose[:6]
            _log.info(f"current world: X={x:.3f} Y={y:.3f} Z={z:.3f} U={u:.3f} V={v:.3f} W={w:.3f}")
            return pose
        else:
            _log.error("failed to query world coordinates")
            return None
    except Exception as exc:
        _log.error(f"exception during world coordinate query: {exc!r}")
        return None


# ────────────── TARGET CALCULATION ──────────────


def compute_target_down_10mm(
    current: tuple[float, ...],
) -> tuple[float, float, float, float, float, float]:
    """Compute target pose by moving 10mm down along Z-axis.

    Args:
        current: Current pose (X, Y, Z, U, V, W)

    Returns:
        Target pose with Z decreased by 10mm
    """
    x, y, z, u, v, w = current[:6]
    target_z = z - 10.0

    _log.info(f"target calculation: Z {z:.3f} → {target_z:.3f} (delta=-10.0mm)")

    return (x, y, target_z, u, v, w)


# ────────────── REGISTER WRITE ──────────────


def write_target_to_registers(
    client: RobotClient,
    x: float,
    y: float,
    z: float,
    u: float,
    v: float,
    w: float,
) -> bool:
    """Write target pose to robot registers 800-805 with proper scaling.

    Position coordinates (X,Y,Z) are scaled ×1000 (mm → µm).
    Angle coordinates (U,V,W) are scaled ×1000 (deg → millideg).

    Args:
        client: Robot client instance
        x, y, z: Position coordinates in mm
        u, v, w: Angle coordinates in degrees

    Returns:
        True if write successful and verified, False otherwise
    """
    _log.info(f"writing target to registers 800-805...")
    _log.info(f"  X={x:.3f} Y={y:.3f} Z={z:.3f} U={u:.3f} V={v:.3f} W={w:.3f}")

    # Scale coordinates
    scaled_x = int(round(x * ROBOT_POS_SCALE))
    scaled_y = int(round(y * ROBOT_POS_SCALE))
    scaled_z = int(round(z * ROBOT_POS_SCALE))
    scaled_u = int(round(u * ROBOT_ANG_SCALE))
    scaled_v = int(round(v * ROBOT_ANG_SCALE))
    scaled_w = int(round(w * ROBOT_ANG_SCALE))

    _log.info(
        f"  scaled: [{scaled_x}, {scaled_y}, {scaled_z}, {scaled_u}, {scaled_v}, {scaled_w}]"
    )

    # Write to registers using rewriteDataList command
    try:
        ok, msg, _ = client.command(
            "rewriteDataList",
            "800",  # Start address
            "6",  # Count
            "0",  # Flags (0 = not savable)
            str(scaled_x),
            str(scaled_y),
            str(scaled_z),
            str(scaled_u),
            str(scaled_v),
            str(scaled_w),
            timeout=3.0,
        )

        if not ok:
            _log.error(f"register write failed: {msg}")
            return False

        _log.info("register write successful")

        # Verify write
        time.sleep(0.1)
        if not client.verify_pose_write(x, y, z, u, v, w):
            _log.error("register verification failed")
            return False

        _log.info("register verification successful")
        return True

    except Exception as exc:
        _log.error(f"exception during register write: {exc!r}")
        return False


# ────────────── MOTION EXECUTION ──────────────


def execute_single_cycle_motion(client: RobotClient) -> bool:
    """Execute motion using Single Cycle mode + Start Button.

    Sequence:
    1. Stop any ongoing motion (stopButton)
    2. Enter Single Loop mode (actionSingleCycle)
    3. Set global speed to 45% (modifyGSPD)
    4. Trigger start (startButton)

    Args:
        client: Robot client instance

    Returns:
        True if commands successful, False otherwise
    """
    try:
        # Step 0: Stop button to reset state
        _log.info("issuing stop button to reset state...")
        ok, msg, _ = client.command("stopButton", timeout=3.0)
        if not ok:
            _log.warning(f"stopButton failed: {msg} (might be OK if already stopped)")

        time.sleep(0.5)

        # Step 1: actionSingleCycle (with retry)
        _log.info("entering Single Loop mode...")

        for attempt in range(3):
            ok, msg, _ = client.command("actionSingleCycle", timeout=3.0)
            if ok:
                break

            _log.warning(f"actionSingleCycle attempt {attempt + 1}/3 failed: {msg}")

            # Try stop button again before retry
            if attempt < 2:
                _log.info("retrying after stop button...")
                client.command("stopButton", timeout=2.0)
                time.sleep(0.5)
        else:
            _log.error("actionSingleCycle failed after 3 attempts")
            return False

        _log.info("Single Loop mode activated")
        time.sleep(0.2)

        # Step 2: Set speed to 45%
        _log.info("setting global speed to 45%...")
        ok, msg, _ = client.command("modifyGSPD", "45", timeout=3.0)
        if not ok:
            _log.error(f"modifyGSPD failed: {msg}")
            return False

        _log.info("speed set to 45%")
        time.sleep(0.2)

        # Step 3: startButton
        _log.info("triggering start button...")
        ok, msg, _ = client.command("startButton", timeout=3.0)
        if not ok:
            _log.error(f"startButton failed: {msg}")
            return False

        _log.info("motion triggered successfully")
        return True

    except Exception as exc:
        _log.error(f"exception during motion execution: {exc!r}")
        return False


# ────────────── MOTION MONITORING ──────────────


def wait_for_motion_start(client: RobotClient, timeout_s: float = 10.0) -> bool:
    """Wait for robot to start moving.

    Args:
        client: Robot client instance
        timeout_s: Maximum time to wait for motion start

    Returns:
        True if motion started, False on timeout or alarm
    """
    _log.info(f"waiting for motion to start (timeout={timeout_s}s)...")

    start_time = time.time()

    while time.time() - start_time < timeout_s:
        if _shutdown_requested:
            _log.warning("shutdown requested during motion start")
            return False

        try:
            # Check alarm
            alarm = client.get_alarm()
            if alarm and alarm != "0":
                _log.error(f"alarm detected: {alarm}")
                return False

            # Check if moving
            moving = client.is_moving()
            if moving:
                _log.info("robot started moving")
                return True

            time.sleep(0.2)

        except Exception as exc:
            _log.error(f"exception while waiting for motion start: {exc!r}")
            return False

    _log.error(f"motion did not start within {timeout_s}s")
    return False


def wait_for_motion_complete(
    client: RobotClient,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.3,
) -> bool:
    """Wait for robot to complete motion and stop.

    Args:
        client: Robot client instance
        timeout_s: Maximum time to wait for motion completion
        poll_interval_s: Interval between status checks

    Returns:
        True if motion completed successfully, False on timeout or alarm
    """
    _log.info(f"waiting for motion to complete (timeout={timeout_s}s)...")

    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < timeout_s:
        if _shutdown_requested:
            _log.warning("shutdown requested during motion")
            return False

        try:
            # Check alarm
            alarm = client.get_alarm()
            if alarm and alarm != "0":
                _log.error(f"alarm detected during motion: {alarm}")
                return False

            # Check if still moving
            moving = client.is_moving()

            # Log status periodically
            if time.time() - last_log_time >= 1.0:
                pose = client.get_world_pose()
                if pose:
                    x, y, z, u, v, w = pose[:6]
                    _log.info(
                        f"moving={moving} alarm={alarm} "
                        f"pos=({x:.1f}, {y:.1f}, {z:.1f}) "
                        f"ang=({u:.1f}, {v:.1f}, {w:.1f})"
                    )
                last_log_time = time.time()

            # Check if stopped
            if not moving:
                _log.info("robot stopped moving")
                return True

            time.sleep(poll_interval_s)

        except Exception as exc:
            _log.error(f"exception during motion monitoring: {exc!r}")
            return False

    _log.error(f"motion did not complete within {timeout_s}s")
    return False


# ────────────── VERIFICATION ──────────────


def verify_z_movement(
    initial: tuple[float, ...],
    final: tuple[float, ...],
    expected_delta: float = -10.0,
    tolerance: float = 2.0,
) -> bool:
    """Verify that Z coordinate changed by expected amount.

    Args:
        initial: Initial pose (X, Y, Z, U, V, W)
        final: Final pose (X, Y, Z, U, V, W)
        expected_delta: Expected Z change in mm (negative = down)
        tolerance: Acceptable deviation in mm

    Returns:
        True if Z delta within tolerance, False otherwise
    """
    initial_z = initial[2]
    final_z = final[2]
    actual_delta = final_z - initial_z
    deviation = abs(actual_delta - expected_delta)

    _log.info(f"Z verification:")
    _log.info(f"  initial Z: {initial_z:.3f} mm")
    _log.info(f"  final Z:   {final_z:.3f} mm")
    _log.info(f"  delta:     {actual_delta:.3f} mm (expected: {expected_delta:.3f} mm)")
    _log.info(f"  deviation: {deviation:.3f} mm (tolerance: {tolerance:.3f} mm)")

    success = deviation <= tolerance

    if success:
        _log.info("✓ Z-axis movement verified successfully")
    else:
        _log.error(f"✗ Z-axis movement verification failed (deviation too large)")

    return success


# ────────────── MAIN EXECUTION ──────────────


def move_down_10mm(
    host: str = "192.168.4.4",
    port: int = 9760,
    timeout: float = 5.0,
) -> bool:
    """Execute complete 10mm downward motion with verification.

    This is the main entry point that orchestrates the full sequence:
    1. Connect to robot
    2. Query current position
    3. Calculate target (Z - 10mm)
    4. Write target to registers
    5. Execute motion with heartbeat
    6. Verify final position
    7. Clean shutdown with graceful_release

    Args:
        host: Robot controller IP address
        port: Robot controller TCP port
        timeout: Network timeout in seconds

    Returns:
        True if motion completed and verified successfully, False otherwise
    """
    global _shutdown_requested

    _install_signal_handlers()
    _log.info("=" * 80)
    _log.info("BORUNTE ROBOT: Move TCP 10mm Down Along Z-Axis")
    _log.info("=" * 80)

    client = RobotClient(host=host, port=port, timeout=timeout)
    hb: Heartbeat | None = None

    try:
        # ────────────── CONNECTION ──────────────

        _log.info(f"connecting to robot at {host}:{port}...")

        if not client.connect():
            _log.error("connection failed")
            return False

        _log.info("connection established")
        time.sleep(0.5)

        # ────────────── QUERY INITIAL POSITION ──────────────

        initial_pose = query_world_coordinates(client)
        if not initial_pose:
            _log.error("failed to query initial position")
            return False

        # ────────────── COMPUTE TARGET ──────────────

        target_pose = compute_target_down_10mm(initial_pose)
        x, y, z, u, v, w = target_pose

        # ────────────── WRITE TARGET TO REGISTERS ──────────────

        if not write_target_to_registers(client, x, y, z, u, v, w):
            _log.error("failed to write target to registers")
            return False

        # ────────────── START HEARTBEAT ──────────────

        hb = Heartbeat(client, period_s=5.0, reassert_s=9.0)
        hb.start()

        # ────────────── EXECUTE MOTION ──────────────

        if not execute_single_cycle_motion(client):
            _log.error("failed to execute motion")
            return False

        # ────────────── WAIT FOR MOTION START ──────────────

        if not wait_for_motion_start(client, timeout_s=10.0):
            _log.error("motion did not start")
            return False

        # ────────────── WAIT FOR MOTION COMPLETE ──────────────

        if not wait_for_motion_complete(client, timeout_s=60.0):
            _log.error("motion did not complete")
            return False

        # ────────────── QUERY FINAL POSITION ──────────────

        time.sleep(0.5)
        final_pose = query_world_coordinates(client)
        if not final_pose:
            _log.error("failed to query final position")
            return False

        # ────────────── VERIFY Z MOVEMENT ──────────────

        if not verify_z_movement(initial_pose, final_pose, expected_delta=-10.0, tolerance=2.0):
            _log.error("Z-axis movement verification failed")
            return False

        # ────────────── SUCCESS ──────────────

        _log.info("=" * 80)
        _log.info("✓ MOTION COMPLETED SUCCESSFULLY")
        _log.info("=" * 80)

        return True

    except KeyboardInterrupt:
        _log.warning("interrupted by user (Ctrl+C)")
        return False

    except Exception as exc:
        _log.error(f"unexpected exception: {exc!r}")
        import traceback
        _log.error(traceback.format_exc())
        return False

    finally:
        # ────────────── CLEANUP ──────────────

        _log.info("performing cleanup...")

        # Use graceful_release from controller.py - it properly:
        # 1. Stops heartbeat thread
        # 2. Sends actionPause, stopButton, actionStop
        # 3. Sends setRemoteMode(0), connectHost(0)
        # 4. Sends endHeartbreak, exitRemoteMonitor, etc.
        # 5. Properly releases robot control without alarms
        graceful_release(client, hb)

        if client.is_connected():
            _log.info("disconnecting TCP socket...")
            client.disconnect()

        _log.info("cleanup complete")


# ────────────── CLI ENTRY POINT ──────────────


def main() -> int:
    """Command-line entry point.

    Returns:
        0 on success, 1 on failure
    """
    success = move_down_10mm()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
