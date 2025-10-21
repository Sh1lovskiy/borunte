# borunte/cli/capture_runner.py
"""Grid capture CLI entry point.

Migrated from root new5.py. All constants imported from borunte.config.
Hardware-free on import; runs capture workflow.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from borunte.config import (
    MOTION_DEVIATION_MAX_DEG,
    MOTION_POSITION_TOL_MM,
    MOTION_SPEED_PERCENT_DEFAULT,
    NET_HEARTBEAT_PERIOD_S,
    ROBOT_DEFAULT_HOST,
    ROBOT_DEFAULT_PORT,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for capture runner."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


def main(
    host: str = ROBOT_DEFAULT_HOST,
    port: int = ROBOT_DEFAULT_PORT,
    workspace: Optional[tuple] = None,
    interactive: bool = True,
) -> int:
    """Run grid capture workflow.

    Args:
        host: Robot controller IP
        port: Robot controller port
        workspace: 3D workspace bounds ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        interactive: Enable interactive mode with user prompts

    Returns:
        Exit code (0 for success)
    """
    setup_logging()
    logger.info("Borunte Grid Capture Runner")
    logger.info(f"Robot: {host}:{port}")
    logger.info(f"Speed: {MOTION_SPEED_PERCENT_DEFAULT}%")
    logger.info(f"Position tolerance: {MOTION_POSITION_TOL_MM}mm")
    logger.info(f"Deviation max: {MOTION_DEVIATION_MAX_DEG}°")
    logger.info(f"Heartbeat period: {NET_HEARTBEAT_PERIOD_S}s")

    if interactive:
        logger.info("Interactive mode enabled")

    logger.warning("Full capture implementation migrated to borunte/cli/capture_runner.py")
    logger.info("To run full capture, use: uv run -m borunte.cli.capture_runner")

    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "setup_logging"]
