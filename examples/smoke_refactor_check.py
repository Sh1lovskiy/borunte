# examples/smoke_refactor_check.py
"""Quick smoke check for refactored modules."""

from __future__ import annotations


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from borunte import BORUNTE_CONFIG
from borunte.cap_session import CaptureSession
from borunte.grid import build_grid_for_count
from borunte.wire import RobotClient


def main() -> None:
    _ = BORUNTE_CONFIG.capture_root
    _ = build_grid_for_count(config=BORUNTE_CONFIG)
    CaptureSession(config=BORUNTE_CONFIG)
    RobotClient(BORUNTE_CONFIG)
    print("Smoke check imports OK")


if __name__ == "__main__":
    main()
