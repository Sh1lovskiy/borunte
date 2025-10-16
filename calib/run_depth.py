# calib/run_depth.py
"""Shim for ``python -m calib.run_depth``."""

from __future__ import annotations

from pathlib import Path

from calib import run_depth_align


def main() -> None:
    run_depth_align(Path("calibration/handeye.json"), [Path("depth/frame.png")])


if __name__ == "__main__":  # pragma: no cover
    main()
