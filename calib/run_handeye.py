# calib/run_handeye.py
"""Shim for ``python -m calib.run_handeye``."""

from __future__ import annotations

from pathlib import Path

from calib import run_handeye


def main() -> None:
    run_handeye(Path("calibration/pnp.json"), [])


if __name__ == "__main__":  # pragma: no cover
    main()
