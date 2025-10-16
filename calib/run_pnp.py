# calib/run_pnp.py
"""Shim for ``python -m calib.run_pnp``."""

from __future__ import annotations

from pathlib import Path

from calib import run_pnp


def main() -> None:
    run_pnp(Path("calibration/detections.json"))


if __name__ == "__main__":  # pragma: no cover
    main()
