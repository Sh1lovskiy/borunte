# calib/run_detect.py
"""Shim for ``python -m calib.run_detect``."""

from __future__ import annotations

from pathlib import Path

from calib import run_detect


def main() -> None:
    run_detect([Path("sample.png")])


if __name__ == "__main__":  # pragma: no cover
    main()
