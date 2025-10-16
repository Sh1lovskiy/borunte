# borunte/run_capture.py
"""Executable shim for ``python -m borunte.run_capture``."""

from __future__ import annotations

from borunte import run_grid_capture
from borunte.config import DEFAULT_CONFIG


def main() -> None:
    run_grid_capture(config=DEFAULT_CONFIG)


if __name__ == "__main__":  # pragma: no cover
    main()
