# main.py
"""Entry point shim for running the capture pipeline."""

from __future__ import annotations

from borunte import BORUNTE_CONFIG
from borunte.runner import run_capture_pipeline


def run() -> None:
    run_capture_pipeline(BORUNTE_CONFIG)


if __name__ == "__main__":
    run()
