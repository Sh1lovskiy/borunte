# vision/analysis/visualize_runner.py
"""Entry point for quick visualization."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from borunte.utils.logger import get_logger
from borunte.vision.config import VisionConfig
from borunte.vision.io.series import load_series
from borunte.vision.processing.depth import depth_to_cloud
from borunte.vision.viz.viewer import Viewer

logger = get_logger(__name__)


def _as_sequence(paths: Sequence[Path | str] | Path | str) -> Sequence[Path | str]:
    if isinstance(paths, (str, Path)):
        return [paths]
    return paths


def run_visualize(
    paths: Sequence[Path | str] | Path | str, *, config: VisionConfig | None = None
) -> None:
    """Visualize RGB-D frames as point clouds."""
    config = config or VisionConfig()
    frames = load_series(_as_sequence(paths))
    logger.info(f"Loaded {len(frames)} frames for visualization")
    viewer = Viewer()
    for frame in frames:
        cloud = depth_to_cloud(frame.depth, frame.color, config)
        viewer.show_cloud(cloud)
    viewer.close()


def main() -> None:
    """Module entry point for ``python -m vision.analysis.visualize_runner``."""
    if len(sys.argv) < 2:
        raise SystemExit("Please provide at least one input path")
    run_visualize(sys.argv[1:])


if __name__ == "__main__":
    main()
