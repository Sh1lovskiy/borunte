# vision/analysis/merge_runner.py
"""Entry point for point cloud merging."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import open3d as o3d

from borunte.utils.logger import get_logger
from borunte.vision.config import VisionConfig
from borunte.vision.io.series import load_series
from borunte.vision.processing.depth import depth_to_cloud
from borunte.vision.processing.merge import CloudMerger
from borunte.vision.viz.viewer import Viewer

logger = get_logger(__name__)


def _as_sequence(paths: Sequence[Path | str] | Path | str) -> Sequence[Path | str]:
    if isinstance(paths, (str, Path)):
        return [paths]
    return paths


def run_merge(
    paths: Sequence[Path | str] | Path | str,
    *,
    output_path: Path | None = None,
    config: VisionConfig | None = None,
    visualize: bool | None = None,
) -> Path:
    """Merge RGB-D sequences into a single point cloud."""
    config = config or VisionConfig()
    visualize_flag = config.visualize if visualize is None else visualize
    frames = load_series(_as_sequence(paths))
    if not frames:
        raise ValueError("No frames loaded")
    logger.info(f"Loaded {len(frames)} frames for merging")

    clouds = [depth_to_cloud(frame.depth, frame.color, config) for frame in frames]
    logger.info("Converted frames to point clouds")

    merger = CloudMerger(
        config=config,
        visualize=visualize_flag,
        visualize_per_frame=config.visualize_per_frame,
        visualize_every_k=config.visualize_every_k,
        visualize_stages=config.visualize_stages,
        viewer_factory=Viewer,
    )
    result = merger.merge(clouds)

    destination = Path(output_path) if output_path else config.saves_root / "merged.ply"
    destination.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(destination), result.cloud)
    logger.info(f"Merged cloud written to {destination}")
    return destination


def main() -> None:
    """Module entry point for ``python -m vision.analysis.merge_runner``."""
    if len(sys.argv) < 2:
        raise SystemExit("Please provide at least one input path")
    run_merge(sys.argv[1:])


if __name__ == "__main__":
    main()
