# vision/analysis/merge_runner.py
"""Entry point for point cloud merging."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from config import get_settings
from utils.error_tracker import ErrorTracker
from utils.io import ensure_directory
from utils.logger import get_logger
from vision.cloud.io import export_statistics, save_point_cloud
from vision.cloud.merge import CloudMerger
from vision.config import DEFAULT_VISION_CONFIG, VisionConfig

LOGGER = get_logger(__name__)


def run_merge(
    paths: Iterable[Path],
    *,
    output_path: Path | None = None,
    config: VisionConfig | None = None,
) -> Path:
    cfg = config or DEFAULT_VISION_CONFIG
    merger = CloudMerger(config=cfg)
    tracker = ErrorTracker(context="vision.merge")
    settings = get_settings()
    target = output_path or (settings.paths.saves_root / "vision" / "merged.npy")
    ensure_directory(target.parent)
    try:
        merged = merger.merge_files(paths)
        save_point_cloud(target, merged)
        export_statistics(target.with_suffix(".json"), merged)
    except Exception as exc:  # noqa: BLE001
        tracker.record("merge", str(exc))
        raise
    tracker.summary()
    LOGGER.info("Merged clouds saved to {}", target)
    return target


def main() -> None:  # pragma: no cover
    run_merge([])


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["run_merge"]
