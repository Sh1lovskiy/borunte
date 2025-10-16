# vision/cloud/merge.py
"""Point cloud merging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from vision.cloud.io import export_statistics, load_point_cloud, save_point_cloud
from vision.cloud.preprocessing import center_cloud, voxel_downsample
from vision.cloud.registration import register_cloud
from vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from utils.logger import get_logger
from utils.progress import track

LOGGER = get_logger(__name__)


class CloudMerger:
    """Combine multiple point clouds using centroid alignment."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or DEFAULT_VISION_CONFIG

    def merge_arrays(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        merged: List[np.ndarray] = []
        reference: np.ndarray | None = None
        for cloud in track(list(clouds), description="Merging clouds"):
            processed = voxel_downsample(cloud, config=self.config)
            processed = center_cloud(processed)
            if reference is None:
                reference = processed
                merged.append(processed)
                LOGGER.debug("Initialised merger with {} points", processed.shape[0])
                continue
            transform = register_cloud(reference, processed, config=self.config)
            transformed = (processed + transform[:3, 3])
            merged.append(transformed)
            LOGGER.debug("Applied translation {}", transform[:3, 3].tolist())
        if not merged:
            raise ValueError("No point clouds provided")
        combined = np.vstack(merged)
        LOGGER.info("Merged {} clouds into {} points", len(merged), combined.shape[0])
        return combined

    def merge_files(self, paths: Iterable[Path]) -> np.ndarray:
        clouds = [load_point_cloud(path) for path in paths]
        return self.merge_arrays(clouds)

    def merge_and_save(self, paths: Iterable[Path], output_path: Path) -> Path:
        merged = self.merge_files(paths)
        save_point_cloud(output_path, merged)
        export_statistics(output_path.with_suffix(".json"), merged)
        return output_path


PointCloudAggregator = CloudMerger  # backward compatibility

__all__ = ["CloudMerger", "PointCloudAggregator"]
