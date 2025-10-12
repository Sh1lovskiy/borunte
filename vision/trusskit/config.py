from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class RunConfig:
    cloud_path: str
    merge_node_radius: float
    raster_res_px: int
    save_tag: str
    normal_bin_edges_deg: Tuple[float, ...]
    orient_sectors: int
    cache_normals: bool
    bbox_points: Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = None
