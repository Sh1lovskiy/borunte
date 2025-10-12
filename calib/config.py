# calib/config.py
"""Configuration for calibration pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import SETTINGS, Settings
from borunte.config import BORUNTE_CONFIG, CharucoConfig


@dataclass(frozen=True)
class DetectionConfig:
    min_charuco_corners: int = BORUNTE_CONFIG.calibration.min_charuco_corners
    save_overlays: bool = BORUNTE_CONFIG.calibration.save_overlay_images
    reproj_rmse_min_px: float = BORUNTE_CONFIG.calibration.reproj_rmse_min_px
    reproj_rmse_max_px: float = BORUNTE_CONFIG.calibration.reproj_rmse_max_px
    reproj_rmse_step_px: float = BORUNTE_CONFIG.calibration.reproj_rmse_step_px
    coverage_threshold: float = 0.2


@dataclass(frozen=True)
class SolverConfig:
    min_frames: int = BORUNTE_CONFIG.calibration.min_sweep_frames
    ransac_reproj_threshold: float = 3.0
    refinement_iters: int = 2
    translation_prior: tuple[float, float, float] = (
        BORUNTE_CONFIG.calibration.prior_t_cam2gripper_m
    )


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    overlay_dirname: str = "overlays"
    detections_file: str = "charuco_detections.json"
    sweep_results_file: str = "handeye_sweep_results.json"
    report_raw: str = "handeye_report_no_validation.txt"
    report_validated: str = "handeye_report_validated.txt"


@dataclass(frozen=True)
class CalibConfig:
    settings: Settings
    charuco: CharucoConfig = BORUNTE_CONFIG.charuco
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(root=SETTINGS.default_calibration_output)
    )
    dataset_override: Optional[Path] = (
        BORUNTE_CONFIG.calibration.dataset_override
    )
    use_stream: str = BORUNTE_CONFIG.calibration.stream


def load_calib_config(settings: Settings = SETTINGS) -> CalibConfig:
    return CalibConfig(settings=settings)


CALIB_CONFIG = load_calib_config()

__all__ = ["CalibConfig", "CALIB_CONFIG", "load_calib_config", "DetectionConfig", "SolverConfig", "OutputConfig"]
