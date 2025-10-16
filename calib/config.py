# calib/config.py
"""Calibration-specific configuration values."""

from __future__ import annotations

from dataclasses import dataclass

from config import get_settings


@dataclass(slots=True)
class DetectorConfig:
    adaptive_thresh_win_size: int
    corner_refinement_win_size: int
    min_corners: int


@dataclass(slots=True)
class SolverConfig:
    max_iterations: int
    determinant_threshold: float
    orthogonality_tolerance: float


@dataclass(slots=True)
class OutputConfig:
    detections_file: str
    pnp_file: str
    handeye_file: str
    depth_file: str


@dataclass(slots=True)
class CalibConfig:
    board_name: str
    square_size: float
    marker_size: float
    detector: DetectorConfig
    solver: SolverConfig
    output: OutputConfig

    @staticmethod
    def from_settings() -> "CalibConfig":
        settings = get_settings()
        detector = DetectorConfig(
            adaptive_thresh_win_size=23,
            corner_refinement_win_size=5,
            min_corners=12,
        )
        solver = SolverConfig(
            max_iterations=100,
            determinant_threshold=0.01,
            orthogonality_tolerance=1e-3,
        )
        output = OutputConfig(
            detections_file="detections.json",
            pnp_file="pnp.json",
            handeye_file="handeye.json",
            depth_file="depth_alignment.json",
        )
        return CalibConfig(
            board_name=settings.calibration.board_name,
            square_size=settings.calibration.square_size,
            marker_size=settings.calibration.marker_size,
            detector=detector,
            solver=solver,
            output=output,
        )


DEFAULT_CALIB_CONFIG = CalibConfig.from_settings()

__all__ = ["CalibConfig", "DEFAULT_CALIB_CONFIG", "DetectorConfig", "OutputConfig", "SolverConfig"]
