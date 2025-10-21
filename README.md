# Borunte Robotics Toolkit

This repository provides a clean baseline for RGB-D acquisition, vision analysis, and calibration
workflows around a Borunte industrial arm. The codebase is organized into modular packages with
focused responsibilities and strongly typed configuration models.

## Repository Structure

```
config.py              # Global settings dataclass with environment overrides
borunte/               # Robot client, capture workflows, and configuration
calib/                 # Calibration entry points and defaults
vision/                # RGB-D series IO, processing, analysis, and visualization
utils/                 # Shared helpers for logging, IO, geometry, progress, and RealSense stubs
```

All Python modules start with a path header comment and expose an English-only public API. Package
`__init__.py` files simply re-export public symbols without triggering side effects.

## Configuration Model

The root [`config.py`](config.py) defines the `Settings` dataclass. It centralizes paths, robot
connection properties, timeouts, and RealSense defaults. Each package layers its own configuration on
top:

- [`vision/config.py`](vision/config.py): camera intrinsics, visualization switches, default capture
  locations, and voxel sizes used by the merger.
- [`borunte/config.py`](borunte/config.py): robot host parameters and capture grid defaults.
- [`calib/config.py`](calib/config.py): default hand-eye calibration matrices and storage paths.

Environment variables prefixed with `BORUNTE_` override `Settings` fields automatically.

## Intrinsics and Paths

Vision routines respect the paths declared in `VisionConfig`, including:

- `capture_root`: default dataset location.
- `image_directory_name`: optional subdirectory containing RGB-D frames.
- `poses_json_name` and `intrinsics_json_name`: metadata filenames stored next to captures.

When generating new data via `borunte.run_grid_capture`, the capture directory contains paired files
named `000_depth.png`, `000_rgb.png`, etc., plus a `poses.json` metadata file that follows the naming
convention above.

## Quickstarts

### Merge RGB-D Sequences

Import and call the merge entry point:

```python
from vision.analysis.merge_runner import run_merge

run_merge("/path/to/capture")
```

Alternatively invoke the module shim:

```
python -m vision.analysis.merge_runner /path/to/capture
```

The merger downsamples each frame using `FRAME_VOX`, optionally visualizes intermediate results via
the Open3D-only viewer, performs centroid alignment, and writes the merged `.ply` cloud to
`VisionConfig.saves_root`.

### Visualize Frames

```python
from vision.analysis.visualize_runner import run_visualize

run_visualize(["capture/000_depth.png", "capture/001_depth.png"])
```

or via `python -m vision.analysis.visualize_runner ...` to step through each frame in a simple
Open3D viewer window.

### Grid Capture

```python
from borunte import BorunteClient, run_grid_capture

with BorunteClient() as client:
    output_dir = run_grid_capture(client)
```

This records a timestamped capture directory under `BorunteConfig.captures_root`, populates RGB-D
placeholders, and logs simulated robot poses.

### Calibration Routines

```python
from calib import run_detect, run_pnp, run_handeye, run_depth_align
```

Each function uses concise logging, shared progress bars, and formatting helpers for readable output.

## Logging and Progress

All user-facing feedback flows through the project logger provided by [`utils/logger.py`](utils/logger.py).
Lengthy operations display a single `tqdm` progress bar. The shared [`ErrorTracker`](utils/error_tracker.py)
collects errors during capture operations, ensuring issues are summarized at the end of a run.

## Outputs

- Merged clouds are written to `VisionConfig.saves_root`.
- Captures store RGB, depth, and pose metadata within a timestamped directory under
  `BorunteConfig.captures_root`.
- Calibration utilities return NumPy arrays or dataclasses for downstream consumption.

## Development

This project uses modern Python tooling with `uv` for dependency management.

### Setup

```bash
uv sync --dev
```

### Code Quality

Run linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

### Type Checking

Run mypy for static type analysis:

```bash
uv run mypy borunte calib vision utils
```

### Testing

Run the test suite:

```bash
uv run pytest
uv run pytest --cov=borunte --cov-report=html
uv run pytest -m "not hardware"
```

### Running Modules

Execute modules via `uv run`:

```bash
uv run -m vision.analysis.merge_runner /path/to/capture
uv run -m vision.analysis.visualize_runner /path/to/frames
```

### Configuration

All configuration is centralized in `borunte/config.py`. Override settings via environment variables:

```bash
export BORUNTE_ROBOT_HOST=192.168.1.100
export BORUNTE_LOG_LEVEL=DEBUG
```

## Changelog

- **v0.2.0**: Refactored with centralized config, SOLID principles, comprehensive typing, pytest suite, and modern tooling (ruff, mypy).
- **v0.1.0**: Initial refactor with modular packages, Open3D-only visualization, atomic IO helpers, and consistent logging.
