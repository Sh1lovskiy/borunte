# Borunte Robotics and Calibration Toolkit

This repository contains the Borunte robot TCP client, RealSense capture utilities, and offline calibration pipelines. The code is organised into Python packages so it can be imported or executed via `uv run -m package.module`.

## Repository layout

- `config.py` – project-wide settings shared by all packages.
- `borunte/` – robot client, capture session helpers, RealSense preview, and the high level runner.
- `calib/` – ChArUco detection utilities and offline calibration entry points.
- `utils/` – logging and error tracking helpers.
- `examples/` – small scripts such as `smoke_refactor_check.py` for sanity checks.

## Configuration

Configuration is centralised in dataclasses. The root `config.Settings` reads defaults from the repository and honours environment variables (e.g. `BORUNTE_DATA_ROOT`, `BORUNTE_ROBOT_HOST`).

Each package provides its own config wrapper:

- `borunte.config.BorunteConfig` exposes capture profiles, RealSense preview parameters, robot network settings, and motion limits.
- `calib.config.CalibConfig` defines detector thresholds, solver options, and output filenames.

Use the provided objects directly:

```python
from borunte.config import BORUNTE_CONFIG
from calib.config import CALIB_CONFIG
```

All filesystem paths come from these configs and rely on `pathlib.Path`. Output directories are created on demand when writing files.

## Running capture and calibration

The capture pipeline is importable and does not rely on CLI parsing:

```python
from borunte.runner import run_capture_pipeline
from borunte import BORUNTE_CONFIG

run_capture_pipeline(BORUNTE_CONFIG)
```

Alternatively, run the shim:

```bash
uv run -m borunte.runner
```

Long-running loops such as pose traversal and image detection use tqdm progress bars and log concise milestones. The project logger lives in `utils/logger.py`; use `Logger.get_logger()` to obtain a module-scoped logger.

## Logging and outputs

Log files are stored under the directory defined by `Settings.logs_root` (default `./logs`). Capture sessions are written under `Settings.captures_root` with timestamped directories. Calibration outputs are stored below `Settings.default_calibration_output` with per-dataset subdirectories.

## Quickstart

```python
from borunte import BORUNTE_CONFIG, run_capture_pipeline
from borunte.grid import build_grid_for_count
from calib.offline import run_offline

# Inspect config
print(BORUNTE_CONFIG.network.host)

# Build a grid of poses
poses = build_grid_for_count(config=BORUNTE_CONFIG)

# Trigger a dry-run smoke check (no hardware interaction)
from examples.smoke_refactor_check import main as smoke
smoke()

# Run offline calibration on an existing dataset
from calib.config import CALIB_CONFIG
run_offline("captures/latest_session/preview_single", CALIB_CONFIG)
```

## Changelog

- Introduced root `config.Settings` and package-level config dataclasses.
- Replaced ad-hoc constants with structured configs across robot and calibration code.
- Added `borunte.runner` importable pipeline and simplified `main.py` shim.
- Reworked robot TCP client into `RobotClient` with proper logging and retries.
- Simplified offline calibration workflow to use centralised config and atomic outputs.
- Added smoke test script and updated README.
