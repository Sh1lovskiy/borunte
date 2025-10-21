# Borunte Robotics Pipeline

A modular, typed pipeline for RGB-D capture, calibration, point-cloud merging, and analysis. All constants are centralized in `borunte/config.py`. Commands run via `uv`.

## Overview

### Stages

1. **Capture** RGB-D frames and robot TCP poses
2. **Calibrate** camera-gripper (hand-eye) and camera intrinsics
3. **Merge** and crop point clouds into a coherent model
4. **Analyze** and export results

### Key Modules

- `borunte/cli/capture_runner.py` — capture entry point
- `borunte/calib/` — calibration utilities
- `borunte/vision/merge.py` — point cloud merge/crop/alignment
- `borunte/core/geometry/angles.py` — rotation conversions and helpers
- `borunte/config.py` — single source of configuration

## Quick Start

```bash
# Install with dev tools
uv sync --dev

# Lint, types, tests
uv run ruff check .
uv run mypy .
uv run pytest -q
```

## Configuration

All constants live in `borunte/config.py`. They are grouped into frozen dataclasses and enums:

- **PATHS** — capture root, output directories
- **ROBOT** — networking (host, port), scaling (ROBOT_POS_SCALE, ROBOT_ANG_SCALE)
- **VISION** — voxel sizes, crop boxes, depth truncation
- **CALIB** — thresholds, board settings, reprojection checks
- **LOG** — log level and formatting

Access them explicitly in code:

```python
from borunte.config import PATHS, VISION, ROBOT, CALIB
```

**Change only in config.py**; never hardcode values in modules.

## Pipeline

### 1) Capture

Runs motion and frame acquisition. Honors safety flags and timeouts.

```bash
uv run -m borunte.cli.capture_runner \
  --mode grid_with_calib \
  --interactive true \
  --view both
```

Uses: `ROBOT.*`, `PATHS.captures_root`, camera settings under `VISION`.

### 2) Calibration

Hand-eye and intrinsics. Write outputs to `PATHS.calib_dir`.

```bash
uv run -m borunte.calib.run \
  --input ${PATHS.captures_root} \
  --out   ${PATHS.calib_dir}
```

Uses: `CALIB.*`.

### 3) Merge

Voxel downsample, crop, align into a single cloud.

```bash
uv run -m borunte.vision.merge \
  --input ${PATHS.captures_root} \
  --calib ${PATHS.calib_dir} \
  --out   ${PATHS.merge_out}
```

Uses: `VISION.MERGE_VOXEL_SIZE`, `VISION.DEPTH_TRUNC`, depth truncation.

### 4) Analyze and Export

```bash
uv run -m borunte.vision.analyze \
  --input ${PATHS.merge_out} \
  --out   ${PATHS.report_dir}
```

## Repository Layout

```
borunte/
  __init__.py
  config.py                 # all constants and settings

  core/
    geometry/
      angles.py             # rotation conversions, degree/radian helpers

  vision/
    merge.py                # merge/crop/alignment
    analyze.py              # metrics, figures, exports (optional)

  calib/
    run.py                  # calibration entry

  cli/
    capture_runner.py       # capture entry (replaces root new5.py)

  robot/
    client.py               # robot communication
    control.py              # high-level control
    motion.py               # motion planning

  camera/
    realsense.py            # RealSense pipeline
    capture.py              # capture session

  io/
    files.py                # file I/O
    network.py              # network I/O
    rgbd.py                 # RGB-D utilities

  _internal/
    logging.py              # logger setup
    errors.py               # error tracking
    formatting.py           # display formatting
```

### Backward Compatibility

Root-level shims provide compatibility for legacy scripts:

- `degree.py` → re-exports `borunte.core.geometry.angles`
- `merge.py` → runs `borunte.vision.merge`
- `new5.py` → runs `borunte.cli.capture_runner`
- `all.py` → deprecated orchestration script

**These shims are deprecated** and will be removed in a future version.

## Roadmap

- [ ] Consolidate small helpers to reduce module count further
- [ ] Strengthen unit tests for all subsystems
- [ ] Add hardware integration tests gated by `-m "hardware"`
- [ ] Enforce strict typing across legacy modules
- [ ] Structured logs and richer HTML reports in `analyze.py`
- [ ] Optional: add a `docs/` site generated from docstrings

## Troubleshooting

- **Import cycles**: Ensure modules do not import `config.py` and then import back into themselves
- **Hardware-free tests**: Mock `RobotClient` and camera I/O
- **Large files**: Split modules >300 LOC by responsibility

## Development

```bash
# Code quality
uv run ruff check .
uv run mypy .

# Tests
uv run pytest
uv run pytest --cov=borunte --cov-report=html

# Run capture
uv run -m borunte.cli.capture_runner

# Run merge
uv run -m borunte.vision.merge
```

## Migration Notes

**Recent Changes (v0.2.0):**

This version restructures the repository for better organization:

1. **Moved root scripts** to proper packages:
   - `degree.py` → `borunte/core/geometry/angles.py`
   - `merge.py` → `borunte/vision/merge.py`
   - `new5.py` → `borunte/cli/capture_runner.py`
   - `all.py` → Functionality distributed to package modules

2. **Centralized constants** in `borunte/config.py`

3. **Created backward-compatibility shims** at repository root

4. **New package structure** with clear domain boundaries:
   - `core/` — domain logic and math
   - `vision/` — point cloud processing
   - `robot/` — robot control
   - `camera/` — camera operations
   - `cli/` — user-facing commands
   - `io/` — all I/O operations
   - `_internal/` — internal utilities

**Breaking Changes:**

- Import paths have changed. Use new locations:
  ```python
  # Old
  from degree import axis_angle_from_matrix

  # New
  from borunte.core.geometry.angles import axis_angle_from_matrix
  ```

- Root scripts (`degree.py`, `merge.py`, `new5.py`, `all.py`) are now shims and will emit deprecation warnings

**Migration Path:**

1. Update imports to use new package locations
2. Test with deprecation warnings enabled
3. Remove reliance on root-level shims
4. Use `uv run -m borunte.cli.*` for CLI commands

## License

MIT License

## Contributors

Borunte Robotics Team
