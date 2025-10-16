# Borunte Robotics Toolkit

The Borunte repository now packages robot control, calibration, and vision tooling
under a unified configuration-driven architecture. Each module exposes Python
APIs and `python -m` shims for interactive or automated usage without relying on
`argparse`.

## Repository Layout

- `config.py` – Root `Settings` dataclass with environment overrides for paths,
  robot connectivity, RealSense defaults, and calibration parameters.
- `utils/` – Shared helpers for logging, error tracking, IO, geometry, RealSense
  lifecycle management, tqdm progress wrappers, and NumPy formatting.
- `borunte/` – Robot automation utilities with a stateful Modbus client,
  configurable grid capture workflow, and a shim for `python -m borunte.run_capture`.
- `calib/` – Calibration pipelines (`run_detect`, `run_pnp`, `run_handeye`,
  `run_depth_align`) plus shims for direct execution.
- `vision/` – Reorganised point-cloud processing stack with dedicated
  sub-packages for cloud manipulation, skeletonisation, graph analysis, and
  visualisation. Analysis runners expose `run_merge`, `run_skeleton`,
  `run_graph`, and `run_visualize` entrypoints.

## Configuration Model

`config.Settings` aggregates defaults for filesystem paths, robot networking,
RealSense capture, and calibration expectations. Override any value with
environment variables such as:

```bash
export BORUNTE_DATA_ROOT=~/datasets/borunte
export BORUNTE_ROBOT_HOST=10.0.0.2
export BORUNTE_RS_WIDTH=1920
```

Package-specific configs extend these defaults:

- `borunte.config.BorunteConfig` captures register addresses, retry policy,
  grid spacing, and capture speed.
- `calib.config.CalibConfig` bundles board geometry, detector tuning, solver
  thresholds, and output filenames.
- `vision.config.VisionConfig` encapsulates cloud, skeleton, graph, and
  visualisation parameters, ensuring algorithms are free from hardcoded values.

## Quickstart Workflows

### Merge Multiple Point Clouds

```python
from pathlib import Path
from vision.analysis.merge_runner import run_merge

merged_path = run_merge([Path("scan1.npy"), Path("scan2.npy")])
```

### Extract a Skeleton and Build a Graph

```python
from pathlib import Path
from vision.analysis.skeleton_runner import run_skeleton
from vision.analysis.graph_runner import run_graph

skeleton_path = run_skeleton(Path("merged.npy"))
graph_summary_path = run_graph(skeleton_path)
```

### Perform a Grid Capture Session

```python
from borunte import run_grid_capture

session_folder = run_grid_capture(columns=4, rows=3, session_name="test_run")
```

### Execute Hand-Eye Calibration Steps

```python
from pathlib import Path
from calib import run_detect, run_pnp, run_handeye, run_depth_align

detections = run_detect([Path("frame_01.png"), Path("frame_02.png")])
pnp = run_pnp(detections)
handeye = run_handeye(pnp, robot_poses=[[0, 0, 0, 0, 0, 0]])
run_depth_align(handeye, [Path("depth_01.png")])
```

## Logging, Progress, and Outputs

All pipelines rely on `utils.logger` and `utils.error_tracker` to produce
structured logs and summary reports. Long-running loops expose a single `tqdm`
progress bar through `utils.progress.track`. Outputs are stored under the
configured `Settings.paths` directories with metadata written atomically in JSON
form.

## Changelog

- Replaced legacy `vision/merge` and `vision/trusskit` modules with structured
  `vision/cloud`, `vision/skeleton`, `vision/graph`, and `vision/viz` packages.
- Added configuration dataclasses for the root project, vision, calibration, and
  robot control domains.
- Implemented reusable utilities for IO, geometry, RealSense lifecycle
  management, and logging.
- Introduced importable `run_*` entrypoints and associated `python -m` shims for
  grid capture, calibration, and vision analysis tasks.
- Standardised logging, progress indication, and output locations across all
  pipelines.
