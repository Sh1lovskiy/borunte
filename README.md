# Borunte Robotics Toolkit

A modular, typed pipeline for RGB-D capture, hand-eye calibration, point cloud merging, and robotic vision analysis. All configuration is centralized in [borunte/config.py](borunte/config.py).

## Overview

The Borunte toolkit coordinates a Borunte robot arm with an Intel RealSense RGB-D camera to capture synchronized pose and depth data, perform hand-eye calibration, merge point clouds, and analyze the workspace.

### Pipeline Stages

1. **Capture** - Robot moves through workspace grid capturing RGB-D frames with synchronized poses
2. **Calibration** - Hand-eye calibration between robot TCP and camera using ChArUco patterns
3. **Merge** - Align and merge point clouds from multiple viewpoints into coherent 3D model
4. **Analysis** - Crop, filter, and export point clouds for downstream processing

## Directory Structure

```
borunte/
  ├── __init__.py           # Package exports
  ├── config.py             # Centralized configuration
  ├── runner.py             # Main capture pipeline orchestrator
  ├── control/              # Robot control and communication
  │   ├── client.py         # RemoteMonitor JSON TCP client
  │   ├── controller.py     # High-level control (heartbeat, motion)
  │   ├── motion.py         # Single-point motion execution
  │   ├── state.py          # Robot state queries
  │   └── protocols.py      # Abstract protocol interfaces
  ├── cam/                  # Camera and capture session
  │   ├── realsense.py      # Intel RealSense interface
  │   └── session.py        # Capture session I/O management
  ├── grid/                 # Workspace grid generation
  │   ├── generator.py      # 3D grid generation with jitter
  │   └── waypoints.py      # Pose loading from JSON/CSV
  ├── calib/                # Calibration pipeline
  │   └── pipeline.py       # Hand-eye calibration utilities
  ├── vision/               # Point cloud processing
  │   ├── merge.py          # Point cloud merge entry point
  │   ├── analysis/         # Analysis runners
  │   ├── processing/       # Core processing utilities
  │   ├── io/               # I/O utilities
  │   └── viz/              # Visualization
  ├── utils/                # Shared utilities
  │   ├── logger.py         # Loguru configuration
  │   ├── error_tracker.py  # Exception tracking and cleanup
  │   └── ...               # Other utilities
  ├── core/                 # Core geometry utilities
  │   └── geometry/angles.py
  └── cli/                  # Command-line interfaces
      └── capture_runner.py
```

## Quick Start

### Installation

```bash
# Install with uv (recommended)
uv sync --dev
```

### Run Capture Pipeline

```bash
# Run with default config
uv run -m main
```

### Merge Point Clouds

```bash
# Merge from latest capture
uv run -m borunte.vision.merge
```

## Configuration

All settings are in [borunte/config.py](borunte/config.py). Access via:

```python
from borunte.config import BORUNTE_CONFIG, get_settings

workspace = BORUNTE_CONFIG.motion.workspace_m
```

Environment variables:
- `BORUNTE_ROBOT_HOST` - Robot IP (default: 192.168.4.4)
- `BORUNTE_ROBOT_PORT` - TCP port (default: 9760)
- `BORUNTE_RS_WIDTH` - Camera width (default: 1280)
- `BORUNTE_RS_HEIGHT` - Camera height (default: 720)

## Key Modules

- **borunte/control/** - Robot communication and motion control
- **borunte/cam/** - RealSense capture and session management
- **borunte/grid/** - Workspace grid generation and waypoints
- **borunte/calib/** - Hand-eye calibration pipeline
- **borunte/vision/** - Point cloud merging and processing
- **borunte/utils/** - Logging, error tracking, I/O

## Development

```bash
# Format and lint
uv run ruff format borunte/
uv run ruff check borunte/ --fix

# Type check
uv run mypy borunte/

# Test
uv run pytest
```

## Architecture

- **Centralized Config** - Single source in config.py
- **Type Safety** - Full mypy strict mode
- **Error Handling** - Centralized error tracker
- **Logging** - Structured loguru with module tags
- **PEP8** - 100-char lines, clear naming

## Workflow

1. Configure robot IP and workspace in config.py
2. Run capture: `uv run -m main`
3. Data saved to `captures/<timestamp>/`
4. Merge point clouds: `uv run -m borunte.vision.merge`
5. Visualize with Open3D

## Dependencies

- pyrealsense2, numpy, opencv, open3d
- pymodbus, loguru, tqdm
- Dev: ruff, mypy, pytest

## License

[Your License]
