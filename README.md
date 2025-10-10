# Borunte Robot Calibration and Capture System

This repository provides a modular, production-grade Python system for controlling a **Borunte industrial robot**, managing **RGB-D camera capture** (Intel RealSense), and performing **offline hand–eye calibration**.
The system is designed for repeatable grid-based scanning, dataset acquisition, and accurate calibration between robot base, gripper, and camera coordinates.

---

## Features

* **Robot Control Layer**

  * Ethernet communication via Borunte’s JSON RemoteMonitor protocol (port 9760)
  * Direct TCP motion commands (MoveL, start/stop, emergency halt)
  * State monitoring (mode, alarm, motion)
  * Heartbeat watchdog for continuous operation

* **Camera Management**

  * Live RealSense RGB-D streaming with synchronized frame capture
  * Interactive or automatic capture modes
  * On-screen preview with RGB and depth visualization
  * Automatic saving of `rs2_params.json` for calibration reuse

* **Grid and Waypoint Execution**

  * 3D workspace definition with deterministic grid generation
  * Optional predefined waypoint execution (`waypoints.json`)
  * Pose logging to `poses.json` for each capture point

* **Calibration and Analysis**

  * ChArUco/ArUco/Checkerboard detection with OpenCV ≥4.7
  * Full corner and ID extraction for robust PnP estimation
  * Multiple hand–eye solvers (Tsai, Park, Horaud, Daniilidis, Andreff)
  * Detailed reports with rotation/translation diagnostics and RMSE filtering

---

## Repository Structure

```
borunte/
├── cam_rs.py           # RealSense streaming and capture
├── cap_session.py      # Dataset folder management
├── config.py           # IP, port, workspace, motion parameters
├── control.py          # Robot control and emergency handling
├── grid.py             # Grid generation utilities
├── motion.py           # Low-level movement functions
├── state.py            # Robot state and pose queries
├── waypoints.py        # Waypoint loading and validation
├── wire.py             # Socket I/O layer for JSON protocol
calib/
├── charuco.py          # Board generation and pose detection
├── handeye.py          # Multiple hand–eye algorithms
├── io_utils.py         # Data path helpers
├── offline.py          # Offline calibration orchestrator
utils/
├── logger.py           # Log formatting and runtime tags
├── error_tracker.py    # Context-managed error handling
├── keyboard.py         # Optional key event listener
├── settings.py         # Shared constants
main.py                 # Entry point (grid capture + calibration)
pyproject.toml          # Project configuration
```

---

## Typical Workflow

### 1. Grid or Waypoint Capture

```bash
uv run -m main
```

**Modes:**

* `RUN_MODE=grid` – execute generated workspace grid
* `RUN_MODE=grid_with_calib` – capture and immediately run calibration
* `RUN_MODE=calib_only` – run calibration on existing dataset

### 2. Offline Calibration

After capture, calibration results are saved to:

```
captures/<timestamp>/.../handeye_report_validated.txt
```
Reports include:

* ChArUco reprojection RMSE per frame
* Transformation spread diagnostics
* Best translation vector by prior distance

---

## Requirements

* Python ≥3.10
* OpenCV ≥4.7
* Intel RealSense SDK (`pyrealsense2`)
* NumPy, tqdm, loguru
* Borunte robot connected via TCP (default: 192.168.4.4:9760)

---

## Example Dataset Layout

```
captures/
  20251010_093305/
    preview_single/
      000_rgb.png
      000_depth.png
    poses.json
    rs2_params.json
```
