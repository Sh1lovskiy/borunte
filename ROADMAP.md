# Borunte Robotics Toolkit - Codebase Understanding Roadmap

## Overview
This roadmap helps navigate the Borunte robotics toolkit codebase by exploring packages, modules, and methods systematically to build deep understanding of the system architecture.

## Stage 1: Core Package Understanding (Understanding the Foundation)

### 1.1 borunte/config.py - Central Configuration System
**Key Dataclasses:**
- `Settings` - Main immutable configuration container
- `Pose`, `CameraIntrinsics`, `HandEyeTransform` - Core data types
- Configuration sections: `RobotConfig`, `RealSenseConfig`, `VisionConfig`, etc.

**Key Methods:**
- `get_settings()` - Factory function with environment overrides
- Constants for scaling: `ROBOT_POS_SCALE`, `ROBOT_ANG_SCALE`
- File naming constants: `FILENAME_POSES_JSON`, `FILENAME_INTRINSICS_JSON`

**Understanding Goal:** Know how all configuration flows through the system

### 1.2 borunte/control/ - Robot Communication Package
**Files:**
- `client.py` - Primary TCP communication with robot
- `controller.py` - High-level control (heartbeat, graceful release)
- `motion.py` - Motion execution logic
- `state.py` - Robot state queries
- `protocols.py` - Abstract interfaces

**Key Methods:**
- `RobotClient.connect()` - Establish TCP connection
- `RobotClient.move_to_pose()` - Verified motion sequence
- `RobotClient.read_registers()` - Safe register access
- `RobotClient.rewrite_pose()` - Pose writing with verification
- `Heartbeat` class - Connection maintenance
- `graceful_release()` - Safe robot control release

**Understanding Goal:** Understand robot communication protocol and safe motion execution

### 1.3 borunte/cam/ - Camera Management Package
**Files:**
- `realsense.py` - Intel RealSense interface
- `session.py` - Capture session management
- `__init__.py` - Package exports

**Key Methods:**
- `PreviewStreamer` class - Preview window with capture controls
- `capture_one_pair()` - Single RGB-D capture
- `CaptureSession` class - File I/O and session management
- `_depth_to_viz()` - Depth visualization processing

**Understanding Goal:** Understand camera capture pipeline and session organization

## Stage 2: Application Logic Packages

### 2.1 borunte/grid/ - Workspace Planning Package
**Files:**
- `generator.py` - 3D grid creation
- `waypoints.py` - Custom pose loading
- `__init__.py` - Package exports

**Key Methods:**
- `build_grid_for_count()` - Create ordered XYZUVW lattice
- `_counts_from_total()` - Calculate grid dimensions
- `_per_point_jitter()` - Add pose variation

**Understanding Goal:** Understand how workspace coverage is planned and executed

### 2.2 borunte/calib/ - Calibration Package
**Files:**
- `charuco.py` - ChArUco pattern detection
- `pipeline.py` - Calibration workflow
- `__init__.py` - Package exports

**Key Methods:**
- `detect_charuco()` - ChArUco board detection
- `estimate_cam_pose()` - Camera pose estimation
- `solve_handeye()` - Hand-eye calibration solution
- `draw_overlay()` - Visualization of results

**Understanding Goal:** Understand hand-eye calibration process from detection to solution

### 2.3 borunte/vision/ - Point Cloud Processing Package
**Subpackages:**
- `analysis/` - Analysis tools
- `io/` - Input/output utilities
- `processing/` - Core processing
- `viz/` - Visualization tools
- `__init__.py` - Package exports

**Key Methods:**
- Point cloud merging algorithms
- Data processing pipelines
- Visualization functions

**Understanding Goal:** Understand 3D processing and analysis capabilities

## Stage 3: Utility and Support Packages

### 3.1 borunte/utils/ - Shared Utilities Package
**Files:**
- `error_tracker.py` - Exception handling and cleanup
- `logger.py` - Structured logging
- `geometry.py` - Geometric operations
- `io.py` - Input/output operations
- `progress.py` - Progress tracking
- `rs.py` - RealSense utilities
- `view.py` - Visualization utilities
- `__init__.py` - Package exports

**Key Methods:**
- `ErrorTracker` class - Centralized error handling
- `get_logger()` - Module-specific logging
- Various geometry and I/O utilities

**Understanding Goal:** Understand cross-cutting concerns and utility functions

### 3.2 borunte/core/ - Core Domain Logic Package
**Subpackages:**
- `geometry/` - Geometric operations
- `__init__.py` - Package exports

**Understanding Goal:** Understand fundamental geometric operations

## Stage 4: Entry Points and Orchestration

### 4.1 borunte/runner.py - Main Pipeline Orchestrator
**Key Methods:**
- `run_capture_pipeline()` - Main capture workflow
- `main()` - CLI entry point with 4 modes
- `_setup_robot_mode()` - Robot preparation
- `_capture_preview()` and `_capture_auto()` - Different capture modes

**Understanding Goal:** Understand how all components are coordinated

### 4.2 borunte/__init__.py - Package Interface
**Exports:**
- `Settings`, `get_settings` - Configuration access
- `__version__` - Version information

**Understanding Goal:** Understand public API of the package

## Stage 5: Testing Infrastructure

### 5.1 tests/ - Test Suite
**Files:**
- `test_client.py` - Robot client tests
- `test_config.py` - Configuration tests
- `test_integration.py` - Integration tests
- `conftest.py` - Test fixtures

**Key Components:**
- `mock_client` fixture - Mock robot client for testing
- `sample_pose`, `sample_intrinsics`, `sample_hand_eye` - Test data
- `temp_captures_dir` - Temporary directory for tests

**Understanding Goal:** Understand how the system is validated

## Deep Understanding Path

### Week 1: Configuration Foundation
1. Study `borunte/config.py` completely
2. Understand how `Settings` dataclass aggregates all configurations
3. Identify all constants and their usage patterns
4. Trace configuration flow through the system

### Week 2: Robot Communication
1. Study `borunte/control/client.py` completely
2. Understand the TCP JSON protocol
3. Map out the complete `move_to_pose()` sequence
4. Understand error handling and verification steps

### Week 3: Camera System
1. Study `borunte/cam/realsense.py` completely
2. Understand preview window architecture with threading
3. Learn RGB-D capture process
4. Understand session file organization

### Week 4: Grid Generation
1. Study `borunte/grid/generator.py` completely
2. Understand 3D lattice generation algorithm
3. Learn about workspace coverage strategies
4. See how poses are calculated and jittered

### Week 5: Calibration Pipeline
1. Study `borunte/calib/` modules completely
2. Understand ChArUco detection and pose estimation
3. Learn hand-eye calibration mathematics
4. See result validation and output

### Week 6: Main Pipeline
1. Study `borunte/runner.py` completely
2. Understand how all components are orchestrated
3. Learn error handling and cleanup procedures
4. Map out complete capture workflow

### Week 7: Utilities and Testing
1. Study `borunte/utils/` modules
2. Understand error tracking system
3. Learn logging methodology
4. Review tests to understand expected behavior

### Week 8: Integration and Refinement
1. Review complete system architecture
2. Understand inter-package communication
3. Identify optimization opportunities
4. Document knowledge gaps for further study

## Key Architectural Patterns

### 1. Immutable Configuration Pattern
- `Settings` dataclass is frozen to prevent accidental changes
- Factory function `get_settings()` creates with environment overrides
- Configuration flows as parameters rather than globals

### 2. Resource Management Pattern
- Context managers and cleanup procedures
- `graceful_release()` function for safe resource cleanup
- Error tracking with exception hooks

### 3. Threading and Concurrency Pattern
- `PreviewStreamer` runs in background thread
- Robot heartbeat in separate thread
- Thread-safe access with locks where needed

### 4. Type Safety Pattern
- Full mypy strict mode compliance
- Dataclasses for type-safe data structures
- Type hints throughout the codebase

## Critical Methods to Master

### Robot Control
- `RobotClient.move_to_pose()` - Complete motion sequence
- `RobotClient.verify_pose_write()` - Pose verification
- `Heartbeat.start()` - Connection maintenance

### Capture Pipeline
- `run_capture_pipeline()` - Complete workflow orchestrator
- `capture_one_pair()` - Single capture operation
- `build_grid_for_count()` - Pose planning

### Calibration
- `detect_charuco()` - Pattern detection
- `solve_handeye()` - Calibration solution
- `PreviewStreamer.capture` - User-controlled capture

## Dependency Flow Understanding

### Input → Processing → Output Flow:
1. Robot pose commands → TCP communication → Robot movement
2. RealSense frames → ChArUco detection → Camera pose
3. Multiple poses → Hand-eye calibration → Robot-camera transform
4. Capture sessions → Point cloud merging → 3D model

## Module Interactions Map

```
config.py → (all modules) - Provides configuration
control/ ↔ runner.py - Robot control from main pipeline
cam/ ↔ runner.py - Camera capture from main pipeline
grid/ ↔ runner.py - Grid planning for capture
calib/ ↔ runner.py - Calibration in post-processing
utils/ → (all modules) - Provides utilities
```

This roadmap provides a systematic approach to deeply understand your Borunte robotics toolkit codebase, starting from the configuration foundation and building up to the complete system integration.