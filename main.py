"""
Main entry point for Borunte Robotics Calibration Toolkit
Demonstrates the core functionality of the toolkit
"""
import numpy as np
from borunte.cli.capture_runner import main as capture_main
from borunte.calib.handeye import calibrate_hand_eye, calibrate_hand_eye_multi
from borunte.cam.realsense_d415 import RealSenseD415
from borunte.control.client import RobotClient
from borunte.grid import build_grid_for_count


def main():
    print("Borunte Robotics Calibration Toolkit")
    print("=====================================")
    
    # Example usage of core functionality
    print("\n1. Demonstrating grid generation...")
    rows, cols = build_grid_for_count(20, rows=None, cols=None)
    print(f"  - Generated grid: {rows}x{cols} for 20 points")
    
    print("\n2. Demonstrating hand-eye calibration...")
    
    # Show how actual calibration would work (with dummy data)
    print("\n3. Running hand-eye calibration with dummy data...")
    
    # In a real scenario, we would capture poses from the robot and camera
    # For this example, we'll use dummy data to demonstrate the interface
    dummy_tcp_poses = [np.eye(4) for _ in range(5)]  # 5 dummy TCP poses (robot)
    dummy_board_poses = [np.eye(4) for _ in range(5)]  # 5 dummy board poses (camera)
    
    try:
        # Single method calibration
        R, t, metrics = calibrate_hand_eye(dummy_tcp_poses, dummy_board_poses, method="Tsai")
        print(f"  - Single method calibration completed successfully")
        print(f"  - Rotation matrix shape: {R.shape}")
        print(f"  - Translation vector: {t}")
        print(f"  - Calibration angle: {metrics.get('angle_deg', 'N/A'):.2f}°")
        print(f"  - Translation norm: {metrics.get('trans_norm_m', 'N/A'):.3f}m")
    except Exception as e:
        print(f"  - Single method calibration failed with error: {e}")
    
    try:
        # Multi-method calibration
        multi_results = calibrate_hand_eye_multi(dummy_tcp_poses, dummy_board_poses)
        print(f"\n  - Multi-method calibration completed")
        print(f"  - Successful methods: {multi_results.get('successful_methods', [])}")
        print(f"  - Total poses used: {multi_results.get('total_poses', 0)}")
    except Exception as e:
        print(f"  - Multi method calibration failed with error: {e}")
    
    print("\n4. Available CLI tools (use from command line):")
    print("  - borunte-capture: Automated grid capture")
    print("  - borunte-waypoints: Waypoint management")  
    print("  - borunte-visualize: Data visualization")
    print("  - borunte-merge: Merge analysis results")
    
    print("\n5. Example of running capture workflow:")
    print("  - This would connect to a real robot and camera, capture calibration data")
    print("  - To run: borunte-capture --host ROBOT_IP --port PORT --interactive")
    
    print("\nDone!")


if __name__ == "__main__":
    main()