"""
Test for grid capture functionality with plotly visualization.
This test file demonstrates the use of offline plotly for visualizing 
calibration data and grid capture results.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any


def create_test_grid_points(
    workspace: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total_points: int = 20
) -> List[List[float]]:
    """
    Create test grid points similar to the original build_grid_for_count function
    
    Args:
        workspace: Tuple of (x_range, y_range, z_range) defining the workspace
        total_points: Total number of points to generate
    
    Returns:
        List of [x, y, z, rx, ry, rz] poses
    """
    (x0, x1), (y0, y1), (z0, z1) = workspace
    
    # Calculate grid dimensions
    sqrt_count = int(np.ceil(np.sqrt(total_points)))
    rows = sqrt_count
    cols = sqrt_count
    
    # Create grid
    x_vals = np.linspace(x0, x1, cols)
    y_vals = np.linspace(y0, y1, rows)
    
    poses = []
    for x in x_vals:
        for y in y_vals:
            # Default orientation values
            pose = [float(x), float(y), (z0 + z1) / 2, 0.0, 0.0, 0.0]
            poses.append(pose)
            if len(poses) >= total_points:
                break
        if len(poses) >= total_points:
            break
            
    return poses


def visualize_grid_capture(
    grid_points: List[List[float]], 
    title: str = "Grid Capture Visualization"
) -> str:
    """
    Visualize grid capture points using plotly offline
    
    Args:
        grid_points: List of [x, y, z, rx, ry, rz] poses
        title: Title for the visualization
    
    Returns:
        HTML string containing the visualization
    """
    x_coords = [point[0] for point in grid_points]
    y_coords = [point[1] for point in grid_points]
    z_coords = [point[2] for point in grid_points]
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=5,
                color=z_coords,  # Color by z-coordinate
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"Point {i}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" 
                  for i, p in enumerate(grid_points)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        width=800,
        height=600
    )
    
    # Return HTML string using plotly offline functionality
    html_str = plot(
        fig,
        output_type='div',
        include_plotlyjs=True,
        filename=None,
        auto_open=False
    )
    
    return html_str


def test_grid_capture_functionality():
    """Test the grid capture functionality with visualization"""
    print("Testing grid capture functionality...")
    
    # Define a test workspace
    workspace = (
        (-0.2, 0.2),  # X range: -200mm to 200mm
        (-0.1, 0.3),  # Y range: -100mm to 300mm  
        (0.1, 0.5)    # Z range: 100mm to 500mm
    )
    
    # Generate test grid points
    grid_points = create_test_grid_points(workspace, total_points=16)
    
    print(f"Generated {len(grid_points)} grid points")
    print(f"First point: {grid_points[0]}")
    print(f"Last point: {grid_points[-1]}")
    
    # Visualize the grid
    html_visualization = visualize_grid_capture(
        grid_points, 
        title="Test Grid Capture Points"
    )
    
    # Save visualization to file
    output_file = Path("test_grid_visualization.html")
    output_file.write_text(html_visualization, encoding="utf-8")
    
    print(f"Visualization saved to {output_file}")
    
    # Test with varying parameters
    print("\nTesting with different workspace...")
    workspace2 = (
        (-0.1, 0.1),
        (0.0, 0.2),
        (0.3, 0.4)
    )
    
    grid_points2 = create_test_grid_points(workspace2, total_points=12)
    html_visualization2 = visualize_grid_capture(
        grid_points2,
        title="Denser Grid Capture Points"
    )
    
    output_file2 = Path("test_grid_visualization_dense.html")
    output_file2.write_text(html_visualization2, encoding="utf-8")
    
    print(f"Denser grid visualization saved to {output_file2}")
    
    return grid_points, grid_points2


def save_grid_for_calibration(grid_points: List[List[float]], filename: str):
    """Save grid points in a format suitable for calibration"""
    data = {
        "grid_points": grid_points,
        "timestamp": str(np.datetime64('now')),
        "total_points": len(grid_points),
        "workspace_bounds": {
            "x": [min(p[0] for p in grid_points), max(p[0] for p in grid_points)],
            "y": [min(p[1] for p in grid_points), max(p[1] for p in grid_points)],
            "z": [min(p[2] for p in grid_points), max(p[2] for p in grid_points)]
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Grid data saved to {filename}")


if __name__ == "__main__":
    # Run the test
    grid1, grid2 = test_grid_capture_functionality()
    
    # Save grid data for calibration
    save_grid_for_calibration(grid1, "test_calibration_data_1.json")
    save_grid_for_calibration(grid2, "test_calibration_data_2.json")
    
    print("\nAll tests completed successfully!")