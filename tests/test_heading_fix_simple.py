"""Simple test to verify heading fix works correctly."""

import numpy as np
import json
import gzip
from pathlib import Path
from unittest.mock import Mock
from navsim.common.bench2drive_scene import Bench2DriveScene


def test_extract_ego_status_with_real_annotation():
    """Test _extract_ego_status with a real annotation file."""

    # Load a real annotation from the mini dataset
    data_folder = Path("/workspace/Bench2Drive-mini")
    scene_dirs = list(data_folder.glob("*"))

    if not scene_dirs:
        print("No scenes found in mini dataset")
        return

    scene_name = scene_dirs[0].name
    # Find first available annotation file
    anno_files = list((data_folder / scene_name / "anno").glob("*.json.gz"))
    if not anno_files:
        print(f"No annotation files found in {scene_name}")
        return
    anno_path = sorted(anno_files)[0]

    if not anno_path.exists():
        print(f"Annotation file not found: {anno_path}")
        return

    # Load annotation
    with gzip.open(anno_path, "rt") as f:
        anno = json.load(f)

    print(f"Testing with scene: {scene_name}")
    print(f"Annotation theta: {anno.get('theta', 'N/A')} degrees")

    # Create a mock scene object
    scene = Mock(spec=Bench2DriveScene)

    # Bind the method to our mock object
    scene._extract_ego_status = Bench2DriveScene._extract_ego_status.__get__(
        scene, Bench2DriveScene
    )

    # Call the method
    ego_status = scene._extract_ego_status(anno)

    print(f"\nExtracted ego status:")
    print(f"  Position: x={ego_status.ego_pose[0]:.2f}, y={ego_status.ego_pose[1]:.2f}")
    print(f"  Heading: {np.degrees(ego_status.ego_pose[2]):.2f} degrees")
    print(f"  Velocity: vx={ego_status.ego_velocity[0]:.2f}, vy={ego_status.ego_velocity[1]:.2f}")

    # Check if ego vehicle exists in bounding boxes
    ego_found = False
    for box in anno.get("bounding_boxes", []):
        if box.get("class") == "ego_vehicle":
            ego_found = True
            print(f"\nEgo vehicle found in bounding boxes:")
            print(f"  Bounding box rotation[2] (yaw): {box['rotation'][2]} degrees")
            print(f"  Bounding box location: {box['location']}")

            # Verify heading came from bounding box
            expected_heading_deg = -box["rotation"][2]  # CW to CCW conversion
            actual_heading_deg = np.degrees(ego_status.ego_pose[2])
            print(f"\nHeading verification:")
            print(f"  Expected (from bbox): {expected_heading_deg:.2f} degrees")
            print(f"  Actual: {actual_heading_deg:.2f} degrees")
            print(f"  Match: {np.isclose(expected_heading_deg, actual_heading_deg, atol=0.1)}")

            # Verify position came from bounding box
            print(f"\nPosition verification:")
            print(
                f"  Expected x: {box['location'][0]:.2f}, Actual x: {ego_status.ego_pose[0]:.2f}"
            )
            print(
                f"  Expected y: {box['location'][1]:.2f}, Actual y: {ego_status.ego_pose[1]:.2f}"
            )
            print(
                f"  Position match: {np.isclose(box['location'][0], ego_status.ego_pose[0], atol=0.1) and np.isclose(box['location'][1], ego_status.ego_pose[1], atol=0.1)}"
            )
            break

    if not ego_found:
        print("\nNo ego vehicle found in bounding boxes - using fallback values")
        print(f"  Fallback theta: {anno.get('theta', 'N/A')} degrees")
        print(f"  Fallback position: x={anno.get('x', 'N/A')}, y={anno.get('y', 'N/A')}")


if __name__ == "__main__":
    test_extract_ego_status_with_real_annotation()
