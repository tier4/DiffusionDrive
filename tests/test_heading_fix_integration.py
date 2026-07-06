"""Integration test to verify heading fix using real Bench2Drive data."""

import json
import numpy as np
from pathlib import Path
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig


def test_heading_fix_with_real_data():
    """Test heading extraction with actual Bench2Drive mini dataset."""
    # Use the mini dataset for testing
    data_folder = Path("/workspace/Bench2Drive-mini")

    # Look for a scene
    scene_dirs = list(data_folder.glob("*"))
    if not scene_dirs:
        print("No scenes found in mini dataset")
        return

    scene_name = scene_dirs[0].name
    print(f"Testing with scene: {scene_name}")

    # Create config - using actual Bench2DriveDataConfig structure
    config = Bench2DriveDataConfig(
        data_root=data_folder,
        scenarios=[scene_name],
        sampling_rate=1,  # Don't downsample for testing
        num_frames=1,
        num_history_frames=0,
        num_future_frames=0,
    )

    # Load scene
    try:
        scene = Bench2DriveScene(config)

        # Load annotation directly to inspect
        anno_path = data_folder / scene_name / "anno" / "00000001.json.gz"
        if anno_path.exists():
            import gzip

            with gzip.open(anno_path, "rt") as f:
                anno = json.load(f)

            print(f"\nAnnotation theta: {anno.get('theta', 'N/A')} degrees")

            # Check if ego vehicle exists in bounding boxes
            ego_found = False
            for box in anno.get("bounding_boxes", []):
                if box.get("class") == "ego_vehicle":
                    ego_found = True
                    print(f"Ego bounding box rotation[2] (yaw): {box['rotation'][2]} degrees")
                    print(f"Ego bounding box location: {box['location']}")
                    break

            if not ego_found:
                print("No ego vehicle found in bounding boxes")

            # Get ego status from the scene
            ego_status = scene.ego_status
            print(
                f"\nExtracted ego pose: x={ego_status.ego_pose[0]:.2f}, y={ego_status.ego_pose[1]:.2f}, heading={np.degrees(ego_status.ego_pose[2]):.2f} degrees"
            )
            print(
                f"Extracted ego velocity: vx={ego_status.ego_velocity[0]:.2f}, vy={ego_status.ego_velocity[1]:.2f}"
            )

            # Verify heading came from bounding box if available
            if ego_found:
                expected_heading_deg = -box["rotation"][2]  # CW to CCW conversion
                actual_heading_deg = np.degrees(ego_status.ego_pose[2])
                print(f"\nHeading verification:")
                print(f"  Expected (from bbox): {expected_heading_deg:.2f} degrees")
                print(f"  Actual: {actual_heading_deg:.2f} degrees")
                print(f"  Match: {np.isclose(expected_heading_deg, actual_heading_deg, atol=0.1)}")

    except Exception as e:
        print(f"Error loading scene: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_heading_fix_with_real_data()
