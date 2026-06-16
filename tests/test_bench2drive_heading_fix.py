import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig


class TestBench2DriveHeadingFix:
    """Test heading extraction from ego vehicle bounding box."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock(spec=Bench2DriveDataConfig)
        config.data_folder = Path("/workspace/Bench2Drive-mini")
        config.scene_name = "test_scene"
        config.frame_index = 0
        return config

    @pytest.fixture
    def mock_scene(self, mock_config):
        """Create a mock scene for testing."""
        # We'll patch the __init__ to avoid file loading
        with patch.object(Bench2DriveScene, "__init__", lambda x, y: None):
            scene = Bench2DriveScene(None)
            scene.config = mock_config
            return scene

    def test_extract_ego_heading_from_bounding_box_current_implementation(self, mock_scene):
        """Test that heading is now correctly extracted from bounding box."""
        test_anno = {
            "theta": 4.863,  # No longer used after fix
            "speed": 5.0,
            "acceleration": [0.1, 0.2, 0.0],
            "command_near": 3,
            "x": 100.0,
            "y": 200.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, -171.349],  # This is now correctly used
                    "location": [100.0, 200.0, 0.5],
                    "world2ego": np.eye(4).tolist(),
                }
            ],
        }

        ego_status = mock_scene._extract_ego_status(test_anno)

        # After fix: uses bounding box rotation (-171.349 degrees)
        # With CW to CCW conversion: -(-171.349) = 171.349 degrees = 2.991 radians
        expected_heading = -np.radians(-171.349)
        assert np.isclose(ego_status.ego_pose[2], expected_heading, atol=0.001)

    def test_extract_ego_heading_from_bounding_box_fixed_implementation(self, mock_scene):
        """Test that ego heading should be extracted from bounding box rotation."""
        test_anno = {
            "theta": 4.863,  # This should NOT be used
            "speed": 5.0,
            "acceleration": [0.1, 0.2, 0.0],
            "command_near": 3,
            "x": 100.0,
            "y": 200.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, -171.349],  # Actual ego yaw
                    "location": [100.0, 200.0, 0.5],
                    "world2ego": np.eye(4).tolist(),
                }
            ],
        }

        # After fix, this should extract heading from bounding box
        # Expected heading: -171.349 degrees = -2.990 radians
        # With CW to CCW conversion: 2.990 radians
        expected_heading_fixed = -np.radians(-171.349)

        # After fix, this now works correctly
        ego_status = mock_scene._extract_ego_status(test_anno)
        assert np.isclose(ego_status.ego_pose[2], expected_heading_fixed, atol=0.001)

    def test_ego_position_from_bounding_box(self, mock_scene):
        """Test that ego position should come from bounding box location."""
        test_anno = {
            "theta": 4.863,
            "speed": 5.0,
            "acceleration": [0.1, 0.2, 0.0],
            "command_near": 3,
            "x": 100.0,  # Current implementation uses this
            "y": 200.0,  # Current implementation uses this
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, -171.349],
                    "location": [105.0, 205.0, 0.5],  # Should use this instead
                    "world2ego": np.eye(4).tolist(),
                }
            ],
        }

        ego_status = mock_scene._extract_ego_status(test_anno)

        # After fix: uses location from bounding box
        assert np.isclose(ego_status.ego_pose[0], 105.0, atol=0.001)
        assert np.isclose(ego_status.ego_pose[1], 205.0, atol=0.001)

    def test_fallback_when_no_ego_in_bounding_boxes(self, mock_scene):
        """Test fallback behavior when ego vehicle not found in bounding boxes."""
        test_anno = {
            "theta": 4.863,
            "speed": 5.0,
            "acceleration": [0.1, 0.2, 0.0],
            "command_near": 3,
            "x": 100.0,
            "y": 200.0,
            "bounding_boxes": [
                {
                    "class": "vehicle",  # Not ego_vehicle
                    "rotation": [0, 0, 45.0],
                    "location": [110.0, 210.0, 0.5],
                }
            ],
        }

        ego_status = mock_scene._extract_ego_status(test_anno)

        # Should fall back to using anno['theta']
        expected_heading = -np.radians(4.863)
        assert np.isclose(ego_status.ego_pose[2], expected_heading, atol=0.001)

    def test_velocity_components_with_correct_heading(self, mock_scene):
        """Test that velocity components should use correct heading."""
        test_anno = {
            "theta": 0.0,  # Currently used for velocity calculation
            "speed": 10.0,
            "acceleration": [0.0, 0.0, 0.0],
            "command_near": 3,
            "x": 100.0,
            "y": 200.0,
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, 90.0],  # Should be used instead
                    "location": [100.0, 200.0, 0.5],
                    "world2ego": np.eye(4).tolist(),
                }
            ],
        }

        ego_status = mock_scene._extract_ego_status(test_anno)

        # After fix: uses rotation from bounding box (90 degrees)
        # At 90 degrees, velocity should be (0, 10)
        assert np.isclose(ego_status.ego_velocity[0], 0.0, atol=0.001)
        assert np.isclose(ego_status.ego_velocity[1], 10.0, atol=0.001)


from pathlib import Path
