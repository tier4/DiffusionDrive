import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_dataloader import Bench2DriveConfig
from pathlib import Path


class TestBench2DriveTrajectoryTransform:
    """Test trajectory transformation using world2ego matrix."""

    @pytest.fixture
    def mock_scene(self):
        """Create a mock scene for testing."""
        with patch.object(Bench2DriveScene, "__init__", lambda x, y, z: None):
            scene = Bench2DriveScene(None, None)
            scene.frames = [
                "frame0",
                "frame1",
                "frame2",
                "frame3",
                "frame4",
                "frame5",
                "frame6",
                "frame7",
                "frame8",
                "frame9",
            ]
            scene.history_frames = 1
            scene.data_folder = Path("/workspace/Bench2Drive-mini")
            scene.scene_name = "test_scene"
            return scene

    def test_trajectory_current_implementation(self, mock_scene):
        """Test current trajectory implementation that now uses bounding box and world2ego."""
        # Mock annotations for current and future frames
        # Create world2ego matrix for 30-degree rotation
        theta_rad = np.radians(30.0)
        world2ego = np.array(
            [
                [
                    np.cos(theta_rad),
                    np.sin(theta_rad),
                    0,
                    -100 * np.cos(theta_rad) - 200 * np.sin(theta_rad),
                ],
                [
                    -np.sin(theta_rad),
                    np.cos(theta_rad),
                    0,
                    100 * np.sin(theta_rad) - 200 * np.cos(theta_rad),
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        current_anno = {
            "x": 100.0,
            "y": 200.0,
            "theta": 45.0,  # No longer used
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, 30.0],  # Now used
                    "location": [100.0, 200.0, 0.5],
                    "world2ego": world2ego.tolist(),
                }
            ],
        }

        future_annos = [
            {"x": 105.0, "y": 205.0, "theta": 50.0},  # +5m forward, +5m right
            {"x": 110.0, "y": 210.0, "theta": 55.0},  # +10m forward, +10m right
        ]

        # Mock the _load_annotation method
        def mock_load_annotation(frame_idx):
            if frame_idx == 1:
                return current_anno
            elif frame_idx == 2:
                return future_annos[0]
            elif frame_idx == 3:
                return future_annos[1]
            else:
                return {"x": 100.0, "y": 200.0, "theta": 45.0}

        mock_scene._load_annotation = mock_load_annotation

        # Get trajectory
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # Verify it's using theta for transformation
        assert isinstance(trajectory, torch.Tensor)
        assert trajectory.shape[0] == 8  # NUM_FUTURE_WAYPOINTS

        # With world2ego matrix, the future position (105, 205) is transformed
        # The world2ego matrix transforms the future world position
        future_world = np.array([105.0, 205.0, 0.0, 1.0])
        future_ego = world2ego @ future_world
        expected_x = future_ego[0]
        expected_y = future_ego[1]

        assert np.isclose(trajectory[0, 0].item(), expected_x, atol=0.1)
        assert np.isclose(trajectory[0, 1].item(), expected_y, atol=0.1)

    def test_trajectory_should_use_world2ego_matrix(self, mock_scene):
        """Test that trajectory should use world2ego matrix from ego bounding box."""
        # Create world2ego matrix for 30-degree rotation
        theta_rad = np.radians(30.0)
        world2ego = np.array(
            [
                [np.cos(theta_rad), np.sin(theta_rad), 0, -100],
                [-np.sin(theta_rad), np.cos(theta_rad), 0, -200],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        current_anno = {
            "x": 100.0,  # Currently used
            "y": 200.0,  # Currently used
            "theta": 45.0,  # Currently used (incorrect)
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, 30.0],  # Actual ego heading
                    "location": [100.0, 200.0, 0.5],
                    "world2ego": world2ego.tolist(),  # Should use this for transformation
                }
            ],
        }

        # After fix, trajectory points should be transformed using world2ego matrix
        # This test documents expected behavior after fix
        pass

    def test_trajectory_ego_relative_accuracy(self, mock_scene):
        """Test that trajectories are properly ego-relative."""
        # Set up annotations where ego doesn't move
        stationary_anno = {"x": 100.0, "y": 200.0, "theta": 0.0, "bounding_boxes": []}

        mock_scene._load_annotation = lambda idx: stationary_anno

        # Get trajectory
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # For stationary ego, all trajectory points should be at origin
        for i in range(trajectory.shape[0]):
            assert np.isclose(trajectory[i, 0].item(), 0.0, atol=0.1)
            assert np.isclose(trajectory[i, 1].item(), 0.0, atol=0.1)
            assert np.isclose(trajectory[i, 2].item(), 0.0, atol=0.1)

    def test_trajectory_with_ego_motion(self, mock_scene):
        """Test trajectory calculation with ego motion."""
        # Ego moves forward 5m per timestep
        annotations = {
            1: {"x": 100.0, "y": 200.0, "theta": 0.0},
            2: {"x": 105.0, "y": 200.0, "theta": 0.0},
            3: {"x": 110.0, "y": 200.0, "theta": 0.0},
            4: {"x": 115.0, "y": 200.0, "theta": 0.0},
        }

        mock_scene._load_annotation = lambda idx: annotations.get(idx, annotations[1])

        # Get trajectory from frame 1
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # First waypoint should be 5m ahead
        assert np.isclose(trajectory[0, 0].item(), 5.0, atol=0.1)
        assert np.isclose(trajectory[0, 1].item(), 0.0, atol=0.1)

        # Second waypoint should be 10m ahead
        assert np.isclose(trajectory[1, 0].item(), 10.0, atol=0.1)
        assert np.isclose(trajectory[1, 1].item(), 0.0, atol=0.1)

    def test_trajectory_extrapolation(self, mock_scene):
        """Test trajectory extrapolation when running out of frames."""
        # Only provide 2 future frames
        annotations = {
            1: {"x": 100.0, "y": 200.0, "theta": 0.0},
            2: {"x": 105.0, "y": 200.0, "theta": 0.0},
            3: {"x": 110.0, "y": 200.0, "theta": 0.0},
        }

        mock_scene.frames = ["frame0", "frame1", "frame2", "frame3"]  # Only 4 frames
        mock_scene._load_annotation = lambda idx: annotations.get(idx, annotations[1])

        # Get trajectory
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # Should still return 8 waypoints
        assert trajectory.shape[0] == 8

        # Last waypoints should repeat the last known position
        assert torch.allclose(trajectory[2], trajectory[3])
        assert torch.allclose(trajectory[3], trajectory[4])
