import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from navsim.common.bench2drive_scene import Bench2DriveScene
from navsim.common.bench2drive_dataloader import Bench2DriveDataConfig
from pathlib import Path


class TestBench2DriveTrajectoryTransform:
    """Test trajectory transformation using world2ego matrix."""

    @pytest.fixture
    def mock_scene(self):
        """Create a mock scene for testing."""
        with patch.object(Bench2DriveScene, "__init__", lambda x, y, z: None):
            scene = Bench2DriveScene(None, None)
            scene.anno_paths = [
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

        # Create future annotations with proper bounding boxes
        future_annos = {}
        for i in range(50):
            # Move 1m forward and 1m right per frame
            future_annos[i] = {
                "x": 100.0 + i,
                "y": 200.0 + i,
                "theta": 30.0 + i * 0.5,
                "bounding_boxes": [
                    {
                        "class": "ego_vehicle",
                        "rotation": [0, 0, 30.0 + i * 0.5],
                        "location": [100.0 + i, 200.0 + i, 0.5],
                        "world2ego": world2ego.tolist(),
                    }
                ],
            }

        # Mock the _load_annotation method
        def mock_load_annotation(frame_idx):
            if frame_idx == 1:
                return current_anno
            elif frame_idx in future_annos:
                return future_annos[frame_idx]
            else:
                return {"x": 100.0, "y": 200.0, "theta": 45.0}

        mock_scene._load_annotation = mock_load_annotation

        # Get trajectory
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # Verify it's using theta for transformation
        assert isinstance(trajectory, torch.Tensor)
        assert trajectory.shape[0] == 8  # NUM_FUTURE_WAYPOINTS

        # With consecutive frames, first waypoint is at frame 2 (1 + 1)
        # Position at frame 2 is (102, 202)
        future_world = np.array([102.0, 202.0, 0.0, 1.0])
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
        world2ego = np.array([
            [1, 0, 0, -100.0],
            [0, 1, 0, -200.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        stationary_anno = {
            "x": 100.0, 
            "y": 200.0, 
            "theta": 0.0, 
            "bounding_boxes": [
                {
                    "class": "ego_vehicle",
                    "rotation": [0, 0, 0.0],
                    "location": [100.0, 200.0, 0.5],
                    "world2ego": world2ego.tolist(),
                }
            ]
        }

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
        # Ego moves forward 1m per frame
        annotations = {}
        for i in range(50):
            # Create world2ego matrix that transforms current ego position to (0,0)
            # For each frame, ego is at (100+i, 200), so translation is (-100-i, -200)
            world2ego = np.array([
                [1, 0, 0, -100.0 - i],
                [0, 1, 0, -200.0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            annotations[i] = {
                "x": 100.0 + i, 
                "y": 200.0, 
                "theta": 0.0,
                "bounding_boxes": [
                    {
                        "class": "ego_vehicle",
                        "rotation": [0, 0, 0.0],
                        "location": [100.0 + i, 200.0, 0.5],
                        "world2ego": world2ego.tolist(),
                    }
                ],
            }

        mock_scene.anno_paths = [f"frame{i}" for i in range(50)]
        mock_scene._load_annotation = lambda idx: annotations.get(idx, annotations[0])

        # Get trajectory from frame 1
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # With consecutive frames:
        # First waypoint at frame 2 (1 + 1) should be 1m ahead
        assert np.isclose(trajectory[0, 0].item(), 1.0, atol=0.1)
        assert np.isclose(trajectory[0, 1].item(), 0.0, atol=0.1)

        # Second waypoint at frame 3 (1 + 2) should be 2m ahead
        assert np.isclose(trajectory[1, 0].item(), 2.0, atol=0.1)
        assert np.isclose(trajectory[1, 1].item(), 0.0, atol=0.1)

    def test_trajectory_extrapolation(self, mock_scene):
        """Test trajectory extrapolation when running out of frames."""
        # Only provide limited future frames
        annotations = {}
        for i in range(5):  # Only 5 frames total
            x_pos = 100.0 + i * 5.0
            # Create proper world2ego matrix for each frame
            world2ego = np.array([
                [1, 0, 0, -x_pos],
                [0, 1, 0, -200.0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            annotations[i] = {
                "x": x_pos, 
                "y": 200.0, 
                "theta": 0.0,
                "bounding_boxes": [
                    {
                        "class": "ego_vehicle",
                        "rotation": [0, 0, 0.0],
                        "location": [x_pos, 200.0, 0.5],
                        "world2ego": world2ego.tolist(),
                    }
                ],
            }

        mock_scene.anno_paths = [f"frame{i}" for i in range(5)]  # Only 5 frames
        mock_scene._load_annotation = lambda idx: annotations.get(idx, annotations[0])

        # Get trajectory from frame 1
        # With consecutive frames, we need frames: 2, 3, 4, 5, 6, 7, 8, 9
        # But we only have up to frame 4
        trajectory = mock_scene.get_future_trajectory(frame_idx=1)

        # Should still return 8 waypoints
        assert trajectory.shape[0] == 8

        # First waypoint (frame 2) should be valid
        assert np.isclose(trajectory[0, 0].item(), 5.0, atol=0.1)  # 100+2*5 - (100+1*5) = 5
        
        # Second waypoint (frame 3) should be valid 
        assert np.isclose(trajectory[1, 0].item(), 10.0, atol=0.1)  # 100+3*5 - (100+1*5) = 10
        
        # Third waypoint (frame 4) should be valid
        assert np.isclose(trajectory[2, 0].item(), 15.0, atol=0.1)  # 100+4*5 - (100+1*5) = 15

        # Remaining waypoints should repeat the last known position (frame 4)
        for i in range(3, 8):
            assert torch.allclose(trajectory[i], trajectory[2])

    def test_trajectory_temporal_sampling(self, mock_scene):
        """Test that trajectory samples consecutive frames.
        
        With the sampling_rate handled by dataloader, trajectories sample consecutive frames.
        Each frame represents 0.5s intervals (regardless of original sampling rate).
        This gives a 4-second prediction horizon (8 * 0.5s = 4s).
        """
        # Simulate scene after dataloader processing (frames already at 0.5s intervals)
        # Ego moves forward at constant 10 m/s
        annotations = {}
        for i in range(20):  # 10 seconds of data at 0.5s intervals
            x_pos = 100.0 + i * 5.0  # 10 m/s * 0.5s = 5m per frame
            world2ego = np.array([
                [1, 0, 0, -x_pos],
                [0, 1, 0, -200.0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            annotations[i] = {
                "x": x_pos,
                "y": 200.0,
                "theta": 0.0,
                "bounding_boxes": [
                    {
                        "class": "ego_vehicle",
                        "rotation": [0, 0, 0.0],
                        "location": [x_pos, 200.0, 0.5],
                        "world2ego": world2ego.tolist(),
                    }
                ]
            }
        
        mock_scene.anno_paths = [f"frame{i}" for i in range(20)]
        mock_scene._load_annotation = lambda idx: annotations.get(idx, annotations[0])
        
        # Get trajectory from frame 4
        trajectory = mock_scene.get_future_trajectory(frame_idx=4)
        
        # Check that we have 8 waypoints
        assert trajectory.shape[0] == 8, f"Expected 8 waypoints, got {trajectory.shape[0]}"
        
        # Now samples consecutive frames (each 0.5s apart, 5m apart in this case)
        for i in range(8):
            expected_distance = (i + 1) * 5.0  # 5m per 0.5s interval
            assert np.isclose(trajectory[i, 0].item(), expected_distance, atol=0.1), \
                f"Waypoint {i}: expected {expected_distance}m, got {trajectory[i, 0].item()}m"
    
    def test_trajectory_sampling_interval_for_bench2drive(self, mock_scene):
        """Test that NUM_FUTURE_WAYPOINTS and sampling works correctly for B2D.
        
        This test documents the behavior for Bench2Drive trajectories:
        - Dataloader handles temporal sampling (10Hz->2Hz if sampling_rate=5)  
        - Trajectory samples consecutive frames (already 0.5s intervals)
        - 8 waypoints total = 4 second horizon
        """
        from navsim.common.bench2drive_constants import NUM_FUTURE_WAYPOINTS
        
        # Verify we're using 8 future waypoints
        assert NUM_FUTURE_WAYPOINTS == 8, f"Expected 8 waypoints, got {NUM_FUTURE_WAYPOINTS}"
        
        # Create scenario where ego moves at varying speeds
        # Frames are already at 0.5s intervals after dataloader processing
        annotations = {}
        positions = [100.0]  # Starting position at frame 0
        for i in range(1, 20):
            # Speed varies: 5 m/s for first 5 frames (frames 0-4), then 15 m/s
            if i <= 5:  # First 6 frames (0-5)
                speed = 5.0  # m/s
            else:
                speed = 15.0  # m/s
            
            # Calculate position based on previous position
            new_x = positions[-1] + speed * 0.5  # distance = speed * dt (0.5s per frame)
            positions.append(new_x)
        
        # Create annotations with correct positions
        for i in range(20):
            current_x = positions[i]
            
            # Create world2ego matrix for current position
            world2ego = np.array([
                [1, 0, 0, -current_x],
                [0, 1, 0, -200.0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            annotations[i] = {
                "x": current_x,
                "y": 200.0,
                "theta": 0.0,
                "bounding_boxes": [
                    {
                        "class": "ego_vehicle",
                        "rotation": [0, 0, 0.0],
                        "location": [current_x, 200.0, 0.5],
                        "world2ego": world2ego.tolist(),
                    }
                ]
            }
        
        mock_scene.anno_paths = [f"frame{i}" for i in range(20)]
        mock_scene._load_annotation = lambda idx: annotations.get(idx, annotations[0])
        
        # Get trajectory from frame 0
        trajectory = mock_scene.get_future_trajectory(frame_idx=0)
        
        # Samples consecutive frames 1,2,3,4,5,6,7,8 (each 0.5s apart)
        # Expected distances at 0.5s intervals:
        # - Frame 1: 5 m/s * 0.5s = 2.5m
        # - Frame 2: 5 m/s * 1.0s = 5.0m  
        # - Frame 3: 5 m/s * 1.5s = 7.5m
        # - Frame 4: 5 m/s * 2.0s = 10.0m
        # - Frame 5: 5 m/s * 2.5s = 12.5m
        # - Frame 6: 12.5m + 15 m/s * 0.5s = 20.0m
        # - Frame 7: 12.5m + 15 m/s * 1.0s = 27.5m 
        # - Frame 8: 12.5m + 15 m/s * 1.5s = 35.0m
        
        expected_distances = [2.5, 5.0, 7.5, 10.0, 12.5, 20.0, 27.5, 35.0]
        
        print(f"Trajectory points (consecutive frame sampling):")
        for i in range(8):
            actual = trajectory[i, 0].item()
            expected = expected_distances[i]
            print(f"  Waypoint {i}: {actual:.2f}m ahead (expected {expected:.2f}m)")
            assert np.isclose(actual, expected, atol=0.1), \
                f"Waypoint {i}: expected {expected}m, got {actual}m"
