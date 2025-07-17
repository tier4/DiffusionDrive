"""
Feature builder for Bench2Drive dataset.
Adapts Bench2Drive data to DiffusionDrive's expected format.

Key Dimension Adaptations:
=========================
LiDAR Processing:
- Bench2Drive provides 85m x 85m LiDAR coverage
- DiffusionDrive expects 64m x 64m coverage
- Solution: Process at full 85m (340x340), then resize to 64m (256x256)
- Benefit: Preserves maximum information while maintaining compatibility

BEV Semantic Maps:
- Dimensions: 128 x 256 (H x W) to match DiffusionDrive
- Different from LiDAR BEV (256 x 256) as per original design
- Currently uses placeholder implementation (see BENCH2DRIVE_INTEGRATION.md)

Camera Processing:
- Stitch 3 front cameras to 1024 x 256 (W x H)
- Matches DiffusionDrive's expected input format
"""

from typing import Dict, List, Any

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, EgoStatus
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

# Import Bench2Drive constants
from navsim.common.bench2drive_constants import (
    BENCH2DRIVE_LIDAR_RANGE_M,
    BENCH2DRIVE_LIDAR_SIZE,
    DIFFUSIONDRIVE_LIDAR_SIZE,
    LIDAR_PIXELS_PER_METER,
    LIDAR_NORMALIZATION_FACTOR,
    BEV_SEMANTIC_HEIGHT,
    BEV_SEMANTIC_WIDTH,
    NUM_FUTURE_WAYPOINTS,
    MAX_AGENTS,
)


class Bench2DriveFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for Bench2Drive data.
    Converts raw sensor data to model-ready features.
    """

    def __init__(self, config: TransfuserConfig):
        """
        Initialize feature builder.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Build features from Bench2Drive agent input.

        Args:
            agent_input: Input containing cameras, lidars, and ego statuses

        Returns:
            Dictionary with camera, lidar, and status features
        """
        features = {}

        # Process camera features
        features["camera_feature"] = self._get_camera_feature(agent_input.cameras)

        # Process LiDAR features
        features["lidar_feature"] = self._get_lidar_feature(agent_input.lidars)

        # Process status features
        features["status_feature"] = self._get_status_feature(agent_input.ego_statuses)

        return features

    def _get_camera_feature(self, cameras: List[any]) -> torch.Tensor:
        """
        Process camera data into model-ready features.

        Bench2Drive has 6 cameras, but we only use 3 front cameras for DiffusionDrive.
        Stitches CAM_F0, CAM_L0, CAM_R0 horizontally.

        Args:
            cameras: List of Camera objects (we use the last/current one)

        Returns:
            Stitched camera tensor [3, 256, 1024]
        """
        # Get current frame cameras
        current_cameras = cameras[-1]  # Cameras object

        # Select front cameras in left-to-right order for stitching
        # cam_l0 (left), cam_f0 (front), cam_r0 (right)
        front_camera_images = []

        # Check and collect valid camera images
        for cam_name, cam in [
            ("cam_l0", current_cameras.cam_l0),
            ("cam_f0", current_cameras.cam_f0),
            ("cam_r0", current_cameras.cam_r0),
        ]:
            if cam.image is None:
                print(f"Warning: {cam_name} image is None, using placeholder")
                # Create placeholder image
                placeholder = np.zeros((900, 1600, 3), dtype=np.uint8)
                front_camera_images.append(placeholder)
            elif len(cam.image.shape) != 3 or cam.image.shape[2] != 3:
                print(
                    f"Warning: {cam_name} has invalid shape {cam.image.shape}, using placeholder"
                )
                placeholder = np.zeros((900, 1600, 3), dtype=np.uint8)
                front_camera_images.append(placeholder)
            else:
                front_camera_images.append(cam.image)

        # Stitch images horizontally first (in numpy for efficiency)
        # Original size: 900x1600 per camera -> 900x4800 total
        try:
            stitched_image = np.concatenate(front_camera_images, axis=1)
        except Exception as e:
            print(f"Error concatenating images: {e}")
            # Return black image as fallback
            return torch.zeros((3, 256, 1024), dtype=torch.float32)

        # Check if stitched image is valid before resizing
        if stitched_image.size == 0:
            print("Warning: Stitched image is empty, returning black image")
            return torch.zeros((3, 256, 1024), dtype=torch.float32)

        # Convert to PIL Image for resizing
        stitched_pil = Image.fromarray(stitched_image.astype(np.uint8))

        # Resize the stitched image to target size [256, 1024] using high-quality LANCZOS
        stitched_resized = stitched_pil.resize((1024, 256), Image.Resampling.LANCZOS)

        # Convert back to numpy array
        stitched_resized = np.array(stitched_resized)

        # Convert to tensor and change to CHW format
        stitched = torch.from_numpy(stitched_resized).permute(2, 0, 1).float()

        return stitched

    def _get_lidar_feature(self, lidars: List[Any]) -> torch.Tensor:
        """
        Process LiDAR data into BEV histogram.
        
        Note: Bench2Drive provides 85m x 85m LiDAR coverage, but DiffusionDrive
        expects 64m x 64m. We first process the full 85m range to preserve
        maximum information, then resize to 64m (256x256) for compatibility.

        Args:
            lidars: List of LiDAR objects (we use the last/current one)

        Returns:
            BEV histogram tensor [1, 256, 256]
        """
        # Get current frame LiDAR
        current_lidar = lidars[-1].lidar_pc  # [N, 4] (x, y, z, intensity)

        if len(current_lidar) == 0:
            # Return empty BEV if no points
            return torch.zeros((1, DIFFUSIONDRIVE_LIDAR_SIZE, DIFFUSIONDRIVE_LIDAR_SIZE), dtype=torch.float32)

        # Extract x, y coordinates (keep in CARLA coordinates)
        points_xy = current_lidar[:, :2]

        # Step 1: Process at Bench2Drive's native 85m x 85m resolution
        # This preserves all the LiDAR information from the original data
        pixel_coords_full = np.zeros_like(points_xy)
        pixel_coords_full[:, 0] = (points_xy[:, 0] + BENCH2DRIVE_LIDAR_RANGE_M / 2) * LIDAR_PIXELS_PER_METER
        pixel_coords_full[:, 1] = (points_xy[:, 1] + BENCH2DRIVE_LIDAR_RANGE_M / 2) * LIDAR_PIXELS_PER_METER

        # Clip to valid range for full resolution
        pixel_coords_full = np.clip(pixel_coords_full, 0, BENCH2DRIVE_LIDAR_SIZE - 1).astype(np.int32)

        # Create histogram at full resolution (340x340 for 85m at 4 pixels/meter)
        hist_full = np.zeros((BENCH2DRIVE_LIDAR_SIZE, BENCH2DRIVE_LIDAR_SIZE), dtype=np.float32)
        
        # Count points in each pixel using vectorized operation
        np.add.at(hist_full, (pixel_coords_full[:, 1], pixel_coords_full[:, 0]), 1)

        # Step 2: Resize to DiffusionDrive's expected 64m x 64m (256x256)
        # Using PIL for high-quality downsampling with LANCZOS filter
        hist_image = Image.fromarray(hist_full)
        hist_resized = hist_image.resize(
            (DIFFUSIONDRIVE_LIDAR_SIZE, DIFFUSIONDRIVE_LIDAR_SIZE), 
            Image.LANCZOS
        )
        hist = np.array(hist_resized, dtype=np.float32)
        
        # Ensure non-negative values after resize (LANCZOS can produce small negative values)
        hist = np.maximum(hist, 0.0)

        # Normalize using log scale and clip
        # This helps handle the dynamic range of point counts
        hist = np.log1p(hist)  # log(1 + x) to handle zeros
        hist = hist / LIDAR_NORMALIZATION_FACTOR  # Normalize
        hist = np.clip(hist, 0.0, 1.0)  # Clip to [0, 1]

        # Convert to tensor and add channel dimension
        bev_tensor = torch.from_numpy(hist).unsqueeze(0)  # [1, H, W]

        return bev_tensor.float()

    def _get_status_feature(self, ego_statuses: List[EgoStatus]) -> torch.Tensor:
        """
        Process ego status into feature vector.

        Args:
            ego_statuses: List of EgoStatus objects (we use the last/current one)

        Returns:
            Status feature tensor [8] containing:
            [one_hot_command(4), velocity(2), acceleration(2)]
        """
        # Get current status
        current_status = ego_statuses[-1]

        # Convert driving command to one-hot
        # driving_command is already a numpy array, extract the scalar value
        command_value = int(current_status.driving_command[0])
        command_one_hot = F.one_hot(
            torch.tensor(command_value, dtype=torch.long), num_classes=4
        ).float()

        # Get velocity and acceleration
        velocity = torch.tensor(current_status.ego_velocity, dtype=torch.float32)
        acceleration = torch.tensor(current_status.ego_acceleration, dtype=torch.float32)

        # Concatenate all features
        status_feature = torch.cat([command_one_hot, velocity, acceleration])  # [4]  # [2]  # [2]

        return status_feature

    @property
    def required_features(self) -> List[str]:
        """List of required feature names."""
        return ["camera_feature", "lidar_feature", "status_feature"]


class Bench2DriveTargetBuilder(AbstractTargetBuilder):
    """
    Target builder for Bench2Drive data.
    Builds training targets from scene data.
    """

    def __init__(self, config: TransfuserConfig):
        """
        Initialize target builder.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        """
        Build targets from Bench2Drive scene.

        Args:
            scene: Bench2DriveScene object

        Returns:
            Dictionary with trajectory, agent_states, agent_labels, bev_semantic_map
        """
        targets = {}

        # Get future trajectory
        targets["trajectory"] = scene.get_future_trajectory()

        # Get agent states and labels
        agent_states, agent_labels = scene.get_agents()
        targets["agent_states"] = agent_states
        targets["agent_labels"] = agent_labels

        # Get BEV semantic map
        targets["bev_semantic_map"] = scene.get_bev_semantic_map()

        return targets

    @property
    def required_targets(self) -> List[str]:
        """List of required target names."""
        return ["trajectory", "agent_states", "agent_labels", "bev_semantic_map"]
