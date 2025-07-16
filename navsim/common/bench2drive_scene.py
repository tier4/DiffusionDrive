"""
Bench2Drive scene representation for CARLA-native training.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import laspy
import numpy as np
import torch

from navsim.common.dataclasses import AgentInput, EgoStatus
from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig,
    load_bench2drive_annotation,
    map_carla_command_to_discrete,
)


# Import the real dataclasses
from navsim.common.dataclasses import Camera as CameraDataclass, Cameras, Lidar


# Simple wrapper classes for Bench2Drive
class Camera:
    """Simple camera wrapper for Bench2Drive that provides images tensor."""

    def __init__(self, images):
        self.images = images


class LiDAR:
    """Simple LiDAR wrapper for Bench2Drive that provides point cloud tensor."""

    def __init__(self, lidar_pc):
        self.lidar_pc = lidar_pc


class Bench2DriveScene:
    """
    Scene representation for Bench2Drive dataset.
    Implements CARLA-native loading without coordinate transformations.
    """

    def __init__(
        self,
        scene_info: Dict,
        config: Bench2DriveConfig,
        planner: Optional[any] = None,
        trajectory_sampling: Optional[any] = None,
    ):
        """
        Initialize Bench2Drive scene.

        Args:
            scene_info: Scene metadata from loader
            config: Dataset configuration
            planner: Optional planner for expert demonstrations
            trajectory_sampling: Trajectory sampling configuration
        """
        self.scene_info = scene_info
        self.config = config
        self.planner = planner
        self.trajectory_sampling = trajectory_sampling

        # Cache for loaded data
        self._annotations_cache = {}
        self._sensor_data_cache = {}

        # Frame information
        self.frames = scene_info["frames"]
        self.base_path = scene_info["base_path"]

        # Calculate frame indices
        self.num_frames = len(self.frames)
        self.history_frames = config.num_history_frames
        self.future_frames = config.num_future_frames

        # Token for compatibility
        self.token = scene_info.get(
            "token", f"{scene_info['scenario']}_{scene_info['start_idx']:05d}"
        )

    def get_agent_input(self, frame_idx: int = -1) -> AgentInput:
        """
        Get agent input for a specific frame.

        Args:
            frame_idx: Frame index (-1 for last frame)

        Returns:
            AgentInput object with cameras, lidars, and ego statuses
        """
        # Default to last frame (current frame in sequence)
        if frame_idx == -1:
            frame_idx = self.history_frames

        # Prepare lists for multi-frame data
        cameras_list = []
        lidars_list = []
        ego_statuses_list = []

        # Load data for history frames including current frame
        # If history_frames=4, we want frames at indices [frame_idx-3, frame_idx-2, frame_idx-1, frame_idx]
        for i in range(max(0, frame_idx - self.history_frames + 1), frame_idx + 1):
            # Load annotation
            anno = self._load_annotation(i)

            # Load sensor data
            cameras = self._load_cameras(i, anno)
            lidar = self._load_lidar(i, anno)
            ego_status = self._extract_ego_status(anno)

            cameras_list.append(cameras)
            lidars_list.append(lidar)
            ego_statuses_list.append(ego_status)

        return AgentInput(
            ego_statuses=ego_statuses_list,
            cameras=cameras_list,
            lidars=lidars_list,
        )

    def _load_annotation(self, frame_idx: int) -> Dict:
        """Load annotation for a specific frame."""
        if frame_idx in self._annotations_cache:
            return self._annotations_cache[frame_idx]

        anno_path = self.frames[frame_idx]
        anno = load_bench2drive_annotation(anno_path)

        self._annotations_cache[frame_idx] = anno
        return anno

    def _load_cameras(self, frame_idx: int, anno: Dict) -> Cameras:
        """
        Load camera data for a frame.

        Returns Camera object with 8 views (matching NavSim format).
        """
        # Extract frame number from annotation filename (e.g., "00000.json.gz" -> "00000")
        frame_id = self.frames[frame_idx].stem.split(".")[0]  # e.g., "00000"
        camera_base = self.base_path / "camera"

        # Map NavSim 8-camera format to actual Bench2Drive camera names
        # Bench2Drive has 6 cameras: rgb_front, rgb_front_left, rgb_front_right, rgb_back, rgb_back_left, rgb_back_right
        # NavSim expects 8 cameras, so we duplicate some views
        camera_mapping = {
            "CAM_FRONT": "rgb_front",
            "CAM_FRONT_LEFT": "rgb_front_left",
            "CAM_SIDE_LEFT": "rgb_front_left",  # Duplicate front-left for missing side-left
            "CAM_BACK_LEFT": "rgb_back_left",
            "CAM_BACK": "rgb_back",
            "CAM_BACK_RIGHT": "rgb_back_right",
            "CAM_SIDE_RIGHT": "rgb_front_right",  # Duplicate front-right for missing side-right
            "CAM_FRONT_RIGHT": "rgb_front_right",
        }

        images = []
        for navsim_name in [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_SIDE_LEFT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
            "CAM_SIDE_RIGHT",
            "CAM_FRONT_RIGHT",
        ]:
            b2d_name = camera_mapping[navsim_name]
            img_path = camera_base / b2d_name / f"{frame_id}.jpg"

            if img_path.exists():
                # Load image using PIL (loads in RGB format)
                img_pil = Image.open(img_path)

                # Apply same JPEG compression as training to avoid train-val gap
                # From anno.md: JPG compressed with quality 20
                # Save to buffer with quality 20 and reload
                from io import BytesIO

                buffer = BytesIO()
                img_pil.save(buffer, format="JPEG", quality=20)
                buffer.seek(0)
                img_pil = Image.open(buffer)

                # Convert to numpy array (RGB format)
                img = np.array(img_pil)
            else:
                # Create placeholder for missing camera
                img = np.zeros((900, 1600, 3), dtype=np.uint8)

            # Keep image as numpy array in HWC format
            images.append(img)

        # Create individual Camera objects for each view
        camera_objects = []
        for img in images:
            # Create Camera dataclass with the numpy image
            cam = CameraDataclass(image=img)
            camera_objects.append(cam)

        # Create Cameras object with all 8 views
        # Mapping based on the order in the loop above:
        # 0: CAM_FRONT -> rgb_front
        # 1: CAM_FRONT_LEFT -> rgb_front_left
        # 2: CAM_SIDE_LEFT -> rgb_front_left (duplicate)
        # 3: CAM_BACK_LEFT -> rgb_back_left
        # 4: CAM_BACK -> rgb_back
        # 5: CAM_BACK_RIGHT -> rgb_back_right
        # 6: CAM_SIDE_RIGHT -> rgb_front_right (duplicate)
        # 7: CAM_FRONT_RIGHT -> rgb_front_right
        return Cameras(
            cam_f0=camera_objects[0],  # Front
            cam_l0=camera_objects[1],  # Front-left
            cam_l1=camera_objects[2],  # Side-left (duplicate of front-left)
            cam_l2=camera_objects[3],  # Back-left
            cam_b0=camera_objects[4],  # Back
            cam_r2=camera_objects[5],  # Back-right
            cam_r1=camera_objects[6],  # Side-right (duplicate of front-right)
            cam_r0=camera_objects[7],  # Front-right
        )

    def _load_lidar(self, frame_idx: int, anno: Dict) -> Lidar:
        """
        Load LiDAR data for a frame.

        Returns LiDAR object with point cloud.
        """
        # Extract frame number from annotation filename (e.g., "00000.json.gz" -> "00000")
        frame_id = self.frames[frame_idx].stem.split(".")[0]
        lidar_path = self.base_path / "lidar" / f"{frame_id}.laz"

        if lidar_path.exists():
            try:
                # Load LAZ file
                las = laspy.read(str(lidar_path))

                # Extract points - keep in CARLA coordinates (no transformation)
                points = np.vstack([las.x, las.y, las.z]).T

                # Add intensity if available
                if hasattr(las, "intensity"):
                    intensity = las.intensity.reshape(-1, 1)
                else:
                    intensity = np.zeros((len(points), 1))

                # Combine into point cloud [N, 4] (x, y, z, intensity)
                point_cloud = np.hstack([points, intensity]).astype(np.float32)
            except Exception as e:
                print(f"Warning: Failed to load LiDAR from {lidar_path}: {e}")
                print("Using empty point cloud as fallback")
                point_cloud = np.zeros((0, 4), dtype=np.float32)
        else:
            # Empty point cloud if file not found
            point_cloud = np.zeros((0, 4), dtype=np.float32)

        # Create Lidar object with numpy array (not tensor)
        return Lidar(lidar_pc=point_cloud)

    def _extract_ego_status(self, anno: Dict) -> EgoStatus:
        """
        Extract ego vehicle status from annotation.

        Returns EgoStatus object.
        """
        # Extract velocity (m/s)
        # Note: Speed is scalar, need to convert to vector
        speed = anno.get("speed", 0.0)
        theta = anno.get("theta", 0.0)  # In degrees

        # Convert to velocity components
        # Note: Keep in CARLA coordinates (no transformation)
        theta_rad = np.radians(theta)
        vx = speed * np.cos(theta_rad)
        vy = speed * np.sin(theta_rad)
        ego_velocity = np.array([vx, vy], dtype=np.float32)

        # Extract acceleration (m/s²)
        # Acceleration is a list [x, y, z] in the annotation
        accel = anno.get("acceleration", [0.0, 0.0, 0.0])
        ego_acceleration = np.array([accel[0], accel[1]], dtype=np.float32)

        # Extract driving command
        # Use command_near as it's for immediate navigation
        carla_command = anno.get("command_near", 3)  # Default to STRAIGHT
        driving_command = map_carla_command_to_discrete(carla_command)

        # Extract position (for trajectory)
        ego_x = anno.get("x", 0.0)
        ego_y = anno.get("y", 0.0)
        ego_heading = -np.radians(theta)  # CW to CCW conversion

        # Create ego pose (x, y, heading)
        ego_pose = np.array([ego_x, ego_y, ego_heading], dtype=np.float64)

        return EgoStatus(
            ego_pose=ego_pose,
            ego_velocity=ego_velocity,
            ego_acceleration=ego_acceleration,
            driving_command=np.array([driving_command], dtype=np.int32),
            in_global_frame=True,  # Bench2Drive data is in global frame
        )

    def get_future_trajectory(self, frame_idx: int = -1) -> torch.Tensor:
        """
        Get future trajectory for training targets.

        Args:
            frame_idx: Frame index (-1 for last history frame)

        Returns:
            Trajectory tensor [num_waypoints, 3] with (x, y, heading)
        """
        if frame_idx == -1:
            frame_idx = self.history_frames

        # Collect future positions
        trajectory = []

        # Get ego position at current frame
        current_anno = self._load_annotation(frame_idx)
        current_x = current_anno.get("x", 0.0)
        current_y = current_anno.get("y", 0.0)
        current_theta = -np.radians(current_anno.get("theta", 0.0))  # Convert to radians CCW

        # Sample future frames (every 5 frames = 0.5s at 2Hz after downsampling)
        for i in range(1, 9):  # 8 future waypoints
            future_idx = frame_idx + i

            if future_idx < len(self.frames):
                future_anno = self._load_annotation(future_idx)
                future_x = future_anno.get("x", 0.0)
                future_y = future_anno.get("y", 0.0)
                future_theta = -np.radians(future_anno.get("theta", 0.0))

                # Convert to ego-centric coordinates (no coordinate system transformation)
                # Just translate and rotate to ego frame
                dx = future_x - current_x
                dy = future_y - current_y

                # Rotate to ego frame
                cos_theta = np.cos(-current_theta)
                sin_theta = np.sin(-current_theta)

                ego_x = dx * cos_theta - dy * sin_theta
                ego_y = dx * sin_theta + dy * cos_theta
                ego_heading = future_theta - current_theta

                # Normalize heading to [-pi, pi]
                ego_heading = np.arctan2(np.sin(ego_heading), np.cos(ego_heading))

                trajectory.append([ego_x, ego_y, ego_heading])
            else:
                # Extrapolate last known position if we run out of frames
                if trajectory:
                    trajectory.append(trajectory[-1])
                else:
                    trajectory.append([0.0, 0.0, 0.0])

        return torch.tensor(trajectory, dtype=torch.float64)

    def get_agents(self, frame_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get other agents' states and labels.

        Returns:
            agent_states: Tensor [max_agents, 5] with (x, y, heading, length, width)
            agent_labels: Boolean tensor [max_agents] indicating valid agents
        """
        if frame_idx == -1:
            frame_idx = self.history_frames

        anno = self._load_annotation(frame_idx)

        # Extract ego position for ego-centric conversion
        ego_x = anno.get("x", 0.0)
        ego_y = anno.get("y", 0.0)
        ego_theta = -np.radians(anno.get("theta", 0.0))  # Convert to radians CCW

        # Process vehicles from bounding_boxes
        max_agents = 30  # From transfuser_config.py
        agent_states = np.zeros((max_agents, 5), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        bboxes = anno.get("bounding_boxes", {})
        agent_idx = 0

        # Process vehicles (excluding ego)
        for obj_type in ["vehicle", "pedestrian"]:
            if obj_type in bboxes:
                for obj in bboxes[obj_type]:
                    if agent_idx >= max_agents:
                        break

                    # Skip static vehicles or ego
                    if obj.get("state") == "static" or obj.get("id") == bboxes.get(
                        "ego_vehicle", {}
                    ).get("id"):
                        continue

                    # Extract position (keep in CARLA coordinates)
                    loc = obj.get("location", {})
                    obj_x = loc.get("x", 0.0)
                    obj_y = loc.get("y", 0.0)

                    # Convert to ego-centric
                    dx = obj_x - ego_x
                    dy = obj_y - ego_y

                    cos_theta = np.cos(-ego_theta)
                    sin_theta = np.sin(-ego_theta)

                    ego_centric_x = dx * cos_theta - dy * sin_theta
                    ego_centric_y = dx * sin_theta + dy * cos_theta

                    # Extract rotation and convert to ego-centric
                    rot = obj.get("rotation", {})
                    obj_yaw = -np.radians(rot.get("yaw", 0.0))  # Convert to radians CCW
                    ego_centric_yaw = obj_yaw - ego_theta

                    # Extract size (extent in CARLA is half-size)
                    extent = obj.get("extent", {})
                    length = 2 * extent.get("x", 2.5)  # Default car length
                    width = 2 * extent.get("y", 1.0)  # Default car width

                    # Store agent state
                    agent_states[agent_idx] = [
                        ego_centric_x,
                        ego_centric_y,
                        ego_centric_yaw,
                        length,
                        width,
                    ]
                    agent_labels[agent_idx] = True
                    agent_idx += 1

        return torch.from_numpy(agent_states), torch.from_numpy(agent_labels)

    def get_bev_semantic_map(self, frame_idx: int = -1) -> torch.Tensor:
        """
        Get BEV semantic segmentation map.

        Returns:
            BEV map tensor [H, W] with semantic labels
        """
        # Create BEV map with basic road layout
        # Classes: 0=background, 1=road, 2=walkway, 3=lane, 4=static, 5=vehicle, 6=pedestrian
        bev_map = np.zeros((128, 256), dtype=np.float32)

        # Add a simple road pattern (most areas are drivable)
        # This is a placeholder - real implementation would use semantic segmentation
        bev_map[20:108, 64:192] = 1.0  # Road area (main driving area)

        # Add some lane markings
        bev_map[50:58, 120:136] = 3.0  # Center lane

        # Add vehicles from agent detection
        agents, labels = self.get_agents(frame_idx)
        if isinstance(agents, torch.Tensor):
            agents = agents.numpy()
            labels = labels.numpy()

        # Project agents to BEV
        for i, (agent, valid) in enumerate(zip(agents, labels)):
            if valid:
                x, y, heading, length, width = agent
                # Convert to BEV pixel coordinates
                # BEV is 256x128, covering 64m x 32m
                # Center is at (128, 64)
                px = int(128 + x * 4)  # 4 pixels per meter
                py = int(64 - y * 4)  # Flip y axis

                # Simple box representation
                half_len = int(length * 2)  # 4 pixels per meter / 2
                half_wid = int(width * 2)

                # Clip to valid range
                x1 = max(0, px - half_wid)
                x2 = min(256, px + half_wid)
                y1 = max(0, py - half_len)
                y2 = min(128, py + half_len)

                if x2 > x1 and y2 > y1:
                    bev_map[y1:y2, x1:x2] = 5.0  # Vehicle class

        return torch.from_numpy(bev_map).float()

    def __len__(self) -> int:
        """Get number of frames in scene."""
        return self.num_frames
