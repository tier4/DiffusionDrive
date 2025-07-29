"""
Bench2Drive scene representation for CARLA-native training.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

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
from navsim.common.bench2drive_constants import (
    B2D_CLASS_TO_NAVSIM,
    BEV_SEMANTIC_HEIGHT,
    BEV_SEMANTIC_WIDTH,
    NUM_FUTURE_WAYPOINTS,
    MAX_AGENTS,
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
        # Find ego vehicle in bounding boxes for accurate heading
        ego_box = None
        bounding_boxes = anno.get("bounding_boxes", [])
        for box in bounding_boxes:
            if box.get("class") == "ego_vehicle":
                ego_box = box
                break

        # Extract position and heading from ego bounding box if available
        if ego_box:
            # Use bounding box location for position
            ego_x = ego_box["location"][0]
            ego_y = ego_box["location"][1]

            # Extract yaw from rotation (z-component)
            ego_yaw_degrees = ego_box["rotation"][2]
            ego_heading = -np.radians(ego_yaw_degrees)  # CW to CCW conversion

            # For velocity, use the correct heading
            theta_rad = np.radians(ego_yaw_degrees)
        else:
            # Fallback to annotation values if ego not found in bounding boxes
            ego_x = anno.get("x", 0.0)
            ego_y = anno.get("y", 0.0)
            theta = anno.get("theta", 0.0)  # In degrees
            ego_heading = -np.radians(theta)  # CW to CCW conversion
            theta_rad = np.radians(theta)

        # Extract velocity (m/s)
        speed = anno.get("speed", 0.0)
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

        # Get ego information at current frame
        current_anno = self._load_annotation(frame_idx)

        # Find ego vehicle in bounding boxes for accurate transformation
        ego_box = None
        world2ego_matrix = None
        for box in current_anno.get("bounding_boxes", []):
            if box.get("class") == "ego_vehicle":
                ego_box = box
                # Get world2ego transformation matrix
                world2ego_matrix = np.array(box.get("world2ego", np.eye(4).tolist()))
                break

        if ego_box:
            # Use ego bounding box information
            current_x = ego_box["location"][0]
            current_y = ego_box["location"][1]
            current_theta = -np.radians(ego_box["rotation"][2])  # CW to CCW
        else:
            # Fallback to annotation values
            current_x = current_anno.get("x", 0.0)
            current_y = current_anno.get("y", 0.0)
            current_theta = -np.radians(current_anno.get("theta", 0.0))

        # Sample future frames (every 5 frames = 0.5s at 2Hz after downsampling)
        for i in range(1, NUM_FUTURE_WAYPOINTS + 1):  # NUM_FUTURE_WAYPOINTS future waypoints
            future_idx = frame_idx + i

            if future_idx < len(self.frames):
                future_anno = self._load_annotation(future_idx)

                # Find future ego position
                future_ego_box = None
                for box in future_anno.get("bounding_boxes", []):
                    if box.get("class") == "ego_vehicle":
                        future_ego_box = box
                        break

                if future_ego_box:
                    future_x = future_ego_box["location"][0]
                    future_y = future_ego_box["location"][1]
                    future_theta = -np.radians(future_ego_box["rotation"][2])
                else:
                    future_x = future_anno.get("x", 0.0)
                    future_y = future_anno.get("y", 0.0)
                    future_theta = -np.radians(future_anno.get("theta", 0.0))

                # Convert to ego-centric coordinates using world2ego matrix if available
                if world2ego_matrix is not None:
                    # Transform future position using world2ego matrix
                    future_world = np.array([future_x, future_y, 0.0, 1.0])
                    future_ego = world2ego_matrix @ future_world
                    ego_x = future_ego[0]
                    ego_y = future_ego[1]
                else:
                    # Fallback to manual transformation
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

    def get_agents(self, frame_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get other agents' states and labels.

        Returns:
            agent_states: Tensor [max_agents, 5] with (x, y, heading, length, width)
            agent_labels: Boolean tensor [max_agents] indicating valid agents
            agent_types: Tensor [max_agents] with NavSim class IDs for BEV rendering
        """
        if frame_idx == -1:
            frame_idx = self.history_frames

        anno = self._load_annotation(frame_idx)

        # Find ego vehicle in bounding boxes for accurate transformation
        ego_box = None
        world2ego_matrix = None
        for box in anno.get("bounding_boxes", []):
            if box.get("class") == "ego_vehicle":
                ego_box = box
                # Get world2ego transformation matrix
                world2ego_matrix = np.array(box.get("world2ego", np.eye(4).tolist()))
                break

        if ego_box:
            # Use ego bounding box information
            ego_x = ego_box["location"][0]
            ego_y = ego_box["location"][1]
            ego_theta = -np.radians(ego_box["rotation"][2])  # CW to CCW
        else:
            # Fallback to annotation values
            ego_x = anno.get("x", 0.0)
            ego_y = anno.get("y", 0.0)
            ego_theta = -np.radians(anno.get("theta", 0.0))  # Convert to radians CCW

        # Process vehicles from bounding_boxes
        max_agents = MAX_AGENTS  # Maximum number of agents to track
        agent_states = np.zeros((max_agents, 5), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        agent_types = np.zeros(max_agents, dtype=np.int32)  # NavSim class IDs

        bboxes = anno.get("bounding_boxes", [])
        agent_idx = 0

        # Handle both list and dict formats
        if isinstance(bboxes, list):
            # List format (Bench2Drive mini dataset)
            for obj in bboxes:
                if agent_idx >= max_agents:
                    break

                # Get object class and map to NavSim type
                obj_class = obj.get("class", "")
                navsim_class = B2D_CLASS_TO_NAVSIM.get(obj_class, -1)

                # Skip ego vehicle or unmapped classes
                if navsim_class == -1:
                    continue

                # Only process vehicles (5) and pedestrians (6)
                if navsim_class in [5, 6]:
                    # Extract position
                    if isinstance(obj.get("location"), list):
                        obj_x, obj_y, obj_z = obj["location"]
                    else:
                        loc = obj.get("location", {})
                        obj_x = loc.get("x", 0.0)
                        obj_y = loc.get("y", 0.0)

                    # Convert to ego-centric using world2ego matrix if available
                    if world2ego_matrix is not None and "world2ego" in obj:
                        # Use object's world2ego matrix for transformation
                        obj_world = np.array([obj_x, obj_y, 0.0, 1.0])
                        obj_ego = world2ego_matrix @ obj_world
                        ego_centric_x = obj_ego[0]
                        ego_centric_y = obj_ego[1]
                    else:
                        # Fallback to manual transformation
                        dx = obj_x - ego_x
                        dy = obj_y - ego_y

                        cos_theta = np.cos(-ego_theta)
                        sin_theta = np.sin(-ego_theta)

                        ego_centric_x = dx * cos_theta - dy * sin_theta
                        ego_centric_y = dx * sin_theta + dy * cos_theta

                    # Filter by lidar range (32m)
                    distance = np.sqrt(ego_centric_x**2 + ego_centric_y**2)
                    if distance > 32.0:
                        continue

                    # Extract rotation and convert to ego-centric
                    if isinstance(obj.get("rotation"), list):
                        obj_yaw_degrees = obj["rotation"][2]
                    else:
                        obj_yaw_degrees = obj.get("rotation", {}).get("yaw", 0.0)
                    obj_yaw = -np.radians(obj_yaw_degrees)
                    ego_centric_yaw = obj_yaw - ego_theta

                    # Extract size or use defaults based on type
                    if navsim_class == 6:  # Pedestrian
                        length = 0.8  # Pedestrian length
                        width = 0.6  # Pedestrian width
                    else:  # Vehicle
                        length = 4.0  # Default car length
                        width = 1.8  # Default car width

                    # Store agent state
                    agent_states[agent_idx] = [
                        ego_centric_x,
                        ego_centric_y,
                        ego_centric_yaw,
                        length,
                        width,
                    ]
                    agent_labels[agent_idx] = True
                    agent_types[agent_idx] = navsim_class
                    agent_idx += 1

        elif isinstance(bboxes, dict):
            # Dict format (original expected format)
            for obj_type in ["vehicle", "pedestrian"]:
                navsim_class = 5 if obj_type == "vehicle" else 6
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

                        # Filter by lidar range (32m)
                        distance = np.sqrt(ego_centric_x**2 + ego_centric_y**2)
                        if distance > 32.0:
                            continue

                        # Extract rotation and convert to ego-centric
                        rot = obj.get("rotation", {})
                        obj_yaw = -np.radians(rot.get("yaw", 0.0))  # Convert to radians CCW
                        ego_centric_yaw = obj_yaw - ego_theta

                        # Extract size (extent in CARLA is half-size)
                        extent = obj.get("extent", {})
                        if navsim_class == 6:  # Pedestrian
                            length = min(2 * extent.get("x", 0.4), 0.8)  # Cap pedestrian size
                            width = min(2 * extent.get("y", 0.3), 0.6)
                        else:
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
                        agent_types[agent_idx] = navsim_class
                        agent_idx += 1

        return (
            torch.from_numpy(agent_states),
            torch.from_numpy(agent_labels),
            torch.from_numpy(agent_types),
        )

    def get_bev_semantic_map(self, frame_idx: int = -1) -> torch.Tensor:
        """
        Get BEV semantic segmentation map.

        Returns:
            BEV map tensor [H, W] with semantic labels in native NavSim format

        Note: Returns 128×256 format which is the original NavSim BEV semantic format.
        This rectangular shape matches the frontal RGB camera coverage area and
        optimizes for forward-driving scenarios where lateral awareness is more
        critical than forward/backward range.
        """
        from navsim.common.bev_semantic_utils import generate_simple_bev_semantic

        # Get trajectory for road generation
        trajectory = None
        if (
            hasattr(self, "_future_trajectory_cache")
            and frame_idx in self._future_trajectory_cache
        ):
            # Use cached trajectory if available
            traj_data = self._future_trajectory_cache[frame_idx]
            trajectory = traj_data  # Already in ego-centric coordinates
        else:
            # Try to generate trajectory from future frames
            try:
                future_positions = []
                for i in range(min(8, len(self.frames) - frame_idx - 1)):
                    future_frame = self._load_annotation(frame_idx + i + 1)
                    if future_frame is not None:
                        ego_x = future_frame.get("x", 0.0)
                        ego_y = future_frame.get("y", 0.0)
                        ego_theta = future_frame.get("theta", 0.0)

                        # Convert to ego-centric coordinates
                        current_frame = self._load_annotation(frame_idx)
                        if current_frame is not None:
                            curr_x = current_frame.get("x", 0.0)
                            curr_y = current_frame.get("y", 0.0)
                            curr_theta = current_frame.get("theta", 0.0)

                            # Transform to ego-centric
                            dx = ego_x - curr_x
                            dy = ego_y - curr_y
                            cos_theta = np.cos(curr_theta)
                            sin_theta = np.sin(curr_theta)

                            ego_centric_x = dx * cos_theta + dy * sin_theta
                            ego_centric_y = -dx * sin_theta + dy * cos_theta
                            ego_centric_heading = ego_theta - curr_theta

                            future_positions.append(
                                [ego_centric_x, ego_centric_y, ego_centric_heading]
                            )

                if future_positions:
                    trajectory = np.array(future_positions)
            except:
                # If trajectory generation fails, use None
                trajectory = None

        # Get agents for vehicle generation
        agents, labels, agent_types = self.get_agents(frame_idx)
        if isinstance(agents, torch.Tensor):
            agents = agents.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # Check if cached BEV exists
        map_bev = None
        if hasattr(self.config, "bev_cache_dir") and self.config.bev_cache_dir:
            cache_dir = Path(self.config.bev_cache_dir)
            scenario_name = self.scene_info["scenario"]  # Get scenario name
            frame_number = self.frames[frame_idx].stem.split(".")[0]
            cache_path = cache_dir / scenario_name / f"{frame_number}.npz"

            if cache_path.exists():
                try:
                    cached_data = np.load(cache_path)
                    # Use front_bev if available, otherwise use full_bev
                    if "front_bev" in cached_data:
                        map_bev = cached_data["front_bev"]
                    elif "full_bev" in cached_data:
                        # Extract front half
                        full_bev = cached_data["full_bev"]
                        map_bev = full_bev[:128, :]  # Front half
                    print(f"Loaded cached BEV from {cache_path}")
                except Exception as e:
                    print(f"Failed to load cached BEV: {e}")

        # If no cached BEV, try to generate from HD map if available
        if map_bev is None and hasattr(self.config, "map_dir") and self.config.map_dir:
            try:
                from navsim.common.bev_map_utils import load_map_data, generate_bev_from_map

                # Extract town name from scenario
                parts = self.scene_info["scenario"].split("_")
                town_name = None
                for part in parts:
                    if part.startswith("Town"):
                        town_name = part
                        break

                if town_name:
                    map_path = Path(self.config.map_dir) / f"{town_name}_HD_map.npz"
                    if map_path.exists():
                        # Load map data
                        map_data = load_map_data(map_path)

                        # Get world2ego transform
                        anno = self._load_annotation(frame_idx)
                        if anno and "bounding_boxes" in anno:
                            world2ego = np.array(anno["bounding_boxes"][0]["world2ego"])

                            # Generate BEV from map
                            map_bev = generate_bev_from_map(
                                map_data=map_data,
                                world2ego=world2ego,
                                bev_height=BEV_SEMANTIC_HEIGHT,
                                bev_width=BEV_SEMANTIC_WIDTH,
                                resolution=0.25,
                                lane_thickness=0.4,
                                max_distance=50.0,
                            )
                            print(f"Generated BEV from HD map for {town_name}")
            except Exception as e:
                print(f"Failed to generate BEV from map: {e}")

        # Generate BEV semantic map using the correct dataset-level approach
        bev_map = generate_simple_bev_semantic(
            trajectory=trajectory if map_bev is None else None,  # Only use trajectory if no map
            agents=agents,
            agent_labels=labels,
            agent_types=(
                agent_types if isinstance(agent_types, np.ndarray) else agent_types.numpy()
            ),
            bev_height=BEV_SEMANTIC_HEIGHT,
            bev_width=BEV_SEMANTIC_WIDTH,
            resolution=0.25,  # 0.25 meters per pixel
            map_bev=map_bev,  # Use map-based BEV if available
        )

        return torch.from_numpy(bev_map).float()

    def __len__(self) -> int:
        """Get number of frames in scene."""
        return self.num_frames
