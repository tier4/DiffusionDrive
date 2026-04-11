"""
Bench2Drive scene representation for CARLA-native training.
"""

from navsim.common.abstract_carla_scene import AbstractCarlaScene
from navsim.common.dataclasses import AgentInput, EgoStatus
from navsim.common.bench2drive_dataloader import (
    Bench2DriveDataConfig,
    load_bench2drive_annotation,
    map_carla_command_to_discrete,
)
from navsim.common.bev_semantic_utils import generate_simple_bev_semantic
from navsim.common.bench2drive_constants import (
    B2D_CLASS_TO_NAVSIM,
    BEV_SEMANTIC_HEIGHT,
    BEV_SEMANTIC_RANGE_M,
    BEV_SEMANTIC_WIDTH,
    BEV_SEMANTIC_RESOLUTION,
    NUM_FUTURE_WAYPOINTS,
    MAX_AGENTS,
    BENCH2DRIVE_LIDAR_RANGE_M,
    FUTURE_TRAJECTORY_FRAME_STRIDE,
)
from navsim.common.bev_map_utils import (
    transform_points_to_ego,
    transform_heading_to_ego,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle

# Import the real dataclasses
from navsim.common.dataclasses import Camera as CameraDataclass, Cameras, Lidar

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from PIL import Image
import laspy
import numpy as np
import torch

logger = logging.getLogger(__name__)


# Simple wrapper classes for Bench2Drive
class Camera:
    """Simple camera wrapper for Bench2Drive that provides images tensor."""

    def __init__(self, images):
        self.images = images


class LiDAR:
    """Simple LiDAR wrapper for Bench2Drive that provides point cloud tensor."""

    def __init__(self, lidar_pc):
        self.lidar_pc = lidar_pc


# TODO: should use coodinates system as ENUM
class Bench2DriveScene(AbstractCarlaScene):
    """
    Scene representation for Bench2Drive dataset.
    Implements CARLA-native loading without coordinate transformations.
    """

    def __init__(
        self,
        scene_info: Dict,
        config: Bench2DriveDataConfig,
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
        self.loader = scene_info.get("loader")  # Store loader reference
        self.scene_name = scene_info.get("scenario", "unknown")  # Store scene name

        # Cache for loaded data
        self._annotations_cache = {}
        self._sensor_data_cache = {}

        # Log sampling rate info once at initialization
        self._log_sampling_rate_once()

        # Frame information - handle both modes
        # DEPRECATED: 10Hz sliding window mode removed to simplify codebase
        # Keeping code commented for potential future revival
        # if config.sliding_mode and "all_frames" in scene_info:
        #     # True sliding window mode
        #     self.all_frames = scene_info["all_frames"]
        #     self.start_idx = scene_info["start_idx"]  # Current frame in 10Hz
        #     self.base_path = Path(scene_info["all_frames"][0]).parent.parent
        #     self.anno_paths = None  # Not used in sliding mode
        # else:
        #     # Legacy mode
        #     self.all_frames = None
        #     self.start_idx = scene_info.get("start_idx", 0)
        #     self.anno_paths = scene_info["frames"]
        #     self.base_path = scene_info["base_path"]

        # Use legacy mode only (2Hz pre-downsampled frames)
        self.all_frames = None
        self.start_idx = scene_info.get("start_idx", 0)
        self.anno_paths = scene_info["frames"]
        self.base_path = scene_info["base_path"]

        self.history_frames = config.num_history_frames
        self.future_frames = config.num_future_frames

        # Token for compatibility
        self.token = scene_info.get(
            "token", f"{scene_info['scenario']}_{self.start_idx:05d}"
        )

    def _log_sampling_rate_once(self):
        """Log sampling rate information once at scene initialization."""
        if not hasattr(self.config, 'sampling_rate'):
            return

        sampling_rate = self.config.sampling_rate
        if sampling_rate == 1:
            logger.info(
                f"Scene {self.scene_name}: Using 10Hz data (sampling_rate=1), "
                f"will skip 5 frames for 0.5s GT intervals"
            )
        elif sampling_rate == 5:
            logger.info(
                f"Scene {self.scene_name}: Using 2Hz data (sampling_rate=5), "
                f"consecutive frames already at 0.5s intervals"
            )

    def get_agent_input(self, frame_idx: int = -1) -> AgentInput:
        """
        Get agent input for a specific frame.

        Args:
            frame_idx: Frame index (-1 for last frame)

        Returns:
            AgentInput object with cameras, lidars, and ego statuses
        """
        # Default to current frame (frame 0 when no history, otherwise last history frame)
        if frame_idx == -1:
            frame_idx = max(0, self.history_frames - 1)

        # Prepare lists for multi-frame data
        cameras_list = []
        lidars_list = []
        ego_statuses_list = []

        # DEPRECATED: 10Hz sliding window mode removed
        # if self.config.sliding_mode and self.all_frames is not None:
        #     # TRUE SLIDING WINDOW MODE: On-demand downsampling
        #     # Current frame is at self.start_idx in 10Hz
        #     # When history_frames=0: Only load current frame at start_idx
        #     # When history_frames>0: Load history frames before start_idx
        #
        #     if self.history_frames == 0:
        #         # No history: only load current frame
        #         abs_idx = self.start_idx
        #         if abs_idx < 0 or abs_idx >= len(self.all_frames):
        #             raise ValueError(f"Invalid frame index {abs_idx} for scene {self.token}")
        #
        #         anno = self._load_annotation_absolute(abs_idx)
        #         cameras = self._load_cameras_absolute(abs_idx, anno)
        #         lidar = self._load_lidar_absolute(abs_idx, anno)
        #         ego_status = self._extract_ego_status(anno)
        #
        #         cameras_list.append(cameras)
        #         lidars_list.append(lidar)
        #         ego_statuses_list.append(ego_status)
        #     else:
        #         # Load history frames (including current frame)
        #         for hist_offset in range(self.history_frames - 1, -1, -1):
        #             # Get the 10Hz index for this history frame
        #             abs_idx = self.start_idx - (hist_offset * self.config.sampling_rate)
        #
        #             # Ensure index is valid
        #             if abs_idx < 0 or abs_idx >= len(self.all_frames):
        #                 raise ValueError(f"Invalid frame index {abs_idx} for scene {self.token}")
        #
        #             # Load annotation at this absolute index
        #             anno = self._load_annotation_absolute(abs_idx)
        #
        #             # Load sensor data
        #             cameras = self._load_cameras_absolute(abs_idx, anno)
        #             lidar = self._load_lidar_absolute(abs_idx, anno)
        #             ego_status = self._extract_ego_status(anno)
        #
        #             cameras_list.append(cameras)
        #             lidars_list.append(lidar)
        #             ego_statuses_list.append(ego_status)
        # else:

        # LEGACY MODE: Use pre-downsampled frames
        # When history_frames=0: Only load current frame (frame_idx)
        # When history_frames>0: Load history frames including current
        if self.history_frames == 0:
            # Only load current frame
            i = frame_idx
            anno = self._load_annotation(i)
            cameras = self._load_cameras(i, anno)
            lidar = self._load_lidar(i, anno)
            ego_status = self._extract_ego_status(anno)

            cameras_list.append(cameras)
            lidars_list.append(lidar)
            ego_statuses_list.append(ego_status)
        else:
            # Load history frames including current frame
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

        # TODO: need to add a check for empty images or status here
        return AgentInput(
            ego_statuses=ego_statuses_list,
            cameras=cameras_list,
            lidars=lidars_list,
        )

    # TODO: REFACTOR - These two methods are nearly identical, differing only in:
    #   1. Path source: self.anno_paths[frame_idx] vs self.all_frames[abs_idx]
    #   2. Index type: relative vs absolute
    # Should be merged into a single method like:
    #   def _load_annotation(self, idx: int, absolute: bool = False) -> Dict
    # This would eliminate code duplication and make the code more maintainable.

    def _load_annotation(self, frame_idx: int) -> Dict:
        """Load annotation for a specific frame (legacy mode)."""
        if frame_idx in self._annotations_cache:
            return self._annotations_cache[frame_idx]

        anno_path = self.anno_paths[frame_idx]
        anno = load_bench2drive_annotation(anno_path)

        self._annotations_cache[frame_idx] = anno
        return anno

    # DEPRECATED: 10Hz sliding window mode removed
    # def _load_annotation_absolute(self, abs_idx: int) -> Dict:
    #     """Load annotation for an absolute frame index (sliding window mode)."""
    #     if abs_idx in self._annotations_cache:
    #         return self._annotations_cache[abs_idx]
    #
    #     anno_path = self.all_frames[abs_idx]
    #     anno = load_bench2drive_annotation(anno_path)
    #
    #     self._annotations_cache[abs_idx] = anno
    #     return anno

    def _load_cameras(self, frame_idx: int, anno: Dict) -> Cameras:
        """
        Load camera data for a frame.

        Returns Camera object with 8 views (matching NavSim format).
        """
        # Extract frame number from annotation filename (e.g., "00000.json.gz" -> "00000")
        frame_id = self.anno_paths[frame_idx].stem.split(".")[0]  # e.g., "00000"
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
                # Missing camera data should not be silently ignored
                raise FileNotFoundError(
                    f"Camera image not found at {img_path}. "
                    f"Missing sensor data should not be replaced with fake values."
                )

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

    # DEPRECATED: 10Hz sliding window mode removed
    # def _load_cameras_absolute(self, abs_idx: int, anno: Dict) -> Cameras:
    #     """
    #     Load camera data for an absolute frame index (sliding window mode).
    #
    #     Returns Camera object with 8 views (matching NavSim format).
    #     """
    #     # Extract frame number from annotation filename
    #     frame_path = self.all_frames[abs_idx]
    #     frame_id = frame_path.stem.split(".")[0]  # e.g., "00000"
    #     camera_base = self.base_path / "camera"
    #
    #     # Map NavSim 8-camera format to actual Bench2Drive camera names
    #     camera_mapping = {
    #         "CAM_FRONT": "rgb_front",
    #         "CAM_FRONT_LEFT": "rgb_front_left",
    #         "CAM_SIDE_LEFT": "rgb_front_left",  # Duplicate front-left for missing side-left
    #         "CAM_BACK_LEFT": "rgb_back_left",
    #         "CAM_BACK": "rgb_back",
    #         "CAM_BACK_RIGHT": "rgb_back_right",
    #         "CAM_SIDE_RIGHT": "rgb_front_right",  # Duplicate front-right for missing side-right
    #         "CAM_FRONT_RIGHT": "rgb_front_right",
    #     }
    #
    #     images = []
    #     for navsim_name in [
    #         "CAM_FRONT",
    #         "CAM_FRONT_LEFT",
    #         "CAM_SIDE_LEFT",
    #         "CAM_BACK_LEFT",
    #         "CAM_BACK",
    #         "CAM_BACK_RIGHT",
    #         "CAM_SIDE_RIGHT",
    #         "CAM_FRONT_RIGHT",
    #     ]:
    #         b2d_name = camera_mapping[navsim_name]
    #         img_path = camera_base / b2d_name / f"{frame_id}.jpg"
    #
    #         if img_path.exists():
    #             # Load image using PIL (loads in RGB format)
    #             img_pil = Image.open(img_path)
    #
    #             # Apply same JPEG compression as training to avoid train-val gap
    #             from io import BytesIO
    #
    #             buffer = BytesIO()
    #             img_pil.save(buffer, format="JPEG", quality=20)
    #             buffer.seek(0)
    #             img_pil = Image.open(buffer)
    #
    #             # Convert to numpy array (RGB format)
    #             img = np.array(img_pil)
    #         else:
    #             # Missing camera data should not be silently ignored
    #             raise FileNotFoundError(
    #                 f"Camera image not found at {img_path}. "
    #                 f"Missing sensor data should not be replaced with fake values."
    #             )
    #
    #         images.append(img)
    #
    #     # Create individual Camera objects for each view
    #     camera_objects = []
    #     for img in images:
    #         cam = CameraDataclass(image=img)
    #         camera_objects.append(cam)
    #
    #     return Cameras(
    #         cam_f0=camera_objects[0],  # Front
    #         cam_l0=camera_objects[1],  # Front-left
    #         cam_l1=camera_objects[2],  # Side-left (duplicate of front-left)
    #         cam_l2=camera_objects[3],  # Back-left
    #         cam_b0=camera_objects[4],  # Back
    #         cam_r2=camera_objects[5],  # Back-right
    #         cam_r1=camera_objects[6],  # Side-right (duplicate of front-right)
    #         cam_r0=camera_objects[7],  # Front-right
    #     )

    def _load_lidar(self, frame_idx: int, anno: Dict) -> Lidar:
        """
        Load LiDAR data for a frame.

        Returns LiDAR object with point cloud.
        """
        # Extract frame number from annotation filename (e.g., "00000.json.gz" -> "00000")
        frame_id = self.anno_paths[frame_idx].stem.split(".")[0]
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
                # LiDAR loading errors should not be silently ignored
                raise RuntimeError(
                    f"Failed to load LiDAR data from {lidar_path}: {e}. "
                    f"Corrupted sensor data should not be replaced with fake values."
                )
        else:
            # Missing LiDAR data should not be silently ignored
            raise FileNotFoundError(
                f"LiDAR file not found at {lidar_path}. "
                f"Missing sensor data should not be replaced with fake values."
            )

        # Create Lidar object with numpy array (not tensor)
        return Lidar(lidar_pc=point_cloud)

    # DEPRECATED: 10Hz sliding window mode removed
    # def _load_lidar_absolute(self, abs_idx: int, anno: Dict) -> Lidar:
    #     """
    #     Load LiDAR data for an absolute frame index (sliding window mode).
    #
    #     Returns LiDAR object with point cloud.
    #     """
    #     # Extract frame number from annotation filename
    #     frame_path = self.all_frames[abs_idx]
    #     frame_id = frame_path.stem.split(".")[0]
    #     lidar_path = self.base_path / "lidar" / f"{frame_id}.laz"
    #
    #     if lidar_path.exists():
    #         try:
    #             # Load LAZ file
    #             las = laspy.read(str(lidar_path))
    #
    #             # Extract points - keep in CARLA coordinates (no transformation)
    #             points = np.vstack([las.x, las.y, las.z]).T
    #
    #             # Add intensity if available
    #             if hasattr(las, "intensity"):
    #                 intensity = las.intensity.reshape(-1, 1)
    #             else:
    #                 intensity = np.zeros((len(points), 1))
    #
    #             # Combine into point cloud [N, 4] (x, y, z, intensity)
    #             point_cloud = np.hstack([points, intensity]).astype(np.float32)
    #         except Exception as e:
    #             # LiDAR loading errors should not be silently ignored
    #             raise RuntimeError(
    #                 f"Failed to load LiDAR data from {lidar_path}: {e}. "
    #                 f"Corrupted sensor data should not be replaced with fake values."
    #             )
    #     else:
    #         # Missing LiDAR data should not be silently ignored
    #         raise FileNotFoundError(
    #             f"LiDAR file not found at {lidar_path}. "
    #             f"Missing sensor data should not be replaced with fake values."
    #         )
    #
    #     # Create Lidar object with numpy array (not tensor)
    #     return Lidar(lidar_pc=point_cloud)

    def _extract_ego_status(self, anno: Dict) -> EgoStatus:
        """
        Extract ego vehicle status from annotation.

        Returns EgoStatus object.
        """
        # Find ego vehicle in bounding boxes - REQUIRED
        ego_box = None
        bounding_boxes = anno["bounding_boxes"]
        for box in bounding_boxes:
            if box["class"] == "ego_vehicle":
                ego_box = box
                break

        if ego_box is None:
            raise ValueError(f"Ego vehicle not found in bounding boxes for frame")

        # Use bounding box center for position
        ego_x = ego_box["center"][0]
        ego_y = ego_box["center"][1]

        # Extract yaw from rotation (z-component)
        ego_yaw_degrees = ego_box["rotation"][2]
        ego_heading = np.radians(ego_yaw_degrees)  # Keep CARLA CW convention

        # For velocity, use the correct heading
        theta_rad = np.radians(ego_yaw_degrees)

        # Extract velocity (m/s)
        speed = anno["speed"]
        vx = speed * np.cos(theta_rad)
        vy = speed * np.sin(theta_rad)  # CARLA left-handed: +Y right, CW heading positive
        ego_velocity = np.array([vx, vy], dtype=np.float32)

        # Extract acceleration (m/s²)
        # Acceleration is a list [x, y, z] in the annotation
        accel = anno["acceleration"]
        ego_acceleration = np.array([accel[0], accel[1]], dtype=np.float32)

        # Extract driving command
        # Use command_near as it's for immediate navigation
        carla_command = anno["command_near"]
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
            # Use frame 0 when no history, otherwise use last history frame
            frame_idx = max(0, self.history_frames - 1)

        # Collect future positions
        trajectory = []

        # Get ego information at current frame
        # DEPRECATED: 10Hz sliding window mode removed
        # if self.config.sliding_mode and self.all_frames is not None:
        #     # True sliding window mode - current frame is at start_idx
        #     current_anno = self._load_annotation_absolute(self.start_idx)
        # else:
        #     # Legacy mode
        #     current_anno = self._load_annotation(frame_idx)

        # Use legacy mode only
        current_anno = self._load_annotation(frame_idx)

        # Find ego vehicle in bounding boxes - REQUIRED
        ego_box = None
        for box in current_anno["bounding_boxes"]:
            if box["class"] == "ego_vehicle":
                ego_box = box
                break

        if ego_box is None:
            raise ValueError(
                f"Ego vehicle not found in bounding boxes for trajectory frame {frame_idx}"
            )

        # Get current heading with normalization
        ego_heading_deg = ego_box["rotation"][2]
        ego_heading_rad = normalize_angle(np.radians(ego_heading_deg))

        # Get current position
        ego_position_world = np.array(ego_box["center"])  # [x, y, z] in world coordinates

        # Sample future frames at 0.5s intervals (model expectation)
        # CRITICAL: Always use FUTURE_TRAJECTORY_FRAME_STRIDE for GT trajectory
        # Even when evaluating at 10Hz (sampling_rate=1), GT must be at 2Hz
        sampling_rate = self.config.sampling_rate if hasattr(self.config, 'sampling_rate') else 5

        # DEPRECATED: 10Hz sliding window mode removed
        # if self.config.sliding_mode and self.all_frames is not None:
        #     # TRUE SLIDING WINDOW MODE code removed - see git history
        #     pass
        # else:

        # LEGACY MODE: Use pre-downsampled frames
        if sampling_rate == 1:
            # 10Hz evaluation mode: raw data at 0.1s intervals
            # CRITICAL FIX: Always use FUTURE_TRAJECTORY_FRAME_STRIDE for 2Hz GT
            frame_stride = FUTURE_TRAJECTORY_FRAME_STRIDE  # Always 5 frames for 0.5s intervals
        elif sampling_rate == 5:
            # 2Hz training mode: pre-sampled data already at 0.5s intervals
            frame_stride = 1  # Use consecutive frames (already at 0.5s)
        else:
            raise ValueError(
                f"Unsupported sampling_rate={sampling_rate}. "
                f"DiffusionDrive only supports:\n"
                f"  - sampling_rate=1 (10Hz evaluation data)\n"
                f"  - sampling_rate=5 (2Hz training data)\n"
                f"Model expects 0.5s intervals between trajectory points."
            )

        # Limit to available frames to prevent out-of-bounds errors
        max_future_frames = min(
            NUM_FUTURE_WAYPOINTS,
            (len(self.anno_paths) - frame_idx - 1) // frame_stride
        )

        for i in range(1, max_future_frames + 1):
            future_idx = frame_idx + (i * frame_stride)

            # Double-check bounds (should not fail with the limit above)
            if future_idx >= len(self.anno_paths):
                logger.warning(
                    f"Stopping trajectory at {len(trajectory)} waypoints "
                    f"due to insufficient future frames"
                )
                break

            future_anno = self._load_annotation(future_idx)

            # Find future ego position
            future_ego_box = None
            for box in future_anno["bounding_boxes"]:
                if box["class"] == "ego_vehicle":
                    future_ego_box = box
                    break

            if future_ego_box is None:
                raise ValueError(f"Ego vehicle not found in future frame {future_idx}")

            # 2. Transform Heading using standardized function with normalization
            future_heading_deg = future_ego_box["rotation"][2]
            future_heading_rad = normalize_angle(np.radians(future_heading_deg))
            future_heading_in_ego = transform_heading_to_ego(
                future_heading_rad, ego_heading_rad, normalize=True
            )

            # transform to ego position in carla
            # I did not using the world2ego cause after investigation
            # confirm that the accuracy is bad
            future_world_position = np.array(future_ego_box["center"])
            future_x_in_ego, future_y_in_ego, _ = transform_points_to_ego(
                points=future_world_position,
                ego_points=ego_position_world,
                ego_heading_rad=ego_heading_rad,
            ).tolist()[0]

            trajectory.append([future_x_in_ego, future_y_in_ego, future_heading_in_ego])

        # Return None if we don't have enough future frames
        # This prevents training on fake/padded data
        if len(trajectory) < NUM_FUTURE_WAYPOINTS:
            logger.debug(
                f"Frame {frame_idx}: Only {len(trajectory)}/{NUM_FUTURE_WAYPOINTS} "
                f"future waypoints available. Returning None to skip this sample."
            )
            return None

        return torch.tensor(trajectory, dtype=torch.float64)

    def get_agents(self, frame_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get other agents' states and labels.

        Returns:
            agent_states: Tensor [max_agents, 5] with (x, y, heading, length, width)
            agent_labels: Boolean tensor [max_agents] indicating valid agents
            agent_types: Tensor [max_agents] with NavSim class IDs for BEV rendering

        Key Difference from NAVSIM Implementation:
        -------------------------------------------
        NAVSIM: Uses rectangular/square filtering (-32m to +32m in X/Y axes)
                This creates a 64m x 64m square region centered on ego
                Naturally emphasizes frontal and side views

        B2D:    Uses circular/radial filtering (42.5m radius, 360° coverage)
                Filters agents by Euclidean distance from ego vehicle
                Provides uniform coverage in all directions
                Better for complex scenarios requiring full situational awareness
        """
        if frame_idx == -1:
            # Use frame 0 when no history, otherwise use last history frame
            frame_idx = max(0, self.history_frames - 1)

        # Load annotation based on mode
        # DEPRECATED: 10Hz sliding window mode removed
        # if self.config.sliding_mode and self.all_frames is not None:
        #     # True sliding window mode - use absolute index
        #     anno = self._load_annotation_absolute(self.start_idx)
        # else:
        #     # Legacy mode - use relative frame index
        #     anno = self._load_annotation(frame_idx)

        # Use legacy mode only
        anno = self._load_annotation(frame_idx)

        # Find ego vehicle in bounding boxes - REQUIRED
        ego_box = None
        for box in anno["bounding_boxes"]:
            if box["class"] == "ego_vehicle":
                ego_box = box
                break

        if ego_box is None:
            raise ValueError(
                f"Ego vehicle not found in bounding boxes for agents frame {frame_idx}"
            )

        # Get ego position and heading for coordinate transforms
        ego_points = np.array(ego_box["center"])  # [x, y, z] in world coordinates
        ego_heading_deg = ego_box["rotation"][2]  # Current ego heading in degrees
        ego_heading_rad = normalize_angle(np.radians(ego_heading_deg))

        # Process vehicles from bounding_boxes
        max_agents = MAX_AGENTS  # Maximum number of agents to track
        agent_states = np.zeros((max_agents, 5), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        agent_types = np.zeros(max_agents, dtype=np.int32)  # NavSim class IDs

        bboxes = anno["bounding_boxes"]  # Required field - fail fast if missing
        agent_idx = 0

        # Normalize bboxes format and process objects uniformly
        objects_to_process = []

        if isinstance(bboxes, list):
            # List format (Bench2Drive mini dataset)
            for obj in bboxes:
                obj_class = obj["class"]  # Required field - fail fast if missing

                # Skip ego vehicle and traffic elements
                if obj_class == "ego_vehicle":
                    continue
                if obj_class in ["traffic_light", "traffic_sign"]:
                    continue  # Already drawn on static BEV maps

                # Only process vehicles and pedestrians (dynamic objects)
                if obj_class in ["vehicle", "walker"]:
                    navsim_class = B2D_CLASS_TO_NAVSIM[obj_class]
                    objects_to_process.append((obj, obj_class, navsim_class))

        elif isinstance(bboxes, dict):
            # Dict format (original expected format)
            ego_id = bboxes.get("ego_vehicle", {}).get("id")
            for obj_type in ["vehicle", "walker"]:  # Fixed: pedestrians are "walker"
                if obj_type in bboxes:
                    navsim_class = B2D_CLASS_TO_NAVSIM[obj_type]
                    for obj in bboxes[obj_type]:
                        # Skip ego vehicle
                        if obj.get("id") == ego_id:
                            continue
                        objects_to_process.append((obj, obj_type, navsim_class))

        # Process all objects with unified logic
        for obj, obj_class, navsim_class in objects_to_process:
            if agent_idx >= max_agents:
                break

            # Use center coordinates for reliable world position (comprehensive fix plan)
            obj_center = obj["center"]  # [x, y, z] in world coordinates
            obj_world_pos = np.array([[obj_center[0], obj_center[1], obj_center[2]]])

            # Transform to ego coordinates using standard function
            ego_coords = transform_points_to_ego(
                obj_world_pos, ego_points, ego_heading_rad, left_to_right=False
            )
            ego_centric_x = ego_coords[0, 0]
            ego_centric_y = ego_coords[0, 1]

            # Use annotation distance instead of manual calculation
            distance = obj["distance"]  # Pre-calculated distance to ego
            # Circular filtering at 32m radius (BENCH2DRIVE_LIDAR_RANGE_M/2 = 64/2 = 32m)
            # Comparable to NAVSIM's 32m square region
            if distance > BENCH2DRIVE_LIDAR_RANGE_M / 2:
                continue

            # Extract rotation and convert to ego-centric using simpler angle subtraction
            rotation = obj["rotation"]  # Required field - fail fast if missing
            if isinstance(rotation, list):
                obj_yaw_degrees = rotation[2]  # Yaw is at index 2: [pitch, roll, yaw]
            else:
                obj_yaw_degrees = rotation["yaw"]  # Dict format
            # Transform heading to ego-centric coordinates with normalization
            obj_yaw_rad = np.radians(obj_yaw_degrees)
            ego_centric_yaw = transform_heading_to_ego(
                obj_yaw_rad, ego_heading_rad, normalize=True
            )

            # Extract size from extent field (comprehensive fix plan)
            extent = obj["extent"]  # Required field - fail fast if missing
            if isinstance(extent, list):
                # extent = [half_length, half_width, half_height]
                length = 2 * extent[0]  # Full length = 2 * half_length
                width = 2 * extent[1]  # Full width = 2 * half_width
            else:
                # Dict format: extent = {"x": half_length, "y": half_width, "z": half_height}
                length = 2 * extent["x"]  # Full length = 2 * half_length
                width = 2 * extent["y"]  # Full width = 2 * half_width

            # Apply size limits for pedestrians to prevent unrealistic dimensions
            if navsim_class == 6:  # Pedestrian
                length = min(length, 0.8)  # Cap pedestrian length
                width = min(width, 0.6)  # Cap pedestrian width

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

        # Get agents for vehicle generation
        agents, labels, agent_types = self.get_agents(frame_idx)
        if isinstance(agents, torch.Tensor):
            agents = agents.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # Check if cached BEV exists
        map_bev = None
        cached_generation_type = None
        if hasattr(self.config, "bev_cache_dir") and self.config.bev_cache_dir:
            cache_dir = Path(self.config.bev_cache_dir)
            scenario_name = self.scene_info["scenario"]  # Get scenario name
            # Get frame number based on mode
            # DEPRECATED: 10Hz sliding window mode removed
            # if self.config.sliding_mode and self.all_frames is not None:
            #     # True sliding window mode - use absolute index
            #     frame_number = self.all_frames[self.start_idx].stem.split(".")[0]
            # else:
            #     # Legacy mode - use relative frame index
            #     frame_number = self.anno_paths[frame_idx].stem.split(".")[0]

            # Use legacy mode only
            frame_number = self.anno_paths[frame_idx].stem.split(".")[0]
            cache_path = cache_dir / scenario_name / f"{frame_number}.npz"

            if cache_path.exists():
                try:
                    cached_data = np.load(cache_path)

                    # Validate cache metadata to reject stale caches from old spatial config
                    if "resolution" in cached_data:
                        cached_res = float(cached_data["resolution"])
                        expected_res = BEV_SEMANTIC_RESOLUTION
                        if abs(cached_res - expected_res) > 1e-3:
                            raise ValueError(
                                f"Stale BEV cache: resolution={cached_res:.4f} m/px "
                                f"but current config expects {expected_res:.4f} m/px. "
                                f"Regenerate caches with: python3 scripts/generate_bev_cache.py"
                            )
                    if "range_m" in cached_data:
                        cached_range = float(cached_data["range_m"])
                        expected_range = BEV_SEMANTIC_RANGE_M
                        if abs(cached_range - expected_range) > 1e-3:
                            raise ValueError(
                                f"Stale BEV cache: range={cached_range:.1f}m "
                                f"but current config expects {expected_range:.1f}m. "
                                f"Regenerate caches with: python3 scripts/generate_bev_cache.py"
                            )

                    # Use front_bev if available, otherwise use full_bev
                    if "front_bev" in cached_data:
                        map_bev = cached_data["front_bev"]
                    elif "full_bev" in cached_data:
                        # Extract front half
                        full_bev = cached_data["full_bev"]
                        map_bev = full_bev[:128, :]  # Front half

                    # Check generation type if available
                    if "generation_type" in cached_data:
                        cached_generation_type = str(cached_data["generation_type"])
                    else:
                        cached_generation_type = "vector"
                except ValueError:
                    raise  # Re-raise validation errors (stale cache detection)
                except Exception as e:
                    raise RuntimeError(f"Failed to load cached BEV from {cache_path}: {e}") from e
            else:
                # If BEV cache directory is provided but file doesn't exist, raise error
                generation_hint = ""
                if hasattr(self.config, "bev_generation_type"):
                    generation_hint = f" ({self.config.bev_generation_type} mode)"
                raise FileNotFoundError(
                    f"BEV cache file not found: {cache_path}\n"
                    f"Please run generate_segmentation_bev_cache.py{generation_hint} first to generate BEV maps for scenario '{scenario_name}'"
                )

        # Check if we're in debug mode and can generate on-the-fly
        if map_bev is None and getattr(self.config, 'debug_mode', False) and hasattr(self.config, 'map_dir') and self.config.map_dir:
            # Debug mode: Generate BEV on-the-fly from HD map
            from navsim.common.bev_generation_factory import BEVGeneratorFactory
            from navsim.common.bev_map_utils import MapProcessor, load_map_data

            # Get generation type from config (default to vector if not specified)
            generation_type = getattr(self.config, 'debug_generation_type', 'vector')
            print(f"Debug mode: Generating BEV on-the-fly using {generation_type} format for frame {frame_idx}")

            # Create generator based on specified type
            generator_params = {
                "generation_type": generation_type,
                "bev_height": 128,
                "bev_width": 256,
                "resolution": BEV_SEMANTIC_RESOLUTION,
                "view_type": "front"
            }

            # Add type-specific parameters
            if generation_type == "segmentation":
                generator_params["lane_width"] = 5.0  # Use 5m width for better coverage

            generator = BEVGeneratorFactory.create_generator(**generator_params)

            # Load map data
            # Extract town name from scenario
            scenario_name = self.scene_info["scenario"]
            town_name = None
            for part in scenario_name.split("_"):
                if part.startswith("Town"):
                    town_name = part
                    break
            if not town_name:
                raise ValueError(f"Cannot extract town name from scenario: {scenario_name}")

            map_path = Path(self.config.map_dir) / f"{town_name}_HD_map.npz"
            if not map_path.exists():
                raise FileNotFoundError(f"HD map not found: {map_path}")

            map_data = load_map_data(map_path)
            map_processor = MapProcessor(map_data)

            # Get ego state
            current_anno = self._load_annotation(frame_idx)
            ego_box = None
            for box in current_anno["bounding_boxes"]:
                if box["class"] == "ego_vehicle":
                    ego_box = box
                    break

            if ego_box is None:
                raise ValueError(f"Ego vehicle not found in frame {frame_idx}")

            ego_points = np.array(ego_box["center"])
            ego_heading_rad = np.radians(ego_box["rotation"][2])

            # Generate BEV
            map_bev = generator.generate_from_map(
                map_data=map_processor,
                ego_points=ego_points,
                ego_heading_rad=ego_heading_rad
            )

        # For training (non-debug mode), BEV cache is required
        elif map_bev is None:
            raise ValueError(
                f"BEV cache not found. Training requires pre-generated BEV cache.\n"
                f"Run: python3 scripts/generate_bev_cache.py --generation-type [vector|segmentation]\n"
                f"Cache directory: {self.config.bev_cache_dir if hasattr(self.config, 'bev_cache_dir') else 'NOT SET'}\n"
                f"Scenario: {self.scene_info.get('scenario', 'unknown')}"
            )

        # Use two-stage approach (static cache + dynamic overlay)
        bev_map = generate_simple_bev_semantic(
            agents=agents,
            agent_labels=labels,
            agent_types=(
                agent_types if isinstance(agent_types, np.ndarray) else agent_types.numpy()
            ),
            bev_height=BEV_SEMANTIC_HEIGHT,
            bev_width=BEV_SEMANTIC_WIDTH,
            resolution=BEV_SEMANTIC_RESOLUTION,  # 0.332m/pixel for 85m coverage
            map_bev=map_bev,  # Use map-based BEV if available
        )

        return torch.from_numpy(bev_map).float()

    def __len__(self) -> int:
        """Get number of frames in scene."""
        return len(self.anno_paths)
