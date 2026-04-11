"""
TaCarla scene representation.
Maps Parquet data to NAVSIM-compatible interfaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import laspy
import numpy as np
import torch
from PIL import Image

from navsim.common.abstract_carla_scene import AbstractCarlaScene
from navsim.common.abstract_carla_dataloader import map_carla_command_to_discrete
from navsim.common.dataclasses import AgentInput, EgoStatus, Camera, Cameras, Lidar
from navsim.common.bev_map_utils import transform_points_to_ego, transform_heading_to_ego
from navsim.common.tacarla_constants import (
    TACARLA_BEV_RGB_TO_CLASS,
    TACARLA_BEV_ORIGINAL_SIZE,
    TACARLA_NUM_BEV_CLASSES,
    TACARLA_CLASS_TO_NAVSIM,
    TACARLA_CAMERA_TAR_DIRS,
    BEV_SEMANTIC_HEIGHT,
    BEV_SEMANTIC_WIDTH,
    MAX_AGENTS,
    NUM_FUTURE_WAYPOINTS,
    AGENT_DISTANCE_THRESHOLD_M,
)

logger = logging.getLogger(__name__)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class TaCarlaScene(AbstractCarlaScene):
    """
    Scene for TaCarla dataset.
    Reads Parquet rows and extracted sensor data.
    """

    def __init__(self, scene_info: Dict, config):
        self.scene_info = scene_info
        self.config = config
        self.route_name = scene_info["route_name"]
        self.parquet_path = scene_info["parquet_path"]
        self.frame_indices = scene_info["frame_indices"]  # 10Hz indices
        self.token = scene_info["token"]

        # Lazy-loaded data
        self._df = None
        self._rows_cache = {}

    def _ensure_loaded(self):
        """Load Parquet data lazily."""
        if self._df is None:
            import pyarrow.parquet as pq
            columns = [
                "measurements", "new_boxes", "lanes",
                "front", "front_left", "front_right",
                "back", "back_left", "back_right",
                "lidar", "bev_label_image",
            ]
            table = pq.read_table(self.parquet_path, columns=columns)
            self._df = table.to_pandas()

    def _get_row(self, frame_idx: int) -> Dict:
        """Get a specific Parquet row by frame index."""
        self._ensure_loaded()
        if frame_idx not in self._rows_cache:
            row = self._df.iloc[frame_idx]
            self._rows_cache[frame_idx] = row
        return self._rows_cache[frame_idx]

    def _get_current_frame_idx(self) -> int:
        """Get the 10Hz index of the current frame (first frame, since num_history=0)."""
        return self.frame_indices[self.config.num_history_frames]

    def _get_sensor_path(self, row, sensor_col: str) -> Path:
        """
        Convert Parquet sensor path string to local extracted path.
        Parquet stores: ' /leaderboard_plant_pdm_Town12/route_name/detection/...'
        Local extraction: sensor_root/route_name/detection/...
        """
        raw_path = row[sensor_col].strip()
        # Extract relative path after the route name
        # Path format: /leaderboard_plant_pdm_TownXX/route_name/subpath
        parts = raw_path.split("/")
        # Find route name in path (it matches self.route_name)
        route_idx = None
        for i, part in enumerate(parts):
            if part == self.route_name:
                route_idx = i
                break

        if route_idx is not None:
            # Reconstruct: sensor_root / route_name / rest_of_path
            subpath = "/".join(parts[route_idx:])
            return Path(self.config.sensor_root) / subpath
        else:
            # Fallback: use the relative path after the town directory
            # /leaderboard_plant_pdm_Town12/route_name/...
            if len(parts) >= 3:
                subpath = "/".join(parts[2:])
                return Path(self.config.sensor_root) / subpath
            raise ValueError(f"Cannot parse sensor path: {raw_path}")

    # ---- AgentInput ----

    def get_agent_input(self, frame_idx: int = -1) -> AgentInput:
        """Get agent input for the current frame."""
        current_10hz = self._get_current_frame_idx()
        row = self._get_row(current_10hz)

        # Get previous row for acceleration derivation
        prev_10hz = current_10hz - 1
        prev_row = self._get_row(prev_10hz) if prev_10hz >= 0 else None

        ego_status = self._extract_ego_status(row, prev_row)
        cameras = self._load_cameras(row, current_10hz)
        lidar = self._load_lidar(row, current_10hz)

        return AgentInput(
            ego_statuses=[ego_status],
            cameras=[cameras],
            lidars=[lidar],
        )

    def _extract_ego_status(self, row, prev_row=None) -> EgoStatus:
        """Extract ego status from Parquet measurements."""
        m = row["measurements"]

        # Ego pose: [x, y, heading] in world coordinates
        x = float(m["x"])
        y = float(m["y"])
        heading = float(m["theta"])
        ego_pose = np.array([x, y, heading], dtype=np.float64)

        # Velocity: decompose scalar speed via heading
        speed = float(m["speed"])
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        ego_velocity = np.array([vx, vy], dtype=np.float32)

        # Acceleration: derive from consecutive frames
        if prev_row is not None:
            prev_m = prev_row["measurements"]
            prev_speed = float(prev_m["speed"])
            prev_heading = float(prev_m["theta"])
            prev_vx = prev_speed * np.cos(prev_heading)
            prev_vy = prev_speed * np.sin(prev_heading)
            dt = 0.1  # 10Hz
            ax = (vx - prev_vx) / dt
            ay = (vy - prev_vy) / dt
            # Clamp to reasonable values
            ax = np.clip(ax, -20.0, 20.0)
            ay = np.clip(ay, -20.0, 20.0)
            ego_acceleration = np.array([ax, ay], dtype=np.float32)
        else:
            ego_acceleration = np.array([0.0, 0.0], dtype=np.float32)

        # Driving command: raw CARLA enum (1-6), NOT B2D shifted
        command_raw = int(m["command"])
        command_discrete = map_carla_command_to_discrete(command_raw)
        # One-hot encode (4 classes: LEFT, STRAIGHT, RIGHT, UNKNOWN)
        driving_command = np.zeros(4, dtype=np.float32)
        driving_command[command_discrete] = 1.0

        return EgoStatus(
            ego_pose=ego_pose,
            ego_velocity=ego_velocity,
            ego_acceleration=ego_acceleration,
            driving_command=driving_command,
            in_global_frame=True,
        )

    def _load_cameras(self, row, frame_10hz: int) -> Cameras:
        """Load camera images from extracted sensor data."""
        camera_images = {}

        for cam_name in ["front", "front_left", "front_right", "back", "back_left", "back_right"]:
            try:
                img_path = self._get_sensor_path(row, cam_name)
                img = Image.open(img_path)
                camera_images[cam_name] = np.array(img)
            except Exception as e:
                logger.warning(f"Cannot load camera {cam_name} for frame {frame_10hz}: {e}")
                camera_images[cam_name] = np.zeros((900, 1600, 3), dtype=np.uint8)

        # Map TaCarla 6 cameras to NAVSIM 8-camera Cameras dataclass
        # Duplicate front_left/front_right for side cameras
        return Cameras(
            cam_f0=Camera(image=camera_images["front"]),
            cam_l0=Camera(image=camera_images["front_left"]),
            cam_l1=Camera(image=camera_images["front_left"]),  # duplicate
            cam_l2=Camera(image=camera_images["back_left"]),
            cam_r0=Camera(image=camera_images["front_right"]),
            cam_r1=Camera(image=camera_images["front_right"]),  # duplicate
            cam_r2=Camera(image=camera_images["back_right"]),
            cam_b0=Camera(image=camera_images["back"]),
        )

    def _load_lidar(self, row, frame_10hz: int) -> Lidar:
        """Load LiDAR point cloud from .laz file."""
        try:
            lidar_path = self._get_sensor_path(row, "lidar")
            las = laspy.read(str(lidar_path))
            # Extract x, y, z, intensity
            points = np.stack([
                las.x.astype(np.float32),
                las.y.astype(np.float32),
                las.z.astype(np.float32),
                getattr(las, 'intensity', np.zeros(len(las.x))).astype(np.float32),
            ], axis=0)  # [4, N]

            # Pad to [6, N] to match NAVSIM format (ring_id=0, lidar_id=0)
            n_points = points.shape[1]
            padded = np.zeros((6, n_points), dtype=np.float32)
            padded[:4] = points
            return Lidar(lidar_pc=padded)
        except Exception as e:
            logger.warning(f"Cannot load LiDAR for frame {frame_10hz}: {e}")
            return Lidar(lidar_pc=np.zeros((6, 1), dtype=np.float32))

    # ---- Future Trajectory ----

    def get_future_trajectory(self, frame_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get 8 future ego poses at 0.5s intervals in ego-local coordinates.
        Returns [8, 3] tensor (x, y, heading) or None.
        """
        current_10hz = self._get_current_frame_idx()
        self._ensure_loaded()

        # Collect 8 future frames at 0.5s intervals (every 5 frames at 10Hz)
        future_indices = []
        for i in range(1, NUM_FUTURE_WAYPOINTS + 1):
            future_10hz = current_10hz + i * 5  # 0.5s intervals
            if future_10hz >= len(self._df):
                return None
            future_indices.append(future_10hz)

        # Get current ego pose (origin)
        current_row = self._get_row(current_10hz)
        current_m = current_row["measurements"]
        ego_x = float(current_m["x"])
        ego_y = float(current_m["y"])
        ego_heading = float(current_m["theta"])

        # Extract future poses in world coordinates
        future_poses_world = []
        for fut_idx in future_indices:
            fut_row = self._get_row(fut_idx)
            fut_m = fut_row["measurements"]
            future_poses_world.append([
                float(fut_m["x"]),
                float(fut_m["y"]),
                float(fut_m["theta"]),
            ])

        future_poses_world = np.array(future_poses_world, dtype=np.float64)  # [8, 3]

        # Transform to ego-local coordinates
        ego_pos = np.array([ego_x, ego_y])
        positions = future_poses_world[:, :2]  # [8, 2]
        headings = future_poses_world[:, 2]  # [8]

        # Transform positions
        local_positions = transform_points_to_ego(positions, ego_pos, ego_heading)

        # Transform headings
        local_headings = transform_heading_to_ego(headings, ego_heading)

        # Combine
        trajectory = np.zeros((NUM_FUTURE_WAYPOINTS, 3), dtype=np.float32)
        trajectory[:, :2] = local_positions.astype(np.float32)
        trajectory[:, 2] = local_headings.astype(np.float32)

        return torch.from_numpy(trajectory)

    # ---- Agents ----

    def get_agents(self, frame_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get agent states from new_boxes column.
        Returns (agent_states [30,5], agent_labels [30], agent_types [30]).
        """
        current_10hz = self._get_current_frame_idx()
        row = self._get_row(current_10hz)

        # Get ego pose for world-to-ego transform
        m = row["measurements"]
        ego_x = float(m["x"])
        ego_y = float(m["y"])
        ego_heading = float(m["theta"])
        ego_pos = np.array([ego_x, ego_y])

        # Parse new_boxes
        boxes = row["new_boxes"]
        agent_states = np.zeros((MAX_AGENTS, 5), dtype=np.float32)
        agent_labels = np.zeros(MAX_AGENTS, dtype=bool)
        agent_types = np.zeros(MAX_AGENTS, dtype=np.int32)

        agent_idx = 0
        for box in boxes:
            if agent_idx >= MAX_AGENTS:
                break

            # Get class
            box_class = box.get("class", "")
            if box_class not in TACARLA_CLASS_TO_NAVSIM:
                continue

            # Skip ego vehicle (position is typically [-1.3, 0, -2.5])
            position = np.array(box["position"], dtype=np.float64)
            if np.allclose(position, [-1.3, 0.0, -2.5], atol=0.5):
                continue

            # Get world location
            world_loc = np.array(box["world_location"], dtype=np.float64)

            # Distance check
            dist = np.sqrt((world_loc[0] - ego_x)**2 + (world_loc[1] - ego_y)**2)
            if dist > AGENT_DISTANCE_THRESHOLD_M:
                continue

            # Transform to ego coordinates
            agent_pos_world = world_loc[:2]
            agent_pos_ego = transform_points_to_ego(
                agent_pos_world.reshape(1, 2), ego_pos, ego_heading
            )[0]

            # Get heading from vehicle_rotation (pitch, yaw, roll) in radians
            rotation = np.array(box.get("vehicle_rotation", [0, 0, 0]), dtype=np.float64)
            agent_heading_world = rotation[1]  # yaw
            agent_heading_ego = transform_heading_to_ego(
                np.array([agent_heading_world]), ego_heading
            )[0]

            # Get dimensions from extent [half_width, half_length, half_height]
            extent = np.array(box["extent"], dtype=np.float32)
            length = extent[1] * 2  # full length
            width = extent[0] * 2  # full width

            # Cap pedestrian dimensions
            navsim_type = TACARLA_CLASS_TO_NAVSIM[box_class]
            if navsim_type == 6:  # pedestrian
                length = min(length, 0.8)
                width = min(width, 0.6)

            agent_states[agent_idx] = [
                agent_pos_ego[0], agent_pos_ego[1],
                agent_heading_ego, length, width
            ]
            agent_labels[agent_idx] = True
            agent_types[agent_idx] = navsim_type
            agent_idx += 1

        return (
            torch.from_numpy(agent_states),
            torch.from_numpy(agent_labels),
            torch.from_numpy(agent_types),
        )

    # ---- BEV Semantic Map ----

    def get_bev_semantic_map(self, frame_idx: int = -1) -> torch.Tensor:
        """
        Load BEV semantic map from TaCarla's bev_label_image.
        200x200 RGB -> integer class labels, resized to model input.
        """
        current_10hz = self._get_current_frame_idx()
        row = self._get_row(current_10hz)

        # Build BEV image path
        # Parquet stores the directory path, actual file is {frame_idx:04d}.png
        bev_dir_path = self._get_sensor_path(row, "bev_label_image")
        bev_file = bev_dir_path / f"{current_10hz:04d}.png"

        try:
            img = Image.open(bev_file)
            arr = np.array(img)  # [200, 200, 3]
        except Exception as e:
            logger.warning(f"Cannot load BEV image {bev_file}: {e}")
            return torch.zeros((BEV_SEMANTIC_HEIGHT, BEV_SEMANTIC_WIDTH), dtype=torch.long)

        # Map RGB to class labels
        class_map = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        for rgb, cls_id in TACARLA_BEV_RGB_TO_CLASS.items():
            mask = np.all(arr == np.array(rgb, dtype=np.uint8), axis=-1)
            class_map[mask] = cls_id

        # Resize to model input dimensions using nearest-neighbor
        # (preserves class boundaries, no interpolation artifacts)
        class_map_resized = cv2.resize(
            class_map,
            (BEV_SEMANTIC_WIDTH, BEV_SEMANTIC_HEIGHT * 2),  # Full BEV first (256x256)
            interpolation=cv2.INTER_NEAREST,
        )

        # Take front half (ego is centered in TaCarla BEV, we want front-half)
        # TaCarla ego is at row ~100/200 (center), so front = top half
        front_half = class_map_resized[:BEV_SEMANTIC_HEIGHT, :]  # [128, 256]

        return torch.from_numpy(front_half.astype(np.int64))

    def __len__(self) -> int:
        """Number of frames in this scene window."""
        return len(self.frame_indices)
