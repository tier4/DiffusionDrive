"""Convert live CARLA leaderboard sensor data into the training AgentInput.

Parity contract (sources):
- LiDAR frame: data_collect.py:46-68 (identity rotation + translation [-0.39, 0, 1.84])
- Velocity:    bench2drive_scene.py:552-560 (GLOBAL frame: vx=v*cos(theta), vy=-v*sin(theta))
- Accel:       data_collect.py:79 (raw IMU accelerometer [:2])
- Command:     bench2drive_dataloader.map_carla_command_to_discrete
This module contains NO feature math -- Bench2DriveFeatureBuilder does that.
"""
import math
from typing import Dict

import numpy as np

from navsim.common.bench2drive_dataloader import map_carla_command_to_discrete
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar

LIDAR_TO_EGO_TRANSLATION = np.array([-0.39, 0.0, 1.84], dtype=np.float32)


def carla_bgra_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected BGRA image [H,W,4], got shape {img.shape}")
    return np.ascontiguousarray(img[:, :, 2::-1])


def lidar_sensor_to_ego(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected lidar points [N,>=3], got shape {points.shape}")
    out = np.zeros((points.shape[0], 4), dtype=np.float32)
    out[:, :3] = points[:, :3] + LIDAR_TO_EGO_TRANSLATION
    if points.shape[1] >= 4:
        out[:, 3] = points[:, 3]
    return out


def world_yaw_from_compass(compass: float) -> float:
    """CARLA world yaw (rad) from leaderboard IMU compass. Candidate formula
    theta = compass - pi/2, validated against privileged yaw in the smoke test."""
    theta = float(compass) - math.pi / 2.0
    return math.atan2(math.sin(theta), math.cos(theta))


def build_ego_status(
    speed: float,
    compass: float,
    imu_accel_xy: np.ndarray,
    road_option_value: int,
    map_xy: np.ndarray,
) -> EgoStatus:
    theta = world_yaw_from_compass(compass)
    velocity = np.array(
        [speed * math.cos(theta), -speed * math.sin(theta)], dtype=np.float32
    )
    command = map_carla_command_to_discrete(int(road_option_value))
    return EgoStatus(
        ego_pose=np.array([map_xy[0], map_xy[1], theta], dtype=np.float64),
        ego_velocity=velocity,
        ego_acceleration=np.asarray(imu_accel_xy[:2], dtype=np.float32),
        driving_command=np.array([command], dtype=np.int32),
        in_global_frame=True,
    )


def build_agent_input(
    camera_images: Dict[str, np.ndarray],
    lidar_points: np.ndarray,
    speed: float,
    compass: float,
    imu_accel_xy: np.ndarray,
    road_option_value: int,
    map_xy: np.ndarray,
) -> AgentInput:
    required = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
    for name in required:
        if name not in camera_images or camera_images[name] is None:
            raise ValueError(f"Missing camera image: {name}")
    cameras = Cameras(
        cam_f0=Camera(image=carla_bgra_to_rgb(camera_images["CAM_FRONT"])),
        cam_l0=Camera(image=carla_bgra_to_rgb(camera_images["CAM_FRONT_LEFT"])),
        cam_r0=Camera(image=carla_bgra_to_rgb(camera_images["CAM_FRONT_RIGHT"])),
        cam_l1=Camera(), cam_l2=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
    )
    lidar = Lidar(lidar_pc=lidar_sensor_to_ego(lidar_points))
    ego_status = build_ego_status(speed, compass, imu_accel_xy, road_option_value, map_xy)
    return AgentInput(ego_statuses=[ego_status], cameras=[cameras], lidars=[lidar])
