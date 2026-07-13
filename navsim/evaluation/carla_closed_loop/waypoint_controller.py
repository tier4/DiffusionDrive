"""Trajectory-to-control conversion for closed-loop CARLA evaluation.

Frames: model trajectory is CARLA left-handed EGO frame (x fwd, y right,
yaw CW-positive). control_pid (vendored TCP/VAD) wants [lateral_right, forward]
with straight-ahead = (0, +d). World->ego: x_e = cos(t)dx + sin(t)dy,
y_e = -sin(t)dx + cos(t)dy. See tests for hand-computed cases.

Steer-sign derivation (pid_controller.py:73): with waypoints stored as
[lateral_right, forward], aim = waypoints[0] = [aim[0]=lateral, aim[1]=forward],
so angle = degrees(pi/2 - arctan2(forward, lateral)) / 90.
  - straight ahead: [0, d] -> arctan2(d, 0) = pi/2 -> angle = 0.
  - aim to the RIGHT: lateral > 0 -> arctan2(forward, +lateral) < pi/2
    -> angle > 0 -> turn_controller (K_P>0) -> steer > 0 = RIGHT in CARLA.
  - aim to the LEFT: lateral < 0 -> angle < 0 -> steer < 0 = LEFT.
Signs are derived, not tuned: the curve-direction test passes without flips.
"""
import math
from typing import Dict

import numpy as np

from navsim.evaluation.carla_closed_loop.input_adapter import world_yaw_from_compass
from navsim.evaluation.carla_closed_loop.pid_controller import PIDController


def target_to_ego_frame(
    target_map_xy: np.ndarray, ego_map_xy: np.ndarray, theta_world: float
) -> np.ndarray:
    delta = np.asarray(target_map_xy, dtype=np.float64) - np.asarray(ego_map_xy, dtype=np.float64)
    c, s = math.cos(theta_world), math.sin(theta_world)
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])


def trajectory_to_pid_waypoints(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(f"Expected trajectory [N,>=2], got {trajectory.shape}")
    return np.stack([trajectory[:, 1], trajectory[:, 0]], axis=1).astype(np.float64)


class WaypointController:
    def __init__(self):
        self._pid = PIDController()

    def control(
        self,
        trajectory: np.ndarray,
        speed: float,
        target_map_xy: np.ndarray,
        ego_map_xy: np.ndarray,
        compass: float,
    ) -> Dict:
        theta = world_yaw_from_compass(compass)
        target_ego = target_to_ego_frame(target_map_xy, ego_map_xy, theta)
        target_pid = np.array([target_ego[1], target_ego[0]])
        waypoints = trajectory_to_pid_waypoints(np.asarray(trajectory))
        steer, throttle, brake, metadata = self._pid.control_pid(
            waypoints, np.float64(speed), target_pid
        )
        return {
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake) if not isinstance(brake, (bool, np.bool_)) else float(bool(brake)),
            "metadata": metadata,
        }
