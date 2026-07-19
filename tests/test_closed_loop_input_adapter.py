"""Tests for the CARLA closed-loop input adapter (no CARLA install required)."""
import subprocess
import sys


def test_package_imports_without_carla():
    """The closed-loop package must import in an env with no carla/leaderboard."""
    code = (
        "import sys; sys.modules['carla'] = None; sys.modules['leaderboard'] = None; "
        "import navsim.evaluation.carla_closed_loop.pid_controller; "
        "import navsim.evaluation.carla_closed_loop.route_planner; "
        "print('OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


import math

import numpy as np

from navsim.common.bench2drive_dataloader import map_carla_command_to_discrete
from navsim.evaluation.carla_closed_loop.input_adapter import (
    build_agent_input,
    build_ego_status,
    carla_bgra_to_rgb,
    lidar_sensor_to_ego,
    world_yaw_from_compass,
)


class TestInputAdapter:
    def test_bgra_to_rgb(self):
        bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        bgra[..., 0] = 10  # B
        bgra[..., 1] = 20  # G
        bgra[..., 2] = 30  # R
        rgb = carla_bgra_to_rgb(bgra)
        assert rgb.shape == (2, 2, 3)
        assert rgb[0, 0].tolist() == [30, 20, 10]

    def test_lidar_translation_matches_data_collect(self):
        # data_collect.py:46-68 -- identity rotation + translation [-0.39, 0, 1.84]
        pts = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
        out = lidar_sensor_to_ego(pts)
        np.testing.assert_allclose(out[0, :3], [0.61, 2.0, 4.84], atol=1e-6)
        assert out[0, 3] == np.float32(0.5)
        assert out.dtype == np.float32

    def test_world_yaw_from_compass(self):
        # candidate: theta = compass - pi/2, wrapped to [-pi, pi]
        assert abs(world_yaw_from_compass(math.pi / 2) - 0.0) < 1e-9
        assert abs(world_yaw_from_compass(0.0) - (-math.pi / 2)) < 1e-9
        assert abs(world_yaw_from_compass(2 * math.pi) - (-math.pi / 2)) < 1e-9

    def test_ego_status_velocity_is_global_frame(self):
        # bench2drive_scene.py:552-560: vx = v*cos(theta), vy = -v*sin(theta)
        theta = math.pi / 6
        compass = theta + math.pi / 2
        status = build_ego_status(
            speed=10.0, compass=compass, imu_accel_xy=np.array([0.1, -0.2]),
            road_option_value=3, map_xy=np.array([5.0, 6.0]),
        )
        np.testing.assert_allclose(
            status.ego_velocity, [10 * math.cos(theta), -10 * math.sin(theta)], atol=1e-5
        )
        np.testing.assert_allclose(status.ego_acceleration, [0.1, -0.2], atol=1e-6)
        # CARLA STRAIGHT(3) -> NavSim STRAIGHT(1) via map_carla_command_to_discrete
        assert status.driving_command[0] == 1
        assert status.in_global_frame is True

    def test_ego_status_command_uses_shared_mapper(self):
        # The adapter must defer to the shared map_carla_command_to_discrete
        # (never re-implement it). Per that mapper, CARLA VOID(-1) is normalized
        # to LANEFOLLOW and therefore maps to NavSim STRAIGHT(1) -- not UNKNOWN.
        void_status = build_ego_status(
            speed=0.0, compass=0.0, imu_accel_xy=np.zeros(2),
            road_option_value=-1, map_xy=np.zeros(2),
        )
        assert void_status.driving_command[0] == map_carla_command_to_discrete(-1)
        assert void_status.driving_command[0] == 1  # STRAIGHT
        # A genuinely out-of-range command maps to UNKNOWN(3).
        unknown_status = build_ego_status(
            speed=0.0, compass=0.0, imu_accel_xy=np.zeros(2),
            road_option_value=0, map_xy=np.zeros(2),
        )
        assert unknown_status.driving_command[0] == 3  # UNKNOWN

    def test_build_agent_input_shapes(self):
        cams = {
            name: np.zeros((900, 1600, 4), dtype=np.uint8)
            for name in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
        }
        lidar = np.zeros((100, 4), dtype=np.float32)
        ai = build_agent_input(
            camera_images=cams, lidar_points=lidar, speed=1.0, compass=0.0,
            imu_accel_xy=np.zeros(2), road_option_value=4, map_xy=np.zeros(2),
        )
        assert len(ai.cameras) == 1 and len(ai.lidars) == 1 and len(ai.ego_statuses) == 1
        assert ai.cameras[-1].cam_f0.image.shape == (900, 1600, 3)
        assert ai.cameras[-1].cam_l0.image.shape == (900, 1600, 3)
        assert ai.cameras[-1].cam_r0.image.shape == (900, 1600, 3)
        assert ai.lidars[-1].lidar_pc.shape == (100, 4)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
