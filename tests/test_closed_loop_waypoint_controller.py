"""Geometry tests for the closed-loop waypoint controller (hand-computed cases)."""
import math

import numpy as np
import pytest

from navsim.evaluation.carla_closed_loop.waypoint_controller import (
    WaypointController,
    target_to_ego_frame,
    trajectory_to_pid_waypoints,
)


class TestFrames:
    def test_target_straight_ahead(self):
        # ego at origin heading +x; target 10m ahead -> ego frame (10, 0)
        out = target_to_ego_frame(np.array([10.0, 0.0]), np.array([0.0, 0.0]), 0.0)
        np.testing.assert_allclose(out, [10.0, 0.0], atol=1e-9)

    def test_target_to_the_right(self):
        # heading +x; target at world (0, 5) = 5m to the RIGHT in left-handed CARLA
        out = target_to_ego_frame(np.array([0.0, 5.0]), np.array([0.0, 0.0]), 0.0)
        np.testing.assert_allclose(out, [0.0, 5.0], atol=1e-9)

    def test_target_rotated_ego(self):
        # ego heading 90deg (CW toward +y); target 10m ahead is world (0, 10)
        out = target_to_ego_frame(np.array([0.0, 10.0]), np.array([0.0, 0.0]), math.pi / 2)
        np.testing.assert_allclose(out, [10.0, 0.0], atol=1e-9)

    def test_trajectory_axis_swap(self):
        traj = np.array([[2.0, 0.5, 0.0]])  # x fwd 2m, y right 0.5m, heading
        wps = trajectory_to_pid_waypoints(traj)
        np.testing.assert_allclose(wps, [[0.5, 2.0]])  # [lateral, forward]


class TestControl:
    def test_straight_line_yields_near_zero_steer(self):
        ctrl = WaypointController()
        traj = np.array([[(i + 1) * 2.0, 0.0, 0.0] for i in range(8)])
        result = ctrl.control(
            trajectory=traj, speed=2.0,
            target_map_xy=np.array([20.0, 0.0]), ego_map_xy=np.zeros(2),
            compass=math.pi / 2,  # world yaw 0
        )
        assert abs(result["steer"]) < 0.05
        assert result["brake"] in (0.0, 1.0) or isinstance(result["brake"], (bool, np.bool_))

    def test_right_curve_steers_right_left_curve_steers_left(self):
        ctrl_r = WaypointController()
        curve_r = np.array([[(i + 1) * 2.0, 0.15 * (i + 1) ** 2, 0.0] for i in range(8)])
        res_r = ctrl_r.control(curve_r, 2.0, np.array([16.0, 8.0]), np.zeros(2), math.pi / 2)
        ctrl_l = WaypointController()
        curve_l = curve_r.copy(); curve_l[:, 1] *= -1
        res_l = ctrl_l.control(curve_l, 2.0, np.array([16.0, -8.0]), np.zeros(2), math.pi / 2)
        assert res_r["steer"] > 0.0   # CARLA: positive steer = right
        assert res_l["steer"] < 0.0

    def test_stationary_target_triggers_brake(self):
        ctrl = WaypointController()
        traj = np.zeros((8, 3))  # model says "stay put"
        res = ctrl.control(traj, 5.0, np.array([1.0, 0.0]), np.zeros(2), math.pi / 2)
        assert bool(res["brake"]) is True
        assert res["throttle"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
