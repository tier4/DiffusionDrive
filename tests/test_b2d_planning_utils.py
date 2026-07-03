import numpy as np
import torch
import pytest


def test_planning_metric_no_agents():
    """With no valid agents, occupancy should be all zeros and no collisions."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    metric = PlanningMetric()

    # No valid agents (all labels False)
    agent_states = np.zeros((1, 4, 5, 5), dtype=np.float32)  # [B, T, N, 5]
    agent_labels = np.zeros((1, 4, 5), dtype=bool)  # [B, T, N] all False

    seg, ped = metric.get_label(agent_states, agent_labels, num_timesteps=4)
    assert seg.shape == (1, 4, 200, 200)
    assert ped.shape == (1, 4, 200, 200)
    assert seg.sum() == 0
    assert ped.sum() == 0


def test_planning_metric_collision_free():
    """Trajectory at origin with no obstacles should have zero collisions."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    metric = PlanningMetric()

    # Trajectory going straight ahead from origin
    trajs = np.zeros((1, 4, 3), dtype=np.float32)
    trajs[0, :, 0] = [1.0, 2.0, 3.0, 4.0]  # x forward

    gt_trajs = trajs.copy()

    # Empty occupancy
    occupancy = np.zeros((1, 4, 200, 200), dtype=np.float32)

    obj_coll, obj_box_coll = metric.evaluate_coll(trajs, gt_trajs, occupancy)
    assert obj_coll.shape == (1, 4)
    assert obj_box_coll.shape == (1, 4)
    assert obj_coll.sum() == 0
    assert obj_box_coll.sum() == 0


def test_polygon_rasterization():
    """A simple square polygon should fill the expected region."""
    from navsim.evaluate.b2d_planning_utils import polygon_simple

    # Square from (2,2) to (5,5)
    r = np.array([2, 2, 5, 5], dtype=np.float64)
    c = np.array([2, 5, 5, 2], dtype=np.float64)

    rr, cc = polygon_simple(r, c, shape=(10, 10))
    assert len(rr) > 0
    # All points should be within the polygon bounds
    assert np.all(rr >= 2) and np.all(rr <= 5)
    assert np.all(cc >= 2) and np.all(cc <= 5)


def test_rasterize_polygon_fills_box():
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    # axis-aligned box in pixel space: rows 10..20, cols 30..40 (corners as (row, col))
    corners = np.array([[10, 30], [10, 40], [20, 40], [20, 30]], dtype=np.int32)
    rr, cc = pm._rasterize_polygon(corners, 200, 200)

    filled = set(zip(rr.tolist(), cc.tolist()))
    assert (15, 35) in filled          # interior
    assert (10, 30) in filled          # corner (cv2 includes boundary)
    assert rr.min() >= 10 and rr.max() <= 20
    assert cc.min() >= 30 and cc.max() <= 40


def test_rasterize_polygon_clips_out_of_bounds():
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    # box partially outside the canvas must be clipped, not wrapped or crashed
    corners = np.array([[-5, -5], [-5, 5], [5, 5], [5, -5]], dtype=np.int32)
    rr, cc = pm._rasterize_polygon(corners, 200, 200)
    assert len(rr) > 0
    assert rr.min() >= 0 and cc.min() >= 0


def test_moving_agent_occupancy_tracks_motion():
    """Occupancy at timestep t must reflect where agents are AT t, not at t=0."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    T, N = 2, 1
    states = np.zeros((T, N, 5), dtype=np.float32)
    labels = np.ones((T, N), dtype=bool)
    states[0, 0] = [5.0, 0.0, 0.0, 4.0, 2.0]    # t=0: agent at x=5
    states[1, 0] = [10.0, 0.0, 0.0, 4.0, 2.0]   # t=1: agent moved to x=10

    seg, _ = pm.get_label(states, labels, num_timesteps=2)

    assert seg.shape == (1, 2, 200, 200)
    # pixel of (5,0) is (110,100); of (10,0) is (120,100) — see grid cheat-sheet
    assert seg[0, 0, 110, 100] == 1 and seg[0, 0, 120, 100] == 0
    assert seg[0, 1, 120, 100] == 1 and seg[0, 1, 110, 100] == 0


def test_get_label_rejects_static_agent_states():
    """The old frozen-at-t0 [N, 5] input shape must be an explicit error."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    states = np.zeros((5, 5), dtype=np.float32)   # legacy [N, 5]
    labels = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match="per-timestep"):
        pm.get_label(states, labels, num_timesteps=2)


def test_get_label_rejects_legacy_batched_static_states():
    """Legacy [B, N, 5] (B != num_timesteps) must raise, not be silently
    misread as [T, N, 5]."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    pm = PlanningMetric()
    states = np.zeros((32, 5, 5), dtype=np.float32)   # old [B, N, 5], B=32
    labels = np.zeros((32, 5), dtype=bool)
    with pytest.raises(ValueError, match="per-timestep"):
        pm.get_label(states, labels, num_timesteps=6)
