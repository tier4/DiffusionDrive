import numpy as np
import torch
import pytest


def test_planning_metric_no_agents():
    """With no valid agents, occupancy should be all zeros and no collisions."""
    from navsim.evaluate.b2d_planning_utils import PlanningMetric

    metric = PlanningMetric()

    # No valid agents (all labels False)
    agent_states = np.zeros((1, 5, 5), dtype=np.float32)  # [B, N, 5]
    agent_labels = np.zeros((1, 5), dtype=bool)  # [B, N] all False

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
