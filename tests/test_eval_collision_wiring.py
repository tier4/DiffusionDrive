"""Test that collision metrics work when agent states are provided."""

import numpy as np
import pytest


def test_compute_metrics_with_agents_returns_collision():
    """compute_metrics should return per-horizon and average collision rates."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = np.arange(1, 9)
    pred = gt.copy()

    agent_states = np.zeros((5, 5), dtype=np.float32)
    agent_labels = np.zeros(5, dtype=bool)

    result = metrics.compute_metrics(
        pred_trajectory=pred,
        gt_trajectory=gt,
        gt_agent_states=agent_states,
        gt_agent_labels=agent_labels,
    )

    assert "collision" in result
    col = result["collision"]
    # Per-horizon cumulative collision (VAD format)
    assert "col_1.0s" in col
    assert "col_2.0s" in col
    assert "col_3.0s" in col
    assert "box_col_1.0s" in col
    assert "box_col_2.0s" in col
    assert "box_col_3.0s" in col
    # Overall averages
    assert "col_avg" in col
    assert "box_col_avg" in col


def test_compute_metrics_without_agents_returns_empty_collision():
    """compute_metrics should return empty collision dict without agent states."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = np.arange(1, 9)
    pred = gt.copy()

    result = metrics.compute_metrics(pred_trajectory=pred, gt_trajectory=gt)

    assert result["collision"] == {}


def test_aggregate_preserves_collision_metrics():
    """Aggregation should include collision metrics when present."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = np.arange(1, 9)

    agent_states = np.zeros((5, 5), dtype=np.float32)
    agent_labels = np.zeros(5, dtype=bool)

    m1 = metrics.compute_metrics(gt.copy(), gt, agent_states, agent_labels)
    m2 = metrics.compute_metrics(gt.copy(), gt, agent_states, agent_labels)

    agg = metrics.aggregate_metrics([m1, m2])
    assert "col_avg" in agg["collision"]
    assert "box_col_avg" in agg["collision"]
    assert "col_1.0s" in agg["collision"]
