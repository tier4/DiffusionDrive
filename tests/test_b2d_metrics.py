import numpy as np
import pytest


def test_gt_as_prediction_gives_zero_l2():
    """When prediction equals GT, all L2 metrics should be exactly 0."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.1, 0.01],
        [4.0, 0.2, 0.02],
        [5.0, 0.3, 0.03],
        [6.0, 0.4, 0.04],
        [7.0, 0.5, 0.05],
        [8.0, 0.6, 0.06],
    ], dtype=np.float32)

    result = metrics.compute_metrics(pred_trajectory=gt.copy(), gt_trajectory=gt)

    # Absolute L2 must be exactly 0 at all horizons
    for key, val in result["absolute_l2"].items():
        assert val == pytest.approx(0.0, abs=1e-7), f"absolute_l2[{key}] = {val}"

    # Offset L2 must also be exactly 0
    for key, val in result["offset_l2"].items():
        assert val == pytest.approx(0.0, abs=1e-7), f"offset_l2[{key}] = {val}"


def test_absolute_l2_basic():
    """Verify absolute L2 at specific horizons with a known offset."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = np.arange(1, 9)  # x = 1..8

    pred = gt.copy()
    pred[:, 1] = 1.0  # y offset of 1.0 at every step

    result = metrics.compute_metrics(pred_trajectory=pred, gt_trajectory=gt)

    # At every timestep, L2 distance = 1.0 (y offset only)
    # VAD period-average at 0.5s (step 1): mean([1.0]) = 1.0
    # VAD period-average at 1.0s (step 2): mean([1.0, 1.0]) = 1.0
    assert result["absolute_l2"]["L2_0.5s"] == pytest.approx(1.0)
    assert result["absolute_l2"]["L2_1.0s"] == pytest.approx(1.0)
    assert result["absolute_l2"]["L2_3.0s"] == pytest.approx(1.0)
    assert result["absolute_l2"]["L2_avg_vad"] == pytest.approx(1.0)
    assert result["absolute_l2"]["L2_avg_full"] == pytest.approx(1.0)


def test_offset_l2_differs_from_absolute():
    """Offset L2 should differ from absolute L2 for drifting predictions."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = np.arange(1, 9)

    # Prediction drifts: each step has cumulative error
    pred = gt.copy()
    pred[:, 1] = np.arange(1, 9) * 0.1  # y drift: 0.1, 0.2, ... 0.8

    result = metrics.compute_metrics(pred_trajectory=pred, gt_trajectory=gt)

    # Absolute L2 at step 1: sqrt(0.1^2) = 0.1
    # Absolute L2 at step 8: sqrt(0.8^2) = 0.8
    assert result["absolute_l2"]["L2_0.5s"] == pytest.approx(0.1, abs=1e-5)
    assert result["absolute_l2"]["L2_4.0s"] == pytest.approx(
        np.mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), abs=1e-5
    )

    # Offset L2 should be different (each offset error is 0.1)
    # GT offsets y: [0, 0, 0, ...], pred offsets y: [0.1, 0.1, 0.1, ...]
    # So offset L2 at every step = 0.1
    assert result["offset_l2"]["L2_0.5s"] == pytest.approx(0.1, abs=1e-5)
    assert result["offset_l2"]["L2_avg_vad"] == pytest.approx(0.1, abs=1e-5)


def test_vad_horizons_vs_full():
    """VAD average uses 6 timesteps, full average uses all 8."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)
    pred = np.zeros((8, 3), dtype=np.float32)
    # Error increases with time
    pred[:, 1] = np.arange(1, 9) * 0.5

    result = metrics.compute_metrics(pred_trajectory=pred, gt_trajectory=gt)

    # VAD avg covers L2_0.5s..L2_3.0s (6 horizons)
    # Full avg covers L2_0.5s..L2_4.0s (8 horizons)
    assert "L2_avg_vad" in result["absolute_l2"]
    assert "L2_avg_full" in result["absolute_l2"]
    # Full includes higher-error timesteps, so full_avg >= vad_avg
    assert result["absolute_l2"]["L2_avg_full"] >= result["absolute_l2"]["L2_avg_vad"]


def test_aggregate_metrics():
    """Aggregation should average across samples."""
    from navsim.evaluate.b2d_metrics import B2DOpenLoopMetrics

    metrics = B2DOpenLoopMetrics(num_timesteps=8, timestep_sec=0.5)

    gt = np.zeros((8, 3), dtype=np.float32)

    pred1 = gt.copy()
    pred1[:, 1] = 1.0  # L2 = 1.0 everywhere

    pred2 = gt.copy()
    pred2[:, 1] = 3.0  # L2 = 3.0 everywhere

    m1 = metrics.compute_metrics(pred_trajectory=pred1, gt_trajectory=gt)
    m2 = metrics.compute_metrics(pred_trajectory=pred2, gt_trajectory=gt)

    agg = metrics.aggregate_metrics([m1, m2])
    assert agg["absolute_l2"]["L2_0.5s"] == pytest.approx(2.0)
    assert agg["absolute_l2"]["L2_avg_vad"] == pytest.approx(2.0)
