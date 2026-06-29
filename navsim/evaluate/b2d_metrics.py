"""
Open-loop evaluation metrics for Bench2Drive.

Computes:
- Absolute position L2 (VAD-compatible): L2 on ego-centric (x, y) positions.
- Offset L2: L2 on frame-to-frame displacements.
- Collision rates: point and bounding-box collision against occupancy.

VAD method: period-average L2 over [0, t] at each horizon.
DiffusionDrive uses 8 waypoints at 0.5s intervals (4.0s total).
VAD-comparable horizons: 0.5s–3.0s (first 6 of 8 timesteps).
"""

import numpy as np
from typing import Dict, List, Optional


# Horizons reported for VAD comparison (6 timesteps, 0.5s–3.0s)
_VAD_NUM_TIMESTEPS = 6


class B2DOpenLoopMetrics:
    """Compute open-loop planning metrics for B2D evaluation."""

    def __init__(self, num_timesteps: int = 8, timestep_sec: float = 0.5):
        """
        Args:
            num_timesteps: Total trajectory waypoints (DiffusionDrive = 8).
            timestep_sec: Duration per waypoint step (DiffusionDrive = 0.5s).
        """
        self.num_timesteps = num_timesteps
        self.timestep_sec = timestep_sec

        self.horizon_labels = [
            f"L2_{(i + 1) * timestep_sec:.1f}s" for i in range(num_timesteps)
        ]

    def compute_metrics(
        self,
        pred_trajectory: np.ndarray,
        gt_trajectory: np.ndarray,
        gt_agent_states: Optional[np.ndarray] = None,
        gt_agent_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute all metrics for a single sample.

        Args:
            pred_trajectory: [T, 3] predicted (x, y, heading) in ego-centric coords.
            gt_trajectory: [T, 3] ground truth (x, y, heading) in ego-centric coords.
            gt_agent_states: Optional [N, 5] agent (x, y, heading, length, width).
            gt_agent_labels: Optional [N] boolean validity mask.

        Returns:
            Dict with keys: absolute_l2, offset_l2, collision.
        """
        T = min(pred_trajectory.shape[0], gt_trajectory.shape[0], self.num_timesteps)

        pred = pred_trajectory[:T, :2]
        gt = gt_trajectory[:T, :2]

        # --- Absolute position L2 ---
        abs_l2_per_step = np.linalg.norm(pred - gt, axis=1)  # [T]
        absolute_l2 = self._compute_horizon_metrics(abs_l2_per_step, T)

        # --- Offset L2 ---
        pred_with_origin = np.vstack([np.zeros((1, 2)), pred])
        gt_with_origin = np.vstack([np.zeros((1, 2)), gt])
        pred_offsets = np.diff(pred_with_origin, axis=0)  # [T, 2]
        gt_offsets = np.diff(gt_with_origin, axis=0)  # [T, 2]
        offset_l2_per_step = np.linalg.norm(pred_offsets - gt_offsets, axis=1)
        offset_l2 = self._compute_horizon_metrics(offset_l2_per_step, T)

        # --- Collision ---
        collision = {}
        if gt_agent_states is not None and gt_agent_labels is not None:
            collision = self._compute_collision(
                pred_trajectory[:T], gt_trajectory[:T],
                gt_agent_states, gt_agent_labels, T,
            )

        return {
            "absolute_l2": absolute_l2,
            "offset_l2": offset_l2,
            "collision": collision,
        }

    def _compute_horizon_metrics(self, l2_per_step: np.ndarray, T: int) -> Dict[str, float]:
        """Compute period-average L2 at each horizon plus averages."""
        result = {}

        vad_horizon_values = []
        all_horizon_values = []

        for t in range(T):
            period_avg = float(np.mean(l2_per_step[: t + 1]))
            label = self.horizon_labels[t]
            result[label] = period_avg
            all_horizon_values.append(period_avg)
            if t < _VAD_NUM_TIMESTEPS:
                vad_horizon_values.append(period_avg)

        result["L2_avg_vad"] = float(np.mean(vad_horizon_values)) if vad_horizon_values else 0.0
        result["L2_avg_full"] = float(np.mean(all_horizon_values)) if all_horizon_values else 0.0

        return result

    def _compute_collision(
        self,
        pred_traj: np.ndarray,
        gt_traj: np.ndarray,
        gt_agent_states: np.ndarray,
        gt_agent_labels: np.ndarray,
        T: int,
    ) -> Dict[str, float]:
        """Compute collision rates using PlanningMetric."""
        from navsim.evaluate.b2d_planning_utils import PlanningMetric

        pm = PlanningMetric()

        if gt_agent_states.ndim == 2:
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]

        seg, ped = pm.get_label(gt_agent_states, gt_agent_labels, num_timesteps=T)
        import torch
        occupancy = torch.logical_or(seg, ped)

        pred_batch = pred_traj[np.newaxis, ...]  # [1, T, 3]
        gt_batch = gt_traj[np.newaxis, ...]

        obj_coll, obj_box_coll = pm.evaluate_coll(pred_batch, gt_batch, occupancy)

        return {
            "obj_col_avg": float(obj_coll.float().mean()),
            "obj_box_col_avg": float(obj_box_coll.float().mean()),
        }

    def aggregate_metrics(self, all_sample_metrics: List[Dict]) -> Dict:
        """
        Average metrics across samples.

        Args:
            all_sample_metrics: List of dicts from compute_metrics().

        Returns:
            Dict with same structure, values averaged.
        """
        if not all_sample_metrics:
            return {"absolute_l2": {}, "offset_l2": {}, "collision": {}}

        result = {}
        for top_key in ("absolute_l2", "offset_l2", "collision"):
            sub_dicts = [m[top_key] for m in all_sample_metrics if m[top_key]]
            if not sub_dicts:
                result[top_key] = {}
                continue
            all_keys = sub_dicts[0].keys()
            result[top_key] = {
                k: float(np.mean([d[k] for d in sub_dicts])) for k in all_keys
            }

        return result
