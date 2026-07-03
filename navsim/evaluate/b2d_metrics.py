"""
Open-loop evaluation metrics for Bench2Drive.

Computes:
- Absolute position L2 (VAD-compatible): L2 on ego-centric (x, y) positions.
- Offset L2: L2 on frame-to-frame displacements.
- Collision rates: point and bounding-box collision against occupancy.

VAD method: period-average L2 over [0, t] at each horizon.
DiffusionDrive uses 8 waypoints at 0.5s intervals (4.0s total).

Per-horizon collision (col_{t}s / box_col_{t}s) is the mean of per-step
collision flags over [0, t]. col_avg_vad / box_col_avg_vad (and the L2
equivalent, L2_avg_vad) are the mean of the 1s/2s/3s horizon values.
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from navsim.evaluate.b2d_planning_utils import PlanningMetric


# VAD reports 1s/2s/3s horizons; at 0.5s per step these are indices 1, 3, 5.
_VAD_HORIZON_INDICES = (1, 3, 5)


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

        if abs(timestep_sec - 0.5) > 1e-9:
            raise ValueError(
                f"VAD horizon indices assume timestep_sec=0.5, got {timestep_sec}"
            )
        self._planning_metric = PlanningMetric()

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
            gt_agent_states: Optional [T, N, 5] agent (x, y, heading, length, width).
            gt_agent_labels: Optional [T, N] boolean validity mask.

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

        all_horizon_values = []

        for t in range(T):
            period_avg = float(np.mean(l2_per_step[: t + 1]))
            label = self.horizon_labels[t]
            result[label] = period_avg
            all_horizon_values.append(period_avg)

        vad_values = [all_horizon_values[i] for i in _VAD_HORIZON_INDICES if i < T]
        result["L2_avg_vad"] = float(np.mean(vad_values)) if vad_values else 0.0
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
        """Collision rates with VAD/STP3 semantics.

        Per-horizon values are the MEAN of per-step collision flags over
        steps [0, t] (VAD: plan_obj_col_{t}s = obj_coll.mean()). Timesteps
        where the GT trajectory itself collides are masked inside
        evaluate_coll. col_any_* keys keep the cumulative-any diagnostic.
        """
        gt_agent_states = np.asarray(gt_agent_states)
        gt_agent_labels = np.asarray(gt_agent_labels).astype(bool)
        if gt_agent_states.ndim == 3:  # [T, N, 5] → [1, T, N, 5]
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]

        seg, ped = self._planning_metric.get_label(
            gt_agent_states, gt_agent_labels, num_timesteps=T
        )
        occupancy = torch.logical_or(seg, ped)

        obj_coll, obj_box_coll = self._planning_metric.evaluate_coll(
            pred_traj[np.newaxis, ...], gt_traj[np.newaxis, ...], occupancy
        )

        result = {}
        for t in range(T):
            label = f"{(t + 1) * self.timestep_sec:.1f}s"
            result[f"col_{label}"] = float(obj_coll[0, : t + 1].float().mean())
            result[f"box_col_{label}"] = float(obj_box_coll[0, : t + 1].float().mean())
            result[f"col_any_{label}"] = float(obj_coll[0, : t + 1].any().float())
            result[f"box_col_any_{label}"] = float(obj_box_coll[0, : t + 1].any().float())

        for prefix, flags in (("col", obj_coll), ("box_col", obj_box_coll)):
            vad_values = [
                result[f"{prefix}_{(i + 1) * self.timestep_sec:.1f}s"]
                for i in _VAD_HORIZON_INDICES
                if i < T
            ]
            result[f"{prefix}_avg_vad"] = float(np.mean(vad_values)) if vad_values else 0.0
            result[f"{prefix}_avg_full"] = float(flags.float().mean())

        return result

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
