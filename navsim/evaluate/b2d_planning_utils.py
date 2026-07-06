"""
Planning metric utilities for collision detection.
Ported from VAD's metric_stp3.py via reference/Bench2DriveZoo.
"""

import cv2
import numpy as np
import torch
from typing import Tuple

EGO_WIDTH = 1.85
EGO_LENGTH = 4.084


class PlanningMetric:
    """
    Collision metric calculator adapted from STP3/VAD.
    Uses a 200x200 BEV grid at 0.5m resolution covering [-50, 50]m.
    """

    def __init__(self):
        self.X_BOUND = [-50.0, 50.0, 0.5]
        self.Y_BOUND = [-50.0, 50.0, 0.5]
        self.Z_BOUND = [-10.0, 10.0, 20.0]

        bev_resolution, bev_start_position, bev_dimension = self._calc_bev_params(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = EGO_WIDTH
        self.H = EGO_LENGTH

    @staticmethod
    def _calc_bev_params(x_bounds, y_bounds, z_bounds):
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor(
            [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_dimension = torch.tensor(
            [int((row[1] - row[0]) / row[2]) for row in [x_bounds, y_bounds, z_bounds]],
            dtype=torch.long,
        )
        return bev_resolution, bev_start_position, bev_dimension

    def _get_agent_corners(self, x, y, heading, length, width):
        """Get four corners of an agent bounding box in world coords."""
        half_l = length / 2.0
        half_w = width / 2.0
        corners_local = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return corners_local @ rot.T + np.array([x, y])

    def _world_to_pixel(self, points):
        """Convert world coordinates to pixel indices on the BEV grid."""
        pixels = np.zeros_like(points)
        pixels[:, 0] = (points[:, 0] - self.bev_start_position[0]) / self.bev_resolution[0]
        pixels[:, 1] = (points[:, 1] - self.bev_start_position[1]) / self.bev_resolution[1]
        return np.round(pixels).astype(np.int32)

    def _rasterize_polygon(self, pixel_corners, height, width):
        """Fill a polygon on a (height, width) canvas.

        Args:
            pixel_corners: [4, 2] integer corners as (row, col).

        Returns:
            (rr, cc): int32 arrays of filled pixel indices, clipped to canvas.
        """
        canvas = np.zeros((height, width), dtype=np.uint8)
        # cv2 points are (x=col, y=row)
        cv2.fillPoly(canvas, [pixel_corners[:, [1, 0]].astype(np.int32)], 1)
        rr, cc = np.nonzero(canvas)
        return rr.astype(np.int32), cc.astype(np.int32)

    def get_label(
        self,
        gt_agent_states: np.ndarray,
        gt_agent_labels: np.ndarray,
        num_timesteps: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate per-timestep BEV occupancy grids from agent states.

        Matches STP3/VAD get_birds_eye_view_label: agents are rasterized at
        their position AT each timestep (metric_stp3.py:118-133), never
        frozen at t=0.

        Args:
            gt_agent_states: [B, T, N, 5] or [T, N, 5] of
                (x, y, heading, length, width) in the current ego frame. The
                3-D form is only accepted when its leading dimension equals
                ``num_timesteps`` exactly, to avoid misreading a legacy
                static [B, N, 5] batch as per-timestep states.
            gt_agent_labels: [B, T, N] or [T, N] boolean validity mask.
            num_timesteps: Number of future timesteps T to rasterize.

        Returns:
            (segmentation, pedestrian) each [B, T, H, W]. The pedestrian
            tensor is always zeros (walkers are merged into segmentation
            upstream); it is kept for VAD API compatibility only.
        """
        gt_agent_states = np.asarray(gt_agent_states)
        gt_agent_labels = np.asarray(gt_agent_labels)
        if gt_agent_states.ndim == 3:
            if gt_agent_states.shape[0] != num_timesteps:
                raise ValueError(
                    "gt_agent_states with ndim==3 must be [T, N, 5] with T "
                    f"exactly equal to num_timesteps ({num_timesteps}); got "
                    f"shape {gt_agent_states.shape}. Legacy static [B, N, 5] "
                    "input is not accepted here — per-timestep states are "
                    "required. If you need a batch, pass an explicit 4-D "
                    "[B, T, N, 5] array instead."
                )
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]
        if gt_agent_states.ndim != 4:
            raise ValueError(
                "gt_agent_states must be per-timestep [B, T, N, 5] or [T, N, 5]; "
                f"got shape {gt_agent_states.shape}. Static [N, 5] states would "
                "freeze agents at t=0 and invalidate the collision metric."
            )
        if gt_agent_states.shape[1] < num_timesteps:
            raise ValueError(
                f"Need agent states for {num_timesteps} timesteps, "
                f"got {gt_agent_states.shape[1]}"
            )

        B = gt_agent_states.shape[0]
        T = num_timesteps
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])

        segmentation = torch.zeros((B, T, H, W))
        pedestrian = torch.zeros((B, T, H, W))

        for b in range(B):
            for t in range(T):
                for agent_idx in range(gt_agent_states.shape[2]):
                    if not gt_agent_labels[b, t, agent_idx]:
                        continue
                    x, y, heading, length, width = gt_agent_states[b, t, agent_idx]
                    corners = self._get_agent_corners(x, y, heading, length, width)
                    pixel_corners = self._world_to_pixel(corners)
                    rr, cc = self._rasterize_polygon(pixel_corners, H, W)
                    if len(rr) > 0:
                        segmentation[b, t, rr, cc] = 1

        return segmentation, pedestrian

    def _traj_point_to_pixel(self, x, y):
        """Quantize a trajectory point with the SAME rounding as _world_to_pixel."""
        px = int(np.round((x - self.bev_start_position[0]) / self.bev_resolution[0]))
        py = int(np.round((y - self.bev_start_position[1]) / self.bev_resolution[1]))
        return px, py

    def _box_collision_flags(self, traj, occupancy):
        """Per-timestep ego-box collision flags for one trajectory.

        Args:
            traj: [T, >=2] (x, y[, heading]) trajectory.
            occupancy: [T, H, W] occupancy grid (tensor or ndarray).

        Returns:
            np.ndarray [T] bool.
        """
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])
        T = traj.shape[0]
        flags = np.zeros(T, dtype=bool)
        for t in range(T):
            x, y = traj[t, 0], traj[t, 1]
            heading = traj[t, 2] if traj.shape[1] > 2 else 0.0
            corners = self._get_agent_corners(x, y, heading, self.H, self.W)
            pixel_corners = self._world_to_pixel(corners)
            rr, cc = self._rasterize_polygon(pixel_corners, H, W)
            if len(rr) > 0:
                occ = occupancy[t]
                if torch.is_tensor(occ):
                    occ = occ.numpy()
                flags[t] = bool(np.any(occ[rr, cc] > 0))
        return flags

    def evaluate_coll(
        self,
        trajs: np.ndarray,
        gt_trajs: np.ndarray,
        occupancy: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check trajectory collisions against per-timestep occupancy grids.

        STP3/VAD semantics (metric_stp3.py:267-287): timesteps where the GT
        trajectory itself box-collides are masked out of BOTH the point and
        box collision flags (annotation noise / unavoidable-contact filter).

        Args:
            trajs: [B, T, 3] predicted trajectories (x, y, heading).
            gt_trajs: [B, T, 3] ground truth — used for collision masking.
            occupancy: [B, T, H, W] occupancy grids.

        Returns:
            (obj_coll, obj_box_coll) each [B, T] {0,1} tensors.
        """
        if torch.is_tensor(trajs):
            trajs = trajs.cpu().numpy()
        if torch.is_tensor(gt_trajs):
            gt_trajs = gt_trajs.cpu().numpy()

        B, T, _ = trajs.shape
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])

        obj_coll = np.zeros((B, T))
        obj_box_coll = np.zeros((B, T))

        for b in range(B):
            occ_b = occupancy[b]
            gt_box_coll = self._box_collision_flags(gt_trajs[b], occ_b)

            for t in range(T):
                if gt_box_coll[t]:
                    continue  # STP3 m1 mask
                x, y = trajs[b, t, 0], trajs[b, t, 1]
                px, py = self._traj_point_to_pixel(x, y)
                if 0 <= px < H and 0 <= py < W:
                    occ = occ_b[t]
                    if torch.is_tensor(occ):
                        occ = occ.numpy()
                    if occ[px, py] > 0:
                        obj_coll[b, t] = 1

            pred_box_coll = self._box_collision_flags(trajs[b], occ_b)
            # STP3 m2 mask
            obj_box_coll[b] = np.logical_and(
                pred_box_coll, np.logical_not(gt_box_coll)
            ).astype(float)

        return torch.tensor(obj_coll), torch.tensor(obj_box_coll)
