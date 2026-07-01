"""
Planning metric utilities for collision detection.
Ported from VAD's metric_stp3.py via reference/Bench2DriveZoo.
"""

import numpy as np
import torch
from typing import Tuple

EGO_WIDTH = 1.85
EGO_LENGTH = 4.084


def polygon_simple(r, c, shape=None):
    """
    Rasterize a polygon to pixel indices using ray-casting.

    Args:
        r: Row coordinates of polygon vertices.
        c: Column coordinates of polygon vertices.
        shape: (max_rows, max_cols) for bounds clipping.

    Returns:
        (rr, cc): Row and column indices of pixels inside the polygon.
    """
    if shape is not None:
        max_r_bound, max_c_bound = shape
    else:
        max_r_bound = int(np.max(r)) + 1
        max_c_bound = int(np.max(c)) + 1

    min_r = max(0, int(np.min(r)))
    min_c = max(0, int(np.min(c)))
    max_r = min(max_r_bound, int(np.max(r)) + 1)
    max_c = min(max_c_bound, int(np.max(c)) + 1)

    rr = []
    cc = []

    n = len(r)
    for i in range(min_r, max_r):
        for j in range(min_c, max_c):
            inside = False
            p1x, p1y = r[0], c[0]
            for k in range(1, n + 1):
                p2x, p2y = r[k % n], c[k % n]
                if j > min(p1y, p2y):
                    if j <= max(p1y, p2y):
                        if i <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (j - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or i <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                rr.append(i)
                cc.append(j)

    return np.array(rr, dtype=np.int32), np.array(cc, dtype=np.int32)


class PlanningMetric:
    """
    Collision metric calculator adapted from STP3/VAD.
    Uses a 200x200 BEV grid at 0.5m resolution covering [-50, 50]m.
    """

    def __init__(self):
        self.X_BOUND = [-50.0, 50.0, 0.5]
        self.Y_BOUND = [-50.0, 50.0, 0.5]
        self.Z_BOUND = [-10.0, 10.0, 20.0]

        dx, bx, _ = self._gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx = dx[:2]
        self.bx = bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self._calc_bev_params(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = EGO_WIDTH
        self.H = EGO_LENGTH

    @staticmethod
    def _gen_dx_bx(xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
        return dx, bx, nx

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

    def get_label(
        self,
        gt_agent_states: np.ndarray,
        gt_agent_labels: np.ndarray,
        num_timesteps: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate BEV occupancy grids from agent states.

        Args:
            gt_agent_states: [B, N, 5] array of (x, y, heading, length, width).
            gt_agent_labels: [B, N] boolean validity mask.
            num_timesteps: Number of future timesteps.

        Returns:
            (segmentation, pedestrian) each [B, T, H, W].
        """
        if gt_agent_states.ndim == 2:
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]

        B = gt_agent_states.shape[0]
        T = num_timesteps
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])

        segmentation = torch.zeros((B, T, H, W))
        pedestrian = torch.zeros((B, T, H, W))

        for b in range(B):
            for agent_idx in range(gt_agent_states.shape[1]):
                if not gt_agent_labels[b, agent_idx]:
                    continue
                x, y, heading, length, width = gt_agent_states[b, agent_idx]
                corners = self._get_agent_corners(x, y, heading, length, width)
                pixel_corners = self._world_to_pixel(corners)

                rr, cc = polygon_simple(
                    pixel_corners[:, 0], pixel_corners[:, 1], shape=(H, W)
                )
                valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                rr, cc = rr[valid], cc[valid]
                if len(rr) > 0:
                    for t in range(T):
                        segmentation[b, t, rr, cc] = 1

        return segmentation, pedestrian

    def evaluate_coll(
        self,
        trajs: np.ndarray,
        gt_trajs: np.ndarray,
        occupancy: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check trajectory collisions against occupancy grid.

        Args:
            trajs: [B, T, 3] predicted trajectories (x, y, heading).
            gt_trajs: [B, T, 3] ground truth (unused in collision check).
            occupancy: [B, T, H, W] occupancy grids.

        Returns:
            (obj_coll, obj_box_coll) each [B, T] boolean tensors.
        """
        if torch.is_tensor(trajs):
            trajs = trajs.cpu().numpy()

        B, T, _ = trajs.shape
        H = int(self.bev_dimension[0])
        W = int(self.bev_dimension[1])

        obj_coll = np.zeros((B, T))
        obj_box_coll = np.zeros((B, T))

        for b in range(B):
            for t in range(T):
                x, y = trajs[b, t, 0], trajs[b, t, 1]
                heading = trajs[b, t, 2] if trajs.shape[2] > 2 else 0.0

                px = int((x - self.bev_start_position[0]) / self.bev_resolution[0])
                py = int((y - self.bev_start_position[1]) / self.bev_resolution[1])

                if 0 <= px < H and 0 <= py < W:
                    occ = occupancy[b, t]
                    if torch.is_tensor(occ):
                        occ = occ.numpy()
                    if occ[px, py] > 0:
                        obj_coll[b, t] = 1

                corners = self._get_agent_corners(x, y, heading, self.H, self.W)
                pixel_corners = self._world_to_pixel(corners)
                rr, cc = polygon_simple(pixel_corners[:, 0], pixel_corners[:, 1], shape=(H, W))
                valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                rr, cc = rr[valid], cc[valid]
                if len(rr) > 0:
                    occ = occupancy[b, t]
                    if torch.is_tensor(occ):
                        occ = occ.numpy()
                    if np.any(occ[rr, cc] > 0):
                        obj_box_coll[b, t] = 1

        return torch.tensor(obj_coll), torch.tensor(obj_box_coll)
