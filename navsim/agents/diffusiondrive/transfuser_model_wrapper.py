"""
Wrapper for V2TransfuserModel that adds trajectory normalization support.
This allows using different normalization for different datasets without modifying the original model.
"""

import torch
import torch.nn as nn
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig


class V2TransfuserModelWrapper(V2TransfuserModel):
    """Wrapper that adds dataset-specific trajectory normalization to V2TransfuserModel.

    Instead of monkey-patching norm_odo/denorm_odo, this registers normalization
    parameters as buffers on the trajectory head. Buffers survive deepcopy, DDP,
    EMA, and state_dict save/load.
    """

    def __init__(self, config: Bench2DriveConfig):
        """Initialize with extended config."""
        super().__init__(config)
        self.extended_config = config

        # Get normalization parameters and register as buffers on the trajectory head
        norm_params = config.get_normalization_params()

        head = self._trajectory_head
        head.register_buffer("_norm_x_offset", torch.tensor(norm_params["x_offset"], dtype=torch.float32))
        head.register_buffer("_norm_x_scale", torch.tensor(norm_params["x_scale"], dtype=torch.float32))
        head.register_buffer("_norm_y_offset", torch.tensor(norm_params["y_offset"], dtype=torch.float32))
        head.register_buffer("_norm_y_scale", torch.tensor(norm_params["y_scale"], dtype=torch.float32))
        head.register_buffer("_norm_heading_offset", torch.tensor(norm_params["heading_offset"], dtype=torch.float32))
        head.register_buffer("_norm_heading_scale", torch.tensor(norm_params["heading_scale"], dtype=torch.float32))

        # Override norm_odo and denorm_odo with buffer-based versions
        # We bind these as proper methods so they move with the module
        import types
        head.norm_odo = types.MethodType(_norm_odo, head)
        head.denorm_odo = types.MethodType(_denorm_odo, head)


def _norm_odo(self, odo_info_fut):
    """Normalize trajectory using registered buffers."""
    x = odo_info_fut[..., 0:1]
    y = odo_info_fut[..., 1:2]
    head = odo_info_fut[..., 2:3] if odo_info_fut.shape[-1] > 2 else torch.zeros_like(x)

    # No clamping here -- matches original norm_odo behavior in transfuser_model_v2.py
    # Clamping would zero gradients for extreme maneuvers in B2D training data
    x = 2 * (x + self._norm_x_offset) / self._norm_x_scale - 1
    y = 2 * (y + self._norm_y_offset) / self._norm_y_scale - 1
    head = 2 * (head + self._norm_heading_offset) / self._norm_heading_scale - 1

    if odo_info_fut.shape[-1] > 2:
        return torch.cat([x, y, head], dim=-1)
    return torch.cat([x, y], dim=-1)


def _denorm_odo(self, odo_info_fut):
    """Denormalize trajectory using registered buffers."""
    x = odo_info_fut[..., 0:1]
    y = odo_info_fut[..., 1:2]
    head = odo_info_fut[..., 2:3] if odo_info_fut.shape[-1] > 2 else torch.zeros_like(x)

    x = (x + 1) / 2 * self._norm_x_scale - self._norm_x_offset
    y = (y + 1) / 2 * self._norm_y_scale - self._norm_y_offset
    head = (head + 1) / 2 * self._norm_heading_scale - self._norm_heading_offset

    if odo_info_fut.shape[-1] > 2:
        return torch.cat([x, y, head], dim=-1)
    return torch.cat([x, y], dim=-1)


def create_diffusiondrive_model(config: Bench2DriveConfig) -> V2TransfuserModelWrapper:
    """Factory function to create the model with proper configuration."""
    return V2TransfuserModelWrapper(config)
