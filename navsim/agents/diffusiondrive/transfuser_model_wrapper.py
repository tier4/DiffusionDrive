"""
Wrapper for V2TransfuserModel that adds trajectory normalization support.
This allows using different normalization for different datasets without modifying the original model.
"""

import torch
from typing import Dict
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from navsim.agents.diffusiondrive.trajectory_normalizer import TrajectoryNormalizer
from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig
import logging

logger = logging.getLogger(__name__)


class V2TransfuserModelWrapper(V2TransfuserModel):
    """Wrapper that adds trajectory normalization to V2TransfuserModel."""

    def __init__(self, config: ExtendedTransfuserConfig):
        """Initialize with extended config."""
        # Initialize parent with base config
        super().__init__(config)

        # Store extended config
        self.extended_config = config

        # Initialize trajectory normalizer
        self.trajectory_normalizer = TrajectoryNormalizer(dataset_type=config.dataset_type)
        
        # Get normalization parameters
        norm_params = config.get_normalization_params()
        
        # Override the trajectory head's norm_odo and denorm_odo methods
        # Store original methods for reference
        self._trajectory_head._orig_norm_odo = self._trajectory_head.norm_odo
        self._trajectory_head._orig_denorm_odo = self._trajectory_head.denorm_odo
        
        # Create new normalization methods using config parameters
        def norm_odo_override(odo_info_fut):
            """Override norm_odo with config-based normalization."""
            odo_info_fut_x = odo_info_fut[..., 0:1]
            odo_info_fut_y = odo_info_fut[..., 1:2]
            odo_info_fut_head = odo_info_fut[..., 2:3] if odo_info_fut.shape[-1] > 2 else torch.zeros_like(odo_info_fut_x)
            
            # Normalize using config parameters
            odo_info_fut_x = 2 * (odo_info_fut_x + norm_params["x_offset"]) / norm_params["x_scale"] - 1
            odo_info_fut_y = 2 * (odo_info_fut_y + norm_params["y_offset"]) / norm_params["y_scale"] - 1
            odo_info_fut_head = 2 * (odo_info_fut_head + norm_params["heading_offset"]) / norm_params["heading_scale"] - 1
            
            # Clamp to ensure values are in [-1, 1]
            odo_info_fut_x = torch.clamp(odo_info_fut_x, min=-1.0, max=1.0)
            odo_info_fut_y = torch.clamp(odo_info_fut_y, min=-1.0, max=1.0)
            odo_info_fut_head = torch.clamp(odo_info_fut_head, min=-1.0, max=1.0)
            
            if odo_info_fut.shape[-1] > 2:
                return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
            else:
                return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        
        def denorm_odo_override(odo_info_fut):
            """Override denorm_odo with config-based denormalization."""
            odo_info_fut_x = odo_info_fut[..., 0:1]
            odo_info_fut_y = odo_info_fut[..., 1:2]
            odo_info_fut_head = odo_info_fut[..., 2:3] if odo_info_fut.shape[-1] > 2 else torch.zeros_like(odo_info_fut_x)
            
            # Denormalize using config parameters
            odo_info_fut_x = (odo_info_fut_x + 1) / 2 * norm_params["x_scale"] - norm_params["x_offset"]
            odo_info_fut_y = (odo_info_fut_y + 1) / 2 * norm_params["y_scale"] - norm_params["y_offset"]
            odo_info_fut_head = (odo_info_fut_head + 1) / 2 * norm_params["heading_scale"] - norm_params["heading_offset"]
            
            if odo_info_fut.shape[-1] > 2:
                return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
            else:
                return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        
        # Replace the methods
        self._trajectory_head.norm_odo = norm_odo_override
        self._trajectory_head.denorm_odo = denorm_odo_override
        
        logger.info(f"Initialized model wrapper with {config.dataset_type} normalization:")
        logger.info(f"  X: offset={norm_params['x_offset']}, scale={norm_params['x_scale']}")
        logger.info(f"  Y: offset={norm_params['y_offset']}, scale={norm_params['y_scale']}")
        logger.info(f"  Heading: offset={norm_params['heading_offset']}, scale={norm_params['heading_scale']}")


def create_diffusiondrive_model(config: ExtendedTransfuserConfig) -> V2TransfuserModelWrapper:
    """Factory function to create the model with proper configuration."""
    return V2TransfuserModelWrapper(config)
