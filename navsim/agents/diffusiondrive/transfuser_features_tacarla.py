"""
Feature and target builders for TaCarla dataset.
Reuses camera/lidar/status logic from B2D builder since sensor formats match.
Target builder handles TaCarla-specific BEV (6 classes).
"""

from typing import Dict, List

import torch

from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
)
from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractTargetBuilder,
)


class TaCarlaFeatureBuilder(Bench2DriveFeatureBuilder):
    """
    Feature builder for TaCarla data.
    Inherits from B2D since camera (900x1600 JPG), LiDAR (.laz),
    and status (command + vel + accel) formats are identical.
    """

    def __init__(self, config: TaCarlaConfig):
        super().__init__(config)


class TaCarlaTargetBuilder(AbstractTargetBuilder):
    """
    Target builder for TaCarla data.
    Handles TaCarla's 6-class BEV maps (vs B2D's 7-class).
    """

    def __init__(self, config: TaCarlaConfig):
        super().__init__()
        self.config = config

    def get_unique_name(self) -> str:
        return "transfuser_target"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        """Build targets from TaCarla scene."""
        targets = {}

        # Current frame (num_history_frames=0, so frame 0 is current)
        current_frame_idx = self.config.num_history_frames if hasattr(self.config, 'num_history_frames') else 0

        # Future trajectory [8, 3]
        trajectory = scene.get_future_trajectory(current_frame_idx)
        if trajectory is None:
            return None
        targets["trajectory"] = trajectory

        # Agent states [30, 5] + labels [30]
        agent_states, agent_labels, agent_types = scene.get_agents(current_frame_idx)
        targets["agent_states"] = agent_states
        targets["agent_labels"] = agent_labels

        # BEV semantic map (TaCarla: 6 classes)
        targets["bev_semantic_map"] = scene.get_bev_semantic_map(current_frame_idx)

        return targets

    @property
    def required_targets(self) -> List[str]:
        return ["trajectory", "agent_states", "agent_labels", "bev_semantic_map"]
