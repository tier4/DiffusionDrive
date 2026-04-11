"""
TaCarla configuration with trajectory normalization support.
Extends Bench2DriveConfig pattern for TaCarla-specific needs.
"""

from dataclasses import dataclass
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig


@dataclass
class TaCarlaConfig(Bench2DriveConfig):
    """TaCarla configuration with dataset-specific trajectory normalization."""

    dataset_type: str = "tacarla"

    # BEV class count (TaCarla has 6 classes, not 7)
    num_bev_classes: int = 6

    # Override anchor path (to be set after generating TaCarla-specific anchors)
    plan_anchor_path: str = "/workspace/DiffusionDrive/download/kmeans_tacarla_traj_20.npy"

    # TaCarla trajectory normalization parameters
    # Placeholder values — must be recomputed from actual TaCarla data
    # using scripts/generate_b2d_plan_anchors.py (adapted for TaCarla cache)
    tc_traj_norm_x_offset: float = 1.0
    tc_traj_norm_x_scale: float = 60.0
    tc_traj_norm_y_offset: float = 30.0
    tc_traj_norm_y_scale: float = 60.0
    tc_traj_norm_heading_offset: float = 2.0
    tc_traj_norm_heading_scale: float = 4.0

    def __post_init__(self):
        if self.dataset_type not in ("navsim", "bench2drive", "tacarla"):
            raise ValueError(
                f"Invalid dataset_type: {self.dataset_type}. "
                "Must be 'navsim', 'bench2drive', or 'tacarla'."
            )

    def get_normalization_params(self):
        """Get the appropriate normalization parameters based on dataset_type."""
        if self.dataset_type == "tacarla":
            return {
                "x_offset": self.tc_traj_norm_x_offset,
                "x_scale": self.tc_traj_norm_x_scale,
                "y_offset": self.tc_traj_norm_y_offset,
                "y_scale": self.tc_traj_norm_y_scale,
                "heading_offset": self.tc_traj_norm_heading_offset,
                "heading_scale": self.tc_traj_norm_heading_scale,
            }
        return super().get_normalization_params()
