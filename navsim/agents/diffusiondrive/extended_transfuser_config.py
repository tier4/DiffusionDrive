"""
Extended TransFuser configuration with trajectory normalization support.
This extends the original TransfuserConfig without modifying it.
"""

from dataclasses import dataclass
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


@dataclass
class ExtendedTransfuserConfig(TransfuserConfig):
    """Extended configuration with dataset-specific trajectory normalization."""

    # Dataset type flag
    dataset_type: str = "navsim"  # "navsim" or "bench2drive"
    
    # Override default anchor paths with correct locations
    plan_anchor_path: str = "/workspace/DiffusionDrive/download/kmeans_navsim_traj_20.npy"

    # Trajectory normalization parameters
    # NavSim ranges (default)
    traj_norm_x_offset: float = 1.2  # -x_min
    traj_norm_x_scale: float = 56.9  # x_max - x_min
    traj_norm_y_offset: float = 20.0  # -y_min
    traj_norm_y_scale: float = 46.0  # y_max - y_min
    traj_norm_heading_offset: float = 2.0  # -heading_min
    traj_norm_heading_scale: float = 3.9  # heading_max - heading_min

    # Bench2Drive ranges (to be used when dataset_type='bench2drive')
    # Based on comprehensive analysis of 5000 samples
    # Full data range: X:[-35.58, 33.66], Y:[-36.84, 34.61]
    # Using proper normalization to ensure all values fit in [-1, 1]
    b2d_traj_norm_x_offset: float = 37.5    # Shift to make range positive: -(-37.5) = 37.5
    b2d_traj_norm_x_scale: float = 75.0     # Total range with margin: 2 * 37.5 = 75
    b2d_traj_norm_y_offset: float = 37.5    # Symmetric with X for simplicity
    b2d_traj_norm_y_scale: float = 75.0     # Total range with margin
    b2d_traj_norm_heading_offset: float = 0.12      # Shift to make range positive
    b2d_traj_norm_heading_scale: float = 0.24       # Total range: 2 * 0.12

    def get_normalization_params(self):
        """Get the appropriate normalization parameters based on dataset_type."""
        if self.dataset_type == "bench2drive":
            return {
                "x_offset": self.b2d_traj_norm_x_offset,
                "x_scale": self.b2d_traj_norm_x_scale,
                "y_offset": self.b2d_traj_norm_y_offset,
                "y_scale": self.b2d_traj_norm_y_scale,
                "heading_offset": self.b2d_traj_norm_heading_offset,
                "heading_scale": self.b2d_traj_norm_heading_scale,
            }
        else:  # default to navsim
            return {
                "x_offset": self.traj_norm_x_offset,
                "x_scale": self.traj_norm_x_scale,
                "y_offset": self.traj_norm_y_offset,
                "y_scale": self.traj_norm_y_scale,
                "heading_offset": self.traj_norm_heading_offset,
                "heading_scale": self.traj_norm_heading_scale,
            }
