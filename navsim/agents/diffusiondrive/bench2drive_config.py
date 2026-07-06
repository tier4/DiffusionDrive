"""
Bench2Drive configuration with trajectory normalization support.
This extends the original TransfuserConfig for B2D-specific needs.
"""

from dataclasses import dataclass
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


@dataclass
class Bench2DriveConfig(TransfuserConfig):
    """Bench2Drive configuration with dataset-specific trajectory normalization."""

    # Dataset type flag
    dataset_type: str = "bench2drive"  # "navsim" or "bench2drive"

    # Explicit spatial params matching 64m crop (same as parent TransfuserConfig,
    # but stated explicitly to document that B2D data is cropped to this range)
    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    pixels_per_meter: float = 4.0

    # Override default anchor paths with correct locations
    plan_anchor_path: str = "/mnt/nvme1/diffusiondrive/weights/kmeans_b2d_v2_traj_20.npy"

    # Trajectory normalization parameters
    # NavSim ranges (default)
    traj_norm_x_offset: float = 1.2  # -x_min
    traj_norm_x_scale: float = 56.9  # x_max - x_min
    traj_norm_y_offset: float = 20.0  # -y_min
    traj_norm_y_scale: float = 46.0  # y_max - y_min
    traj_norm_heading_offset: float = 2.0  # -heading_min
    traj_norm_heading_scale: float = 3.9  # heading_max - heading_min

    # Bench2Drive ranges (to be used when dataset_type='bench2drive')
    # Generated from kmeans_b2d_v2_traj_20_metadata.json with 35942 samples
    # Actual data ranges: X:[-0.67, 58.94], Y:[-32.96, 31.65], Heading:[-2.31, 2.34]
    b2d_traj_norm_x_offset: float = 0.6710353515014572  # -x_min
    b2d_traj_norm_x_scale: float = 59.60934967542077  # x_max - x_min
    b2d_traj_norm_y_offset: float = 32.956087756550545  # -y_min
    b2d_traj_norm_y_scale: float = 64.60912817798321  # y_max - y_min
    b2d_traj_norm_heading_offset: float = 2.310837927351096  # -heading_min
    b2d_traj_norm_heading_scale: float = 4.6519694452598985  # heading_max - heading_min

    def __post_init__(self):
        if self.dataset_type not in ("navsim", "bench2drive"):
            raise ValueError(
                f"Invalid dataset_type: {self.dataset_type}. "
                "Must be 'navsim' or 'bench2drive'."
            )

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
