"""
Unified trajectory normalization system for DiffusionDrive.
Handles normalization for different datasets with automatic detection and validation.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)


class TrajectoryNormalizer:
    """
    Handles trajectory normalization for different datasets.

    Supports automatic dataset detection, percentile-based normalization,
    and safety checks to ensure outputs are in [-1, 1] range.
    """

    # Default normalization profiles for known datasets
    NORMALIZATION_PROFILES = {
        "navsim": {
            "x": {"offset": 1.2, "scale": 56.9},
            "y": {"offset": 20.0, "scale": 46.0},
            "heading": {"offset": 2.0, "scale": 3.9},
            "characteristics": "forward-biased, real-world driving",
            "verified": True,  # These parameters are verified to work well
        },
        "bench2drive": {
            # Based on empirical analysis: X:[-34.6, 32.5], Y:[-32.4, 32.8]
            "x": {"offset": 35.0, "scale": 70.0},  # Covers full range with margin
            "y": {"offset": 35.0, "scale": 70.0},  # Symmetric with X
            "heading": {"offset": 0.05, "scale": 0.5},  # Small heading variation
            "characteristics": "centered, simulation-based",
            "verified": True,
        },
        "custom": {
            "x": {"offset": 0.0, "scale": 1.0},
            "y": {"offset": 0.0, "scale": 1.0},
            "heading": {"offset": 0.0, "scale": 1.0},
            "characteristics": "user-defined",
            "verified": False,
        },
    }

    def __init__(self, dataset_type: str = "auto", config_path: Optional[str] = None):
        """
        Initialize the normalizer.

        Args:
            dataset_type: 'navsim', 'bench2drive', 'custom', or 'auto' for detection
            config_path: Path to custom normalization config JSON
        """
        self.dataset_type = dataset_type
        self.config_path = config_path
        self.normalization_params = None

        # Load normalization parameters
        if dataset_type == "auto":
            logger.info("Auto-detection mode enabled for trajectory normalization")
            self.normalization_params = None  # Will be set on first use
        elif dataset_type in self.NORMALIZATION_PROFILES:
            self.normalization_params = self.NORMALIZATION_PROFILES[dataset_type].copy()
            logger.info(f"Using {dataset_type} normalization profile")
        elif config_path:
            self.load_custom_config(config_path)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

    def detect_dataset_from_path(self, path: str) -> str:
        """
        Detect dataset type from file path or cache path.

        Args:
            path: File or directory path

        Returns:
            Detected dataset type
        """
        path_lower = str(path).lower()

        if "bench2drive" in path_lower or "b2d" in path_lower:
            return "bench2drive"
        elif "navsim" in path_lower or "navtrain" in path_lower or "navtest" in path_lower:
            return "navsim"
        else:
            logger.warning(f"Could not detect dataset type from path: {path}")
            return "custom"

    def fit_from_data(
        self, trajectories: Union[np.ndarray, torch.Tensor], percentile: float = 95.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute normalization parameters from trajectory data.

        Args:
            trajectories: Shape (N, T, 3) or (N*T, 3) with (x, y, heading)
            percentile: Percentile to use for robust bounds (default: 95)

        Returns:
            Dictionary of normalization parameters
        """
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.cpu().numpy()

        # Reshape if needed
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, 3)

        params = {}

        # Compute robust bounds for each dimension
        for i, dim in enumerate(["x", "y", "heading"]):
            values = trajectories[:, i]

            # Use percentiles to avoid outliers
            low_p = (100 - percentile) / 2
            high_p = 100 - low_p
            low_val, high_val = np.percentile(values, [low_p, high_p])

            # Compute center and range
            center = (low_val + high_val) / 2
            range_val = high_val - low_val

            # Add safety margin (10%)
            margin = 0.1 * range_val if range_val > 0 else 1.0

            # Ensure minimum scale for numerical stability
            min_scales = {"x": 1.0, "y": 1.0, "heading": 0.5}
            scale = max(range_val + margin, min_scales[dim])

            # Offset to center the range
            offset = -low_val + margin / 2

            params[dim] = {
                "offset": float(offset),
                "scale": float(scale),
                "center": float(center),
                "range": float(range_val),
                "percentile_low": float(low_val),
                "percentile_high": float(high_val),
            }

        # Log statistics
        logger.info("Computed normalization parameters from data:")
        for dim, p in params.items():
            logger.info(
                f"  {dim}: offset={p['offset']:.3f}, scale={p['scale']:.3f}, "
                f"range=[{p['percentile_low']:.3f}, {p['percentile_high']:.3f}]"
            )

        return params

    def normalize(
        self, trajectories: Union[np.ndarray, torch.Tensor], clamp: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize trajectories to [-1, 1] range.

        Args:
            trajectories: Input trajectories with shape (..., 3) where last dim is (x, y, heading)
            clamp: Whether to clamp output to [-1, 1]

        Returns:
            Normalized trajectories with same shape and type as input
        """
        if self.normalization_params is None:
            raise ValueError(
                "Normalization parameters not set. Call fit_from_data() first or specify dataset_type."
            )

        is_torch = isinstance(trajectories, torch.Tensor)
        if is_torch:
            device = trajectories.device
            trajectories_np = trajectories.cpu().numpy()
        else:
            trajectories_np = trajectories

        # Normalize each dimension
        normalized = np.zeros_like(trajectories_np)
        for i, dim in enumerate(["x", "y", "heading"]):
            params = self.normalization_params[dim]
            normalized[..., i] = (
                2 * (trajectories_np[..., i] + params["offset"]) / params["scale"] - 1
            )

        # Clamp to [-1, 1] if requested
        if clamp:
            normalized = np.clip(normalized, -1, 1)

        # Check for invalid values
        if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
            logger.warning("NaN or Inf detected in normalized trajectories!")

        # Return in same format as input
        if is_torch:
            return torch.from_numpy(normalized).to(device)
        else:
            return normalized

    def denormalize(
        self, normalized: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize trajectories from [-1, 1] back to original range.

        Args:
            normalized: Normalized trajectories

        Returns:
            Denormalized trajectories
        """
        if self.normalization_params is None:
            raise ValueError("Normalization parameters not set.")

        is_torch = isinstance(normalized, torch.Tensor)
        if is_torch:
            device = normalized.device
            normalized_np = normalized.cpu().numpy()
        else:
            normalized_np = normalized

        # Denormalize each dimension
        denormalized = np.zeros_like(normalized_np)
        for i, dim in enumerate(["x", "y", "heading"]):
            params = self.normalization_params[dim]
            denormalized[..., i] = (normalized_np[..., i] + 1) / 2 * params["scale"] - params[
                "offset"
            ]

        # Return in same format as input
        if is_torch:
            return torch.from_numpy(denormalized).to(device)
        else:
            return denormalized

    def load_custom_config(self, config_path: str):
        """Load custom normalization configuration from JSON file."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Ensure the config has the expected structure
        if "x" in config and isinstance(config["x"], dict):
            self.normalization_params = config
        else:
            # Handle flat config format
            self.normalization_params = {"x": config, "y": config, "heading": config}
        logger.info(f"Loaded custom normalization config from {config_path}")

    def save_config(self, save_path: str):
        """Save current normalization configuration to JSON file."""
        if self.normalization_params is None:
            raise ValueError("No normalization parameters to save")

        with open(save_path, "w") as f:
            json.dump(self.normalization_params, f, indent=2)

        logger.info(f"Saved normalization config to {save_path}")

    def get_config_dict(self) -> Dict:
        """Get normalization parameters as dictionary for model config."""
        if self.normalization_params is None:
            raise ValueError("Normalization parameters not set")

        return {
            f"traj_norm_{dim}_{key}": value
            for dim in ["x", "y", "heading"]
            for key, value in self.normalization_params[dim].items()
            if key in ["offset", "scale"]
        }

    def validate_normalization(
        self, trajectories: Union[np.ndarray, torch.Tensor], tolerance: float = 0.01
    ) -> Dict[str, float]:
        """
        Validate that normalization produces values in [-1, 1].

        Args:
            trajectories: Test trajectories
            tolerance: Allowed deviation from [-1, 1]

        Returns:
            Dictionary with validation metrics
        """
        normalized = self.normalize(trajectories, clamp=False)

        if isinstance(normalized, torch.Tensor):
            normalized = normalized.cpu().numpy()

        metrics = {}
        for i, dim in enumerate(["x", "y", "heading"]):
            values = normalized[..., i].flatten()

            metrics[f"{dim}_min"] = float(values.min())
            metrics[f"{dim}_max"] = float(values.max())
            metrics[f"{dim}_out_of_range"] = float(
                np.sum(np.abs(values) > 1 + tolerance) / len(values)
            )
            metrics[f"{dim}_mean"] = float(values.mean())
            metrics[f"{dim}_std"] = float(values.std())

        # Overall metrics
        all_values = normalized.reshape(-1)
        metrics["overall_out_of_range"] = float(
            np.sum(np.abs(all_values) > 1 + tolerance) / len(all_values)
        )
        metrics["has_nan"] = bool(np.any(np.isnan(all_values)))
        metrics["has_inf"] = bool(np.any(np.isinf(all_values)))

        return metrics
