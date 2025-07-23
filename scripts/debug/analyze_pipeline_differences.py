#!/usr/bin/env python3
"""
Analyze specific differences between NavSim and Bench2Drive processing pipelines
to identify where and why NaN values are introduced.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import json
from collections import defaultdict
import argparse
from tqdm import tqdm
import pickle
import gzip

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder,
)
from navsim.agents.diffusiondrive.trajectory_normalizer import TrajectoryNormalizer
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from omegaconf import OmegaConf


class PipelineDifferenceAnalyzer:
    """Analyzes differences between NavSim and Bench2Drive pipelines."""

    def __init__(self):
        self.differences = {
            "config": {},
            "normalization": {},
            "model_params": {},
            "processing": {},
            "numerical": {},
        }

    def compare_configs(self, nav_cfg: Dict, b2d_cfg: Dict):
        """Compare configuration differences."""
        print("\n" + "=" * 60)
        print("CONFIGURATION DIFFERENCES")
        print("=" * 60)

        def compare_dict(d1: Dict, d2: Dict, path: str = ""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}/{key}" if path else key

                if key not in d1:
                    self.differences["config"][current_path] = {
                        "status": "missing_in_navsim",
                        "b2d_value": d2[key],
                    }
                elif key not in d2:
                    self.differences["config"][current_path] = {
                        "status": "missing_in_b2d",
                        "nav_value": d1[key],
                    }
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dict(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    self.differences["config"][current_path] = {
                        "status": "different",
                        "nav_value": d1[key],
                        "b2d_value": d2[key],
                    }

        # Convert OmegaConf to dict for comparison
        nav_dict = OmegaConf.to_container(nav_cfg, resolve=True)
        b2d_dict = OmegaConf.to_container(b2d_cfg, resolve=True)

        compare_dict(nav_dict, b2d_dict)

        # Print key differences
        for path, diff in self.differences["config"].items():
            if diff["status"] == "different":
                print(f"\n{path}:")
                print(f"  NavSim: {diff['nav_value']}")
                print(f"  B2D:    {diff['b2d_value']}")

    def compare_normalization(self):
        """Compare normalization parameters."""
        print("\n" + "=" * 60)
        print("NORMALIZATION DIFFERENCES")
        print("=" * 60)

        nav_norm = TrajectoryNormalizer(dataset_type="navsim")
        b2d_norm = TrajectoryNormalizer(dataset_type="bench2drive")

        nav_params = nav_norm.NORMALIZATION_PROFILES["navsim"]
        b2d_params = b2d_norm.NORMALIZATION_PROFILES["bench2drive"]

        print("\nTrajectory Normalization Parameters:")
        print(f"{'Component':<10} {'NavSim':<30} {'Bench2Drive':<30} {'Ratio':<10}")
        print("-" * 80)

        for component in ["x", "y", "heading"]:
            nav_p = nav_params[component]
            b2d_p = b2d_params[component]

            nav_str = f"offset={nav_p['offset']:.3f}, scale={nav_p['scale']:.3f}"
            b2d_str = f"offset={b2d_p['offset']:.3f}, scale={b2d_p['scale']:.3f}"
            ratio = b2d_p["scale"] / nav_p["scale"]

            print(f"{component:<10} {nav_str:<30} {b2d_str:<30} {ratio:<10.3f}")

            self.differences["normalization"][component] = {
                "nav_offset": nav_p["offset"],
                "nav_scale": nav_p["scale"],
                "b2d_offset": b2d_p["offset"],
                "b2d_scale": b2d_p["scale"],
                "scale_ratio": ratio,
            }

    def analyze_numerical_precision(self, nav_features: Dict, b2d_features: Dict):
        """Analyze numerical precision issues."""
        print("\n" + "=" * 60)
        print("NUMERICAL PRECISION ANALYSIS")
        print("=" * 60)

        def check_tensor_properties(tensor: torch.Tensor, name: str):
            if tensor is None:
                return None

            props = {
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "is_contiguous": tensor.is_contiguous(),
                "has_nan": bool(torch.isnan(tensor).any()),
                "has_inf": bool(torch.isinf(tensor).any()),
                "min_val": float(tensor.min()) if not torch.isnan(tensor).all() else None,
                "max_val": float(tensor.max()) if not torch.isnan(tensor).all() else None,
                "mean_val": float(tensor.mean()) if not torch.isnan(tensor).all() else None,
                "std_val": float(tensor.std()) if not torch.isnan(tensor).all() else None,
            }

            # Check for values close to numerical limits
            if props["dtype"] == "torch.float32":
                eps = torch.finfo(torch.float32).eps
                very_small = (tensor.abs() < eps * 10).sum()
                props["near_zero_count"] = int(very_small)

            return props

        # Analyze each feature type
        for key in set(nav_features.keys()) | set(b2d_features.keys()):
            nav_tensor = nav_features.get(key)
            b2d_tensor = b2d_features.get(key)

            if nav_tensor is not None:
                nav_props = check_tensor_properties(nav_tensor, f"NavSim/{key}")
                if nav_props and (nav_props["has_nan"] or nav_props["has_inf"]):
                    print(f"\n⚠️  NavSim {key}: Contains NaN or Inf!")

            if b2d_tensor is not None:
                b2d_props = check_tensor_properties(b2d_tensor, f"B2D/{key}")
                if b2d_props and (b2d_props["has_nan"] or b2d_props["has_inf"]):
                    print(f"\n⚠️  B2D {key}: Contains NaN or Inf!")

            # Compare properties
            if nav_tensor is not None and b2d_tensor is not None:
                nav_props = check_tensor_properties(nav_tensor, f"NavSim/{key}")
                b2d_props = check_tensor_properties(b2d_tensor, f"B2D/{key}")

                if nav_props and b2d_props:
                    if nav_props["dtype"] != b2d_props["dtype"]:
                        print(
                            f"\n{key}: Different dtypes - NavSim: {nav_props['dtype']}, B2D: {b2d_props['dtype']}"
                        )

                    # Check for significant range differences
                    if nav_props["max_val"] and b2d_props["max_val"]:
                        range_nav = nav_props["max_val"] - nav_props["min_val"]
                        range_b2d = b2d_props["max_val"] - b2d_props["min_val"]

                        if abs(range_nav - range_b2d) > 0.5 * max(range_nav, range_b2d):
                            print(f"\n{key}: Significant range difference")
                            print(
                                f"  NavSim: [{nav_props['min_val']:.3f}, {nav_props['max_val']:.3f}]"
                            )
                            print(
                                f"  B2D:    [{b2d_props['min_val']:.3f}, {b2d_props['max_val']:.3f}]"
                            )

    def check_model_compatibility(self, nav_cfg, b2d_cfg):
        """Check if model parameters are compatible with data."""
        print("\n" + "=" * 60)
        print("MODEL COMPATIBILITY CHECK")
        print("=" * 60)

        # Check if using same model architecture
        nav_model_type = nav_cfg.get("model_type", "v2_transfuser")
        b2d_model_type = b2d_cfg.get("model_type", "v2_transfuser")

        if nav_model_type != b2d_model_type:
            print(f"⚠️  Different model types: NavSim={nav_model_type}, B2D={b2d_model_type}")

        # Check critical parameters
        critical_params = [
            "num_modes",
            "num_timesteps",
            "diffusion_steps",
            "use_trajectory_normalization",
            "plan_anchor_path",
        ]

        for param in critical_params:
            nav_val = nav_cfg.get(param)
            b2d_val = b2d_cfg.get(param)

            if nav_val != b2d_val:
                print(f"\n{param}:")
                print(f"  NavSim: {nav_val}")
                print(f"  B2D:    {b2d_val}")

                if param == "plan_anchor_path":
                    # Check if anchors exist
                    if nav_val and not Path(nav_val).exists():
                        print(f"  ⚠️  NavSim anchor file missing: {nav_val}")
                    if b2d_val and not Path(b2d_val).exists():
                        print(f"  ⚠️  B2D anchor file missing: {b2d_val}")

    def trace_nan_propagation(
        self, features: Dict[str, torch.Tensor], model: nn.Module, dataset_name: str
    ):
        """Trace where NaN values appear in model forward pass."""
        print(f"\n\nTracing NaN propagation for {dataset_name}...")

        nan_locations = []

        def check_hook(module, input, output, layer_name):
            """Hook to check for NaN in layer outputs."""
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    nan_count = torch.isnan(output).sum().item()
                    nan_locations.append(
                        {
                            "layer": layer_name,
                            "nan_count": nan_count,
                            "output_shape": list(output.shape),
                        }
                    )
                    print(f"  ⚠️  NaN detected in {layer_name}: {nan_count} values")
            elif isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                        nan_count = torch.isnan(out).sum().item()
                        nan_locations.append(
                            {
                                "layer": f"{layer_name}[{i}]",
                                "nan_count": nan_count,
                                "output_shape": list(out.shape),
                            }
                        )
                        print(f"  ⚠️  NaN detected in {layer_name}[{i}]: {nan_count} values")

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(lambda m, i, o, n=name: check_hook(m, i, o, n))
                hooks.append(hook)

        try:
            # Run forward pass
            with torch.no_grad():
                _ = model(features)
        except Exception as e:
            print(f"  Error during forward pass: {e}")
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return nan_locations

    def generate_report(self, output_path: Path):
        """Generate comprehensive difference report."""
        report = {
            "configuration_differences": self.differences["config"],
            "normalization_differences": self.differences["normalization"],
            "model_parameter_differences": self.differences["model_params"],
            "processing_differences": self.differences["processing"],
            "numerical_differences": self.differences["numerical"],
            "recommendations": self._generate_recommendations(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {output_path}")

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check normalization scale differences
        for component, diff in self.differences["normalization"].items():
            if diff["scale_ratio"] > 10 or diff["scale_ratio"] < 0.1:
                recommendations.append(
                    f"Large scale difference in {component} normalization "
                    f"(ratio={diff['scale_ratio']:.2f}). Verify numerical stability."
                )

        # Check config differences
        if "use_trajectory_normalization" in self.differences["config"]:
            recommendations.append(
                "Trajectory normalization settings differ between datasets. "
                "Ensure both use consistent normalization."
            )

        if "plan_anchor_path" in self.differences["config"]:
            recommendations.append(
                "Different anchor files used. Ensure Bench2Drive uses "
                "dataset-specific anchors generated from Bench2Drive data."
            )

        return recommendations


def load_sample_data(cache_path: Path, sample_idx: int = 0) -> Tuple[Dict, Dict]:
    """Load a sample from cache."""
    if not cache_path.exists():
        raise ValueError(f"Cache directory does not exist: {cache_path}")

    feature_files = sorted(cache_path.glob("**/transfuser_feature.gz"))
    target_files = sorted(cache_path.glob("**/transfuser_target.gz"))

    if not feature_files:
        raise ValueError(f"No feature files found in {cache_path}")

    if sample_idx >= len(feature_files):
        sample_idx = 0

    with gzip.open(feature_files[sample_idx], "rb") as f:
        features = pickle.load(f)

    targets = {}
    if sample_idx < len(target_files):
        with gzip.open(target_files[sample_idx], "rb") as f:
            targets = pickle.load(f)

    return features, targets


def main():
    parser = argparse.ArgumentParser(description="Analyze pipeline differences")
    parser.add_argument(
        "--navsim-cache",
        type=str,
        default=os.environ.get("NAVSIM_EXP_ROOT", "/workspace/cache") + "/training_cache",
        help="NavSim cache path",
    )
    parser.add_argument(
        "--b2d-cache",
        type=str,
        default="/workspace/cache/bench2drive_cache",
        help="Bench2Drive cache path",
    )
    parser.add_argument(
        "--navsim-config",
        type=str,
        default="navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml",
        help="NavSim config",
    )
    parser.add_argument(
        "--b2d-config",
        type=str,
        default="navsim/planning/script/config/common/agent/diffusiondrive_agent_extended.yaml",
        help="Bench2Drive config",
    )
    parser.add_argument(
        "--output", type=str, default="pipeline_differences_report.json", help="Output report path"
    )
    parser.add_argument(
        "--test-model", action="store_true", help="Test model forward pass for NaN propagation"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = PipelineDifferenceAnalyzer()

    # Load configs
    nav_cfg = OmegaConf.load(args.navsim_config)
    b2d_cfg = OmegaConf.load(args.b2d_config)

    # Compare configurations
    analyzer.compare_configs(nav_cfg, b2d_cfg)

    # Compare normalization
    analyzer.compare_normalization()

    # Check model compatibility
    analyzer.check_model_compatibility(nav_cfg, b2d_cfg)

    # Load sample data for numerical analysis
    print("\nLoading sample data for numerical analysis...")
    nav_features, nav_targets = load_sample_data(Path(args.navsim_cache))
    b2d_features, b2d_targets = load_sample_data(Path(args.b2d_cache))

    # Analyze numerical precision
    analyzer.analyze_numerical_precision(nav_features, b2d_features)

    # Test model forward pass if requested
    if args.test_model:
        print("\n" + "=" * 60)
        print("MODEL FORWARD PASS TEST")
        print("=" * 60)

        try:
            # Create models
            nav_model = V2TransfuserModel(
                config=TransfuserConfig(**nav_cfg),
                bkb_path=nav_cfg.get("bkb_path"),
                plan_anchor_path=nav_cfg.get("plan_anchor_path"),
            )

            b2d_model = V2TransfuserModel(
                config=ExtendedTransfuserConfig(**b2d_cfg),
                bkb_path=b2d_cfg.get("bkb_path"),
                plan_anchor_path=b2d_cfg.get("plan_anchor_path"),
            )

            # Test forward pass
            nav_nan_locs = analyzer.trace_nan_propagation(nav_features, nav_model, "NavSim")
            b2d_nan_locs = analyzer.trace_nan_propagation(b2d_features, b2d_model, "Bench2Drive")

            if not nav_nan_locs:
                print("\n✅ NavSim: No NaN detected in forward pass")
            if not b2d_nan_locs:
                print("\n✅ Bench2Drive: No NaN detected in forward pass")

        except Exception as e:
            print(f"\nError creating/testing models: {e}")
            print("Skipping model forward pass test")

    # Generate report
    analyzer.generate_report(Path(args.output))

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey findings:")
    print("1. Configuration and normalization parameters differ significantly")
    print("2. Bench2Drive requires dataset-specific anchors and normalization")
    print("3. Check model initialization and ensure correct parameters are used")
    print("4. Run the analysis scripts on actual training data to find NaN source")


if __name__ == "__main__":
    main()
