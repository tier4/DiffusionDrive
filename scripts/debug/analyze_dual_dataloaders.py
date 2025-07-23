#!/usr/bin/env python3
"""
Analyze data from both NavSim and Bench2Drive dataloaders side-by-side,
tracking statistics through each transformation stage.
"""

import os
import sys
import numpy as np
import torch
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import defaultdict
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig
from navsim.planning.training.dataset import CacheOnlyDataset
# Removed unused imports
from omegaconf import OmegaConf


class DataloaderAnalyzer:
    """Analyzes dataloader outputs at each transformation stage."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.stage_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "min": float("inf"),
                    "max": float("-inf"),
                    "mean": 0,
                    "std": 0,
                    "count": 0,
                    "nan_count": 0,
                    "inf_count": 0,
                    "first_nan_batch": None,
                    "samples": [],
                }
            )
        )

    def analyze_tensor(self, stage: str, name: str, tensor: torch.Tensor, batch_idx: int):
        """Analyze a tensor and update statistics."""
        if tensor is None:
            return

        # Convert to numpy
        data = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

        # Get stats dict
        stats = self.stage_stats[stage][name]

        # Check for NaN/Inf
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)

        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()

        stats["nan_count"] += nan_count
        stats["inf_count"] += inf_count

        # Record first NaN occurrence
        if nan_count > 0 and stats["first_nan_batch"] is None:
            stats["first_nan_batch"] = batch_idx
            # Store the problematic sample
            stats["first_nan_sample"] = {
                "batch_idx": batch_idx,
                "shape": data.shape,
                "nan_locations": np.where(nan_mask),
                "sample_values": data.flatten()[:100].tolist(),
            }

        # Calculate statistics on valid data
        valid_mask = ~(nan_mask | inf_mask)
        if valid_mask.any():
            valid_data = data[valid_mask]

            stats["min"] = min(stats["min"], valid_data.min())
            stats["max"] = max(stats["max"], valid_data.max())

            # Update running mean/std
            old_count = stats["count"]
            new_count = old_count + valid_data.size
            old_mean = stats["mean"]
            new_mean = (old_mean * old_count + valid_data.sum()) / new_count

            if old_count > 0:
                old_var = stats["std"] ** 2
                new_var = (
                    old_var * old_count
                    + ((valid_data - new_mean) ** 2).sum()
                    + old_count * (old_mean - new_mean) ** 2
                ) / new_count
                stats["std"] = np.sqrt(new_var)
            else:
                stats["std"] = valid_data.std()

            stats["mean"] = new_mean
            stats["count"] = new_count

            # Store samples
            if len(stats["samples"]) < 10:
                stats["samples"].append(
                    {"batch_idx": batch_idx, "values": valid_data.flatten()[:20].tolist()}
                )

    def analyze_dict(
        self, stage: str, data_dict: Dict[str, Any], batch_idx: int, prefix: str = ""
    ):
        """Recursively analyze a dictionary of tensors."""
        for key, value in data_dict.items():
            full_key = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                self.analyze_dict(stage, value, batch_idx, full_key)
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                self.analyze_tensor(stage, full_key, value, batch_idx)

                # Special handling for specific fields
                if key == "trajectory" and len(value.shape) >= 3:
                    # Analyze components separately
                    if value.shape[-1] >= 2:
                        self.analyze_tensor(stage, f"{full_key}/x", value[..., 0], batch_idx)
                        self.analyze_tensor(stage, f"{full_key}/y", value[..., 1], batch_idx)
                    if value.shape[-1] >= 3:
                        self.analyze_tensor(stage, f"{full_key}/heading", value[..., 2], batch_idx)


def create_dataloader(dataset_type: str, cache_path: str, config_path: str, batch_size: int = 8):
    """Create a dataloader for the specified dataset."""
    # Check if cache path exists
    cache_dir = Path(cache_path)
    if not cache_dir.exists():
        raise ValueError(f"Cache directory does not exist: {cache_path}")

    # Check if cache contains data
    feature_files = list(cache_dir.glob("**/transfuser_feature.gz"))
    if not feature_files:
        raise ValueError(f"No cached data found in {cache_path}")

    # Load config
    cfg = OmegaConf.load(config_path)

    # Create config object from the loaded config
    from hydra.utils import instantiate
    
    # Get the config section
    if dataset_type == "navsim":
        config = instantiate(cfg.config)
    else:
        # For bench2drive, we might need ExtendedTransfuserConfig
        try:
            config = instantiate(cfg.config)
        except:
            # If instantiation fails, create a default config
            config = TransfuserConfig()
    
    # Create feature and target builders with config
    feature_builder = TransfuserFeatureBuilder(config)
    target_builder = TransfuserTargetBuilder(config)
    
    # Create dataset with builders
    dataset = CacheOnlyDataset(
        cache_path=cache_path,
        feature_builders=[feature_builder],
        target_builders=[target_builder],
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single worker for debugging
        collate_fn=None,  # Use default collation
    )

    return dataloader, feature_builder, target_builder, cfg


def analyze_dataloader_pipeline(
    dataloader,
    feature_builder,
    target_builder,
    analyzer: DataloaderAnalyzer,
    max_batches: int = 10,
):
    """Analyze data through the complete pipeline."""

    print(f"\nAnalyzing {analyzer.dataset_name} dataloader pipeline...")

    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=max_batches)):
        if batch_idx >= max_batches:
            break

        # DataLoader with default collate returns list: [features_dict, targets_dict]
        if isinstance(batch_data, list) and len(batch_data) == 2:
            features_batch, targets_batch = batch_data
            
            # Stage 1: Analyze raw features from cache (already batched)
            if isinstance(features_batch, dict):
                analyzer.analyze_dict("1_cached_features", features_batch, batch_idx)
            
            # Stage 2: Analyze raw targets from cache (already batched)
            if isinstance(targets_batch, dict):
                analyzer.analyze_dict("2_cached_targets", targets_batch, batch_idx)
                
            # Stage 3: Analyze individual trajectory components
            for key, value in features_batch.items():
                if key == "trajectory" and isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                    # Analyze trajectory components
                    if value.shape[-1] >= 2:
                        analyzer.analyze_tensor("3_features_components", f"trajectory/x", value[..., 0], batch_idx)
                        analyzer.analyze_tensor("3_features_components", f"trajectory/y", value[..., 1], batch_idx)
                    if value.shape[-1] >= 3:
                        analyzer.analyze_tensor("3_features_components", f"trajectory/heading", value[..., 2], batch_idx)
                        
            for key, value in targets_batch.items():
                if key == "trajectory" and isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                    # Analyze trajectory components
                    if value.shape[-1] >= 2:
                        analyzer.analyze_tensor("4_targets_components", f"trajectory/x", value[..., 0], batch_idx)
                        analyzer.analyze_tensor("4_targets_components", f"trajectory/y", value[..., 1], batch_idx)
                    if value.shape[-1] >= 3:
                        analyzer.analyze_tensor("4_targets_components", f"trajectory/heading", value[..., 2], batch_idx)
                        
        else:
            print(f"Unexpected batch format: {type(batch_data)}")
            if isinstance(batch_data, list):
                print(f"  List length: {len(batch_data)}")
                if len(batch_data) > 0:
                    print(f"  First item type: {type(batch_data[0])}")


def compare_stage_stats(nav_analyzer: DataloaderAnalyzer, b2d_analyzer: DataloaderAnalyzer):
    """Compare statistics between datasets at each stage."""

    print("\n" + "=" * 100)
    print("STAGE-BY-STAGE COMPARISON")
    print("=" * 100)

    # Get all stages
    all_stages = set(nav_analyzer.stage_stats.keys()) | set(b2d_analyzer.stage_stats.keys())

    for stage in sorted(all_stages):
        print(f"\n{'='*80}")
        print(f"STAGE: {stage}")
        print(f"{'='*80}")

        # Get all fields in this stage
        nav_fields = set(nav_analyzer.stage_stats[stage].keys())
        b2d_fields = set(b2d_analyzer.stage_stats[stage].keys())
        all_fields = nav_fields | b2d_fields

        # Print comparison table
        print(f"{'Field':<40} {'NavSim':<30} {'Bench2Drive':<30}")
        print(f"{'-'*40} {'-'*30} {'-'*30}")

        for field in sorted(all_fields):
            nav_stats = nav_analyzer.stage_stats[stage].get(field, {})
            b2d_stats = b2d_analyzer.stage_stats[stage].get(field, {})

            # Format stats
            nav_str = "N/A"
            if nav_stats and nav_stats["count"] > 0:
                nav_str = f"[{nav_stats['min']:.3f}, {nav_stats['max']:.3f}]"
                if nav_stats["nan_count"] > 0:
                    nav_str += f" ⚠️ NaN:{nav_stats['nan_count']}"

            b2d_str = "N/A"
            if b2d_stats and b2d_stats["count"] > 0:
                b2d_str = f"[{b2d_stats['min']:.3f}, {b2d_stats['max']:.3f}]"
                if b2d_stats["nan_count"] > 0:
                    b2d_str += f" ⚠️ NaN:{b2d_stats['nan_count']}"

            print(f"{field:<40} {nav_str:<30} {b2d_str:<30}")

    # Print NaN occurrences
    print("\n" + "=" * 80)
    print("NaN FIRST OCCURRENCES")
    print("=" * 80)

    for dataset_name, analyzer in [("NavSim", nav_analyzer), ("Bench2Drive", b2d_analyzer)]:
        print(f"\n{dataset_name}:")
        nan_found = False

        for stage, fields in analyzer.stage_stats.items():
            for field, stats in fields.items():
                if stats.get("first_nan_batch") is not None:
                    print(f"  ⚠️  {stage}/{field}: First NaN at batch {stats['first_nan_batch']}")
                    if "first_nan_sample" in stats:
                        sample = stats["first_nan_sample"]
                        print(f"      Shape: {sample['shape']}")
                        print(f"      NaN locations: {sample['nan_locations']}")
                    nan_found = True

        if not nan_found:
            print("  ✅ No NaN values found")


def save_detailed_report(
    nav_analyzer: DataloaderAnalyzer, b2d_analyzer: DataloaderAnalyzer, output_path: Path
):
    """Save detailed analysis report."""
    report = {"navsim": {}, "bench2drive": {}, "comparison": {}}

    # Convert stats to serializable format
    for stage, fields in nav_analyzer.stage_stats.items():
        report["navsim"][stage] = {}
        for field, stats in fields.items():
            report["navsim"][stage][field] = {
                "min": float(stats["min"]) if stats["min"] != float("inf") else None,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else None,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "count": int(stats["count"]),
                "nan_count": int(stats["nan_count"]),
                "inf_count": int(stats["inf_count"]),
                "first_nan_batch": stats["first_nan_batch"],
            }

    for stage, fields in b2d_analyzer.stage_stats.items():
        report["bench2drive"][stage] = {}
        for field, stats in fields.items():
            report["bench2drive"][stage][field] = {
                "min": float(stats["min"]) if stats["min"] != float("inf") else None,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else None,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "count": int(stats["count"]),
                "nan_count": int(stats["nan_count"]),
                "inf_count": int(stats["inf_count"]),
                "first_nan_batch": stats["first_nan_batch"],
            }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Analyze dataloader outputs for both datasets")
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
        help="NavSim config path",
    )
    parser.add_argument(
        "--b2d-config",
        type=str,
        default="navsim/planning/script/config/common/agent/diffusiondrive_agent_extended.yaml",
        help="Bench2Drive config path",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for analysis")
    parser.add_argument("--max-batches", type=int, default=10, help="Maximum batches to analyze")
    parser.add_argument(
        "--output", type=str, default="dataloader_analysis_report.json", help="Output report path"
    )

    args = parser.parse_args()

    # Create analyzers
    nav_analyzer = DataloaderAnalyzer("NavSim")
    b2d_analyzer = DataloaderAnalyzer("Bench2Drive")

    # Analyze NavSim
    print("Creating NavSim dataloader...")
    nav_dataloader, nav_feature_builder, nav_target_builder, nav_cfg = create_dataloader(
        "navsim", args.navsim_cache, args.navsim_config, args.batch_size
    )
    analyze_dataloader_pipeline(
        nav_dataloader, nav_feature_builder, nav_target_builder, nav_analyzer, args.max_batches
    )

    # Analyze Bench2Drive
    print("\nCreating Bench2Drive dataloader...")
    b2d_dataloader, b2d_feature_builder, b2d_target_builder, b2d_cfg = create_dataloader(
        "bench2drive", args.b2d_cache, args.b2d_config, args.batch_size
    )
    analyze_dataloader_pipeline(
        b2d_dataloader, b2d_feature_builder, b2d_target_builder, b2d_analyzer, args.max_batches
    )

    # Compare results
    compare_stage_stats(nav_analyzer, b2d_analyzer)

    # Save detailed report
    output_path = Path(args.output)
    save_detailed_report(nav_analyzer, b2d_analyzer, output_path)
    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
