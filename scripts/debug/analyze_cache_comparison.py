#!/usr/bin/env python3
"""
Parallel analysis of NavSim and Bench2Drive cached data to identify fundamental differences.
This script analyzes raw cached data before any processing or transformations.
"""

import os
import sys
import numpy as np
import torch
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Any
import json
from collections import defaultdict
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class CacheAnalyzer:
    """Analyzes cached data for a single dataset."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.stats = defaultdict(
            lambda: {
                "min": float("inf"),
                "max": float("-inf"),
                "mean": 0,
                "std": 0,
                "count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "shape_counts": defaultdict(int),
                "percentiles": {},
            }
        )
        self.raw_samples = defaultdict(list)

    def update_stats(self, field_name: str, data: np.ndarray):
        """Update running statistics for a field."""
        if data is None:
            return

        # Convert to numpy if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Record shape
        self.stats[field_name]["shape_counts"][str(data.shape)] += 1

        # Check for invalid values
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        valid_mask = ~(nan_mask | inf_mask)

        stats = self.stats[field_name]
        stats["nan_count"] += nan_mask.sum()
        stats["inf_count"] += inf_mask.sum()

        if valid_mask.any():
            valid_data = data[valid_mask].flatten()

            # Update min/max
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

            # Store raw samples for percentile calculation
            if len(self.raw_samples[field_name]) < 10000:  # Limit memory usage
                self.raw_samples[field_name].extend(valid_data[:100].tolist())

    def analyze_features(self, features: Dict[str, Any], prefix: str = "features"):
        """Recursively analyze feature dictionary."""
        for key, value in features.items():
            if value is None:
                continue

            field_path = f"{prefix}/{key}"

            if isinstance(value, dict):
                self.analyze_features(value, field_path)
            elif isinstance(value, (np.ndarray, list, torch.Tensor)):
                if isinstance(value, list):
                    value = np.array(value)
                elif isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                self.update_stats(field_path, value)

                # Special handling for specific fields
                if key == "trajectory" and len(value.shape) >= 2:
                    # Analyze x, y, heading separately
                    if value.shape[-1] >= 2:
                        self.update_stats(f"{field_path}/x", value[..., 0])
                        self.update_stats(f"{field_path}/y", value[..., 1])
                    if value.shape[-1] >= 3:
                        self.update_stats(f"{field_path}/heading", value[..., 2])

                elif key == "velocity" and len(value.shape) >= 1:
                    if value.shape[-1] >= 2:
                        self.update_stats(f"{field_path}/vx", value[..., 0])
                        self.update_stats(f"{field_path}/vy", value[..., 1])

    def finalize_percentiles(self):
        """Calculate percentiles from collected samples."""
        for field_name, samples in self.raw_samples.items():
            if samples:
                arr = np.array(samples)
                # Skip percentile calculation for boolean arrays
                if arr.dtype == bool:
                    self.stats[field_name]["percentiles"] = {
                        "p1": None, "p5": None, "p25": None, "p50": None,
                        "p75": None, "p95": None, "p99": None
                    }
                else:
                    self.stats[field_name]["percentiles"] = {
                        "p1": np.percentile(arr, 1),
                        "p5": np.percentile(arr, 5),
                        "p25": np.percentile(arr, 25),
                        "p50": np.percentile(arr, 50),
                        "p75": np.percentile(arr, 75),
                        "p95": np.percentile(arr, 95),
                        "p99": np.percentile(arr, 99),
                    }

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        for name, stats in self.stats.items():
            summary[name] = {
                "min": float(stats["min"]) if stats["min"] != float("inf") else None,
                "max": float(stats["max"]) if stats["max"] != float("-inf") else None,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "count": int(stats["count"]),
                "nan_count": int(stats["nan_count"]),
                "inf_count": int(stats["inf_count"]),
                "nan_ratio": stats["nan_count"] / max(1, stats["count"] + stats["nan_count"]),
                "shapes": dict(stats["shape_counts"]),
                "percentiles": stats.get("percentiles", {}),
            }
        return summary


def analyze_cache_files(
    cache_dir: Path, dataset_name: str, max_samples: int = 1000
) -> CacheAnalyzer:
    """Analyze cached files for a dataset."""
    # Check if cache directory exists
    if not cache_dir.exists():
        raise ValueError(f"{dataset_name}: Cache directory does not exist: {cache_dir}")
        
    analyzer = CacheAnalyzer(dataset_name)

    # Find cached files
    feature_files = sorted(cache_dir.glob("**/transfuser_feature.gz"))[:max_samples]
    target_files = sorted(cache_dir.glob("**/transfuser_target.gz"))[:max_samples]

    print(
        f"\n{dataset_name}: Found {len(feature_files)} feature files, {len(target_files)} target files"
    )
    
    # Check if any files were found
    if len(feature_files) == 0 and len(target_files) == 0:
        raise ValueError(
            f"{dataset_name}: No cache files found in {cache_dir}\n"
            f"Looking for: **/transfuser_feature.gz and **/transfuser_target.gz"
        )

    # Analyze features
    for feat_file in tqdm(feature_files, desc=f"Analyzing {dataset_name} features"):
        try:
            with gzip.open(feat_file, "rb") as f:
                features = pickle.load(f)
            analyzer.analyze_features(features, "features")
        except Exception as e:
            print(f"Error loading {feat_file}: {e}")

    # Analyze targets
    for tgt_file in tqdm(target_files, desc=f"Analyzing {dataset_name} targets"):
        try:
            with gzip.open(tgt_file, "rb") as f:
                targets = pickle.load(f)
            analyzer.analyze_features(targets, "targets")
        except Exception as e:
            print(f"Error loading {tgt_file}: {e}")

    # Calculate percentiles
    analyzer.finalize_percentiles()

    return analyzer


def compare_field_stats(nav_stats: Dict, b2d_stats: Dict, field_name: str) -> Dict:
    """Compare statistics for a specific field between datasets."""
    comparison = {
        "field": field_name,
        "navsim": nav_stats,
        "bench2drive": b2d_stats,
        "differences": {},
    }

    # Calculate differences
    if nav_stats and b2d_stats:
        # Range difference
        nav_range = (nav_stats.get("max", 0) or 0) - (nav_stats.get("min", 0) or 0)
        b2d_range = (b2d_stats.get("max", 0) or 0) - (b2d_stats.get("min", 0) or 0)

        comparison["differences"]["range_ratio"] = (
            b2d_range / nav_range if nav_range > 0 else float("inf")
        )
        comparison["differences"]["mean_diff"] = abs(
            (nav_stats.get("mean", 0) or 0) - (b2d_stats.get("mean", 0) or 0)
        )
        comparison["differences"]["std_ratio"] = (b2d_stats.get("std", 0) or 0) / (
            nav_stats.get("std", 1) or 1
        )

        # Check if distributions overlap
        nav_min, nav_max = nav_stats.get("min", 0), nav_stats.get("max", 0)
        b2d_min, b2d_max = b2d_stats.get("min", 0), b2d_stats.get("max", 0)

        if (
            nav_min is not None
            and nav_max is not None
            and b2d_min is not None
            and b2d_max is not None
        ):
            overlap = max(0, min(nav_max, b2d_max) - max(nav_min, b2d_min))
            comparison["differences"]["overlap_ratio"] = (
                overlap / nav_range if nav_range > 0 else 0
            )

    return comparison


def print_comparison_table(comparisons: List[Dict], critical_fields: List[str]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print("FIELD COMPARISON SUMMARY")
    print("=" * 120)

    # Headers
    print(
        f"{'Field':<40} {'NavSim Range':<20} {'B2D Range':<20} {'Range Ratio':<12} {'Overlap':<10}"
    )
    print("-" * 120)

    # Sort by range ratio
    comparisons.sort(key=lambda x: abs(x["differences"].get("range_ratio", 1) - 1), reverse=True)

    for comp in comparisons:
        field = comp["field"]
        nav = comp["navsim"]
        b2d = comp["bench2drive"]
        diff = comp["differences"]

        if nav and b2d:
            nav_range = f"[{nav.get('min', 'N/A'):.3f}, {nav.get('max', 'N/A'):.3f}]"
            b2d_range = f"[{b2d.get('min', 'N/A'):.3f}, {b2d.get('max', 'N/A'):.3f}]"
            range_ratio = diff.get("range_ratio", 0)
            overlap = diff.get("overlap_ratio", 0)

            # Highlight critical fields or significant differences
            marker = (
                "⚠️ " if field in critical_fields or range_ratio > 2 or range_ratio < 0.5 else "  "
            )

            print(
                f"{marker}{field:<38} {nav_range:<20} {b2d_range:<20} "
                f"{range_ratio:<12.2f} {overlap:<10.1%}"
            )


def plot_distribution_comparison(
    nav_analyzer: CacheAnalyzer, b2d_analyzer: CacheAnalyzer, field_name: str, output_dir: Path
):
    """Plot distribution comparison for a specific field."""
    nav_samples = nav_analyzer.raw_samples.get(field_name, [])
    b2d_samples = b2d_analyzer.raw_samples.get(field_name, [])

    if not nav_samples or not b2d_samples:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histograms
    ax1.hist(nav_samples[:1000], bins=50, alpha=0.5, label="NavSim", density=True)
    ax1.hist(b2d_samples[:1000], bins=50, alpha=0.5, label="Bench2Drive", density=True)
    ax1.set_xlabel(field_name)
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.set_title(f"Distribution Comparison: {field_name}")

    # Box plots
    ax2.boxplot([nav_samples[:1000], b2d_samples[:1000]], labels=["NavSim", "Bench2Drive"])
    ax2.set_ylabel(field_name)
    ax2.set_title("Box Plot Comparison")

    plt.tight_layout()

    # Save plot
    safe_name = field_name.replace("/", "_")
    plt.savefig(output_dir / f"dist_comparison_{safe_name}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare NavSim and Bench2Drive cache statistics")
    parser.add_argument(
        "--navsim-cache",
        type=str,
        default=os.environ.get("NAVSIM_EXP_ROOT", "/workspace/cache") + "/training_cache",
        help="Path to NavSim cache",
    )
    parser.add_argument(
        "--b2d-cache",
        type=str,
        default="/workspace/cache/bench2drive_cache",
        help="Path to Bench2Drive cache",
    )
    parser.add_argument(
        "--max-samples", type=int, default=1000, help="Maximum samples to analyze per dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cache_analysis_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--plot-fields",
        nargs="+",
        default=["features/trajectory/x", "features/trajectory/y", "features/trajectory/heading"],
        help="Fields to plot distributions for",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Analyze both datasets
    print("Analyzing NavSim cache...")
    nav_analyzer = analyze_cache_files(Path(args.navsim_cache), "NavSim", args.max_samples)
    nav_summary = nav_analyzer.get_summary()
    
    print("\nAnalyzing Bench2Drive cache...")
    b2d_analyzer = analyze_cache_files(Path(args.b2d_cache), "Bench2Drive", args.max_samples)
    b2d_summary = b2d_analyzer.get_summary()

    # Compare fields
    all_fields = set(nav_summary.keys()) | set(b2d_summary.keys())
    comparisons = []

    critical_fields = [
        "features/trajectory/x",
        "features/trajectory/y",
        "features/trajectory/heading",
        "targets/trajectory/x",
        "targets/trajectory/y",
        "targets/trajectory/heading",
    ]

    for field in all_fields:
        nav_stats = nav_summary.get(field, {})
        b2d_stats = b2d_summary.get(field, {})

        if nav_stats or b2d_stats:
            comparison = compare_field_stats(nav_stats, b2d_stats, field)
            comparisons.append(comparison)

    # Print comparison table
    print_comparison_table(comparisons, critical_fields)

    # Print warnings
    print("\n" + "=" * 80)
    print("WARNINGS AND ANOMALIES")
    print("=" * 80)

    # Check for NaN/Inf
    for dataset, summary in [("NavSim", nav_summary), ("Bench2Drive", b2d_summary)]:
        print(f"\n{dataset}:")
        has_issues = False
        for field, stats in summary.items():
            if stats.get("nan_count", 0) > 0:
                print(
                    f"  ⚠️  {field}: {stats['nan_count']} "
                    f"NaN values ({stats.get('nan_ratio', 0):.1%})"
                )
                has_issues = True
            if stats.get("inf_count", 0) > 0:
                print(f"  ⚠️  {field}: {stats['inf_count']} Inf values")
                has_issues = True
        if not has_issues:
            print("  ✅ No NaN/Inf values found")

    # Plot distributions for selected fields
    print(f"\nPlotting distributions for: {args.plot_fields}")
    for field in args.plot_fields:
        if field in nav_summary and field in b2d_summary:
            plot_distribution_comparison(nav_analyzer, b2d_analyzer, field, output_dir)

    # Save detailed results
    results = {
        "navsim_summary": nav_summary,
        "bench2drive_summary": b2d_summary,
        "comparisons": comparisons,
    }

    with open(output_dir / "cache_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("  - Detailed JSON: cache_comparison_results.json")
    print("  - Distribution plots: dist_comparison_*.png")


if __name__ == "__main__":
    main()
