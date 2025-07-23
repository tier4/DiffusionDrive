#!/usr/bin/env python3
"""
Deep dive analysis of feature transformations for both NavSim and Bench2Drive.
Tracks data through each specific transformation step with parallel processing.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import gzip
from collections import defaultdict
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.extended_transfuser_config import ExtendedTransfuserConfig
from navsim.agents.diffusiondrive.trajectory_normalizer import TrajectoryNormalizer
# Import what we actually need
from navsim.common.dataclasses import AgentInput
from omegaconf import OmegaConf


class TransformationStats:
    """Lightweight stats container for parallel processing."""
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0.0
        self.sum_sq = 0.0
        self.count = 0
        self.nan_count = 0
        self.samples = []
    
    def update(self, data: np.ndarray):
        """Update statistics with new data."""
        self.nan_count += np.isnan(data).sum()
        
        if not np.isnan(data).all():
            valid_data = data[~np.isnan(data)]
            self.min_val = min(self.min_val, valid_data.min())
            self.max_val = max(self.max_val, valid_data.max())
            self.sum_val += valid_data.sum()
            self.sum_sq += (valid_data ** 2).sum()
            self.count += len(valid_data)
            
            # Store a few samples
            if len(self.samples) < 5:
                self.samples.append(data.flatten()[:20].tolist())
    
    def merge(self, other: 'TransformationStats'):
        """Merge statistics from another instance."""
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.sum_val += other.sum_val
        self.sum_sq += other.sum_sq
        self.count += other.count
        self.nan_count += other.nan_count
        self.samples.extend(other.samples[:max(0, 5 - len(self.samples))])
    
    def get_stats(self):
        """Get computed statistics."""
        if self.count > 0:
            mean = self.sum_val / self.count
            variance = (self.sum_sq / self.count) - (mean ** 2)
            std = np.sqrt(max(0, variance))  # Avoid negative variance due to numerical errors
        else:
            mean = 0
            std = 0
            
        return {
            'min': float(self.min_val) if self.min_val != float('inf') else 0,
            'max': float(self.max_val) if self.max_val != float('-inf') else 0,
            'mean': float(mean),
            'std': float(std),
            'nan_count': int(self.nan_count)
        }


def analyze_single_sample(cache_path: Path, sample_idx: int, dataset_type: str) -> Dict[str, Dict[str, TransformationStats]]:
    """Analyze a single sample and return statistics."""
    stats = defaultdict(lambda: {'before': TransformationStats(), 'after': TransformationStats()})
    
    try:
        # Load sample
        feature_files = sorted(cache_path.glob("**/transfuser_feature.gz"))
        target_files = sorted(cache_path.glob("**/transfuser_target.gz"))
        
        if sample_idx >= len(feature_files):
            return stats
            
        with gzip.open(feature_files[sample_idx], 'rb') as f:
            features = pickle.load(f)
            
        targets = {}
        if sample_idx < len(target_files):
            with gzip.open(target_files[sample_idx], 'rb') as f:
                targets = pickle.load(f)
        
        # Analyze camera
        if 'camera_feature' in features:
            camera_data = features['camera_feature']
            if isinstance(camera_data, torch.Tensor):
                camera_data = camera_data.cpu().numpy()
            
            stats['camera/normalized']['before'].update(camera_data)
            stats['camera/normalized']['after'].update(camera_data)
            
            # Simulate denormalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            denormalized = camera_data * std.mean() + mean.mean()
            stats['camera/denormalized']['before'].update(camera_data)
            stats['camera/denormalized']['after'].update(denormalized)
        
        # Analyze lidar
        if 'lidar_feature' in features:
            lidar_data = features['lidar_feature']
            if isinstance(lidar_data, torch.Tensor):
                lidar_data = lidar_data.cpu().numpy()
            
            stats['lidar/bev_histogram']['before'].update(lidar_data)
            stats['lidar/bev_histogram']['after'].update(lidar_data)
            
            # Density analysis
            non_zero = lidar_data[lidar_data > 0]
            if len(non_zero) > 0:
                density = len(non_zero) / lidar_data.size
                density_array = np.full_like(lidar_data, density)
                stats['lidar/density']['before'].update(lidar_data)
                stats['lidar/density']['after'].update(density_array)
        
        # Analyze trajectory
        if 'trajectory' in targets:
            trajectory = targets['trajectory']
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.cpu().numpy()
            
            stats['trajectory/ego_relative']['before'].update(trajectory)
            stats['trajectory/ego_relative']['after'].update(trajectory)
            
            # Apply normalization
            normalizer = TrajectoryNormalizer(dataset_type=dataset_type)
            traj_tensor = torch.from_numpy(trajectory).float()
            normalized = normalizer.normalize(traj_tensor)
            if isinstance(normalized, torch.Tensor):
                normalized_np = normalized.numpy()
            else:
                normalized_np = normalized
            
            stats['trajectory/normalized']['before'].update(trajectory.flatten())
            stats['trajectory/normalized']['after'].update(normalized_np.flatten())
            
            # Track components
            if trajectory.shape[-1] >= 3:
                stats['trajectory/x']['before'].update(trajectory[..., 0])
                stats['trajectory/x']['after'].update(normalized_np[..., 0])
                stats['trajectory/y']['before'].update(trajectory[..., 1])
                stats['trajectory/y']['after'].update(normalized_np[..., 1])
                stats['trajectory/heading']['before'].update(trajectory[..., 2])
                stats['trajectory/heading']['after'].update(normalized_np[..., 2])
        
        # Analyze status
        if 'status_feature' in features:
            status = features['status_feature']
            if isinstance(status, torch.Tensor):
                status = status.cpu().numpy()
            
            stats['status/processed']['before'].update(status)
            stats['status/processed']['after'].update(status)
            
            if len(status) >= 3:
                stats['status/driving_command']['before'].update(status[0:1])
                stats['status/driving_command']['after'].update(status[0:1])
                
                if len(status) >= 5:
                    velocities = status[1:3]
                    accelerations = status[3:5]
                    stats['status/velocities']['before'].update(velocities)
                    stats['status/velocities']['after'].update(velocities)
                    stats['status/accelerations']['before'].update(accelerations)
                    stats['status/accelerations']['after'].update(accelerations)
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
    
    return stats


def merge_stats(all_stats: List[Dict[str, Dict[str, TransformationStats]]]) -> Dict[str, Dict[str, Dict]]:
    """Merge statistics from all samples."""
    merged = defaultdict(lambda: {'before': TransformationStats(), 'after': TransformationStats()})
    
    for sample_stats in all_stats:
        for transform_name, transform_stats in sample_stats.items():
            merged[transform_name]['before'].merge(transform_stats['before'])
            merged[transform_name]['after'].merge(transform_stats['after'])
    
    # Convert to regular dict with computed stats
    final_stats = {}
    for transform_name, transform_stats in merged.items():
        before_stats = transform_stats['before'].get_stats()
        after_stats = transform_stats['after'].get_stats()
        
        # Compute change stats
        change_stats = {
            'min_diff': after_stats['min'] - before_stats['min'],
            'max_diff': after_stats['max'] - before_stats['max'],
            'mean_diff': after_stats['mean'] - before_stats['mean'],
            'std_ratio': after_stats['std'] / (before_stats['std'] + 1e-6)
        }
        
        final_stats[transform_name] = {
            'before': before_stats,
            'after': after_stats,
            'change': change_stats,
            'samples': transform_stats['before'].samples[:5]  # Keep a few samples
        }
    
    return final_stats


def print_transformation_summary(nav_stats: Dict, b2d_stats: Dict):
    """Print summary of transformations for both datasets."""
    print("\n" + "="*100)
    print("TRANSFORMATION ANALYSIS SUMMARY")
    print("="*100)
    
    # Get all transformations
    all_transforms = set(nav_stats.keys()) | set(b2d_stats.keys())
    
    for transform in sorted(all_transforms):
        print(f"\n{transform}:")
        print(f"{'Dataset':<15} {'Before':<30} {'After':<30} {'Change':<20}")
        print(f"{'-'*15} {'-'*30} {'-'*30} {'-'*20}")
        
        # NavSim stats
        if transform in nav_stats:
            stats = nav_stats[transform]
            before = f"[{stats['before']['min']:.3f}, {stats['before']['max']:.3f}]"
            after = f"[{stats['after']['min']:.3f}, {stats['after']['max']:.3f}]"
            change = f"Δμ={stats['change']['mean_diff']:.3f}"
            
            if stats['before']['nan_count'] > 0:
                before += f" ⚠️ NaN:{stats['before']['nan_count']}"
            if stats['after']['nan_count'] > 0:
                after += f" ⚠️ NaN:{stats['after']['nan_count']}"
                
            print(f"{'NavSim':<15} {before:<30} {after:<30} {change:<20}")
        
        # Bench2Drive stats
        if transform in b2d_stats:
            stats = b2d_stats[transform]
            before = f"[{stats['before']['min']:.3f}, {stats['before']['max']:.3f}]"
            after = f"[{stats['after']['min']:.3f}, {stats['after']['max']:.3f}]"
            change = f"Δμ={stats['change']['mean_diff']:.3f}"
            
            if stats['before']['nan_count'] > 0:
                before += f" ⚠️ NaN:{stats['before']['nan_count']}"
            if stats['after']['nan_count'] > 0:
                after += f" ⚠️ NaN:{stats['after']['nan_count']}"
                
            print(f"{'Bench2Drive':<15} {before:<30} {after:<30} {change:<20}")


def plot_transformation_effects(nav_stats: Dict, b2d_stats: Dict, output_dir: Path):
    """Plot before/after distributions for key transformations."""
    key_transforms = ['trajectory/normalized', 'trajectory/x', 'trajectory/y', 'trajectory/heading']
    
    fig, axes = plt.subplots(len(key_transforms), 2, figsize=(12, 4 * len(key_transforms)))
    if len(key_transforms) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, transform in enumerate(key_transforms):
        # NavSim plot
        if transform in nav_stats and nav_stats[transform]['samples']:
            sample = nav_stats[transform]['samples'][0]
            
            axes[idx, 0].hist(sample, bins=30, alpha=0.5, label='Sample', density=True)
            axes[idx, 0].axvline(nav_stats[transform]['before']['min'], color='r', linestyle='--', label='Min/Max')
            axes[idx, 0].axvline(nav_stats[transform]['before']['max'], color='r', linestyle='--')
            axes[idx, 0].axvline(nav_stats[transform]['before']['mean'], color='g', linestyle='-', label='Mean')
            axes[idx, 0].set_title(f'NavSim: {transform}')
            axes[idx, 0].legend()
            
        # Bench2Drive plot
        if transform in b2d_stats and b2d_stats[transform]['samples']:
            sample = b2d_stats[transform]['samples'][0]
            
            axes[idx, 1].hist(sample, bins=30, alpha=0.5, label='Sample', density=True)
            axes[idx, 1].axvline(b2d_stats[transform]['before']['min'], color='r', linestyle='--', label='Min/Max')
            axes[idx, 1].axvline(b2d_stats[transform]['before']['max'], color='r', linestyle='--')
            axes[idx, 1].axvline(b2d_stats[transform]['before']['mean'], color='g', linestyle='-', label='Mean')
            axes[idx, 1].set_title(f'Bench2Drive: {transform}')
            axes[idx, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transformation_effects.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Deep dive into feature transformations with parallel processing")
    parser.add_argument('--navsim-cache', type=str,
                       default=os.environ.get('NAVSIM_EXP_ROOT', '/workspace/cache') + '/training_cache',
                       help='NavSim cache path')
    parser.add_argument('--b2d-cache', type=str,
                       default='/workspace/navsim_workspace/cache/bench2drive_Base_cache/',
                       help='Bench2Drive cache path')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to analyze')
    parser.add_argument('--output-dir', type=str, default='transform_analysis_parallel',
                       help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all CPUs)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine number of jobs
    if args.n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = args.n_jobs
    
    print(f"Using {n_jobs} parallel jobs")
    
    # Analyze NavSim samples
    print(f"Analyzing {args.num_samples} NavSim samples...")
    nav_cache = Path(args.navsim_cache)
    
    if nav_cache.exists():
        # Count available samples
        feature_files = sorted(nav_cache.glob("**/transfuser_feature.gz"))
        num_samples = min(args.num_samples, len(feature_files))
        print(f"Found {len(feature_files)} samples, analyzing {num_samples}")
        
        # Parallel processing
        nav_results = Parallel(n_jobs=n_jobs)(
            delayed(analyze_single_sample)(nav_cache, i, 'navsim')
            for i in tqdm(range(num_samples), desc="NavSim")
        )
        
        # Merge results
        nav_stats = merge_stats(nav_results)
    else:
        print(f"NavSim cache not found at {nav_cache}")
        nav_stats = {}
    
    # Analyze Bench2Drive samples
    print(f"\nAnalyzing {args.num_samples} Bench2Drive samples...")
    b2d_cache = Path(args.b2d_cache)
    
    if b2d_cache.exists():
        # Count available samples
        feature_files = sorted(b2d_cache.glob("**/transfuser_feature.gz"))
        num_samples = min(args.num_samples, len(feature_files))
        print(f"Found {len(feature_files)} samples, analyzing {num_samples}")
        
        # Parallel processing
        b2d_results = Parallel(n_jobs=n_jobs)(
            delayed(analyze_single_sample)(b2d_cache, i, 'bench2drive')
            for i in tqdm(range(num_samples), desc="Bench2Drive")
        )
        
        # Merge results
        b2d_stats = merge_stats(b2d_results)
    else:
        print(f"Bench2Drive cache not found at {b2d_cache}")
        b2d_stats = {}
    
    # Print summary
    print_transformation_summary(nav_stats, b2d_stats)
    
    # Plot effects
    if nav_stats and b2d_stats:
        plot_transformation_effects(nav_stats, b2d_stats, output_dir)
    
    # Save detailed results
    results = {
        'navsim': nav_stats,
        'bench2drive': b2d_stats
    }
    
    with open(output_dir / 'transformation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()