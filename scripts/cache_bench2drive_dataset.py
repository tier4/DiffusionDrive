#!/usr/bin/env python3
"""
Dataset caching script for Bench2Drive dataset.
Caches features and targets for training.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cache_bench2drive_dataset(
    data_root: Path,
    cache_path: Path,
    bev_cache_dir: Path = None,
    map_dir: Path = None,
    scenarios: List[str] = None,
) -> None:
    """
    Cache Bench2Drive dataset for training.
    
    Args:
        data_root: Root directory of Bench2Drive dataset
        cache_path: Path to save cached features/targets
        bev_cache_dir: Optional path to pre-generated BEV cache
        map_dir: Optional path to HD maps
        scenarios: List of scenarios to cache (None = all)
    """
    # Create cache directory
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Get scenarios if not specified
    if scenarios is None:
        scenarios = [d.name for d in data_root.iterdir() 
                    if d.is_dir() and not d.name.startswith(".")]
        logger.info(f"Found {len(scenarios)} scenarios in {data_root}")
    
    # Create configuration
    config = Bench2DriveConfig(
        data_root=data_root,
        scenarios=scenarios,
        sampling_rate=5,  # 10Hz -> 2Hz
        num_frames=30,
        num_history_frames=4,
        num_future_frames=26,
        extract_tar=False,
        map_dir=map_dir,
        bev_cache_dir=bev_cache_dir,
    )
    
    # Create scene loader
    scene_loader = Bench2DriveSceneLoader(config)
    logger.info(f"Created scene loader with {len(scene_loader)} scenes")
    
    # Create feature and target builders
    model_config = TransfuserConfig()
    feature_builder = Bench2DriveFeatureBuilder(model_config)
    target_builder = Bench2DriveTargetBuilder(model_config)
    
    # Process all scenes
    failed_scenes = []
    
    for i, token in enumerate(tqdm(scene_loader.get_scene_tokens(), desc="Caching scenes")):
        try:
            # Load scene
            scene = scene_loader.get_scene(token)
            
            # Get agent input for the last frame (where we have full history)
            agent_input = scene.get_agent_input(-1)
            
            # Compute features
            features = feature_builder.compute_features(agent_input)
            
            # Compute targets
            targets = target_builder.compute_targets(scene)
            
            # Extract log name from token
            # Token format is typically: scenario_framenum
            # We need to create directory structure: cache_path/log_name/token/
            log_name = '_'.join(token.split('_')[:-1])  # Remove frame number
            
            # Create directory structure for CacheOnlyDataset compatibility
            token_dir = cache_path / log_name / token
            token_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features and targets separately as expected by CacheOnlyDataset
            for builder in [feature_builder]:
                builder_file = token_dir / (builder.get_unique_name() + ".gz")
                dump_feature_target_to_pickle(builder_file, features)
            
            for builder in [target_builder]:
                builder_file = token_dir / (builder.get_unique_name() + ".gz") 
                dump_feature_target_to_pickle(builder_file, targets)
            
        except Exception as e:
            logger.warning(f"Failed to cache scene {token}: {e}")
            failed_scenes.append((token, str(e)))
            continue
    
    # Report results
    logger.info(f"Successfully cached {len(scene_loader) - len(failed_scenes)} scenes")
    if failed_scenes:
        logger.warning(f"Failed to cache {len(failed_scenes)} scenes:")
        for token, error in failed_scenes[:10]:  # Show first 10 errors
            logger.warning(f"  {token}: {error}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Bench2Drive dataset")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/Bench2Drive-mini"),
                       help="Root directory of Bench2Drive dataset")
    parser.add_argument("--cache-path", type=Path, 
                       default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "bench2drive_cache",
                       help="Path to save cache")
    parser.add_argument("--bev-cache-dir", type=Path, 
                       default=Path("/workspace/Bench2Drive-mini-full_bev_cache"),
                       help="Path to pre-generated BEV cache")
    parser.add_argument("--map-dir", type=Path, default=Path("/workspace/Bench2Drive-Map"),
                       help="Path to HD maps")
    parser.add_argument("--scenarios", nargs="+", default=None,
                       help="Specific scenarios to cache")
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    if args.bev_cache_dir and not args.bev_cache_dir.exists():
        logger.warning(f"BEV cache directory not found: {args.bev_cache_dir}")
        args.bev_cache_dir = None
    
    if args.map_dir and not args.map_dir.exists():
        logger.warning(f"Map directory not found: {args.map_dir}")
        args.map_dir = None
    
    # Run caching
    cache_bench2drive_dataset(
        data_root=args.data_root,
        cache_path=args.cache_path,
        bev_cache_dir=args.bev_cache_dir,
        map_dir=args.map_dir,
        scenarios=args.scenarios,
    )


if __name__ == "__main__":
    main()