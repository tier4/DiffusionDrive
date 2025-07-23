#!/usr/bin/env python3
"""
Dataset caching script for Bench2Drive dataset.
Caches features and targets for training using Ray for parallel processing.
"""

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import time

import ray
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: Should mention I can only use cache now in the training script
@ray.remote
class SceneProcessor:
    """Ray actor for processing scenes in parallel."""

    def __init__(self, config: Bench2DriveConfig, model_config: TransfuserConfig):
        """Initialize the processor with configurations."""
        # Create scene loader
        self.scene_loader = Bench2DriveSceneLoader(config)

        # Create feature and target builders
        self.feature_builder = Bench2DriveFeatureBuilder(model_config)
        self.target_builder = Bench2DriveTargetBuilder(model_config)

    def process_scene(self, token: str, cache_path: Path) -> Tuple[str, Optional[str]]:
        """
        Process a single scene and save to cache.

        Returns:
            Tuple of (token, error_message or None)
        """
        try:
            # Load scene
            scene = self.scene_loader.get_scene(token)

            # Get agent input for the last frame (where we have full history)
            agent_input = scene.get_agent_input(-1)

            # Compute features
            features = self.feature_builder.compute_features(agent_input)

            # Compute targets
            targets = self.target_builder.compute_targets(scene)

            # Extract log name from token
            log_name = "_".join(token.split("_")[:-1])  # Remove frame number

            # Create directory structure for CacheOnlyDataset compatibility
            token_dir = cache_path / log_name / token
            token_dir.mkdir(parents=True, exist_ok=True)

            # Save features and targets separately as expected by CacheOnlyDataset
            for builder in [self.feature_builder]:
                builder_file = token_dir / (builder.get_unique_name() + ".gz")
                dump_feature_target_to_pickle(builder_file, features)

            for builder in [self.target_builder]:
                builder_file = token_dir / (builder.get_unique_name() + ".gz")
                dump_feature_target_to_pickle(builder_file, targets)

            return token, None

        except Exception as e:
            return token, str(e)


@ray.remote
def process_batch_sequential(
    processor: ray.ObjectRef, tokens: List[str], cache_path: Path, desc: str = "Processing batch"
) -> List[Tuple[str, Optional[str]]]:
    """Process a batch of tokens sequentially on a single actor."""
    results = []
    for token in tqdm(tokens, desc=desc, leave=False):
        result = ray.get(processor.process_scene.remote(token, cache_path))
        results.append(result)
    return results


def cache_bench2drive_dataset(
    data_root: Path,
    cache_path: Path,
    bev_cache_dir: Path = None,
    map_dir: Path = None,
    scenarios: List[str] = None,
    num_workers: int = None,
) -> None:
    """
    Cache Bench2Drive dataset for training using Ray for parallel processing.

    Args:
        data_root: Root directory of Bench2Drive dataset
        cache_path: Path to save cached features/targets
        bev_cache_dir: Optional path to pre-generated BEV cache
        map_dir: Optional path to HD maps
        scenarios: List of scenarios to cache (None = all)
        num_workers: Number of parallel workers (None = auto)
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers)

    # Create cache directory
    cache_path.mkdir(parents=True, exist_ok=True)

    # Get scenarios if not specified
    if scenarios is None:
        scenarios = [
            d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
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

    # Create scene loader to get tokens
    scene_loader = Bench2DriveSceneLoader(config)
    all_tokens = scene_loader.get_scene_tokens()
    logger.info(f"Found {len(all_tokens)} scenes to cache")

    # Create model config
    model_config = TransfuserConfig()

    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 workers

    logger.info(f"Using {num_workers} parallel workers")

    # Put configs in Ray object store for efficiency
    config_ref = ray.put(config)
    model_config_ref = ray.put(model_config)
    cache_path_ref = ray.put(cache_path)

    # Create worker actors
    processors = [SceneProcessor.remote(config_ref, model_config_ref) for _ in range(num_workers)]

    # Split tokens among workers
    tokens_per_worker = len(all_tokens) // num_workers
    token_batches = []

    for i in range(num_workers):
        start_idx = i * tokens_per_worker
        if i == num_workers - 1:
            # Last worker gets remaining tokens
            batch = all_tokens[start_idx:]
        else:
            batch = all_tokens[start_idx : start_idx + tokens_per_worker]
        token_batches.append(batch)

    # Process scenes in parallel
    start_time = time.time()

    # Submit all batches to workers
    futures = []
    for i, (processor, tokens) in enumerate(zip(processors, token_batches)):
        if tokens:  # Only submit if there are tokens
            future = process_batch_sequential.remote(
                processor, tokens, cache_path_ref, f"Worker {i+1}"
            )
            futures.append(future)

    # Collect results with progress bar
    all_results = []
    with tqdm(total=len(all_tokens), desc="Caching scenes") as pbar:
        while futures:
            # Wait for any future to complete
            ready_futures, futures = ray.wait(futures, num_returns=1)

            for future in ready_futures:
                batch_results = ray.get(future)
                all_results.extend(batch_results)
                pbar.update(len(batch_results))

    # Process results
    failed_scenes = [(token, error) for token, error in all_results if error is not None]

    # Clean up Ray actors
    for processor in processors:
        ray.kill(processor)

    # Report results
    elapsed_time = time.time() - start_time
    successful_scenes = len(all_tokens) - len(failed_scenes)

    logger.info(f"Successfully cached {successful_scenes} scenes in {elapsed_time:.1f} seconds")
    logger.info(f"Average time per scene: {elapsed_time / len(all_tokens):.2f} seconds")

    if failed_scenes:
        logger.warning(f"Failed to cache {len(failed_scenes)} scenes:")
        for token, error in failed_scenes[:10]:  # Show first 10 errors
            logger.warning(f"  {token}: {error}")
        if len(failed_scenes) > 10:
            logger.warning(f"  ... and {len(failed_scenes) - 10} more errors")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache Bench2Drive dataset with parallel processing"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/Bench2Drive-mini"),
        help="Root directory of Bench2Drive dataset",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "bench2drive_cache",
        help="Path to save cache",
    )
    parser.add_argument(
        "--bev-cache-dir",
        type=Path,
        default=Path("/workspace/Bench2Drive-mini-full_bev_cache"),
        help="Path to pre-generated BEV cache",
    )
    parser.add_argument(
        "--map-dir", type=Path, default=Path("/workspace/Bench2Drive-Map"), help="Path to HD maps"
    )
    parser.add_argument("--scenarios", nargs="+", default=None, help="Specific scenarios to cache")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto based on CPU count)",
    )
    parser.add_argument(
        "--ray-address", type=str, default=None, help="Ray cluster address (default: local)"
    )

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

    # Initialize Ray if cluster address provided
    if args.ray_address:
        ray.init(address=args.ray_address)
        logger.info(f"Connected to Ray cluster at {args.ray_address}")

    try:
        # Run caching
        cache_bench2drive_dataset(
            data_root=args.data_root,
            cache_path=args.cache_path,
            bev_cache_dir=args.bev_cache_dir,
            map_dir=args.map_dir,
            scenarios=args.scenarios,
            num_workers=args.num_workers,
        )
    finally:
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
