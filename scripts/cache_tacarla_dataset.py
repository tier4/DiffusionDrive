#!/usr/bin/env python3
"""
Dataset caching script for TaCarla dataset.
Caches features and targets for training using Ray for parallel processing.

Usage:
    python3 scripts/cache_tacarla_dataset.py \
        --labels-root /mnt/nas/.../TaCarla_labels \
        --sensor-root /mnt/nas/.../TaCarla_extracted \
        --cache-path /mnt/nas/.../tacarla_cache \
        --max-routes 20  # Start small
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import ray
from tqdm import tqdm

from navsim.agents.diffusiondrive.tacarla_config import TaCarlaConfig
from navsim.agents.diffusiondrive.transfuser_features_tacarla import (
    TaCarlaFeatureBuilder,
    TaCarlaTargetBuilder,
)
from navsim.common.tacarla_dataloader import TaCarlaDataConfig, TaCarlaSceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def should_filter_sample(targets):
    """Filter samples that would cause NaN during training."""
    if targets is None:
        return True
    # No valid agents
    if not targets["agent_labels"].any():
        return True
    # Zero trajectory
    import torch
    traj = targets["trajectory"]
    if torch.allclose(traj[:, :2], torch.zeros_like(traj[:, :2]), atol=1e-6):
        return True
    return False


@ray.remote
class TaCarlaSceneProcessor:
    """Ray actor for processing TaCarla scenes."""

    def __init__(self, config: TaCarlaDataConfig, model_config: TaCarlaConfig):
        self.loader = TaCarlaSceneLoader(config)
        self.feature_builder = TaCarlaFeatureBuilder(model_config)
        self.target_builder = TaCarlaTargetBuilder(model_config)

    def process_scene(self, token: str, cache_path: Path) -> Tuple[str, Optional[str]]:
        try:
            scene = self.loader.get_scene(token)
            agent_input = scene.get_agent_input()
            features = self.feature_builder.compute_features(agent_input)
            targets = self.target_builder.compute_targets(scene)

            if targets is None:
                return token, "Skipped: insufficient future frames"

            if should_filter_sample(targets):
                if not targets["agent_labels"].any():
                    return token, "Skipped: no valid agents"
                return token, "Skipped: zero trajectory"

            # Create cache directory
            log_name = "_".join(token.split("_")[:-1])
            token_dir = cache_path / log_name / token
            token_dir.mkdir(parents=True, exist_ok=True)

            # Save features and targets
            feature_file = token_dir / (self.feature_builder.get_unique_name() + ".gz")
            dump_feature_target_to_pickle(feature_file, features)

            target_file = token_dir / (self.target_builder.get_unique_name() + ".gz")
            dump_feature_target_to_pickle(target_file, targets)

            return token, None

        except Exception as e:
            import traceback
            return token, f"{str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Cache TaCarla dataset for training")
    parser.add_argument("--labels-root", type=str, required=True,
                        help="Path to TaCarla_labels/ directory")
    parser.add_argument("--sensor-root", type=str, required=True,
                        help="Path to extracted sensor data directory")
    parser.add_argument("--cache-path", type=str, required=True,
                        help="Output cache directory")
    parser.add_argument("--max-routes", type=int, default=None,
                        help="Limit number of routes (for testing)")
    parser.add_argument("--towns", type=str, nargs="+", default=["Town12", "Town13"],
                        help="Towns to process")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of Ray workers (keep low, 32GB RAM)")
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Build data config
    data_config = TaCarlaDataConfig(
        data_root=Path(args.sensor_root),
        labels_root=Path(args.labels_root),
        sensor_root=Path(args.sensor_root),
        towns=args.towns,
        max_routes=args.max_routes,
    )

    # Build model config
    model_config = TaCarlaConfig()

    # Build scene index (on main process)
    logger.info("Building scene index...")
    loader = TaCarlaSceneLoader(data_config)
    tokens = loader.get_scene_tokens()
    logger.info(f"Total scenes to cache: {len(tokens)}")

    if len(tokens) == 0:
        logger.error("No scenes found. Check labels-root and sensor-root paths.")
        return

    # Initialize Ray
    ray.init(num_cpus=args.num_workers)

    # Create processors
    processors = [
        TaCarlaSceneProcessor.remote(data_config, model_config)
        for _ in range(args.num_workers)
    ]

    # Distribute work
    success_count = 0
    skip_count = 0
    error_count = 0

    # Process in batches
    batch_size = args.num_workers * 4
    for batch_start in range(0, len(tokens), batch_size):
        batch_tokens = tokens[batch_start : batch_start + batch_size]

        futures = []
        for i, token in enumerate(batch_tokens):
            proc = processors[i % len(processors)]
            futures.append(proc.process_scene.remote(token, cache_path))

        results = ray.get(futures)
        for token, error in results:
            if error is None:
                success_count += 1
            elif error.startswith("Skipped"):
                skip_count += 1
            else:
                error_count += 1
                logger.error(f"Error processing {token}: {error[:200]}")

        processed = batch_start + len(batch_tokens)
        logger.info(
            f"Progress: {processed}/{len(tokens)} "
            f"(success={success_count}, skipped={skip_count}, errors={error_count})"
        )

    ray.shutdown()

    logger.info(f"\nCaching complete:")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Cache path: {cache_path}")


if __name__ == "__main__":
    main()
