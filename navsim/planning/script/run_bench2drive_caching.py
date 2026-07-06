"""
Dataset caching script for Bench2Drive data.
Adapts the caching process to work with Bench2Drive's data structure.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import uuid
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset
from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets for Bench2Drive data.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # Create agent
    agent: AbstractAgent = instantiate(cfg.agent)

    # Create Bench2Drive config
    bench2drive_config = instantiate(cfg.train_test_split.bench2drive)

    # Get split-specific scenarios
    split = cfg.get("split", "train")
    if hasattr(bench2drive_config, "scenarios") and isinstance(bench2drive_config.scenarios, dict):
        if split in bench2drive_config.scenarios:
            bench2drive_config.scenarios = bench2drive_config.scenarios[split]
        else:
            logger.warning(f"Split '{split}' not found in scenarios, using all scenarios")

    # Create scene loader with filtered tokens
    scene_loader = Bench2DriveSceneLoader(
        config=bench2drive_config,
        planner=None,
        trajectory_sampling=None,
    )

    # Filter to only requested tokens
    scene_loader.scene_tokens = [t for t in scene_loader.scene_tokens if t in tokens]

    logger.info(
        f"Extracted {len(scene_loader.scene_tokens)} scenarios for "
        f"thread_id={thread_id}, node_id={node_id}."
    )

    # Create dataset and cache
    dataset = Bench2DriveDataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache.cache_path,
        force_cache_computation=cfg.cache.force_cache_computation,
    )
    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for Bench2Drive dataset caching script.
    :param cfg: omegaconf dictionary
    """
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)

    logger.info("Building Bench2Drive SceneLoader")

    # Create Bench2Drive config
    bench2drive_config = instantiate(cfg.train_test_split.bench2drive)

    # Get split-specific scenarios
    split = cfg.get("split", "train")
    if hasattr(bench2drive_config, "scenarios") and isinstance(bench2drive_config.scenarios, dict):
        if split in bench2drive_config.scenarios:
            bench2drive_config.scenarios = bench2drive_config.scenarios[split]
        else:
            logger.warning(f"Split '{split}' not found in scenarios, using all scenarios")

    # Create scene loader
    scene_loader = Bench2DriveSceneLoader(
        config=bench2drive_config,
        planner=None,
        trajectory_sampling=None,
    )

    logger.info(f"Loaded {len(scene_loader)} scenes from Bench2Drive dataset")

    # Distribute tokens across workers
    tokens = scene_loader.get_scene_tokens()

    # Create args for workers
    worker_args = []
    tokens_per_worker = max(1, len(tokens) // worker.number_of_workers)

    for i in range(worker.number_of_workers):
        start_idx = i * tokens_per_worker
        end_idx = (
            start_idx + tokens_per_worker if i < worker.number_of_workers - 1 else len(tokens)
        )

        worker_tokens = tokens[start_idx:end_idx]
        if worker_tokens:
            worker_args.append(
                {
                    "tokens": worker_tokens,
                    "cfg": cfg,
                }
            )

    logger.info(f"Starting caching of {len(tokens)} tokens across {len(worker_args)} workers")

    # Execute caching across workers
    worker_map(worker, cache_features, worker_args)

    logger.info("Caching completed successfully!")


if __name__ == "__main__":
    main()
