"""
Dataset factory for creating appropriate dataset based on configuration.
Supports both NAVSIM and Bench2Drive datasets.
"""

import logging
from typing import List, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating datasets based on configuration."""

    @staticmethod
    def create_dataset(
        cfg: DictConfig,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        use_cache_only: bool = False,
    ) -> Union[Dataset, Bench2DriveDataset, CacheOnlyDataset]:
        """
        Create dataset based on configuration.

        Args:
            cfg: Hydra configuration
            feature_builders: List of feature builders
            target_builders: List of target builders
            use_cache_only: Whether to use cache-only mode

        Returns:
            Dataset instance (NAVSIM or Bench2Drive)
        """
        dataset_type = cfg.get("dataset_type", "navsim")

        logger.info(f"Creating dataset of type: {dataset_type}")

        if dataset_type == "bench2drive":
            return DatasetFactory._create_bench2drive_dataset(
                cfg, feature_builders, target_builders, use_cache_only
            )
        else:
            # Default to NAVSIM dataset
            return DatasetFactory._create_navsim_dataset(
                cfg, feature_builders, target_builders, use_cache_only
            )

    @staticmethod
    def _create_bench2drive_dataset(
        cfg: DictConfig,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        use_cache_only: bool,
    ) -> Union[Bench2DriveDataset, CacheOnlyDataset]:
        """Create Bench2Drive dataset."""

        if use_cache_only:
            # Use generic cache-only dataset
            return CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=None,  # Will use all available
            )
        else:
            # Create Bench2Drive scene loader
            bench2drive_config = instantiate(cfg.bench2drive)

            # Get split-specific scenarios
            split = cfg.get("split", "train")
            if hasattr(bench2drive_config, "scenarios") and isinstance(
                bench2drive_config.scenarios, dict
            ):
                if split in bench2drive_config.scenarios:
                    bench2drive_config.scenarios = bench2drive_config.scenarios[split]
                else:
                    logger.warning(f"Split '{split}' not found in scenarios, using all scenarios")

            scene_loader = Bench2DriveSceneLoader(
                config=bench2drive_config,
                planner=cfg.get("planner"),
                trajectory_sampling=cfg.get("trajectory_sampling"),
            )

            # Create dataset
            return Bench2DriveDataset(
                scene_loader=scene_loader,
                feature_builders=feature_builders,
                target_builders=target_builders,
                cache_path=cfg.get("cache_path"),
                force_cache_computation=cfg.get("force_cache_computation", False),
            )

    @staticmethod
    def _create_navsim_dataset(
        cfg: DictConfig,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        use_cache_only: bool,
    ) -> Union[Dataset, CacheOnlyDataset]:
        """Create NAVSIM dataset."""

        if use_cache_only:
            return CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=cfg.get("log_names"),
            )
        else:
            # Create standard NAVSIM scene loader
            scene_loader = SceneLoader(
                sensor_blobs_path=cfg.sensor_blobs_path,
                data_path=cfg.data_path,
                scene_filter=instantiate(cfg.scene_filter),
                sensor_config=instantiate(cfg.sensor_config),
            )

            # Create dataset
            return Dataset(
                scene_loader=scene_loader,
                feature_builders=feature_builders,
                target_builders=target_builders,
                cache_path=cfg.get("cache_path"),
                force_cache_computation=cfg.get("force_cache_computation", False),
            )
