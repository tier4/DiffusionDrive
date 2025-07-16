"""
Dataset wrapper for Bench2Drive data.
Extends the base Dataset class to work with Bench2DriveSceneLoader.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.training.dataset import (
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)

logger = logging.getLogger(__name__)


class Bench2DriveDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Bench2Drive data.
    Compatible with DiffusionDrive training pipeline.
    """

    def __init__(
        self,
        scene_loader: Bench2DriveSceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        """
        Initialize Bench2Drive dataset.

        Args:
            scene_loader: Bench2Drive scene loader
            feature_builders: List of feature builders
            target_builders: List of target builders
            cache_path: Optional path for caching features/targets
            force_cache_computation: Whether to recompute cache
        """
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Load valid cache paths for Bench2Drive data.

        Cache structure:
        cache_path/
        ├── scenario_name/
        │   └── run_name/
        │       └── frame_idx/
        │           ├── transfuser_feature.gz
        │           └── transfuser_target.gz
        """
        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            # Iterate through scenario directories
            for scenario_path in cache_path.iterdir():
                if not scenario_path.is_dir():
                    continue

                # Iterate through run directories
                for run_path in scenario_path.iterdir():
                    if not run_path.is_dir():
                        continue

                    # Iterate through frame directories
                    for frame_path in run_path.iterdir():
                        if not frame_path.is_dir():
                            continue

                        # Check if all required cache files exist
                        found_caches: List[bool] = []
                        for builder in feature_builders + target_builders:
                            data_dict_path = frame_path / (builder.get_unique_name() + ".gz")
                            found_caches.append(data_dict_path.is_file())

                        if all(found_caches):
                            # Create token from path components
                            token = f"{scenario_path.name}_{run_path.name}_{frame_path.name}"
                            valid_cache_paths[token] = frame_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Compute features/targets for a scene and save to cache.

        Args:
            token: Scene identifier
        """
        scene = self._scene_loader.get_scene(token)
        agent_input = scene.get_agent_input()

        # Parse token to get path components
        parts = token.split("_")
        scenario = parts[0]
        run = "_".join(parts[1:-1])  # Handle multi-part run names
        frame_idx = parts[-1]

        # Create cache directory
        token_path = self._cache_path / scenario / run / frame_idx
        os.makedirs(token_path, exist_ok=True)

        # Cache features
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        # Cache targets
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(
        self, token: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Load cached features/targets for a scene.

        Args:
            token: Scene identifier

        Returns:
            Tuple of (features, targets) dictionaries
        """
        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Cache complete dataset into cache folder."""
        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # Determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.get_scene_tokens()
        else:
            tokens_to_cache = set(self._scene_loader.get_scene_tokens()) - set(
                self._valid_cache_paths.keys()
            )
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"Starting caching of {len(tokens_to_cache)} tokens. "
                f"Already cached: {len(self._valid_cache_paths)}"
            )

        for token in tqdm(tokens_to_cache, desc="Caching Bench2Drive Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features and targets for a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, targets) dictionaries
        """
        token = self._scene_loader.scene_tokens[idx]

        if self._cache_path is not None:
            # Load from cache
            if token not in self._valid_cache_paths:
                # Cache on-the-fly if needed
                self._cache_scene_with_token(token)
            features, targets = self._load_scene_with_token(token)
        else:
            # Compute on-the-fly
            scene = self._scene_loader.get_scene(token)
            agent_input = scene.get_agent_input()

            features = {}
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))

            targets = {}
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)
