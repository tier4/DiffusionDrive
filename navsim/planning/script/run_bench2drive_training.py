"""
Training script for Bench2Drive dataset.
Modified version of run_training.py to support Bench2Drive data.
"""

from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader
from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_bench2drive_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Bench2DriveDataset, Bench2DriveDataset]:
    """
    Builds training and validation datasets for Bench2Drive
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    # Create Bench2Drive configs for train and val
    train_config = instantiate(cfg.train_test_split.bench2drive)
    val_config = instantiate(cfg.train_test_split.bench2drive)
    
    # Get split-specific scenarios
    split = cfg.get('split', 'train')
    
    # Set scenarios based on split
    if hasattr(train_config, 'scenarios') and isinstance(train_config.scenarios, dict):
        train_config.scenarios = train_config.scenarios.get('train', [])
        val_config.scenarios = val_config.scenarios.get('val', [])
    
    # Create scene loaders
    train_scene_loader = Bench2DriveSceneLoader(
        config=train_config,
        planner=None,
        trajectory_sampling=None,
    )
    
    val_scene_loader = Bench2DriveSceneLoader(
        config=val_config,
        planner=None,
        trajectory_sampling=None,
    )
    
    logger.info(f"Loaded {len(train_scene_loader)} training scenes")
    logger.info(f"Loaded {len(val_scene_loader)} validation scenes")
    
    # Create datasets
    train_data = Bench2DriveDataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    
    val_data = Bench2DriveDataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent with Bench2Drive data.
    :param cfg: omegaconf dictionary
    """
    
    # Autograd anomaly detection disabled for performance
    # Uncomment the following lines if debugging NaN issues:
    # import torch
    # torch.autograd.set_detect_anomaly(True)
    # logger.warning("Autograd anomaly detection enabled - this will slow down training!")
    
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    
    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    
    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )
    
    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        
        # Get train/val scenario splits from official JSON split file
        import json
        import os
        from pathlib import Path
        
        # Load official Bench2Drive split
        split_file = Path(__file__).parent / "config" / "common" / "train_test_split" / "bench2drive_base_train_val_split.json"
            
        if split_file.exists():
            logger.info(f"Loading official splits from: {split_file}")
            with open(split_file, 'r') as f:
                official_splits = json.load(f)
            
            # Get validation scenarios from JSON
            val_scenarios_json = official_splits.get('val', [])
            # Remove 'v1/' prefix to match cache directory names
            val_scenarios = [s.replace('v1/', '') for s in val_scenarios_json]
            
            # Get all available cached scenarios
            cached_scenarios = [d for d in os.listdir(cfg.cache_path) if os.path.isdir(os.path.join(cfg.cache_path, d))]
            
            # Training scenarios = all cached scenarios EXCEPT validation scenarios
            train_scenarios = [s for s in cached_scenarios if s not in val_scenarios]
            
            logger.info(f"Using official split: {len(train_scenarios)} train, {len(val_scenarios)} val scenarios")
        else:
            logger.warning(f"Official split file not found at {split_file}, falling back to config")
            # Fallback to config-based splits
            train_scenarios = []
            val_scenarios = []
            
            if hasattr(cfg.train_test_split, 'scenarios'):
                scenarios_config = cfg.train_test_split.scenarios
                if hasattr(scenarios_config, 'train'):
                    train_scenarios = list(scenarios_config['train'])
                    val_scenarios = list(scenarios_config.get('val', []))
                else:
                    train_scenarios = list(scenarios_config) if hasattr(scenarios_config, '__iter__') else []
                    val_scenarios = list(scenarios_config) if hasattr(scenarios_config, '__iter__') else []
            
        logger.info(f"Training scenarios: {train_scenarios}")
        logger.info(f"Validation scenarios: {val_scenarios}")
        
        # For Bench2Drive, we use the standard CacheOnlyDataset with scenario splits
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=train_scenarios,
        )
        logger.info(f"Training dataset length: {len(train_data)}")
        logger.info(f"Training dataset tokens: {getattr(train_data, 'tokens', [])[:5]}...")  # Show first 5
        
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=val_scenarios,
        )
        logger.info(f"Validation dataset length: {len(val_data)}")
        logger.info(f"Validation dataset tokens: {getattr(val_data, 'tokens', [])[:5]}...")  # Show first 5
    else:
        logger.info("Building Bench2Drive SceneLoader")
        train_data, val_data = build_bench2drive_datasets(cfg, agent)
    
    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))
    
    # Instantiate callbacks from the Hydra config
    callbacks = [instantiate(c) for c in cfg.trainer.callbacks] if cfg.trainer.get("callbacks") else []
    # Add agent-specific callbacks
    callbacks.extend(agent.get_training_callbacks())
    
    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)
    
    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()