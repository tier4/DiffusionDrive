#!/usr/bin/env python3
"""
Quick test to see what format the dataloader returns.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from torch.utils.data import DataLoader
from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Load config
cfg = OmegaConf.load("navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml")
config = instantiate(cfg.config)

# Create builders
feature_builder = TransfuserFeatureBuilder(config)
target_builder = TransfuserTargetBuilder(config)

# Create dataset
dataset = CacheOnlyDataset(
    cache_path="/workspace/navsim_workspace/exp/training_cache",
    feature_builders=[feature_builder],
    target_builders=[target_builder],
)

print(f"Dataset length: {len(dataset)}")

# Check single sample
sample = dataset[0]
print(f"\nSingle sample type: {type(sample)}")
if isinstance(sample, tuple):
    print(f"Tuple length: {len(sample)}")
    for i, item in enumerate(sample):
        print(f"  Item {i} type: {type(item)}")
        if isinstance(item, dict):
            print(f"  Item {i} keys: {list(item.keys())[:5]}...")

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

# Check batch
for batch in dataloader:
    print(f"\nBatch type: {type(batch)}")
    if isinstance(batch, list):
        print(f"List length: {len(batch)}")
        if len(batch) > 0:
            print(f"First item type: {type(batch[0])}")
            if isinstance(batch[0], dict):
                print(f"First item keys: {list(batch[0].keys())[:5]}...")
    elif isinstance(batch, tuple):
        print(f"Tuple length: {len(batch)}")
        for i, item in enumerate(batch):
            print(f"  Item {i} type: {type(item)}")
    break