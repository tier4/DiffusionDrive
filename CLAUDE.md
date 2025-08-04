# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffusionDrive is a truncated diffusion model for end-to-end autonomous driving built on top of the NAVSIM framework. It combines traditional perception (TransFuser backbone) with modern generative modeling (diffusion) for real-time trajectory planning.

## Recent Changes (2025-06-26) - Community Contribution

> **Note:** The following changes were made by a community contributor and are not from the original DiffusionDrive authors.

### Script Reorganization

- All scripts moved to organized structure under `scripts/` directory
- Legacy date-based scripts archived in `archive/legacy/`
- New parameterized scripts with CLI arguments
- See [CHANGELOG.md](../CHANGELOG.md) and [Migration Guide](../docs/MIGRATION_GUIDE.md)

## Development Commands

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate navsim

# Install DiffusionDrive specific packages
pip install diffusers einops

# Install the package in development mode
pip install -e .
```

### Dataset Caching (Required before training/evaluation)

```bash
# Cache training dataset
python3 navsim/planning/script/run_dataset_caching.py agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent train_test_split=navtrain

# Cache evaluation dataset  
python3 navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

### Training

```bash
python3 $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=100 \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False
```

### Evaluation

```bash
# Set checkpoint path
export CKPT=/path/to/checkpoint.pth

# Run evaluation
python3 $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=diffusiondrive_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=diffusiondrive_agent_eval
```

### Testing CUDA/PyTorch Setup

```bash
python3 test_cuda_torch.py
```

## Architecture Overview

### Core Components

1. **Agent System** (`navsim/agents/diffusiondrive/`)
   - `transfuser_agent.py`: Main agent class implementing AbstractAgent interface
   - `transfuser_model_v2.py`: V2TransfuserModel with diffusion-based trajectory planning
   - `transfuser_backbone.py`: Multi-modal fusion of camera and LiDAR features
   - `transfuser_config.py`: Configuration dataclass with all hyperparameters

2. **Diffusion Modules** (`navsim/agents/diffusiondrive/modules/`)
   - `blocks.py`: Building blocks for transformer and diffusion models
   - `scheduler.py`: DDIM scheduler for trajectory generation
   - `multimodal_loss.py`: Multi-task loss computation

3. **Feature Processing**
   - `transfuser_features.py`: Feature and target builders for training
   - Camera: 3 front-view cameras stitched to 1024x256
   - LiDAR: Point cloud histogram (256x256)
   - Status: Velocity, acceleration, driving command

4. **Configuration System**
   - Hydra-based YAML configs in `navsim/planning/script/config/`
   - Agent config: `common/agent/diffusiondrive_agent.yaml`
   - Training splits: `common/train_test_split/`

### Key Design Patterns

1. **Multi-Modal Fusion**: TransFuser backbone with cross-attention at 4 resolution levels
2. **Diffusion Planning**: 20 trajectory modes, 8 timesteps each, truncated to 50 diffusion steps
3. **Multi-Task Learning**: Trajectory prediction + agent detection + BEV segmentation
4. **Distributed Computing**: Ray-based parallel evaluation and training

### Important Configuration Parameters

Before training, update these in `transfuser_config.py`:

- `bkb_path`: Path to pretrained ResNet-34 model
- `plan_anchor_path`: Path to k-means clustered trajectory anchors

Environment variables:

- `NAVSIM_DEVKIT_ROOT`: Root directory of the repository
- `NAVSIM_EXP_ROOT`: Directory for experiment outputs and caches

### Model Outputs

1. **Trajectories**: 20 modes × 8 timesteps (4 seconds at 0.5s intervals)
2. **Agent Detection**: Up to 30 bounding boxes with classifications
3. **BEV Segmentation**: 7-class semantic map (road, walkway, vehicles, etc.)

## Troubleshooting

- If dataset caching fails, check disk space and permissions
- For CUDA errors, verify PyTorch installation with `test_cuda_torch.py`
- Training requires significant GPU memory (recommend 32GB+ for batch size 32)
- Evaluation uses Ray distributed computing - ensure sufficient system resources

## Development Notes

- There is not 'python', just 'python3'

## Testing Guidelines

- Use pytest for testing
- Put tests under ./tests folder

## Code Style Guidelines

### Formatting

- Use black for all Python code formatting to ensure consistency.

### Imports

- Group at Top: All module-level imports must be at the top of the file.
- No sys.path Manipulation: Do not modify sys.path to resolve imports. This is a fragile, outdated practice. The project is installed via pip install -e ., so standard absolute and relative imports will work correctly.

### Error Handling: "Fail Fast and Loud"

- Avoid Suppressing Errors: Do not use try...except blocks to silently ignore or hide errors (e.g., except: pass). Code should fail immediately when an unexpected state occurs.
- Raise Specific Exceptions: Instead of catching an error and returning None, allow the original exception to be raised or raise a more specific, descriptive exception (e.g., ValueError, TypeError). This makes debugging significantly easier.
