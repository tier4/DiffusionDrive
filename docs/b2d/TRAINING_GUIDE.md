# DiffusionDrive Training Guide: Start to Finish

This guide provides complete instructions for training DiffusionDrive from scratch using the Bench2Drive dataset.

> **Quick Start**: For a condensed version using the clean solution, see [`docs/BENCH2DRIVE_TRAINING_GUIDE.md`](../BENCH2DRIVE_TRAINING_GUIDE.md)

## Two Training Approaches

1. **Original Approach** (this guide): Modifies configuration files directly
2. **Clean Solution** (recommended): Uses extended agent with automatic normalization

This guide covers the original approach in detail. For the clean solution that doesn't modify original code, see the Quick Start guide above.

## Prerequisites

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate navsim

# Install DiffusionDrive specific packages
pip install diffusers einops

# Install the package in development mode
pip install -e .
```

### 2. Required Data

- **Bench2Drive Dataset**: Main dataset (we'll use Bench2Drive-mini for this guide)
- **Bench2Drive HD Maps**: Map data at `/workspace/Bench2Drive-Map/`
- **Pretrained Models**: ResNet backbone and trajectory anchors

### 3. Environment Variables

```bash
export NAVSIM_DEVKIT_ROOT=/workspace/DiffusionDrive
export NAVSIM_EXP_ROOT=/workspace/DiffusionDrive/experiments
```

## Step 1: Dataset Preparation

### 1.1 Verify Dataset Structure

```bash
# Check dataset structure
ls -la /workspace/Bench2Drive-mini/
# Should show scenario directories like:
# ConstructionObstacle_Town05_Route68_Weather8/
# HardBreakRoute_Town01_Route30_Weather3/
# etc.

# Check HD maps
ls -la /workspace/Bench2Drive-Map/
# Should show map files like:
# Town01_HD_map.npz
# Town05_HD_map.npz
# etc.
```

### 1.2 Generate BEV Cache (Recommended)

Pre-generate BEV maps for faster training:

```bash
# Generate BEV cache for all scenarios
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive-mini \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir data/bev_cache \
    --full-bev \
    --workers 4 \
    --verbose

# This will create:
# data/bev_cache/scenario_name/frame_number.npz
```

**Note**: For the mini dataset, a pre-generated BEV cache is available at:
`/workspace/Bench2Drive-mini-full_bev_cache/`

### 1.3 Dataset Caching

Cache the training dataset for faster loading:

```bash
# Cache training dataset using custom script
python3 scripts/cache_bench2drive_dataset.py \
    --data-root /workspace/Bench2Drive-mini \
    --cache-path $NAVSIM_EXP_ROOT/bench2drive_mini_cache \
    --bev-cache-dir /workspace/Bench2Drive-mini-full_bev_cache \
    --map-dir /workspace/Bench2Drive-Map

# This will create cache files like:
# $NAVSIM_EXP_ROOT/bench2drive_mini_cache/
# ├── AccidentTwoWays_Town12_Route1444_Weather0_00000.pt
# ├── AccidentTwoWays_Town12_Route1444_Weather0_00001.pt
# └── ...
```

**Note**: The standard NavSim caching scripts don't work with Bench2Drive due to structural differences. Use the custom caching script above.

## Step 2: Configuration Setup

### 2.1 Update Agent Configuration

Edit `navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`:

```yaml
# Agent configuration for Bench2Drive
agent:
  _target_: navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent
  
  # Model configuration
  model:
    bkb_path: "path/to/resnet34_pretrained.pth"  # Update this path
    plan_anchor_path: "path/to/trajectory_anchors.pth"  # Update this path
    
  # Training parameters
  lr: 0.0001
  weight_decay: 0.01
  
  # Loss weights
  loss_weights:
    trajectory: 1.0
    agent_detection: 1.0
    bev_semantic: 14.0  # Important: BEV loss weight
    
  # Data parameters
  data_root: "/workspace/Bench2Drive-mini"
  map_dir: "/workspace/Bench2Drive-Map"
  bev_cache_dir: "data/bev_cache"
```

### 2.2 Create Training Split Configuration

Create `navsim/planning/script/config/common/train_test_split/bench2drive_mini.yaml`:

```yaml
# Bench2Drive training split
train_test_split:
  _target_: navsim.planning.training.bench2drive_dataset.Bench2DriveDataset
  
  # Dataset configuration
  data_root: "/workspace/Bench2Drive-mini"
  scenarios: null  # null means all scenarios
  split: "train"
  
  # Sampling configuration
  sampling_rate: 5  # 10Hz -> 2Hz
  num_frames: 30    # 15 seconds at 2Hz
  num_history_frames: 4
  num_future_frames: 26
  
  # BEV configuration
  map_dir: "/workspace/Bench2Drive-Map"
  bev_cache_dir: "data/bev_cache"
  
  # Cache configuration
  cache_path: "${NAVSIM_EXP_ROOT}/training_cache/"
  use_cache_without_dataset: true
  force_cache_computation: false
```

## Step 3: Training Execution

### 3.1 Start Training

> **Important**: The commands below use the original agent. For the clean solution that handles normalization automatically, use `agent=diffusiondrive_agent_extended` and see the [Quick Start Guide](../BENCH2DRIVE_TRAINING_GUIDE.md).

```bash
# Basic training command (original approach)
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_bench2drive \
    train_test_split=bench2drive_mini \
    split=trainval \
    trainer.params.max_epochs=100 \
    dataloader.params.batch_size=32 \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=true \
    force_cache_computation=false

# For clean solution (recommended):
# ./scripts/training/train.sh --agent "diffusiondrive_agent_extended" --dataset "bench2drive"
```

### 3.2 Monitor Training

Training outputs will be saved to:

```
$NAVSIM_EXP_ROOT/training_diffusiondrive_bench2drive/
├── checkpoints/
│   ├── epoch_010.ckpt
│   ├── epoch_020.ckpt
│   └── ...
├── logs/
│   └── tensorboard_logs/
└── config.yaml
```

Monitor with TensorBoard:

```bash
tensorboard --logdir=$NAVSIM_EXP_ROOT/training_diffusiondrive_bench2drive/logs/
```

### 3.3 Key Training Parameters

**Recommended Settings:**

- **Batch Size**: 16-32 (depending on GPU memory)
- **Learning Rate**: 1e-4
- **Epochs**: 100-200
- **GPU Memory**: 16GB+ recommended
- **BEV Loss Weight**: 14.0 (important for BEV quality)

## Step 4: Evaluation

### 4.1 Evaluate Trained Model

```bash
# Set checkpoint path
export CKPT=$NAVSIM_EXP_ROOT/training_diffusiondrive_bench2drive/checkpoints/epoch_100.ckpt

# Run evaluation
python3 navsim/planning/script/run_pdm_score.py \
    train_test_split=bench2drive_mini \
    agent=diffusiondrive_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=eval_diffusiondrive_bench2drive \
    data_root=/workspace/Bench2Drive-mini
```

### 4.2 Evaluation Metrics

The evaluation will compute:

- **PDM Score**: Planning Distance Metric
- **Trajectory Accuracy**: L2 error on future waypoints
- **Agent Detection**: mAP for surrounding vehicles
- **BEV Segmentation**: IoU for semantic classes

## Step 5: Troubleshooting

### 5.1 Common Issues

**CUDA Out of Memory:**

```bash
# Reduce batch size
dataloader.params.batch_size=8

# Use gradient accumulation
trainer.params.accumulate_grad_batches=4
```

**Dataset Loading Issues:**

```bash
# Test dataset loading
python3 -c "
from navsim.planning.training.bench2drive_dataset import Bench2DriveDataset
from navsim.common.bench2drive_dataloader import Bench2DriveConfig
config = Bench2DriveConfig(data_root='/workspace/Bench2Drive-mini', scenarios=['ConstructionObstacle_Town05_Route68_Weather8'])
dataset = Bench2DriveDataset(config)
print(f'Dataset loaded: {len(dataset)} samples')
"
```

**BEV Generation Issues:**

```bash
# Test BEV generation
python3 -c "
from navsim.common.bev_map_utils import load_map_data
map_data = load_map_data('/workspace/Bench2Drive-Map/Town05_HD_map.npz')
print(f'Map loaded: {len(map_data)} roads')
"
```

### 5.2 Performance Optimization

**Use BEV Cache:**

- Always pre-generate BEV cache for faster training
- Cache reduces BEV generation time from 0.5s to 0.01s per frame

**Parallel Processing:**

- Use multiple workers: `dataloader.params.num_workers=4`
- Use distributed training for multiple GPUs

**Memory Management:**

- Monitor GPU memory usage
- Use mixed precision training: `trainer.params.precision=16`

## Step 6: Advanced Training

### 6.1 Multi-Stage Training

```bash
# Stage 1: Feature extraction only
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=stage1_features \
    trainer.params.max_epochs=50 \
    model.train_perception_only=true

# Stage 2: Full end-to-end training
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=stage2_e2e \
    trainer.params.max_epochs=100 \
    model.checkpoint_path=$NAVSIM_EXP_ROOT/stage1_features/checkpoints/epoch_50.ckpt
```

### 6.2 Hyperparameter Tuning

Create different configurations:

```bash
# High learning rate
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=tune_lr_high \
    model.lr=0.0005

# Different BEV loss weight
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=tune_bev_weight \
    model.loss_weights.bev_semantic=20.0
```

## Step 7: Deployment

### 7.1 Model Export

```bash
# Export trained model
python3 scripts/export_model.py \
    --checkpoint $NAVSIM_EXP_ROOT/training_diffusiondrive_bench2drive/checkpoints/epoch_100.ckpt \
    --output models/diffusiondrive_bench2drive.pth
```

### 7.2 Inference Testing

```bash
# Test inference
python3 scripts/test_inference.py \
    --model models/diffusiondrive_bench2drive.pth \
    --data /workspace/Bench2Drive-mini/ConstructionObstacle_Town05_Route68_Weather8
```

## Expected Results

After successful training, you should see:

- **Training Loss**: Decreasing trajectory, agent detection, and BEV losses
- **Validation Metrics**: PDM score < 0.5, trajectory error < 2.0m
- **BEV Quality**: Rich semantic maps with lanes and traffic elements
- **Model Size**: ~200MB for full model

## Resources

- **Memory Requirements**: 16GB+ GPU memory
- **Training Time**: 6-12 hours on RTX 3090
- **Disk Space**: 50GB+ for caches and checkpoints
- **CPU**: 8+ cores recommended for data loading

This guide provides a complete training pipeline from dataset preparation to model deployment for DiffusionDrive on Bench2Drive dataset.
