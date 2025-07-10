# DiffusionDrive Scripts

This directory contains organized scripts for training and evaluating DiffusionDrive models.

## Directory Structure

```text
scripts/
├── training/           # Training scripts
├── evaluation/         # Evaluation scripts
├── utils/             # Shared utilities
└── data/              # Data preparation scripts
```

## Usage Examples

### Training

1. **Single training run with all parameters:**

```bash
./scripts/training/train.sh \
    --name my_experiment \
    --epochs 100 \
    --batch-size 32 \
    --workers 8 \
    --gpus "0,1,2,3,4,5,6,7" \
    --config default_training \
    --agent diffusiondrive_agent \
    --lr 1e-3
```

2. **Batch size sweep with custom learning rates:**

```bash
# Option A: Same learning rate for all batch sizes
./scripts/training/batch_experiments.sh \
    --batch-sizes "32,64,128,256" \
    --epochs 300 \
    --base-name batch_sweep \
    --gpus "0,1,2,3,4,5,6,7" \
    --lr 5e-4

# Option B: Different learning rate for each batch size
./scripts/training/batch_experiments.sh \
    --batch-sizes "32,64,128,256" \
    --epochs 300 \
    --base-name lr_sweep \
    --gpus "0,1,2,3,4,5,6,7" \
    --lr-list "1e-4,2e-4,4e-4,8e-4"
```

### Evaluation

1. **Evaluate single checkpoint:**

```bash
./scripts/evaluation/eval.sh \
    --checkpoint path/to/checkpoint.ckpt \
    --name eval_experiment \
    --agent diffusiondrive_agent
```

2. **Evaluate all checkpoints:**

```bash
./scripts/evaluation/eval_all_checkpoints.sh \
    --dir navsim_workspace/ \
    --pattern "*.ckpt" \
    --agent diffusiondrive_agent
```

## Script Parameters

### train.sh

- `--name`: Experiment name (required)
- `--epochs`: Maximum epochs (default: 100)
- `--batch-size`: Batch size per GPU (default: 32)
- `--workers`: Number of data loading workers (default: 8)
- `--gpus`: GPU devices as comma-separated list (default: 0,1,2,3,4,5,6,7)
- `--config`: Training config name (default: default_training)
- `--agent`: Agent type (default: diffusiondrive_agent)
- `--lr`: Learning rate override (default: uses agent's configured LR - 6e-4 for diffusiondrive_agent, 1e-4 for others)

### batch_experiments.sh

- `--batch-sizes`: Comma-separated list of batch sizes (default: 32,64,128,256)
- `--epochs`: Maximum epochs for all experiments (default: 300)
- `--base-name`: Base name for experiments (default: batch_sweep)
- `--gpus`: GPU devices as comma-separated list (default: 0,1,2,3,4,5,6,7)
- `--lr`: Single learning rate for all experiments (default: uses agent's configured LR)
- `--lr-list`: Comma-separated learning rates for each batch size (default: uses agent's configured LR for all)

**Note**: `--lr` and `--lr-list` are mutually exclusive. When using `--lr-list`, provide exactly one learning rate per batch size.

**Experiment Naming**: The script now includes the learning rate in experiment names:
- Format: `{base_name}_bs{batch_size}_lr{learning_rate}_ep{epochs}`
- Learning rate formatting in names:
  - Scientific notation: `1e-4` → `1e4`
  - Decimal notation: `0.001` → `0p001`
- Default: When no custom LR is specified, uses `lr6e4` (the default for diffusiondrive_agent)

### eval.sh

- `--checkpoint`: Path to checkpoint file (required)
- `--name`: Experiment name (auto-generated if not provided)
- `--agent`: Agent type (default: diffusiondrive_agent)

### eval_all_checkpoints.sh

- `--dir`: Directory containing checkpoints (required)
- `--pattern`: File pattern to match (default: *.ckpt)
- `--agent`: Agent type (default: diffusiondrive_agent)

## Environment Requirements

Before running any scripts, ensure these environment variables are set:

- `NAVSIM_DEVKIT_ROOT`: Root directory of the repository
- `NAVSIM_EXP_ROOT`: Directory for experiment outputs

## Learning Rate Configuration

The training scripts now support learning rate override via the `--lr` flag. This allows you to experiment with different learning rates without modifying the agent configuration files.

**Note**: When training on multiple GPUs with DDP (DistributedDataParallel), the batch size specified is per GPU. For example, with `--batch-size 64` on 8 GPUs, the effective batch size is 512 (64 × 8).

## Logging

All scripts automatically create timestamped log files in the `logs/` directory.
