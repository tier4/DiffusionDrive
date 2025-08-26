# DiffusionDrive Scripts

This directory contains scripts for DiffusionDrive data processing and utilities.

**Note:** Training and evaluation wrapper scripts have been removed. Use direct Python commands instead.

## Directory Structure

```text
scripts/
├── evaluation/         # Original evaluation scripts (maintained for compatibility)
├── data_processing/   # Data preparation scripts  
└── cache_bench2drive_dataset.py  # Bench2Drive cache generation
```

## Training & Evaluation

For comprehensive training and evaluation commands, see:

- **NavSim**: [docs/train_eval.md](../docs/train_eval.md)
- **Bench2Drive**: [docs/b2d/train_eval.md](../docs/b2d/train_eval.md)

### Quick Examples

**NavSim Training:**

```bash
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=my_experiment
```

**Bench2Drive Training:**

```bash
python3 navsim/planning/script/run_bench2drive_training.py \
    agent=diffusiondrive_agent_extended \
    experiment_name=my_b2d_experiment

## Available Scripts

### Data Processing
- `cache_bench2drive_dataset.py` - Generate Bench2Drive training cache
- `generate_bev_cache.py` - Generate BEV map cache
- `generate_b2d_plan_anchors.py` - Generate trajectory anchors for Bench2Drive
- `visualize_trajectory_anchors.py` - Visualize trajectory anchor clusters
