# Bench2Drive Training Guide

This guide shows how to train DiffusionDrive on Bench2Drive dataset.

## Prerequisites

- Environment setup complete (see `docs/b2d/TRAINING_GUIDE.md`)
- Bench2Drive dataset and HD maps available
- Pre-trained backbone: `./download/pytorch_model.bin`

## Quick Training Steps

### 1. Generate BEV Cache (Required)

```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive-mini \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir /workspace/navsim_workspace/cache/bench2drive_mini_bev_cache \
    --full-bev \
    --workers 4
```

### 2. Cache Dataset

```bash
python3 scripts/cache_bench2drive_dataset.py \
    --dataset-root /workspace/Bench2Drive-mini \
    --cache-path /workspace/navsim_workspace/cache/bench2drive_Mini_cache \
    --bev-cache-dir /workspace/Bench2Drive-mini-full_bev_cache \
    --map-dir /workspace/Bench2Drive-Map \
    --split Mini \
    --num-workers 8
```

### 3. Generate Trajectory Anchors

```bash
python3 scripts/generate_trajectory_anchors.py \
    --cache-path /workspace/navsim_workspace/cache/bench2drive_Mini_cache \
    --output-path ./download/kmeans_bench2drive_traj_20.npy \
    --dataset-type bench2drive
```

### 4. Train Model

```bash
# Using the training script
./train_bench2drive_full.sh
```

Or manually:
```bash
python3 navsim/planning/script/run_bench2drive_training.py \
    agent=diffusiondrive_agent_extended \
    agent.config.dataset_type="bench2drive" \
    agent.config.plan_anchor_path="./download/kmeans_bench2drive_traj_20.npy"
```

## Key Configuration

- Use `diffusiondrive_agent_extended` agent
- Set `dataset_type="bench2drive"` 
- Use Bench2Drive-specific anchors
- BEV generation is REQUIRED

## Debugging Tips

If NaN losses occur:
1. Check training logs for "WARNING: Invalid loss detected"
2. Verify correct anchors are loaded
3. Run analysis scripts to check cached data
4. See BENCH2DRIVE_CACHE_INVESTIGATION.md for known issues

## Known Issues Under Investigation

From cache analysis:
- Status feature minimum value: -26.97 (suspicious)
- Heading range: ±0.110 (possibly clipped)
- Zero agent states and labels
- Coordinate system differences

## Related Documentation

- Technical details: [TRAJECTORY_PROCESSING.md](TRAJECTORY_PROCESSING.md)
- Current investigation: [BENCH2DRIVE_CACHE_INVESTIGATION.md](../BENCH2DRIVE_CACHE_INVESTIGATION.md)
- Full setup: [docs/b2d/TRAINING_GUIDE.md](b2d/TRAINING_GUIDE.md)