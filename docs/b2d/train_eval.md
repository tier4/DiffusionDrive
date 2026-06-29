# Bench2Drive Training and Evaluation Guide

This guide provides comprehensive commands for training and evaluating DiffusionDrive models on the Bench2Drive dataset.

## Prerequisites

Ensure environment variables are set (*see [note on naming](#note-on-environment-variable-naming)*):

```bash
export NAVSIM_DEVKIT_ROOT=/workspace/DiffusionDrive
export NAVSIM_EXP_ROOT=/workspace/navsim_workspace/exp
```

## Cache Setup

**Required:** Before training, you must generate both BEV maps and Bench2Drive cache:

### 1. Generate BEV Maps

The system supports two generation modes:

#### Vector Mode (Default - Sparse)
Generates sparse BEV maps with lane lines only (~3,600 pixels per frame).

**Simple:**
```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache"
```

**With options:**
```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache" \
    --generation-type vector \
    --lane-thickness 0.4 \
    --workers 16
```

#### Segmentation Mode (Recommended - Dense)
Generates dense BEV maps with filled road surfaces (~20,500 pixels per frame, 5.6x better coverage).

**Simple:**
```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg" \
    --generation-type segmentation
```

**With options:**
```bash
python3 scripts/generate_bev_cache.py \
    --data-root /workspace/Bench2Drive \
    --map-dir /workspace/Bench2Drive-Map \
    --output-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg" \
    --generation-type segmentation \
    --lane-width 5.0 \
    --fill-drivable \
    --workers 16
```

*Use `--help` for complete parameter documentation.*

### 2. Generate Bench2Drive Cache

After generating BEV maps, create the training cache. Use the appropriate BEV cache directory based on your chosen mode.

**NEW: True Sliding Window Mode (Default, Recommended):**
- Slides through ALL 10Hz frames, downsamples within window
- Generates ~5x more samples (~1000 per scenario vs ~200)
- Better data utilization (95% of frames used)
```bash
python3 scripts/cache_bench2drive_dataset.py \
    --data-root /workspace/Bench2Drive-Base \
    --cache-path "${NAVSIM_EXP_ROOT}/bench2drive_cache_v6" \
    --bev-cache-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg"
```

**Legacy Mode (Downsample-first):**
- Downsamples to 2Hz first, then slides window
- Generates ~200 samples per scenario
- Use for backward compatibility with v4 caches
```bash
python3 scripts/cache_bench2drive_dataset.py \
    --data-root /workspace/Bench2Drive-Base \
    --cache-path "${NAVSIM_EXP_ROOT}/bench2drive_cache_v4" \
    --bev-cache-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg" \
    --use-hardcoded-config
```

**Full (all parameters):**
```bash
python3 scripts/cache_bench2drive_dataset.py \
    --data-root /workspace/Bench2Drive-Base \
    --cache-path "${NAVSIM_EXP_ROOT}/bench2drive_cache_v6" \
    --bev-cache-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg" \
    --map-dir /workspace/Bench2Drive-Map \
    --scenarios "scenario1" "scenario2" \
    --num-workers 16 \
    --ray-address "ray://localhost:10001"
```

**Note:** BEV cache does NOT need to be regenerated when switching between sliding modes.

*Use `--help` for complete parameter documentation.*

## Training

### Simple Training

Use the training script with default settings:

```bash
./scripts/training/train.sh \
    --name "bench2drive_experiment" \
    --epochs 1000 \
    --batch-size 64 \
    --workers 32 \
    --gpus "0,1,2,3,4,5,6,7" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive" \
    --lr 5e-5
```

**Note:** Bench2Drive training automatically uses `default_training_w_callbacks` config, which includes:

- **TimeLoggingCallback**: Tracks training/validation time per epoch
- **ModelCheckpoint**: Saves best 5 models and checkpoints every 10 epochs

### Custom Training

Customize parameters for your specific needs:

```bash
./scripts/training/train.sh \
    --name "my_custom_experiment" \
    --epochs 3000 \
    --batch-size 128 \
    --workers 16 \
    --gpus "0,1,2,3" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive" \
    --lr 1e-4
```

### Training with Custom Paths

Override cache directory or anchor trajectory files:

```bash
# Custom cache directory
./scripts/training/train.sh \
    --name "experiment_custom_cache" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive" \
    --cache-path "/workspace/my_custom_cache/"

# Custom anchor trajectory file
./scripts/training/train.sh \
    --name "experiment_custom_anchors" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive" \
    --anchor-path "/workspace/kmeans_b2d_v3_traj_50.npy"

# Both custom paths
./scripts/training/train.sh \
    --name "experiment_full_custom" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive" \
    --cache-path "/workspace/experimental_cache/" \
    --anchor-path "/workspace/experimental_anchors.npy" \
    --config "default_training_w_callbacks" \
    --lr 5e-5
```

### Mini Dataset Training

For faster experiments with the mini dataset:

```bash
./scripts/training/train.sh \
    --name "bench2drive_mini_test" \
    --epochs 100 \
    --batch-size 32 \
    --workers 8 \
    --gpus "0,1,2,3" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive_mini" \
    --lr 5e-5
```

## Evaluation

### Simple Evaluation

```bash
python3 navsim/planning/script/run_pdm_score.py \
    train_test_split=bench2drive \
    agent=diffusiondrive_agent_b2d \
    agent.checkpoint_path=/path/to/checkpoint.pth
```

### Full Evaluation (All Parameters)

```bash
python3 navsim/planning/script/run_pdm_score.py \
    train_test_split=bench2drive \
    agent=diffusiondrive_agent_b2d \
    worker=ray_distributed \
    agent.checkpoint_path=$CKPT \
    experiment_name=diffusiondrive_agent_bench2drive_eval \
    challenge=open_loop_boxes \
    split=val \
    metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache" \
    output_dir="${NAVSIM_EXP_ROOT}/evaluation_results"
```

## Key Configuration Details

### Training Configuration

- **Script**: `scripts/training/train.sh` (parameterized shell script)
- **Training Python Script**: `run_bench2drive_training.py` (automatically selected for Bench2Drive)
- **Default Config**: `default_training_w_callbacks.yaml` (auto-selected for Bench2Drive)
- **Agent**: `diffusiondrive_agent_b2d` (includes B2D features and normalization)
- **Max Epochs**: 1000-3000 (typical for Bench2Drive)
- **Validation**: Every epoch by default
- **Callbacks**: TimeLoggingCallback and ModelCheckpoint (enabled by default)

### Available Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Experiment name | Required |
| `--epochs` | Maximum training epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--workers` | Number of data workers | 8 |
| `--gpus` | GPU devices (comma-separated) | "0,1,2,3,4,5,6,7" |
| `--agent` | Agent configuration | "diffusiondrive_agent" |
| `--dataset` | Dataset type | Auto-detected |
| `--config` | Training config file | "default_training_w_callbacks" for B2D |
| `--lr` | Learning rate override | Agent default |
| `--cache-path` | Custom cache directory | Auto-determined by dataset |
| `--anchor-path` | Custom anchor trajectory file | Default B2D anchors |
| `--print-config` | Print config and exit | False |

### Checkpoint Saving

Checkpoints are automatically configured with:

- **save_top_k**: 5 best models (keeps only the 5 best checkpoints)
- **Monitor**: `val/loss_epoch` (validation loss)
- **Mode**: `min` (saves models with lowest validation loss)
- **every_n_epochs**: 10 (saves a checkpoint every 10 epochs regardless of performance)
- **save_last**: Not configured by default (add if you want to keep the last epoch)
- **Location**: `${NAVSIM_EXP_ROOT}/training_*/lightning_logs/version_0/checkpoints/`

### Time Logging

Training automatically logs timing metrics:

- **time_epoch**: Time for each training epoch (seconds)
- **time_eval**: Time for each validation epoch (seconds)
- **Logged to**: TensorBoard logs in experiment output directory

### Cache Requirements

- **Training**: Requires pre-computed cache (`use_cache_without_dataset=True`)
- **Path**: `${NAVSIM_EXP_ROOT}/bench2drive_cache/`
- **Force recomputation**: Set `force_cache_computation=False` (default)

## Differences from NavSim

| Parameter | NavSim | Bench2Drive |
|-----------|--------|-------------|
| Training Script | `run_training.py` | `run_bench2drive_training.py` |
| Agent | `diffusiondrive_agent` | `diffusiondrive_agent_b2d` |
| Config | `default_training.yaml` | `default_training_w_callbacks.yaml` |
| Train Split | `navtrain` | `bench2drive` |
| Max Epochs | 100 | 1000-3000 |
| Callbacks | None by default | TimeLogging + ModelCheckpoint |
| Cache Required | Optional | Mandatory |
| Normalization | Hardcoded NavSim values | B2D-specific values |
| Path Overrides | Not available | `--cache-path`, `--anchor-path` |

## Docker Training (Blackwell GPU)

On machines with NVIDIA Blackwell GPUs (sm_120, e.g. RTX PRO 6000), the base Docker image's PyTorch lacks sm_120 support. Use the `diffusiondrive:blackwell` image and `PYTHONPATH` instead of `pip install -e .` (which downgrades torch via `requirements.txt`).

### Build the Blackwell image (one-time)

```bash
cat > /tmp/Dockerfile.blackwell << 'EOF'
FROM diffusiondrive:latest
USER user
RUN pip install --no-cache-dir --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
EOF
docker build -f /tmp/Dockerfile.blackwell -t diffusiondrive:blackwell .
```

### Training

```bash
docker run -d --rm --memory=12g --cpus=8 --gpus=1 --shm-size=4g \
  --name b2d_training \
  -e PYTHONPATH=/workspace/DiffusionDrive \
  -v /path/to/workspace:/workspace \
  -v /mnt:/mnt \
  -w /workspace/DiffusionDrive/navsim/planning/script \
  --entrypoint "" \
  diffusiondrive:blackwell \
  python3 run_bench2drive_training.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    cache_path=/path/to/dataset_cache \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    trainer.params.max_epochs=1000 \
    trainer.params.strategy=auto \
    trainer.params.accelerator=gpu \
    trainer.params.precision=16-mixed \
    +trainer.params.devices=1 \
    dataloader.params.batch_size=128 \
    dataloader.params.num_workers=4 \
    output_dir=/path/to/training_output/<EXPERIMENT_NAME>
```

**Common Hydra overrides:**

| Override | Description | Example |
|----------|-------------|---------|
| `agent.config.trajectory_weight` | Trajectory loss weight | `0.1` |
| `agent.config.trajectory_cls_weight` | Mode classification weight | `0.5` |
| `agent.config.trajectory_reg_weight` | Trajectory regression weight | `0.1` |
| `agent.config.bev_semantic_weight` | BEV semantic loss weight | `32.0` |
| `agent.checkpoint_path` | Resume from checkpoint | `/path/to/checkpoint.ckpt` |
| `agent.lr` | Learning rate | `1e-4` |

### BEV cache for large towns

Town11/12/13 maps (1.4-1.8GB compressed) expand to 20GB+ during KDTree construction. Process one town at a time with swap enabled:

```bash
docker run --rm --memory=24g --memory-swap=-1 --cpus=8 --shm-size=4g \
  -v /path/to/workspace:/workspace \
  -v /mnt:/mnt \
  -w /workspace/DiffusionDrive \
  --entrypoint "" \
  diffusiondrive:latest \
  bash -c "pip install -e . --quiet && python3 scripts/generate_bev_cache.py \
    --data-root /path/to/Bench2Drive-Base \
    --map-dir /path/to/Bench2Drive-Map \
    --output-dir /path/to/bev_cache_seg \
    --generation-type segmentation \
    --scenarios SCENARIO_1 SCENARIO_2 ... \
    --workers 2"
```

---

## Note on Environment Variable Naming

The environment variables use `NAVSIM_*` naming (e.g., `NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`) for historical reasons. This can be confusing since we're training on Bench2Drive, not NavSim data.

**Why `NAVSIM_*` naming:**

- DiffusionDrive was built as an extension of the NavSim framework
- It reuses NavSim's training infrastructure, evaluation pipeline, and configuration system
- The variables refer to the **framework/infrastructure**, not the **dataset**

**What the variables actually mean:**

- `NAVSIM_DEVKIT_ROOT`: DiffusionDrive codebase root directory
- `NAVSIM_EXP_ROOT`: Experiment outputs directory (used for both NavSim and Bench2Drive experiments)

Both NavSim and Bench2Drive training use the same workspace structure and framework - only the dataset differs.
