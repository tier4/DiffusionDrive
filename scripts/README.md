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
```

## Available Scripts

### Data Processing

#### `cache_bench2drive_dataset.py`
Generate Bench2Drive training cache with true sliding window support.

**NEW Default Mode (True Sliding Window):**
- Slides through ALL 10Hz frames for ~5x more training samples
- Generates ~1000 samples per scenario (vs ~200 in legacy mode)
```bash
python3 scripts/cache_bench2drive_dataset.py \
    --data-root /workspace/Bench2Drive-Base \
    --cache-path "${NAVSIM_EXP_ROOT}/bench2drive_cache_v6" \
    --bev-cache-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg"
```

**Legacy Mode:**
- Use `--use-hardcoded-config` flag for backward compatibility
```bash
python3 scripts/cache_bench2drive_dataset.py \
    --data-root /workspace/Bench2Drive-Base \
    --cache-path "${NAVSIM_EXP_ROOT}/bench2drive_cache_v4" \
    --bev-cache-dir "${NAVSIM_EXP_ROOT}/bench2drive_bev_cache_seg" \
    --use-hardcoded-config
```

#### `generate_bev_cache.py`
Generate BEV map cache (required before training cache).

#### `generate_b2d_plan_anchors.py`
Generate trajectory anchors for Bench2Drive.

### Visualization Scripts

#### 1. `visualize_model_predictions.py`

Creates MP4 videos showing DiffusionDrive model predictions with trajectory visualization overlaid on BEV and camera views.

**Features:**

- Bird's eye view with predicted vs ground truth trajectories
- Multiple camera views (front 3 or all 8 cameras)
- Continuous multi-scene videos for longer durations
- Trajectory projection on camera views
- L2 error metrics display

**Usage:**

```bash
# Basic usage with checkpoint
python3 scripts/visualize_model_predictions.py \
    --checkpoint /path/to/model.pth \
    --output predictions.mp4

# All available arguments
python3 scripts/visualize_model_predictions.py \
    --checkpoint /path/to/checkpoint.pth \        # Path to model checkpoint (required)
    --output output_video.mp4 \                   # Output MP4 file path (default: diffusiondrive_predictions.mp4)
    --data-root /path/to/navsim/dataset \         # Path to NAVSIM dataset (default: /workspace/navsim_workspace/dataset)
    --split test \                                 # Dataset split: test or trainval (default: test)
    --duration 60 \                                # Video duration in seconds (default: 60)
    --fps 2 \                                      # Frames per second (default: 2)
    --all-cameras \                                # Show all 8 cameras instead of front 3
    --debug \                                      # Show debug information
    --rebuild-cache \                              # Force rebuild scene metadata cache
    --test-mode \                                  # Use only first 50 scenes for quick testing
    --cache-name my_cache                         # Custom cache file name (default: navsim_scene_metadata_cache_test)
```

**Examples:**

```bash
# Quick test with front cameras only
python3 scripts/visualize_model_predictions.py \
    --checkpoint model.pth \
    --duration 10 \
    --fps 1

# Full visualization with all cameras
python3 scripts/visualize_model_predictions.py \
    --checkpoint model.pth \
    --all-cameras \
    --duration 120 \
    --fps 4 \
    --output full_predictions.mp4

# Debug mode with custom dataset
python3 scripts/visualize_model_predictions.py \
    --checkpoint model.pth \
    --data-root /custom/dataset/path \
    --split trainval \
    --debug \
    --test-mode
```

#### 2. `visualize_trajectory_anchors.py`

Comprehensive visualization tool for trajectory anchors used in DiffusionDrive's diffusion process.

**Features:**

- Overview plot of all trajectory anchors
- Statistical analysis (length, velocity, curvature distributions)
- Clustering visualization by motion type (forward, turns, complex)
- Quality metrics and feasibility checks
- Side-by-side comparison of different anchor sets

**Usage:**

```bash
# Basic usage
python3 scripts/visualize_trajectory_anchors.py \
    --anchor-path /path/to/anchors.npy

# All available arguments
python3 scripts/visualize_trajectory_anchors.py \
    --anchor-path /path/to/anchors.npy \          # Path to anchor file (required)
    --compare-with /path/to/other_anchors.npy \   # Path to another anchor file for comparison
    --output-dir /path/to/output \                # Directory to save plots (default: same as anchor file)
    --show                                        # Show plots interactively instead of saving
```

**Examples:**

```bash
# Visualize NavSim anchors
python3 scripts/visualize_trajectory_anchors.py \
    --anchor-path download/kmeans_navsim_traj_20.npy \
    --output-dir visualization_results/

# Compare NavSim and Bench2Drive anchors
python3 scripts/visualize_trajectory_anchors.py \
    --anchor-path download/kmeans_navsim_traj_20.npy \
    --compare-with download/kmeans_b2d_traj_20.npy \
    --output-dir comparison_plots/

# Interactive visualization
python3 scripts/visualize_trajectory_anchors.py \
    --anchor-path download/kmeans_navsim_traj_20.npy \
    --show
```

**Generated Plots:**

- `*_overview.png`: All anchors overlaid and in grid layout
- `*_statistics.png`: Statistical analysis (lengths, velocities, curvatures)
- `*_clusters.png`: Anchors grouped by motion characteristics
- `*_quality.png`: Quality metrics and feasibility checks (if metadata available)
- `anchor_comparison.png`: Side-by-side comparison (when using --compare-with)

### Environment Setup

Before running visualization scripts, ensure you have:

```bash
# Required packages
pip install matplotlib numpy pillow opencv-python tqdm

# For video generation (visualize_model_predictions.py)
apt-get update && apt-get install -y ffmpeg

# Set environment variables
export NAVSIM_DEVKIT_ROOT=/workspace/DiffusionDrive
export NAVSIM_EXP_ROOT=/workspace/navsim_workspace
export OPENSCENE_DATA_ROOT=/workspace/navsim_workspace/dataset
```
