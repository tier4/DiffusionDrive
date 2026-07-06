# DiffusionDrive Notebooks

This directory contains Jupyter notebooks for exploring the CARLA-NavSim dataset and visualizing DiffusionDrive model outputs.

## Notebooks Overview

### Data Exploration Notebooks

1. **`01_data_loading_basics.ipynb`**
   - Basic data loading and visualization
   - Understanding Transfuser features and targets
   - Dataset structure and shapes
   - Single sample visualization

2. **`02_temporal_analysis.ipynb`**
   - Temporal sequence analysis
   - Frame ordering and continuity
   - Animated visualizations of sequences
   - Multi-frame trajectory analysis

3. **`03_scenario_search.ipynb`**
   - Scenario search and filtering
   - Finding specific driving situations
   - Statistical analysis of scenarios
   - Batch processing tools

### Model Analysis Notebooks

4. **`05_model_output_visualization.ipynb`**
   - DiffusionDrive model output visualization
   - Model loading from checkpoints
   - Prediction visualization:
     - Trajectory predictions (8 waypoints, 4 seconds)
     - Agent detection (up to 30 agents)
     - BEV semantic segmentation (7 classes)
   - Comparison with ground truth
   - Interactive sample browser
   - Batch evaluation and metrics

### Legacy Notebooks

- **`explore_carla_navsim_data.ipynb`** - Original comprehensive notebook (46k+ lines)

### evaluation/
- `compare_eval.ipynb` - Compare evaluation results across different model configurations
  - Analyzes performance metrics from multiple training runs
  - Creates comparison tables for different hyperparameters (epochs, batch sizes)
  - Useful for hyperparameter tuning and model selection

- `visualization_eval.ipynb` - Visualization utilities for evaluation results
  - BEV (Bird's Eye View) plots
  - Camera view visualizations
  - Custom visualization functions

## Key Concepts

### Transfuser Features
- **Camera**: 3 front cameras stitched to 1024×256×3
- **LiDAR**: BEV histogram at 256×256×1
- **Status**: 8-element vector `[left, straight, right, unknown, vx, vy, ax, ay]`

### Transfuser Targets
- **Trajectory**: 8×3 future waypoints (x, y, heading)
- **Agent States**: 30×5 bounding boxes (x, y, heading, length, width)
- **Agent Labels**: 30 binary detection scores
- **BEV Semantic Map**: 7×128×256 segmentation

### Driving Commands
- 0: LEFT
- 1: STRAIGHT
- 2: RIGHT
- 3: UNKNOWN

## Usage

1. Set environment variables:
```python
os.environ['NAVSIM_DEVKIT_ROOT'] = '/workspace/DiffusionDrive'
os.environ['NAVSIM_EXP_ROOT'] = '/workspace/navsim_workspace'
```

2. Ensure cached data exists:
```bash
python navsim/planning/script/run_dataset_caching.py agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent train_test_split=navtrain
```

3. For model visualization, provide checkpoint path:
```python
CHECKPOINT_PATH = '/path/to/checkpoint.ckpt'
```

## Requirements

- PyTorch with CUDA support
- NavSim framework
- Matplotlib for visualizations
- IPyWidgets for interactive browsing
- DiffusionDrive dependencies (diffusers, einops)

## Note

These notebooks were moved from the root directory as part of the project reorganization (Community Contribution).

For NAVSIM visualization tutorials, see the `tutorial/` directory in the root of the repository.