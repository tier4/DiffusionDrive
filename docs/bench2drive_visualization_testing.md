# Bench2Drive Visualization and Testing Guide

This guide explains how to visualize cached Bench2Drive data and test the training dataloader functionality to ensure your integration is working correctly.

## Overview

After implementing Bench2Drive support in DiffusionDrive, it's crucial to verify that:
1. Data caching works correctly and preserves all necessary information
2. The training dataloader provides valid batches for model training
3. Features and targets are properly formatted and aligned

This guide covers two main tools:
- **Visualization Notebook**: Interactive exploration of cached data
- **DataLoader Test Script**: Automated validation of training pipeline

## Prerequisites

1. Environment setup:
   ```bash
   conda activate navsim
   export NAVSIM_DEVKIT_ROOT=/workspace/DiffusionDrive
   export NAVSIM_EXP_ROOT=/workspace/navsim_workspace/exp
   export BENCH2DRIVE_ROOT=/workspace/Bench2Drive-Base
   ```

2. Cached data (run caching first if not done):
   ```bash
   python navsim/planning/script/run_dataset_caching.py \
       agent=diffusiondrive_agent_b2d \
       train_test_split=bench2drive \
       experiment_name=bench2drive_caching \
       split=train
   ```

## Visualization Notebook

### Purpose
The notebook `notebooks/bench2drive_cache_visualization.ipynb` provides interactive visualizations to verify data integrity throughout the caching and loading pipeline.

### Key Sections

1. **Cache Structure Exploration**
   - Lists available cached files
   - Shows metadata about the dataset
   - Helps identify any missing or corrupted cache files

2. **Feature Visualization**
   - **Camera Features**: Displays stitched front camera views
   - **LiDAR BEV**: Shows bird's eye view point cloud histograms
   - **Status Features**: Plots driving commands and velocity profiles

3. **Target Visualization**
   - **Trajectories**: Shows future waypoints and paths
   - **Agent Detection**: Visualizes bounding boxes on BEV
   - **Semantic Maps**: Displays road/lane segmentation

4. **Original vs Cached Comparison**
   - Side-by-side comparisons to ensure caching preserves data fidelity
   - Validates coordinate transformations and command mappings
   - Checks temporal alignment between different modalities

5. **DataLoader Integration**
   - Tests batch loading from the dataset
   - Validates tensor shapes and data types
   - Ensures compatibility with training pipeline

### Running the Notebook

```bash
cd /workspace/DiffusionDrive
jupyter notebook notebooks/bench2drive_cache_visualization.ipynb
```

### What to Look For

✅ **Good Signs**:
- Camera images are properly stitched without artifacts
- LiDAR BEV shows clear vehicle and road boundaries
- Driving commands match expected behavior (turns, straight driving)
- Trajectories are smooth and physically plausible
- Agent boxes align with visible vehicles

❌ **Potential Issues**:
- Black or corrupted images
- Empty LiDAR data
- All-zero command vectors
- Discontinuous trajectories
- Misaligned agent detections

## DataLoader Test Script

### Purpose
The script `test_bench2drive_dataloader.py` provides automated testing of the training dataloader with comprehensive validation and performance metrics.

### Running the Test

Basic usage:
```bash
python test_bench2drive_dataloader.py
```

With custom parameters:
```bash
python test_bench2drive_dataloader.py \
    --data-root /workspace/Bench2Drive-Base \
    --cache-root /workspace/navsim_workspace/exp/bench2drive_training_cache \
    --scenarios ConstructionObstacle ParkingExit \
    --batch-size 32 \
    --num-workers 4 \
    --max-batches 50
```

### Command Line Options

- `--data-root`: Path to Bench2Drive dataset (default: `/workspace/Bench2Drive-Base`)
- `--cache-root`: Path to cache directory (default: `/workspace/navsim_workspace/exp/bench2drive_training_cache`)
- `--scenarios`: List of scenarios to test (default: `["ConstructionObstacle"]`)
- `--batch-size`: Batch size for DataLoader (default: 32)
- `--num-workers`: Number of worker processes (default: 4)
- `--max-batches`: Maximum batches to test, None for all (default: 10)
- `--no-cache`: Test without using cache (forces recomputation)

### Test Report Sections

The script generates a comprehensive report with:

1. **Dataset Statistics**
   - Total samples available
   - Number of scenes per scenario
   - Batch configuration

2. **Performance Metrics**
   - Average batch load time
   - Data throughput (batches/second)
   - Memory usage statistics

3. **Command Distribution**
   - Percentage of LEFT, STRAIGHT, RIGHT, UNKNOWN commands
   - Helps identify dataset imbalances

4. **Speed Statistics**
   - Min/max/average ego vehicle speeds
   - Useful for identifying static scenes

5. **Agent Detection Statistics**
   - Average number of agents per frame
   - Detection range validation

6. **Validation Results**
   - Shape validation for all tensors
   - Value range checks (normalized images, one-hot encoding)
   - Data integrity verification

### Expected Output

```
=== Creating Dataset ===
✓ Scene loader created with 1500 scenes
✓ Dataset created with 1500 samples

=== Creating DataLoader ===
✓ DataLoader created:
  - Batch size: 32
  - Num workers: 4
  - Shuffle: True
  - Pin memory: True

=== Testing DataLoader ===
Loading batches: 100%|████████| 10/10 [00:45<00:00, 4.52s/it]
✓ Loaded 10 batches in 45.23s

============================================================
BENCH2DRIVE DATALOADER TEST REPORT
============================================================

1. DATASET STATISTICS:
   Total samples: 1500
   Total batches tested: 10
   Batch size: 32

2. PERFORMANCE METRICS:
   Average batch load time: 4.523s
   Min batch load time: 3.891s
   Max batch load time: 5.234s
   Total time: 45.23s
   Throughput: 0.22 batches/s

3. COMMAND DISTRIBUTION:
   LEFT: 85 (26.6%)
   RIGHT: 92 (28.8%)
   STRAIGHT: 143 (44.7%)

4. SPEED STATISTICS:
   Min speed: 0.00 m/s
   Max speed: 15.23 m/s
   Average speed: 7.84 m/s

5. AGENT DETECTION STATISTICS:
   Average agents per frame: 4.2
   Min agents: 0
   Max agents: 18

6. VALIDATION RESULTS:
   ✅ All batches passed validation!

============================================================
TEST COMPLETE
============================================================
```

## Common Issues and Solutions

### 1. Empty Cache Files
**Symptom**: Notebook shows no cached data or test script fails to load
**Solution**: 
- Verify caching completed successfully
- Check disk space and permissions
- Re-run caching with `force_cache_computation=True`

### 2. Shape Mismatch Errors
**Symptom**: Validation errors about tensor shapes
**Solution**:
- Verify feature/target builders match model expectations
- Check temporal downsampling is correctly applied
- Ensure camera stitching produces correct dimensions

### 3. Command Distribution Imbalance
**Symptom**: Test shows 90%+ of one command type
**Solution**:
- This might be normal for some scenarios (e.g., highway driving)
- Consider using multiple diverse scenarios
- Check command mapping logic if all commands are UNKNOWN

### 4. Slow Performance
**Symptom**: Very high batch load times (>10s)
**Solution**:
- Ensure cache is being used (not recomputing features)
- Reduce number of workers if CPU-bound
- Check disk I/O performance
- Consider SSD storage for cache

### 5. Memory Issues
**Symptom**: Out of memory errors or system slowdown
**Solution**:
- Reduce batch size
- Limit number of workers
- Enable pin_memory only if GPU available
- Monitor with `nvidia-smi` and `htop`

## Integration with Training

Once visualization and testing confirm correct behavior:

1. **Start Training**:
   ```bash
   ./scripts/train_bench2drive.sh
   ```

2. **Monitor Training**:
   - Check loss curves converge
   - Validate on held-out scenarios
   - Compare with baseline NavSim performance

3. **Debug Training Issues**:
   - Use visualization notebook to inspect problematic batches
   - Run dataloader test with training scenarios
   - Enable debug logging in training script

## Advanced Usage

### Custom Visualizations

Add to the notebook:
```python
def visualize_trajectory_distribution(cached_data):
    """Analyze trajectory patterns in the dataset."""
    trajectories = []
    for idx in range(len(cached_data)):
        features, targets = cached_data[idx]
        traj = targets['trajectory'].numpy()
        trajectories.append(traj)
    
    trajectories = np.array(trajectories)
    
    # Plot trajectory endpoints
    plt.figure(figsize=(10, 10))
    plt.scatter(trajectories[:, -1, 0], trajectories[:, -1, 1], 
                alpha=0.5, s=1)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Endpoint Distribution')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
```

### Performance Profiling

Add to test script:
```python
import cProfile
import pstats

def profile_dataloader(dataloader, num_batches=5):
    """Profile dataloader performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    for i, (features, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

### Scenario-Specific Analysis

Test specific challenging scenarios:
```bash
# Test urban scenarios with many agents
python test_bench2drive_dataloader.py \
    --scenarios CrossingPedestrian BusyIntersection \
    --max-batches 100

# Test highway scenarios
python test_bench2drive_dataloader.py \
    --scenarios HighwayMerge HighwayExit \
    --max-batches 100
```

## Conclusion

Regular visualization and testing ensures:
- Data integrity throughout the pipeline
- Training stability and convergence
- Early detection of integration issues
- Performance optimization opportunities

Run these tools:
- After initial implementation
- When changing data processing code
- Before starting long training runs
- When debugging training issues