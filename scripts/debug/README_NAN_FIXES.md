# NaN Loss Debugging and Fixes for Bench2Drive Training

This directory contains documentation for resolving NaN loss issues when training DiffusionDrive on the Bench2Drive dataset.

## Script Organization

The debugging tools have been reorganized for better project structure:
- **Data Validation Tools**: `navsim/planning/utils/data_validation/`
- **Debugging Utilities**: `navsim/planning/utils/debugging/`
- **Data Processing Scripts**: `scripts/data_processing/`

For convenience, use the wrapper script:
```bash
./scripts/debug/run_validation.sh {validate|check-fixes|test-fixes|monitor|regenerate-cache}
```

## Problem Summary

When switching from NavSim to Bench2Drive dataset, training encounters NaN losses due to:
1. **Data normalization mismatches** - Camera images not normalized to [0,1]
2. **Different LiDAR processing** - Using log normalization instead of simple clipping
3. **BEV semantic map issues** - Float values instead of integer class labels
4. **Missing data validation** - No checks for corrupted cache entries

## Applied Fixes

### 1. Camera Normalization (transfuser_features_b2d.py)
```python
# Added normalization to match NavSim
stitched = stitched / 255.0  # Convert [0,255] to [0,1]
```

### 2. LiDAR Normalization (transfuser_features_b2d.py)
```python
# Changed from log normalization to NavSim's approach
hist = np.clip(hist, 0, hist_max_per_pixel)
hist = hist / hist_max_per_pixel  # Normalize to [0,1]
```

### 3. BEV Semantic Maps (bev_semantic_utils.py)
```python
# Fixed to return integer labels
bev_map = np.zeros((bev_height, bev_width), dtype=np.uint8)
return bev_map.astype(np.uint8)
```

### 4. Training Robustness
- Added autograd anomaly detection
- Added gradient clipping (max_norm=1.0)
- Added feature validation in data pipeline

## Step-by-Step Solution

### Step 1: Validate Current Cache
Check if your existing cache has issues:
```bash
python3 navsim/planning/utils/data_validation/validate_b2d_data.py \
    --cache-path /path/to/cache/bench2drive_mini \
    --use-cache-only \
    --num-batches 10

# Or use the wrapper script:
./scripts/debug/run_validation.sh validate \
    --cache-path /path/to/cache/bench2drive_mini \
    --use-cache-only \
    --num-batches 10
```

### Step 2: Clear and Regenerate Cache
**IMPORTANT**: You must regenerate the cache with the fixed feature builders!

```bash
# Option 1: Manual clearing and regeneration
rm -rf /path/to/cache/bench2drive_mini

# Then run normal caching command with fixed code

# Option 2: Use the safe regeneration script
python3 scripts/data_processing/regenerate_b2d_cache_safe.py \
    --data-root /path/to/bench2drive/data \
    --cache-path /path/to/cache/bench2drive_mini \
    --scenarios ConstructionObstacle \
    --force \
    --backup-old \
    --show-stats

# Or use the wrapper:
./scripts/debug/run_validation.sh regenerate-cache \
    --data-root /path/to/bench2drive/data \
    --cache-path /path/to/cache/bench2drive_mini \
    --scenarios ConstructionObstacle \
    --force \
    --backup-old \
    --show-stats
```

### Step 3: Test Fixes Before Training
Run a quick test to ensure no NaN:
```bash
python3 navsim/planning/utils/debugging/test_nan_fixes.py

# Or use the wrapper:
./scripts/debug/run_validation.sh test-fixes
```

### Step 4: Start Training with Safe Parameters
```bash
# Modify your training script to use lower learning rate
./train_bench2drive_full.sh \
    --lr 1e-5  # Start with very low LR
```

### Step 5: Monitor Training
In a separate terminal, monitor for NaN issues:
```bash
python3 navsim/planning/utils/debugging/monitor_training_nan.py \
    --log-dir /path/to/experiment/logs \
    --interval 30

# Or use the wrapper:
./scripts/debug/run_validation.sh monitor \
    --log-dir /path/to/experiment/logs \
    --interval 30
```

## Debugging Tools

### 1. validate_b2d_data.py (navsim/planning/utils/data_validation/)
- Checks data ranges and statistics
- Identifies normalization issues
- Compares with expected NavSim ranges

### 2. test_nan_fixes.py (navsim/planning/utils/debugging/)
- Tests single batch through model
- Verifies forward pass and loss computation
- Quick sanity check before full training

### 3. regenerate_b2d_cache_safe.py (scripts/data_processing/)
- Safely regenerates cache with validation
- Backs up old cache
- Shows data statistics

### 4. monitor_training_nan.py (navsim/planning/utils/debugging/)
- Real-time training monitoring
- Detects NaN/Inf in losses
- Saves diagnostic reports

### 5. check_nan_fixes_status.py (navsim/planning/utils/data_validation/)
- Verifies all fixes are properly applied
- Run with: `./scripts/debug/run_validation.sh check-fixes`

## Common Issues and Solutions

### Issue: NaN still occurs after fixes
1. **Check cache was regenerated**: Old cache still has unnormalized data
2. **Lower learning rate further**: Try 5e-6 or 1e-6
3. **Check GPU memory**: OOM can cause NaN
4. **Validate BEV generation**: Ensure BEV maps are generated correctly

### Issue: Very slow training with anomaly detection
- Anomaly detection is for debugging only
- Disable after finding the issue:
  ```python
  # Comment out in run_bench2drive_training.py
  # torch.autograd.set_detect_anomaly(True)
  ```

### Issue: Gradient clipping makes training unstable
- Adjust clipping threshold in agent_lightning_module.py:
  ```python
  nn_utils.clip_grad_norm_(self.parameters(), max_norm=5.0)  # Try larger values
  ```

## Verification Checklist

- [ ] Camera images normalized to [0,1]
- [ ] LiDAR histograms in [0,1] range
- [ ] BEV maps have integer values [0,6]
- [ ] Cache regenerated with fixed code
- [ ] Learning rate reduced (≤ 4e-5)
- [ ] Gradient clipping enabled
- [ ] Monitoring script running

## Emergency Recovery

If training still fails:
1. **Enable maximum debugging**:
   ```python
   torch.autograd.set_detect_anomaly(True)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

2. **Use CPU for debugging**:
   ```bash
   CUDA_VISIBLE_DEVICES="" python3 train.py  # Force CPU
   ```

3. **Binary search the issue**:
   - Train on single batch repeatedly
   - Disable model components one by one
   - Check each loss term separately

## Contact

If issues persist after trying all fixes, gather:
1. Full error traceback
2. Output from validation script
3. First few training iterations log
4. Model configuration

Then create an issue with this information.