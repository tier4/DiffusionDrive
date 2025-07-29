# Bench2Drive Retraining Guide with Fixes

This guide explains how to retrain DiffusionDrive with the Bench2Drive fixes applied.

## Why Re-cache is Needed

The fixes we implemented change fundamental data processing:
- **Heading extraction**: Now uses ego bounding box rotation instead of `anno['theta']`
- **Trajectory calculation**: Uses world2ego matrices for proper coordinate transformation  
- **Agent detection**: Now detects both vehicles AND pedestrians with proper class mapping
- **BEV generation**: Renders vehicles (class 5) and pedestrians (class 6) correctly

The old caches contain incorrect data that would lead to poor training performance.

## Process Overview

1. **Clear old BEV cache** (contains incorrect dynamic objects)
2. **Clear old training/metric caches** 
3. **Regenerate BEV maps** with fixed agent detection
4. **Re-cache dataset** with all fixes applied
5. **Train** with clean, corrected data

## Step-by-Step Instructions

### Step 1: Clear Old Caches

```bash
# Remove old BEV cache (most important!)
rm -rf /workspace/navsim_workspace/cache/bev_cache_Bench2Drive-Base

# Remove old training cache
rm -rf /workspace/navsim_workspace/cache/training_cache_bench2drive

# Remove old metric cache (if exists)
rm -rf /workspace/navsim_workspace/cache/metric_cache_bench2drive
```

**Why**: The old BEV cache was generated before our fixes and doesn't contain vehicles/pedestrians properly rendered.

### Step 2: Generate New BEV Cache (Critical!)

Create a script to regenerate BEV maps with the fixes:

```python
# save as generate_bev_cache.py
"""Generate BEV cache with fixed agent detection."""
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from navsim.common.bench2drive_dataloader import Bench2DriveSceneLoader, Bench2DriveConfig

def generate_bev_cache(data_root, cache_dir, max_scenes=None):
    """Generate BEV cache with fixes applied."""
    # Get available scenarios
    scenarios = [d.name for d in data_root.iterdir() if d.is_dir()]
    
    config = Bench2DriveConfig(
        data_root=data_root,
        scenarios=scenarios,
        sampling_rate=5,
        num_frames=30,
    )
    
    print(f"Loading scenes from {data_root}...")
    loader = Bench2DriveSceneLoader(config)
    print(f"Found {len(loader.scene_tokens)} scenes")
    
    # Process scenes
    scenes_to_process = loader.scene_tokens[:max_scenes] if max_scenes else loader.scene_tokens
    
    for token in tqdm(scenes_to_process, desc="Generating BEV maps"):
        try:
            scene = loader.get_scene(token)
            scenario_name = scene.scene_info['scenario']
            
            # Create output directory
            scenario_cache_dir = cache_dir / scenario_name
            scenario_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate BEV for each frame
            for frame_idx in range(len(scene.frames)):
                frame_name = scene.frames[frame_idx].stem.split('.')[0]
                output_path = scenario_cache_dir / f"{frame_name}.npz"
                
                if output_path.exists():
                    continue  # Skip if already exists
                
                # Generate BEV with fixed agent detection
                bev_map = scene.get_bev_semantic_map(frame_idx)
                
                # Save as compressed numpy
                np.savez_compressed(
                    output_path,
                    front_bev=bev_map.numpy().astype(np.uint8)
                )
                
        except Exception as e:
            print(f"Error processing scene {token}: {e}")
            continue
    
    print(f"BEV cache generated at: {cache_dir}")

if __name__ == "__main__":
    data_root = Path("/workspace/Bench2Drive-Base")
    cache_dir = Path("/workspace/navsim_workspace/cache/bev_cache_Bench2Drive-Base")
    
    # For testing: limit to 1000 scenes
    # For full dataset: remove max_scenes parameter
    generate_bev_cache(data_root, cache_dir, max_scenes=1000)
```

Run the BEV generation:
```bash
python3 generate_bev_cache.py
```

**This step is crucial** - the new BEV maps will contain properly rendered vehicles and pedestrians.

### Step 3: Re-cache Training Dataset

```bash
python3 navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent_b2d \
    experiment_name=training_diffusiondrive_bench2drive_fixed \
    train_test_split=bench2drive \
    +split=train \
    cache.cache_path="/workspace/navsim_workspace/cache/training_cache_bench2drive_fixed" \
    worker.threads_per_worker=8
```

### Step 4: Re-cache Evaluation Dataset (Optional)

```bash
python3 navsim/planning/script/run_metric_caching.py \
    train_test_split=bench2drive \
    +split=val \
    cache.cache_path="/workspace/navsim_workspace/cache/metric_cache_bench2drive_fixed"
```

### Step 5: Train with Fixed Data

```bash
python3 navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent_b2d \
    experiment_name=training_diffusiondrive_bench2drive_fixed \
    train_test_split=bench2drive \
    +split=train \
    trainer.params.max_epochs=100 \
    cache_path="/workspace/navsim_workspace/cache/training_cache_bench2drive_fixed" \
    use_cache_without_dataset=True
```

## What's Different in the New Cache

### Before Fixes (Old Cache)
- ❌ Heading from incorrect `anno['theta']` 
- ❌ Trajectories not properly ego-relative
- ❌ Only vehicles detected, no pedestrians
- ❌ BEV maps missing dynamic objects

### After Fixes (New Cache)
- ✅ **Correct heading** from ego bounding box rotation
- ✅ **Proper ego-relative trajectories** using world2ego matrices
- ✅ **Vehicles AND pedestrians** detected with class mapping
- ✅ **BEV maps with dynamic objects** (vehicles class 5, pedestrians class 6)

## Expected Training Improvements

With the fixed data, you should see:
- **No NaN losses** (caused by incorrect coordinate transformations)
- **Better trajectory learning** (proper ego-relative coordinates)
- **Improved agent awareness** (model can see pedestrians)
- **More realistic driving behavior** (correct heading understanding)

## Important Notes

1. **Don't mix old and new caches** - Use either all old or all new
2. **BEV regeneration is critical** - The old BEV cache lacks dynamic objects
3. **Monitor training closely** - The model will learn different patterns with correct data
4. **Expect different convergence** - With proper data, training dynamics will change

## Validation

After caching, you can validate the fixes worked:

```bash
python3 scripts/validate_bench2drive_fixes.py
```

Expected validation results:
- ✅ Heading std > 0.1 rad (much better than near-zero before)
- ✅ Agent detection rate > 50% (including pedestrians)
- ✅ BEV maps contain vehicles and pedestrians

## Troubleshooting

**If training gets NaN losses:**
1. Check that you're using the new cache (not old one)
2. Verify BEV cache was regenerated with fixes
3. Run validation script to confirm data quality

**If no pedestrians are detected:**
1. Check that the class mapping fix is applied (`B2D_CLASS_TO_NAVSIM`)
2. Verify the mini dataset actually contains pedestrians (they're rare ~0.3%)

**If trajectories seem wrong:**
1. Confirm world2ego matrices are being used in trajectory calculation
2. Check that ego heading comes from bounding box, not `anno['theta']`