# Bench2Drive Experiment Versions

Below is the explanation of the different versions of the Bench2Drive experiments on what are their differences.

- `v1`: Initial version of the Bench2Drive experiments
- `v2`: Fix the `v1` which change coordinate system from Carla's left handed to NVSIM's right handed.
- `v3`: Fix agent distance filtering - corrected radius/diameter confusion (2025-01-16)
  - **Issue**: Agent filtering was using 85m as radius instead of diameter
  - **Fix**: Changed filtering from `distance > 85m` to `distance > 42.5m` (85m diameter / 2)
  - **Impact**: Removes ~60% of agents that were outside the LiDAR visible range
  - **Result**: Training and validation losses should now both converge properly
- `v4`: Remove fake data and fix ground truth alignment (2025-01-18)
  - **Issue**: Trajectory padding with fake data and wrong GT alignment for 10Hz evaluation
  - **Fix**: Skip incomplete samples, add frame_stride logic, raise errors for missing sensors
  - **Impact**: No more fake/padded data in training, correct 0.5s interval GT for all sampling rates
  - **Result**: Model trains on real data only, proper evaluation at both 2Hz and 10Hz
- `v5`: Segmentation-based BEV generation (2025-10-17)
  - **Cache Location**: `/workspace/navsim_workspace/cache/Bench2Drive-Base-full_bev_cache-v3/`
  - **Approach**: Generate BEV maps using filled road surfaces instead of just lane lines
  - **Method**: Draw lanes with 5m width and apply morphological operations to connect road segments
  - **Goal**: Test if denser BEV representation improves model performance
- `v6`: True sliding window sampling (2025-10-23)
  - **Cache Location**: `/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache-v6/`
  - **Approach**: Slide through ALL 10Hz frames instead of downsampling first
  - **Method**: Each scene starts at consecutive 10Hz frames with on-demand 2Hz downsampling
  - **Impact**: ~5x more training samples (~1000 per scenario vs ~200 in v4)
  - **Data Utilization**: 95% of frames used (vs 19% in v4)
  - **Backward Compatibility**: Use `--use-hardcoded-config` flag for v4 legacy mode
  - **Note**: BEV cache remains the same, only training cache changes
