# Bench2Drive Experiment Versions

Below is the explanation of the different versions of the Bench2Drive experiments on what are their differences.

- `v1`: Initial version of the Bench2Drive experiments
- `v2`: Fix the `v1` which change coordinate system from Carla's left handed to NVSIM's right handed.
- `v3`: Fix agent distance filtering - corrected radius/diameter confusion (2025-01-16)
  - **Issue**: Agent filtering was using 85m as radius instead of diameter
  - **Fix**: Changed filtering from `distance > 85m` to `distance > 42.5m` (85m diameter / 2)
  - **Impact**: Removes ~60% of agents that were outside the LiDAR visible range
  - **Result**: Training and validation losses should now both converge properly
