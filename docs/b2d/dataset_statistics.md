# Bench2Drive Dataset Statistics

**NOTE:** These statistics are based on the legacy v4 cache (downsample-first, ~200 samples/scenario).
With the new v6 true sliding window implementation, expect:
- **~5x more training samples** (~180,000 total vs 35,942)
- **~1000 samples per scenario** (vs current ~200)
- **95% frame utilization** (vs current 19%)

## Executive Summary: Key Differences

| Aspect | NAVSIM | Bench2Drive |
|--------|--------|-------------|
| **Training Samples** | 103,288 | 35,942 |
| **Scenarios** | 1,192 | 999 |
| **Avg Samples/Scenario** | 86.7 | 36.0 |
| **Sample Range/Scenario** | 1-1,080 | 3-212 |
| **Cameras** | 3 (front-view only) | 8 (360° coverage) |
| **Additional Sensors** | LiDAR | LiDAR + Radar |
| **Scenario Naming** | Timestamp-based (e.g., `2021.07.16.01.22.41_veh-14`) | Semantic names (e.g., `ParkingExit`, `AccidentTwoWays`) |
| **Feature File Size** | ~1MB | ~650KB |
| **Target File Size** | ~7KB | ~2KB |

## Overview

This document provides detailed statistics about the Bench2Drive-Base dataset as used in DiffusionDrive training.

## Dataset Versions

The Bench2Drive dataset has been processed into multiple cache versions for training:

- **v2**: Initial cache version
- **v3**: Updated cache format
- **v4**: Current production version (35,942 samples)
- **v5**: Latest experimental version

## Bench2Drive-Base Statistics (Cache v4)

### Overall Statistics

- **Total Training Samples**: 35,942
- **Total Scenarios**: 999 (out of 1000 in raw dataset)
- **Average Samples per Scenario**: 36.0
- **Median Samples per Scenario**: 28

### Sample Distribution

| Metric | Value |
|--------|-------|
| Minimum samples per scenario | 3 |
| Maximum samples per scenario | 212 |
| Median samples per scenario | 28 |
| Average samples per scenario | 36.0 |

### Top 10 Scenarios by Sample Count

| Scenario | Sample Count |
|----------|-------------|
| ParkingCrossingPedestrian_Town15_Route513_Weather19 | 212 |
| NonSignalizedJunctionLeftTurn_Town04_Route181_Weather15 | 193 |
| CrossingBicycleFlow_Town12_Route1078_Weather12 | 174 |
| HardBreakRoute_Town15_Route59_Weather7 | 147 |
| AccidentTwoWays_Town12_Route1124_Weather18 | 144 |
| EnterActorFlow_Town05_Route271_Weather11 | 139 |
| ParkedObstacle_Town13_Route555_Weather9 | 119 |
| PedestrianCrossing_Town12_Route864_Weather6 | 117 |
| BlockedIntersection_Town05_Route272_Weather12 | 117 |
| ParkingExit_Town13_Route677_Weather1 | 117 |

### Scenarios with Fewest Samples

| Scenario | Sample Count |
|----------|-------------|
| VanillaSignalizedTurnEncounterGreenLight_Town13_Route642_Weather18 | 3 |
| VanillaNonSignalizedTurnEncounterStopsign_Town15_Route509_Weather15 | 3 |
| VanillaNonSignalizedTurnEncounterStopsign_Town13_Route650_Weather0 | 5 |
| VanillaSignalizedTurnEncounterGreenLight_Town12_Route869_Weather11 | 5 |
| VehicleTurningRoute_Town15_Route1380_Weather19 | 5 |
| VanillaNonSignalizedTurnEncounterStopsign_Town12_Route1016_Weather2 | 5 |

## Cache Structure

Each cached sample is stored as a directory containing:

```text
scenario_name_xxxxx/
├── transfuser_feature.gz  # Compressed feature data (~650KB per file)
└── transfuser_target.gz   # Compressed target/label data (~2KB per file)
```

### Cache Paths

Standard cache locations:

```bash
# Training cache versions
/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache-v2/
/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache-v3/
/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache-v4/  # Current
/workspace/navsim_workspace/cache/Bench2Drive-Base-training_cache-v5/

# BEV cache versions
/workspace/navsim_workspace/cache/Bench2Drive-Base-full_bev_cache-v2/
/workspace/navsim_workspace/cache/Bench2Drive-Base-full_bev_cache-v3/
```

## Dataset Characteristics

### Scenario Types

The Bench2Drive dataset includes diverse driving scenarios:

1. **Parking Scenarios**: ParkingExit, ParkingCrossingPedestrian, ParkingCutIn
2. **Intersection Scenarios**: SignalizedJunction, NonSignalizedJunction, TJunction, BlockedIntersection
3. **Highway Scenarios**: HighwayExit, HighwayCutIn, MergerIntoSlowTraffic
4. **Pedestrian/Cyclist Scenarios**: PedestrianCrossing, CrossingBicycleFlow, DynamicObjectCrossing
5. **Emergency Scenarios**: Accident, YieldToEmergencyVehicle, ControlLoss
6. **Traffic Scenarios**: OppositeVehicleRunningRedLight, VehicleTurningRoute, HardBreakRoute
7. **Obstacle Scenarios**: ConstructionObstacle, ParkedObstacle, StaticCutIn

### Sample Variability

The large variation in samples per scenario (3-212) reflects:

- **Scenario Complexity**: More complex scenarios tend to have more training samples
- **Scenario Duration**: Longer scenarios generate more samples
- **Data Quality**: Some scenarios may have frames filtered out during preprocessing
- **Safety-Critical Events**: Scenarios with critical maneuvers may be sampled more densely

## Training Considerations

### Batch Size Recommendations

Based on the dataset size of 35,942 samples:

- **Small batch (32-64)**: ~560-1120 iterations per epoch
- **Medium batch (128-256)**: ~140-280 iterations per epoch
- **Large batch (512-1024)**: ~35-70 iterations per epoch

### Data Loading

With 999 scenarios and varying sample counts:

- Consider using weighted sampling if scenario balance is important
- The median of 28 samples per scenario suggests most scenarios are well-represented
- Scenarios with <10 samples may need special handling or augmentation

## Utility Scripts

To count samples in dataset caches:

**Bench2Drive:**

```bash
python3 /workspace/DiffusionDrive/count_bench2drive_samples.py
```

**NAVSIM:**

```bash
python3 /workspace/DiffusionDrive/count_navsim_samples.py
```

Modify the `cache_path` variable in the scripts to analyze different cache versions.

## NAVSIM Dataset Statistics

### Overall Statistics

- **Total Training Samples**: 103,288
- **Total Scenarios (Logs)**: 1,192
- **Average Samples per Scenario**: 86.7
- **Median Samples per Scenario**: 52

### Sample Distribution

| Metric | Value |
|--------|-------|
| Minimum samples per scenario | 1 |
| Maximum samples per scenario | 1,080 |
| Median samples per scenario | 52 |
| Average samples per scenario | 86.7 |

### Top 10 NAVSIM Scenarios by Sample Count

| Scenario | Sample Count |
|----------|-------------|
| 2021.07.16.01.22.41_veh-14_04315_07102 | 1,080 |
| 2021.06.23.15.56.12_veh-16_01308_04289 | 824 |
| 2021.06.23.14.58.13_veh-35_02037_04783 | 767 |
| 2021.07.16.18.19.22_veh-35_00869_03454 | 758 |
| 2021.07.16.18.49.56_veh-26_00833_03384 | 752 |
| 2021.06.23.15.18.10_veh-26_00165_02848 | 746 |
| 2021.06.23.20.00.35_veh-35_00960_03649 | 736 |
| 2021.07.09.17.06.37_veh-35_02609_05015 | 714 |
| 2021.07.09.23.23.48_veh-26_02228_04624 | 682 |
| 2021.06.23.14.54.32_veh-16_01187_03336 | 670 |

### NAVSIM Cache Structure

```
scenario_log_name/
└── sample_hash/
    ├── transfuser_feature.gz  # Compressed feature data (~1MB per file)
    └── transfuser_target.gz   # Compressed target/label data (~7KB per file)
```

### NAVSIM Cache Paths

```bash
# Training cache
/workspace/navsim_workspace/cache/training_cache/  # 103,288 samples

# Metadata files
/workspace/navsim_workspace/cache/navsim_scene_metadata_cache_trainval.pkl  # 2.39 MB
/workspace/navsim_workspace/cache/navsim_scene_metadata_cache_test.pkl      # 0.25 MB
```

## Comparison: Bench2Drive vs NAVSIM

| Metric | Bench2Drive-Base | NAVSIM |
|--------|-----------------|--------|
| **Total Scenarios** | 999 | 1,192 |
| **Total Training Samples** | 35,942 | 103,288 |
| **Average Samples/Scenario** | 36.0 | 86.7 |
| **Median Samples/Scenario** | 28 | 52 |
| **Min Samples/Scenario** | 3 | 1 |
| **Max Samples/Scenario** | 212 | 1,080 |
| **Scenario Types** | 30+ named types | Date-based logs |
| **Sensors** | 8 cameras, LiDAR, Radar | 3 cameras, LiDAR |
| **Feature File Size** | ~650KB | ~1MB |
| **Target File Size** | ~2KB | ~7KB |

### Key Differences

1. **Dataset Size**: NAVSIM has ~2.9x more training samples than Bench2Drive
2. **Sample Distribution**: NAVSIM has higher variance (1-1,080) vs Bench2Drive (3-212)
3. **Scenario Organization**:
   - Bench2Drive uses semantic scenario names (e.g., "ParkingExit", "AccidentTwoWays")
   - NAVSIM uses timestamp-based log names (e.g., "2021.07.16.01.22.41_veh-14")
4. **Data Density**: NAVSIM averages 86.7 samples per scenario vs 36.0 for Bench2Drive
5. **Feature Size**: NAVSIM features are ~50% larger (1MB vs 650KB)

## Notes

- The cache v4 is missing 1 scenario from the original 1000 (likely filtered during preprocessing)
- Each sample represents a single timestep suitable for training
- The preprocessing includes feature extraction, normalization, and compression
- BEV (Bird's Eye View) maps are cached separately and linked during training
