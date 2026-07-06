# Bench2Drive Official Train/Val Split

## File: bench2drive_base_train_val_split.json

**Source**: `/workspace/Bench2DriveZoo/data/splits/bench2drive_base_train_val_split.json`

**Origin**: Official Bench2Drive dataset train/validation split from the Bench2DriveZoo repository

**Purpose**: Provides the official train/validation split at the instance level for the Bench2Drive base dataset, ensuring reproducible results that are comparable to published benchmarks.

**Structure**:
```json
{
  "val": ["v1/ScenarioName_Town_Route_Weather", ...],
  "train": ["v1/ScenarioName_Town_Route_Weather", ...]  // Note: "train" key exists but is truncated in the file view
}
```

**Usage**: This split should be used instead of manually defined scenario lists to ensure consistency with the official Bench2Drive evaluation protocol.

**Note**: The `v1/` prefix in scenario names corresponds to the directory structure in the Bench2Drive dataset.