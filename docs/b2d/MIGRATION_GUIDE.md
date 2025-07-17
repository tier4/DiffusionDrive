# Script Migration Guide (Community Contribution)

> **Disclaimer:** This migration guide and the associated script reorganization were contributed by a community member and are not part of the original DiffusionDrive project. Use at your own discretion.

This guide helps developers transition from the old script structure to the new organized system.

## Overview of Changes

### Before (Root Directory)
```
DiffusionDrive/
├── train_bs32_20250617.sh
├── train_default_20250609.sh
├── train_bs128ep300_20250609.sh
├── eval.sh
└── ... (8+ scripts)
```

### After (Organized Structure)
```
DiffusionDrive/
└── scripts/
    ├── training/
    │   ├── train.sh
    │   └── batch_experiments.sh
    ├── evaluation/
    │   ├── eval.sh
    │   └── eval_all_checkpoints.sh
    └── utils/
        └── common.sh
```

## Migration Examples

### 1. Single Training Run

**Old way:**
```bash
# Had to create a new script for each configuration
./train_bs32_20250617.sh
```

**New way:**
```bash
# Use parameters with the unified script
./scripts/training/train.sh --name experiment1 --batch-size 32 --epochs 1000
```

### 2. Batch Size Experiments

**Old way:**
```bash
# Run multiple scripts manually
./train_bs32_20250617.sh
./train_bs64_20250617.sh
./train_bs128_20250617.sh
```

**New way:**
```bash
# Single command for batch experiments
./scripts/training/batch_experiments.sh --batch-sizes "32,64,128" --epochs 300
```

### 3. Evaluation

**Old way:**
```bash
# Hardcoded checkpoint paths in eval.sh
./eval.sh
```

**New way:**
```bash
# Pass checkpoint as parameter
./scripts/evaluation/eval.sh --checkpoint navsim_workspace/checkpoints/model.ckpt

# Or evaluate all checkpoints
./scripts/evaluation/eval_all_checkpoints.sh --dir navsim_workspace/
```

## Key Improvements

1. **No more date-based filenames** - Use git for version control
2. **Parameterized scripts** - One script handles all variations
3. **Shared utilities** - Common functions reduce duplication
4. **Better logging** - Automatic timestamped logs in `logs/` directory
5. **Environment checks** - Scripts validate required variables

## Environment Variables

Ensure these are set before running scripts:
```bash
export NAVSIM_DEVKIT_ROOT=/path/to/DiffusionDrive
export NAVSIM_EXP_ROOT=/path/to/experiments
```

## Legacy Scripts

Old scripts are preserved in `archive/legacy/` for reference. They should not be used for new experiments.

## Getting Help

All scripts support `--help`:
```bash
./scripts/training/train.sh --help
./scripts/evaluation/eval.sh --help
```

## Common Issues

### Issue: Script not found
**Solution:** Make sure scripts are executable:
```bash
chmod +x scripts/**/*.sh
```

### Issue: Missing environment variables
**Solution:** The new scripts will check and report missing variables with clear error messages.

### Issue: Different behavior from old scripts
**Solution:** Check the script parameters with `--help`. The new scripts may have different defaults or additional options.