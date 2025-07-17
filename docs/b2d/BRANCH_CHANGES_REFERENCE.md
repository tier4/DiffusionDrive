# Branch Changes Reference: feature/training-callbacks-with-improvements vs tier4-main

## Overview

This document summarizes all changes made in the `feature/training-callbacks-with-improvements` branch compared to `tier4-main` branch.

## Commit History

```text
9654ecc Fix line endings in setup_env.sh
70f8d68 Merge branch 'tier4-main' into feature/training-callbacks-with-improvements
5839fd0 Update evaluation comparison notebook with new experiment results
e86c8f6 Fix PyTorch compatibility and improve environment setup
1e6f371 Merge remote-tracking branch 'upstream/main' into feature/training-callbacks-with-improvements
cf02031 Add simple TODO checklist for reviewing changes
1e6f540 Fix test assertions to match actual script output (Community Contribution)
6a4d5c8 Organize notebooks into structured directory (Community Contribution)
cda8cad Add comprehensive test suite for shell scripts (Community Contribution)
6484304 Reorganize scripts following software engineering best practices (Community Contribution)
fa588b8 Backup current state before script reorganization
5cbe65a Solve callback bug make it not working
c763205 add callbacks in training for more customizable saving options
274ebff revert back to original without try except
ee279e3 init
```

## Detailed Changes by Category

### 1. Training Infrastructure Enhancements

#### Modified Files

- `navsim/planning/script/run_training.py`
  - Added support for external callbacks from Hydra configuration
  - Changed callback instantiation to combine Hydra-configured and agent-specific callbacks
  - Key change:

    ```python
    # Old:
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    
    # New:
    callbacks = [instantiate(c) for c in cfg.trainer.callbacks] if cfg.trainer.get("callbacks") else []
    callbacks.extend(agent.get_training_callbacks())
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)
    ```

#### New Files

- `navsim/planning/script/config/training/default_training_w_callbacks.yaml`
  - New training configuration with ModelCheckpoint callback
  - Configures checkpointing every 10 epochs, saving top 5 models
  - Monitors `val/loss_epoch` metric

### 2. Configuration Updates

#### Modified Files

- `navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`
  - Added default paths:

    ```yaml
    bkb_path: "/workspace/DiffusionDrive/download/pytorch_model.bin"
    plan_anchor_path: "/workspace/DiffusionDrive/download/kmeans_navsim_traj_20.npy"
    ```

### 3. Script Reorganization (Community Contribution)

#### Structure Changes

- All scripts moved from root to organized directories:

  ```text
  scripts/
  ├── README.md
  ├── evaluation/
  │   ├── eval.sh
  │   ├── eval_all_checkpoints.sh
  │   ├── run_cv_pdm_score_evaluation.sh
  │   ├── run_ego_mlp_agent_pdm_score_evaluation.sh
  │   ├── run_human_agent_pdm_score_evaluation.sh
  │   ├── run_metric_caching.sh
  │   └── run_transfuser.sh
  ├── training/
  │   ├── batch_experiments.sh (NEW)
  │   ├── cache_dataset.sh (NEW)
  │   ├── run_ego_mlp_agent_training.sh
  │   ├── run_transfuser_training.sh
  │   └── train.sh (NEW)
  └── utils/
      ├── NOTICE.txt
      └── common.sh
  ```

#### Legacy Scripts Archived

- Moved to `archive/legacy/`:
  - `eval.sh`
  - `train_bs128ep300_20250609.sh`
  - `train_bs256ep1000_20250610.sh`
  - `train_bs256ep300_20250609.sh`
  - `train_bs32_20250617.sh`
  - `train_default_20250609.sh`
  - `train_default_20250618.sh`
  - `train_default_ep300_20250609.sh`

#### New Parameterized Scripts

- `scripts/training/batch_experiments.sh`: Batch training with different configurations
  - Supports: `--batch-sizes`, `--epochs`, `--base-name`, `--gpus` parameters
- `scripts/training/train.sh`: Main training script with CLI arguments
- `scripts/training/cache_dataset.sh`: Dataset caching script
- `scripts/utils/common.sh`: Shared utilities for all scripts

### 4. Environment Setup and Compatibility

#### New Files

- `setup_env.sh`: Environment setup script
- `test_cuda_torch.py`: CUDA/PyTorch testing utility

#### Fixes

- PyTorch 2.7.1 compatibility fix in WarmupCosLR scheduler (from merged tier4-main)
- Fixed line endings in setup_env.sh

### 5. Documentation Additions

#### New Documentation Files

- `CLAUDE.md`: AI assistant guidance for Claude Code
- `TODO.md`: Task tracking checklist
- `docs/ENVIRONMENT_SETUP.md`: Environment setup guide
- `docs/MIGRATION_GUIDE.md`: Migration guide for script changes
- `scripts/README.md`: Scripts documentation
- `notebooks/README.md`: Notebooks documentation
- `tests/scripts/README.md`: Test suite documentation

### 6. Testing Infrastructure

#### New Test Suite

```text
tests/scripts/
├── README.md
├── evaluation/
│   ├── test_eval.sh
│   └── test_eval_all_checkpoints.sh
├── github_workflow_example.yml
├── run_all_tests.sh
├── test_framework.sh
├── test_integration.sh
├── training/
│   ├── test_batch_experiments.sh
│   └── test_train.sh
└── utils/
    └── test_common.sh
```

### 7. Notebooks Organization

#### Structure

```text
notebooks/
├── README.md
└── evaluation/
    ├── compare_eval.ipynb (with new experiment results)
    └── visualization_eval.ipynb
```

### 8. Other Files Added

- `claude-20250625.txt`: Claude interaction log
- `config.txt`: Configuration notes
- `notice.md`: Notice file

## File Change Summary

### Added Files (A)

- CLAUDE.md
- TODO.md
- All files in archive/legacy/
- All files in scripts/ (new structure)
- All files in tests/scripts/
- All files in docs/ (ENVIRONMENT_SETUP.md, MIGRATION_GUIDE.md)
- notebooks/ structure
- setup_env.sh
- test_cuda_torch.py
- navsim/planning/script/config/training/default_training_w_callbacks.yaml

### Modified Files (M)

- CHANGELOG.md
- README.md
- download/*.sh (all download scripts)
- navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml
- navsim/planning/script/run_training.py
- Existing evaluation and training scripts (moved and updated)

## Key Benefits of These Changes

1. **Flexible Training**: Support for custom callbacks through Hydra configuration
2. **Better Organization**: Scripts organized by functionality with clear directory structure
3. **Improved Usability**: CLI arguments instead of hardcoded values in scripts
4. **Testing**: Comprehensive test suite for shell scripts
5. **Documentation**: Clear migration guides and documentation for all components
6. **Environment Setup**: Automated setup scripts and compatibility fixes

## Migration Notes

For users upgrading from tier4-main:

1. Update script paths - all scripts moved to `scripts/` directory
2. Use new parameterized scripts instead of legacy date-based scripts
3. Configure callbacks in Hydra config files for custom training behavior
4. Run tests with `tests/scripts/run_all_tests.sh` to verify setup
