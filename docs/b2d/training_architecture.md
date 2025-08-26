# DiffusionDrive Training Architecture

This document describes the complete training architecture for both NavSim and Bench2Drive datasets.

## 1. NavSim Training (Original)

### Dependencies and Components

| Component | File/Class | Purpose |
|-----------|------------|---------|
| **Training Script** | `navsim/planning/script/run_training.py` | Standard NavSim training script |
| **Training Config** | `config/training/default_training.yaml` | Training hyperparameters (epochs, batch size, etc.) |
| **Agent Config** | `config/common/agent/diffusiondrive_agent.yaml` | Agent specification |
| **Agent Class** | `navsim.agents.diffusiondrive.transfuser_agent.TransfuserAgent` | Basic agent |
| **Model** | `navsim.agents.diffusiondrive.transfuser_model_v2.V2TransfuserModel` | Core diffusion model with hardcoded normalization |
| **Feature Builder** | `navsim.agents.diffusiondrive.transfuser_features.TransfuserFeatureBuilder` | Processes NavSim sensor data |
| **Target Builder** | `navsim.agents.diffusiondrive.transfuser_features.TransfuserTargetBuilder` | Creates training targets |

### Training Command

```bash
python3 $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=100
```

## 2. Bench2Drive Training (Current State - BROKEN)

### Current Issues

| Component | Current State | Problem |
|-----------|--------------|---------|
| **Training Script** | `run_bench2drive_training.py` | ✓ Works correctly |
| **Training Config** | `default_training.yaml` (hardcoded) | ⚠️ No B2D-specific config exists |
| **Agent Config** | `diffusiondrive_agent_b2d.yaml` | ❌ Points to ExtendedTransfuserAgent |
| **Agent Class** | `ExtendedTransfuserAgent` | ❌ Has normalization but uses NavSim features |
| **Model** | `V2TransfuserModelWrapper` | ✓ Has normalization support |
| **Feature Builder** | Wants `Bench2DriveFeatureBuilder` | ❌ Can't instantiate (missing config) |
| **Target Builder** | Wants `Bench2DriveTargetBuilder` | ❌ Can't instantiate (missing config) |

### Core Problem

- `ExtendedTransfuserAgent` was created for normalization support
- It inherits from `TransfuserAgent` which hardcodes **NavSim feature builders**
- The YAML config tries to override feature builders but Hydra can't instantiate them properly
- **Result**: Training fails with instantiation error

## 3. Bench2Drive Training (Fixed)

### Solution Architecture

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Training Script** | `run_bench2drive_training.py` | B2D-specific training script |
| **Training Config** | `default_training.yaml` | Same config, B2D overrides via agent |
| **Agent Config** | `diffusiondrive_agent_b2d.yaml` | Points to new Bench2DriveAgent |
| **Agent Class** | `Bench2DriveAgent` (new) | Combines normalization + B2D features |
| **Model** | `V2TransfuserModelWrapper` | Normalization for B2D data |
| **Feature Builder** | `Bench2DriveFeatureBuilder` | B2D-specific sensor processing |
| **Target Builder** | `Bench2DriveTargetBuilder` | B2D-specific targets |

### Class Inheritance Hierarchy

```
TransfuserAgent (Base NavSim agent)
    ↑
    ├── Uses: TransfuserFeatureBuilder (NavSim features)
    ├── Uses: V2TransfuserModel (no normalization)
    │
ExtendedTransfuserAgent (Adds normalization)
    ↑
    ├── Uses: V2TransfuserModelWrapper (with normalization)
    ├── Inherits: NavSim feature builders (PROBLEM!)
    │
Bench2DriveAgent (Complete B2D solution)
    ↑
    ├── Overrides: Bench2DriveFeatureBuilder (B2D features)
    ├── Overrides: Bench2DriveTargetBuilder (B2D targets)
    ├── Inherits: V2TransfuserModelWrapper (normalization)
```

### Training Command

```bash
./scripts/training/train.sh \
    --name "bench2drive_experiment" \
    --epochs 1000 \
    --batch-size 64 \
    --workers 32 \
    --gpus "0,1,2,3,4,5,6,7" \
    --agent "diffusiondrive_agent_b2d" \
    --dataset "bench2drive" \
    --lr 5e-5
```

## Summary of Changes Required

1. **Create `Bench2DriveAgent` class** (`navsim/agents/diffusiondrive/b2d_agent.py`)
   - Inherits from `ExtendedTransfuserAgent` for normalization
   - Overrides `get_feature_builders()` to return B2D feature builder
   - Overrides `get_target_builders()` to return B2D target builder

2. **Update `diffusiondrive_agent_b2d.yaml`**
   - Change `_target_` to point to `Bench2DriveAgent`
   - Remove feature_builders and target_builders sections (handled in Python)

3. **Keep existing files**
   - `ExtendedTransfuserAgent` - Still needed as base class
   - `ExtendedTransfuserConfig` - Provides dataset_type configuration
   - `V2TransfuserModelWrapper` - Provides normalization logic

## Key Differences

| Aspect | NavSim | Bench2Drive |
|--------|--------|-------------|
| **Agent** | `TransfuserAgent` | `Bench2DriveAgent` |
| **Features** | NavSim coordinate system | CARLA coordinate system |
| **Normalization** | Hardcoded in model | Flexible via config |
| **BEV Generation** | From ground truth | From HD maps |
| **Cache Required** | Optional | Mandatory |
