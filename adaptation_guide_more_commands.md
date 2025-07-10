# Guide: Adapting DiffusionDrive for More Driving Commands

## Overview

Currently, DiffusionDrive expects 4 driving commands (one-hot encoded):

- 0: LEFT (turns, lane changes, sharp curves)
- 1: STRAIGHT
- 2: RIGHT (turns, lane changes, sharp curves)
- 3: UNKNOWN (can be filtered)

To support more commands (e.g., 6, 8, or more), you need to modify several files.

## Required Modifications

### 1. **Configuration File** (`navsim/agents/diffusiondrive/transfuser_config.py`)

Add a new configuration parameter:

```python
@dataclass
class TransfuserConfig:
    # ... existing parameters ...
    
    # Add this new parameter
    num_driving_commands: int = 4  # Change this to your number of commands
```

### 2. **Feature Builder** (`navsim/agents/diffusiondrive/transfuser_features.py`)

No changes needed here! The feature builder already handles any size of driving command array.

### 3. **Model Architecture** (`navsim/agents/diffusiondrive/transfuser_model_v2.py`)

Update the status encoding layer to use the config parameter:

```python
# Line 41 - Change from:
self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

# To:
self._status_encoding = nn.Linear(config.num_driving_commands + 2 + 2, config.tf_d_model)
```

### 4. **Data Loading/Preprocessing**

Ensure your data loader provides one-hot encoded commands with the correct size:

```python
# Example conversion function
def convert_command_to_onehot(command_id: int, num_commands: int) -> np.ndarray:
    """Convert integer command to one-hot encoding."""
    onehot = np.zeros(num_commands, dtype=np.float32)
    if 0 <= command_id < num_commands:
        onehot[command_id] = 1.0
    return onehot
```

### 5. **Hydra Configuration** (`navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`)

Add the new parameter:

```yaml
num_driving_commands: 8  # or whatever number you need
```

## Example: Supporting 8 Commands

If you want to support 8 commands (e.g., for more granular control):

1. **Define your command mapping**:

```python
COMMAND_MAPPING = {
    0: "HARD_LEFT",      # Sharp left turn
    1: "LEFT",           # Normal left turn/lane change
    2: "SLIGHT_LEFT",    # Slight left adjustment
    3: "STRAIGHT",       # Continue straight
    4: "SLIGHT_RIGHT",   # Slight right adjustment
    5: "RIGHT",          # Normal right turn/lane change
    6: "HARD_RIGHT",     # Sharp right turn
    7: "UNKNOWN"         # Unknown/undefined
}
```

2. **Update config**:

```python
num_driving_commands: int = 8
```

3. **Update model**:

```python
self._status_encoding = nn.Linear(config.num_driving_commands + 2 + 2, config.tf_d_model)
```

4. **Update visualization** (optional):

```python
# In your notebook visualization
labels = ['HARD_LEFT', 'LEFT', 'SLIGHT_LEFT', 'STRAIGHT', 
          'SLIGHT_RIGHT', 'RIGHT', 'HARD_RIGHT', 'UNKNOWN', 
          'vel_x', 'vel_y', 'acc_x', 'acc_y']
```

## Testing Your Changes

1. **Verify data shape**:

```python
# In your notebook
print(f"Driving command shape: {features['status_feature'][:num_commands].shape}")
print(f"Expected: ({num_commands},)")
```

2. **Check model input**:

```python
# Add debug print in model forward pass
print(f"Status feature shape: {status_feature.shape}")
# Should be: (batch_size, num_commands + 4)
```

3. **Validate one-hot encoding**:

```python
driving_cmd = status_feature[:, :num_commands]
assert torch.allclose(driving_cmd.sum(dim=1), torch.ones(batch_size))
```

## Important Notes

1. **Backward Compatibility**: If you have pre-trained models with 4 commands, they won't work with the modified architecture.

2. **Data Consistency**: Ensure all your data uses the same number of commands consistently.

3. **Loss Functions**: The model doesn't directly predict commands, so no changes needed in loss computation.

4. **Evaluation**: If you have command-specific metrics, update them to handle the new commands.

## Alternative Approach: Continuous Commands

Instead of one-hot encoding, you could use continuous values:

```python
# Use 2D continuous command (angle, magnitude)
driving_command = np.array([angle, magnitude])  # 2D instead of one-hot

# Update model accordingly
self._status_encoding = nn.Linear(2 + 2 + 2, config.tf_d_model)  # 2 for command + 2 vel + 2 acc
```

This gives more flexibility but requires different data preprocessing.
