# Driving Command Analysis in DiffusionDrive/NAVSIM

## Overview

This document contains the analysis of driving command format in DiffusionDrive (NAVSIM) dataset, including the scripts used to inspect the data and the findings.

## Key Findings

### 1. Command Format

According to the official NAVSIM documentation, driving commands are discrete values indicating whether the intended route is towards the left, straight or right direction. There is also a fourth command representing 'unknown', which can be used to filter out data during training.

**Important Notes from NAVSIM docs:**
- The driving command is based solely on the desired route
- It does NOT entangle information regarding obstacles and traffic signs (unlike prior benchmarks such as nuScenes)
- Left and right commands cover turns, lane changes and sharp curves
- The 'unknown' command can be used to filter out data during training

**Command Values:**
- `0` = LEFT (turns, lane changes left, sharp left curves)
- `1` = STRAIGHT (continue straight)
- `2` = RIGHT (turns, lane changes right, sharp right curves)  
- `3` = UNKNOWN (can be filtered during training)

### 2. Data Structure

- **Type**: Single integer value (0, 1, 2, or 3)
- **Location in code**: `navsim/common/dataclasses.py`, line 138
- **Access path**: `agent_input.ego_statuses[-1].driving_command`
- **Important**: While stored as an integer in the data structure, **the model expects one-hot encoded vectors**. The DiffusionDrive model's status encoding layer (`transfuser_model_v2.py`) uses `nn.Linear(config.num_driving_commands + 2 + 2, config.tf_d_model)` which requires one-hot encoded input. See `adaptation_guide_more_commands.md` for conversion details.

### 3. Usage in Model

The driving command is likely converted to a one-hot encoding before being concatenated with velocity and acceleration. The exact implementation would need verification in the specific model code:

```python
# Possible implementation (needs verification):
driving_cmd = agent_input.ego_statuses[-1].driving_command  # Integer 0-3
driving_cmd_one_hot = F.one_hot(torch.tensor(driving_cmd), num_classes=4).float()  # Convert to 4D one-hot

features["status_feature"] = torch.concatenate([
    driving_cmd_one_hot,  # 4D one-hot encoded
    torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),     # 2D
    torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32), # 2D
])
```

## Analysis Scripts

### Script 1: Basic Pickle File Inspection

```python
import pickle
import numpy as np

# Load a sample pickle file
with open('/workspace/navsim_workspace/dataset/navsim_logs/test/2021.05.25.14.16.10_veh-35_00083_00485.pkl', 'rb') as f:
    data = pickle.load(f)

# Print the structure
print('Type:', type(data))
print('Length:', len(data) if hasattr(data, '__len__') else 'N/A')

# If it's a list, show first frame
if isinstance(data, list) and len(data) > 0:
    print('\nFirst frame keys:', list(data[0].keys()))
    if 'driving_command' in data[0]:
        print('\nDriving command in first few frames:')
        for i in range(min(5, len(data))):
            print(f'  Frame {i}: {data[i]["driving_command"]}')
```

### Script 2: Analyze Command Encoding

```python
import pickle
import numpy as np

# Load a sample pickle file
with open('/workspace/navsim_workspace/dataset/navsim_logs/test/2021.05.25.14.16.10_veh-35_00083_00485.pkl', 'rb') as f:
    data = pickle.load(f)

# Analyze driving commands
commands = set()
for frame in data[:100]:  # Check first 100 frames
    cmd = tuple(frame['driving_command'])
    commands.add(cmd)

print('Unique driving commands found:')
for cmd in sorted(commands):
    print(f'  {list(cmd)}')

# Check if it's one-hot encoded
print('\nChecking pattern (one-hot encoding?):')
for cmd in sorted(commands):
    sum_val = sum(cmd)
    print(f'  {list(cmd)} -> sum = {sum_val}')

# Map to simple commands
print('\nMapping to simple commands:')
for cmd in sorted(commands):
    if cmd == 0:
        print(f'  {cmd} -> LEFT (turns, lane changes, sharp curves)')
    elif cmd == 1:
        print(f'  {cmd} -> STRAIGHT')
    elif cmd == 2:
        print(f'  {cmd} -> RIGHT (turns, lane changes, sharp curves)')
    elif cmd == 3:
        print(f'  {cmd} -> UNKNOWN (can be filtered)')
```

### Script 3: Find All Command Types Across Dataset

```python
import pickle
import os
import glob

# Sample multiple files to find all command types
log_dir = '/workspace/navsim_workspace/dataset/navsim_logs/trainval/'
files = glob.glob(os.path.join(log_dir, '*.pkl'))[:20]  # Check first 20 files

all_commands = {}

for file_path in files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Collect all unique commands from this file
    file_commands = set()
    for frame in data:
        cmd = tuple(frame['driving_command'])
        file_commands.add(cmd)
    
    # Store file name if it has interesting commands
    for cmd in file_commands:
        if cmd not in all_commands:
            all_commands[cmd] = []
        all_commands[cmd].append(os.path.basename(file_path))

print('All unique driving commands found:')
for cmd, files in sorted(all_commands.items()):
    print(f'\n{list(cmd)} found in {len(files)} files')
    print(f'  Example: {files[0]}')

print('\n\nCommand encoding summary:')
print('NAVSIM uses discrete integer values for driving commands:')
print('  0 = LEFT (turns, lane changes, sharp curves)')
print('  1 = STRAIGHT') 
print('  2 = RIGHT (turns, lane changes, sharp curves)')
print('  3 = UNKNOWN (can be filtered during training)')
```

### Script 4: Analyze Command Distribution

```python
import pickle
import numpy as np
from collections import Counter

# Load multiple files and analyze command distribution
files = [
    '/workspace/navsim_workspace/dataset/navsim_logs/trainval/2021.06.09.12.39.51_veh-26_00055_00360.pkl',
    '/workspace/navsim_workspace/dataset/navsim_logs/trainval/2021.06.09.14.50.36_veh-26_00063_00350.pkl',
    '/workspace/navsim_workspace/dataset/navsim_logs/test/2021.09.09.19.10.24_veh-39_00148_00372.pkl'
]

command_counter = Counter()
total_frames = 0

for file_path in files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    for frame in data:
        cmd = tuple(frame['driving_command'])
        command_counter[cmd] += 1
        total_frames += 1

print('Command distribution:')
for cmd, count in command_counter.most_common():
    percentage = (count / total_frames) * 100
    print(f'  {list(cmd)}: {count} frames ({percentage:.1f}%)')
```

## Mapping for Bench2Drive Integration

When converting from CARLA/Bench2Drive commands to NAVSIM format:

```python
def carla_to_navsim_command(carla_command: str) -> int:
    """Convert CARLA command string to NAVSIM integer command"""
    
    # Define mapping based on NAVSIM documentation
    command_map = {
        # Left commands (turns, lane changes, sharp curves)
        'CHANGELANELEFT': 0,
        'TURNLEFT': 0,
        'LEFT': 0,
        
        # Straight commands
        'STRAIGHT': 1,
        'LANEFOLLOW': 1,
        
        # Right commands (turns, lane changes, sharp curves)
        'CHANGELANERIGHT': 2,
        'TURNRIGHT': 2,
        'RIGHT': 2,
        
        # Unknown (can be filtered during training)
        'STOP': 3,  # Could be mapped to unknown or handled differently
        'UNKNOWN': 3
    }
    
    # Default to straight if command not found
    return command_map.get(carla_command.upper(), 1)
```

## Dataset File Structure

Each pickle file in NAVSIM contains a list of frames, where each frame is a dictionary with keys:

- `token`: Unique frame identifier
- `driving_command`: Integer command (0=left, 1=straight, 2=right, 3=unknown)
- `ego2global_translation`: Ego vehicle position
- `ego2global_rotation`: Ego vehicle rotation (quaternion)
- `ego_dynamic_state`: [vx, vy, ax, ay] - velocity and acceleration
- `cams`: Camera data dictionary
- `lidar_path`: Path to LiDAR data
- `anns`: Annotations (bounding boxes, etc.)
- And many more...

## Notes

1. Commands are discrete integers (0-3) based solely on the desired route
2. Commands do NOT include information about obstacles or traffic signs
3. Most frames in highway/straight road scenarios will have command value 1 (straight)
4. Commands 0 and 2 cover all left/right maneuvers including turns, lane changes, and sharp curves
5. Command 3 (unknown) can be used to filter out ambiguous or problematic data during training
6. This design decouples route intention from reactive behavior (unlike nuScenes)

## Usage Example

To use this in your code:

```python
# Access driving command from agent input
driving_cmd = agent_input.ego_statuses[-1].driving_command
print(f"Current command: {driving_cmd}")

# Interpret the command
if driving_cmd == 0:
    action = "Left (turn/lane change/sharp curve)"
elif driving_cmd == 1:
    action = "Straight"
elif driving_cmd == 2:
    action = "Right (turn/lane change/sharp curve)"
elif driving_cmd == 3:
    action = "Unknown (can be filtered)"
else:
    action = "Invalid command"
print(f"Action: {action}")

# If the model expects one-hot encoding:
driving_cmd_one_hot = F.one_hot(torch.tensor(driving_cmd), num_classes=4)
```
