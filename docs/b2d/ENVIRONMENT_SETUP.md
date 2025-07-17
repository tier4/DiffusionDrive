# Environment Setup for DiffusionDrive

This document describes how to set up the required environment variables for DiffusionDrive.

## Required Environment Variables

DiffusionDrive requires the following environment variables to be set:

- `NUPLAN_MAP_VERSION`: Version of NuPlan maps (default: "nuplan-maps-v1.0")
- `NUPLAN_MAPS_ROOT`: Path to the NuPlan maps directory
- `NAVSIM_EXP_ROOT`: Path for experiment outputs and caches
- `NAVSIM_DEVKIT_ROOT`: Path to the DiffusionDrive repository root
- `OPENSCENE_DATA_ROOT`: Path to the OpenScene dataset root

## Setup Instructions

### Option 1: Using the setup script (Recommended)

1. Navigate to the DiffusionDrive repository root
2. Source the setup script:
   ```bash
   source setup_env.sh
   ```

3. The script will use default values if variables are not already set. Update the paths in the script to match your system configuration.

4. To make the setup permanent, add the source command to your shell profile:
   ```bash
   echo "source /path/to/DiffusionDrive/setup_env.sh" >> ~/.bashrc
   ```

### Option 2: Manual setup

Add the following lines to your `~/.bashrc` or `~/.zshrc`:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/your/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/your/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/your/DiffusionDrive"
export OPENSCENE_DATA_ROOT="/path/to/your/navsim_workspace/dataset"
```

Then reload your shell configuration:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Verifying Setup

After setting up the environment, verify the variables are set correctly:

```bash
echo $NAVSIM_DEVKIT_ROOT
echo $NAVSIM_EXP_ROOT
```

These commands should output the paths you configured.