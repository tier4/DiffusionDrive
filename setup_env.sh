#!/bin/bash
# DiffusionDrive Environment Setup Script
# Source this file to set up environment variables: source setup_env.sh

# Check if script is being sourced (not executed)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script should be sourced, not executed."
    echo "Usage: source setup_env.sh"
    exit 1
fi

# Set environment variables for DiffusionDrive
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# Update these paths according to your setup
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/your/navsim_workspace/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/your/navsim_workspace/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(pwd)}"  # Defaults to current directory
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/your/navsim_workspace/dataset}"

# Print current configuration
echo "DiffusionDrive environment variables set:"
echo "  NUPLAN_MAP_VERSION: $NUPLAN_MAP_VERSION"
echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
echo "  NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "  NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "  OPENSCENE_DATA_ROOT: $OPENSCENE_DATA_ROOT"

# Check if paths exist (warn if not)
if [[ ! -d "$NAVSIM_DEVKIT_ROOT" ]]; then
    echo "WARNING: NAVSIM_DEVKIT_ROOT does not exist: $NAVSIM_DEVKIT_ROOT"
fi

echo ""
echo "To make these permanent, add 'source $PWD/setup_env.sh' to your ~/.bashrc"