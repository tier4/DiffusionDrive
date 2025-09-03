#!/bin/bash
# Docker run script for DiffusionDrive
#
# Usage: ./run.sh
#
# Environment variables (optional):
#   WORKSPACE_DIR - Path to workspace directory (default: current directory)
#   DATA_DIR      - Path to data directory (default: /data)
#   CONTAINER_NAME - Name for the container (default: diffusiondrive)

# Enable all GPUs
# Use host's IPC namespace (needed for PyTorch multiprocessing)
# Remove memory lock limits (required for some GPU operations)
# Container name (can be customized via env var)
# Mount workspace directory
# Mount data directory
# Mount SSH keys as read-only for git operations
docker run -it \
    --gpus all \
    --ipc host \
    --net carla_net \
    --ulimit memlock=-1 \
    --name "${CONTAINER_NAME:-diffusiondrive}" \
    -v "${WORKSPACE_DIR:-$(pwd)}:/workspace" \
    -v "${DATA_DIR:-/data}:/data" \
    -v "${HOME}/.ssh:/home/user/.ssh:ro" \
    diffusiondrive:latest

