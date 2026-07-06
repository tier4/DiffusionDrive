# DiffusionDrive Setup Findings

## Overview

This document summarizes the findings from setting up a two-container architecture for CARLA simulation with DiffusionDrive, including container setup, networking requirements, and next steps.

## Container Architecture

### 1. CARLA Simulator Container

**Script**: `@Bench2DriveZoo/run_carla.sh`

```bash
docker run -d --rm --gpus all --ipc host --net carla_net \
  --name carla carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
```

**Key Configuration**:

- Runs CARLA 0.9.15 in headless mode (`-RenderOffScreen`)
- Uses port 2000 for RPC communication
- Connected to `carla_net` network
- GPU acceleration enabled

### 2. Python Client Container  

**Script**: `@DiffusionDrive/docker/run.sh`

```bash
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
```

**Key Configuration**:

- Connected to same `carla_net` network
- Workspace and data directories mounted
- SSH keys mounted for git operations

## Network Configuration

### Docker Network Requirements

- **Network Name**: `carla_net`  
- **Purpose**: Enable communication between CARLA simulator and Python client
- **Creation Command**: `docker network create carla_net`

### Hostname Configuration

- **File**: `@DiffusionDrive/test_carla.py:7`
- **Current Setting**: `client = carla.Client("carla", 2000)`
- **Requirement**: Hostname "carla" must match CARLA container name

## Python Dependencies

### CARLA Python Wheel Installation

**Location**: `@Bench2DriveZoo/carla/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl`

**Installation Required**:

- Install in Python 3.10 environment within DiffusionDrive container
- Command: `pip install /path/to/carla-0.9.15-cp310-cp310-linux_x86_64.whl`

## Test Verification

**Test Script**: `@DiffusionDrive/test_carla.py`

- Connects to CARLA on port 2000
- Validates world access and blueprint library
- Tests vehicle spawning capabilities

## Next Steps

### Immediate Tasks

1. **Development Environment Setup**: Configure inference environment for closed-loop testing
2. **Docker Compose Creation**: Create unified docker-compose.yml for easier container orchestration

### Suggested Docker Compose Structure

```yaml
version: '3.8'
services:
  carla:
    image: carlasim/carla:0.9.15
    # ... configuration from run_carla.sh
  
  diffusiondrive:
    image: diffusiondrive:latest  
    # ... configuration from run.sh
    depends_on:
      - carla

networks:
  carla_net:
    driver: bridge
```

## Status

- ✅ Container scripts identified and analyzed
- ✅ Network requirements documented  
- ✅ Python wheel location found
- ⏳ Development environment setup pending
- ⏳ Docker compose file creation pending
