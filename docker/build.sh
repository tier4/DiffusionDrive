#!/bin/bash
# Docker build script for DiffusionDrive

docker build \
    --build-arg CUDA_VER=12.8.1 \
    --build-arg UBUNTU_VER=22.04 \
    --build-arg PYTHON_VER=3.10 \
    --build-arg USERNAME=user \
    --build-arg USERID=1000 \
    --build-arg GROUPID=1000 \
    -t diffusiondrive:latest \
    -f dockerfile \
    .

echo "Docker image built: diffusiondrive:latest"

