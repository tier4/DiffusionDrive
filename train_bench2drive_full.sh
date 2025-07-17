#!/bin/bash

# Train DiffusionDrive on full Bench2Drive dataset
# This script calls the organized training script with appropriate parameters

./scripts/training/train.sh \
    --name "bench2drive_full_test0" \
    --epochs 100 \
    --batch-size 32 \
    --workers 8 \
    --gpus "0,1,2,3,4,5,6,7" \
    --config "bench2drive_training" \
    --agent "diffusiondrive_agent" \
    --dataset "bench2drive" \
    --lr 0.0001