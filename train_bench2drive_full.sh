#!/bin/bash

# Train DiffusionDrive on full Bench2Drive dataset
# This script calls the organized training script with appropriate parameters

# ./scripts/training/train.sh \
#     --name "bench2drive_Base_test0" \
#     --epochs 1000 \
#     --batch-size 32 \
#     --workers 16 \
#     --gpus "0,1,2,3,4,5,6,7" \
#     --config "bench2drive_training" \
#     --agent "diffusiondrive_agent" \
#     --dataset "bench2drive_Base" \
#     --lr 4e-5

./scripts/training/train.sh \
    --name "bench2drive_mini_test1" \
    --epochs 100 \
    --batch-size 32 \
    --workers 8 \
    --gpus "0,1,2,3,4,5,6,7" \
    --config "bench2drive_training" \
    --agent "diffusiondrive_agent" \
    --dataset "bench2drive_mini" \
    --lr 5e-5
