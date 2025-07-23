#!/bin/bash

# Use the clean extended configuration that doesn't modify original code
./scripts/training/train.sh \
    --name "bench2drive_Base_clean_test0" \
    --epochs 1000 \
    --batch-size 64 \
    --workers 32 \
    --gpus "0,1,2,3,4,5,6,7" \
    --config "bench2drive_training" \
    --agent "diffusiondrive_agent_extended" \
    --dataset "bench2drive" \
    --lr 5e-5

# For mini dataset training:
# ./scripts/training/train.sh \
#     --name "bench2drive_mini_clean" \
#     --epochs 100 \
#     --batch-size 32 \
#     --workers 8 \
#     --gpus "0,1,2,3,4,5,6,7" \
#     --config "bench2drive_training" \
#     --agent "diffusiondrive_agent_extended" \
#     --dataset "bench2drive_mini" \
#     --lr 5e-5
