#!/bin/bash
# Training script for DiffusionDrive with Bench2Drive dataset

# Set environment variables
export NAVSIM_DEVKIT_ROOT=$(pwd)
export NAVSIM_EXP_ROOT="/workspace/navsim_workspace/exp"
export BENCH2DRIVE_ROOT="/workspace/Bench2Drive-Base"

# Create experiment directory
mkdir -p $NAVSIM_EXP_ROOT

echo "Starting Bench2Drive training with DiffusionDrive..."
echo "NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "BENCH2DRIVE_ROOT: $BENCH2DRIVE_ROOT"

# Step 1: Cache the dataset (if not already cached)
echo ""
echo "Step 1: Caching Bench2Drive dataset..."
python3 $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_bench2drive_caching.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    experiment_name=bench2drive_diffusiondrive_caching \
    split=train \
    cache.cache_path="${NAVSIM_EXP_ROOT}/bench2drive_training_cache/" \
    cache.force_cache_computation=False

# Step 2: Run training
echo ""
echo "Step 2: Starting training..."
python3 $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_bench2drive_training.py \
    agent=diffusiondrive_agent_b2d \
    train_test_split=bench2drive \
    experiment_name=bench2drive_diffusiondrive_training \
    split=train \
    trainer.params.max_epochs=100 \
    trainer.params.check_val_every_n_epoch=5 \
    trainer.params.limit_val_batches=100 \
    cache_path="${NAVSIM_EXP_ROOT}/bench2drive_training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    dataloader.params.batch_size=32 \
    dataloader.params.num_workers=8 \
    lr_scheduler.config.warmup_steps=1000 \
    agent.config.lr=0.0003

echo ""
echo "Training complete! Check logs in $NAVSIM_EXP_ROOT/bench2drive_diffusiondrive_training/"