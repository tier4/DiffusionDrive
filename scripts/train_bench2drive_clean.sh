#!/bin/bash
# Clean training script for Bench2Drive using extended configuration
# This avoids modifying any original code

# Environment setup
export NAVSIM_DEVKIT_ROOT="/workspace/DiffusionDrive"
export NAVSIM_EXP_ROOT="/workspace/navsim_workspace/exp"
export USE_RAY="${USE_RAY:-True}"
export NUM_WORKERS="${NUM_WORKERS:-32}"

# Experiment configuration
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_diffusiondrive_bench2drive_clean}"
B2D_CACHE="${B2D_CACHE:-/workspace/navsim_workspace/cache/bench2drive_Base_cache}"
B2D_ANCHORS="${B2D_ANCHORS:-${NAVSIM_DEVKIT_ROOT}/download/kmeans_bench2drive_traj_20.npy}"

echo "Starting Bench2Drive training with clean extended configuration..."
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Cache: ${B2D_CACHE}"
echo "Anchors: ${B2D_ANCHORS}"
echo "Ray enabled: ${USE_RAY} with ${NUM_WORKERS} workers"

# Run training with extended agent configuration
python3 ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_bench2drive_training.py \
    agent=diffusiondrive_agent_extended \
    agent.config.dataset_type="bench2drive" \
    agent.config.plan_anchor_path="${B2D_ANCHORS}" \
    experiment_name="${EXPERIMENT_NAME}" \
    cache_path="${B2D_CACHE}" \
    group="DiffusionDriver_Bench2Drive_Clean" \
    job_name="${EXPERIMENT_NAME}" \
    py_func=train \
    +training=training_bench2drive \
    scenario_builder=bench2drive_scenario_builder \
    worker=single_machine_thread_pool \
    worker.use_ray="${USE_RAY}" \
    worker.max_workers="${NUM_WORKERS}" \
    trainer.params.max_epochs=100 \
    trainer.params.check_val_every_n_epoch=5 \
    trainer.params.limit_val_batches=10 \
    trainer.params.enable_progress_bar=True \
    trainer.params.enable_checkpointing=True \
    trainer.params.enable_model_summary=True \
    lr=1e-4 \
    model.load_pretrained_weights=False \
    data_loader.params.batch_size=32 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=True \
    +lr_scheduler=multistep_lr \
    lr_scheduler.milestones="[50, 70, 90]" \
    lr_scheduler.gamma=0.5 \
    callbacks.visualization_callback.frequency=200 \
    trainer.params.log_every_n_steps=10 \
    hydra.run.dir="${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}" \
    wandb.project="diffusiondrive_bench2drive_clean" \
    wandb.name="${EXPERIMENT_NAME}"

echo "Training completed!"