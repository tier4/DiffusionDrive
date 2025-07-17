#!/bin/bash

# Main training script for DiffusionDrive
# Usage: ./train.sh [options]
#   --name NAME          Experiment name (required)
#   --epochs N           Max epochs (default: 100)
#   --batch-size N       Batch size (default: 32)
#   --workers N          Number of workers (default: 8)
#   --gpus DEVICES       GPU devices (default: 0,1,2,3,4,5,6,7)
#   --config CONFIG      Config name (default: default_training)
#   --agent AGENT        Agent type (default: diffusiondrive_agent)
#   --dataset DATASET    Dataset type (auto-detect if not specified)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/common.sh"

# Default values
EXPERIMENT_NAME=""
MAX_EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=8
GPU_DEVICES="0,1,2,3,4,5,6,7"
CONFIG_NAME="default_training"
AGENT="diffusiondrive_agent"
DATASET_TYPE=""
LEARNING_RATE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            GPU_DEVICES="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --dataset)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --name NAME          Experiment name (required)"
            echo "  --epochs N           Max epochs (default: 100)"
            echo "  --batch-size N       Batch size (default: 32)"
            echo "  --workers N          Number of workers (default: 8)"
            echo "  --gpus DEVICES       GPU devices (default: 0,1,2,3,4,5,6,7)"
            echo "  --config CONFIG      Config name (default: default_training)"
            echo "  --agent AGENT        Agent type (default: diffusiondrive_agent)"
            echo "  --dataset DATASET    Dataset type (auto-detect if not specified)"
            echo "  --lr RATE            Learning rate (optional, overrides agent default)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: --name is required"
    exit 1
fi

# Setup
setup_logging "training_${EXPERIMENT_NAME}"
setup_environment
setup_cuda "$GPU_DEVICES"

# Build full experiment name
FULL_EXPERIMENT_NAME="training_${AGENT}_${EXPERIMENT_NAME}"

# Log configuration
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Experiment: $FULL_EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "  Config: $CONFIG_NAME" | tee -a "$LOG_FILE"
echo "  Agent: $AGENT" | tee -a "$LOG_FILE"
echo "  Max Epochs: $MAX_EPOCHS" | tee -a "$LOG_FILE"
echo "  Batch Size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Workers: $NUM_WORKERS" | tee -a "$LOG_FILE"
if [ ! -z "$LEARNING_RATE" ]; then
    echo "  Learning Rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
fi

# Determine dataset type (use explicit if provided, otherwise auto-detect)
if [ -z "$DATASET_TYPE" ]; then
    DATASET_TYPE="navtrain"
    if [[ "$AGENT" == *"bench2drive"* ]] || [[ "$CONFIG_NAME" == *"bench2drive"* ]]; then
        DATASET_TYPE="bench2drive"
    fi
fi

# Build training arguments
build_training_args "$AGENT" "$FULL_EXPERIMENT_NAME" "$MAX_EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$DATASET_TYPE"

# Add learning rate override if specified
if [ ! -z "$LEARNING_RATE" ]; then
    TRAINING_ARGS+=("agent.lr=$LEARNING_RATE")
fi

# Start training
log_start "Training"

# Run training with appropriate script and config
if [[ "$DATASET_TYPE" == "bench2drive"* ]]; then
    # Use Bench2Drive training script
    python3 -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_bench2drive_training.py" \
        "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
elif [ "$CONFIG_NAME" = "default_training_w_callbacks" ]; then
    python3 -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
        --config-name="$CONFIG_NAME" \
        "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
else
    python3 -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
        "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

# Log completion
log_finish "Training"