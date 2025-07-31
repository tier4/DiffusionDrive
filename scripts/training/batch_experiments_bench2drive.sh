#!/bin/bash

# Batch experiments runner for DiffusionDrive with Bench2Drive dataset
# Usage: ./batch_experiments_bench2drive.sh [options]
#   --batch-sizes "32,64,128,256"    Comma-separated list of batch sizes
#   --epochs N                       Max epochs for all experiments
#   --base-name NAME                 Base experiment name
#   --lr RATE                        Single learning rate for all experiments (optional)
#   --lr-list "1e-4,2e-4,4e-4,8e-4"  Comma-separated learning rates for each batch size

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/common.sh"

# Default values (adapted from train_bench2drive_full.sh)
BATCH_SIZES="32,64,128,256"
MAX_EPOCHS=1000
BASE_NAME="bench2drive_batch_sweep"
GPU_DEVICES="0,1,2,3,4,5,6,7"
LEARNING_RATE=""
LEARNING_RATE_LIST=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --batch-sizes)
        BATCH_SIZES="$2"
        shift 2
        ;;
    --epochs)
        MAX_EPOCHS="$2"
        shift 2
        ;;
    --base-name)
        BASE_NAME="$2"
        shift 2
        ;;
    --gpus)
        GPU_DEVICES="$2"
        shift 2
        ;;
    --lr)
        LEARNING_RATE="$2"
        shift 2
        ;;
    --lr-list)
        LEARNING_RATE_LIST="$2"
        shift 2
        ;;
    --help)
        echo "Usage: $0 [options]"
        echo "  --batch-sizes SIZES  Comma-separated batch sizes (default: 32,64,128,256)"
        echo "  --epochs N           Max epochs (default: 1000)"
        echo "  --base-name NAME     Base experiment name (default: bench2drive_batch_sweep)"
        echo "  --gpus DEVICES       GPU devices (default: 0,1,2,3,4,5,6,7)"
        echo "  --lr RATE            Single learning rate for all experiments (default: 5e-5)"
        echo "  --lr-list RATES      Comma-separated learning rates for each batch size"
        echo ""
        echo "Note: --lr and --lr-list are mutually exclusive. If --lr-list is provided,"
        echo "      it must have the same number of values as batch sizes."
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Setup
setup_logging "batch_experiments_bench2drive_${BASE_NAME}"
setup_environment

# Validate --lr and --lr-list are not both provided
if [ ! -z "$LEARNING_RATE" ] && [ ! -z "$LEARNING_RATE_LIST" ]; then
    echo "Error: --lr and --lr-list cannot be used together" | tee -a "$LOG_FILE"
    exit 1
fi

# Calculate workers as num_gpus * 8
IFS=',' read -ra GPU_ARRAY <<<"$GPU_DEVICES"
WORKERS=$((${#GPU_ARRAY[@]} * 8))

# Convert comma-separated lists to arrays
IFS=',' read -ra BATCH_SIZE_ARRAY <<<"$BATCH_SIZES"
if [ ! -z "$LEARNING_RATE_LIST" ]; then
    IFS=',' read -ra LR_ARRAY <<<"$LEARNING_RATE_LIST"

    # Validate number of learning rates matches number of batch sizes
    if [ ${#LR_ARRAY[@]} -ne ${#BATCH_SIZE_ARRAY[@]} ]; then
        echo "Error: Number of learning rates (${#LR_ARRAY[@]}) must match number of batch sizes (${#BATCH_SIZE_ARRAY[@]})" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

echo "Bench2Drive Batch Experiment Configuration:" | tee -a "$LOG_FILE"
echo "  Base Name: $BASE_NAME" | tee -a "$LOG_FILE"
echo "  Batch Sizes: ${BATCH_SIZE_ARRAY[*]}" | tee -a "$LOG_FILE"
echo "  Max Epochs: $MAX_EPOCHS" | tee -a "$LOG_FILE"
echo "  Workers: $WORKERS (${#GPU_ARRAY[@]} GPUs × 8)" | tee -a "$LOG_FILE"
if [ ! -z "$LEARNING_RATE" ]; then
    echo "  Learning Rate: $LEARNING_RATE (for all experiments)" | tee -a "$LOG_FILE"
elif [ ! -z "$LEARNING_RATE_LIST" ]; then
    echo "  Learning Rates: ${LR_ARRAY[*]}" | tee -a "$LOG_FILE"
else
    echo "  Learning Rate: 5e-5 (default for Bench2Drive)" | tee -a "$LOG_FILE"
fi

# Run experiments for each batch size
for i in "${!BATCH_SIZE_ARRAY[@]}"; do
    BATCH_SIZE="${BATCH_SIZE_ARRAY[$i]}"

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting Bench2Drive experiment with batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"

    # Determine learning rate for this experiment
    if [ ! -z "$LEARNING_RATE_LIST" ]; then
        CURRENT_LR="${LR_ARRAY[$i]}"
        echo "Learning rate: $CURRENT_LR" | tee -a "$LOG_FILE"
    elif [ ! -z "$LEARNING_RATE" ]; then
        CURRENT_LR="$LEARNING_RATE"
        echo "Learning rate: $CURRENT_LR" | tee -a "$LOG_FILE"
    else
        CURRENT_LR="5e-5" # Default for Bench2Drive
        echo "Learning rate: $CURRENT_LR (default)" | tee -a "$LOG_FILE"
    fi
    echo "========================================" | tee -a "$LOG_FILE"

    # Build experiment name with learning rate
    # Format learning rate for filename:
    # - Replace decimal point with 'p' (e.g., 0.001 -> 0p001)
    # - Replace 'e-' with 'e' (e.g., 1e-4 -> 1e4)
    # - Replace 'e+' with 'e' (e.g., 1e+3 -> 1e3)
    LR_FORMATTED=$(echo "$CURRENT_LR" | sed 's/\./p/g' | sed 's/e-/e/g' | sed 's/e+/e/g')
    EXPERIMENT_NAME="${BASE_NAME}_bs${BATCH_SIZE}_lr${LR_FORMATTED}_ep${MAX_EPOCHS}"

    # Build training command using the same structure as train_bench2drive_full.sh
    TRAIN_CMD=("$SCRIPT_DIR/train.sh"
        "--name" "$EXPERIMENT_NAME"
        "--epochs" "$MAX_EPOCHS"
        "--batch-size" "$BATCH_SIZE"
        "--workers" "$WORKERS"
        "--gpus" "$GPU_DEVICES"
        "--config" "bench2drive_training"
        "--agent" "diffusiondrive_agent_extended"
        "--dataset" "bench2drive"
        "--lr" "$CURRENT_LR")

    # Run training script
    "${TRAIN_CMD[@]}"

    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "Bench2Drive experiment $EXPERIMENT_NAME completed successfully" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Bench2Drive experiment $EXPERIMENT_NAME failed" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "All Bench2Drive batch experiments completed" | tee -a "$LOG_FILE"
