#!/bin/bash

# Test script for batch_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../test_framework.sh"

BATCH_SCRIPT="$SCRIPT_DIR/../../../scripts/training/batch_experiments.sh"

echo "Testing batch_experiments.sh script..."
echo "====================================="

# Test help functionality
test_start "batch_experiments.sh shows help"
OUTPUT=$($BATCH_SCRIPT --help 2>&1)
assert_contains "$OUTPUT" "Usage:"
assert_contains "$OUTPUT" "--batch-sizes"
assert_contains "$OUTPUT" "--epochs"
assert_contains "$OUTPUT" "--base-name"
test_end

# Test default behavior
test_start "batch_experiments.sh runs with defaults"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

# Create mock train.sh
MOCK_TRAIN="$SCRIPT_DIR/../../../scripts/training/train.sh"
mkdir -p "$(dirname "$MOCK_TRAIN")"
cat > "$MOCK_BIN/bash" << 'EOF'
#!/bin/bash
# Check if this is our train.sh script
if [[ "$1" =~ train\.sh ]]; then
    echo "Mock train.sh called with: ${@:2}"
    echo "Batch size detected: ${@:2}" | grep -o "batch-size [0-9]*"
    exit 0
fi
# Otherwise, use real bash
/bin/bash "$@"
EOF
chmod +x "$MOCK_BIN/bash"

OUTPUT=$($BATCH_SCRIPT 2>&1)
assert_contains "$OUTPUT" "Starting experiment with batch size:"
assert_contains "$OUTPUT" "batch-size 32"
assert_contains "$OUTPUT" "batch-size 64"
assert_contains "$OUTPUT" "batch-size 128"
assert_contains "$OUTPUT" "All batch experiments completed"

cleanup_test_dir "$TEST_DIR"
test_end

# Test custom batch sizes
test_start "batch_experiments.sh accepts custom batch sizes"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

# Track calls to train.sh
CALL_LOG="$TEST_DIR/calls.log"
cat > "$MOCK_BIN/bash" << EOF
#!/bin/bash
if [[ "\$1" =~ train\.sh ]]; then
    echo "train.sh \${@:2}" >> "$CALL_LOG"
    exit 0
fi
/bin/bash "\$@"
EOF
chmod +x "$MOCK_BIN/bash"

OUTPUT=$($BATCH_SCRIPT --batch-sizes "8,16" --epochs 50 --base-name custom_test 2>&1)

assert_file_exists "$CALL_LOG" "Should track train.sh calls"
CALLS=$(cat "$CALL_LOG")
assert_contains "$CALLS" "--batch-size 8"
assert_contains "$CALLS" "--batch-size 16"
assert_contains "$CALLS" "--epochs 50"
assert_contains "$CALLS" "--name custom_test_bs8"
assert_contains "$CALLS" "--name custom_test_bs16"

cleanup_test_dir "$TEST_DIR"
test_end

# Test GPU constraint
test_start "batch_experiments.sh respects GPU limit"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

# Create a script that checks GPU settings
cat > "$MOCK_BIN/bash" << 'EOF'
#!/bin/bash
if [[ "$1" =~ train\.sh ]]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    exit 0
fi
/bin/bash "$@"
EOF
chmod +x "$MOCK_BIN/bash"

OUTPUT=$($BATCH_SCRIPT --batch-sizes "32" --gpus "0,1" 2>&1)
assert_contains "$OUTPUT" "CUDA_VISIBLE_DEVICES=0,1" "Should use only 2 GPUs"

cleanup_test_dir "$TEST_DIR"
test_end

# Test logging
test_start "batch_experiments.sh creates summary log"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env
export LOG_DIR="$TEST_DIR/logs"

# Simple mock that just succeeds
cat > "$MOCK_BIN/bash" << 'EOF'
#!/bin/bash
if [[ "$1" =~ train\.sh ]]; then
    echo "Mock training batch size ${@:2}"
    exit 0
fi
/bin/bash "$@"
EOF
chmod +x "$MOCK_BIN/bash"

OUTPUT=$($BATCH_SCRIPT --batch-sizes "16" 2>&1)

assert_dir_exists "$LOG_DIR" "Log directory should be created"
LOG_FILE=$(find "$LOG_DIR" -name "batch_experiments_*.log" -type f | head -1)
assert_not_empty "$LOG_FILE" "Summary log file should be created"

if [ -n "$LOG_FILE" ]; then
    LOG_CONTENT=$(cat "$LOG_FILE")
    assert_contains "$LOG_CONTENT" "Batch Sizes: 16"
    assert_contains "$LOG_CONTENT" "Running experiment"
fi

cleanup_test_dir "$TEST_DIR"
test_end

# Print test summary
test_summary