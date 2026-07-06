#!/bin/bash

# Test script for eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../test_framework.sh"

EVAL_SCRIPT="$SCRIPT_DIR/../../../scripts/evaluation/eval.sh"

echo "Testing eval.sh script..."
echo "========================"

# Test parameter validation
test_start "eval.sh requires --checkpoint parameter"
assert_failure "$EVAL_SCRIPT" "Should fail without --checkpoint parameter"
test_end

test_start "eval.sh shows help"
OUTPUT=$($EVAL_SCRIPT --help 2>&1)
assert_contains "$OUTPUT" "Usage:"
assert_contains "$OUTPUT" "--checkpoint"
assert_contains "$OUTPUT" "--name"
assert_contains "$OUTPUT" "--agent"
test_end

# Test with valid checkpoint
test_start "eval.sh validates checkpoint exists"
TEST_DIR=$(create_test_dir)
setup_mock_env

# Test with non-existent checkpoint
OUTPUT=$($EVAL_SCRIPT --checkpoint "/nonexistent/checkpoint.ckpt" 2>&1)
assert_contains "$OUTPUT" "Error: Checkpoint file not found" "Should error on missing checkpoint"

cleanup_test_dir "$TEST_DIR"
test_end

# Test successful evaluation
test_start "eval.sh runs evaluation with valid checkpoint"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

# Create mock checkpoint
CHECKPOINT="$TEST_DIR/model.ckpt"
create_mock_checkpoint "$CHECKPOINT"

# Create python mock that captures arguments
cat > "$MOCK_BIN/python" << 'EOF'
#!/bin/bash
echo "Python evaluation called with:"
echo "$@"
# Check for required arguments
if [[ "$@" =~ agent\.checkpoint_path ]]; then
    echo "Checkpoint path found in arguments"
fi
echo "Evaluation completed successfully"
exit 0
EOF
chmod +x "$MOCK_BIN/python"

OUTPUT=$($EVAL_SCRIPT --checkpoint "$CHECKPOINT" --name test_eval 2>&1)
assert_success "test $? -eq 0" "Evaluation should succeed"
assert_contains "$OUTPUT" "Evaluation completed successfully"
assert_contains "$OUTPUT" "Checkpoint path found"

cleanup_test_dir "$TEST_DIR"
test_end

# Test auto-generated experiment name
test_start "eval.sh auto-generates experiment name when not provided"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

CHECKPOINT="$TEST_DIR/checkpoints/epoch_100.ckpt"
create_mock_checkpoint "$CHECKPOINT"

# Capture the actual command
cat > "$MOCK_BIN/python" << 'EOF'
#!/bin/bash
echo "$@" > /tmp/eval_args.txt
exit 0
EOF
chmod +x "$MOCK_BIN/python"

OUTPUT=$($EVAL_SCRIPT --checkpoint "$CHECKPOINT" 2>&1)
ARGS=$(cat /tmp/eval_args.txt 2>/dev/null || echo "")

# Check that experiment name was auto-generated from checkpoint
assert_contains "$ARGS" "experiment_name=" "Should have experiment name"
assert_contains "$ARGS" "epoch_100" "Auto-generated name should contain checkpoint info"

cleanup_test_dir "$TEST_DIR"
test_end

# Test GPU constraint
test_start "eval.sh respects GPU constraint"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Start with 4 GPUs

CHECKPOINT="$TEST_DIR/model.ckpt"
create_mock_checkpoint "$CHECKPOINT"

cat > "$MOCK_BIN/python" << 'EOF'
#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exit 0
EOF
chmod +x "$MOCK_BIN/python"

OUTPUT=$($EVAL_SCRIPT --checkpoint "$CHECKPOINT" 2>&1)
# The script should use the setup_mock_env which sets CUDA_VISIBLE_DEVICES to "0,1"
assert_contains "$OUTPUT" "Using GPUs: 0,1" "Should limit to 2 GPUs"

cleanup_test_dir "$TEST_DIR"
test_end

# Test logging
test_start "eval.sh creates log files"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env
export LOG_DIR="$TEST_DIR/logs"

CHECKPOINT="$TEST_DIR/model.ckpt"
create_mock_checkpoint "$CHECKPOINT"

OUTPUT=$($EVAL_SCRIPT --checkpoint "$CHECKPOINT" --name log_test 2>&1)

assert_dir_exists "$LOG_DIR" "Log directory should be created"
LOG_FILE=$(find "$LOG_DIR" -name "eval_*.log" -type f | head -1)
assert_not_empty "$LOG_FILE" "Log file should be created"

if [ -n "$LOG_FILE" ]; then
    LOG_CONTENT=$(cat "$LOG_FILE")
    assert_contains "$LOG_CONTENT" "checkpoint"
    assert_contains "$LOG_CONTENT" "experiment_name"
fi

cleanup_test_dir "$TEST_DIR"
test_end

# Print test summary
test_summary