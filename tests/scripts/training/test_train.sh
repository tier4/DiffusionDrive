#!/bin/bash

# Test script for train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../test_framework.sh"

TRAIN_SCRIPT="$SCRIPT_DIR/../../../scripts/training/train.sh"

echo "Testing train.sh script..."
echo "========================="

# Test parameter validation
test_start "train.sh requires --name parameter"
assert_failure "$TRAIN_SCRIPT" "Should fail without --name parameter"
test_end

test_start "train.sh shows help"
OUTPUT=$($TRAIN_SCRIPT --help 2>&1)
assert_contains "$OUTPUT" "Usage:"
assert_contains "$OUTPUT" "--name"
assert_contains "$OUTPUT" "--epochs"
assert_contains "$OUTPUT" "--batch-size"
test_end

# Test with mock environment
test_start "train.sh accepts all parameters"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

# Create a wrapper that captures the command
cat > "$MOCK_BIN/python" << 'EOF'
#!/bin/bash
echo "Python called with arguments:"
echo "$@" > /tmp/train_test_args.txt
echo "Mock training completed successfully"
exit 0
EOF
chmod +x "$MOCK_BIN/python"

# Run train script with parameters
OUTPUT=$($TRAIN_SCRIPT \
    --name test_exp \
    --epochs 10 \
    --batch-size 16 \
    --workers 4 \
    --gpus "0,1" \
    --config test_config \
    --agent test_agent 2>&1)

assert_success "test $? -eq 0" "Train script should succeed"
assert_contains "$OUTPUT" "Mock training completed"

# Check that GPU constraint was applied
assert_equals "0,1" "$CUDA_VISIBLE_DEVICES" "Should limit to 2 GPUs"

cleanup_test_dir "$TEST_DIR"
test_end

# Test default values
test_start "train.sh uses default values"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env

# Capture the actual command arguments
cat > "$MOCK_BIN/python" << 'EOF'
#!/bin/bash
echo "$@" > /tmp/train_default_args.txt
exit 0
EOF
chmod +x "$MOCK_BIN/python"

OUTPUT=$($TRAIN_SCRIPT --name default_test 2>&1)
ARGS=$(cat /tmp/train_default_args.txt 2>/dev/null || echo "")

# Check default values in arguments
assert_contains "$ARGS" "trainer.params.max_epochs=100" "Default epochs should be 100"
assert_contains "$ARGS" "dataloader.params.batch_size=32" "Default batch size should be 32"
assert_contains "$ARGS" "dataloader.params.num_workers=8" "Default workers should be 8"

cleanup_test_dir "$TEST_DIR"
test_end

# Test logging functionality
test_start "train.sh creates log files"
TEST_DIR=$(create_test_dir)
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
create_mock_python "$MOCK_BIN"
export PATH="$MOCK_BIN:$PATH"
setup_mock_env
export LOG_DIR="$TEST_DIR/logs"

OUTPUT=$($TRAIN_SCRIPT --name log_test --epochs 5 2>&1)

assert_dir_exists "$LOG_DIR" "Log directory should be created"
# Find the log file
LOG_FILE=$(find "$LOG_DIR" -name "train_*.log" -type f | head -1)
assert_not_empty "$LOG_FILE" "Log file should be created"

if [ -n "$LOG_FILE" ]; then
    LOG_CONTENT=$(cat "$LOG_FILE")
    assert_contains "$LOG_CONTENT" "experiment_name=log_test"
    assert_contains "$LOG_CONTENT" "max_epochs=5"
fi

cleanup_test_dir "$TEST_DIR"
test_end

# Print test summary
test_summary