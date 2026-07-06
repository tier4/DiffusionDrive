#!/bin/bash

# Test script for eval_all_checkpoints.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../test_framework.sh"

EVAL_ALL_SCRIPT="$SCRIPT_DIR/../../../scripts/evaluation/eval_all_checkpoints.sh"

echo "Testing eval_all_checkpoints.sh script..."
echo "========================================"

# Test help functionality
test_start "eval_all_checkpoints.sh shows help"
OUTPUT=$($EVAL_ALL_SCRIPT --help 2>&1)
assert_contains "$OUTPUT" "Usage:"
assert_contains "$OUTPUT" "--dir"
assert_contains "$OUTPUT" "--pattern"
assert_contains "$OUTPUT" "--agent"
test_end

# Test directory validation
test_start "eval_all_checkpoints.sh validates directory"
OUTPUT=$($EVAL_ALL_SCRIPT --dir /nonexistent/directory 2>&1)
assert_contains "$OUTPUT" "Error: Directory not found" "Should error on missing directory"
test_end

# Test finding checkpoints
test_start "eval_all_checkpoints.sh finds checkpoint files"
TEST_DIR=$(create_test_dir)
CHECKPOINT_DIR="$TEST_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# Create mock checkpoints
create_mock_checkpoint "$CHECKPOINT_DIR/epoch_10.ckpt"
create_mock_checkpoint "$CHECKPOINT_DIR/epoch_20.ckpt"
create_mock_checkpoint "$CHECKPOINT_DIR/best_model.ckpt"
touch "$CHECKPOINT_DIR/not_a_checkpoint.txt"

# Mock eval.sh
MOCK_EVAL="$SCRIPT_DIR/../../../scripts/evaluation/eval.sh"
mkdir -p "$(dirname "$MOCK_EVAL")"
cat > "$TEST_DIR/mock_eval.sh" << 'EOF'
#!/bin/bash
echo "Evaluating checkpoint: $2"
exit 0
EOF
chmod +x "$TEST_DIR/mock_eval.sh"

# Replace eval.sh path in the script for testing
export EVAL_SCRIPT="$TEST_DIR/mock_eval.sh"

OUTPUT=$($EVAL_ALL_SCRIPT --dir "$CHECKPOINT_DIR" 2>&1)
assert_contains "$OUTPUT" "Found 3 checkpoint(s)" "Should find all .ckpt files"
assert_contains "$OUTPUT" "epoch_10.ckpt"
assert_contains "$OUTPUT" "epoch_20.ckpt"
assert_contains "$OUTPUT" "best_model.ckpt"
assert_contains "$OUTPUT" "All evaluations completed"

cleanup_test_dir "$TEST_DIR"
test_end

# Test custom pattern
test_start "eval_all_checkpoints.sh respects custom pattern"
TEST_DIR=$(create_test_dir)
CHECKPOINT_DIR="$TEST_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# Create checkpoints with different patterns
create_mock_checkpoint "$CHECKPOINT_DIR/epoch_10.ckpt"
create_mock_checkpoint "$CHECKPOINT_DIR/epoch_20.ckpt"
create_mock_checkpoint "$CHECKPOINT_DIR/best_model.ckpt"
create_mock_checkpoint "$CHECKPOINT_DIR/model.pth"

export EVAL_SCRIPT="$TEST_DIR/mock_eval.sh"
cat > "$TEST_DIR/mock_eval.sh" << 'EOF'
#!/bin/bash
echo "Evaluating: $(basename "$2")"
exit 0
EOF
chmod +x "$TEST_DIR/mock_eval.sh"

# Test with epoch pattern only
OUTPUT=$($EVAL_ALL_SCRIPT --dir "$CHECKPOINT_DIR" --pattern "epoch_*.ckpt" 2>&1)
assert_contains "$OUTPUT" "Found 2 checkpoint files" "Should find only epoch checkpoints"
assert_contains "$OUTPUT" "epoch_10.ckpt"
assert_contains "$OUTPUT" "epoch_20.ckpt"
assert_contains "$OUTPUT" -v "best_model" "Should not include best_model"

cleanup_test_dir "$TEST_DIR"
test_end

# Test parallel evaluation with GPU constraint
test_start "eval_all_checkpoints.sh runs evaluations with GPU limit"
TEST_DIR=$(create_test_dir)
CHECKPOINT_DIR="$TEST_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
setup_mock_env

# Create multiple checkpoints
for i in 1 2 3 4; do
    create_mock_checkpoint "$CHECKPOINT_DIR/model_$i.ckpt"
done

# Create eval mock that shows GPU usage
cat > "$TEST_DIR/mock_eval.sh" << 'EOF'
#!/bin/bash
echo "Eval $2 using GPUs: $CUDA_VISIBLE_DEVICES"
sleep 0.1  # Simulate work
exit 0
EOF
chmod +x "$TEST_DIR/mock_eval.sh"
export EVAL_SCRIPT="$TEST_DIR/mock_eval.sh"

OUTPUT=$($EVAL_ALL_SCRIPT --dir "$CHECKPOINT_DIR" 2>&1)
assert_contains "$OUTPUT" "Using GPUs: 0,1" "Should show GPU constraint"
assert_contains "$OUTPUT" "Found 4 checkpoint(s)"

cleanup_test_dir "$TEST_DIR"
test_end

# Test summary generation
test_start "eval_all_checkpoints.sh creates summary log"
TEST_DIR=$(create_test_dir)
CHECKPOINT_DIR="$TEST_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
export LOG_DIR="$TEST_DIR/logs"

create_mock_checkpoint "$CHECKPOINT_DIR/model_1.ckpt"
create_mock_checkpoint "$CHECKPOINT_DIR/model_2.ckpt"

cat > "$TEST_DIR/mock_eval.sh" << 'EOF'
#!/bin/bash
echo "Evaluated $(basename "$2")"
exit 0
EOF
chmod +x "$TEST_DIR/mock_eval.sh"
export EVAL_SCRIPT="$TEST_DIR/mock_eval.sh"

OUTPUT=$($EVAL_ALL_SCRIPT --dir "$CHECKPOINT_DIR" 2>&1)

assert_dir_exists "$LOG_DIR" "Log directory should be created"
LOG_FILE=$(find "$LOG_DIR" -name "eval_all_*.log" -type f | head -1)
assert_not_empty "$LOG_FILE" "Summary log should be created"

if [ -n "$LOG_FILE" ]; then
    LOG_CONTENT=$(cat "$LOG_FILE")
    assert_contains "$LOG_CONTENT" "Batch Evaluation Configuration:"
    assert_contains "$LOG_CONTENT" "model_1.ckpt"
    assert_contains "$LOG_CONTENT" "model_2.ckpt"
fi

cleanup_test_dir "$TEST_DIR"
test_end

# Print test summary
test_summary