# Shell Script Tests

This directory contains comprehensive tests for the DiffusionDrive shell scripts.

## Overview

The test suite ensures that all training and evaluation scripts work correctly with a 2-GPU constraint, as requested. Tests are organized by functionality:

- `test_framework.sh` - Core testing utilities and assertions
- `utils/test_common.sh` - Tests for common utility functions
- `training/test_train.sh` - Tests for the main training script
- `training/test_batch_experiments.sh` - Tests for batch experiment runner
- `evaluation/test_eval.sh` - Tests for single checkpoint evaluation
- `evaluation/test_eval_all_checkpoints.sh` - Tests for bulk evaluation
- `test_integration.sh` - Integration tests for script interactions
- `run_all_tests.sh` - Main test runner

## Running Tests

### Run all tests:
```bash
cd tests/scripts
./run_all_tests.sh
```

### Run individual test suites:
```bash
# Test common utilities
./utils/test_common.sh

# Test training scripts
./training/test_train.sh
./training/test_batch_experiments.sh

# Test evaluation scripts
./evaluation/test_eval.sh
./evaluation/test_eval_all_checkpoints.sh

# Test integration
./test_integration.sh
```

## GPU Constraint

All tests enforce a 2-GPU limit by setting `CUDA_VISIBLE_DEVICES=0,1`. This ensures that:
- Scripts respect GPU constraints
- Multi-GPU workflows work with limited resources
- Resource usage is predictable and controlled

## Test Features

### Mock Environment
- Tests use mock Python executables to avoid running actual training/evaluation
- Temporary directories are created and cleaned up automatically
- Environment variables are mocked to avoid dependencies

### Assertions
The test framework provides various assertions:
- `assert_equals` - Check value equality
- `assert_not_empty` - Verify non-empty values
- `assert_file_exists` - Check file existence
- `assert_dir_exists` - Check directory existence
- `assert_success` - Verify command succeeds
- `assert_failure` - Verify command fails
- `assert_contains` - Check string containment

### Coverage
Tests verify:
- ✅ Parameter parsing and validation
- ✅ Help documentation
- ✅ Default values
- ✅ Error handling
- ✅ Log file creation
- ✅ GPU constraint enforcement
- ✅ Environment variable handling
- ✅ Script integration

## Example Output

```
======================================
DiffusionDrive Shell Script Test Suite
======================================

Test Configuration:
  - GPU Limit: 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)
  - NAVSIM_DEVKIT_ROOT: /workspace/DiffusionDrive
  - NAVSIM_EXP_ROOT: /tmp/diffusiondrive_test_exp

Running test suite: test_common
Testing common.sh utilities...
Testing setup_logging creates log directory and file... PASSED
Testing log_start and log_finish work correctly... PASSED
...

Overall Test Summary
====================
Total test suites: 6
Passed: 6
Failed: 0

All test suites passed!
```

## Adding New Tests

To add tests for new scripts:

1. Create a test file: `test_<script_name>.sh`
2. Source the test framework
3. Write test cases using the assertion functions
4. Add to appropriate directory (training/, evaluation/, etc.)
5. The test will be automatically picked up by `run_all_tests.sh`

Example test structure:
```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../test_framework.sh"

echo "Testing my_script.sh..."
echo "======================"

test_start "my_script does something"
# Test code here
assert_equals "expected" "actual"
test_end

test_summary
```