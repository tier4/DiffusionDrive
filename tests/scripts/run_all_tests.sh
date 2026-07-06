#!/bin/bash

# Main test runner for all shell script tests
# This ensures all tests run with only 2 GPUs

# Set up test environment
export CUDA_VISIBLE_DEVICES="0,1"  # Limit to 2 GPUs
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/DiffusionDrive}"
export NAVSIM_EXP_ROOT="/tmp/diffusiondrive_test_exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}DiffusionDrive Shell Script Test Suite${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Test Configuration:"
echo "  - GPU Limit: 2 GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "  - NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo ""

# Make sure all test scripts are executable
find "$SCRIPT_DIR" -name "test_*.sh" -type f -exec chmod +x {} \;

# Track overall results
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

# Function to run a test suite
run_test_suite() {
    local test_file="$1"
    local suite_name=$(basename "$test_file" .sh)
    
    echo -e "\n${YELLOW}Running test suite: $suite_name${NC}"
    echo "----------------------------------------"
    
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    
    if bash "$test_file"; then
        echo -e "${GREEN}✓ $suite_name passed${NC}"
        PASSED_SUITES=$((PASSED_SUITES + 1))
    else
        echo -e "${RED}✗ $suite_name failed${NC}"
        FAILED_SUITES=$((FAILED_SUITES + 1))
    fi
}

# Run all test suites
echo -e "\n${BLUE}Running all test suites...${NC}\n"

# Test common utilities first
if [ -f "$SCRIPT_DIR/utils/test_common.sh" ]; then
    run_test_suite "$SCRIPT_DIR/utils/test_common.sh"
fi

# Test training scripts
for test_file in "$SCRIPT_DIR/training/test_"*.sh; do
    if [ -f "$test_file" ]; then
        run_test_suite "$test_file"
    fi
done

# Test evaluation scripts
for test_file in "$SCRIPT_DIR/evaluation/test_"*.sh; do
    if [ -f "$test_file" ]; then
        run_test_suite "$test_file"
    fi
done

# Print overall summary
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Overall Test Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo "Total test suites: $TOTAL_SUITES"
echo -e "Passed: ${GREEN}$PASSED_SUITES${NC}"
echo -e "Failed: ${RED}$FAILED_SUITES${NC}"

if [ $FAILED_SUITES -eq 0 ]; then
    echo -e "\n${GREEN}All test suites passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some test suites failed!${NC}"
    exit 1
fi