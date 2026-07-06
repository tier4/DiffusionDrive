#!/bin/bash

# Test framework for shell scripts
# Provides assertions and test utilities

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Current test context
CURRENT_TEST=""
TEST_FAILED=0

# Start a new test
test_start() {
    local test_name="$1"
    CURRENT_TEST="$test_name"
    TEST_FAILED=0
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -n "Testing $test_name... "
}

# End current test
test_end() {
    if [ $TEST_FAILED -eq 0 ]; then
        echo -e "${GREEN}PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Assert that two values are equal
assert_equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Values should be equal}"
    
    if [ "$expected" != "$actual" ]; then
        echo -e "\n  ${RED}✗ $message${NC}"
        echo "    Expected: '$expected'"
        echo "    Actual:   '$actual'"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Assert that a value is not empty
assert_not_empty() {
    local value="$1"
    local message="${2:-Value should not be empty}"
    
    if [ -z "$value" ]; then
        echo -e "\n  ${RED}✗ $message${NC}"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Assert that a file exists
assert_file_exists() {
    local file="$1"
    local message="${2:-File should exist}"
    
    if [ ! -f "$file" ]; then
        echo -e "\n  ${RED}✗ $message: $file${NC}"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Assert that a directory exists
assert_dir_exists() {
    local dir="$1"
    local message="${2:-Directory should exist}"
    
    if [ ! -d "$dir" ]; then
        echo -e "\n  ${RED}✗ $message: $dir${NC}"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Assert that a command succeeds
assert_success() {
    local command="$1"
    local message="${2:-Command should succeed}"
    
    if ! eval "$command" >/dev/null 2>&1; then
        echo -e "\n  ${RED}✗ $message: $command${NC}"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Assert that a command fails
assert_failure() {
    local command="$1"
    local message="${2:-Command should fail}"
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "\n  ${RED}✗ $message: $command${NC}"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Assert that output contains a string
assert_contains() {
    local output="$1"
    local expected="$2"
    local message="${3:-Output should contain string}"
    
    if [[ ! "$output" =~ "$expected" ]]; then
        echo -e "\n  ${RED}✗ $message${NC}"
        echo "    Expected to contain: '$expected'"
        echo "    Actual output: '$output'"
        TEST_FAILED=1
        return 1
    fi
    return 0
}

# Create a temporary test directory
create_test_dir() {
    local test_dir="/tmp/diffusiondrive_test_$$"
    mkdir -p "$test_dir"
    echo "$test_dir"
}

# Cleanup test directory
cleanup_test_dir() {
    local test_dir="$1"
    if [[ "$test_dir" =~ ^/tmp/diffusiondrive_test_ ]]; then
        rm -rf "$test_dir"
    fi
}

# Setup mock environment variables
setup_mock_env() {
    export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/DiffusionDrive}"
    export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/tmp/diffusiondrive_test_exp}"
    export CUDA_VISIBLE_DEVICES="0,1"  # Limit to 2 GPUs
}

# Print test summary
test_summary() {
    echo ""
    echo "Test Summary:"
    echo "============="
    echo "Total tests: $TESTS_RUN"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        return 1
    fi
}

# Mock python command for testing
create_mock_python() {
    local mock_dir="$1"
    cat > "$mock_dir/python" << 'EOF'
#!/bin/bash
# Mock python script for testing
echo "Mock Python executed with args: $@"
# Simulate successful execution
exit 0
EOF
    chmod +x "$mock_dir/python"
}

# Create mock checkpoint file
create_mock_checkpoint() {
    local checkpoint_path="$1"
    mkdir -p "$(dirname "$checkpoint_path")"
    echo "mock checkpoint data" > "$checkpoint_path"
}