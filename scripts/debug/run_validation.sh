#!/bin/bash
# Convenience script to run validation tools from their new locations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "DiffusionDrive Validation Tools"
echo "==============================="
echo ""
echo "Available tools have been moved to better locations:"
echo ""
echo "1. Data Validation:"
echo "   python3 $PROJECT_ROOT/navsim/planning/utils/data_validation/validate_b2d_data.py"
echo ""
echo "2. Check NaN Fixes:"
echo "   python3 $PROJECT_ROOT/navsim/planning/utils/data_validation/check_nan_fixes_status.py"
echo ""
echo "3. Test NaN Fixes:"
echo "   python3 $PROJECT_ROOT/navsim/planning/utils/debugging/test_nan_fixes.py"
echo ""
echo "4. Monitor Training:"
echo "   python3 $PROJECT_ROOT/navsim/planning/utils/debugging/monitor_training_nan.py"
echo ""
echo "5. Regenerate Cache:"
echo "   python3 $PROJECT_ROOT/scripts/data_processing/regenerate_b2d_cache_safe.py"
echo ""

# If arguments provided, run the corresponding tool
case "$1" in
    validate)
        shift
        python3 "$PROJECT_ROOT/navsim/planning/utils/data_validation/validate_b2d_data.py" "$@"
        ;;
    check-fixes)
        shift
        python3 "$PROJECT_ROOT/navsim/planning/utils/data_validation/check_nan_fixes_status.py" "$@"
        ;;
    test-fixes)
        shift
        python3 "$PROJECT_ROOT/navsim/planning/utils/debugging/test_nan_fixes.py" "$@"
        ;;
    monitor)
        shift
        python3 "$PROJECT_ROOT/navsim/planning/utils/debugging/monitor_training_nan.py" "$@"
        ;;
    regenerate-cache)
        shift
        python3 "$PROJECT_ROOT/scripts/data_processing/regenerate_b2d_cache_safe.py" "$@"
        ;;
    *)
        echo "Usage: $0 {validate|check-fixes|test-fixes|monitor|regenerate-cache} [args...]"
        ;;
esac