#!/bin/bash

# Script to run all NaN analysis tools for Bench2Drive debugging

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAVSIM_CACHE="/workspace/navsim_workspace/exp/training_cache"
B2D_CACHE="/workspace/navsim_workspace/cache/bench2drive_Base_cache"
OUTPUT_DIR="nan_analysis_results_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}=== Bench2Drive NaN Analysis Suite ===${NC}"
echo "NavSim cache: $NAVSIM_CACHE"
echo "Bench2Drive cache: $B2D_CACHE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run analysis with error handling
run_analysis() {
    local script_name=$1
    local output_name=$2
    shift 2
    local args=$@
    
    echo -e "${YELLOW}Running $script_name...${NC}"
    
    if python3 scripts/debug/$script_name $args > "$OUTPUT_DIR/${output_name}.log" 2>&1; then
        echo -e "${GREEN}✓ $script_name completed successfully${NC}"
        echo "  Output saved to: $OUTPUT_DIR/${output_name}.log"
    else
        echo -e "${RED}✗ $script_name failed!${NC}"
        echo "  Check error log: $OUTPUT_DIR/${output_name}.log"
        tail -n 20 "$OUTPUT_DIR/${output_name}.log"
        return 1
    fi
}

# 1. Cache comparison analysis (10,000 samples)
echo -e "\n${GREEN}[1/4] Cache Comparison Analysis${NC}"
run_analysis "analyze_cache_comparison.py" "cache_comparison" \
    --navsim-cache "$NAVSIM_CACHE" \
    --b2d-cache "$B2D_CACHE" \
    --max-samples 10000 \
    --output-dir "$OUTPUT_DIR/cache_comparison" \
    --plot-fields features/trajectory/x features/trajectory/y features/trajectory/heading \
                  features/status_feature targets/trajectory/x targets/trajectory/y \
                  targets/trajectory/heading targets/agent_states

# 2. Dual dataloader analysis 
echo -e "\n${GREEN}[2/4] Dataloader Pipeline Analysis${NC}"
run_analysis "analyze_dual_dataloaders.py" "dataloader_analysis" \
    --navsim-cache "$NAVSIM_CACHE" \
    --b2d-cache "$B2D_CACHE" \
    --max-batches 100 \
    --output "$OUTPUT_DIR/dataloader_analysis_report.json"

# 3. Feature transformation analysis
echo -e "\n${GREEN}[3/4] Feature Transformation Analysis${NC}"
run_analysis "analyze_feature_transforms.py" "feature_transforms" \
    --navsim-cache "$NAVSIM_CACHE" \
    --b2d-cache "$B2D_CACHE" \
    --num-samples 1000 \
    --output-dir "$OUTPUT_DIR/transform_analysis"

# 4. Pipeline differences analysis
echo -e "\n${GREEN}[4/4] Pipeline Differences Analysis${NC}"
run_analysis "analyze_pipeline_differences.py" "pipeline_differences" \
    --navsim-cache "$NAVSIM_CACHE" \
    --b2d-cache "$B2D_CACHE" \
    --output "$OUTPUT_DIR/pipeline_differences_report.json" \
    --test-model

# Summary
echo -e "\n${GREEN}=== Analysis Complete ===${NC}"
echo "All results saved to: $OUTPUT_DIR"
echo ""
echo "Key files to check:"
echo "  - $OUTPUT_DIR/cache_comparison/cache_comparison_results.json"
echo "  - $OUTPUT_DIR/cache_comparison/dist_comparison_*.png"
echo "  - $OUTPUT_DIR/dataloader_analysis_report.json"
echo "  - $OUTPUT_DIR/transform_analysis/transformation_analysis.json"
echo "  - $OUTPUT_DIR/pipeline_differences_report.json"

# Create summary report
echo -e "\nGenerating summary report..."
cat > "$OUTPUT_DIR/ANALYSIS_SUMMARY.md" << EOF
# Bench2Drive NaN Analysis Summary

Generated: $(date)

## Analysis Results

### 1. Cache Comparison
- Log: cache_comparison.log
- Results: cache_comparison/cache_comparison_results.json
- Plots: cache_comparison/dist_comparison_*.png

### 2. Dataloader Pipeline
- Log: dataloader_analysis.log
- Report: dataloader_analysis_report.json

### 3. Feature Transformations
- Log: feature_transforms.log
- Results: transform_analysis/transformation_analysis.json
- Plots: transform_analysis/transformation_effects.png

### 4. Pipeline Differences
- Log: pipeline_differences.log
- Report: pipeline_differences_report.json

## Quick Checks

1. Check for NaN occurrences:
\`\`\`bash
grep -i "nan" $OUTPUT_DIR/*.log
\`\`\`

2. Check for errors:
\`\`\`bash
grep -i "error" $OUTPUT_DIR/*.log
\`\`\`

3. View key differences:
\`\`\`bash
jq '.comparisons[] | select(.differences.range_ratio > 2 or .differences.range_ratio < 0.5)' $OUTPUT_DIR/cache_comparison/cache_comparison_results.json
\`\`\`
EOF

echo -e "${GREEN}Summary report created: $OUTPUT_DIR/ANALYSIS_SUMMARY.md${NC}"