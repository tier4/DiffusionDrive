#!/bin/bash

# Simple script to run individual NaN analysis tools with all options

# Configuration
export NAVSIM_CACHE="/workspace/navsim_workspace/exp/training_cache"
export B2D_CACHE="/workspace/navsim_workspace/cache/bench2drive_Base_cache"

echo "=== Bench2Drive NaN Analysis Commands ==="
echo ""
echo "1. Cache comparison (with all plot options):"
echo "python3 scripts/debug/analyze_cache_comparison.py \\"
echo "    --navsim-cache \$NAVSIM_CACHE \\"
echo "    --b2d-cache \$B2D_CACHE \\"
echo "    --max-samples 10000 \\"
echo "    --output-dir cache_analysis_output \\"
echo "    --plot-fields features/trajectory/x features/trajectory/y features/trajectory/heading \\"
echo "                  features/status_feature targets/trajectory/x targets/trajectory/y \\"
echo "                  targets/trajectory/heading targets/agent_states"
echo ""
echo "2. Dataloader analysis:"
echo "python3 scripts/debug/analyze_dual_dataloaders.py \\"
echo "    --navsim-cache \$NAVSIM_CACHE \\"
echo "    --b2d-cache \$B2D_CACHE \\"
echo "    --navsim-config navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml \\"
echo "    --b2d-config navsim/planning/script/config/common/agent/diffusiondrive_agent_extended.yaml \\"
echo "    --batch-size 8 \\"
echo "    --max-batches 100 \\"
echo "    --output dataloader_analysis_report.json"
echo ""
echo "3. Feature transformation analysis:"
echo "python3 scripts/debug/analyze_feature_transforms.py \\"
echo "    --navsim-cache \$NAVSIM_CACHE \\"
echo "    --b2d-cache \$B2D_CACHE \\"
echo "    --num-samples 1000 \\"
echo "    --output-dir transform_analysis_output"
echo ""
echo "4. Pipeline differences (with model test):"
echo "python3 scripts/debug/analyze_pipeline_differences.py \\"
echo "    --navsim-cache \$NAVSIM_CACHE \\"
echo "    --b2d-cache \$B2D_CACHE \\"
echo "    --navsim-config navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml \\"
echo "    --b2d-config navsim/planning/script/config/common/agent/diffusiondrive_agent_extended.yaml \\"
echo "    --output pipeline_differences_report.json \\"
echo "    --test-model"
echo ""
echo "=== Quick Run Commands ==="
echo ""
echo "# Minimal test run (small samples)"
echo "python3 scripts/debug/analyze_cache_comparison.py --navsim-cache $NAVSIM_CACHE --b2d-cache $B2D_CACHE --max-samples 10"
echo ""
echo "# Full analysis with all plots"
echo "python3 scripts/debug/analyze_cache_comparison.py --navsim-cache $NAVSIM_CACHE --b2d-cache $B2D_CACHE --max-samples 10000 --output-dir full_analysis --plot-fields features/trajectory/x features/trajectory/y features/trajectory/heading features/status_feature targets/trajectory/x targets/trajectory/y targets/trajectory/heading targets/agent_states"
echo ""

# Uncomment to run all at once with full options
# python3 scripts/debug/analyze_cache_comparison.py --navsim-cache $NAVSIM_CACHE --b2d-cache $B2D_CACHE --max-samples 10000 --output-dir cache_analysis --plot-fields features/trajectory/x features/trajectory/y features/trajectory/heading features/status_feature targets/trajectory/x targets/trajectory/y targets/trajectory/heading targets/agent_states
# python3 scripts/debug/analyze_dual_dataloaders.py --navsim-cache $NAVSIM_CACHE --b2d-cache $B2D_CACHE --batch-size 8 --max-batches 100 --output dataloader_analysis.json
# python3 scripts/debug/analyze_feature_transforms.py --navsim-cache $NAVSIM_CACHE --b2d-cache $B2D_CACHE --num-samples 1000 --output-dir transform_analysis
# python3 scripts/debug/analyze_pipeline_differences.py --navsim-cache $NAVSIM_CACHE --b2d-cache $B2D_CACHE --output pipeline_differences.json --test-model