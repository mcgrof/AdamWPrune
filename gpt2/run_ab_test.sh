#!/bin/bash
# Run A/B test comparing AdamWSpam with momentum pruning vs AdamWPrune with state pruning

set -e

# Create output directory
OUTPUT_DIR="test_matrix_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running GPT-2 A/B Test"
echo "====================="
echo "Output directory: $OUTPUT_DIR"
echo

# Test 1: AdamWSpam with momentum pruning at 50%
echo "[1/2] Running AdamWSpam with momentum pruning at 50%..."
python3 scripts/train_with_monitoring.py \
    --model gpt2 \
    --optimizer adamwspam \
    --pruning-method momentum \
    --target-sparsity 0.5 \
    --config-name "gpt2_adamwspam_momentum_50" \
    --output-dir "$OUTPUT_DIR/gpt2_adamwspam_momentum_50" \
    --json-output

# Test 2: AdamWPrune with state pruning at 50% (using AdamWSpam as base)
echo "[2/2] Running AdamWPrune with state pruning at 50%..."
python3 scripts/train_with_monitoring.py \
    --model gpt2 \
    --optimizer adamwprune \
    --pruning-method state \
    --target-sparsity 0.5 \
    --config-name "gpt2_adamwprune_state_50" \
    --output-dir "$OUTPUT_DIR/gpt2_adamwprune_state_50" \
    --json-output

# Compare results
echo
echo "Comparing results..."
python3 gpt2/compare_ab_results.py "$OUTPUT_DIR"

# Generate graphs if available
if [ -f "scripts/generate_comparison_graphs.py" ]; then
    echo
    echo "Generating comparison graphs..."
    python3 scripts/generate_comparison_graphs.py "$OUTPUT_DIR"
fi

echo
echo "A/B test complete! Results in: $OUTPUT_DIR"
