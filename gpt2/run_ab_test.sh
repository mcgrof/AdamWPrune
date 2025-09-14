#!/bin/bash
# Run A/B test comparing AdamWSpam with momentum pruning vs AdamWPrune with state pruning
# This bypasses the test matrix framework to run exactly 2 tests

set -e

# Load the base configuration
echo "Loading base configuration..."
make defconfig-gpt2-ab-test

# Create output directory
OUTPUT_DIR="test_matrix_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running GPT-2 A/B Test"
echo "====================="
echo "Output directory: $OUTPUT_DIR"
echo
echo "Tests to run:"
echo "1. AdamWSpam with momentum pruning at 50%"
echo "2. AdamWPrune with state pruning at 50%"
echo

# Test 1: AdamWSpam with momentum pruning at 50%
echo "[1/2] Running AdamWSpam with momentum pruning at 50%..."
TEST_DIR="$OUTPUT_DIR/gpt2_adamwspam_momentum_50"
mkdir -p "$TEST_DIR"

python3 gpt2/train.py \
    --optimizer adamwspam \
    --pruning-method momentum \
    --target-sparsity 0.5 \
    --json-output \
    2>&1 | tee "$TEST_DIR/training.log"

# Save metrics
if [ -f "training_metrics.json" ]; then
    mv training_metrics.json "$TEST_DIR/"
fi

echo "[1/2] Complete!"
echo

# Test 2: AdamWPrune with state pruning at 50% (using AdamWSpam as base)
echo "[2/2] Running AdamWPrune with state pruning at 50%..."
TEST_DIR="$OUTPUT_DIR/gpt2_adamwprune_state_50"
mkdir -p "$TEST_DIR"

python3 gpt2/train.py \
    --optimizer adamwprune \
    --pruning-method state \
    --target-sparsity 0.5 \
    --json-output \
    2>&1 | tee "$TEST_DIR/training.log"

# Save metrics
if [ -f "training_metrics.json" ]; then
    mv training_metrics.json "$TEST_DIR/"
fi

echo "[2/2] Complete!"
echo

# Compare results
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
echo
echo "To view detailed results:"
echo "  - Logs: $OUTPUT_DIR/*/training.log"
echo "  - Metrics: $OUTPUT_DIR/*/training_metrics.json"
echo "  - Graphs: $OUTPUT_DIR/graphs/ (if generated)"
