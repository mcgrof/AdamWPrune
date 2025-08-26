#!/bin/bash
# Convenience script to re-run only AdamWPrune tests in an existing test matrix results directory

if [ $# -lt 1 ]; then
    echo "Usage: $0 <existing_results_directory> [additional_options]"
    echo ""
    echo "Example:"
    echo "  $0 test_matrix_results_20250826_181029"
    echo "  $0 test_matrix_results_20250826_181029 --dry-run"
    echo ""
    echo "This will re-run all AdamWPrune tests and update the results in the specified directory."
    exit 1
fi

RESULTS_DIR=$1
shift  # Remove first argument, pass rest to the script

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory '$RESULTS_DIR' does not exist"
    exit 1
fi

echo "Re-running AdamWPrune tests in: $RESULTS_DIR"
echo ""

# Run the test matrix with the adamwprune filter
python3 scripts/run_test_matrix.py \
    --rerun-dir "$RESULTS_DIR" \
    --filter-optimizer adamwprune \
    "$@"
