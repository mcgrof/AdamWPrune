#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Convenience script for running parallel test matrix with optimal settings
# for high-memory GPUs like the AMD Radeon Pro W7900 (48GB)

set -e

# Default values
PARALLEL_JOBS=8
RESULTS_DIR=""
DRY_RUN=false
FILTER_OPTIMIZER=""

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run test matrix with parallel execution optimized for high-memory GPUs.

OPTIONS:
    -j, --jobs N         Number of parallel jobs (default: 8)
    -r, --rerun-dir DIR  Existing results directory to update
    -n, --dry-run        Show what would be run without executing
    -f, --filter OPT     Filter to specific optimizer(s) (comma-separated)
    -h, --help           Show this help message

EXAMPLES:
    $0                           # Run with 8 parallel jobs
    $0 -j 16                     # Run with 16 parallel jobs
    $0 -j 4 -f adamwprune        # Run only AdamWPrune tests with 4 jobs
    $0 -r test_matrix_results_20250826_181029  # Re-run failed tests

MEMORY RECOMMENDATIONS:
    GPU Memory    Recommended Jobs
    < 16GB        1-4 jobs
    24GB          4-8 jobs  
    48GB (W7900)  8-20 jobs
    > 48GB        16-50 jobs

Each LeNet-5 training job uses ~50-100MB GPU memory.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -r|--rerun-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--filter)
            FILTER_OPTIMIZER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "Parallel Test Matrix Execution"
echo "=============================="
echo "Jobs: $PARALLEL_JOBS"
echo "GPU Memory Monitoring: Enabled"

# Check if .config exists
if [[ ! -f .config ]]; then
    echo "Error: .config not found. Run 'make menuconfig' first."
    exit 1
fi

# Build command
CMD=(python3 scripts/run_test_matrix.py --parallel "$PARALLEL_JOBS")

if [[ -n "$RESULTS_DIR" ]]; then
    CMD+=(--rerun-dir "$RESULTS_DIR")
fi

if [[ "$DRY_RUN" == "true" ]]; then
    CMD+=(--dry-run)
fi

if [[ -n "$FILTER_OPTIMIZER" ]]; then
    CMD+=(--filter-optimizer "$FILTER_OPTIMIZER")
fi

echo "Command: ${CMD[*]}"
echo

# Execute
exec "${CMD[@]}"