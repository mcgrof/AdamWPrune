# Continuing Interrupted Test Runs

The AdamWPrune test framework includes robust support for continuing test matrices that were interrupted due to system crashes, power failures, or manual termination. This feature ensures you don't lose progress on long-running experiments.

## Quick Start

If your test run was interrupted, simply run:

```bash
make continue
```

This will automatically:
1. Find your latest test matrix directory
2. Identify incomplete runs
3. Clean up incomplete runs after confirmation
4. Re-run the incomplete tests
5. Continue with all remaining tests from the original test plan

If you only want to re-run incomplete tests without adding new ones:

```bash
make continue-incomplete
```

## How It Works

### Identifying Incomplete Runs

A training run is considered complete when it has:
- `output.log` containing "Training with monitoring completed successfully!"
- GPU statistics PNG file generated
- `training_metrics.json` file present

Any run missing these markers is considered incomplete.

### The Continuation Process

When you run `make continue` or use the `--continue-dir` flag, the system:

1. **Scans the test directory** to identify complete and incomplete runs
2. **Shows a continuation plan** including:
   - Number of complete runs
   - Number of incomplete runs to remove
   - List of tests that need to be run
   - Time estimate based on completed runs
3. **Asks for confirmation** before removing incomplete runs
4. **Removes incomplete runs** to ensure a clean state
5. **Continues with remaining tests** using the same configuration

## Usage Examples

### Basic Continuation

Complete the entire test matrix (re-run incomplete + run remaining tests):

```bash
make continue
```

### Re-run Only Incomplete Tests

Re-run only the tests that were started but didn't complete:

```bash
make continue-incomplete
```

### Continue Specific Directory

Continue a specific test matrix directory:

```bash
# Re-run only incomplete tests
python3 scripts/run_test_matrix.py --continue-dir test_matrix_results_20250908_121537 --incomplete-only

# Complete entire matrix
python3 scripts/run_test_matrix.py --continue-dir test_matrix_results_20250908_121537
```

### Check Incomplete Runs Without Removing

See what runs are incomplete without removing them:

```bash
python3 scripts/clean_incomplete_runs.py test_matrix_results_20250908_121537 --dry-run
```

### List All Runs Status

See both complete and incomplete runs:

```bash
python3 scripts/clean_incomplete_runs.py test_matrix_results_20250908_121537 --list-complete
```

## Example Workflow

Here's a typical scenario after a system crash:

```bash
$ make continue
Found latest test matrix directory: test_matrix_results_20250908_121537
Checking for incomplete runs...
Test matrix directory: test_matrix_results_20250908_121537
Total runs: 6
Complete runs: 2
Incomplete runs: 1

Test Matrix Continuation Plan:
  Complete runs: 2
  Incomplete runs to remove: 1
  Tests to run: 4
  Estimated time: 1.2h (based on 2 completed runs)

Tests to be run:
  - resnet50_adamw_movement_70
  - resnet50_adamwadv_movement_70
  - resnet50_adamwspam_movement_70
  - resnet50_adamwprune_state_70

============================================================
Remove incomplete runs and continue? (y/N): y

Cleaning incomplete runs...
  Removed: resnet50_adamw_movement_70

Continuing with 4 remaining tests...
```

## Time Estimation

The continuation feature provides time estimates based on completed runs:

- **Per-test estimates**: Shows estimated time for each individual test
- **Total remaining time**: Calculates total time for all remaining tests
- **Updates dynamically**: Estimates improve as more tests complete

Time estimates appear in the format:
- Seconds: `45s`
- Minutes: `5.2m`
- Hours: `1.5h`

## Advanced Features

### JSON Output

Export incomplete run information for scripting:

```bash
python3 scripts/clean_incomplete_runs.py test_matrix_results_20250908_121537 \
    --json-output incomplete_runs.json
```

### Integration with Parallel Execution

Continuation works seamlessly with parallel execution. The system will continue with the same parallelism settings from the original run.

### Preserving Results

The continuation feature preserves all completed results:
- Existing `all_results.json` is updated, not replaced
- Complete runs are never touched
- Summary reports include both original and continued runs

## Troubleshooting

### No Test Matrix Found

If you see "No test_matrix_results_* directories found", you need to start a test matrix first:

```bash
make test-matrix
```

### All Runs Complete

If there are no incomplete runs, the system will report:

```
No incomplete runs found. Test matrix is complete.
```

### Missing Configuration

The continuation feature requires the original `config.txt` or `config.yaml` in the test directory. If missing, you'll see an error.

## Technical Details

### File Structure

A complete test run directory contains:
```
resnet50_sgd_movement_70/
├── output.log                    # Training output with completion message
├── training_metrics.json         # Performance metrics
├── gpu_stats_*.json              # GPU monitoring data
├── gpu_stats_*_plot.png          # GPU performance visualization
└── gpu_stats_*_summary.txt       # GPU stats summary
```

### Clean Script

The `scripts/clean_incomplete_runs.py` script can be used standalone:

```bash
# Check status
./scripts/clean_incomplete_runs.py test_matrix_results_20250908_121537 --dry-run

# Remove incomplete runs with confirmation
./scripts/clean_incomplete_runs.py test_matrix_results_20250908_121537

# Export results
./scripts/clean_incomplete_runs.py test_matrix_results_20250908_121537 \
    --json-output status.json
```

### Continuation Flag

The `--continue-dir` flag in `scripts/run_test_matrix.py`:

```bash
python3 scripts/run_test_matrix.py --continue-dir test_matrix_results_20250908_121537
```

This flag:
- Loads the original configuration
- Identifies incomplete runs
- Calculates what needs to be re-run
- Provides time estimates
- Handles the complete workflow

## Best Practices

1. **Always check status first**: Use `--dry-run` to see what will be removed
2. **Review the plan**: Check the continuation plan before confirming
3. **Monitor estimates**: Time estimates improve with more completed runs
4. **Keep logs**: Incomplete run logs are deleted, so save them if needed for debugging
5. **Use parallel execution**: For faster completion, especially when continuing many tests

## See Also

- [Test Matrix Documentation](test-matrix.md)
- [Parallel Execution Guide](parallel.md)
- [Configuration Guide](config.md)
