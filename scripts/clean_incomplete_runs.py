#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Clean incomplete training runs from a test matrix results directory.

This script identifies incomplete training runs by checking for:
1. "Training with monitoring completed successfully" in output.log
2. Presence of a PNG file for gpu_stats

Incomplete runs are removed to allow for clean continuation.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import json


def is_run_complete(test_dir):
    """Check if a training run is complete."""
    test_path = Path(test_dir)

    # Check for output.log
    output_log = test_path / "output.log"
    if not output_log.exists():
        return False

    # Check for completion message in output.log
    try:
        with open(output_log, "r") as f:
            content = f.read()
            if "Training with monitoring completed successfully!" not in content:
                return False
    except Exception as e:
        print(f"Error reading {output_log}: {e}")
        return False

    # Check for PNG file
    png_files = list(test_path.glob("*.png"))
    if not png_files:
        return False

    # Check for training_metrics.json (additional validation)
    metrics_file = test_path / "training_metrics.json"
    if not metrics_file.exists():
        return False

    return True


def is_run_failed(test_dir):
    """Check if a test run failed with an error."""
    test_path = Path(test_dir)
    output_log = test_path / "output.log"

    if not output_log.exists():
        return False

    try:
        with open(output_log, "r") as f:
            content = f.read()
            # Check for common failure indicators
            if any(indicator in content for indicator in [
                "ERROR - Training failed",
                "AttributeError:",
                "Traceback (most recent call last):",
                "RuntimeError:",
                "ValueError:",
                "KeyError:",
                "ImportError:",
                "ModuleNotFoundError:"
            ]):
                return True
    except Exception:
        pass

    return False


def find_incomplete_runs(matrix_dir):
    """Find all incomplete runs in a test matrix directory."""
    matrix_path = Path(matrix_dir)
    if not matrix_path.exists():
        print(f"Error: Directory {matrix_dir} does not exist")
        sys.exit(1)

    incomplete_runs = []
    complete_runs = []
    failed_runs = []

    # Iterate through all subdirectories
    for test_dir in matrix_path.iterdir():
        if not test_dir.is_dir():
            continue

        # Skip special directories
        if test_dir.name in ["graphs", "logs", ".git"]:
            continue

        # Check if this test run is complete
        if is_run_complete(test_dir):
            complete_runs.append(test_dir)
        elif is_run_failed(test_dir):
            failed_runs.append(test_dir)
        else:
            incomplete_runs.append(test_dir)

    return complete_runs, incomplete_runs, failed_runs


def remove_incomplete_runs(incomplete_runs, dry_run=False):
    """Remove incomplete training runs."""
    if not incomplete_runs:
        print("No incomplete runs found.")
        return

    print(f"\nFound {len(incomplete_runs)} incomplete run(s):")
    for run_dir in incomplete_runs:
        print(f"  - {run_dir.name}")

    if dry_run:
        print("\nDry run mode - no files will be deleted.")
        return

    # Ask for confirmation
    print(f"\nAbout to remove {len(incomplete_runs)} incomplete run(s).")
    response = input("Continue? (y/N): ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    # Remove incomplete runs
    for run_dir in incomplete_runs:
        try:
            shutil.rmtree(run_dir)
            print(f"Removed: {run_dir.name}")
        except Exception as e:
            print(f"Error removing {run_dir}: {e}")


def get_test_configuration(matrix_dir):
    """Extract test configuration from the matrix directory."""
    matrix_path = Path(matrix_dir)

    # Look for config.txt or config.yaml
    config_file = matrix_path / "config.txt"
    if not config_file.exists():
        config_file = matrix_path / "config.yaml"

    if config_file.exists():
        return config_file

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Clean incomplete training runs from test matrix results"
    )
    parser.add_argument("matrix_dir", help="Path to test matrix results directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing",
    )
    parser.add_argument(
        "--list-complete", action="store_true", help="Also list complete runs"
    )
    parser.add_argument("--json-output", help="Output results to JSON file")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also remove failed runs (runs with errors)"
    )

    args = parser.parse_args()

    # Find incomplete runs
    complete_runs, incomplete_runs, failed_runs = find_incomplete_runs(args.matrix_dir)

    print(f"Test matrix directory: {args.matrix_dir}")
    print(f"Total runs: {len(complete_runs) + len(incomplete_runs) + len(failed_runs)}")
    print(f"Complete runs: {len(complete_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Incomplete runs: {len(incomplete_runs)}")

    if args.list_complete and complete_runs:
        print(f"\nComplete runs ({len(complete_runs)}):")
        for run_dir in complete_runs:
            print(f"  ✓ {run_dir.name}")

    # List failed runs if any
    if failed_runs:
        print(f"\nFailed runs ({len(failed_runs)}):")
        for run_dir in failed_runs:
            print(f"  ✗ {run_dir.name}")

    # Determine what to remove
    runs_to_remove = incomplete_runs.copy()
    if args.include_failed:
        runs_to_remove.extend(failed_runs)

    # Save to JSON if requested
    if args.json_output:
        results = {
            "matrix_dir": str(args.matrix_dir),
            "complete_runs": [str(r.name) for r in complete_runs],
            "incomplete_runs": [str(r.name) for r in incomplete_runs],
            "failed_runs": [str(r.name) for r in failed_runs],
            "total_runs": len(complete_runs) + len(incomplete_runs) + len(failed_runs),
        }
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.json_output}")

    # Remove incomplete/failed runs
    if runs_to_remove:
        if args.include_failed:
            print(f"\nWill remove {len(runs_to_remove)} run(s) (incomplete + failed)")
        remove_incomplete_runs(runs_to_remove, dry_run=args.dry_run)

    # Show what tests need to be run to complete the matrix
    if incomplete_runs and not args.dry_run:
        print(
            f"\n{len(incomplete_runs)} test(s) removed and need to be re-run to complete the test matrix."
        )


if __name__ == "__main__":
    main()
