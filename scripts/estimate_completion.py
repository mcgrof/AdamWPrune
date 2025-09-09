#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Estimate completion time for running test matrix.

This script:
1. Finds test matrix directories
2. Checks for running train.py processes
3. Analyzes completed and in-progress tests
4. Estimates time to completion
"""

import os
import sys
import json
import psutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import re
import subprocess


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "Unknown"
    elif seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        return f"{int(days)}d {int(hours)}h"


def find_running_train_processes():
    """Find all running train.py processes and their working directories."""
    train_processes = []

    try:
        for proc in psutil.process_iter(
            ["pid", "name", "cmdline", "cwd", "create_time"]
        ):
            try:
                # Check if this is a python process running train.py
                cmdline = proc.info["cmdline"]
                if cmdline and "python" in cmdline[0].lower():
                    for arg in cmdline:
                        if "train.py" in arg:
                            train_processes.append(
                                {
                                    "pid": proc.info["pid"],
                                    "cwd": proc.info["cwd"],
                                    "cmdline": cmdline,
                                    "start_time": proc.info["create_time"],
                                }
                            )
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"Error scanning processes: {e}")

    return train_processes


def find_test_matrix_dirs(base_dir="."):
    """Find all test_matrix_results_* directories."""
    base_path = Path(base_dir)
    matrix_dirs = sorted(base_path.glob("test_matrix_results_*"))
    return matrix_dirs


def analyze_test_directory(matrix_dir):
    """Analyze a test matrix directory for completion status."""
    matrix_path = Path(matrix_dir)

    results = {
        "directory": str(matrix_path),
        "complete_tests": [],
        "incomplete_tests": [],
        "in_progress_tests": [],
        "total_tests": 0,
        "average_time": 0,
    }

    # Check for config to understand total expected tests
    config_file = matrix_path / "config.txt"
    if not config_file.exists():
        config_file = matrix_path / "config.yaml"

    # Analyze each test subdirectory
    for test_dir in matrix_path.iterdir():
        if not test_dir.is_dir() or test_dir.name in ["graphs", "logs", ".git"]:
            continue

        output_log = test_dir / "output.log"
        metrics_file = test_dir / "training_metrics.json"

        # Check if test is complete
        is_complete = False
        elapsed_time = None

        if output_log.exists():
            try:
                with open(output_log, "r") as f:
                    content = f.read()
                    if "Training with monitoring completed successfully!" in content:
                        is_complete = True
                        # Try to get elapsed time from metrics
                        if metrics_file.exists():
                            with open(metrics_file, "r") as mf:
                                metrics = json.load(mf)
                                # Try different field names for elapsed time
                                elapsed_time = metrics.get("elapsed_time", 0)
                                if not elapsed_time:
                                    elapsed_time = metrics.get("total_time", 0)
            except Exception:
                pass

        if is_complete:
            results["complete_tests"].append(
                {"name": test_dir.name, "elapsed_time": elapsed_time}
            )
        else:
            # Check if it's currently running
            is_running = False
            for proc in find_running_train_processes():
                # Check if process is running for this test
                # Look in command line args for the output directory
                cmdline_str = " ".join(proc["cmdline"])
                if (
                    str(test_dir) in proc["cwd"]
                    or test_dir.name in proc["cwd"]
                    or str(test_dir) in cmdline_str
                    or test_dir.name in cmdline_str
                ):
                    is_running = True
                    # Estimate progress based on output.log
                    current_epoch = 0
                    total_epochs = 100  # default

                    if output_log.exists():
                        try:
                            with open(output_log, "r") as f:
                                content = f.read()
                                # Find epoch numbers
                                epoch_matches = re.findall(r"Epoch[:\s]+(\d+)", content)
                                if epoch_matches:
                                    current_epoch = int(epoch_matches[-1])
                                # Try to find total epochs
                                total_matches = re.findall(
                                    r"Epoch[:\s]+\d+/(\d+)", content
                                )
                                if total_matches:
                                    total_epochs = int(total_matches[0])
                        except Exception:
                            pass

                    results["in_progress_tests"].append(
                        {
                            "name": test_dir.name,
                            "pid": proc["pid"],
                            "current_epoch": current_epoch,
                            "total_epochs": total_epochs,
                            "start_time": proc["start_time"],
                        }
                    )
                    break

            if not is_running:
                results["incomplete_tests"].append({"name": test_dir.name})

    # Calculate average time from completed tests
    completed_times = [
        t["elapsed_time"]
        for t in results["complete_tests"]
        if t["elapsed_time"] is not None
    ]
    if completed_times:
        results["average_time"] = sum(completed_times) / len(completed_times)

    return results


def estimate_completion_time(analysis):
    """Estimate time to complete all tests."""
    estimates = {}

    # Get average time per test
    avg_time = analysis["average_time"]
    if avg_time == 0:
        # No completed tests to base estimate on
        return None

    # Estimate for in-progress tests
    for test in analysis["in_progress_tests"]:
        if test["total_epochs"] > 0:
            progress = test["current_epoch"] / test["total_epochs"]
            elapsed = datetime.now().timestamp() - test["start_time"]
            if progress > 0:
                total_expected = elapsed / progress
                remaining = total_expected - elapsed
                estimates[test["name"]] = {
                    "remaining_time": remaining,
                    "progress_percent": progress * 100,
                    "current_epoch": test["current_epoch"],
                    "total_epochs": test["total_epochs"],
                }

    # Add time for incomplete tests (will need full run)
    incomplete_count = len(analysis["incomplete_tests"])
    incomplete_time = incomplete_count * avg_time

    # Calculate total remaining
    in_progress_remaining = sum(
        e["remaining_time"] for e in estimates.values() if e["remaining_time"] > 0
    )
    total_remaining = in_progress_remaining + incomplete_time

    return {
        "in_progress_estimates": estimates,
        "incomplete_count": incomplete_count,
        "incomplete_time": incomplete_time,
        "total_remaining": total_remaining,
        "average_test_time": avg_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate completion time for running test matrix"
    )
    parser.add_argument(
        "--matrix-dir", help="Specific test matrix directory to analyze"
    )
    parser.add_argument(
        "--all", action="store_true", help="Analyze all test matrix directories"
    )
    parser.add_argument("--json-output", help="Output results to JSON file")

    args = parser.parse_args()

    # Find directories to analyze
    if args.matrix_dir:
        matrix_dirs = [Path(args.matrix_dir)]
    elif args.all:
        matrix_dirs = find_test_matrix_dirs()
    else:
        # Find latest with running process
        matrix_dirs = find_test_matrix_dirs()
        running_procs = find_running_train_processes()

        # Filter to directories with running processes
        active_dirs = []
        for matrix_dir in matrix_dirs:
            for proc in running_procs:
                if str(matrix_dir) in proc["cwd"]:
                    active_dirs.append(matrix_dir)
                    break

        if active_dirs:
            matrix_dirs = active_dirs
        elif matrix_dirs:
            # Use latest if no running processes
            matrix_dirs = [matrix_dirs[-1]]
        else:
            print("No test matrix directories found")
            sys.exit(1)

    # Analyze each directory
    all_results = {}
    for matrix_dir in matrix_dirs:
        if not matrix_dir.exists():
            print(f"Error: Directory {matrix_dir} does not exist")
            continue

        print(f"\nAnalyzing: {matrix_dir}")
        print("=" * 60)

        analysis = analyze_test_directory(matrix_dir)
        estimates = estimate_completion_time(analysis)

        # Display results
        print(f"Complete tests: {len(analysis['complete_tests'])}")
        print(f"In-progress tests: {len(analysis['in_progress_tests'])}")
        print(f"Incomplete tests: {len(analysis['incomplete_tests'])}")

        if analysis["average_time"] > 0:
            print(f"Average test time: {format_time(analysis['average_time'])}")

        if estimates:
            print("\nIn-Progress Tests:")
            for test_name, est in estimates["in_progress_estimates"].items():
                print(f"  {test_name}:")
                print(
                    f"    Progress: {est['progress_percent']:.1f}% (Epoch {est['current_epoch']}/{est['total_epochs']})"
                )
                print(f"    Time remaining: {format_time(est['remaining_time'])}")

            if estimates["incomplete_count"] > 0:
                print(f"\nIncomplete tests to re-run: {estimates['incomplete_count']}")
                print(
                    f"Estimated time for incomplete: {format_time(estimates['incomplete_time'])}"
                )

            print("\n" + "=" * 60)
            print(
                f"TOTAL TIME TO COMPLETION: {format_time(estimates['total_remaining'])}"
            )

            # Estimate completion time
            completion_time = datetime.now() + timedelta(
                seconds=estimates["total_remaining"]
            )
            print(
                f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            print("\nNo running tests or completed tests to base estimate on")

        all_results[str(matrix_dir)] = {"analysis": analysis, "estimates": estimates}

    # Save JSON output if requested
    if args.json_output:
        # Convert to JSON-serializable format
        output_data = {}
        for dir_name, data in all_results.items():
            output_data[dir_name] = {
                "complete_count": len(data["analysis"]["complete_tests"]),
                "in_progress_count": len(data["analysis"]["in_progress_tests"]),
                "incomplete_count": len(data["analysis"]["incomplete_tests"]),
                "average_time": data["analysis"]["average_time"],
            }
            if data["estimates"]:
                output_data[dir_name]["total_remaining_seconds"] = data["estimates"][
                    "total_remaining"
                ]
                output_data[dir_name]["estimated_completion"] = (
                    datetime.now()
                    + timedelta(seconds=data["estimates"]["total_remaining"])
                ).isoformat()

        with open(args.json_output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.json_output}")


if __name__ == "__main__":
    main()
