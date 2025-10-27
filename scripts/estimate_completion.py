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
import itertools
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
    """Find all running train.py or train_ra_mla.py processes and their working directories."""
    train_processes = []

    try:
        for proc in psutil.process_iter(
            ["pid", "name", "cmdline", "cwd", "create_time"]
        ):
            try:
                # Check if this is a python process running train.py or train_ra_mla.py
                cmdline = proc.info["cmdline"]
                if cmdline and "python" in cmdline[0].lower():
                    for arg in cmdline:
                        if "train.py" in arg or "train_ra_mla.py" in arg:
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


def parse_test_config(config_path):
    """Parse config.txt to extract test matrix settings."""
    config = {}

    if not config_path.exists():
        return config

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse CONFIG_KEY=value or CONFIG_KEY="value"
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove CONFIG_ prefix
                if key.startswith("CONFIG_"):
                    key = key[7:]
                # Remove quotes from value
                value = value.strip('"')
                config[key] = value

    return config


def generate_expected_tests(config):
    """Generate expected test combinations from config."""
    combinations = []

    # Determine models
    models = []
    if config.get("MODEL_SELECT_RESNET50") == "y":
        models = ["resnet50"]
    elif config.get("MODEL_SELECT_RESNET18") == "y":
        models = ["resnet18"]
    elif config.get("MODEL_SELECT_LENET5") == "y":
        models = ["lenet5"]
    elif config.get("MODEL_SELECT_GPT2") == "y":
        models = ["gpt2"]

    # Determine optimizers
    optimizers = []
    if config.get("OPTIMIZER_MODE_MULTIPLE") == "y":
        if config.get("OPTIMIZER_ENABLE_SGD") == "y":
            optimizers.append("sgd")
        if config.get("OPTIMIZER_ENABLE_ADAM") == "y":
            optimizers.append("adam")
        if config.get("OPTIMIZER_ENABLE_ADAMW") == "y":
            optimizers.append("adamw")
        if config.get("OPTIMIZER_ENABLE_ADAMWADV") == "y":
            optimizers.append("adamwadv")
        if config.get("OPTIMIZER_ENABLE_ADAMWSPAM") == "y":
            optimizers.append("adamwspam")
        if config.get("OPTIMIZER_ENABLE_ADAMWPRUNE") == "y":
            optimizers.append("adamwprune")

    # Determine pruning methods
    pruning_methods = []
    if config.get("PRUNING_MODE_MULTIPLE") == "y":
        if config.get("PRUNING_ENABLE_MOVEMENT") == "y":
            pruning_methods.append("movement")
        if config.get("PRUNING_ENABLE_STATE") == "y":
            pruning_methods.append("state")
        if config.get("PRUNING_ENABLE_MAGNITUDE") == "y":
            pruning_methods.append("magnitude")

    # Determine sparsity levels
    sparsity_levels = []
    if config.get("SPARSITY_ENABLE_50") == "y" or config.get("TEST_SPARSITY_50") == "y":
        sparsity_levels.append("50")
    if config.get("SPARSITY_ENABLE_70") == "y" or config.get("TEST_SPARSITY_70") == "y":
        sparsity_levels.append("70")
    if config.get("SPARSITY_ENABLE_90") == "y" or config.get("TEST_SPARSITY_90") == "y":
        sparsity_levels.append("90")
    if config.get("SPARSITY_ENABLE_95") == "y" or config.get("TEST_SPARSITY_95") == "y":
        sparsity_levels.append("95")
    if config.get("SPARSITY_ENABLE_99") == "y" or config.get("TEST_SPARSITY_99") == "y":
        sparsity_levels.append("99")

    # Check for AdamWPrune variants (bitter0, bitter1, bitter2)
    adamwprune_variants = []
    if "adamwprune" in optimizers:
        if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER0") == "y":
            adamwprune_variants.append("bitter0")
        if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER1") == "y":
            adamwprune_variants.append("bitter1")
        if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER2") == "y":
            adamwprune_variants.append("bitter2")
        # Default to bitter0 if no variants specified
        if not adamwprune_variants:
            adamwprune_variants = ["bitter0"]

    # Generate all valid combinations
    for model, optimizer, pruning in itertools.product(
        models, optimizers, pruning_methods
    ):
        # Skip invalid combinations
        if optimizer == "adamwprune" and pruning not in ["none", "state"]:
            continue  # AdamWPrune only works with state-based pruning
        if optimizer != "adamwprune" and pruning == "state":
            continue  # State-based pruning is only for AdamWPrune

        # For no pruning, sparsity is always 0
        if pruning == "none":
            if optimizer == "adamwprune":
                # Generate a test for each variant
                for variant in adamwprune_variants:
                    test_name = f"{model}_{optimizer}_{variant}_none"
                    combinations.append(test_name)
            else:
                test_name = f"{model}_{optimizer}_none"
                combinations.append(test_name)
        else:
            # For pruning methods, add each sparsity level
            for sparsity in sparsity_levels:
                if optimizer == "adamwprune":
                    # Generate a test for each variant
                    for variant in adamwprune_variants:
                        test_name = (
                            f"{model}_{optimizer}_{variant}_{pruning}_{sparsity}"
                        )
                        combinations.append(test_name)
                else:
                    test_name = f"{model}_{optimizer}_{pruning}_{sparsity}"
                    combinations.append(test_name)

    return combinations


def analyze_test_directory(matrix_dir):
    """Analyze a test matrix directory for completion status."""
    matrix_path = Path(matrix_dir)

    results = {
        "directory": str(matrix_path),
        "complete_tests": [],
        "incomplete_tests": [],
        "failed_tests": [],
        "in_progress_tests": [],
        "not_started_tests": [],
        "total_tests": 0,
        "average_time": 0,
        "expected_tests": [],
    }

    # Parse config to understand total expected tests
    config_file = matrix_path / "config.txt"
    if config_file.exists():
        config = parse_test_config(config_file)
        results["expected_tests"] = generate_expected_tests(config)
        results["total_tests"] = len(results["expected_tests"])

    # Track which expected tests we've seen
    seen_tests = set()

    # Analyze each test subdirectory
    for test_dir in matrix_path.iterdir():
        if not test_dir.is_dir() or test_dir.name in ["graphs", "logs", ".git"]:
            continue

        seen_tests.add(test_dir.name)
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
            # Check if test failed
            is_failed = False
            if output_log.exists():
                try:
                    with open(output_log, "r") as f:
                        content = f.read()
                        if (
                            "ERROR - Training failed" in content
                            or "AttributeError" in content
                            or "Traceback" in content
                        ):
                            is_failed = True
                            # Try to extract error message
                            error_msg = ""
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if "AttributeError" in line or "ERROR" in line:
                                    error_msg = line.strip()
                                    break
                            results["failed_tests"].append(
                                {"name": test_dir.name, "error": error_msg}
                            )
                except Exception:
                    pass

            if is_failed:
                continue

            # Check if it's currently running
            is_running = False

            # First check if this test has actually started (has output.log)
            if not output_log.exists():
                results["incomplete_tests"].append({"name": test_dir.name})
                continue

            for proc in find_running_train_processes():
                # Check if process is running for this SPECIFIC test
                # Look for exact match in command line args for the output directory
                cmdline_str = " ".join(proc["cmdline"])
                # Need exact match to avoid matching "resnet50_adam_movement_9" with "resnet50_adam_movement_90"
                # Look for the full path or the test name in the json output path
                if (
                    str(test_dir.absolute()) in cmdline_str
                    or f"/{test_dir.name}/training_metrics.json" in cmdline_str
                ):
                    is_running = True
                    # Estimate progress based on output.log
                    current_epoch = 0
                    total_epochs = 10  # default (more reasonable than 100)
                    current_iter = 0
                    total_iters = 0

                    if output_log.exists():
                        try:
                            with open(output_log, "r") as f:
                                content = f.read()

                                # Check if this is GPT-2 (iteration-based)
                                iter_matches = re.findall(r"Iter\s+(\d+)\s+\|", content)
                                if iter_matches:
                                    # GPT-2 uses iterations, not epochs
                                    current_iter = int(iter_matches[-1])

                                    # Look for max_iters in command
                                    max_iter_matches = re.findall(
                                        r"--max-iters\s+(\d+)", content
                                    )
                                    if max_iter_matches:
                                        total_iters = int(max_iter_matches[0])
                                    else:
                                        # Default for GPT-2 train.py when --max-iters not specified
                                        total_iters = 10000

                                    # Look for epochs in command (GPT-2 specific)
                                    epoch_arg_matches = re.findall(
                                        r"--epochs\s+(\d+)", content
                                    )
                                    if epoch_arg_matches:
                                        total_epochs = int(epoch_arg_matches[0])
                                        # Estimate progress based on iterations
                                        # Rough estimate: assume linear progress through epochs
                                        if total_iters > 0:
                                            current_epoch = (
                                                current_iter / total_iters
                                            ) * total_epochs
                                        else:
                                            # If no max_iters, estimate from training patterns
                                            # GPT-2 typically does ~50k iters for full training
                                            estimated_iters_per_epoch = 5000
                                            current_epoch = (
                                                current_iter / estimated_iters_per_epoch
                                            )
                                            if current_epoch > total_epochs:
                                                current_epoch = total_epochs * 0.9
                                else:
                                    # CNN models use epochs
                                    epoch_matches = re.findall(
                                        r"Epoch[:\s]+(\d+)", content
                                    )
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
                            "current_iter": current_iter,
                            "total_iters": total_iters,
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

    # Find tests that haven't been started yet
    for expected_test in results["expected_tests"]:
        if expected_test not in seen_tests:
            results["not_started_tests"].append({"name": expected_test})

    return results


def estimate_completion_time(analysis):
    """Estimate time to complete all tests."""
    estimates = {}

    # Get average time per test
    avg_time = analysis["average_time"]
    has_completed_tests = avg_time > 0
    if avg_time == 0:
        # No completed tests to base estimate on - use a default estimate
        # Typical test takes around 30-60 minutes for ResNet50 on CIFAR100
        # GPT-2 training typically takes 20-30 hours
        avg_time = 45 * 60  # 45 minutes as default for CNN models

    # Estimate for in-progress tests
    for test in analysis["in_progress_tests"]:
        # Use iterations if available (more accurate for GPT-2), otherwise epochs
        using_iterations = (
            test.get("total_iters", 0) > 0 and test.get("current_iter", 0) > 0
        )
        if using_iterations:
            progress = test["current_iter"] / test["total_iters"]
        elif test["total_epochs"] > 0:
            progress = test["current_epoch"] / test["total_epochs"]
        else:
            progress = 0

        elapsed = datetime.now().timestamp() - test["start_time"]

        # For early progress (< 5%), use average time instead of extrapolation
        # which can be wildly inaccurate - BUT only if we have completed tests
        if progress < 0.05 and has_completed_tests:
            # Use historical average, adjusted for progress already made
            remaining = avg_time * (1 - progress)
        elif progress > 0:
            # For later progress, use actual progress-based estimation
            total_expected = elapsed / progress
            remaining = total_expected - elapsed

            # Sanity check: if estimated total time is way off from average,
            # blend the estimates (this handles variations in early epochs)
            # SKIP blending for iteration-based training (GPT-2) when no completed tests,
            # as the default avg_time is for CNN models and will be completely wrong
            should_blend = (
                has_completed_tests  # Only blend if we have real historical data
                and not using_iterations  # Don't blend for iteration-based (more accurate)
                and total_expected > avg_time * 2
            )

            if should_blend:
                # Blend with historical average if estimate seems too high
                weight = min(
                    progress * 2, 1.0
                )  # Give more weight to actual as progress increases
                remaining = (remaining * weight) + (
                    avg_time * (1 - progress) * (1 - weight)
                )
        else:
            continue

        estimates[test["name"]] = {
            "remaining_time": max(0, remaining),  # Never negative
            "progress_percent": progress * 100,
            "current_epoch": test["current_epoch"],
            "total_epochs": test["total_epochs"],
            "current_iter": test.get("current_iter", 0),
            "total_iters": test.get("total_iters", 0),
            "using_average": progress < 0.05,  # Flag if using historical average
        }

    # Add time for incomplete tests (will need full run)
    incomplete_count = len(analysis["incomplete_tests"])
    incomplete_time = incomplete_count * avg_time

    # Add time for failed tests (will need to be fixed and re-run)
    failed_count = len(analysis.get("failed_tests", []))
    failed_time = failed_count * avg_time

    # Add time for not-started tests (with variant awareness)
    not_started_count = len(analysis["not_started_tests"])
    not_started_time = 0

    # Check for bitter2 variants that need extra time
    for test in analysis["not_started_tests"]:
        test_name = test.get("name", "")
        if "adamwprune_bitter2" in test_name:
            # Bitter2 runs 21% more iterations
            not_started_time += avg_time * 1.21
        else:
            not_started_time += avg_time

    # Calculate total remaining
    in_progress_remaining = sum(
        e["remaining_time"] for e in estimates.values() if e["remaining_time"] > 0
    )
    total_remaining = (
        in_progress_remaining + incomplete_time + failed_time + not_started_time
    )

    return {
        "in_progress_estimates": estimates,
        "incomplete_count": incomplete_count,
        "incomplete_time": incomplete_time,
        "failed_count": failed_count,
        "failed_time": failed_time,
        "not_started_count": not_started_count,
        "not_started_time": not_started_time,
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
        # Check both cwd and command line args for the directory path
        active_dirs = []
        for matrix_dir in matrix_dirs:
            for proc in running_procs:
                cmdline_str = " ".join(proc["cmdline"])
                # Check if this process is working on this specific test matrix
                if str(matrix_dir) in proc["cwd"] or str(matrix_dir) in cmdline_str:
                    active_dirs.append(matrix_dir)
                    break

        if active_dirs:
            matrix_dirs = active_dirs
        elif matrix_dirs:
            # Sort directories by timestamp in name and use the most recent
            # Format is test_matrix_results_YYYYMMDD_HHMMSS
            sorted_dirs = sorted(matrix_dirs, key=lambda x: x.name, reverse=True)
            matrix_dirs = [sorted_dirs[0]]
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
        total_expected = len(analysis.get("expected_tests", []))
        if total_expected > 0:
            print(f"Expected tests: {total_expected}")
        print(f"Complete tests: {len(analysis['complete_tests'])}")

        # Show failed tests prominently
        if len(analysis.get("failed_tests", [])) > 0:
            print(f"\n⚠️  FAILED TESTS: {len(analysis['failed_tests'])}")
            for failed in analysis["failed_tests"]:
                error_msg = failed.get("error", "Unknown error")
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                print(f"   ✗ {failed['name']}: {error_msg}")
            print()

        print(f"In-progress tests: {len(analysis['in_progress_tests'])}")
        print(f"Incomplete tests: {len(analysis['incomplete_tests'])}")
        print(f"Not started tests: {len(analysis['not_started_tests'])}")

        if analysis["average_time"] > 0:
            print(f"Average test time: {format_time(analysis['average_time'])}")

        if estimates:
            print("\nIn-Progress Tests:")
            for test_name, est in estimates["in_progress_estimates"].items():
                print(f"  {test_name}:")
                # Show iterations for GPT-2, epochs for CNN models
                if est.get("current_iter", 0) > 0:
                    print(
                        f"    Progress: {est['progress_percent']:.1f}% (Iter {est.get('current_iter', 0)}, Epoch {est['current_epoch']:.1f}/{est['total_epochs']})"
                    )
                else:
                    print(
                        f"    Progress: {est['progress_percent']:.1f}% (Epoch {est['current_epoch']}/{est['total_epochs']})"
                    )
                time_str = format_time(est["remaining_time"])
                if est.get("using_average", False):
                    time_str += " (based on average)"
                print(f"    Time remaining: {time_str}")

            if estimates.get("failed_count", 0) > 0:
                print(
                    f"\n⚠️  Failed tests to fix and re-run: {estimates['failed_count']}"
                )
                print(
                    f"Estimated time for failed (after fixing): {format_time(estimates.get('failed_time', 0))}"
                )

            if estimates["incomplete_count"] > 0:
                print(f"\nIncomplete tests to re-run: {estimates['incomplete_count']}")
                print(
                    f"Estimated time for incomplete: {format_time(estimates['incomplete_time'])}"
                )

            if estimates.get("not_started_count", 0) > 0:
                print(f"Not started tests to run: {estimates['not_started_count']}")
                print(
                    f"Estimated time for not started: {format_time(estimates.get('not_started_time', 0))}"
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
