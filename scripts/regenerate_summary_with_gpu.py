#!/usr/bin/env python3
"""
Regenerate summary report using real GPU memory measurements.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from glob import glob
import re


def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return None


def get_gpu_memory_stats(test_dir):
    """Extract GPU memory statistics from monitoring files."""
    gpu_stats = {}

    # Find GPU monitoring files ONLY in the specified test directory
    gpu_files = []
    
    if os.path.exists(test_dir):
        gpu_files.extend(
            glob(os.path.join(test_dir, "**/gpu_stats_*.json"), recursive=True)
        )
        gpu_files.extend(
            glob(os.path.join(test_dir, "**/gpu_training_*.json"), recursive=True)
        )
        gpu_files.extend(
            glob(
                os.path.join(test_dir, "**/gpu_inference_*.json"), recursive=True
            )
        )

    for gpu_file in gpu_files:
        # Extract test name from path
        parent_dir = Path(gpu_file).parent.name
        filename = Path(gpu_file).stem

        # Determine test name from filename or parent dir
        test_name = parent_dir
        if "gpu_training_" in filename:
            # Parse training file: gpu_training_adamwprune_state_70
            parts = filename.replace("gpu_training_", "").split("_")
            if len(parts) >= 3:
                test_name = f"resnet18_{parts[0]}_{parts[1]}_{parts[2]}"
        elif "gpu_stats_" in filename:
            # Already has full name in parent dir
            test_name = parent_dir

        # Load GPU data
        data = load_json(gpu_file)
        if not data:
            continue

        # Extract memory values
        samples = data if isinstance(data, list) else data.get("samples", [])
        if not samples:
            # Check if data has summary field directly
            if isinstance(data, dict) and "summary" in data:
                summary = data["summary"]
                if "mean_memory_mb" in summary:
                    gpu_stats[test_name] = {
                        "mean": summary.get("mean_memory_mb", 0),
                        "max": summary.get("max_memory_mb", 0),
                        "min": summary.get("min_memory_mb", 0),
                        "std": 0,
                    }
            continue

        memory_values = []
        for s in samples:
            if "memory_used" in s:
                val = s["memory_used"]
                # Filter out idle values (< 100 MB usually indicates idle)
                if val > 100:
                    memory_values.append(val)
            elif "memory_mb" in s:
                memory_values.append(s["memory_mb"])
            elif "memory_used_mb" in s:
                memory_values.append(s["memory_used_mb"])

        if memory_values:
            import numpy as np

            # Store under multiple possible keys for matching
            gpu_stats[test_name] = {
                "mean": np.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "std": np.std(memory_values),
            }

            # Also store under alternative keys for AdamWPrune
            if "adamwprune" in test_name.lower():
                # Store under both state and movement variants
                alt_name = test_name.replace("_state_", "_movement_")
                if alt_name != test_name:
                    gpu_stats[alt_name] = gpu_stats[test_name]
                alt_name = test_name.replace("_movement_", "_state_")
                if alt_name != test_name:
                    gpu_stats[alt_name] = gpu_stats[test_name]

    return gpu_stats


def regenerate_summary(results_dir, output_file="summary_report.txt"):
    """Regenerate summary report with real GPU memory data."""

    # Load all results
    all_results_file = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(all_results_file):
        print(f"Error: {all_results_file} not found")
        return False

    with open(all_results_file, "r") as f:
        all_results = json.load(f)

    # Get GPU memory stats
    gpu_stats = get_gpu_memory_stats(results_dir)

    # Process results
    test_results = []

    # Handle both dict and list formats
    if isinstance(all_results, list):
        # List format - construct test names from fields
        for result in all_results:
            if isinstance(result, dict):
                # Construct test ID from fields
                model = result.get("model", "unknown")
                optimizer = result.get("optimizer", "unknown")
                pruning = result.get("pruning_method", "none")
                sparsity = int(result.get("target_sparsity", 0) * 100)
                test_name = f"{model}_{optimizer}_{pruning}_{sparsity}"

                test_info = {
                    "test_id": test_name,
                    "accuracy": result.get(
                        "final_accuracy", result.get("test_accuracy", 0)
                    ),
                    "sparsity": result.get("final_sparsity", 0),
                    "time": result.get("total_time", 0),
                    "status": "✓ Success",
                    "epochs": result.get("epochs", 0),
                    "optimizer": optimizer,
                }

                # Add real GPU memory data if available
                # Try exact match first
                if test_name in gpu_stats:
                    test_info["gpu_memory_mean"] = gpu_stats[test_name]["mean"]
                    test_info["gpu_memory_max"] = gpu_stats[test_name]["max"]
                # For "none" pruning with _0 suffix, try without the suffix
                elif pruning == "none" and test_name.endswith("_0"):
                    alt_name = test_name[:-2]  # Remove "_0"
                    if alt_name in gpu_stats:
                        test_info["gpu_memory_mean"] = gpu_stats[alt_name]["mean"]
                        test_info["gpu_memory_max"] = gpu_stats[alt_name]["max"]
                # Also check for alternative naming (e.g., adamwprune state vs movement)
                elif optimizer == "adamwprune":
                    # Try alternative naming patterns
                    alt_names = [
                        f"{model}_{optimizer}_state_{sparsity}",
                        f"{model}_{optimizer}_movement_{sparsity}",
                        f"{optimizer}_state_{sparsity}",
                        f"{optimizer}_movement_{sparsity}",
                    ]
                    for alt_name in alt_names:
                        if alt_name in gpu_stats:
                            test_info["gpu_memory_mean"] = gpu_stats[alt_name]["mean"]
                            test_info["gpu_memory_max"] = gpu_stats[alt_name]["max"]
                            break

                test_results.append(test_info)
    else:
        # Dict format
        for test_name, result in all_results.items():
            if isinstance(result, dict) and "final_accuracy" in result:
                test_info = {
                    "test_id": test_name,
                    "accuracy": result.get("final_accuracy", 0),
                    "sparsity": result.get("final_sparsity", 0),
                    "time": result.get("total_time", 0),
                    "status": (
                        "✓ Success" if result.get("success", True) else "✗ Failed"
                    ),
                    "epochs": result.get("epochs_completed", 0),
                }

                # Add real GPU memory data if available
                # Try exact match first
                if test_name in gpu_stats:
                    test_info["gpu_memory_mean"] = gpu_stats[test_name]["mean"]
                    test_info["gpu_memory_max"] = gpu_stats[test_name]["max"]
                # For "none" pruning with _0 suffix, try without the suffix
                elif pruning == "none" and test_name.endswith("_0"):
                    alt_name = test_name[:-2]  # Remove "_0"
                    if alt_name in gpu_stats:
                        test_info["gpu_memory_mean"] = gpu_stats[alt_name]["mean"]
                        test_info["gpu_memory_max"] = gpu_stats[alt_name]["max"]
                # Also check for alternative naming (e.g., adamwprune state vs movement)
                elif optimizer == "adamwprune":
                    # Try alternative naming patterns
                    alt_names = [
                        f"{model}_{optimizer}_state_{sparsity}",
                        f"{model}_{optimizer}_movement_{sparsity}",
                        f"{optimizer}_state_{sparsity}",
                        f"{optimizer}_movement_{sparsity}",
                    ]
                    for alt_name in alt_names:
                        if alt_name in gpu_stats:
                            test_info["gpu_memory_mean"] = gpu_stats[alt_name]["mean"]
                            test_info["gpu_memory_max"] = gpu_stats[alt_name]["max"]
                            break

                # Extract optimizer info
                parts = test_name.split("_")
                if len(parts) >= 2:
                    test_info["optimizer"] = parts[1]

                test_results.append(test_info)

    # Sort by accuracy
    test_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Generate report
    with open(os.path.join(results_dir, output_file), "w") as f:
        f.write("Test Matrix Summary Report (With Real GPU Memory Data)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"From: {results_dir}\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        total_tests = len(test_results)
        successful_tests = sum(1 for t in test_results if "Success" in t["status"])
        failed_tests = total_tests - successful_tests

        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Successful: {successful_tests}\n")
        f.write(f"Failed: {failed_tests}\n\n")

        # Results table with GPU memory
        f.write("Results Table:\n")
        f.write("-" * 105 + "\n")
        f.write(
            f"{'Test ID':<40} {'Accuracy':>8} {'Sparsity':>8} {'GPU Mean (MiB)':>14} {'GPU Max (MiB)':>13} {'Status':<10}\n"
        )
        f.write("-" * 105 + "\n")

        for result in test_results:
            gpu_mean = (
                f"{result.get('gpu_memory_mean', 0):.1f}"
                if "gpu_memory_mean" in result
                else "N/A"
            )
            gpu_max = (
                f"{result.get('gpu_memory_max', 0):.1f}"
                if "gpu_memory_max" in result
                else "N/A"
            )

            f.write(
                f"{result['test_id']:<40} "
                f"{result['accuracy']:>8.4f} "
                f"{result['sparsity']:>8.4f} "
                f"{gpu_mean:>14} "
                f"{gpu_max:>13} "
                f"{result['status']:<10}\n"
            )

        f.write("-" * 105 + "\n\n")

        # Best performers by accuracy
        f.write("Best Performers:\n")
        f.write("-" * 80 + "\n")
        f.write("Top Results by Accuracy:\n")
        for i, result in enumerate(test_results[:10], 1):
            gpu_info = ""
            if "gpu_memory_mean" in result:
                gpu_info = f" (GPU: {result['gpu_memory_mean']:.1f} MB)"
            f.write(f"{i}. {result['test_id']}: {result['accuracy']:.4f}{gpu_info}\n")

        f.write("\n")

        # Best by optimizer (with GPU memory)
        optimizer_best = {}
        for result in test_results:
            opt = result.get("optimizer", "unknown")
            if (
                opt not in optimizer_best
                or result["accuracy"] > optimizer_best[opt]["accuracy"]
            ):
                optimizer_best[opt] = result

        f.write("Best by Optimizer:\n")
        for opt in sorted(optimizer_best.keys()):
            result = optimizer_best[opt]
            gpu_info = ""
            if "gpu_memory_mean" in result:
                gpu_info = f", GPU: {result['gpu_memory_mean']:.1f} MB"
            f.write(
                f"  {opt}: {result['test_id']} ({result['accuracy']:.4f}{gpu_info})\n"
            )

        f.write("\n")

        # GPU Memory Efficiency Analysis (using real data)
        f.write("GPU Memory Efficiency Analysis (Real Measurements):\n")
        f.write("-" * 80 + "\n")

        # Filter results with GPU data
        gpu_results = [r for r in test_results if "gpu_memory_mean" in r]

        if gpu_results:
            # Sort by GPU memory efficiency (accuracy per MB)
            for r in gpu_results:
                r["efficiency"] = r["accuracy"] / r["gpu_memory_mean"] * 100

            gpu_results.sort(key=lambda x: x["efficiency"], reverse=True)

            f.write("Most Memory-Efficient (Accuracy per 100MB GPU):\n")
            for i, result in enumerate(gpu_results[:10], 1):
                f.write(
                    f"{i}. {result['test_id']}\n"
                    f"   Accuracy: {result['accuracy']:.2f}%, "
                    f"GPU Memory: {result['gpu_memory_mean']:.1f} MB, "
                    f"Efficiency Score: {result['efficiency']:.2f}\n"
                )

            f.write("\n")

            # Sort by absolute GPU memory usage
            gpu_results.sort(key=lambda x: x["gpu_memory_mean"])

            f.write("Lowest GPU Memory Usage:\n")
            for i, result in enumerate(gpu_results[:10], 1):
                f.write(
                    f"{i}. {result['test_id']}\n"
                    f"   GPU Memory: {result['gpu_memory_mean']:.1f} MB (max: {result['gpu_memory_max']:.1f} MB), "
                    f"Accuracy: {result['accuracy']:.2f}%\n"
                )

            f.write("\n")

            # AdamWPrune specific analysis
            adamwprune_results = [
                r for r in gpu_results if "adamwprune" in r["test_id"].lower()
            ]
            if adamwprune_results:
                f.write("AdamWPrune Performance (Real GPU Measurements):\n")
                f.write("-" * 80 + "\n")

                for result in adamwprune_results:
                    f.write(f"Configuration: {result['test_id']}\n")
                    f.write(f"  Accuracy: {result['accuracy']:.2f}%\n")
                    f.write(f"  Sparsity achieved: {result['sparsity']:.1%}\n")
                    f.write(
                        f"  GPU Memory (mean): {result['gpu_memory_mean']:.1f} MB\n"
                    )
                    f.write(f"  GPU Memory (peak): {result['gpu_memory_max']:.1f} MB\n")

                    # Compare with other optimizers
                    other_opts = [
                        r
                        for r in gpu_results
                        if "adamwprune" not in r["test_id"].lower()
                    ]
                    if other_opts:
                        avg_other_memory = sum(
                            r["gpu_memory_mean"] for r in other_opts
                        ) / len(other_opts)
                        memory_savings = avg_other_memory - result["gpu_memory_mean"]
                        savings_pct = (memory_savings / avg_other_memory) * 100

                        f.write(
                            f"  Memory savings vs others: {memory_savings:.1f} MB ({savings_pct:.1f}%)\n"
                        )

                f.write("\n")

            # Overall GPU memory comparison
            f.write("GPU Memory Comparison (All Optimizers):\n")
            optimizer_memory = {}
            for result in gpu_results:
                opt = result.get("optimizer", "unknown")
                if opt not in optimizer_memory:
                    optimizer_memory[opt] = []
                optimizer_memory[opt].append(result["gpu_memory_mean"])

            import numpy as np

            for opt in sorted(optimizer_memory.keys()):
                memories = optimizer_memory[opt]
                mean_mem = np.mean(memories)
                f.write(
                    f"  {opt:12s}: {mean_mem:7.1f} MB (avg of {len(memories)} runs)\n"
                )
        else:
            f.write("No GPU memory data available. Run with GPU monitoring enabled.\n")

    print(f"Summary report regenerated: {os.path.join(results_dir, output_file)}")
    return True


def main():
    if len(sys.argv) < 2:
        # Try to find the most recent test results directory
        results_dirs = sorted(
            [d for d in os.listdir(".") if d.startswith("test_matrix_results_")],
            reverse=True,
        )
        if results_dirs:
            results_dir = results_dirs[0]
            print(f"Using most recent results: {results_dir}")
        else:
            print("Usage: python regenerate_summary_with_gpu.py <results_directory>")
            print("No test_matrix_results_* directories found in current directory")
            sys.exit(1)
    else:
        results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found")
        sys.exit(1)

    regenerate_summary(results_dir)


if __name__ == "__main__":
    main()
