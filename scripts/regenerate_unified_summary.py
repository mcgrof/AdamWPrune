#!/usr/bin/env python3
"""
Regenerate unified summary report that handles both CNN (accuracy) and GPT-2 (perplexity) metrics.
"""

import json
import os
import sys
import math
from datetime import datetime
from pathlib import Path

# Import the unified metrics extractor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_metrics import extract_metrics, format_metric


def load_gpu_stats(test_dir):
    """Load GPU statistics from the test directory."""
    # Find GPU stats file
    gpu_files = list(Path(test_dir).glob("gpu_stats_*.json"))
    if not gpu_files:
        return None

    with open(gpu_files[0], "r") as f:
        data = json.load(f)

    if not data:
        return None

    # Calculate statistics
    memory_values = [entry.get("memory_used", 0) for entry in data]
    if memory_values:
        return {
            "mean": sum(memory_values) / len(memory_values),
            "max": max(memory_values),
            "min": min(memory_values),
        }
    return None


def regenerate_summary(results_dir):
    """Regenerate summary report with unified metrics handling."""

    # Scan for all test directories
    test_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith(".")
    ]

    results = []
    for test_id in sorted(test_dirs):
        test_path = os.path.join(results_dir, test_id)

        # Extract metrics
        metrics = extract_metrics(test_path)
        if metrics:
            metrics["test_id"] = test_id

            # Load GPU stats
            gpu_stats = load_gpu_stats(test_path)
            if gpu_stats:
                metrics["gpu_memory_mean"] = gpu_stats["mean"]
                metrics["gpu_memory_max"] = gpu_stats["max"]

            # Parse test configuration from ID
            parts = test_id.split("_")
            if len(parts) >= 2:
                metrics["model"] = parts[0]
                metrics["optimizer"] = parts[1]
                if len(parts) >= 3:
                    metrics["pruning"] = parts[2]
                else:
                    metrics["pruning"] = "none"
                if len(parts) >= 4:
                    metrics["sparsity_target"] = float(parts[3]) / 100
                else:
                    metrics["sparsity_target"] = 0.0

            results.append(metrics)

    # Save all results
    all_results_file = os.path.join(results_dir, "all_results_unified.json")
    with open(all_results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Create summary report
    report_file = os.path.join(results_dir, "summary_report_unified.txt")

    with open(report_file, "w") as f:
        f.write("Unified Test Matrix Summary Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"From: {results_dir}\n")
        f.write("=" * 120 + "\n\n")

        # Summary statistics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        f.write(f"Total tests: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        # Group by model type
        gpt2_results = [r for r in results if r.get("model") == "gpt2"]
        cnn_results = [r for r in results if r.get("model") != "gpt2"]

        # GPT-2 Results Table
        if gpt2_results:
            f.write("GPT-2 Language Model Results:\n")
            f.write("-" * 120 + "\n")
            f.write(
                f"{'Test ID':<40} {'PPL (↓)':<12} {'ΔPPL':<10} {'Sparsity':<10} "
                f"{'Latency@512':<15} {'GPU (MB)':<12} {'Status':<10}\n"
            )
            f.write("-" * 120 + "\n")

            for r in sorted(
                gpt2_results, key=lambda x: x.get("best_metric", float("inf"))
            ):
                test_id = r["test_id"]
                ppl = r.get("best_metric", float("inf"))
                delta_ppl = r.get("delta_ppl", 0.0)
                sparsity = r.get("final_sparsity", 0.0) * 100

                # Latency info
                lat_p50 = r.get("latency_seq512_p50", -1)
                if lat_p50 > 0:
                    latency_str = f"{lat_p50:.1f}ms"
                else:
                    latency_str = "N/A"

                # GPU memory
                gpu_mem = r.get("gpu_memory_max", 0) or r.get(
                    "gpu_memory_reserved_mb", 0
                )
                if gpu_mem > 0:
                    gpu_str = f"{gpu_mem:.1f}"
                else:
                    gpu_str = "N/A"

                status = "✓" if r.get("success") else "✗"

                f.write(
                    f"{test_id:<40} {format_metric(ppl, 'perplexity'):<12} "
                    f"{delta_ppl:+10.2f} {sparsity:9.1f}% "
                    f"{latency_str:<15} {gpu_str:<12} {status:<10}\n"
                )

            # Best performers
            f.write("\nBest GPT-2 Models (by perplexity, lower is better):\n")
            best_gpt2 = sorted(
                [r for r in gpt2_results if r.get("success")],
                key=lambda x: x.get("best_metric", float("inf")),
            )[:5]
            for i, r in enumerate(best_gpt2, 1):
                f.write(
                    f"{i}. {r['test_id']}: PPL={format_metric(r['best_metric'], 'perplexity')}, "
                    f"ΔPPL={r.get('delta_ppl', 0):+.2f}\n"
                )
            f.write("\n")

        # CNN Results Table
        if cnn_results:
            f.write("CNN Model Results:\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Test ID':<40} {'Accuracy (↑)':<12} {'Sparsity':<10} "
                f"{'GPU (MB)':<12} {'Time (s)':<10} {'Status':<10}\n"
            )
            f.write("-" * 100 + "\n")

            for r in sorted(cnn_results, key=lambda x: -x.get("best_metric", 0)):
                test_id = r["test_id"]
                acc = r.get("best_metric", 0.0)
                sparsity = r.get("final_sparsity", 0.0) * 100

                # GPU memory
                gpu_mem = r.get("gpu_memory_max", 0)
                if gpu_mem > 0:
                    gpu_str = f"{gpu_mem:.1f}"
                else:
                    gpu_str = "N/A"

                # Training time
                time_s = r.get("total_time", 0)
                if time_s > 0:
                    time_str = f"{time_s:.1f}"
                else:
                    time_str = "N/A"

                status = "✓" if r.get("success") else "✗"

                f.write(
                    f"{test_id:<40} {format_metric(acc, 'accuracy'):<12} "
                    f"{sparsity:9.1f}% {gpu_str:<12} {time_str:<10} {status:<10}\n"
                )

            # Best performers
            f.write("\nBest CNN Models (by accuracy, higher is better):\n")
            best_cnn = sorted(
                [r for r in cnn_results if r.get("success")],
                key=lambda x: -x.get("best_metric", 0),
            )[:5]
            for i, r in enumerate(best_cnn, 1):
                f.write(
                    f"{i}. {r['test_id']}: {format_metric(r['best_metric'], 'accuracy')}\n"
                )

        # Overall analysis
        f.write("\n" + "=" * 120 + "\n")
        f.write("Analysis Summary:\n")
        f.write("-" * 120 + "\n")

        # Group by optimizer
        optimizers = {}
        for r in results:
            if r.get("success"):
                opt = r.get("optimizer", "unknown")
                if opt not in optimizers:
                    optimizers[opt] = []
                optimizers[opt].append(r)

        f.write("\nPerformance by Optimizer:\n")
        for opt, opt_results in sorted(optimizers.items()):
            f.write(f"\n{opt.upper()}:\n")

            # Separate by model type
            opt_gpt2 = [r for r in opt_results if r.get("model") == "gpt2"]
            opt_cnn = [r for r in opt_results if r.get("model") != "gpt2"]

            if opt_gpt2:
                best = min(opt_gpt2, key=lambda x: x.get("best_metric", float("inf")))
                f.write(
                    f"  Best GPT-2: {best['test_id']} "
                    f"(PPL={format_metric(best['best_metric'], 'perplexity')})\n"
                )

            if opt_cnn:
                best = max(opt_cnn, key=lambda x: x.get("best_metric", 0))
                f.write(
                    f"  Best CNN: {best['test_id']} "
                    f"({format_metric(best['best_metric'], 'accuracy')})\n"
                )

    print(f"Summary report saved to: {report_file}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: regenerate_unified_summary.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found")
        sys.exit(1)

    success = regenerate_summary(results_dir)
    sys.exit(0 if success else 1)
