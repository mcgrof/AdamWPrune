#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate summary report from existing test matrix results."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def regenerate_summary(results_dir):
    """Regenerate summary report from all_results.json."""
    json_file = os.path.join(results_dir, "all_results.json")

    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found")
        return False

    # Load results
    with open(json_file, "r") as f:
        results = json.load(f)

    # Create new summary report
    report_file = os.path.join(results_dir, "summary_report.txt")

    with open(report_file, "w") as f:
        f.write("Test Matrix Summary Report (Regenerated)\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"From: {results_dir}\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        f.write(f"Total tests: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")

        # Results table
        f.write("Results Table:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Test ID':<40} {'Accuracy':<12} {'Sparsity':<12} {'Time (s)':<12} {'Status':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for result in results:
            test_id = result.get("test_id", "unknown")
            accuracy = (
                result.get("final_accuracy", result.get("best_accuracy", 0.0))
                if result.get("success")
                else 0.0
            )
            sparsity = (
                result.get("final_sparsity", 0.0) if result.get("success") else 0.0
            )
            elapsed = result.get("elapsed_time", 0.0)
            status = "✓ Success" if result.get("success") else "✗ Failed"

            f.write(
                f"{test_id:<40} {accuracy:<12.4f} {sparsity:<12.4f} {elapsed:<12.2f} {status:<10}\n"
            )

        f.write("-" * 80 + "\n\n")

        # Best performers
        if successful > 0:
            f.write("Best Performers:\n")
            f.write("-" * 80 + "\n")

            # Sort by accuracy
            success_results = [r for r in results if r.get("success", False)]
            success_results.sort(
                key=lambda x: x.get("final_accuracy", x.get("best_accuracy", 0)),
                reverse=True,
            )

            f.write("Top 20 by Accuracy:\n")
            for i, result in enumerate(success_results[:20], 1):
                test_id = result.get("test_id", "unknown")
                acc = result.get("final_accuracy", result.get("best_accuracy", 0))
                f.write(f"{i}. {test_id}: {acc:.4f}\n")

            f.write("\n")

            # Group by optimizer and find best
            by_optimizer = {}
            for r in success_results:
                opt = r.get("optimizer", "unknown")
                if opt not in by_optimizer:
                    by_optimizer[opt] = []
                by_optimizer[opt].append(r)

            f.write("Best by Optimizer:\n")
            for opt in sorted(by_optimizer.keys()):
                best = max(
                    by_optimizer[opt],
                    key=lambda x: x.get("final_accuracy", x.get("best_accuracy", 0)),
                )
                test_id = best.get("test_id", "unknown")
                acc = best.get("final_accuracy", best.get("best_accuracy", 0))
                f.write(f"  {opt}: {test_id} ({acc:.4f})\n")

            f.write("\n")

            # Memory efficiency analysis
            f.write("Memory Efficiency Analysis:\n")
            f.write("-" * 80 + "\n")
            f.write("Note: Memory shown as multiplier of weight tensor size\n")
            f.write("Training memory = weights + optimizer states + pruning overhead\n")
            f.write(
                "Inference memory = active weights only (benefits from sparsity)\n\n"
            )

            # Memory usage relative to weight tensor size (W)
            # SGD: W (weights only, no optimizer state)
            # Adam/AdamW: W + 2W (weights + exp_avg + exp_avg_sq)
            # Movement pruning adds: W (scores) + W (initial_weights) + 0.03W (masks)
            # State pruning adds: 0.03W (boolean mask only)

            # Base optimizer memory (as multiplier of weight size)
            optimizer_memory = {
                "sgd": 1.0,  # Just weights
                "adam": 3.0,  # Weights + exp_avg + exp_avg_sq
                "adamw": 3.0,  # Same as Adam
                "adamwadv": 3.1,  # Adam + AMSGrad max_exp_avg_sq
                "adamwspam": 3.1,  # Adam + spike tracking
                "adamwprune": 3.0,  # Same as Adam (reuses states for pruning)
            }

            # Additional memory for pruning methods (as multiplier of weight size)
            pruning_overhead = {
                "none": 0.0,
                "magnitude": 0.03,  # Boolean mask (1 bit per weight ≈ 0.03x float32)
                "movement": 2.03,  # scores (1x) + initial_weights (1x) + mask (0.03x)
                "state": 0.03,  # Boolean mask only for AdamWPrune
            }

            # Calculate memory efficiency scores
            memory_results = []
            for r in success_results:
                opt = r.get("optimizer", "unknown")
                test_id = r.get("test_id", "unknown")

                # Parse pruning method from test_id
                if "adamwprune" in test_id and (
                    "state" in test_id or "none" not in test_id
                ):
                    # AdamWPrune with any sparsity uses state pruning
                    pruning = "state" if "none" not in test_id else "none"
                elif "movement" in test_id:
                    pruning = "movement"
                elif "magnitude" in test_id:
                    pruning = "magnitude"
                else:
                    pruning = "none"
                acc = r.get("final_accuracy", r.get("best_accuracy", 0))
                sparsity = r.get("final_sparsity", 0.0)

                # Calculate total training memory
                base_memory = optimizer_memory.get(opt, 1.0)
                pruning_mem = pruning_overhead.get(pruning, 0.0)
                total_memory = base_memory + pruning_mem

                # Inference memory: only active weights matter
                # With sparsity, we can use sparse storage formats
                # Conservative estimate: sparse format achieves ~50% of theoretical compression
                inference_memory = 1.0 * (1 - sparsity * 0.5)

                # Memory efficiency = accuracy / training memory usage
                efficiency = acc / total_memory if total_memory > 0 else 0

                memory_results.append(
                    {
                        "test_id": test_id,
                        "optimizer": opt,
                        "pruning": pruning,
                        "accuracy": acc,
                        "sparsity": sparsity,
                        "training_memory": total_memory,
                        "inference_memory": inference_memory,
                        "efficiency": efficiency,
                    }
                )

            # Sort by efficiency
            memory_results.sort(key=lambda x: x["efficiency"], reverse=True)

            f.write("Top 20 Most Memory-Efficient (Accuracy/Training Memory Ratio):\n")
            for i, result in enumerate(memory_results[:20], 1):
                f.write(f"{i}. {result['test_id']}\n")
                f.write(f"   Accuracy: {result['accuracy']:.2f}%, ")
                f.write(f"Training Memory: {result['training_memory']:.2f}x weights, ")
                f.write(f"Efficiency Score: {result['efficiency']:.2f}\n")

            f.write("\n")

            # Best for inference (considering sparsity)
            memory_results.sort(key=lambda x: x["inference_memory"])
            f.write("Top 20 Lowest Inference Memory (with sparsity benefits):\n")
            for i, result in enumerate(memory_results[:20], 1):
                f.write(f"{i}. {result['test_id']}\n")
                f.write(f"   Accuracy: {result['accuracy']:.2f}%, ")
                f.write(f"Sparsity: {result['sparsity']:.1%}, ")
                f.write(
                    f"Inference Memory: {result['inference_memory']:.2f}x weights\n"
                )

            f.write("\n")

            # Special highlight for AdamWPrune
            adamwprune_results = [
                r for r in memory_results if r["optimizer"] == "adamwprune"
            ]
            if adamwprune_results:
                f.write(
                    "AdamWPrune Performance (State-Based Pruning with Minimal Overhead):\n"
                )
                f.write("-" * 80 + "\n")
                f.write("Memory breakdown for AdamWPrune:\n")
                f.write("  - Weights: 1.0x\n")
                f.write("  - Adam states (exp_avg, exp_avg_sq): 2.0x\n")
                f.write("  - Pruning overhead (boolean mask): 0.03x\n")
                f.write("  - Total training memory: 3.03x weights\n")
                f.write("\nComparison with other pruning methods:\n")
                f.write(
                    "  - SGD + movement pruning: 3.03x (1x weights + 2.03x pruning)\n"
                )
                f.write(
                    "  - AdamW + movement pruning: 5.03x (3x Adam + 2.03x pruning)\n"
                )
                f.write("  - AdamWPrune: 3.03x (3x Adam + 0.03x mask)\n")
                f.write("\nTop AdamWPrune configurations:\n")
                for result in sorted(
                    adamwprune_results, key=lambda x: x["accuracy"], reverse=True
                )[:3]:
                    f.write(f"  {result['test_id']}:\n")
                    f.write(f"    Accuracy: {result['accuracy']:.2f}%, ")
                    f.write(f"Sparsity: {result['sparsity']:.1%}\n")
                    f.write(
                        f"    Training Memory: {result['training_memory']:.2f}x weights, "
                    )
                    f.write(
                        f"Inference Memory: {result['inference_memory']:.2f}x weights\n"
                    )

    print(f"Summary report regenerated: {report_file}")
    return True


def main():
    if len(sys.argv) < 2:
        # Find the most recent test_matrix_results directory
        pattern = "test_matrix_results_*"
        dirs = sorted(Path(".").glob(pattern))
        if dirs:
            results_dir = str(dirs[-1])
            print(f"Using most recent results: {results_dir}")
        else:
            print("Error: No test_matrix_results_* directories found")
            print("Usage: python regenerate_summary.py [results_dir]")
            sys.exit(1)
    else:
        results_dir = sys.argv[1]

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    if regenerate_summary(results_dir):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
