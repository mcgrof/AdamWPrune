#!/usr/bin/env python3
"""
Generate consolidated GPU memory summary from all available data.
Includes test matrix results and standalone runs.
"""

import json
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime


def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return None


def extract_gpu_stats(data):
    """Extract GPU memory statistics from data, supporting both single and multi-GPU formats."""
    samples = None

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", [])

    if not samples:
        return None

    memory_values = []
    per_gpu_data = {}
    has_multi_gpu = False

    for s in samples:
        # Check for multi-GPU aggregate data first
        if "aggregate_stats" in s:
            memory_values.append(s["aggregate_stats"]["total_memory_used"])
            has_multi_gpu = True

            # Extract per-GPU breakdown
            if "multi_gpu_data" in s:
                for gpu_data in s["multi_gpu_data"]:
                    gpu_idx = gpu_data["gpu_index"]
                    if gpu_idx not in per_gpu_data:
                        per_gpu_data[gpu_idx] = []
                    per_gpu_data[gpu_idx].append(gpu_data["memory_used"])

        # Fall back to single GPU data
        elif "memory_used" in s:
            memory_values.append(s["memory_used"])
        elif "memory_mb" in s:
            memory_values.append(s["memory_mb"])
        elif "memory_used_mb" in s:
            memory_values.append(s["memory_used_mb"])

    if not memory_values:
        return None

    result = {
        "mean": np.mean(memory_values),
        "max": max(memory_values),
        "min": min(memory_values),
        "std": np.std(memory_values),
        "samples": len(memory_values),
        "multi_gpu": has_multi_gpu,
    }

    # Add per-GPU statistics if available
    if per_gpu_data:
        result["per_gpu_stats"] = {}
        for gpu_idx, gpu_memory_vals in per_gpu_data.items():
            result["per_gpu_stats"][gpu_idx] = {
                "mean": np.mean(gpu_memory_vals),
                "max": max(gpu_memory_vals),
                "min": min(gpu_memory_vals),
                "std": np.std(gpu_memory_vals),
                "samples": len(gpu_memory_vals),
            }

    return result


def main():
    """Generate consolidated GPU memory summary."""

    print("=" * 70)
    print("CONSOLIDATED GPU MEMORY SUMMARY")
    print("Including all available GPU monitoring data")
    print("=" * 70)

    # Collect all GPU data
    all_gpu_data = {}

    # Find all GPU monitoring files (including multi-GPU)
    gpu_files = glob("**/gpu_*.json", recursive=True)
    # Remove multi-GPU files from main list to avoid double processing
    gpu_files = [f for f in gpu_files if "_multi_gpu.json" not in f]

    # Also find multi-GPU files
    multi_gpu_files = glob("**/gpu_*_multi_gpu.json", recursive=True)

    # Process both regular and multi-GPU files
    for gpu_file in gpu_files + multi_gpu_files:
        path = Path(gpu_file)

        # Determine optimizer and phase
        filename = path.stem
        parent = path.parent.name

        optimizer = None
        phase = None
        sparsity = None

        # Parse filename patterns
        if "gpu_training_" in filename:
            phase = "training"
            parts = filename.replace("gpu_training_", "").split("_")
            if parts:
                optimizer = parts[0]
                if len(parts) > 2:
                    sparsity = parts[2]
        elif "gpu_inference_" in filename:
            phase = "inference"
            optimizer = "adamwprune"  # Our inference data is from AdamWPrune
        elif "gpu_stats_" in filename:
            phase = "training"
            # Extract from parent directory
            if "resnet18_" in parent:
                parts = parent.split("_")
                if len(parts) > 1:
                    optimizer = parts[1]
                    if len(parts) > 3:
                        sparsity = parts[3]

        if optimizer and phase:
            # Load and process data
            data = load_json(gpu_file)
            stats = extract_gpu_stats(data)

            if stats:
                key = f"{optimizer}_{phase}"
                # Keep the best (lowest memory) if multiple runs
                if (
                    key not in all_gpu_data
                    or stats["mean"] < all_gpu_data[key]["stats"]["mean"]
                ):
                    all_gpu_data[key] = {
                        "optimizer": optimizer,
                        "phase": phase,
                        "sparsity": sparsity,
                        "stats": stats,
                        "file": str(gpu_file),
                    }

    # Organize by optimizer
    optimizers = {}
    for key, data in all_gpu_data.items():
        opt = data["optimizer"]
        if opt not in optimizers:
            optimizers[opt] = {}
        optimizers[opt][data["phase"]] = data

    # Print consolidated report
    print("\nGPU MEMORY BY OPTIMIZER:")
    print("-" * 70)

    # Sort by training memory if available
    sorted_opts = sorted(
        optimizers.keys(),
        key=lambda x: optimizers[x]
        .get("training", {})
        .get("stats", {})
        .get("mean", float("inf")),
    )

    for opt in sorted_opts:
        data = optimizers[opt]
        print(f"\n{opt.upper()}:")

        if "training" in data:
            stats = data["training"]["stats"]
            sparsity = data["training"].get("sparsity", "unknown")
            print(f"  Training (sparsity target: {sparsity}%):")
            print(f"    Mean:    {stats['mean']:7.1f} MB")
            print(f"    Max:     {stats['max']:7.1f} MB")
            print(f"    Std Dev: {stats['std']:7.1f} MB")
            print(f"    Samples: {stats['samples']}")

        if "inference" in data:
            stats = data["inference"]["stats"]
            print(f"  Inference:")
            print(f"    Mean:    {stats['mean']:7.1f} MB")
            print(f"    Max:     {stats['max']:7.1f} MB")
            print(f"    Std Dev: {stats['std']:7.1f} MB")
            print(f"    Samples: {stats['samples']}")

            # Calculate reduction if both phases available
            if "training" in data:
                train_mean = data["training"]["stats"]["mean"]
                infer_mean = data["inference"]["stats"]["mean"]
                reduction = train_mean - infer_mean
                reduction_pct = (reduction / train_mean) * 100
                print(f"  Memory Reduction: {reduction:.1f} MB ({reduction_pct:.1f}%)")

    # Overall comparison
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS:")
    print("-" * 70)

    # Training comparison
    training_data = [
        (opt, optimizers[opt]["training"]["stats"]["mean"])
        for opt in optimizers
        if "training" in optimizers[opt]
    ]
    training_data.sort(key=lambda x: x[1])

    if training_data:
        print("\nTraining Memory Ranking (Lower is Better):")
        for i, (opt, mem) in enumerate(training_data, 1):
            print(f"  {i}. {opt.upper():12s}: {mem:7.1f} MB")

        best = training_data[0]
        worst = training_data[-1]
        if len(training_data) > 1:
            savings = worst[1] - best[1]
            savings_pct = (savings / worst[1]) * 100
            print(f"\nBest vs Worst:")
            print(
                f"  {best[0].upper()} saves {savings:.1f} MB ({savings_pct:.1f}%) vs {worst[0].upper()}"
            )

    # Inference data if available
    inference_data = [
        (opt, optimizers[opt]["inference"]["stats"]["mean"])
        for opt in optimizers
        if "inference" in optimizers[opt]
    ]

    if inference_data:
        print("\nInference Memory:")
        for opt, mem in inference_data:
            print(f"  {opt.upper():12s}: {mem:7.1f} MB")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("-" * 70)

    if "adamwprune" in optimizers:
        adamwprune_train = (
            optimizers["adamwprune"].get("training", {}).get("stats", {}).get("mean", 0)
        )
        if adamwprune_train and training_data:
            other_avg = np.mean(
                [mem for opt, mem in training_data if opt != "adamwprune"]
            )
            savings = other_avg - adamwprune_train
            savings_pct = (savings / other_avg) * 100
            print(f"\n1. AdamWPrune Training Efficiency:")
            print(
                f"   Uses {adamwprune_train:.1f} MB vs {other_avg:.1f} MB average of others"
            )
            print(f"   Saves {savings:.1f} MB ({savings_pct:.1f}%) on average")

        if "inference" in optimizers["adamwprune"]:
            print(f"\n2. AdamWPrune Inference:")
            infer_stats = optimizers["adamwprune"]["inference"]["stats"]
            print(f"   Uses only {infer_stats['mean']:.1f} MB during inference")
            if adamwprune_train:
                reduction = adamwprune_train - infer_stats["mean"]
                reduction_pct = (reduction / adamwprune_train) * 100
                print(
                    f"   {reduction:.1f} MB reduction ({reduction_pct:.1f}%) from training"
                )

    print("\n3. Overall Memory Efficiency Champion:")
    if training_data:
        print(
            f"   {training_data[0][0].upper()} with {training_data[0][1]:.1f} MB during training"
        )

    # Multi-GPU analysis if available
    has_multi_gpu_data = any(
        data.get("stats", {}).get("multi_gpu", False)
        for opt_data in optimizers.values()
        for data in opt_data.values()
    )

    if has_multi_gpu_data:
        print("\n" + "=" * 70)
        print("MULTI-GPU ANALYSIS (4x A10G):")
        print("-" * 70)

        for opt in sorted_opts:
            data = optimizers[opt]
            if "training" in data:
                stats = data["training"]["stats"]
                if stats.get("multi_gpu", False) and "per_gpu_stats" in stats:
                    print(f"\n{opt.upper()} Training - Per-GPU Breakdown:")
                    per_gpu = stats["per_gpu_stats"]

                    # Calculate load balance
                    gpu_means = [per_gpu[str(i)]["mean"] for i in range(len(per_gpu)) if str(i) in per_gpu]
                    if gpu_means:
                        avg_memory = np.mean(gpu_means)
                        std_memory = np.std(gpu_means)
                        cv = (std_memory / avg_memory) * 100 if avg_memory > 0 else 0

                        for gpu_idx in sorted(per_gpu.keys(), key=int):
                            gpu_stats = per_gpu[gpu_idx]
                            print(f"    GPU {gpu_idx}: {gpu_stats['mean']:7.1f} MB (±{gpu_stats['std']:5.1f})")

                        print(f"    Total:    {stats['mean']:7.1f} MB")
                        print(f"    Balance:  CV = {cv:5.1f}% ({'Excellent' if cv < 5 else 'Good' if cv < 10 else 'Fair' if cv < 20 else 'Poor'})")

        print("\nLoad Balance Legend:")
        print("  Excellent: CV < 5%  (Very well balanced)")
        print("  Good:      CV < 10% (Well balanced)")
        print("  Fair:      CV < 20% (Acceptable)")
        print("  Poor:      CV ≥ 20% (Imbalanced - investigate)")

    print("\n" + "=" * 70)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
