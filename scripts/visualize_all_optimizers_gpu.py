#!/usr/bin/env python3
"""
Generate GPU memory comparison graphs for all optimizers.
Creates separate graphs for training and inference phases.
"""

import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from glob import glob
import re


def load_gpu_data(json_file):
    """Load GPU monitoring data from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def extract_memory_timeline(data):
    """Extract memory usage timeline from GPU monitoring data."""
    # Handle different formats
    samples = None

    # Check if data is already a list (direct JSON array)
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        if "samples" in data:
            samples = data["samples"]
        elif "stats" in data:
            samples = data["stats"]

    if not samples:
        return None, None

    timestamps = []
    memory_used = []

    for i, s in enumerate(samples):
        # Get timestamp - use index if no timestamp field
        if "elapsed_seconds" in s:
            timestamps.append(s["elapsed_seconds"])
        elif "timestamp" in s:
            # If string timestamp, use index
            timestamps.append(i)
        elif "time" in s:
            timestamps.append(s["time"])
        else:
            timestamps.append(i)

        # Get memory (try different field names)
        if "memory_used" in s:
            # Memory in MB
            memory_used.append(s["memory_used"])
        elif "memory_mb" in s:
            memory_used.append(s["memory_mb"])
        elif "memory_used_mb" in s:
            memory_used.append(s["memory_used_mb"])
        elif "memory" in s:
            if isinstance(s["memory"], dict) and "used" in s["memory"]:
                memory_used.append(s["memory"]["used"])
            else:
                memory_used.append(s["memory"])
        elif "gpu_memory" in s:
            memory_used.append(s["gpu_memory"])

    if not memory_used:
        return None, None

    return timestamps, memory_used


def get_optimizer_name(filename):
    """Extract optimizer name from filename."""
    filename = Path(filename).stem

    # Patterns to try
    patterns = [
        r"gpu_(?:training|inference)_([^_]+)_",
        r"gpu_stats_resnet18_([^_]+)_",
        r"resnet18_([^_]+)_",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)

    # Fallback - extract optimizer from path
    path_parts = Path(filename).parts
    for part in path_parts:
        if "resnet18_" in part:
            match = re.search(r"resnet18_([^_]+)_", part)
            if match:
                return match.group(1)

    return "unknown"


def create_training_comparison(gpu_files, output_file):
    """Create comparison graph for training GPU memory usage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GPU Memory Usage During Training - All Optimizers",
        fontsize=16,
        fontweight="bold",
    )

    # Color map for optimizers
    optimizers = []
    colors = {}
    color_list = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Collect data from all files
    optimizer_data = {}

    for gpu_file in gpu_files:
        data = load_gpu_data(gpu_file)
        optimizer = get_optimizer_name(gpu_file)

        if optimizer not in colors:
            colors[optimizer] = color_list[len(colors) % len(color_list)]
            optimizers.append(optimizer)

        timestamps, memory = extract_memory_timeline(data)

        if timestamps and memory:
            optimizer_data[optimizer] = {
                "timestamps": timestamps,
                "memory": memory,
                "mean": np.mean(memory),
                "std": np.std(memory),
                "max": max(memory),
                "min": min(memory),
                "file": gpu_file,
            }

    if not optimizer_data:
        print("No valid GPU data found for training")
        return

    # Sort optimizers for consistent ordering
    optimizers = sorted(optimizer_data.keys())

    # Plot 1: Memory over time
    ax1 = axes[0, 0]
    for opt in optimizers:
        data = optimizer_data[opt]
        # Subsample if too many points
        if len(data["timestamps"]) > 500:
            indices = np.linspace(0, len(data["timestamps"]) - 1, 500, dtype=int)
            ts = [data["timestamps"][i] for i in indices]
            mem = [data["memory"][i] for i in indices]
        else:
            ts = data["timestamps"]
            mem = data["memory"]

        ax1.plot(
            ts, mem, label=opt.upper(), color=colors[opt], alpha=0.7, linewidth=1.5
        )

    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Memory Used (MB)", fontsize=10)
    ax1.set_title("Memory Usage Over Time", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean memory comparison
    ax2 = axes[0, 1]
    x = np.arange(len(optimizers))
    means = [optimizer_data[opt]["mean"] for opt in optimizers]
    stds = [optimizer_data[opt]["std"] for opt in optimizers]

    bars = ax2.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[colors[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + stds[i] + 20,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xlabel("Optimizer", fontsize=10)
    ax2.set_ylabel("Mean Memory (MB)", fontsize=10)
    ax2.set_title("Average Memory Usage", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([opt.upper() for opt in optimizers], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Max memory comparison
    ax3 = axes[1, 0]
    max_values = [optimizer_data[opt]["max"] for opt in optimizers]

    bars = ax3.bar(
        x,
        max_values,
        color=[colors[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 10,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax3.set_xlabel("Optimizer", fontsize=10)
    ax3.set_ylabel("Max Memory (MB)", fontsize=10)
    ax3.set_title("Peak Memory Usage", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([opt.upper() for opt in optimizers], rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary table
    table_data = [["Optimizer", "Mean (MB)", "Max (MB)", "Std (MB)"]]

    # Sort by mean memory usage
    sorted_opts = sorted(optimizers, key=lambda x: optimizer_data[x]["mean"])

    for opt in sorted_opts:
        data = optimizer_data[opt]
        row = [
            opt.upper(),
            f"{data['mean']:.1f}",
            f"{data['max']:.1f}",
            f"{data['std']:.1f}",
        ]
        table_data.append(row)

    # Add efficiency ranking
    best_opt = sorted_opts[0]
    table_data.append(["", "", "", ""])
    table_data.append(["Best (Lowest Mean)", best_opt.upper(), "", ""])

    # Create table
    table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best performer
    for i in range(1, len(sorted_opts) + 1):
        if sorted_opts[i - 1] == best_opt:
            for j in range(4):
                table[(i, j)].set_facecolor("#90EE90")
        else:
            table[(i, 0)].set_facecolor("#f0f0f0")

    ax4.set_title(
        "Memory Usage Summary (Sorted by Efficiency)", fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved training comparison to {output_file}")

    # Print summary
    print("\nTraining GPU Memory Summary:")
    print("-" * 60)
    for opt in sorted_opts:
        data = optimizer_data[opt]
        print(f"{opt:12s}: Mean={data['mean']:7.1f} MB, Max={data['max']:7.1f} MB")
    print("-" * 60)
    print(
        f"Most Efficient: {best_opt.upper()} ({optimizer_data[best_opt]['mean']:.1f} MB mean)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GPU memory usage across all optimizers"
    )
    parser.add_argument("--test-dir", help="Test matrix results directory")
    parser.add_argument(
        "--training-output",
        default="gpu_all_training_comparison.png",
        help="Output file for training comparison",
    )
    parser.add_argument(
        "--inference-output",
        default="gpu_all_inference_comparison.png",
        help="Output file for inference comparison",
    )

    args = parser.parse_args()

    # Find GPU monitoring files
    if args.test_dir:
        # Look in test matrix directory
        training_files = glob(f"{args.test_dir}/**/gpu_stats_*.json", recursive=True)
        training_files.extend(
            glob(f"{args.test_dir}/**/gpu_training_*.json", recursive=True)
        )

        inference_files = glob(
            f"{args.test_dir}/**/gpu_inference_*.json", recursive=True
        )
    else:
        # Look in current directory and subdirectories
        training_files = glob("**/gpu_stats_*.json", recursive=True)
        training_files.extend(glob("**/gpu_training_*.json", recursive=True))

        inference_files = glob("**/gpu_inference_*.json", recursive=True)

    print(f"Found {len(training_files)} training GPU files")
    print(f"Found {len(inference_files)} inference GPU files")

    if training_files:
        print("\nGenerating training comparison...")
        create_training_comparison(training_files, args.training_output)

    if inference_files:
        print("\nGenerating inference comparison...")
        # For now, use the same function but label it differently
        # You can create a separate function if inference data has different structure
        create_training_comparison(inference_files, args.inference_output)

    if not training_files and not inference_files:
        print("No GPU monitoring files found!")
        print("Make sure to run training with GPU monitoring enabled.")


if __name__ == "__main__":
    main()
