#!/usr/bin/env python3
"""
Visualize GPU memory usage comparison between training and inference phases.
"""

import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from matplotlib.patches import Rectangle


def load_gpu_data(json_file):
    """Load GPU monitoring data from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def extract_memory_stats(data):
    """Extract memory statistics from GPU monitoring data."""
    samples = data.get("samples", [])
    if not samples:
        return None

    # Handle different field names for memory
    memory_used = []
    for s in samples:
        if "memory_used_mb" in s:
            memory_used.append(s["memory_used_mb"])
        elif "memory_mb" in s:
            memory_used.append(s["memory_mb"])

    memory_free = [s.get("memory_free_mb", 0) for s in samples]
    timestamps = [s["timestamp"] for s in samples if "timestamp" in s]

    # Convert timestamps to relative seconds
    if timestamps:
        start_time = timestamps[0]
        timestamps = [(t - start_time) for t in timestamps]

    return {
        "timestamps": timestamps,
        "memory_used": memory_used,
        "memory_free": memory_free,
        "mean_used": np.mean(memory_used) if memory_used else 0,
        "std_used": np.std(memory_used) if memory_used else 0,
        "max_used": max(memory_used) if memory_used else 0,
        "min_used": min(memory_used) if memory_used else 0,
    }


def create_comparison_plot(training_data, inference_data, output_file):
    """Create comparison plot of training vs inference GPU memory usage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GPU Memory Usage: Training vs Inference Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Extract stats
    train_stats = extract_memory_stats(training_data)
    infer_stats = extract_memory_stats(inference_data)

    if not train_stats or not infer_stats:
        print("Error: Could not extract memory statistics from data")
        return

    # Plot 1: Memory usage over time
    ax1 = axes[0, 0]
    ax1.plot(
        train_stats["timestamps"],
        train_stats["memory_used"],
        label="Training",
        color="blue",
        alpha=0.7,
        linewidth=1,
    )
    ax1.plot(
        infer_stats["timestamps"],
        infer_stats["memory_used"],
        label="Inference",
        color="green",
        alpha=0.7,
        linewidth=1,
    )
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Memory Used (MB)", fontsize=10)
    ax1.set_title("Memory Usage Over Time", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Statistical comparison
    ax2 = axes[0, 1]
    categories = ["Mean", "Max", "Min"]
    x = np.arange(len(categories))
    width = 0.35

    train_values = [
        train_stats["mean_used"],
        train_stats["max_used"],
        train_stats["min_used"],
    ]
    infer_values = [
        infer_stats["mean_used"],
        infer_stats["max_used"],
        infer_stats["min_used"],
    ]

    bars1 = ax2.bar(
        x - width / 2, train_values, width, label="Training", color="blue", alpha=0.7
    )
    bars2 = ax2.bar(
        x + width / 2, infer_values, width, label="Inference", color="green", alpha=0.7
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 10,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax2.set_xlabel("Statistic", fontsize=10)
    ax2.set_ylabel("Memory (MB)", fontsize=10)
    ax2.set_title("Memory Usage Statistics", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Memory distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(
        train_stats["memory_used"],
        bins=30,
        alpha=0.5,
        label="Training",
        color="blue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.hist(
        infer_stats["memory_used"],
        bins=30,
        alpha=0.5,
        label="Inference",
        color="green",
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_xlabel("Memory Used (MB)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title("Memory Usage Distribution", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary data
    summary_data = [
        ["Metric", "Training", "Inference", "Difference"],
        [
            "Mean (MB)",
            f'{train_stats["mean_used"]:.1f}',
            f'{infer_stats["mean_used"]:.1f}',
            f'{train_stats["mean_used"] - infer_stats["mean_used"]:.1f}',
        ],
        [
            "Std Dev (MB)",
            f'{train_stats["std_used"]:.1f}',
            f'{infer_stats["std_used"]:.1f}',
            f'{train_stats["std_used"] - infer_stats["std_used"]:.1f}',
        ],
        [
            "Max (MB)",
            f'{train_stats["max_used"]:.1f}',
            f'{infer_stats["max_used"]:.1f}',
            f'{train_stats["max_used"] - infer_stats["max_used"]:.1f}',
        ],
        [
            "Min (MB)",
            f'{train_stats["min_used"]:.1f}',
            f'{infer_stats["min_used"]:.1f}',
            f'{train_stats["min_used"] - infer_stats["min_used"]:.1f}',
        ],
        [
            "Samples",
            f'{len(train_stats["memory_used"])}',
            f'{len(infer_stats["memory_used"])}',
            "-",
        ],
    ]

    # Create table
    table = ax4.table(cellText=summary_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style data rows
    for i in range(1, len(summary_data)):
        for j in range(4):
            if j == 0:  # Metric column
                table[(i, j)].set_facecolor("#f0f0f0")
                table[(i, j)].set_text_props(weight="bold")
            else:
                table[(i, j)].set_facecolor("white")

    ax4.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

    # Add model info if available
    model_info = f"Model: {training_data.get('model_name', 'Unknown')}"
    if "device" in training_data:
        model_info += f" | Device: {training_data['device']}"
    fig.text(0.5, 0.02, model_info, ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {output_file}")

    return {"training": train_stats, "inference": infer_stats}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GPU memory usage comparison"
    )
    parser.add_argument(
        "--training", required=True, help="Path to training GPU monitoring JSON file"
    )
    parser.add_argument(
        "--inference", required=True, help="Path to inference GPU monitoring JSON file"
    )
    parser.add_argument(
        "--output", default="gpu_comparison.png", help="Output plot filename"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.training}")
    training_data = load_gpu_data(args.training)

    print(f"Loading inference data from {args.inference}")
    inference_data = load_gpu_data(args.inference)

    # Create visualization
    stats = create_comparison_plot(training_data, inference_data, args.output)

    # Print summary
    if stats:
        print("\n=== GPU Memory Usage Summary ===")
        print(
            f"Training:  Mean={stats['training']['mean_used']:.1f} MB, "
            f"Max={stats['training']['max_used']:.1f} MB, "
            f"Samples={len(stats['training']['memory_used'])}"
        )
        print(
            f"Inference: Mean={stats['inference']['mean_used']:.1f} MB, "
            f"Max={stats['inference']['max_used']:.1f} MB, "
            f"Samples={len(stats['inference']['memory_used'])}"
        )
        print(
            f"Difference: {stats['training']['mean_used'] - stats['inference']['mean_used']:.1f} MB (mean)"
        )


if __name__ == "__main__":
    main()
