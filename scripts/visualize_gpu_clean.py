#!/usr/bin/env python3
"""
Clean GPU memory comparison visualization.
Shows only real data, no confusing explanations.
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
        return json.load(f)


def extract_memory_stats(data):
    """Extract memory statistics from GPU monitoring data."""
    samples = None

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", [])

    if not samples:
        return None

    memory_values = []
    for s in samples:
        if "memory_used" in s:
            memory_values.append(s["memory_used"])
        elif "memory_mb" in s:
            memory_values.append(s["memory_mb"])

    if not memory_values:
        return None

    return {
        "mean": np.mean(memory_values),
        "std": np.std(memory_values),
        "max": max(memory_values),
        "min": min(memory_values),
    }


def get_optimizer_info(filepath):
    """Extract optimizer name and settings from filepath."""
    path = Path(filepath)
    parent_dir = path.parent.name

    optimizer = "unknown"

    # Extract from directory name
    if "resnet18_" in parent_dir:
        parts = parent_dir.split("_")
        if len(parts) >= 2:
            optimizer = parts[1]
    else:
        # Extract from filename
        filename = path.stem
        if "gpu_training_" in filename:
            optimizer = filename.replace("gpu_training_", "").split("_")[0]
        elif "gpu_stats_resnet18_" in filename:
            optimizer = filename.replace("gpu_stats_resnet18_", "").split("_")[0]

    return optimizer


def create_clean_comparison(training_data, inference_data, output_file):
    """Create clean comparison plot without confusing elements."""

    # Create figure with better layout
    fig = plt.figure(figsize=(18, 10))

    # Use GridSpec for better control
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

    # Sort optimizers by training memory
    optimizers = sorted(training_data.keys(), key=lambda x: training_data[x]["mean"])

    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(optimizers)))
    color_map = dict(zip(optimizers, colors))

    # ========== Plot 1: Training Memory Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(optimizers))

    means = [training_data[opt]["mean"] for opt in optimizers]
    stds = [training_data[opt]["std"] for opt in optimizers]

    bars = ax1.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for i, (bar, std) in enumerate(zip(bars, stds)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 20,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax1.set_ylabel("Memory (MB)", fontsize=11)
    ax1.set_title("Training GPU Memory Usage", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    # Rotate labels to prevent overlap
    ax1.set_xticklabels(
        [opt.upper() for opt in optimizers], rotation=45, ha="right", fontsize=10
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # ========== Plot 2: Inference Memory (if available) ==========
    ax2 = fig.add_subplot(gs[0, 1])

    if inference_data:
        # Show actual inference data we have
        infer_opts = list(inference_data.keys())
        x_infer = np.arange(len(infer_opts))
        infer_means = [inference_data[opt]["mean"] for opt in infer_opts]

        bars = ax2.bar(
            x_infer,
            infer_means,
            color="lightcoral",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 10,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax2.set_ylabel("Memory (MB)", fontsize=11)
        ax2.set_title("Inference GPU Memory Usage", fontsize=12, fontweight="bold")
        ax2.set_xticks(x_infer)
        ax2.set_xticklabels([opt.upper() for opt in infer_opts], fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(
            0.5,
            0.5,
            "No Inference Data Available",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax2.set_title("Inference GPU Memory Usage", fontsize=12, fontweight="bold")
        ax2.set_xticks([])

    # ========== Plot 3: Memory Efficiency Ranking ==========
    ax3 = fig.add_subplot(gs[0, 2])

    y_pos = np.arange(len(optimizers))
    values = [training_data[opt]["mean"] for opt in optimizers]

    bars = ax3.barh(
        y_pos,
        values,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax3.text(
            val + 5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.0f} MB",
            va="center",
            fontsize=10,
        )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([opt.upper() for opt in optimizers], fontsize=10)
    ax3.set_xlabel("Memory Usage (MB)", fontsize=11)
    ax3.set_title("Training Memory Ranking", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="x")

    # ========== Plot 4: Peak Memory Comparison ==========
    ax4 = fig.add_subplot(gs[1, 0])

    max_values = [training_data[opt]["max"] for opt in optimizers]

    bars = ax4.bar(
        x,
        max_values,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 10,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax4.set_ylabel("Max Memory (MB)", fontsize=11)
    ax4.set_title("Peak Memory During Training", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [opt.upper() for opt in optimizers], rotation=45, ha="right", fontsize=10
    )
    ax4.grid(True, alpha=0.3, axis="y")

    # ========== Plot 5: Memory Savings vs Worst ==========
    ax5 = fig.add_subplot(gs[1, 1])

    worst_mean = max(training_data[opt]["mean"] for opt in optimizers)
    savings = [worst_mean - training_data[opt]["mean"] for opt in optimizers]
    savings_pct = [(s / worst_mean * 100) for s in savings]

    bars = ax5.bar(
        x,
        savings_pct,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    for bar, pct in zip(bars, savings_pct):
        height = bar.get_height()
        if height > 0:
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax5.set_ylabel("Savings (%)", fontsize=11)
    ax5.set_title("Memory Savings vs Worst Case", fontsize=12, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(
        [opt.upper() for opt in optimizers], rotation=45, ha="right", fontsize=10
    )
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_ylim(0, max(savings_pct) * 1.15 if savings_pct else 20)

    # ========== Plot 6: Summary Statistics ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Create clean summary table
    table_data = []
    table_data.append(["Optimizer", "Mean (MB)", "Max (MB)", "Std (MB)"])

    for opt in optimizers:
        data = training_data[opt]
        row = [
            opt.upper(),
            f"{data['mean']:.1f}",
            f"{data['max']:.1f}",
            f"{data['std']:.1f}",
        ]
        table_data.append(row)

    # Add best/worst summary
    table_data.append(["", "", "", ""])
    best_opt = optimizers[0]
    worst_opt = optimizers[-1]
    savings = training_data[worst_opt]["mean"] - training_data[best_opt]["mean"]
    savings_pct = (savings / training_data[worst_opt]["mean"]) * 100

    table_data.append(
        [
            "BEST",
            best_opt.upper(),
            f"{training_data[best_opt]['mean']:.1f}",
            f"-{savings:.0f} MB",
        ]
    )
    table_data.append(
        [
            "WORST",
            worst_opt.upper(),
            f"{training_data[worst_opt]['mean']:.1f}",
            f"({savings_pct:.1f}% more)",
        ]
    )

    # Create table
    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best performer
    for j in range(4):
        table[(1, j)].set_facecolor("#90EE90")  # Light green

    # Style summary rows
    row_idx = len(optimizers) + 2
    for j in range(4):
        table[(row_idx, j)].set_facecolor("#e8f5e9")
        table[(row_idx, j)].set_text_props(weight="bold")
        table[(row_idx + 1, j)].set_facecolor("#ffebee")

    ax6.set_title("Summary Statistics", fontsize=12, fontweight="bold")

    # Main title
    fig.suptitle(
        "GPU Memory Usage Comparison - ResNet-18 Training (70% Sparsity Target)",
        fontsize=16,
        fontweight="bold",
    )

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved clean comparison to {output_file}")

    return optimizers


def main():
    parser = argparse.ArgumentParser(description="Generate clean GPU memory comparison")
    parser.add_argument(
        "--output",
        default="gpu_comparison_clean.png",
        help="Output file for comparison",
    )

    args = parser.parse_args()

    # Find all GPU files from the test matrix run
    test_dir = "test_matrix_results_20250828_190015"
    training_files = glob(f"{test_dir}/**/gpu_stats_*.json", recursive=True)

    # Also include standalone runs
    training_files.extend(glob("resnet18/gpu_training_*.json"))

    inference_files = glob("resnet18/gpu_inference_*.json")

    print(f"Found {len(training_files)} training GPU files")
    print(f"Found {len(inference_files)} inference GPU files")

    # Process training data
    training_data = {}
    for file in training_files:
        data = load_gpu_data(file)
        stats = extract_memory_stats(data)
        if stats:
            optimizer = get_optimizer_info(file)
            # Keep the best (lowest memory) run for each optimizer
            if (
                optimizer not in training_data
                or stats["mean"] < training_data[optimizer]["mean"]
            ):
                training_data[optimizer] = stats

    # Process inference data
    inference_data = {}
    for file in inference_files:
        data = load_gpu_data(file)
        stats = extract_memory_stats(data)
        if stats:
            # We only have AdamWPrune inference data
            inference_data["adamwprune"] = stats

    if not training_data:
        print("No valid training data found!")
        return

    # Create visualization
    sorted_opts = create_clean_comparison(training_data, inference_data, args.output)

    # Print summary
    print("\n" + "=" * 70)
    print("GPU Memory Usage Summary")
    print("=" * 70)

    print("\nTraining Phase (sorted by efficiency):")
    for opt in sorted_opts:
        stats = training_data[opt]
        print(
            f"  {opt.upper():12s}: Mean={stats['mean']:7.1f} MB, Max={stats['max']:7.1f} MB"
        )

    if inference_data:
        print("\nInference Phase:")
        for opt, stats in inference_data.items():
            print(
                f"  {opt.upper():12s}: Mean={stats['mean']:7.1f} MB, Max={stats['max']:7.1f} MB"
            )

    print("\n" + "=" * 70)
    best = sorted_opts[0]
    worst = sorted_opts[-1]
    savings = training_data[worst]["mean"] - training_data[best]["mean"]
    print(
        f"Most Efficient: {best.upper()} saves {savings:.1f} MB ({savings/training_data[worst]['mean']*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
