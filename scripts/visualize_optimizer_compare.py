#!/usr/bin/env python3
"""
Visualize GPU memory usage comparison across different optimizers.
Compare both training and inference phases.
"""

import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from glob import glob


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

    if not memory_used:
        return None

    return {
        "mean": np.mean(memory_used),
        "std": np.std(memory_used),
        "max": max(memory_used),
        "min": min(memory_used),
        "samples": len(memory_used),
    }


def find_gpu_files(directory):
    """Find all GPU monitoring files in directory."""
    train_files = {}
    infer_files = {}

    # Look for training files
    for f in glob(f"{directory}/gpu_training_*.json"):
        name = Path(f).stem.replace("gpu_training_", "")
        train_files[name] = f

    # Look for inference files
    for f in glob(f"{directory}/gpu_inference_*.json"):
        name = Path(f).stem.replace("gpu_inference_", "")
        # Try to match with training files
        for train_key in train_files:
            if train_key in name or name in train_key:
                infer_files[train_key] = f
                break

    return train_files, infer_files


def create_battle_comparison(directory, output_file):
    """Create comparison plots for all optimizers in battle."""
    train_files, infer_files = find_gpu_files(directory)

    if not train_files:
        print("No GPU monitoring files found!")
        return

    # Collect data for all optimizers
    optimizer_data = {}

    for name in train_files:
        train_data = load_gpu_data(train_files[name])
        train_stats = extract_memory_stats(train_data)

        infer_stats = None
        if name in infer_files:
            infer_data = load_gpu_data(infer_files[name])
            infer_stats = extract_memory_stats(infer_data)

        if train_stats:
            optimizer_data[name] = {"train": train_stats, "infer": infer_stats}

    if not optimizer_data:
        print("No valid data found!")
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Optimizer Battle: GPU Memory Usage Comparison", fontsize=16, fontweight="bold"
    )

    optimizers = sorted(optimizer_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizers)))

    # Plot 1: Training memory comparison
    ax1 = axes[0, 0]
    x = np.arange(len(optimizers))
    width = 0.8

    train_means = [optimizer_data[opt]["train"]["mean"] for opt in optimizers]
    train_stds = [optimizer_data[opt]["train"]["std"] for opt in optimizers]

    bars = ax1.bar(
        x,
        train_means,
        width,
        yerr=train_stds,
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + train_stds[i] + 20,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1.set_xlabel("Optimizer", fontsize=10)
    ax1.set_ylabel("Memory (MB)", fontsize=10)
    ax1.set_title(
        "Training Phase: Mean GPU Memory Usage", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [opt.replace("_", "\n") for opt in optimizers], rotation=0, ha="center"
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Inference memory comparison
    ax2 = axes[0, 1]

    infer_means = []
    infer_stds = []
    valid_optimizers = []
    valid_colors = []

    for i, opt in enumerate(optimizers):
        if optimizer_data[opt]["infer"]:
            infer_means.append(optimizer_data[opt]["infer"]["mean"])
            infer_stds.append(optimizer_data[opt]["infer"]["std"])
            valid_optimizers.append(opt)
            valid_colors.append(colors[i])

    if infer_means:
        x2 = np.arange(len(valid_optimizers))
        bars = ax2.bar(
            x2,
            infer_means,
            width,
            yerr=infer_stds,
            capsize=5,
            color=valid_colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + infer_stds[i] + 20,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax2.set_xlabel("Optimizer", fontsize=10)
        ax2.set_ylabel("Memory (MB)", fontsize=10)
        ax2.set_title(
            "Inference Phase: Mean GPU Memory Usage", fontsize=12, fontweight="bold"
        )
        ax2.set_xticks(x2)
        ax2.set_xticklabels(
            [opt.replace("_", "\n") for opt in valid_optimizers],
            rotation=0,
            ha="center",
        )
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(
            0.5,
            0.5,
            "No inference data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax2.set_title(
            "Inference Phase: Mean GPU Memory Usage", fontsize=12, fontweight="bold"
        )

    # Plot 3: Training vs Inference comparison
    ax3 = axes[1, 0]

    if infer_means:
        x3 = np.arange(len(valid_optimizers))
        width3 = 0.35

        train_vals = [optimizer_data[opt]["train"]["mean"] for opt in valid_optimizers]

        bars1 = ax3.bar(
            x3 - width3 / 2,
            train_vals,
            width3,
            label="Training",
            color="blue",
            alpha=0.7,
        )
        bars2 = ax3.bar(
            x3 + width3 / 2,
            infer_means,
            width3,
            label="Inference",
            color="green",
            alpha=0.7,
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 10,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax3.set_xlabel("Optimizer", fontsize=10)
        ax3.set_ylabel("Memory (MB)", fontsize=10)
        ax3.set_title(
            "Training vs Inference Memory Usage", fontsize=12, fontweight="bold"
        )
        ax3.set_xticks(x3)
        ax3.set_xticklabels(
            [opt.replace("_", "\n") for opt in valid_optimizers],
            rotation=0,
            ha="center",
        )
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(
            0.5,
            0.5,
            "Comparison requires inference data",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax3.set_title(
            "Training vs Inference Memory Usage", fontsize=12, fontweight="bold"
        )

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary table
    table_data = [["Optimizer", "Train Mean", "Train Max", "Infer Mean", "Infer Max"]]

    for opt in optimizers:
        row = [opt.replace("_", " ")]
        train_data = optimizer_data[opt]["train"]
        row.append(f"{train_data['mean']:.1f}")
        row.append(f"{train_data['max']:.1f}")

        if optimizer_data[opt]["infer"]:
            infer_data = optimizer_data[opt]["infer"]
            row.append(f"{infer_data['mean']:.1f}")
            row.append(f"{infer_data['max']:.1f}")
        else:
            row.append("-")
            row.append("-")

        table_data.append(row)

    # Create table
    table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            if j == 0:  # Optimizer column
                table[(i, j)].set_facecolor("#f0f0f0")
                table[(i, j)].set_text_props(weight="bold")
            else:
                table[(i, j)].set_facecolor("white")

    ax4.set_title("Memory Usage Summary (MB)", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved battle comparison to {output_file}")

    # Print summary
    print("\n=== Optimizer Battle Summary ===")
    for opt in optimizers:
        print(f"\n{opt}:")
        train_data = optimizer_data[opt]["train"]
        print(
            f"  Training:  Mean={train_data['mean']:.1f} MB, Max={train_data['max']:.1f} MB"
        )
        if optimizer_data[opt]["infer"]:
            infer_data = optimizer_data[opt]["infer"]
            print(
                f"  Inference: Mean={infer_data['mean']:.1f} MB, Max={infer_data['max']:.1f} MB"
            )
            print(f"  Difference: {train_data['mean'] - infer_data['mean']:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optimizer battle GPU memory usage"
    )
    parser.add_argument(
        "--dir", default=".", help="Directory containing GPU monitoring files"
    )
    parser.add_argument(
        "--output", default="optimizer_battle_gpu.png", help="Output plot filename"
    )

    args = parser.parse_args()

    create_battle_comparison(args.dir, args.output)


if __name__ == "__main__":
    main()
