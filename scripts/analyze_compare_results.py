#!/usr/bin/env python3
"""
Analyze and visualize results from optimizer state pruning battle.
Generates comprehensive comparison graphs for training and inference GPU memory usage.
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


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_optimizer_info(filename):
    """Extract optimizer name and settings from filename."""
    # Pattern: gpu_training_{optimizer}_{method}_{sparsity}.json
    match = re.match(
        r"gpu_(training|inference)_(.+?)_(state|movement|magnitude)_(\d+)", filename
    )
    if match:
        phase, optimizer, method, sparsity = match.groups()
        return {
            "phase": phase,
            "optimizer": optimizer,
            "method": method,
            "sparsity": int(sparsity),
        }
    return None


def collect_battle_data(directory):
    """Collect all GPU monitoring data from battle runs."""
    data = {"training": {}, "inference": {}}

    # Find all GPU monitoring files
    gpu_files = glob(f"{directory}/gpu_*.json")

    for file in gpu_files:
        filename = Path(file).stem
        info = extract_optimizer_info(filename)

        if info:
            phase = info["phase"]
            optimizer = info["optimizer"]

            # Load the data
            file_data = load_json(file)
            samples = file_data.get("samples", [])

            if samples:
                # Extract memory usage
                memory_usage = []
                for s in samples:
                    if "memory_mb" in s:
                        memory_usage.append(s["memory_mb"])
                    elif "memory_used_mb" in s:
                        memory_usage.append(s["memory_used_mb"])

                if memory_usage:
                    data[phase][optimizer] = {
                        "file": file,
                        "info": info,
                        "memory": memory_usage,
                        "mean": np.mean(memory_usage),
                        "std": np.std(memory_usage),
                        "max": max(memory_usage),
                        "min": min(memory_usage),
                        "samples": len(memory_usage),
                    }

    return data


def create_comprehensive_battle_plot(data, output_file):
    """Create comprehensive battle visualization."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Get optimizer list
    all_optimizers = set(list(data["training"].keys()) + list(data["inference"].keys()))
    optimizers = sorted(all_optimizers)
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))
    color_map = dict(zip(optimizers, colors))

    # Plot 1: Training Memory Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if data["training"]:
        x = np.arange(len(data["training"]))
        train_optimizers = sorted(data["training"].keys())
        means = [data["training"][opt]["mean"] for opt in train_optimizers]
        stds = [data["training"][opt]["std"] for opt in train_optimizers]

        bars = ax1.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=[color_map[opt] for opt in train_optimizers],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels
        for i, (bar, std) in enumerate(zip(bars, stds)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 20,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax1.set_xlabel("Optimizer")
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("Training: Mean GPU Memory", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(train_optimizers, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Inference Memory Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if data["inference"]:
        x = np.arange(len(data["inference"]))
        infer_optimizers = sorted(data["inference"].keys())
        means = [data["inference"][opt]["mean"] for opt in infer_optimizers]
        stds = [data["inference"][opt]["std"] for opt in infer_optimizers]

        bars = ax2.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=[color_map[opt] for opt in infer_optimizers],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels
        for i, (bar, std) in enumerate(zip(bars, stds)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 20,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax2.set_xlabel("Optimizer")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_title("Inference: Mean GPU Memory", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(infer_optimizers, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Memory Reduction (Training -> Inference)
    ax3 = fig.add_subplot(gs[0, 2])
    common_optimizers = [
        opt
        for opt in optimizers
        if opt in data["training"] and opt in data["inference"]
    ]

    if common_optimizers:
        x = np.arange(len(common_optimizers))
        reductions = []
        percentages = []

        for opt in common_optimizers:
            train_mean = data["training"][opt]["mean"]
            infer_mean = data["inference"][opt]["mean"]
            reduction = train_mean - infer_mean
            percentage = (reduction / train_mean) * 100
            reductions.append(reduction)
            percentages.append(percentage)

        bars = ax3.bar(
            x,
            reductions,
            color=[color_map[opt] for opt in common_optimizers],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 10,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax3.set_xlabel("Optimizer")
        ax3.set_ylabel("Memory Reduction (MB)")
        ax3.set_title("Memory Reduction: Training → Inference", fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(common_optimizers, rotation=45, ha="right")
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Plot 4: Max Memory Usage Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    width = 0.35
    x = np.arange(len(common_optimizers))

    if common_optimizers:
        train_max = [data["training"][opt]["max"] for opt in common_optimizers]
        infer_max = [
            data["inference"][opt]["max"] if opt in data["inference"] else 0
            for opt in common_optimizers
        ]

        bars1 = ax4.bar(
            x - width / 2,
            train_max,
            width,
            label="Training",
            color="steelblue",
            alpha=0.7,
        )
        bars2 = ax4.bar(
            x + width / 2,
            infer_max,
            width,
            label="Inference",
            color="lightcoral",
            alpha=0.7,
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 10,
                        f"{height:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax4.set_xlabel("Optimizer")
        ax4.set_ylabel("Max Memory (MB)")
        ax4.set_title("Maximum GPU Memory Usage", fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(common_optimizers, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

    # Plot 5: Memory Efficiency Score
    ax5 = fig.add_subplot(gs[1, 1])
    if common_optimizers:
        # Calculate efficiency score: lower memory * consistency (low std)
        scores = []
        for opt in common_optimizers:
            train_data = data["training"][opt]
            infer_data = data["inference"].get(opt, train_data)
            # Efficiency = 1000 / (mean_memory * (1 + std/mean))
            train_score = 1000 / (
                train_data["mean"] * (1 + train_data["std"] / train_data["mean"])
            )
            infer_score = 1000 / (
                infer_data["mean"] * (1 + infer_data["std"] / infer_data["mean"])
            )
            scores.append((train_score + infer_score) / 2)

        bars = ax5.bar(
            x,
            scores,
            color=[color_map[opt] for opt in common_optimizers],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax5.set_xlabel("Optimizer")
        ax5.set_ylabel("Efficiency Score")
        ax5.set_title("Memory Efficiency Score (Higher is Better)", fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(common_optimizers, rotation=45, ha="right")
        ax5.grid(True, alpha=0.3, axis="y")

    # Plot 6: Sample Count Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if optimizers:
        train_samples = [
            data["training"][opt]["samples"] if opt in data["training"] else 0
            for opt in optimizers
        ]
        infer_samples = [
            data["inference"][opt]["samples"] if opt in data["inference"] else 0
            for opt in optimizers
        ]

        x = np.arange(len(optimizers))
        bars1 = ax6.bar(
            x - width / 2,
            train_samples,
            width,
            label="Training",
            color="darkgreen",
            alpha=0.7,
        )
        bars2 = ax6.bar(
            x + width / 2,
            infer_samples,
            width,
            label="Inference",
            color="darkorange",
            alpha=0.7,
        )

        ax6.set_xlabel("Optimizer")
        ax6.set_ylabel("Sample Count")
        ax6.set_title("GPU Monitoring Sample Count", fontweight="bold")
        ax6.set_xticks(x)
        ax6.set_xticklabels(optimizers, rotation=45, ha="right")
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

    # Plot 7-9: Summary Table spanning bottom row
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    # Create summary table
    table_data = [
        [
            "Optimizer",
            "Train Mean",
            "Train Max",
            "Train Std",
            "Infer Mean",
            "Infer Max",
            "Infer Std",
            "Reduction",
            "Efficiency",
        ]
    ]

    for opt in optimizers:
        row = [opt]

        # Training data
        if opt in data["training"]:
            train_data = data["training"][opt]
            row.extend(
                [
                    f"{train_data['mean']:.1f}",
                    f"{train_data['max']:.1f}",
                    f"{train_data['std']:.1f}",
                ]
            )
        else:
            row.extend(["-", "-", "-"])

        # Inference data
        if opt in data["inference"]:
            infer_data = data["inference"][opt]
            row.extend(
                [
                    f"{infer_data['mean']:.1f}",
                    f"{infer_data['max']:.1f}",
                    f"{infer_data['std']:.1f}",
                ]
            )
        else:
            row.extend(["-", "-", "-"])

        # Reduction
        if opt in data["training"] and opt in data["inference"]:
            reduction = data["training"][opt]["mean"] - data["inference"][opt]["mean"]
            percentage = (reduction / data["training"][opt]["mean"]) * 100
            row.append(f"{reduction:.0f} ({percentage:.1f}%)")

            # Efficiency score
            train_eff = 1000 / (
                data["training"][opt]["mean"]
                * (1 + data["training"][opt]["std"] / data["training"][opt]["mean"])
            )
            infer_eff = 1000 / (
                data["inference"][opt]["mean"]
                * (1 + data["inference"][opt]["std"] / data["inference"][opt]["mean"])
            )
            row.append(f"{(train_eff + infer_eff)/2:.2f}")
        else:
            row.extend(["-", "-"])

        table_data.append(row)

    # Create table
    table = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if j == 0:  # Optimizer column
                table[(i, j)].set_facecolor("#f0f0f0")
                table[(i, j)].set_text_props(weight="bold")
            else:
                table[(i, j)].set_facecolor("white")

    ax_table.set_title(
        "Comprehensive GPU Memory Usage Summary (All values in MB)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    # Main title
    fig.suptitle(
        "State Pruning Battle: GPU Memory Analysis", fontsize=16, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved comprehensive battle analysis to {output_file}")

    return table_data


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("GPU Memory Usage Summary - State Pruning Battle")
    print("=" * 80)

    # Training summary
    if data["training"]:
        print("\nTraining Phase:")
        for opt in sorted(data["training"].keys()):
            d = data["training"][opt]
            print(
                f"  {opt:15s}: Mean={d['mean']:7.1f} MB, Max={d['max']:7.1f} MB, "
                f"Std={d['std']:6.1f} MB, Samples={d['samples']:4d}"
            )

    # Inference summary
    if data["inference"]:
        print("\nInference Phase:")
        for opt in sorted(data["inference"].keys()):
            d = data["inference"][opt]
            print(
                f"  {opt:15s}: Mean={d['mean']:7.1f} MB, Max={d['max']:7.1f} MB, "
                f"Std={d['std']:6.1f} MB, Samples={d['samples']:4d}"
            )

    # Comparison summary
    common = set(data["training"].keys()) & set(data["inference"].keys())
    if common:
        print("\nMemory Reduction (Training → Inference):")
        for opt in sorted(common):
            train_mean = data["training"][opt]["mean"]
            infer_mean = data["inference"][opt]["mean"]
            reduction = train_mean - infer_mean
            percentage = (reduction / train_mean) * 100
            print(f"  {opt:15s}: {reduction:6.1f} MB ({percentage:5.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU memory usage from state pruning battle"
    )
    parser.add_argument(
        "--dir", default=".", help="Directory containing GPU monitoring JSON files"
    )
    parser.add_argument(
        "--output",
        default="state_pruning_battle_analysis.png",
        help="Output plot filename",
    )

    args = parser.parse_args()

    # Collect all battle data
    print(f"Analyzing GPU monitoring data in {args.dir}")
    data = collect_battle_data(args.dir)

    if not data["training"] and not data["inference"]:
        print("No GPU monitoring data found!")
        print("Make sure to run the battle with CONFIG_GPU_MONITOR=y")
        return

    # Create comprehensive visualization
    table_data = create_comprehensive_battle_plot(data, args.output)

    # Print summary
    print_summary(data)

    print(f"\nVisualization saved to {args.output}")


if __name__ == "__main__":
    main()
