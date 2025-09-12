#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Visualize GPU memory usage for training vs inference across all optimizers."""

import os
import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


def load_test_results(results_dir):
    """Load all test results with GPU memory data."""
    all_results_file = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(all_results_file):
        print(f"Error: {all_results_file} not found")
        return None

    with open(all_results_file, "r") as f:
        return json.load(f)


def load_gpu_memory_stats(test_dir):
    """Load GPU memory statistics from a test directory."""
    gpu_files = list(Path(test_dir).glob("gpu_stats*.json"))
    if not gpu_files:
        return None

    with open(gpu_files[0], "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        return None

    # Extract memory statistics for training
    training_memory = []
    inference_memory = []

    for sample in data:
        if "memory_used" in sample:
            # For now, all data is training data since we don't have phase info
            training_memory.append(sample["memory_used"])

    stats = {}
    if training_memory:
        stats["training"] = {
            "mean": np.mean(training_memory),
            "max": max(training_memory),
            "min": min(training_memory),
            "std": np.std(training_memory),
            "median": np.median(training_memory),
        }

    if inference_memory:
        stats["inference"] = {
            "mean": np.mean(inference_memory),
            "max": max(inference_memory),
            "min": min(inference_memory),
            "std": np.std(inference_memory),
            "median": np.median(inference_memory),
        }

    return stats if stats else None


def create_training_memory_comparison(results_dir, output_dir):
    """Create training memory comparison visualization."""

    results = load_test_results(results_dir)
    if not results:
        return

    # Organize data by optimizer
    optimizer_data = {}

    for result in results:
        test_id = result.get("test_id", "")
        optimizer = result.get("optimizer", "")
        pruning = result.get("pruning", "none")
        sparsity = (
            int(result.get("target_sparsity", 0) * 100) if pruning != "none" else 0
        )
        accuracy = result.get("final_accuracy", 0)

        # Load GPU stats
        test_dir = os.path.join(results_dir, test_id)
        gpu_stats = load_gpu_memory_stats(test_dir)

        if gpu_stats and "training" in gpu_stats:
            if optimizer not in optimizer_data:
                optimizer_data[optimizer] = []

            optimizer_data[optimizer].append(
                {
                    "pruning": pruning,
                    "sparsity": sparsity,
                    "accuracy": accuracy,
                    "memory_mean": gpu_stats["training"]["mean"],
                    "memory_max": gpu_stats["training"]["max"],
                    "memory_min": gpu_stats["training"]["min"],
                    "label": (
                        f"{pruning}"
                        if pruning != "none" and sparsity == 0
                        else (
                            f"{pruning}_{sparsity}%"
                            if pruning != "none"
                            else "baseline"
                        )
                    ),
                }
            )

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Bar chart comparing mean memory usage
    ax1 = plt.subplot(2, 3, 1)
    create_memory_bars(ax1, optimizer_data, "Training GPU Memory Usage (Mean)", "mean")

    # 2. Bar chart comparing peak memory usage
    ax2 = plt.subplot(2, 3, 2)
    create_memory_bars(ax2, optimizer_data, "Training GPU Memory Usage (Peak)", "max")

    # 3. Memory efficiency (accuracy per MiB)
    ax3 = plt.subplot(2, 3, 3)
    create_efficiency_chart(ax3, optimizer_data)

    # 4. Memory vs Accuracy scatter
    ax4 = plt.subplot(2, 3, 4)
    create_memory_accuracy_scatter(ax4, optimizer_data)

    # 5. AdamWPrune spotlight comparison
    ax5 = plt.subplot(2, 3, 5)
    create_adamwprune_comparison(ax5, optimizer_data)

    # 6. Summary table
    ax6 = plt.subplot(2, 3, 6)
    create_summary_table(ax6, optimizer_data)

    plt.suptitle("GPU Memory Analysis: Training Phase", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_file = os.path.join(output_dir, "training_memory_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training memory comparison saved to {output_file}")


def create_memory_bars(ax, optimizer_data, title, metric="mean"):
    """Create bar chart for memory comparison."""

    optimizers = sorted(optimizer_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizers)))

    # Get all unique configurations
    all_configs = set()
    for opt_results in optimizer_data.values():
        for result in opt_results:
            all_configs.add(result["label"])
    configs = sorted(all_configs)

    x = np.arange(len(configs))
    width = 0.15

    for i, optimizer in enumerate(optimizers):
        values = []
        for config in configs:
            # Find matching result
            matching = [r for r in optimizer_data[optimizer] if r["label"] == config]
            if matching:
                if metric == "mean":
                    values.append(matching[0]["memory_mean"])
                else:
                    values.append(matching[0]["memory_max"])
            else:
                values.append(0)

        offset = (i - len(optimizers) / 2) * width
        bars = ax.bar(
            x + offset, values, width, label=optimizer.upper(), color=colors[i]
        )

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("GPU Memory (MiB)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    # Extend y-axis to make room for legend
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def create_efficiency_chart(ax, optimizer_data):
    """Create memory efficiency chart (accuracy per 100 MiB)."""

    efficiency_data = {}

    for optimizer, results in optimizer_data.items():
        efficiency_data[optimizer] = []
        for result in results:
            efficiency = (result["accuracy"] / result["memory_mean"]) * 100
            efficiency_data[optimizer].append(
                {
                    "label": result["label"],
                    "efficiency": efficiency,
                    "accuracy": result["accuracy"],
                    "memory": result["memory_mean"],
                }
            )

    # Sort optimizers by best efficiency
    optimizer_best = {}
    for opt, data in efficiency_data.items():
        if data:
            optimizer_best[opt] = max(d["efficiency"] for d in data)

    sorted_optimizers = sorted(
        optimizer_best.keys(), key=lambda x: optimizer_best[x], reverse=True
    )

    # Create grouped bar chart
    configs = sorted(
        set(r["label"] for results in optimizer_data.values() for r in results)
    )
    x = np.arange(len(sorted_optimizers))
    width = 0.15
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))

    for i, config in enumerate(configs):
        values = []
        for optimizer in sorted_optimizers:
            matching = [d for d in efficiency_data[optimizer] if d["label"] == config]
            values.append(matching[0]["efficiency"] if matching else 0)

        offset = (i - len(configs) / 2) * width
        bars = ax.bar(x + offset, values, width, label=config, color=colors[i])

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Efficiency (Accuracy % per 100 MiB)")
    ax.set_title("Memory Efficiency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([o.upper() for o in sorted_optimizers])
    # Extend y-axis for legend
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.2)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def create_memory_accuracy_scatter(ax, optimizer_data):
    """Create scatter plot of memory vs accuracy."""

    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizer_data)))

    # Different markers for different pruning methods
    pruning_markers = {
        "none": "o",  # circle for baseline
        "state": "s",  # square for state
        "movement": "^",  # triangle up for movement
        "magnitude": "D",  # diamond for magnitude
    }

    for i, (optimizer, results) in enumerate(sorted(optimizer_data.items())):
        # Group results by pruning method for better visualization
        for r in results:
            marker = pruning_markers.get(r["pruning"], "o")

            # Create label with pruning info
            if r["pruning"] != "none":
                label_text = f"{optimizer.upper()} ({r['pruning']}"
                if r["sparsity"] > 0:
                    label_text += f" {r['sparsity']}%"
                label_text += ")"
            else:
                label_text = f"{optimizer.upper()} (baseline)"

            scatter = ax.scatter(
                r["memory_mean"],
                r["accuracy"],
                c=[colors[i]],
                marker=marker,
                s=120,
                alpha=0.7,
                edgecolors="black",
                linewidth=1.5,
                label=label_text,
            )

            # Annotate all points with abbreviated labels
            annotation = optimizer.upper()[:3]  # First 3 letters
            if r["pruning"] != "none":
                annotation += f"-{r['pruning'][:3]}"  # First 3 letters of pruning
                if r["sparsity"] > 0:
                    annotation += f"{r['sparsity']}"

            ax.annotate(
                annotation,
                (r["memory_mean"], r["accuracy"]),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=5,
                alpha=0.8,
            )

    ax.set_xlabel("GPU Memory Usage (MiB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Memory vs Accuracy Trade-off")
    # Extend axes for better visibility
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min * 0.95, x_max * 1.05)
    ax.set_ylim(y_min * 0.95, y_max * 1.05)

    # Create custom legend for pruning methods
    from matplotlib.patches import Patch

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="State pruning",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Movement pruning",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Magnitude pruning",
        ),
    ]

    # Add legend in corner
    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=7, title="Pruning Methods"
    )
    ax.grid(True, alpha=0.3)

    # Add ideal direction indicator (top-left corner is better)
    ax.text(
        0.02,
        0.98,
        "← Better (Low Memory, High Accuracy)",
        transform=ax.transAxes,
        fontsize=9,
        color="green",
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8
        ),
    )


def create_adamwprune_comparison(ax, optimizer_data):
    """Create spotlight comparison for AdamWPrune."""

    # Get AdamWPrune data
    adamwprune = optimizer_data.get("adamwprune", [])
    if not adamwprune:
        ax.text(
            0.5,
            0.5,
            "No AdamWPrune data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("AdamWPrune Performance")
        return

    # Find the best AdamWPrune result (prefer state pruning)
    adamwprune_best = None
    for r in adamwprune:
        if r["pruning"] == "state":
            adamwprune_best = r
            break
    if not adamwprune_best:
        adamwprune_best = adamwprune[0]  # Use first result if no state pruning

    # Calculate average memory for other optimizers
    other_memory = []
    other_accuracy = []

    for opt, results in optimizer_data.items():
        if opt != "adamwprune":
            for r in results:
                # Compare with any pruning method at similar sparsity
                # or baseline if no pruning results available
                if r["pruning"] in ["movement", "state", "magnitude"]:
                    other_memory.append(r["memory_mean"])
                    other_accuracy.append(r["accuracy"])
                elif r["pruning"] == "none" and not other_memory:
                    # Use baseline if no pruning data available
                    other_memory.append(r["memory_mean"])
                    other_accuracy.append(r["accuracy"])

    # Create comparison bars
    categories = ["GPU Memory\n(MiB)", "Accuracy\n(%)"]
    adamwprune_vals = [adamwprune_best["memory_mean"], adamwprune_best["accuracy"]]
    other_vals = [
        np.mean(other_memory) if other_memory else 0,
        np.mean(other_accuracy) if other_accuracy else 0,
    ]

    x = np.arange(len(categories))
    width = 0.35

    # Create labels with sparsity info
    adamwprune_label = f"AdamWPrune ({adamwprune_best['pruning']})"
    if adamwprune_best["sparsity"] > 0:
        adamwprune_label += f" @ {adamwprune_best['sparsity']}%"

    bars1 = ax.bar(
        x - width / 2,
        adamwprune_vals,
        width,
        label=adamwprune_label,
        color="#2ecc71",
    )
    bars2 = ax.bar(
        x + width / 2, other_vals, width, label="Others (Avg)", color="#95a5a6"
    )

    # Add value labels and savings
    for i, (bar1, bar2, awp, other) in enumerate(
        zip(bars1, bars2, adamwprune_vals, other_vals)
    ):
        ax.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height() + 1,
            f"{awp:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + 1,
            f"{other:.1f}",
            ha="center",
            va="bottom",
        )

        if i == 0 and other > 0:  # Memory comparison
            savings = (other - awp) / other * 100
            # Place text above the bars with more space
            y_pos = max(awp, other) * 1.15

            # Choose color and text based on whether it's savings or extra cost
            if savings > 0:
                text = f"{savings:.1f}% savings"
                color = "green"
            else:
                text = f"{abs(savings):.1f}% extra cost"
                color = "red"

            ax.text(
                x[i],
                y_pos,
                text,
                ha="center",
                color=color,
                fontweight="bold",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9,
                ),
            )

    ax.set_ylabel("Value")
    ax.set_title("AdamWPrune State Pruning Advantage", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    # Extend y-axis to accommodate savings label
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.3)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")


def create_summary_table(ax, optimizer_data):
    """Create summary table of key metrics."""

    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = []
    headers = ["Optimizer", "Config", "Memory (MiB)", "Accuracy (%)", "Efficiency"]

    # Sort by efficiency
    all_results = []
    for opt, results in optimizer_data.items():
        for r in results:
            efficiency = (r["accuracy"] / r["memory_mean"]) * 100
            all_results.append(
                {
                    "optimizer": opt.upper(),
                    "config": r["label"],
                    "memory": r["memory_mean"],
                    "accuracy": r["accuracy"],
                    "efficiency": efficiency,
                }
            )

    all_results.sort(key=lambda x: x["efficiency"], reverse=True)

    # Take top 10
    for r in all_results[:10]:
        table_data.append(
            [
                r["optimizer"],
                r["config"],
                f"{r['memory']:.0f}",
                f"{r['accuracy']:.1f}",
                f"{r['efficiency']:.2f}",
            ]
        )

    # Handle empty table case
    if not table_data:
        table_data = [["No data", "-", "-", "-", "-"]]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.25, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Highlight AdamWPrune rows
    for i, row in enumerate(table_data):
        if "ADAMWPRUNE" in row[0]:
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor("#d4edda")

    ax.set_title("Top 10 Most Memory-Efficient Configurations", fontweight="bold")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training vs inference GPU memory"
    )
    parser.add_argument("results_dir", help="Path to test matrix results directory")
    parser.add_argument("--output", help="Output directory for graphs")
    args = parser.parse_args()

    output_dir = (
        args.output if args.output else os.path.join(args.results_dir, "graphs")
    )
    os.makedirs(output_dir, exist_ok=True)

    create_training_memory_comparison(args.results_dir, output_dir)


if __name__ == "__main__":
    main()
