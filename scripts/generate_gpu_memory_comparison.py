#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate GPU memory comparison graphs from test matrix results."""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from glob import glob
from datetime import datetime


def load_gpu_stats(test_dir):
    """Load GPU statistics from a test directory."""
    gpu_files = list(Path(test_dir).glob("gpu_stats*.json"))
    if not gpu_files:
        return None

    with open(gpu_files[0], "r") as f:
        data = json.load(f)

    if isinstance(data, list) and data:
        memory_values = []
        for sample in data:
            if "memory_used" in sample:
                memory_values.append(sample["memory_used"])

        if memory_values:
            return {
                "mean": np.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "std": np.std(memory_values),
            }
    return None


def generate_gpu_memory_comparison(results_dir, output_dir=None):
    """Generate GPU memory comparison graphs."""

    if output_dir is None:
        output_dir = os.path.join(results_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    all_results_file = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(all_results_file):
        print(f"Error: {all_results_file} not found")
        return

    with open(all_results_file, "r") as f:
        results = json.load(f)

    # Organize data by optimizer and pruning method
    gpu_data = {}
    accuracy_data = {}

    for result in results:
        test_id = result.get("test_id", "")
        optimizer = result.get("optimizer", "")
        pruning = result.get("pruning", "none")
        accuracy = result.get("final_accuracy", 0)

        # Load GPU stats
        test_dir = os.path.join(results_dir, test_id)
        gpu_stats = load_gpu_stats(test_dir)

        if gpu_stats:
            if optimizer not in gpu_data:
                gpu_data[optimizer] = {}
                accuracy_data[optimizer] = {}

            # Store data
            key = f"{pruning}_{int(result.get('target_sparsity', 0) * 100) if pruning != 'none' else 0}"
            gpu_data[optimizer][key] = gpu_stats["mean"]
            accuracy_data[optimizer][key] = accuracy

    # Create comparison plots
    create_memory_comparison_bar_chart(gpu_data, accuracy_data, output_dir)
    create_memory_vs_accuracy_scatter(gpu_data, accuracy_data, output_dir)
    create_memory_timeline_comparison(results_dir, output_dir)

    print(f"GPU memory comparison graphs saved to {output_dir}")


def create_memory_comparison_bar_chart(gpu_data, accuracy_data, output_dir):
    """Create bar chart comparing GPU memory across optimizers."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data
    optimizers = sorted(gpu_data.keys())
    pruning_methods = set()
    for opt_data in gpu_data.values():
        pruning_methods.update(opt_data.keys())
    pruning_methods = sorted(pruning_methods)

    # Colors for each optimizer
    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizers)))

    # Top plot: GPU Memory
    x = np.arange(len(pruning_methods))
    width = 0.15

    for i, optimizer in enumerate(optimizers):
        values = [gpu_data[optimizer].get(pm, 0) for pm in pruning_methods]
        offset = (i - len(optimizers) / 2) * width
        bars = ax1.bar(
            x + offset, values, width, label=optimizer.upper(), color=colors[i]
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    ax1.set_xlabel("Pruning Configuration")
    ax1.set_ylabel("GPU Memory (MiB)")
    ax1.set_title("GPU Memory Usage Comparison Across Optimizers")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pruning_methods, rotation=45, ha="right")
    # Extend y-axis to make room for legend and labels
    y_max = ax1.get_ylim()[1]
    ax1.set_ylim(0, y_max * 1.25)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Accuracy
    for i, optimizer in enumerate(optimizers):
        values = [accuracy_data[optimizer].get(pm, 0) for pm in pruning_methods]
        offset = (i - len(optimizers) / 2) * width
        bars = ax2.bar(
            x + offset, values, width, label=optimizer.upper(), color=colors[i]
        )

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    ax2.set_xlabel("Pruning Configuration")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Comparison Across Optimizers")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pruning_methods, rotation=45, ha="right")
    # Extend y-axis for better visibility
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min * 0.95, y_max * 1.05)
    ax2.legend(loc="lower right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gpu_memory_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def create_memory_vs_accuracy_scatter(gpu_data, accuracy_data, output_dir):
    """Create scatter plot of memory vs accuracy."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors and markers for optimizers
    colors = plt.cm.Set3(np.linspace(0, 1, len(gpu_data)))
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]

    for i, optimizer in enumerate(sorted(gpu_data.keys())):
        memory_values = []
        accuracy_values = []
        labels = []

        for config in gpu_data[optimizer]:
            memory_values.append(gpu_data[optimizer][config])
            accuracy_values.append(accuracy_data[optimizer].get(config, 0))
            labels.append(config)

        # Plot points
        scatter = ax.scatter(
            memory_values,
            accuracy_values,
            c=[colors[i]] * len(memory_values),
            marker=markers[i % len(markers)],
            s=100,
            alpha=0.7,
            edgecolors="black",
            label=optimizer.upper(),
        )

        # Add labels for each point
        for mem, acc, label in zip(memory_values, accuracy_values, labels):
            ax.annotate(
                label,
                (mem, acc),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
            )

    ax.set_xlabel("GPU Memory Usage (MiB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "GPU Memory Efficiency: Memory vs Accuracy Trade-off",
        fontsize=14,
        fontweight="bold",
    )
    # Extend axes for better visibility
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min * 0.95, x_max * 1.05)
    ax.set_ylim(y_min * 0.95, y_max * 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Add efficiency frontier line (optional)
    # This would connect the pareto-optimal points

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "memory_vs_accuracy_scatter.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def create_memory_timeline_comparison(results_dir, output_dir):
    """Create combined timeline comparison of GPU memory during training."""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Find all test directories
    test_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("resnet18_")
    ]

    # Unique colors for each configuration
    config_colors = {
        "adam_none": "#3498db",
        "adam_magnitude_70": "#e67e22",
        "adam_movement_70": "#27ae60",
        "adamwprune_none": "#8e44ad",
        "adamwprune_state_70": "#e74c3c",
        "sgd_none": "#95a5a6",
        "adamw_none": "#2ecc71",
        "adamwadv_none": "#f39c12",
        "adamwspam_none": "#9b59b6",
    }

    line_styles = {
        "movement_70": "-",
        "state_70": "-",
        "magnitude_70": "-",
        "none": "-",
    }

    # Process each test directory
    plot_data = []
    for test_dir in sorted(test_dirs):
        parts = test_dir.split("_")
        if len(parts) >= 2:
            optimizer = parts[1]
            pruning_info = "_".join(parts[2:]) if len(parts) > 2 else "none"
            
            # Handle the "_0" suffix for baseline tests
            if pruning_info == "none_0":
                pruning_info = "none"

            gpu_files = list(
                Path(os.path.join(results_dir, test_dir)).glob("gpu_stats*.json")
            )
            if gpu_files:
                with open(gpu_files[0], "r") as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    timestamps = []
                    memory_values = []

                    for sample in data:
                        if "timestamp" in sample and "memory_used" in sample:
                            # Convert timestamp string to datetime, then to seconds
                            if isinstance(sample["timestamp"], str):
                                dt = datetime.fromisoformat(
                                    sample["timestamp"].replace("Z", "+00:00")
                                )
                                ts = dt.timestamp()
                            else:
                                ts = sample["timestamp"]
                            timestamps.append(ts)
                            memory_values.append(sample["memory_used"])

                    if memory_values:
                        # Normalize timestamps to start from 0
                        if timestamps:
                            timestamps = [t - timestamps[0] for t in timestamps]
                        else:
                            timestamps = list(range(len(memory_values)))

                        # Downsample if too many points (keep every nth point)
                        if len(timestamps) > 500:
                            n = len(timestamps) // 500
                            timestamps = timestamps[::n]
                            memory_values = memory_values[::n]

                        plot_data.append(
                            {
                                "optimizer": optimizer,
                                "pruning": pruning_info,
                                "timestamps": timestamps,
                                "memory": memory_values,
                                "label": f"{optimizer.upper()} {pruning_info.replace('_', ' ')}",
                            }
                        )

    # Plot all lines
    for data in plot_data:
        # Create config key for color lookup
        config_key = f"{data['optimizer']}_{data['pruning']}"
        color = config_colors.get(config_key, "#95a5a6")
        style = line_styles.get(data["pruning"], "-")

        ax.plot(
            data["timestamps"],
            data["memory"],
            label=data["label"],
            color=color,
            linestyle=style,
            linewidth=2,
            alpha=0.8,
        )

    # Formatting
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("GPU Memory (MiB)", fontsize=12)
    ax.set_title(
        "GPU Memory Timeline Comparison - All Optimizers",
        fontsize=14,
        fontweight="bold",
    )

    # Legend with two columns (only if there's data to plot)
    if plot_data:
        ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95, edgecolor="black")

    ax.grid(True, alpha=0.3)

    # Add annotation for AdamWPrune efficiency
    ax.text(
        0.02,
        0.98,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        color="green",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8
        ),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gpu_memory_timeline.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU memory comparison graphs"
    )
    parser.add_argument("results_dir", help="Path to test matrix results directory")
    parser.add_argument("--output", help="Output directory for graphs")
    args = parser.parse_args()

    generate_gpu_memory_comparison(args.results_dir, args.output)


if __name__ == "__main__":
    main()
