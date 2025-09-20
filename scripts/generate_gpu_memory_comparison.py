#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate GPU memory comparison graphs from test matrix results."""

import os
import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from glob import glob
from datetime import datetime


def load_gpu_stats(test_dir):
    """Load GPU statistics from a test directory, supporting both single and multi-GPU formats."""
    gpu_files = list(Path(test_dir).glob("gpu_stats*.json"))
    multi_gpu_files = list(Path(test_dir).glob("gpu_stats*_multi_gpu.json"))

    # Try to load multi-GPU data first
    if multi_gpu_files:
        with open(multi_gpu_files[0], "r") as f:
            data = json.load(f)

        if isinstance(data, list) and data:
            # Extract aggregate stats from multi-GPU data
            total_memory_values = []
            per_gpu_memory = {}

            for sample in data:
                if "aggregate_stats" in sample:
                    total_memory_values.append(sample["aggregate_stats"]["total_memory_used"])

                # Also collect per-GPU data for detailed analysis
                if "multi_gpu_data" in sample:
                    for gpu_data in sample["multi_gpu_data"]:
                        gpu_idx = gpu_data["gpu_index"]
                        if gpu_idx not in per_gpu_memory:
                            per_gpu_memory[gpu_idx] = []
                        per_gpu_memory[gpu_idx].append(gpu_data["memory_used"])

            if total_memory_values:
                result = {
                    "mean": np.mean(total_memory_values),
                    "max": max(total_memory_values),
                    "min": min(total_memory_values),
                    "std": np.std(total_memory_values),
                    "multi_gpu": True,
                    "per_gpu_stats": {}
                }

                # Add per-GPU statistics
                for gpu_idx, memory_vals in per_gpu_memory.items():
                    result["per_gpu_stats"][gpu_idx] = {
                        "mean": np.mean(memory_vals),
                        "max": max(memory_vals),
                        "min": min(memory_vals),
                        "std": np.std(memory_vals),
                    }

                return result

    # Fall back to single GPU format
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
                "multi_gpu": False,
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

    # Check if we have multi-GPU data and create additional plots
    has_multi_gpu = any(data.get("multi_gpu", False) for data in gpu_data.values() for data in [data] if isinstance(data, dict))
    if has_multi_gpu:
        create_per_gpu_breakdown_charts(gpu_data, output_dir)
        create_gpu_load_balance_analysis(gpu_data, output_dir)

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

    # Find all test directories (model-agnostic)
    test_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and any(
            d.startswith(f"{model}_") for model in ["lenet5", "resnet18", "resnet50"]
        )
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
        ax.legend(
            loc="upper right", fontsize=9, ncol=2, framealpha=0.95, edgecolor="black"
        )

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


def create_per_gpu_breakdown_charts(gpu_data, output_dir):
    """Create per-GPU memory breakdown charts for multi-GPU setups."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Per-GPU Memory Analysis (4x A10G GPUs)", fontsize=16, fontweight="bold")

    # Collect all per-GPU data
    all_per_gpu_data = {}

    for optimizer, configs in gpu_data.items():
        for config, stats in configs.items():
            if isinstance(stats, dict) and stats.get("multi_gpu", False):
                per_gpu_stats = stats.get("per_gpu_stats", {})
                for gpu_idx, gpu_stats in per_gpu_stats.items():
                    if gpu_idx not in all_per_gpu_data:
                        all_per_gpu_data[gpu_idx] = {}
                    key = f"{optimizer}_{config}"
                    all_per_gpu_data[gpu_idx][key] = gpu_stats["mean"]

    if not all_per_gpu_data:
        plt.close()
        return

    # Create subplot for each GPU
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_per_gpu_data)))

    for i, (gpu_idx, gpu_data) in enumerate(sorted(all_per_gpu_data.items())):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        if gpu_data:
            configs = list(gpu_data.keys())
            values = list(gpu_data.values())

            bars = ax.bar(range(len(configs)), values, color=colors[i], alpha=0.7)
            ax.set_title(f"GPU {gpu_idx} Memory Usage", fontweight="bold")
            ax.set_ylabel("Memory (MiB)")
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"GPU {gpu_idx} - No Data")

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_gpu_memory_breakdown.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def create_gpu_load_balance_analysis(gpu_data, output_dir):
    """Create GPU load balance analysis showing memory distribution across GPUs."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("GPU Load Balance Analysis (4x A10G)", fontsize=16, fontweight="bold")

    # Collect load balance data
    balance_data = {}

    for optimizer, configs in gpu_data.items():
        for config, stats in configs.items():
            if isinstance(stats, dict) and stats.get("multi_gpu", False):
                per_gpu_stats = stats.get("per_gpu_stats", {})
                if len(per_gpu_stats) >= 4:  # Ensure we have 4 GPUs
                    key = f"{optimizer}_{config}"
                    gpu_memories = [per_gpu_stats[i]["mean"] for i in range(4)]

                    # Calculate load balance metrics
                    avg_memory = np.mean(gpu_memories)
                    std_memory = np.std(gpu_memories)
                    cv = (std_memory / avg_memory) * 100 if avg_memory > 0 else 0  # Coefficient of variation

                    balance_data[key] = {
                        "memories": gpu_memories,
                        "avg": avg_memory,
                        "std": std_memory,
                        "cv": cv
                    }

    if not balance_data:
        plt.close()
        return

    # Left plot: Memory distribution across GPUs
    configs = list(balance_data.keys())
    gpu_indices = [0, 1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))

    width = 0.15
    x = np.arange(len(gpu_indices))

    for i, (config, data) in enumerate(balance_data.items()):
        offset = (i - len(configs)/2) * width
        bars = ax1.bar(x + offset, data["memories"], width,
                      label=config, color=colors[i], alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, data["memories"]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax1.set_xlabel("GPU Index")
    ax1.set_ylabel("Memory Usage (MiB)")
    ax1.set_title("Memory Distribution Across GPUs")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"GPU {i}" for i in gpu_indices])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right plot: Load balance coefficient (lower is better)
    configs_short = [config.replace("_", "\n") for config in configs]
    cv_values = [data["cv"] for data in balance_data.values()]

    bars = ax2.bar(range(len(configs)), cv_values,
                   color=plt.cm.RdYlGn_r(np.array(cv_values)/max(cv_values)))

    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Load Imbalance (CV %)")
    ax2.set_title("GPU Load Balance\n(Lower is Better)")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs_short, rotation=45, ha="right", fontsize=8)

    # Add value labels
    for bar, val in zip(bars, cv_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # Add balance quality indicators
    ax2.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="Excellent (<5%)")
    ax2.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="Good (<10%)")
    ax2.axhline(y=20, color="red", linestyle="--", alpha=0.7, label="Poor (>20%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_load_balance_analysis.png"),
                dpi=150, bbox_inches="tight")
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
