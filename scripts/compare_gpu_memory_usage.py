#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare actual GPU memory usage from monitoring data."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def load_gpu_stats(results_dir):
    """Load GPU stats for all optimizers."""
    results_dir = Path(results_dir)
    gpu_data = {}

    # Find all GPU stats files
    for gpu_file in results_dir.glob("*/gpu_stats*.json"):
        optimizer = gpu_file.parent.name.split("_")[1]

        with open(gpu_file) as f:
            data = json.load(f)
            gpu_data[optimizer] = data

    return gpu_data


def extract_memory_stats(gpu_data):
    """Extract memory statistics from GPU data."""
    memory_stats = {}

    for optimizer, data in gpu_data.items():
        # Data is a list of GPU stat entries
        if isinstance(data, list) and len(data) > 0:
            # Get memory usage in GB (convert from MB)
            used_mem = []
            for entry in data:
                if "memory_used" in entry:
                    used_gb = entry["memory_used"] / 1024.0  # Convert MB to GB
                    used_mem.append(used_gb)

            if used_mem:
                memory_stats[optimizer] = {
                    "min": min(used_mem),
                    "max": max(used_mem),
                    "avg": np.mean(used_mem),
                    "median": np.median(used_mem),
                    "timeline": used_mem,
                }

    return memory_stats


def create_gpu_memory_comparison(results_dir):
    """Create GPU memory usage comparison from actual monitoring data."""

    gpu_data = load_gpu_stats(results_dir)
    memory_stats = extract_memory_stats(gpu_data)

    if not memory_stats:
        print("No GPU memory data found!")
        return

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Actual GPU Memory Usage Comparison (from monitoring)\nResNet-18 Training with 70% Sparsity",
        fontsize=16,
        fontweight="bold",
    )

    optimizers = list(memory_stats.keys())
    colors = {
        "sgd": "#1f77b4",
        "adam": "#ff7f0e",
        "adamw": "#2ca02c",
        "adamwadv": "#d62728",
        "adamwspam": "#9467bd",
        "adamwprune": "#8c564b",
    }

    # Plot 1: Average GPU Memory Usage Bar Chart
    avg_memory = [memory_stats[opt]["avg"] for opt in optimizers]
    bars1 = ax1.bar(
        optimizers,
        avg_memory,
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    # Highlight AdamWPrune
    adamwprune_idx = (
        optimizers.index("adamwprune") if "adamwprune" in optimizers else -1
    )
    if adamwprune_idx >= 0:
        bars1[adamwprune_idx].set_linewidth(3)
        bars1[adamwprune_idx].set_edgecolor("darkred")

    ax1.set_ylabel("GPU Memory Usage (GB)", fontsize=12)
    ax1.set_title("Average GPU Memory Usage During Training", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels and savings
    for i, (bar, val) in enumerate(zip(bars1, avg_memory)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{val:.2f} GB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # Calculate savings vs AdamWPrune
        if "adamwprune" in memory_stats and optimizers[i] != "adamwprune":
            adamwprune_mem = memory_stats["adamwprune"]["avg"]
            if val > adamwprune_mem:
                extra = ((val - adamwprune_mem) / adamwprune_mem) * 100
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height / 2,
                    f"+{extra:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    # Plot 2: Memory Usage Over Time
    for opt in optimizers:
        if opt in memory_stats and "timeline" in memory_stats[opt]:
            timeline = memory_stats[opt]["timeline"]
            # Sample if too many points
            if len(timeline) > 100:
                indices = np.linspace(0, len(timeline) - 1, 100, dtype=int)
                timeline = [timeline[i] for i in indices]

            ax2.plot(
                range(len(timeline)),
                timeline,
                label=opt.upper(),
                color=colors.get(opt, "gray"),
                linewidth=2 if opt == "adamwprune" else 1,
                alpha=0.8,
            )

    ax2.set_xlabel("Time (samples)", fontsize=12)
    ax2.set_ylabel("GPU Memory (GB)", fontsize=12)
    ax2.set_title("GPU Memory Usage Over Training Time", fontsize=14)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Peak Memory Usage Comparison
    peak_memory = [memory_stats[opt]["max"] for opt in optimizers]
    bars3 = ax3.bar(
        optimizers,
        peak_memory,
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    if adamwprune_idx >= 0:
        bars3[adamwprune_idx].set_linewidth(3)
        bars3[adamwprune_idx].set_edgecolor("darkred")

    ax3.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax3.set_title("Peak GPU Memory Usage", fontsize=14)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars3, peak_memory):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{val:.2f} GB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 4: Memory Efficiency Summary Table
    ax4.axis("off")

    # Create summary statistics
    summary_text = (
        "GPU Memory Usage Summary (Actual Measurements)\n" + "=" * 50 + "\n\n"
    )

    # Sort by average memory usage
    sorted_opts = sorted(optimizers, key=lambda x: memory_stats[x]["avg"])

    summary_text += f"{'Optimizer':<12} {'Average':<10} {'Peak':<10} {'Min':<10}\n"
    summary_text += "-" * 42 + "\n"

    for opt in sorted_opts:
        stats = memory_stats[opt]
        summary_text += f"{opt.upper():<12} {stats['avg']:.2f} GB    {stats['max']:.2f} GB    {stats['min']:.2f} GB\n"

    # Split the display into two sections to avoid overlap
    ax4.text(
        0.05,
        0.95,
        summary_text,
        fontsize=10,
        family="monospace",
        transform=ax4.transAxes,
        verticalalignment="top",
    )

    # Add key findings in a separate box on the right
    if "adamwprune" in memory_stats and "adam" in memory_stats:
        adamwprune_avg = memory_stats["adamwprune"]["avg"]
        adam_avg = memory_stats["adam"]["avg"]
        savings = ((adam_avg - adamwprune_avg) / adam_avg) * 100

        findings = "Key Findings:\n" + "-" * 30 + "\n"
        findings += f"AdamWPrune: {adamwprune_avg:.2f} GB avg\n"
        findings += f"Adam:       {adam_avg:.2f} GB avg\n"
        findings += f"Savings:    {savings:.1f}%\n"

        if "sgd" in memory_stats:
            sgd_avg = memory_stats["sgd"]["avg"]
            findings += f"\nSGD:        {sgd_avg:.2f} GB avg\n"
            diff = abs(adamwprune_avg - sgd_avg)
            if adamwprune_avg < sgd_avg:
                findings += f"AdamWPrune beats SGD by {diff:.3f} GB!"
            else:
                findings += f"Within {diff:.3f} GB of SGD"

        # Place findings in a box on the right side (moved down)
        from matplotlib.patches import FancyBboxPatch

        bbox = FancyBboxPatch(
            (0.55, 0.10),
            0.4,
            0.35,
            boxstyle="round,pad=0.02",
            facecolor="lightgreen",
            alpha=0.3,
            transform=ax4.transAxes,
        )
        ax4.add_patch(bbox)

        ax4.text(
            0.57,
            0.42,
            findings,
            fontsize=11,
            transform=ax4.transAxes,
            color="darkgreen",
            fontweight="bold",
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig("gpu_memory_comparison_actual.png", dpi=150, bbox_inches="tight")
    print("Saved actual GPU memory comparison to: gpu_memory_comparison_actual.png")

    # Also create a simplified comparison
    create_simple_memory_comparison(memory_stats)


def create_simple_memory_comparison(memory_stats):
    """Create a simple bar chart of actual GPU memory usage."""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort optimizers for better visualization
    optimizers = sorted(memory_stats.keys(), key=lambda x: memory_stats[x]["avg"])
    avg_memory = [memory_stats[opt]["avg"] for opt in optimizers]

    colors = []
    for opt in optimizers:
        if opt == "adamwprune":
            colors.append("#8c564b")
        elif opt == "sgd":
            colors.append("#1f77b4")
        elif "adam" in opt:
            colors.append("#ff7f0e")
        else:
            colors.append("gray")

    bars = ax.barh(optimizers, avg_memory, color=colors, edgecolor="black", linewidth=2)

    # Highlight AdamWPrune
    for i, opt in enumerate(optimizers):
        if opt == "adamwprune":
            bars[i].set_linewidth(3)
            bars[i].set_edgecolor("darkred")

    ax.set_xlabel("Average GPU Memory Usage (GB)", fontsize=14)
    ax.set_title(
        "Actual GPU Memory Usage Comparison\n(Measured During Training)",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, avg_memory):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f} GB",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

    # Add vertical line at AdamWPrune level
    if "adamwprune" in memory_stats:
        adamwprune_mem = memory_stats["adamwprune"]["avg"]
        ax.axvline(
            x=adamwprune_mem, color="green", linestyle="--", alpha=0.5, linewidth=2
        )
        ax.text(
            adamwprune_mem,
            ax.get_ylim()[1] * 0.95,
            "AdamWPrune Level",
            ha="center",
            fontsize=11,
            color="green",
        )

    # Capitalize optimizer names for display
    ax.set_yticklabels([opt.upper() for opt in optimizers])

    plt.tight_layout()
    plt.savefig("gpu_memory_simple_actual.png", dpi=150, bbox_inches="tight")
    print("Saved simple GPU memory comparison to: gpu_memory_simple_actual.png")


if __name__ == "__main__":
    results_dir = (
        sys.argv[1] if len(sys.argv) > 1 else "test_matrix_results_20250827_231931"
    )
    create_gpu_memory_comparison(results_dir)
