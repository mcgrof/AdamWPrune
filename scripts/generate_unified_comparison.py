#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate unified comparison plot for all optimizers."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def load_metrics(results_dir):
    """Load all metrics from test results."""
    results_dir = Path(results_dir)
    all_metrics = {}

    for test_dir in results_dir.glob("resnet18_*_70"):
        metrics_file = test_dir / "training_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                optimizer = test_dir.name.split("_")[1]
                all_metrics[optimizer] = data

    return all_metrics


def create_unified_comparison(results_dir, output_file="unified_comparison.png"):
    """Create a unified comparison plot."""
    metrics = load_metrics(results_dir)

    if not metrics:
        print("No metrics found!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "ResNet-18 Optimizer Comparison (70% Sparsity Target)",
        fontsize=16,
        fontweight="bold",
    )

    colors = {
        "sgd": "blue",
        "adam": "orange",
        "adamw": "green",
        "adamwadv": "red",
        "adamwspam": "purple",
        "adamwprune": "brown",
    }

    # Plot 1: Accuracy Evolution
    for optimizer, data in metrics.items():
        if "test_accuracy" in data:
            epochs = data.get("epochs", list(range(1, len(data["test_accuracy"]) + 1)))
            ax1.plot(
                epochs,
                data["test_accuracy"],
                label=optimizer.upper(),
                color=colors.get(optimizer, "gray"),
                linewidth=2,
                alpha=0.8,
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy Evolution")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final Accuracy Bar Chart
    final_accs = {}
    for optimizer, data in metrics.items():
        if "test_accuracy" in data and data["test_accuracy"]:
            final_accs[optimizer] = data["test_accuracy"][-1]

    optimizers = list(final_accs.keys())
    accuracies = list(final_accs.values())

    bars = ax2.bar(
        optimizers, accuracies, color=[colors.get(o, "gray") for o in optimizers]
    )
    ax2.set_ylabel("Final Test Accuracy (%)")
    ax2.set_title("Final Accuracy Comparison")
    ax2.set_ylim([85, 93])

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
        )

    # Plot 3: Sparsity Achievement
    for optimizer, data in metrics.items():
        if "sparsity" in data:
            epochs = data.get("epochs", list(range(1, len(data["sparsity"]) + 1)))
            sparsities = [s * 100 for s in data["sparsity"]]
            ax3.plot(
                epochs,
                sparsities,
                label=optimizer.upper(),
                color=colors.get(optimizer, "gray"),
                linewidth=2,
                alpha=0.8,
            )

    ax3.axhline(y=70, color="red", linestyle="--", label="Target (70%)", alpha=0.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Sparsity (%)")
    ax3.set_title("Sparsity Evolution")
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])

    # Plot 4: Memory Efficiency Table
    ax4.axis("tight")
    ax4.axis("off")

    # Create efficiency data
    memory_data = {
        "sgd": (91.80, 3.03, 30.30),
        "adam": (90.63, 5.03, 18.02),
        "adamw": (90.50, 5.03, 17.99),
        "adamwspam": (90.25, 5.13, 17.59),
        "adamwadv": (90.11, 5.13, 17.57),
        "adamwprune": (88.44, 3.03, 29.19),
    }

    table_data = []
    for opt in optimizers:
        if opt in memory_data:
            acc, mem, eff = memory_data[opt]
            table_data.append([opt.upper(), f"{acc:.2f}%", f"{mem:.2f}x", f"{eff:.2f}"])

    table = ax4.table(
        cellText=table_data,
        colLabels=["Optimizer", "Accuracy", "Memory", "Efficiency"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color code the table
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i + 1, j)]
            if j == 0:  # Optimizer name
                cell.set_facecolor(colors.get(optimizers[i], "gray"))
                cell.set_alpha(0.3)

    ax4.set_title(
        "Memory Efficiency Comparison", fontsize=12, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved unified comparison to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "test_matrix_results_20250827_231931"

    create_unified_comparison(results_dir)
