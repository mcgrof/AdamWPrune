#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate comparison graphs for each optimizer from test matrix results."""

import json
import os
import sys
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_test_results(results_dir):
    """Load all test results from a test matrix directory."""
    json_file = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found")
        return None

    with open(json_file, "r") as f:
        return json.load(f)


def load_metrics_file(results_dir, test_id):
    """Load training metrics for a specific test."""
    metrics_file = os.path.join(results_dir, test_id, "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None


def group_by_optimizer(results):
    """Group test results by optimizer."""
    grouped = defaultdict(list)
    for result in results:
        if result.get("success", False):
            optimizer = result.get("optimizer", "unknown")
            grouped[optimizer].append(result)
    return grouped


def extract_plot_data(metrics):
    """Extract data for plotting from metrics."""
    if not metrics:
        return None

    # Check if we have data in either format
    if "epochs" not in metrics and "test_accuracy" not in metrics:
        return None

    epochs = []
    accuracies = []
    losses = []
    sparsities = []

    # Handle different JSON formats
    if "epochs" in metrics and isinstance(metrics["epochs"], list):
        if metrics["epochs"] and isinstance(metrics["epochs"][0], dict):
            # LeNet-5 format - epochs is a list of dictionaries
            for epoch_data in metrics["epochs"]:
                epochs.append(epoch_data["epoch"])
                accuracies.append(epoch_data["accuracy"])
                losses.append(epoch_data["avg_loss"])
                sparsities.append(epoch_data.get("sparsity", 0) * 100)
        else:
            # ResNet-18 format - separate lists
            epochs = metrics["epochs"]
            accuracies = metrics.get("test_accuracy", [])
            losses = metrics.get("train_loss", [])
            sparsities = [s * 100 for s in metrics.get("sparsity", [])]
    elif "test_accuracy" in metrics:
        # Alternative ResNet-18 format
        epochs = list(range(1, len(metrics["test_accuracy"]) + 1))
        accuracies = metrics.get("test_accuracy", [])
        losses = metrics.get("train_loss", [])
        sparsities = [s * 100 for s in metrics.get("sparsity", [])]

    return {
        "epochs": epochs,
        "accuracies": accuracies,
        "losses": losses,
        "sparsities": sparsities,
        "final_accuracy": metrics.get("final_accuracy", 0),
        "final_sparsity": metrics.get("final_sparsity", 0) * 100,
        "training_time": metrics.get(
            "total_time", metrics.get("total_training_time", 0)
        ),
        "compression_ratio": metrics.get("compression_ratio", 1.0),
    }


def create_optimizer_comparison(optimizer, tests, results_dir, output_dir):
    """Create comparison graphs for a single optimizer."""
    # Detect model from test results
    model_name = tests[0].get("model", "unknown") if tests else "unknown"
    display_name = model_name.upper() if model_name != "unknown" else "Model"

    # Sort tests by sparsity level
    tests.sort(key=lambda x: x.get("final_sparsity", 0))

    # Prepare data for plotting
    plot_data = {}
    for test in tests:
        test_id = test["test_id"]
        metrics = load_metrics_file(results_dir, test_id)
        if metrics:
            data = extract_plot_data(metrics)
            if data:
                # Determine label based on pruning method and sparsity
                if "none" in test_id:
                    label = f"{optimizer.upper()} Baseline"
                else:
                    sparsity = test.get("final_sparsity", 0) * 100
                    if "movement" in test_id:
                        pruning = "Movement"
                    elif "magnitude" in test_id:
                        pruning = "Magnitude"
                    elif "state" in test_id:
                        pruning = "State"
                    else:
                        pruning = "Unknown"
                    label = f"{optimizer.upper()} {pruning} {sparsity:.0f}%"
                plot_data[label] = data

    if not plot_data:
        print(f"  No valid data for {optimizer}")
        return

    # Create figure 1: Model comparison (6 panels)
    fig1 = plt.figure(figsize=(18, 14))
    fig1.suptitle(
        f"{display_name} Model Comparison: {optimizer.upper()}",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: Training Accuracy Evolution
    ax1 = plt.subplot(3, 3, 1)
    for label, data in plot_data.items():
        ax1.plot(data["epochs"], data["accuracies"], marker="o", label=label)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy Evolution")
    # Extend y-axis for legend visibility
    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min, y_max * 1.1)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Training Loss Evolution
    ax2 = plt.subplot(3, 3, 2)
    for label, data in plot_data.items():
        ax2.plot(data["epochs"], data["losses"], marker="s", label=label)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Loss Evolution")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Panel 3: Sparsity Progression
    ax3 = plt.subplot(3, 3, 3)
    has_sparsity = False
    for label, data in plot_data.items():
        if max(data["sparsities"]) > 0:
            ax3.plot(data["epochs"], data["sparsities"], marker="^", label=label)
            has_sparsity = True
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Sparsity (%)")
    ax3.set_title("Sparsity Progression")
    if has_sparsity:
        ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 110])  # Extra space for legend

    # Panel 4: Final Accuracy Comparison
    ax4 = plt.subplot(3, 3, 4)
    labels = list(plot_data.keys())
    final_accs = [data["final_accuracy"] for data in plot_data.values()]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax4.bar(range(len(labels)), final_accs, color=colors)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(
        [l.replace(" ", "\n") for l in labels], rotation=45, ha="right", fontsize=8
    )
    ax4.set_ylabel("Final Accuracy (%)")
    ax4.set_title("Final Accuracy Comparison")
    # Dynamic y-limits with padding
    y_min = min(final_accs) * 0.98 if final_accs else 90
    y_max = max(final_accs) * 1.02 if final_accs else 100
    ax4.set_ylim([y_min, y_max])
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel 5: Training Time Comparison
    ax5 = plt.subplot(3, 3, 5)
    times = [data["training_time"] for data in plot_data.values()]
    bars = ax5.bar(range(len(labels)), times, color=colors)
    ax5.set_xticks(range(len(labels)))
    ax5.set_xticklabels(
        [l.replace(" ", "\n") for l in labels], rotation=45, ha="right", fontsize=8
    )
    ax5.set_ylabel("Training Time (seconds)")
    ax5.set_title("Training Time")
    ax5.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time in zip(bars, times):
        if time > 0:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time:.1f}s",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # Panel 6: Accuracy vs Sparsity Trade-off
    ax6 = plt.subplot(3, 3, 6)
    sparsities = [data["final_sparsity"] for data in plot_data.values()]
    ax6.scatter(
        sparsities, final_accs, s=100, c=colors, edgecolors="black", linewidth=1.5
    )
    for i, label in enumerate(labels):
        ax6.annotate(
            label.split()[-1] if "%" in label else "Baseline",
            (sparsities[i], final_accs[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    ax6.set_xlabel("Sparsity (%)")
    ax6.set_ylabel("Final Accuracy (%)")
    ax6.set_title("Accuracy vs Sparsity Trade-off")
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([-5, 105])
    # Dynamic y-limits
    if final_accs:
        y_min = min(final_accs) * 0.98
        y_max = max(final_accs) * 1.02
        ax6.set_ylim([y_min, y_max])
    else:
        ax6.set_ylim([90, 100])

    # Panel 7: Memory Efficiency
    ax7 = plt.subplot(3, 3, 7)
    # Calculate memory factors based on optimizer and pruning
    memory_factors = []
    for label in labels:
        base = (
            3.0
            if optimizer in ["adam", "adamw", "adamwadv", "adamwspam", "adamwprune"]
            else 1.0
        )
        if "Movement" in label:
            overhead = 2.03
        elif "Magnitude" in label:
            overhead = 0.03
        elif "State" in label:
            overhead = 0.03
        else:
            overhead = 0.0
        memory_factors.append(base + overhead)

    bars = ax7.bar(range(len(labels)), memory_factors, color=colors)
    ax7.set_xticks(range(len(labels)))
    ax7.set_xticklabels(
        [l.replace(" ", "\n") for l in labels], rotation=45, ha="right", fontsize=8
    )
    ax7.set_ylabel("Memory (x weights)")
    ax7.set_title("Training Memory Requirements")
    ax7.grid(True, alpha=0.3, axis="y")

    # Panel 8: Inference Memory (with sparsity benefits)
    ax8 = plt.subplot(3, 3, 8)
    inference_mem = [
        1.0 * (1 - s / 100 * 0.5) for s in sparsities
    ]  # Conservative 50% compression
    bars = ax8.bar(range(len(labels)), inference_mem, color=colors)
    ax8.set_xticks(range(len(labels)))
    ax8.set_xticklabels(
        [l.replace(" ", "\n") for l in labels], rotation=45, ha="right", fontsize=8
    )
    ax8.set_ylabel("Memory (x weights)")
    ax8.set_title("Inference Memory (with sparsity)")
    ax8.grid(True, alpha=0.3, axis="y")

    # Panel 9: Efficiency Score (Accuracy / Training Memory)
    ax9 = plt.subplot(3, 3, 9)
    efficiency = [acc / mem for acc, mem in zip(final_accs, memory_factors)]
    bars = ax9.bar(range(len(labels)), efficiency, color=colors)
    ax9.set_xticks(range(len(labels)))
    ax9.set_xticklabels(
        [l.replace(" ", "\n") for l in labels], rotation=45, ha="right", fontsize=8
    )
    ax9.set_ylabel("Efficiency Score")
    ax9.set_title("Memory Efficiency (Accuracy/Memory)")
    ax9.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save figure 1
    model_comparison_path = os.path.join(
        output_dir, f"{optimizer}_model_comparison.png"
    )
    plt.savefig(model_comparison_path, dpi=100, bbox_inches="tight")
    print(f"  Saved: {model_comparison_path}")
    plt.close()

    # Create figure 2: Accuracy evolution focused view
    fig2 = plt.figure(figsize=(12, 8))
    fig2.suptitle(
        f"{display_name} Accuracy Evolution: {optimizer.upper()}",
        fontsize=16,
        fontweight="bold",
    )

    ax = plt.subplot(1, 1, 1)
    for label, data in plot_data.items():
        ax.plot(
            data["epochs"],
            data["accuracies"],
            marker="o",
            linewidth=2,
            label=label,
            markersize=6,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    # Dynamic y-limits with space for legend
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 0.98, min(100, y_max * 1.05))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure 2
    accuracy_evolution_path = os.path.join(
        output_dir, f"{optimizer}_accuracy_evolution.png"
    )
    plt.savefig(accuracy_evolution_path, dpi=100, bbox_inches="tight")
    print(f"  Saved: {accuracy_evolution_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        # Find most recent test results
        pattern = "test_matrix_results_*"
        dirs = sorted(Path(".").glob(pattern))
        if dirs:
            results_dir = str(dirs[-1])
            print(f"Using most recent results: {results_dir}")
        else:
            print("Error: No test_matrix_results_* directories found")
            print(
                "Usage: python generate_optimizer_graphs.py [results_dir] [output_dir]"
            )
            sys.exit(1)
    else:
        results_dir = sys.argv[1]

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(results_dir, "graphs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load test results
    results = load_test_results(results_dir)
    if not results:
        sys.exit(1)

    # Group by optimizer
    grouped = group_by_optimizer(results)

    print(f"Generating graphs for {len(grouped)} optimizers...")

    # Generate graphs for each optimizer
    for optimizer, tests in grouped.items():
        print(f"\nProcessing {optimizer}...")
        create_optimizer_comparison(optimizer, tests, results_dir, output_dir)

    print(f"\nAll graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
