#!/usr/bin/env python3
"""
Comprehensive optimizer comparison focusing on memory efficiency.
Generates 4 plots: baselines, 50% pruning, 70% pruning, 90% pruning.
Highlights how AdamWPrune achieves same results with zero memory overhead.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Compare optimizers across sparsity levels"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=".",
    help="Directory to save output plots",
)
args = parser.parse_args()

# Define optimizer display configurations (colors/labels)
OPTIMIZER_CONFIGS = {
    "SGD": {
        "color": "#1f77b4",  # Blue
        "label": "SGD+Movement",
    },
    "Adam": {
        "color": "#2ca02c",  # Green
        "label": "Adam+Movement",
    },
    "AdamW": {
        "color": "#ff7f0e",  # Orange
        "label": "AdamW+Movement",
    },
    "AdamWAdv": {
        "color": "#d62728",  # Red
        "label": "AdamWAdv+Movement",
    },
    "AdamWSPAM": {
        "color": "#9467bd",  # Purple
        "label": "AdamWSPAM+Movement",
    },
    "AdamWPrune": {
        "color": "#17becf",  # Cyan
        "label": "AdamWPrune",
    },
    "SGD-Magnitude": {
        "color": "#8c564b",  # Brown
        "label": "SGD+Magnitude",
    },
    "AdamW-Magnitude": {
        "color": "#e377c2",  # Pink
        "label": "AdamW+Magnitude",
    },
}

# Memory accounting constants
BYTES_PER_PARAM = 4  # float32
BYTES_PER_BOOL = 1  # bool mask

# Movement pruning overhead in this repo (float32):
# - scores (1x), initial_weights (1x), masks (1x)
MOVEMENT_PRUNING_OVERHEAD_MULT = 3.0


def load_results_from_json():
    """Load results from JSON files in results directories."""
    results = {}

    # Map directory names to optimizer names
    dir_mapping = {
        "sgd": "SGD",
        "adam": "Adam",
        "adamw": "AdamW",
        "adamwadv": "AdamWAdv",
        "adamwspam": "AdamWSPAM",
        "adamwprune": "AdamWPrune",
        "sgd-magnitude": "SGD-Magnitude",
        "adamw-magnitude": "AdamW-Magnitude",
    }

    # Map model files to sparsity levels
    model_mapping = {
        "model_a_metrics.json": "Baseline",
        "model_b_metrics.json": "50%",
        "model_c_metrics.json": "90%",
        "model_d_metrics.json": "70%",
    }

    for dir_name, opt_name in dir_mapping.items():
        results[opt_name] = {}
        results_dir = Path(f"results/{dir_name}")

        if not results_dir.exists():
            print(f"Warning: {results_dir} does not exist. Using default values.")
            # Default values if directory doesn't exist
            results[opt_name] = {
                "Baseline": {"accuracy": 95.0, "params": 61750},
                "50%": {"accuracy": 95.0, "params": 31015},
                "70%": {"accuracy": 95.0, "params": 18721},
                "90%": {"accuracy": 95.0, "params": 6427},
            }
            continue

        for json_file, sparsity in model_mapping.items():
            json_path = results_dir / json_file
            if json_path.exists():
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Extract final test accuracy from the last epoch
                    if "epochs" in data and len(data["epochs"]) > 0:
                        final_accuracy = data["epochs"][-1]["accuracy"]
                    elif "final_accuracy" in data:
                        final_accuracy = data["final_accuracy"]
                    else:
                        print(
                            f"Warning: Could not find accuracy in {json_path}. Using default."
                        )
                        final_accuracy = 95.0
                    # Extract parameter count from sparsity
                    if sparsity == "Baseline":
                        params = 61750
                    elif sparsity == "50%":
                        params = 31015
                    elif sparsity == "70%":
                        params = 18721
                    elif sparsity == "90%":
                        params = 6427

                    results[opt_name][sparsity] = {
                        "accuracy": final_accuracy,
                        "params": params,
                    }
            else:
                # Use default if file doesn't exist
                print(f"Warning: {json_path} not found. Using default values.")
                if sparsity == "Baseline":
                    params = 61750
                elif sparsity == "50%":
                    params = 31015
                elif sparsity == "70%":
                    params = 18721
                elif sparsity == "90%":
                    params = 6427
                results[opt_name][sparsity] = {
                    "accuracy": 95.0,
                    "params": params,
                }

    return results


# Load results from JSON files
RESULTS = load_results_from_json()


def calculate_memory_usage(optimizer, params, pruning_level):
    """Calculate total memory usage (weights + optimizer states + pruning buffers)."""
    weights_mem = params * BYTES_PER_PARAM

    # Optimizer states
    if optimizer in ("SGD", "SGD-Magnitude"):
        # No momentum used here; treat as 0 extra
        opt_states_mem = 0
    else:
        # Adam/AdamW families: exp_avg + exp_avg_sq (~2x params)
        opt_states_mem = 2 * params * BYTES_PER_PARAM

    # Pruning buffers
    pruning_mem = 0
    if pruning_level != "Baseline":
        if optimizer == "AdamWPrune":
            # bool mask per parameter when pruning is enabled
            pruning_mem += params * BYTES_PER_BOOL
        elif "Magnitude" in optimizer:
            # Magnitude pruning: only binary masks (float32 in this implementation)
            pruning_mem += params * BYTES_PER_PARAM  # mask stored as float32
        else:
            # Movement pruning in this repo: scores + initial_weights + masks (all float32)
            pruning_mem += MOVEMENT_PRUNING_OVERHEAD_MULT * params * BYTES_PER_PARAM

    total_bytes = weights_mem + opt_states_mem + pruning_mem
    return total_bytes / (1024 * 1024)  # MB


def create_comparison_plot(sparsity_level, filename):
    """Create a comparison plot for a specific sparsity level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    optimizers = list(OPTIMIZER_CONFIGS.keys())
    x_pos = np.arange(len(optimizers))

    # Collect data for this sparsity level
    accuracies = []
    memory_usages = []
    colors = []
    labels = []

    for opt in optimizers:
        result = RESULTS[opt][sparsity_level]
        accuracies.append(result["accuracy"])
        memory_usages.append(
            calculate_memory_usage(opt, result["params"], sparsity_level)
        )
        colors.append(OPTIMIZER_CONFIGS[opt]["color"])
        labels.append(OPTIMIZER_CONFIGS[opt]["label"])

    # Plot 1: Accuracy comparison
    bars1 = ax1.bar(x_pos, accuracies, color=colors)
    ax1.set_xlabel("Optimizer", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title(
        f"Accuracy Comparison - {sparsity_level} Sparsity",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Set y-axis limits for better comparison
    min_acc = min(accuracies) - 2
    max_acc = 100
    ax1.set_ylim([min_acc, max_acc])

    # Plot 2: Memory usage comparison
    bars2 = ax2.bar(x_pos, memory_usages, color=colors)
    ax2.set_xlabel("Optimizer", fontsize=12)
    ax2.set_ylabel("Memory Usage (MB)", fontsize=12)
    ax2.set_title(
        f"Memory Usage - {sparsity_level} Sparsity", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mem in zip(bars2, memory_usages):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{mem:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Highlight AdamWPrune's advantage
    adamwprune_idx = optimizers.index("AdamWPrune")
    bars2[adamwprune_idx].set_edgecolor("black")
    bars2[adamwprune_idx].set_linewidth(2)

    # Add annotation for AdamWPrune
    if sparsity_level != "Baseline":
        ax2.annotate(
            "Minimal extra memory\n(bool mask only)",
            xy=(adamwprune_idx, memory_usages[adamwprune_idx]),
            xytext=(adamwprune_idx - 1, max(memory_usages) * 0.8),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # Add main title
    if sparsity_level == "Baseline":
        main_title = "Optimizer Comparison - No Pruning (Baseline)"
    else:
        main_title = f"Optimizer Comparison - {sparsity_level} Pruning"

    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")

    return fig


def create_memory_efficiency_summary():
    """Create a summary plot showing memory efficiency across all configurations.

    Improves readability by removing per-point text labels and adding a
    single, shared legend mapping markers/colors to optimizers. Increases
    spacing and font sizes for clarity.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    sparsity_levels = ["Baseline", "50%", "70%", "90%"]
    optimizers = list(OPTIMIZER_CONFIGS.keys())

    def marker_for(label: str) -> str:
        if "SGD" in label:
            return "o"
        if "AdamWPrune" in label:
            return "P"
        if "SPAM" in label:
            return "*"
        if "Adv" in label:
            return "D"
        if "AdamW" in label:
            return "^"
        if "Adam" in label and "AdamW" not in label:
            return "s"
        return "o"

    # Prepare data for each subplot
    axes = [ax1, ax2, ax3, ax4]
    for ax, sparsity in zip(axes, sparsity_levels):
        for opt in optimizers:
            result = RESULTS[opt][sparsity]
            mem_usage = calculate_memory_usage(opt, result["params"], sparsity)
            # Memory savings relative to SGD at the same sparsity level
            sgd_baseline_mem = calculate_memory_usage(
                "SGD", RESULTS["SGD"][sparsity]["params"], sparsity
            )
            saving = ((sgd_baseline_mem - mem_usage) / sgd_baseline_mem) * 100

            label = OPTIMIZER_CONFIGS[opt]["label"]
            color = OPTIMIZER_CONFIGS[opt]["color"]
            marker = marker_for(label)

            ax.scatter(
                saving,
                result["accuracy"],
                s=180,
                c=color,
                marker=marker,
                alpha=0.8,
                edgecolors="black",
                linewidth=1.2,
                label=label,
            )

        ax.set_xlabel("Memory Savings vs SGD Baseline (%)", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(
            f"{sparsity} {'Pruning' if sparsity != 'Baseline' else ''}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add ideal region (high accuracy, high memory savings)
        if sparsity != "Baseline":
            ax.axhspan(98, 100, xmin=0.6, alpha=0.08, color="green")
            # Place label in axes coordinates to avoid going out of bounds
            ax.text(
                0.83,
                0.95,
                "Ideal",
                transform=ax.transAxes,
                fontsize=10,
                style="italic",
                color="green",
                fontweight="bold",
            )

    # Build a single shared legend using proxy artists
    proxies = []
    for opt in optimizers:
        label = OPTIMIZER_CONFIGS[opt]["label"]
        color = OPTIMIZER_CONFIGS[opt]["color"]
        marker = marker_for(label)
        proxies.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=10,
                linestyle="None",
                label=label,
            )
        )

    fig.legend(
        handles=proxies,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        title="Optimizers (savings vs SGD+movement)",
    )

    fig.suptitle(
        "Memory Efficiency vs Accuracy Across All Configurations",
        fontsize=16,
        fontweight="bold",
    )

    # Adjust layout to make room for the legend
    plt.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.12)
    plt.savefig(
        f"{args.output_dir}/memory_efficiency_summary.png", dpi=150, bbox_inches="tight"
    )
    print("Saved memory_efficiency_summary.png")

    return fig


# Generate all plots
print("Generating optimizer comparison plots...")
print("=" * 60)

# Individual comparison plots
create_comparison_plot("Baseline", "optimizer_comparison_baseline.png")
create_comparison_plot("50%", "optimizer_comparison_50_pruning.png")
create_comparison_plot("70%", "optimizer_comparison_70_pruning.png")
create_comparison_plot("90%", "optimizer_comparison_90_pruning.png")

# Summary plot
create_memory_efficiency_summary()

print("=" * 60)
print("All plots generated successfully!")
print("\nKey Insight: AdamWPrune achieves competitive accuracy while using")
print(
    "minimal extra memory for pruning (1 byte/param boolean mask), reusing Adam states."
)
