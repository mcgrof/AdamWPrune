#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_metrics(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def extract_data(metrics):
    epochs = []
    accuracies = []
    losses = []
    sparsities = []

    for epoch_data in metrics["epochs"]:
        epochs.append(epoch_data["epoch"])
        accuracies.append(epoch_data["accuracy"])
        losses.append(epoch_data["avg_loss"])
        if "sparsity" in epoch_data:
            sparsities.append(epoch_data["sparsity"] * 100)
        else:
            sparsities.append(0)

    config = metrics["config"]
    final_acc = metrics["final_accuracy"]
    training_time = metrics["total_training_time"]

    # Get compression ratio
    if "total_params" in metrics and "non_zero_params" in metrics:
        compression = metrics["total_params"] / metrics["non_zero_params"]
    else:
        compression = 1.0

    return {
        "epochs": epochs,
        "accuracies": accuracies,
        "losses": losses,
        "sparsities": sparsities,
        "final_accuracy": final_acc,
        "training_time": training_time,
        "compression_ratio": compression,
        "pruning_method": config.get("pruning_method", "none"),
        "target_sparsity": config.get("target_sparsity", 0),
        "optimizer": config.get("optimizer", "sgd"),
    }


parser = argparse.ArgumentParser(description="LeNet-5 training with optional pruning")
parser.add_argument(
    "--test-prefix",
    type=str,
    default="SGD",
    choices=["SGD", "Adam", "AdamW", "AdamWAdv", "AdamWSPAM", "AdamWPrune"],
    help='Prefix to use for tests labels (default: "SGD")',
)
parser.add_argument(
    "--compare-output",
    type=str,
    default="model_comparison.png",
    help="Where to output model comparison graph metrics",
)
parser.add_argument(
    "--accuracy-output",
    type=str,
    default="accuracy_evolution.png",
    help="Where to output model accuracy evolution graph metrics",
)
args = parser.parse_args()

# Load all model metrics
models = {
    f"{args.test_prefix} Baseline": load_metrics("model_a_metrics.json"),
    f"{args.test_prefix} 50% Pruning": load_metrics("model_b_metrics.json"),
    f"{args.test_prefix} 90% Pruning": load_metrics("model_c_metrics.json"),
    f"{args.test_prefix} 70% Pruning": load_metrics("model_d_metrics.json"),
}

# Extract data for each model
model_data = {}
for name, metrics in models.items():
    model_data[name] = extract_data(metrics)

# Sort models by sparsity for better visualization
sorted_models = sorted(
    model_data.items(),
    key=lambda x: x[1]["target_sparsity"] if x[1]["target_sparsity"] is not None else 0,
)

# Create a full comparison figure
fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    f"LeNet-5 Model Comparison: {args.test_prefix} vs Movement Pruning",
    fontsize=16,
    fontweight="bold",
)

# Define colors for different optimizers
sgd_colors = ["#1f77b4", "#aec7e8", "#174a7e", "#6baed6"]  # Blues
adam_colors = ["#2ca02c", "#98df8a", "#27ae60", "#52c77e"]  # Greens
adamw_colors = ["#ff7f0e", "#ffbb78", "#ff9800", "#ffc947"]  # Oranges
adamwadv_colors = ["#d62728", "#ff9896", "#e74c3c", "#ffb3ba"]  # Reds
adamwspam_colors = ["#9467bd", "#c5b0d5", "#8c564b", "#c49c94"]  # Purples
adamwprune_colors = ["#17becf", "#9edae5", "#bcbd22", "#dbdb8d"]  # Cyans/Yellow-greens


def get_color_and_style(name):
    # Determine the optimizer type from the name
    if "AdamWPrune" in name or "adamwprune" in name.lower():
        colors = adamwprune_colors
    elif "AdamWSPAM" in name or "adamwspam" in name.lower():
        colors = adamwspam_colors
    elif "AdamWAdv" in name or "adamwadv" in name.lower():
        colors = adamwadv_colors
    elif "AdamW" in name or "adamw" in name.lower():
        colors = adamw_colors
    elif "Adam" in name or "adam" in name.lower():
        colors = adam_colors
    else:  # SGD
        colors = sgd_colors

    # Determine which variant (baseline, 50%, 70%, 90%) with distinct line styles
    if "Baseline" in name:
        return colors[0], "-"  # Solid line
    elif "50%" in name:
        return colors[1], "--"  # Dashed line
    elif "70%" in name:
        return colors[2], "-."  # Dash-dot line
    else:  # 90%
        return colors[3], ":"  # Dotted line


# 1. Accuracy over epochs
ax1 = plt.subplot(2, 3, 1)
for name, data in sorted_models:
    color, style = get_color_and_style(name)
    ax1.plot(
        data["epochs"],
        data["accuracies"],
        marker="o",
        label=name,
        linewidth=2.5,
        markersize=4,
        color=color,
        linestyle=style,
    )
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_title("Test Accuracy During Training")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Loss over epochs
ax2 = plt.subplot(2, 3, 2)
for name, data in sorted_models:
    color, style = get_color_and_style(name)
    ax2.plot(
        data["epochs"],
        data["losses"],
        marker="s",
        label=name,
        linewidth=2.5,
        markersize=4,
        color=color,
        linestyle=style,
    )
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Training Loss")
ax2.set_title("Training Loss Convergence")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_yscale("log")

# 3. Sparsity progression (for pruned models)
ax3 = plt.subplot(2, 3, 3)
for name, data in sorted_models:
    if data["pruning_method"] != "none":
        color, style = get_color_and_style(name)
        ax3.plot(
            data["epochs"],
            data["sparsities"],
            marker="^",
            label=name,
            linewidth=2.5,
            markersize=4,
            color=color,
            linestyle=style,
        )
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Sparsity (%)")
ax3.set_title("Sparsity Progression During Training")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Final accuracy comparison
ax4 = plt.subplot(2, 3, 4)
names = [name.replace(" Pruning", "\nPruning") for name, _ in sorted_models]
final_accs = [data["final_accuracy"] for _, data in sorted_models]
bar_colors = [get_color_and_style(name)[0] for name, _ in sorted_models]
bars = ax4.bar(range(len(names)), final_accs, color=bar_colors)
ax4.set_xticks(range(len(names)))
ax4.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
ax4.set_ylabel("Final Test Accuracy (%)")
ax4.set_title("Final Model Accuracy Comparison")
ax4.set_ylim([94, 100])
for i, (bar, acc) in enumerate(zip(bars, final_accs)):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )
ax4.grid(True, alpha=0.3, axis="y")

# 5. Compression ratio comparison
ax5 = plt.subplot(2, 3, 5)
compression_ratios = [data["compression_ratio"] for _, data in sorted_models]
bars = ax5.bar(range(len(names)), compression_ratios, color=bar_colors)
ax5.set_xticks(range(len(names)))
ax5.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
ax5.set_ylabel("Compression Ratio")
ax5.set_title("Model Compression Achieved")
for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{ratio:.2f}x",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )
ax5.grid(True, alpha=0.3, axis="y")

# 6. Accuracy vs Compression trade-off
ax6 = plt.subplot(2, 3, 6)
compression_ratios = [data["compression_ratio"] for _, data in sorted_models]
final_accs = [data["final_accuracy"] for _, data in sorted_models]
labels = [name for name, _ in sorted_models]

for i, (comp, acc, label) in enumerate(zip(compression_ratios, final_accs, labels)):
    color = get_color_and_style(label)[0]
    # Different markers for different optimizers
    if "SGD" in label or "sgd" in label.lower():
        marker = "o"
    elif "AdamWPrune" in label or "adamwprune" in label.lower():
        marker = "P"  # Plus (filled)
    elif "AdamWSPAM" in label or "adamwspam" in label.lower():
        marker = "*"  # Star
    elif "AdamWAdv" in label or "adamwadv" in label.lower():
        marker = "D"  # Diamond
    elif "AdamW" in label or "adamw" in label.lower():
        marker = "^"  # Triangle up
    else:  # Adam
        marker = "s"  # Square
    ax6.scatter(
        comp,
        acc,
        s=200,
        c=color,
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
        marker=marker,
    )
    # Shorter labels for the plot
    short_label = label.replace(" Baseline", "").replace(" Pruning", "")
    ax6.annotate(
        short_label, (comp, acc), xytext=(5, 5), textcoords="offset points", fontsize=7
    )

ax6.set_xlabel("Compression Ratio")
ax6.set_ylabel("Test Accuracy (%)")
ax6.set_title("Accuracy vs Compression Trade-off")
ax6.grid(True, alpha=0.3)
ax6.set_ylim([94, 100])

plt.tight_layout()
plt.savefig(args.compare_output, dpi=150, bbox_inches="tight")
print(f"Saved full comparison plot as {args.compare_output}")

# Create a detailed performance table
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 80)
print(
    f"{'Model':<25} {'Accuracy':<12} {'Compression':<12} {'Parameters':<15} {'Training Time':<12}"
)
print("-" * 80)

for name, data in sorted_models:
    if data["pruning_method"] != "none":
        params = f"{models[name]['non_zero_params']:,}/{models[name]['total_params']:,}"
    else:
        params = "61,750/61,750"

    print(
        f"{name:<25} {data['final_accuracy']:<12.2f}% {data['compression_ratio']:<12.2f}x "
        f"{params:<15} {data['training_time']:<12.2f}s"
    )

print("=" * 80)

# Create individual accuracy evolution plot
fig2, ax = plt.subplots(figsize=(14, 7))
fig2.suptitle(
    f"Test Accuracy Evolution: {args.test_prefix} vs Movement Pruning",
    fontsize=14,
    fontweight="bold",
)

for name, data in sorted_models:
    color, style = get_color_and_style(name)
    # Different markers for different optimizers
    if "SGD" in name or "sgd" in name.lower():
        marker = "o"
    elif "AdamWPrune" in name or "adamwprune" in name.lower():
        marker = "P"  # Plus (filled)
    elif "AdamWSPAM" in name or "adamwspam" in name.lower():
        marker = "*"  # Star
    elif "AdamWAdv" in name or "adamwadv" in name.lower():
        marker = "D"  # Diamond
    elif "AdamW" in name or "adamw" in name.lower():
        marker = "^"  # Triangle up
    else:  # Adam
        marker = "s"  # Square
    ax.plot(
        data["epochs"],
        data["accuracies"],
        marker=marker,
        label=name,
        linewidth=3.0,
        markersize=6,
        linestyle=style,
        color=color,
    )

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([94, 100])

# Add annotations for key insights
ax.axhline(y=98, color="gray", linestyle=":", alpha=0.5)
ax.text(10.5, 98, "98% accuracy threshold", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(args.accuracy_output, dpi=150, bbox_inches="tight")
print(f"\nSaved accuracy evolution plot as {args.accuracy_output}")

print("\nPlots generated successfully!")
print(f"- {args.compare_output}: Comprehensive 6-panel comparison")
print(f"- {args.accuracy_output}: Detailed accuracy evolution plot")
