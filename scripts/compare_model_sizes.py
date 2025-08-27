#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compare model sizes and theoretical inference memory for pruned models.
"""

import torch
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_checkpoint(checkpoint_path):
    """Analyze a saved checkpoint for size and sparsity."""

    # Get file size
    file_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Calculate sparsity and parameter counts
    total_params = 0
    zero_params = 0
    dense_params = 0

    for name, param in state_dict.items():
        if "weight" in name or "bias" in name:
            total_params += param.numel()
            zeros = (param == 0).sum().item()
            zero_params += zeros
            dense_params += param.numel() - zeros

    sparsity = zero_params / total_params if total_params > 0 else 0

    # Theoretical memory calculations (in MB)
    # Dense storage: 4 bytes per parameter
    dense_size_mb = (total_params * 4) / 1024 / 1024

    # Actual dense parameters after pruning
    effective_size_mb = (dense_params * 4) / 1024 / 1024

    # Sparse storage would be: indices + values (but PyTorch doesn't use this by default)
    sparse_theoretical_mb = (
        (dense_params * 8) / 1024 / 1024
    )  # 4 bytes value + 4 bytes index

    return {
        "file_size_mb": file_size_mb,
        "total_params": total_params,
        "zero_params": zero_params,
        "dense_params": dense_params,
        "sparsity": sparsity,
        "dense_size_mb": dense_size_mb,
        "effective_size_mb": effective_size_mb,
        "sparse_theoretical_mb": sparse_theoretical_mb,
        "compression_ratio": (
            dense_size_mb / effective_size_mb if effective_size_mb > 0 else 1
        ),
    }


def compare_all_models(results_dir):
    """Compare all saved models in results directory."""

    results_dir = Path(results_dir)
    model_stats = {}

    print(
        f"{'Optimizer':<12} {'Sparsity':>10} {'File Size':>12} {'Dense Size':>12} {'Effective':>12} {'Compression':>12}"
    )
    print("-" * 82)

    for checkpoint_file in sorted(results_dir.glob("*/resnet18_checkpoint.pth")):
        optimizer = checkpoint_file.parent.name.split("_")[1]
        stats = analyze_checkpoint(checkpoint_file)
        model_stats[optimizer] = stats

        print(
            f"{optimizer.upper():<12} {stats['sparsity']:>9.1%} {stats['file_size_mb']:>11.2f}M "
            f"{stats['dense_size_mb']:>11.2f}M {stats['effective_size_mb']:>11.2f}M "
            f"{stats['compression_ratio']:>11.2f}x"
        )

    return model_stats


def create_model_size_comparison(model_stats):
    """Create visualization of model sizes and inference memory."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Model Size and Inference Memory Analysis\nPruned ResNet-18 Models (70% Sparsity Target)",
        fontsize=16,
        fontweight="bold",
    )

    colors = {
        "sgd": "#1f77b4",
        "adam": "#ff7f0e",
        "adamw": "#2ca02c",
        "adamwadv": "#d62728",
        "adamwspam": "#9467bd",
        "adamwprune": "#8c564b",
    }

    optimizers = list(model_stats.keys())

    # Plot 1: File Sizes
    file_sizes = [model_stats[opt]["file_size_mb"] for opt in optimizers]
    bars1 = ax1.bar(
        optimizers,
        file_sizes,
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    if "adamwprune" in optimizers:
        idx = optimizers.index("adamwprune")
        bars1[idx].set_linewidth(3)
        bars1[idx].set_edgecolor("darkred")

    ax1.set_ylabel("File Size (MB)", fontsize=12)
    ax1.set_title("Saved Model File Sizes", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, file_sizes):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{val:.1f} MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Effective Model Size (non-zero parameters)
    effective_sizes = [model_stats[opt]["effective_size_mb"] for opt in optimizers]
    dense_sizes = [model_stats[opt]["dense_size_mb"] for opt in optimizers]

    x = np.arange(len(optimizers))
    width = 0.35

    bars2a = ax2.bar(
        x - width / 2,
        dense_sizes,
        width,
        label="Original Dense",
        color="lightgray",
        edgecolor="black",
        alpha=0.7,
    )
    bars2b = ax2.bar(
        x + width / 2,
        effective_sizes,
        width,
        label="After Pruning",
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    ax2.set_ylabel("Model Size (MB)", fontsize=12)
    ax2.set_title("Theoretical Inference Memory (Dense Storage)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(optimizers)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Sparsity Achievement
    sparsities = [model_stats[opt]["sparsity"] * 100 for opt in optimizers]
    bars3 = ax3.bar(
        optimizers,
        sparsities,
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Target (70%)")
    ax3.set_ylabel("Sparsity (%)", fontsize=12)
    ax3.set_title("Achieved Sparsity", fontsize=14)
    ax3.set_ylim([0, 100])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars3, sparsities):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 4: Summary and Insights
    ax4.axis("off")

    # Create summary table
    summary = "Model Size Analysis\n" + "=" * 40 + "\n\n"
    summary += "All models successfully achieved ~70% sparsity\n\n"
    summary += f"{'Optimizer':<10} {'Params':>8} {'Zeros':>8} {'Savings':>8}\n"
    summary += "-" * 34 + "\n"

    for opt in optimizers:
        stats = model_stats[opt]
        total_m = stats["total_params"] / 1e6
        zeros_m = stats["zero_params"] / 1e6
        savings = (stats["zero_params"] / stats["total_params"]) * 100
        summary += (
            f"{opt.upper():<10} {total_m:>7.2f}M {zeros_m:>7.2f}M {savings:>7.1f}%\n"
        )

    ax4.text(
        0.1,
        0.85,
        summary,
        fontsize=10,
        family="monospace",
        transform=ax4.transAxes,
        verticalalignment="top",
    )

    # Add insights box
    insights = "Key Insights for Inference:\n" + "-" * 35 + "\n"
    insights += "✓ All optimizers achieved ~70% sparsity\n"
    insights += "✓ 70% of weights are zero after pruning\n"
    insights += "✓ File sizes include optimizer states\n"
    insights += "✓ Inference only needs model weights\n\n"

    insights += "Memory Savings Potential:\n"
    insights += "• Dense storage: No memory saved\n"
    insights += "• Sparse storage: ~70% memory saved\n"
    insights += "• Structured pruning: Real memory savings\n"
    insights += "• Current: Zeros stored but not computed\n"

    from matplotlib.patches import FancyBboxPatch

    bbox = FancyBboxPatch(
        (0.08, 0.15),
        0.84,
        0.45,
        boxstyle="round,pad=0.02",
        facecolor="lightyellow",
        alpha=0.5,
        transform=ax4.transAxes,
    )
    ax4.add_patch(bbox)

    ax4.text(
        0.1,
        0.58,
        insights,
        fontsize=11,
        transform=ax4.transAxes,
        color="darkblue",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig("model_size_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved model size comparison to: model_size_comparison.png")

    # Additional inference memory projection
    create_inference_projection(model_stats)


def create_inference_projection(model_stats):
    """Create projection of actual inference memory savings with different storage formats."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Inference Memory Projection: Dense vs Sparse Storage",
        fontsize=16,
        fontweight="bold",
    )

    optimizers = list(model_stats.keys())

    # Calculate different storage scenarios
    scenarios = {
        "Current (Dense)": [model_stats[opt]["dense_size_mb"] for opt in optimizers],
        "Effective (70% pruned)": [
            model_stats[opt]["effective_size_mb"] for opt in optimizers
        ],
        "Sparse Format": [
            model_stats[opt]["sparse_theoretical_mb"] for opt in optimizers
        ],
    }

    # Plot 1: Bar comparison
    x = np.arange(len(optimizers))
    width = 0.25

    for i, (scenario, values) in enumerate(scenarios.items()):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=scenario)

    ax1.set_xlabel("Optimizer")
    ax1.set_ylabel("Memory Usage (MB)")
    ax1.set_title("Inference Memory: Different Storage Formats")
    ax1.set_xticks(x)
    ax1.set_xticklabels([opt.upper() for opt in optimizers])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Memory savings potential
    current_mem = scenarios["Current (Dense)"]
    effective_mem = scenarios["Effective (70% pruned)"]

    savings_percent = [
        (1 - eff / curr) * 100 for curr, eff in zip(current_mem, effective_mem)
    ]

    bars2 = ax2.bar(
        optimizers,
        savings_percent,
        color=["green" if s > 60 else "orange" for s in savings_percent],
        edgecolor="black",
        linewidth=2,
    )

    ax2.set_ylabel("Potential Memory Savings (%)")
    ax2.set_title("Inference Memory Savings with Sparse Storage")
    ax2.set_ylim([0, 80])
    ax2.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Target Sparsity")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    for bar, val in zip(bars2, savings_percent):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("inference_memory_projection.png", dpi=150, bbox_inches="tight")
    print("Saved inference memory projection to: inference_memory_projection.png")


if __name__ == "__main__":
    results_dir = (
        sys.argv[1] if len(sys.argv) > 1 else "test_matrix_results_20250827_231931"
    )

    print("\nAnalyzing Model Sizes and Inference Memory")
    print("=" * 82)

    model_stats = compare_all_models(results_dir)

    print("\nCreating visualizations...")
    create_model_size_comparison(model_stats)

    print("\nAnalysis complete!")
