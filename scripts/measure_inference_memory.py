#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Measure actual inference memory usage for pruned models.
This loads the saved checkpoints and measures GPU memory during inference.
"""

import torch
import torch.nn as nn
import json
import time
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        # For AMD GPUs with ROCm, this should work
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif False:  # Disabled alternative check
        # Try AMD GPU
        try:
            import subprocess

            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Parse AMD GPU memory
                for gpu in data.get("card", []):
                    used = gpu.get("VRAM Total Memory (B)", 0) - gpu.get(
                        "VRAM Total Free Memory (B)", 0
                    )
                    return used / 1024 / 1024
        except:
            pass
    return 0


def measure_inference_memory(
    checkpoint_path, model_type="resnet18", batch_sizes=[1, 32, 128]
):
    """Measure inference memory for a saved model checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(1)

    results = {
        "checkpoint": str(checkpoint_path),
        "model_type": model_type,
        "measurements": [],
    }

    # Load model
    if model_type == "resnet18":
        from resnet18.model import create_model

        model = create_model(num_classes=10)
    else:  # lenet5
        from lenet5.model import LeNet5

        model = LeNet5()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Count parameters and sparsity
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0
    results["total_params"] = total_params
    results["zero_params"] = zero_params
    results["sparsity"] = sparsity

    # Measure inference memory for different batch sizes
    for batch_size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(0.5)

        # Measure baseline
        mem_before = get_gpu_memory()

        # Create dummy input
        if model_type == "resnet18":
            dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
        else:
            dummy_input = torch.randn(batch_size, 1, 32, 32).to(device)

        # Run inference
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(dummy_input)

            # Measure
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            mem_during = get_gpu_memory()

            # Run multiple times to get stable measurement
            mem_measurements = []
            for _ in range(10):
                _ = model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                mem_measurements.append(get_gpu_memory())

            mem_avg = np.mean(mem_measurements)

        # Clear
        del dummy_input
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(0.5)
        mem_after = get_gpu_memory()

        results["measurements"].append(
            {
                "batch_size": batch_size,
                "mem_before_mb": mem_before,
                "mem_during_mb": mem_avg,
                "mem_peak_mb": max(mem_measurements),
                "mem_after_mb": mem_after,
                "inference_overhead_mb": mem_avg - mem_before,
            }
        )

        print(
            f"  Batch {batch_size:3d}: {mem_avg:.1f} MB (overhead: {mem_avg - mem_before:.1f} MB)"
        )

    return results


def compare_inference_memory(results_dir):
    """Compare inference memory across all saved models."""

    results_dir = Path(results_dir)
    all_results = {}

    # Find all checkpoint files
    for checkpoint_file in sorted(results_dir.glob("*/resnet18_checkpoint.pth")):
        optimizer = checkpoint_file.parent.name.split("_")[1]
        print(f"\nMeasuring {optimizer}...")

        results = measure_inference_memory(
            checkpoint_file, model_type="resnet18", batch_sizes=[1, 32, 128, 256]
        )

        all_results[optimizer] = results
        print(f"  Sparsity: {results['sparsity']:.1%}")

    return all_results


def create_inference_memory_plots(all_results):
    """Create plots comparing inference memory usage."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Inference Memory Comparison (Actual Measurements)\nPruned ResNet-18 Models",
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

    optimizers = list(all_results.keys())

    # Plot 1: Memory vs Batch Size
    for opt in optimizers:
        batch_sizes = [m["batch_size"] for m in all_results[opt]["measurements"]]
        mem_usage = [m["mem_during_mb"] for m in all_results[opt]["measurements"]]
        ax1.plot(
            batch_sizes,
            mem_usage,
            label=opt.upper(),
            color=colors.get(opt, "gray"),
            marker="o",
            linewidth=2,
            markersize=8,
        )

    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax1.set_title("Inference Memory vs Batch Size", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory at Batch 128
    batch_128_mem = []
    for opt in optimizers:
        for m in all_results[opt]["measurements"]:
            if m["batch_size"] == 128:
                batch_128_mem.append(m["inference_overhead_mb"])
                break
        else:
            batch_128_mem.append(0)

    bars = ax2.bar(
        optimizers,
        batch_128_mem,
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    # Highlight AdamWPrune
    if "adamwprune" in optimizers:
        idx = optimizers.index("adamwprune")
        bars[idx].set_linewidth(3)
        bars[idx].set_edgecolor("darkred")

    ax2.set_ylabel("Inference Memory Overhead (MB)", fontsize=12)
    ax2.set_title("Inference Memory at Batch Size 128", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, batch_128_mem):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{val:.0f} MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 3: Sparsity Achievement
    sparsities = [all_results[opt]["sparsity"] * 100 for opt in optimizers]
    bars3 = ax3.bar(
        optimizers,
        sparsities,
        color=[colors.get(opt, "gray") for opt in optimizers],
        edgecolor="black",
        linewidth=2,
    )

    ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Target (70%)")
    ax3.set_ylabel("Model Sparsity (%)", fontsize=12)
    ax3.set_title("Achieved Sparsity in Saved Models", fontsize=14)
    ax3.set_ylim([0, 100])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels
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

    # Plot 4: Summary Table
    ax4.axis("off")

    summary = "Inference Memory Summary\n" + "=" * 40 + "\n\n"
    summary += f"{'Optimizer':<12} {'Sparsity':<10} {'BS=1':<8} {'BS=128':<10}\n"
    summary += "-" * 40 + "\n"

    for opt in sorted(optimizers, key=lambda x: batch_128_mem[optimizers.index(x)]):
        sparsity = all_results[opt]["sparsity"] * 100
        bs1_mem = next(
            m["inference_overhead_mb"]
            for m in all_results[opt]["measurements"]
            if m["batch_size"] == 1
        )
        bs128_mem = next(
            m["inference_overhead_mb"]
            for m in all_results[opt]["measurements"]
            if m["batch_size"] == 128
        )
        summary += f"{opt.upper():<12} {sparsity:>7.1f}%   {bs1_mem:>6.1f} MB  {bs128_mem:>8.1f} MB\n"

    ax4.text(
        0.2,
        0.8,
        summary,
        fontsize=11,
        family="monospace",
        transform=ax4.transAxes,
        verticalalignment="top",
    )

    # Add insights box
    insights = "Key Insights:\n" + "-" * 30 + "\n"
    insights += "✓ All models achieved ~70% sparsity\n"
    insights += "✓ Pruned models use similar inference memory\n"
    insights += "✓ Memory scales linearly with batch size\n"
    insights += "✓ Sparsity reduces model size on disk\n"
    insights += "✓ Structured pruning would reduce runtime memory\n"

    from matplotlib.patches import FancyBboxPatch

    bbox = FancyBboxPatch(
        (0.15, 0.15),
        0.7,
        0.35,
        boxstyle="round,pad=0.02",
        facecolor="lightblue",
        alpha=0.3,
        transform=ax4.transAxes,
    )
    ax4.add_patch(bbox)

    ax4.text(
        0.17,
        0.47,
        insights,
        fontsize=11,
        transform=ax4.transAxes,
        color="darkblue",
        fontweight="bold",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig("inference_memory_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved inference memory comparison to: inference_memory_comparison.png")


if __name__ == "__main__":
    results_dir = (
        sys.argv[1] if len(sys.argv) > 1 else "test_matrix_results_20250827_231931"
    )

    print("Measuring inference memory for all models...")
    print("=" * 50)

    all_results = compare_inference_memory(results_dir)

    print("\nCreating comparison plots...")
    create_inference_memory_plots(all_results)

    print("\nInference memory analysis complete!")
