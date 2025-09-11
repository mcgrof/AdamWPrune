#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Universal training plot generator for AdamWPrune experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path


def load_metrics(file_path):
    """Load training metrics from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{file_path}'.")
        return None


def extract_data(metrics):
    """Extract data from either LeNet-5 or ResNet-18 JSON format."""
    data = {}

    # Detect format and extract accordingly
    if "epochs" in metrics and isinstance(metrics["epochs"], list):
        # Check if it's LeNet-5 format (list of dicts) or ResNet-18 format (list of ints)
        if metrics["epochs"] and isinstance(metrics["epochs"][0], dict):
            # LeNet-5 format - epochs is a list of dictionaries
            epochs_data = metrics["epochs"]
            data["epochs"] = [e["epoch"] for e in epochs_data]
            data["accuracies"] = [e.get("accuracy", 0) for e in epochs_data]
            data["losses"] = [e.get("avg_loss", 0) for e in epochs_data]
            data["sparsities"] = [e.get("sparsity", 0) * 100 for e in epochs_data]
            data["times"] = [e.get("epoch_time", 0) for e in epochs_data]
        else:
            # ResNet-18 format - epochs is a list of integers
            data["epochs"] = metrics["epochs"]
            data["accuracies"] = metrics.get("test_accuracy", [])
            data["losses"] = metrics.get("train_loss", [])
            data["sparsities"] = [s * 100 for s in metrics.get("sparsity", [])]
            data["times"] = metrics.get("epoch_time", [])
    elif "test_accuracy" in metrics:
        # ResNet-18 format
        data["epochs"] = metrics.get(
            "epochs", list(range(1, len(metrics["test_accuracy"]) + 1))
        )
        data["accuracies"] = metrics.get("test_accuracy", [])
        data["losses"] = metrics.get("train_loss", [])
        data["sparsities"] = [s * 100 for s in metrics.get("sparsity", [])]
        data["times"] = metrics.get("epoch_time", [])
    else:
        print("Warning: Unrecognized metrics format")
        return None

    # Extract metadata
    data["model"] = "Neural Network"
    data["optimizer"] = "unknown"
    data["target_sparsity"] = 0

    if "config" in metrics:
        cfg = metrics["config"]
        data["model"] = cfg.get("model", metrics.get("model", "Neural Network"))
        data["optimizer"] = cfg.get("optimizer", metrics.get("optimizer", "unknown"))
        data["target_sparsity"] = (cfg.get("target_sparsity") or 0) * 100
    elif "model" in metrics:
        data["model"] = metrics["model"]
        data["optimizer"] = metrics.get("optimizer", "unknown")
        data["target_sparsity"] = (metrics.get("target_sparsity") or 0) * 100
    elif "optimizer" in metrics:
        data["optimizer"] = metrics["optimizer"]
        data["target_sparsity"] = (metrics.get("target_sparsity") or 0) * 100
    else:
        data["target_sparsity"] = 0

    # Get final stats
    data["final_accuracy"] = data["accuracies"][-1] if data["accuracies"] else 0
    data["final_sparsity"] = data["sparsities"][-1] if data["sparsities"] else 0
    data["best_accuracy"] = max(data["accuracies"]) if data["accuracies"] else 0

    return data


def create_plot(data, output_file):
    """Create a training plot."""
    # Determine subplot layout based on available data
    has_sparsity = any(s > 0 for s in data["sparsities"])
    num_plots = 4 if has_sparsity else 3

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Title
    title = f"{data['model'].upper()} Training - {data['optimizer'].upper()}"
    if data["target_sparsity"] > 0:
        title += f" (Target Sparsity: {data['target_sparsity']:.0f}%)"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot 1: Accuracy
    if data["accuracies"]:
        ax1 = axes[0]
        ax1.plot(data["epochs"], data["accuracies"], "b-o", linewidth=2, markersize=6)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax1.set_title("Test Accuracy Evolution", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([max(0, min(data["accuracies"]) - 5), 100])

        # Annotate final
        final_acc = data["final_accuracy"]
        ax1.annotate(
            f"Final: {final_acc:.2f}%",
            xy=(data["epochs"][-1], final_acc),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

    # Plot 2: Loss
    if data["losses"] and any(l > 0 for l in data["losses"]):
        ax2 = axes[1]
        valid_losses = [(e, l) for e, l in zip(data["epochs"], data["losses"]) if l > 0]
        if valid_losses:
            epochs_l, losses_l = zip(*valid_losses)
            ax2.plot(epochs_l, losses_l, "r-o", linewidth=2, markersize=6)
            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("Training Loss", fontsize=12)
            ax2.set_title("Training Loss", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale("log")

    # Plot 3: Sparsity (if available)
    plot_idx = 2
    if has_sparsity:
        ax3 = axes[plot_idx]
        ax3.plot(
            data["epochs"][: len(data["sparsities"])],
            data["sparsities"],
            "m-o",
            linewidth=2,
            markersize=6,
            label="Actual",
        )
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Sparsity (%)", fontsize=12)
        ax3.set_title("Model Sparsity Evolution", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])

        # Add target line
        if data["target_sparsity"] > 0:
            ax3.axhline(
                y=data["target_sparsity"],
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f'Target: {data["target_sparsity"]:.0f}%',
            )

        # Annotate final
        if data["sparsities"]:
            final_sp = data["final_sparsity"]
            ax3.annotate(
                f"Final: {final_sp:.1f}%",
                xy=(data["epochs"][len(data["sparsities"]) - 1], final_sp),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )
        ax3.legend(loc="best")
        plot_idx += 1

    # Plot 4: Training time
    if data["times"] and any(t > 0 for t in data["times"]):
        ax4 = axes[plot_idx]
        ax4.bar(
            data["epochs"][: len(data["times"])],
            data["times"],
            color="green",
            alpha=0.7,
        )
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("Time (seconds)", fontsize=12)
        ax4.set_title("Training Time per Epoch", fontsize=14)
        ax4.grid(True, alpha=0.3, axis="y")

        # Add average line
        avg_time = np.mean(data["times"])
        ax4.axhline(
            y=avg_time,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Avg: {avg_time:.2f}s",
        )
        ax4.legend()

    # Hide unused subplot if only 3 plots
    if not has_sparsity:
        axes[3].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate training plots from metrics JSON"
    )
    parser.add_argument("metrics_file", help="Path to training_metrics.json")
    parser.add_argument(
        "--output", default=None, help="Output filename (without extension)"
    )
    args = parser.parse_args()

    # Load metrics
    metrics = load_metrics(args.metrics_file)
    if metrics is None:
        sys.exit(1)

    # Extract data
    data = extract_data(metrics)
    if data is None:
        sys.exit(1)

    # Determine output filename
    if args.output:
        output_base = args.output
    else:
        output_base = Path(args.metrics_file).stem

    # Add _plot suffix if not present
    if not output_base.endswith("_plot"):
        output_file = f"{output_base}_plot.png"
    else:
        output_file = f"{output_base}.png"

    # Create plot
    create_plot(data, output_file)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
