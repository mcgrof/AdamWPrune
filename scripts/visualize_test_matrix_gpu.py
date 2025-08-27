#!/usr/bin/env python3
"""
Visualize GPU memory from a single test matrix run.
Only uses data from the specified test results directory.
"""

import json
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from glob import glob
import sys


def load_gpu_data(json_file):
    """Load GPU monitoring data from JSON file."""
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def extract_memory_stats(data):
    """Extract memory statistics from GPU monitoring data."""
    if not data:
        return None

    samples = None

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", data.get("stats", []))

    if not samples:
        return None

    memory_values = []
    for s in samples:
        # Try different field names for memory
        if "memory_used" in s:
            memory_values.append(s["memory_used"])
        elif "memory_mb" in s:
            memory_values.append(s["memory_mb"])
        elif "memory_used_mb" in s:
            memory_values.append(s["memory_used_mb"])

    if not memory_values:
        return None

    return {
        "mean": np.mean(memory_values),
        "std": np.std(memory_values),
        "max": max(memory_values),
        "min": min(memory_values),
        "samples": len(memory_values),
    }


def get_optimizer_from_dir(dir_name):
    """Extract optimizer name from directory name."""
    # Format: resnet18_optimizer_method_sparsity
    parts = dir_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def collect_test_matrix_data(test_dir):
    """Collect GPU data ONLY from the specified test matrix directory."""
    training_data = {}
    inference_data = {}

    # Find all subdirectories (one per test)
    test_dirs = [d for d in Path(test_dir).iterdir() if d.is_dir()]

    for test_subdir in test_dirs:
        test_name = test_subdir.name
        optimizer = get_optimizer_from_dir(test_name)

        # Look for GPU monitoring files in this test directory
        gpu_files = list(test_subdir.glob("gpu_*.json"))

        for gpu_file in gpu_files:
            filename = gpu_file.name

            # Determine if it's training or inference
            is_training = False
            is_inference = False

            if "training" in filename or "stats" in filename:
                is_training = True
            elif "inference" in filename:
                is_inference = True
            else:
                # Default to training if unclear
                is_training = True

            # Load and process data
            data = load_gpu_data(gpu_file)
            stats = extract_memory_stats(data)

            if stats:
                if is_training:
                    # Store training data
                    if (
                        optimizer not in training_data
                        or stats["mean"] < training_data[optimizer]["mean"]
                    ):
                        training_data[optimizer] = stats
                        training_data[optimizer]["test_name"] = test_name
                        training_data[optimizer]["file"] = str(gpu_file)
                elif is_inference:
                    # Store inference data
                    if (
                        optimizer not in inference_data
                        or stats["mean"] < inference_data[optimizer]["mean"]
                    ):
                        inference_data[optimizer] = stats
                        inference_data[optimizer]["test_name"] = test_name
                        inference_data[optimizer]["file"] = str(gpu_file)

    return training_data, inference_data


def create_test_matrix_visualization(training_data, inference_data, output_file):
    """Create visualization using only data from the same test run."""

    if not training_data:
        print("No training data found in test matrix results!")
        return None

    # Create figure
    fig = plt.figure(figsize=(18, 10))

    # Use GridSpec for layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

    # Sort optimizers by training memory
    optimizers = sorted(training_data.keys(), key=lambda x: training_data[x]["mean"])

    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(optimizers), 3)))
    color_map = dict(zip(optimizers, colors))

    # ========== Plot 1: Training Memory Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(optimizers))

    means = [training_data[opt]["mean"] for opt in optimizers]
    stds = [training_data[opt]["std"] for opt in optimizers]

    bars = ax1.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for i, (bar, std) in enumerate(zip(bars, stds)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 20,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax1.set_ylabel("Memory (MB)", fontsize=11)
    ax1.set_title("Training GPU Memory Usage", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [opt.upper() for opt in optimizers], rotation=45, ha="right", fontsize=10
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # ========== Plot 2: Inference Memory ==========
    ax2 = fig.add_subplot(gs[0, 1])

    if inference_data:
        # Only show inference data from the same test run
        infer_optimizers = sorted(
            inference_data.keys(), key=lambda x: inference_data[x]["mean"]
        )
        x_infer = np.arange(len(infer_optimizers))
        infer_means = [inference_data[opt]["mean"] for opt in infer_optimizers]
        infer_stds = [inference_data[opt]["std"] for opt in infer_optimizers]

        bars = ax2.bar(
            x_infer,
            infer_means,
            yerr=infer_stds,
            capsize=5,
            color=[color_map.get(opt, "gray") for opt in infer_optimizers],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        for i, (bar, std) in enumerate(zip(bars, infer_stds)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 20,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax2.set_ylabel("Memory (MB)", fontsize=11)
        ax2.set_title("Inference GPU Memory Usage", fontsize=12, fontweight="bold")
        ax2.set_xticks(x_infer)
        ax2.set_xticklabels(
            [opt.upper() for opt in infer_optimizers],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(
            0.5,
            0.5,
            "No Inference Data in Test Run",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax2.set_title("Inference GPU Memory Usage", fontsize=12, fontweight="bold")
        ax2.set_xticks([])

    # ========== Plot 3: Training vs Inference Comparison ==========
    ax3 = fig.add_subplot(gs[0, 2])

    # Only compare optimizers that have both training and inference data
    common_opts = sorted(
        [opt for opt in optimizers if opt in inference_data],
        key=lambda x: training_data[x]["mean"],
    )

    if common_opts:
        x_comp = np.arange(len(common_opts))
        width = 0.35

        train_means = [training_data[opt]["mean"] for opt in common_opts]
        infer_means = [inference_data[opt]["mean"] for opt in common_opts]

        bars1 = ax3.bar(
            x_comp - width / 2,
            train_means,
            width,
            label="Training",
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )
        bars2 = ax3.bar(
            x_comp + width / 2,
            infer_means,
            width,
            label="Inference",
            color="lightcoral",
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 10,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax3.set_ylabel("Memory (MB)", fontsize=11)
        ax3.set_title("Training vs Inference", fontsize=12, fontweight="bold")
        ax3.set_xticks(x_comp)
        ax3.set_xticklabels(
            [opt.upper() for opt in common_opts], rotation=45, ha="right", fontsize=10
        )
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(
            0.5,
            0.5,
            "No Common Data\nfor Comparison",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        ax3.set_title("Training vs Inference", fontsize=12, fontweight="bold")
        ax3.set_xticks([])

    # ========== Plot 4: Peak Memory ==========
    ax4 = fig.add_subplot(gs[1, 0])

    max_values = [training_data[opt]["max"] for opt in optimizers]

    bars = ax4.bar(
        x,
        max_values,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 10,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax4.set_ylabel("Max Memory (MB)", fontsize=11)
    ax4.set_title("Peak Training Memory", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [opt.upper() for opt in optimizers], rotation=45, ha="right", fontsize=10
    )
    ax4.grid(True, alpha=0.3, axis="y")

    # ========== Plot 5: Memory Efficiency ==========
    ax5 = fig.add_subplot(gs[1, 1])

    # Horizontal bar chart for efficiency ranking
    y_pos = np.arange(len(optimizers))
    values = [training_data[opt]["mean"] for opt in optimizers]

    bars = ax5.barh(
        y_pos,
        values,
        color=[color_map[opt] for opt in optimizers],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax5.text(
            val + 5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.0f} MB",
            va="center",
            fontsize=10,
        )

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([opt.upper() for opt in optimizers], fontsize=10)
    ax5.set_xlabel("Memory Usage (MB)", fontsize=11)
    ax5.set_title(
        "Efficiency Ranking (Lower is Better)", fontsize=12, fontweight="bold"
    )
    ax5.grid(True, alpha=0.3, axis="x")

    # ========== Plot 6: Summary Table ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Create summary table
    table_data = []
    table_data.append(["Optimizer", "Train Mean", "Train Max", "Samples"])

    for opt in optimizers:
        data = training_data[opt]
        row = [
            opt.upper(),
            f"{data['mean']:.1f}",
            f"{data['max']:.1f}",
            f"{data['samples']}",
        ]
        table_data.append(row)

    # Add summary
    table_data.append(["", "", "", ""])
    best_opt = optimizers[0]
    worst_opt = optimizers[-1]
    savings = training_data[worst_opt]["mean"] - training_data[best_opt]["mean"]
    savings_pct = (savings / training_data[worst_opt]["mean"]) * 100

    table_data.append(
        [
            "BEST",
            best_opt.upper(),
            f"{training_data[best_opt]['mean']:.1f} MB",
            f"-{savings:.0f} MB",
        ]
    )

    # Create table
    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best
    for j in range(4):
        table[(1, j)].set_facecolor("#90EE90")

    ax6.set_title("Summary", fontsize=12, fontweight="bold")

    # Main title
    fig.suptitle(
        "GPU Memory Analysis - Test Matrix Results", fontsize=16, fontweight="bold"
    )

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_file}")

    return optimizers


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GPU memory from test matrix results"
    )
    parser.add_argument("test_dir", nargs="?", help="Test matrix results directory")
    parser.add_argument(
        "--output",
        help="Output file for visualization (default: saves in test directory)",
    )

    args = parser.parse_args()

    # Find test directory
    if args.test_dir:
        test_dir = args.test_dir
    else:
        # Find most recent test_matrix_results directory
        test_dirs = sorted(glob("test_matrix_results_*"), reverse=True)
        if not test_dirs:
            print("Error: No test_matrix_results_* directories found!")
            print("Run 'make' first to generate test results.")
            sys.exit(1)
        test_dir = test_dirs[0]
        print(f"Using most recent test results: {test_dir}")

    if not Path(test_dir).exists():
        print(f"Error: Directory {test_dir} does not exist!")
        sys.exit(1)

    # Collect data ONLY from this test run
    training_data, inference_data = collect_test_matrix_data(test_dir)

    if not training_data and not inference_data:
        print(f"No GPU monitoring data found in {test_dir}")
        print("Make sure to run with CONFIG_GPU_MONITOR=y")
        sys.exit(1)

    print(f"Found {len(training_data)} training datasets")
    print(f"Found {len(inference_data)} inference datasets")

    # Default output path in test directory
    if args.output:
        output_file = args.output
    else:
        output_file = Path(test_dir) / "gpu_comparison.png"

    # Create visualization
    sorted_opts = create_test_matrix_visualization(
        training_data, inference_data, output_file
    )

    if sorted_opts:
        # Print summary
        print("\n" + "=" * 70)
        print(f"GPU Memory Summary from: {test_dir}")
        print("=" * 70)

        print("\nTraining Phase:")
        for opt in sorted_opts:
            stats = training_data[opt]
            print(
                f"  {opt.upper():12s}: {stats['mean']:7.1f} MB "
                f"(max: {stats['max']:7.1f} MB, samples: {stats['samples']})"
            )

        if inference_data:
            print("\nInference Phase:")
            for opt in sorted(inference_data.keys()):
                stats = inference_data[opt]
                print(
                    f"  {opt.upper():12s}: {stats['mean']:7.1f} MB "
                    f"(max: {stats['max']:7.1f} MB, samples: {stats['samples']})"
                )

        print("\n" + "=" * 70)
        best = sorted_opts[0]
        worst = sorted_opts[-1]
        savings = training_data[worst]["mean"] - training_data[best]["mean"]
        print(
            f"Most Efficient: {best.upper()} saves {savings:.1f} MB "
            f"({savings/training_data[worst]['mean']*100:.1f}%) vs {worst.upper()}"
        )


if __name__ == "__main__":
    main()
