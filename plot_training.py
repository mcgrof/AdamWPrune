#!/usr/bin/env python3
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
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{file_path}'.")
        sys.exit(1)


def plot_single_training(metrics, output_prefix="training"):
    """Create plots for a single training run."""
    epochs_data = metrics["epochs"]
    epochs = [e["epoch"] for e in epochs_data]
    accuracies = [e["accuracy"] for e in epochs_data]
    avg_losses = [e["avg_loss"] for e in epochs_data if e["avg_loss"] > 0]
    epoch_times = [e["epoch_time"] for e in epochs_data]

    # Create figure with subplots - now 2x3 for additional loss detail plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("LeNet-5 Training Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Accuracy over epochs
    ax1 = axes[0, 0]
    ax1.plot(epochs, accuracies, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Test Accuracy Evolution", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(accuracies) - 1, 100])

    # Add final accuracy annotation
    final_acc = accuracies[-1]
    ax1.annotate(
        f"Final: {final_acc:.2f}%",
        xy=(epochs[-1], final_acc),
        xytext=(epochs[-1] - 1, final_acc - 2),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    # Plot 2: Loss over epochs (average per epoch)
    ax2 = axes[0, 1]
    if avg_losses:
        loss_epochs = epochs[: len(avg_losses)]
        ax2.plot(loss_epochs, avg_losses, "r-o", linewidth=2, markersize=8)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Average Loss", fontsize=12)
        ax2.set_title("Training Loss Evolution (Per Epoch)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    # NEW Plot 3: Detailed loss curve (all loss points, especially for hockey stick)
    ax3 = axes[0, 2]
    all_losses = []
    all_steps = []
    step_counter = 0

    for epoch_data in epochs_data:
        if "losses" in epoch_data and epoch_data["losses"]:
            epoch_losses = epoch_data["losses"]
            # Calculate steps for this epoch
            steps_in_epoch = len(epoch_losses)
            for i, loss in enumerate(epoch_losses):
                all_losses.append(loss)
                # Calculate global step number
                all_steps.append(step_counter + i + 1)
            step_counter += steps_in_epoch

    if all_losses:
        ax3.plot(all_steps, all_losses, "g-", linewidth=2, alpha=0.8)
        ax3.scatter(
            all_steps[:20],
            all_losses[:20],
            c="red",
            s=20,
            alpha=0.6,
            label="Early training",
        )
        ax3.set_xlabel("Training Step", fontsize=12)
        ax3.set_ylabel("Loss", fontsize=12)
        ax3.set_title("Detailed Loss Curve (Hockey Stick View)", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

        # Add theoretical initial loss line
        theoretical_loss = 2.303  # -log(1/10) for 10 classes
        ax3.axhline(
            y=theoretical_loss,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"Theoretical init: {theoretical_loss:.3f}",
        )
        ax3.legend()

    # Plot 4: Epoch training time
    ax4 = axes[1, 0]
    ax4.bar(epochs, epoch_times, color="green", alpha=0.7)
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Time (seconds)", fontsize=12)
    ax4.set_title("Training Time per Epoch", fontsize=14)
    ax4.grid(True, alpha=0.3, axis="y")

    # Add average time line
    avg_time = np.mean(epoch_times)
    ax4.axhline(y=avg_time, color="red", linestyle="--", label=f"Avg: {avg_time:.2f}s")
    ax4.legend()

    # Plot 5: Summary statistics
    ax5 = axes[1, 1]
    ax5.axis("off")

    # Create summary text
    summary_text = f"""Training Summary:
    
• Model: LeNet-5
• Device: {metrics.get('device', {}).get('type', 'N/A').upper()}
• GPU: {metrics.get('device', {}).get('name', 'N/A')}
• Batch Size: {metrics.get('config', {}).get('batch_size', 'N/A')}
• Learning Rate: {metrics.get('config', {}).get('learning_rate', 'N/A')}
• Epochs: {metrics.get('config', {}).get('num_epochs', 'N/A')}

Performance:
• Final Accuracy: {final_acc:.2f}%
• Best Accuracy: {max(accuracies):.2f}% (Epoch {epochs[accuracies.index(max(accuracies))]})
• Total Training Time: {metrics.get('total_training_time', 0):.2f}s
• Avg Time/Epoch: {metrics.get('avg_time_per_epoch', 0):.2f}s

Dataset:
• Training Samples: {metrics.get('dataset', {}).get('train_samples', 'N/A'):,}
• Test Samples: {metrics.get('dataset', {}).get('test_samples', 'N/A'):,}
"""

    ax5.text(
        0.1,
        0.9,
        summary_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Plot 6: Zoomed early training (first 10 steps)
    ax6 = axes[1, 2]
    if all_losses and len(all_losses) > 0:
        # Plot first 10-20 steps for detailed hockey stick view
        early_steps = min(10, len(all_losses))
        ax6.plot(
            all_steps[:early_steps],
            all_losses[:early_steps],
            "go-",
            linewidth=2,
            markersize=8,
        )
        ax6.set_xlabel("Training Step", fontsize=12)
        ax6.set_ylabel("Loss", fontsize=12)
        ax6.set_title("Early Training Detail (First 10 Steps)", fontsize=14)
        ax6.grid(True, alpha=0.3)

        # Add theoretical initial loss line
        ax6.axhline(
            y=theoretical_loss,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"Theoretical: {theoretical_loss:.3f}",
        )

        # Annotate the drop
        if len(all_losses) > 1:
            initial_loss = all_losses[0]
            final_early_loss = all_losses[min(9, len(all_losses) - 1)]
            drop_percent = ((initial_loss - final_early_loss) / initial_loss) * 100
            ax6.annotate(
                f"Drop: {drop_percent:.1f}%",
                xy=(5, (initial_loss + final_early_loss) / 2),
                fontsize=10,
                color="blue",
            )
        ax6.legend()
    else:
        ax6.axis("off")
        ax6.text(
            0.5,
            0.5,
            "No detailed loss data available",
            transform=ax6.transAxes,
            ha="center",
            va="center",
        )

    # Adjust layout and save
    plt.tight_layout()
    output_file = f"{output_prefix}_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Show plot
    plt.show()


def plot_comparison(metrics_list, labels, output_prefix="comparison"):
    """Create comparison plots for multiple training runs (for A/B testing)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Comparison (A/B Testing)", fontsize=16, fontweight="bold")

    colors = ["blue", "red", "green", "orange", "purple"]

    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        epochs_data = metrics["epochs"]
        epochs = [e["epoch"] for e in epochs_data]
        accuracies = [e["accuracy"] for e in epochs_data]
        ax1.plot(
            epochs,
            accuracies,
            f"{colors[i % len(colors)]}-o",
            linewidth=2,
            markersize=6,
            label=label,
            alpha=0.8,
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy Comparison", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Loss comparison
    ax2 = axes[0, 1]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        epochs_data = metrics["epochs"]
        epochs = [e["epoch"] for e in epochs_data]
        avg_losses = [e["avg_loss"] for e in epochs_data if e["avg_loss"] > 0]
        if avg_losses:
            loss_epochs = epochs[: len(avg_losses)]
            ax2.plot(
                loss_epochs,
                avg_losses,
                f"{colors[i % len(colors)]}-o",
                linewidth=2,
                markersize=6,
                label=label,
                alpha=0.8,
            )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Average Loss", fontsize=12)
    ax2.set_title("Loss Comparison", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    ax2.legend()

    # Plot 3: Final accuracy bar chart
    ax3 = axes[1, 0]
    final_accuracies = []
    for metrics in metrics_list:
        epochs_data = metrics["epochs"]
        accuracies = [e["accuracy"] for e in epochs_data]
        final_accuracies.append(accuracies[-1] if accuracies else 0)

    x_pos = np.arange(len(labels))
    bars = ax3.bar(x_pos, final_accuracies, color=colors[: len(labels)])
    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("Final Accuracy (%)", fontsize=12)
    ax3.set_title("Final Accuracy Comparison", fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
        )

    # Plot 4: Training time comparison
    ax4 = axes[1, 1]
    total_times = []
    for metrics in metrics_list:
        total_times.append(metrics.get("total_training_time", 0))

    bars = ax4.bar(x_pos, total_times, color=colors[: len(labels)])
    ax4.set_xlabel("Model", fontsize=12)
    ax4.set_ylabel("Total Training Time (s)", fontsize=12)
    ax4.set_title("Training Time Comparison", fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, time in zip(bars, total_times):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time:.1f}s",
            ha="center",
            va="bottom",
        )

    # Adjust layout and save
    plt.tight_layout()
    output_file = f"{output_prefix}_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {output_file}")

    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot LeNet-5 training metrics for analysis and A/B testing"
    )
    parser.add_argument(
        "metrics_files", nargs="+", help="Path(s) to training_metrics.json file(s)"
    )
    parser.add_argument(
        "--labels", nargs="+", help="Labels for each metrics file (for comparison)"
    )
    parser.add_argument(
        "--output", default="training", help="Output file prefix (default: training)"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display the plot, only save it"
    )

    args = parser.parse_args()

    # Disable showing if requested
    if args.no_show:
        plt.switch_backend("Agg")

    # Load metrics from all files
    metrics_list = []
    for file_path in args.metrics_files:
        metrics_list.append(load_metrics(file_path))

    # Determine labels
    if args.labels:
        if len(args.labels) != len(args.metrics_files):
            print(
                f"Error: Number of labels ({len(args.labels)}) must match number of files ({len(args.metrics_files)})"
            )
            sys.exit(1)
        labels = args.labels
    else:
        # Use file names as labels
        labels = [Path(f).stem for f in args.metrics_files]

    # Create plots
    if len(metrics_list) == 1:
        # Single training run
        plot_single_training(metrics_list[0], args.output)
    else:
        # Multiple training runs - create comparison
        plot_comparison(metrics_list, labels, args.output)


if __name__ == "__main__":
    main()
