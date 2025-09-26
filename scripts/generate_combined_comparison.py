#!/usr/bin/env python3
"""
Generate combined comparison graphs showing all pruning methods including AdamWPrune.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(results_dir, test_id):
    """Load metrics for a specific test."""
    metrics_file = os.path.join(results_dir, test_id, "training_metrics.json")

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def create_combined_accuracy_evolution(results_dir, output_dir):
    """Create a graph comparing all pruning methods including AdamWPrune."""

    # Load all_results.json
    results_file = os.path.join(results_dir, "all_results.json")
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return

    with open(results_file, 'r') as f:
        all_results = json.load(f)

    plt.figure(figsize=(14, 8))

    # Colors for different methods
    colors = {
        'adam_none': 'blue',
        'adam_magnitude': 'orange',
        'adam_movement': 'green',
        'adamwprune_none': 'purple',
        'adamwprune_state': 'red'
    }

    # Process each test result
    for result in all_results:
        optimizer = result.get('optimizer', 'unknown')
        pruning = result.get('pruning_method', 'none')
        sparsity = int(result.get('target_sparsity', 0) * 100)

        # Create test ID
        model = result.get('model', 'resnet18')
        if pruning == 'none':
            test_id = f"{model}_{optimizer}_{pruning}"
        else:
            test_id = f"{model}_{optimizer}_{pruning}_{sparsity}"

        # Load metrics
        metrics = load_metrics(results_dir, test_id)
        if not metrics:
            continue

        # Extract accuracy data
        if 'test_accuracy' in metrics:
            accuracies = metrics['test_accuracy']
            epochs = list(range(1, len(accuracies) + 1))
        else:
            continue

        # Determine label and color
        if optimizer == 'adam' and pruning == 'none':
            label = 'Adam Baseline'
            color = colors['adam_none']
        elif optimizer == 'adam' and pruning == 'magnitude':
            label = f'Adam Magnitude {sparsity}%'
            color = colors['adam_magnitude']
        elif optimizer == 'adam' and pruning == 'movement':
            label = f'Adam Movement {sparsity}%'
            color = colors['adam_movement']
        elif optimizer == 'adamwprune' and pruning == 'none':
            label = 'AdamWPrune Baseline'
            color = colors['adamwprune_none']
        elif optimizer == 'adamwprune' and pruning == 'state':
            label = f'AdamWPrune State {sparsity}%'
            color = colors['adamwprune_state']
        else:
            continue

        # Plot
        plt.plot(epochs, accuracies, label=label, color=color, linewidth=2, alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('All Pruning Methods Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(50, 95)

    # Save the plot
    output_file = os.path.join(output_dir, 'all_methods_comparison.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_memory_accuracy_comparison(results_dir, output_dir):
    """Create a bar chart comparing memory and accuracy for all methods."""

    # Load summary report
    summary_file = os.path.join(results_dir, "summary_report.txt")
    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found")
        return

    # Parse summary report to get memory data
    test_data = {}
    with open(summary_file, 'r') as f:
        lines = f.readlines()
        in_table = False
        for line in lines:
            if 'Test ID' in line and 'Accuracy' in line and 'GPU Mean' in line:
                in_table = True
                continue
            if in_table:
                if '---' in line or '═' in line:
                    continue
                if 'Best Performers' in line or 'Best by' in line:
                    break
                if line.strip() and 'resnet18' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        test_id = parts[0]
                        try:
                            accuracy = float(parts[1])
                            gpu_mem = parts[3]
                            if gpu_mem != 'N/A':
                                gpu_mem = float(gpu_mem)
                            else:
                                gpu_mem = 1307.5  # Use baseline for Adam
                            test_data[test_id] = {'accuracy': accuracy, 'memory': gpu_mem}
                        except ValueError:
                            continue

    if not test_data:
        print("No test data found")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for plotting
    methods = []
    accuracies = []
    memories = []
    colors_list = []

    # Order: baselines, then pruning methods
    order = [
        'resnet18_adam_none_0',
        'resnet18_adamwprune_none_0',
        'resnet18_adam_magnitude_70',
        'resnet18_adam_movement_70',
        'resnet18_adamwprune_state_70'
    ]

    labels_map = {
        'resnet18_adam_none_0': 'Adam\nBaseline',
        'resnet18_adamwprune_none_0': 'AdamWPrune\nBaseline',
        'resnet18_adam_magnitude_70': 'Adam\nMagnitude 70%',
        'resnet18_adam_movement_70': 'Adam\nMovement 70%',
        'resnet18_adamwprune_state_70': 'AdamWPrune\nState 70%'
    }

    colors_map = {
        'resnet18_adam_none_0': 'steelblue',
        'resnet18_adamwprune_none_0': 'purple',
        'resnet18_adam_magnitude_70': 'orange',
        'resnet18_adam_movement_70': 'green',
        'resnet18_adamwprune_state_70': 'crimson'
    }

    for test_id in order:
        if test_id in test_data:
            methods.append(labels_map[test_id])
            accuracies.append(test_data[test_id]['accuracy'])
            memories.append(test_data[test_id]['memory'])
            colors_list.append(colors_map[test_id])

    x = np.arange(len(methods))
    width = 0.35

    # Create bars
    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color=colors_list, alpha=0.8)
    bars2 = ax2.bar(x + width/2, memories, width, label='GPU Memory (MB)', color=colors_list, alpha=0.5)

    # Add value labels on bars
    for bar, val in zip(bars1, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, memories):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax.set_title('Memory and Accuracy Comparison: All Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(85, 92)
    ax2.set_ylim(1200, 1600)

    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'all_methods_memory_accuracy.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_combined_comparison.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Create output directory
    output_dir = os.path.join(results_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    # Generate graphs
    create_combined_accuracy_evolution(results_dir, output_dir)
    create_memory_accuracy_comparison(results_dir, output_dir)

    print(f"All graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
