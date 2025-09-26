#!/usr/bin/env python3
"""
Generate markdown reports for key test results in data/key_results/
Each report provides detailed analysis of test matrix results.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime


def load_json_file(filepath):
    """Load and return JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def format_accuracy(acc):
    """Format accuracy value."""
    if acc is None or acc == "N/A":
        return "N/A"
    # Handle case where acc is a list (epoch accuracies)
    if isinstance(acc, list):
        if len(acc) > 0:
            acc = acc[-1]  # Use last epoch accuracy
        else:
            return "N/A"
    return f"{acc:.2f}%"


def format_memory(mem):
    """Format memory value."""
    if mem is None or mem == "N/A":
        return "N/A"
    return f"{mem:.1f} MB"


def parse_test_id(test_id):
    """Parse test ID into components."""
    parts = test_id.split('_')
    model = parts[0] if len(parts) > 0 else "unknown"
    optimizer = parts[1] if len(parts) > 1 else "unknown"
    pruning = parts[2] if len(parts) > 2 else "none"
    sparsity = parts[3] if len(parts) > 3 else "0"
    return model, optimizer, pruning, sparsity


def generate_report(results_dir):
    """Generate markdown report for a test matrix results directory."""
    results_path = Path(results_dir)
    report_name = results_path.name

    # Load summary report if exists
    summary_file = results_path / "summary_report.txt"

    # Load test matrix config (try all_results.json if test_matrix_config.json doesn't exist)
    config_file = results_path / "test_matrix_config.json"
    if not config_file.exists():
        config_file = results_path / "all_results.json"
    config_data = load_json_file(config_file)

    # Find all test result directories
    test_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("resnet18_")])

    # Start building markdown
    md_lines = []
    md_lines.append(f"# Test Matrix Results: {report_name}")
    md_lines.append("")

    # Add generation timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines.append(f"*Generated: {timestamp}*")
    md_lines.append("")

    # Add overview section
    md_lines.append("## Overview")
    md_lines.append("")

    if config_data:
        md_lines.append(f"- **Total Tests**: {len(test_dirs)}")
        if 'model' in config_data:
            md_lines.append(f"- **Model**: {config_data['model']}")
        if 'dataset' in config_data:
            md_lines.append(f"- **Dataset**: {config_data['dataset']}")
        md_lines.append("")

    # Add graphs section
    graphs_dir = results_path / "graphs"
    if graphs_dir.exists():
        md_lines.append("## Visualization Graphs")
        md_lines.append("")

        graph_files = sorted(graphs_dir.glob("*.png"))
        for graph_file in graph_files:
            graph_name = graph_file.stem.replace('_', ' ').title()
            rel_path = f"graphs/{graph_file.name}"
            md_lines.append(f"### {graph_name}")
            md_lines.append(f"![{graph_name}]({rel_path})")
            md_lines.append("")

    # Add detailed results table
    md_lines.append("## Detailed Test Results")
    md_lines.append("")
    md_lines.append("| Test Configuration | Accuracy | Sparsity | GPU Mean | GPU Max | Training Time |")
    md_lines.append("|-------------------|----------|----------|----------|---------|---------------|")

    for test_dir in test_dirs:
        test_id = test_dir.name
        model, optimizer, pruning, sparsity = parse_test_id(test_id)

        # Load result JSON (try training_metrics.json if result.json doesn't exist)
        result_file = test_dir / "result.json"
        if not result_file.exists():
            result_file = test_dir / "training_metrics.json"
        result_data = load_json_file(result_file)

        if result_data:
            accuracy = result_data.get('test_accuracy', 'N/A')
            actual_sparsity = result_data.get('sparsity_achieved', sparsity)
            training_time = result_data.get('total_training_time', 'N/A')

            # Look for GPU stats
            gpu_stats_files = list(test_dir.glob("gpu_stats_*.json"))
            gpu_mean = "N/A"
            gpu_max = "N/A"

            if gpu_stats_files:
                gpu_data = load_json_file(gpu_stats_files[0])
                if gpu_data and 'summary' in gpu_data:
                    gpu_mean = format_memory(gpu_data['summary'].get('mean_memory_mb'))
                    gpu_max = format_memory(gpu_data['summary'].get('max_memory_mb'))

            # Format training time
            if isinstance(training_time, (int, float)):
                hours = int(training_time // 3600)
                minutes = int((training_time % 3600) // 60)
                time_str = f"{hours}h {minutes}m"
            else:
                time_str = "N/A"

            # Add row
            config_str = f"{optimizer.upper()}"
            if pruning != "none":
                config_str += f" + {pruning}"

            md_lines.append(f"| {test_id} | {format_accuracy(accuracy)} | {actual_sparsity} | {gpu_mean} | {gpu_max} | {time_str} |")

    md_lines.append("")

    # Add individual test plots
    md_lines.append("## Individual Training Plots")
    md_lines.append("")

    for test_dir in test_dirs:
        test_id = test_dir.name
        plot_files = sorted(test_dir.glob("*_plot.png"))

        if plot_files:
            md_lines.append(f"### {test_id}")
            for plot_file in plot_files:
                rel_path = f"{test_id}/{plot_file.name}"
                plot_name = plot_file.stem.replace('_', ' ').title()
                md_lines.append(f"![{plot_name}]({rel_path})")
            md_lines.append("")

    # Add summary report content if exists
    if summary_file.exists():
        md_lines.append("## Summary Report")
        md_lines.append("")
        md_lines.append("```")
        with open(summary_file, 'r') as f:
            md_lines.append(f.read())
        md_lines.append("```")
        md_lines.append("")

    # Add key findings section
    md_lines.append("## Key Findings")
    md_lines.append("")

    # Analyze and add key findings
    best_accuracy = None
    best_accuracy_test = None
    best_memory = float('inf')
    best_memory_test = None

    for test_dir in test_dirs:
        result_file = test_dir / "result.json"
        if not result_file.exists():
            result_file = test_dir / "training_metrics.json"
        result_data = load_json_file(result_file)

        if result_data:
            accuracy = result_data.get('test_accuracy')
            if accuracy and (best_accuracy is None or accuracy > best_accuracy):
                best_accuracy = accuracy
                best_accuracy_test = test_dir.name

            # Check GPU memory
            gpu_stats_files = list(test_dir.glob("gpu_stats_*.json"))
            if gpu_stats_files:
                gpu_data = load_json_file(gpu_stats_files[0])
                if gpu_data and 'summary' in gpu_data:
                    mean_mem = gpu_data['summary'].get('mean_memory_mb')
                    if mean_mem and mean_mem < best_memory:
                        best_memory = mean_mem
                        best_memory_test = test_dir.name

    if best_accuracy_test:
        md_lines.append(f"- **Best Accuracy**: {format_accuracy(best_accuracy)} ({best_accuracy_test})")
    if best_memory_test and best_memory != float('inf'):
        md_lines.append(f"- **Lowest Memory Usage**: {format_memory(best_memory)} ({best_memory_test})")

    md_lines.append("")

    # Write the markdown file
    output_file = results_path / "report.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))

    return output_file


def main():
    """Generate reports for all test results in key_results/."""
    key_results_dir = Path(__file__).parent.parent / "key_results"

    if not key_results_dir.exists():
        print(f"Key results directory not found: {key_results_dir}")
        return

    # Find all test result directories
    result_dirs = [d for d in key_results_dir.iterdir() if d.is_dir() and d.name.startswith("test_matrix_results_")]

    if not result_dirs:
        print(f"No test result directories found in {key_results_dir}")
        return

    print(f"Generating reports for {len(result_dirs)} test result directories...")

    reports = []
    for result_dir in sorted(result_dirs):
        print(f"Processing {result_dir.name}...")
        report_file = generate_report(result_dir)
        reports.append((result_dir.name, report_file))
        print(f"  Generated: {report_file}")

    # Generate index file
    index_file = key_results_dir / "index.md"
    with open(index_file, 'w') as f:
        f.write("# Key Test Results Index\n\n")
        f.write("This directory contains key test matrix results that demonstrate important findings.\n\n")
        f.write("## Available Results\n\n")

        for dir_name, report_file in reports:
            f.write(f"- [{dir_name}]({dir_name}/report.md)\n")

        f.write("\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\nGenerated index: {index_file}")
    print("Done!")


if __name__ == "__main__":
    main()
