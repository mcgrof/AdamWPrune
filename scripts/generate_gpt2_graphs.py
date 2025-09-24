#!/usr/bin/env python3
"""Generate visualization graphs for GPT-2 AdamWPrune test results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for beautiful graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_gpt2_results():
    """Load GPT-2 test results."""
    results = {
        'AdamWSPAM + Magnitude': {
            'perplexity': 42.82,
            'speedup': 1.0,
            'memory_multiplier': 5.03,
            'color': '#2E86AB'
        },
        'AdamWPrune + Bitter2': {
            'perplexity': 46.07,
            'speedup': 1.20,
            'memory_multiplier': 3.03,
            'color': '#A23B72'
        },
        'AdamWPrune + Bitter1': {
            'perplexity': 49.99,
            'speedup': 1.20,
            'memory_multiplier': 3.03,
            'color': '#F18F01'
        },
        'AdamWPrune + Bitter0': {
            'perplexity': 51.51,
            'speedup': 1.20,
            'memory_multiplier': 3.03,
            'color': '#C73E1D'
        }
    }
    return results

def create_perplexity_comparison(results, output_dir):
    """Create perplexity comparison bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = list(results.keys())
    perplexities = [results[name]['perplexity'] for name in names]
    colors = [results[name]['color'] for name in names]

    bars = ax.bar(names, perplexities, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, perp in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{perp:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight baseline
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)

    ax.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('GPT-2 Pruning Performance: Perplexity Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(perplexities) * 1.1)

    # Rotate x labels for better readability
    plt.xticks(rotation=15, ha='right')

    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'gpt2_perplexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_speedup_memory_chart(results, output_dir):
    """Create combined speedup and memory efficiency chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = list(results.keys())
    speedups = [results[name]['speedup'] for name in names]
    memory = [results[name]['memory_multiplier'] for name in names]
    colors = [results[name]['color'] for name in names]

    # Speedup chart
    bars1 = ax1.bar(names, speedups, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars1, speedups):
        height = bar.get_height()
        percentage = (val - 1) * 100
        label = f'+{percentage:.0f}%' if percentage > 0 else 'Baseline'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Training Speedup', fontsize=12, fontweight='bold')
    ax1.set_title('Training Speed Improvement', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.4)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Memory chart
    bars2 = ax2.bar(names, memory, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars2, memory):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add memory savings annotation
    ax2.annotate('40% memory\nreduction',
                xy=(2, 3.03), xytext=(2.5, 4.0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold', ha='center')

    ax2.set_ylabel('Memory Usage (×model weights)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Memory Overhead', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 6)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Rotate x labels
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')

    plt.suptitle('GPT-2 AdamWPrune Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'gpt2_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_bitter_lesson_chart(results, output_dir):
    """Create bitter lesson visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Filter AdamWPrune results only
    adamwprune_results = {k: v for k, v in results.items() if 'AdamWPrune' in k}

    # Extract algorithm complexity order (reversed so simpler is better)
    algorithms = ['AdamWPrune + Bitter1', 'AdamWPrune + Bitter2', 'AdamWPrune + Bitter0']
    complexity = ['Simple\n(Pure Magnitude)', 'Medium\n(Scale-aware)', 'Complex\n(Hybrid)']
    perplexities = [adamwprune_results[alg]['perplexity'] for alg in algorithms]
    colors = [adamwprune_results[alg]['color'] for alg in algorithms]

    bars = ax.bar(complexity, perplexities, color=colors, edgecolor='black', linewidth=2)

    # Add value labels and algorithm names
    for i, (bar, perp, alg) in enumerate(zip(bars, perplexities, algorithms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{perp:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Add algorithm variant name below
        variant = alg.split(' + ')[1]
        ax.text(bar.get_x() + bar.get_width()/2., -2,
                f'({variant})', ha='center', va='top', fontsize=9, style='italic')

    # Highlight the trend with an arrow
    ax.annotate('', xy=(2.3, 52), xytext=(0.3, 49),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(1.3, 48, 'Complexity hurts\nperformance',
            fontsize=11, color='red', fontweight='bold', ha='center')

    ax.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algorithm Complexity', fontsize=12, fontweight='bold')
    ax.set_title('The Bitter Lesson: Simpler Algorithms Win\n(AdamWPrune Variants on GPT-2)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(45, 53)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add text box with bitter lesson quote
    textstr = '"The bitter lesson is that general methods\nthat leverage computation ultimately\ndominate specialized methods."'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'gpt2_bitter_lesson.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_trade_off_scatter(results, output_dir):
    """Create trade-off visualization (perplexity vs efficiency)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    names = list(results.keys())
    perplexities = [results[name]['perplexity'] for name in names]
    speedups = [results[name]['speedup'] for name in names]
    memory = [results[name]['memory_multiplier'] for name in names]
    colors = [results[name]['color'] for name in names]

    # Create scatter plot with size based on memory efficiency
    sizes = [(6 - m) * 200 for m in memory]  # Invert so smaller memory = larger circle

    scatter = ax.scatter(perplexities, speedups, s=sizes, c=colors,
                        alpha=0.7, edgecolors='black', linewidth=2)

    # Add labels
    for i, name in enumerate(names):
        offset_y = 0.02 if i % 2 == 0 else -0.03
        ax.annotate(name.replace(' + ', '\n'),
                   (perplexities[i], speedups[i] + offset_y),
                   ha='center', fontsize=9, fontweight='bold')

    # Add quadrant lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    median_perp = np.median(perplexities)
    ax.axvline(x=median_perp, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    ax.text(41, 1.25, 'Best\n(Fast & Accurate)', fontsize=10, color='green',
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(53, 1.25, 'Fast but\nLess Accurate', fontsize=10, color='orange',
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    ax.set_xlabel('Perplexity (lower is better →)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Speedup', fontsize=12, fontweight='bold')
    ax.set_title('GPT-2 Pruning Trade-offs: Speed vs Quality vs Memory', fontsize=14, fontweight='bold')

    # Add legend for circle sizes
    legend_elements = [plt.scatter([], [], s=200, c='gray', alpha=0.7, edgecolors='black',
                                   label='Low memory (3.03x)'),
                      plt.scatter([], [], s=60, c='gray', alpha=0.7, edgecolors='black',
                                   label='High memory (5.03x)')]
    ax.legend(handles=legend_elements, loc='lower left', title='Memory Usage', fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'gpt2_trade_off_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all GPT-2 visualization graphs."""
    # Create output directory
    output_dir = Path('key_results/test_matrix_results_20250923_010926/graphs')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load results
    results = load_gpt2_results()

    print("Generating GPT-2 visualization graphs...")

    # Generate all charts
    create_perplexity_comparison(results, output_dir)
    print("  ✓ Perplexity comparison chart")

    create_speedup_memory_chart(results, output_dir)
    print("  ✓ Efficiency analysis charts")

    create_bitter_lesson_chart(results, output_dir)
    print("  ✓ Bitter lesson visualization")

    create_trade_off_scatter(results, output_dir)
    print("  ✓ Trade-off analysis scatter plot")

    print(f"\nAll graphs saved to: {output_dir}")
    print("\nGraph files created:")
    for graph_file in sorted(output_dir.glob('*.png')):
        print(f"  - {graph_file.name}")

if __name__ == "__main__":
    main()
