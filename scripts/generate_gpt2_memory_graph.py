#!/usr/bin/env python3
"""Generate GPU memory consumption graph for GPT-2 experiments."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for beautiful graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_gpu_memory_comparison():
    """Create GPU memory consumption comparison for GPT-2."""
    # Data from GPU stats summaries
    # AMD W7900 has 48GB of memory
    gpu_total_memory = 48 * 1024  # MiB

    tests = {
        'AdamWSPAM\n+ Magnitude': {
            'memory_percent': 56.1,
            'memory_mib': gpu_total_memory * 0.561,
            'duration': 29849,
            'perplexity': 42.82,
            'color': '#2E86AB'
        },
        'AdamWPrune\n+ Bitter0': {
            'memory_percent': 51.5,
            'memory_mib': gpu_total_memory * 0.515,
            'duration': 24750,
            'perplexity': 51.51,
            'color': '#C73E1D'
        },
        'AdamWPrune\n+ Bitter1': {
            'memory_percent': 51.5,
            'memory_mib': gpu_total_memory * 0.515,
            'duration': 24723,
            'perplexity': 49.99,
            'color': '#F18F01'
        },
        'AdamWPrune\n+ Bitter2': {
            'memory_percent': 51.5,
            'memory_mib': gpu_total_memory * 0.515,
            'duration': 29959,
            'perplexity': 46.07,
            'color': '#A23B72'
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Memory consumption bar chart
    names = list(tests.keys())
    memory_mib = [tests[name]['memory_mib'] for name in names]
    memory_percent = [tests[name]['memory_percent'] for name in names]
    colors = [tests[name]['color'] for name in names]
    perplexities = [tests[name]['perplexity'] for name in names]

    bars1 = ax1.bar(range(len(names)), memory_mib, color=colors, edgecolor='black', linewidth=1.5)

    # Add memory values on bars
    for i, (bar, mem_mib, mem_pct) in enumerate(zip(bars1, memory_mib, memory_percent)):
        # Memory in GiB
        mem_gib = mem_mib / 1024
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{mem_gib:.1f} GiB\n({mem_pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight the memory savings
    ax1.axhline(y=memory_mib[0], color='red', linestyle='--', alpha=0.5, label='AdamWSPAM baseline')
    savings_pct = (1 - memory_mib[1]/memory_mib[0]) * 100
    # Place text inside Bitter2 bar (index 3) which has best perplexity among AdamWPrune
    ax1.text(3, memory_mib[3]/2, f'{savings_pct:.1f}%\nmemory\nreduction',
            fontsize=11, color='white', fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', alpha=0.8))

    ax1.set_ylabel('GPU Memory Usage (MiB)', fontsize=12, fontweight='bold')
    ax1.set_title('GPU Memory Consumption', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=0, ha='center')
    ax1.set_ylim(0, max(memory_mib) * 1.15)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right')

    # Training time vs perplexity scatter
    durations_hours = [tests[name]['duration'] / 3600 for name in names]

    scatter = ax2.scatter(perplexities, durations_hours,
                         s=[200 if 'AdamWSPAM' in name else 150 for name in names],
                         c=colors, alpha=0.7, edgecolors='black', linewidth=2)

    # Add labels
    for i, name in enumerate(names):
        label = name.replace('\n', ' ')
        offset_y = 0.3 if i % 2 == 0 else -0.3
        ax2.annotate(label,
                    (perplexities[i], durations_hours[i] + offset_y),
                    ha='center', fontsize=9, fontweight='bold')

    # Add quadrants
    ax2.axhline(y=np.mean(durations_hours), color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=np.mean(perplexities), color='gray', linestyle='--', alpha=0.3)

    # Label best and worst quadrants
    ax2.text(43, 8.8, 'Best\n(Fast & Accurate)', fontsize=9, color='green',
            ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax2.text(51, 8.8, 'Worst\n(Slow & Inaccurate)', fontsize=9, color='red',
            ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    ax2.set_xlabel('Perplexity (lower is better →)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    plt.suptitle('GPT-2 GPU Memory and Training Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    output_dir = Path('images/gpt2')
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / 'gpt2_gpu_memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("GPU memory analysis graph saved to images/gpt2/gpt2_gpu_memory_analysis.png")

if __name__ == "__main__":
    create_gpu_memory_comparison()
