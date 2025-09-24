#!/usr/bin/env python3
"""Generate perplexity projection visualization for GPT-2 experiments."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for beautiful graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_perplexity_projection():
    """Create visualization of projected iterations needed to match baseline perplexity."""

    # Data from analysis
    baseline_ppl = 42.82
    improvement_rate = 0.75  # perplexity per 1000 iterations

    variants = {
        'AdamWSPAM\n(Baseline)': {
            'current_ppl': 42.82,
            'current_iters': 10000,
            'projected_iters': 10000,
            'color': '#2E86AB'
        },
        'Bitter2\n(Scale-aware)': {
            'current_ppl': 46.07,
            'current_iters': 12100,
            'projected_iters': 16433,
            'color': '#A23B72'
        },
        'Bitter1\n(Pure Magnitude)': {
            'current_ppl': 49.99,
            'current_iters': 10000,
            'projected_iters': 19560,
            'color': '#F18F01'
        },
        'Bitter0\n(Hybrid)': {
            'current_ppl': 51.51,
            'current_iters': 10000,
            'projected_iters': 21587,
            'color': '#C73E1D'
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Current vs Projected Iterations
    names = list(variants.keys())
    current_iters = [variants[n]['current_iters'] for n in names]
    projected_iters = [variants[n]['projected_iters'] for n in names]
    colors = [variants[n]['color'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, current_iters, width, label='Actual iterations',
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, projected_iters, width, label='Projected to match baseline',
                    color=colors, alpha=0.4, edgecolor='black', linewidth=1.5, hatch='//')

    # Add value labels
    for bar1, bar2, curr, proj in zip(bars1, bars2, current_iters, projected_iters):
        # Current iterations
        ax1.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 300,
                f'{curr:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Projected iterations
        if proj > curr:
            ax1.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 300,
                    f'{proj:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Show percentage increase
            pct_increase = (proj / 10000 - 1) * 100
            ax1.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height()/2,
                    f'+{pct_increase:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    ax1.set_ylabel('Training Iterations', fontsize=12, fontweight='bold')
    ax1.set_title('Iterations Needed to Match Baseline Perplexity\n(Projected at 0.75 ppl/1000 iters improvement)',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=0, ha='center')
    ax1.legend(loc='upper left')
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Add baseline reference line
    ax1.axhline(y=10000, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(3.5, 10500, 'Baseline', ha='right', fontsize=9, color='red', style='italic')

    # Right plot: Time-Cost Analysis
    # Assuming ~0.83 hours per 1000 iterations (from actual data)
    hours_per_1k = 0.83

    time_data = []
    cost_data = []
    ppl_data = []

    for name in names:
        actual_time = variants[name]['current_iters'] * hours_per_1k / 1000
        projected_time = variants[name]['projected_iters'] * hours_per_1k / 1000
        current_ppl = variants[name]['current_ppl']

        time_data.append(actual_time)
        cost_data.append(projected_time / actual_time if projected_time > actual_time else 1.0)
        ppl_data.append(current_ppl)

    # Create scatter plot
    scatter = ax2.scatter(time_data, ppl_data, s=200, c=colors,
                         alpha=0.7, edgecolors='black', linewidth=2)

    # Add projected lines
    for i, name in enumerate(names):
        if variants[name]['projected_iters'] > variants[name]['current_iters']:
            projected_time = variants[name]['projected_iters'] * hours_per_1k / 1000
            # Draw arrow from current to projected
            ax2.annotate('', xy=(projected_time, baseline_ppl),
                        xytext=(time_data[i], ppl_data[i]),
                        arrowprops=dict(arrowstyle='->', color=colors[i],
                                      lw=2, alpha=0.5, linestyle='--'))
            # Add time label
            ax2.text(projected_time, baseline_ppl + 0.5,
                    f'{projected_time:.1f}h', ha='center', va='bottom',
                    fontsize=8, color=colors[i], fontweight='bold')

    # Add labels for current positions
    for i, name in enumerate(names):
        clean_name = name.replace('\n', ' ')
        ax2.text(time_data[i], ppl_data[i] + 1,
                clean_name, ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Add target line
    ax2.axhline(y=baseline_ppl, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(17, baseline_ppl + 0.3, 'Target perplexity',
            ha='right', fontsize=10, color='green', fontweight='bold')

    ax2.set_xlabel('Training Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Time Investment to Reach Baseline Quality',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_xlim(5, 20)
    ax2.set_ylim(41, 54)

    # Add text box with key insight
    textstr = 'Key Insight:\nEven with 2x training time,\nAdamWPrune would still use\n40% less memory'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, fontweight='bold')

    plt.suptitle('GPT-2 Perplexity Matching Projections', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    output_dir = Path('images/gpt2')
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / 'gpt2_perplexity_projections.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Perplexity projection graph saved to images/gpt2/gpt2_perplexity_projections.png")

    # Print analysis summary
    print("\n" + "="*60)
    print("PROJECTION SUMMARY")
    print("="*60)
    print("\nTo match baseline perplexity (42.82):")
    for name, data in variants.items():
        if data['current_iters'] < data['projected_iters']:
            clean_name = name.replace('\n', ' ')
            additional = data['projected_iters'] - data['current_iters']
            pct_more = (data['projected_iters'] / 10000 - 1) * 100
            print(f"\n{clean_name:25}: +{additional:,} iterations ({pct_more:+.0f}% vs baseline)")
            print(f"  Current: {data['current_ppl']:.2f} ppl at {data['current_iters']:,} iters")
            print(f"  Projected: {baseline_ppl:.2f} ppl at {data['projected_iters']:,} iters")

if __name__ == "__main__":
    create_perplexity_projection()