#!/usr/bin/env python3
"""
Compare A/B test results for AdamWSpam vs AdamWPrune.
Usage: python3 gpt2/compare_ab_results.py test_matrix_results_*/
"""

import json
import sys
from pathlib import Path
import numpy as np

def load_metrics(results_dir):
    """Load metrics from test matrix results."""
    results = {}
    results_path = Path(results_dir)

    # Look for GPT2 results
    for test_dir in results_path.glob("gpt2_*"):
        optimizer = test_dir.name.split("_")[1]
        pruning = "_".join(test_dir.name.split("_")[2:]) if len(test_dir.name.split("_")) > 2 else "none"

        metrics_file = test_dir / "training_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                key = f"{optimizer}_{pruning}"
                results[key] = metrics

    return results

def compare_results(results):
    """Compare AdamWSpam vs AdamWPrune results."""
    print("\n" + "="*60)
    print("A/B Test Results: AdamWSpam vs AdamWPrune (50% pruning)")
    print("="*60)

    # Expected keys
    spam_key = "adamwspam_none"
    prune_key = "adamwprune_state_50"

    if spam_key in results and prune_key in results:
        spam = results[spam_key]
        prune = results[prune_key]

        # Final metrics comparison
        print("\nFinal Metrics:")
        print("-" * 40)
        print(f"{'Optimizer':<20} {'Loss':<12} {'Perplexity':<12}")
        print("-" * 40)

        spam_loss = spam.get('final_loss', spam.get('train_losses', [0])[-1] if 'train_losses' in spam else 0)
        prune_loss = prune.get('final_loss', prune.get('train_losses', [0])[-1] if 'train_losses' in prune else 0)

        spam_ppl = np.exp(spam_loss) if spam_loss > 0 else 0
        prune_ppl = np.exp(prune_loss) if prune_loss > 0 else 0

        print(f"{'AdamWSpam (baseline)':<20} {spam_loss:<12.4f} {spam_ppl:<12.2f}")
        print(f"{'AdamWPrune (50%)':<20} {prune_loss:<12.4f} {prune_ppl:<12.2f}")

        # Performance difference
        print("\nPerformance Comparison:")
        print("-" * 40)
        if spam_loss > 0:
            loss_diff = ((prune_loss - spam_loss) / spam_loss) * 100
            print(f"Loss difference: {loss_diff:+.2f}%")

            if loss_diff < 5:
                print("✓ AdamWPrune achieves similar performance with 50% pruning!")
            elif loss_diff < 10:
                print("✓ AdamWPrune shows acceptable degradation with 50% pruning")
            else:
                print("⚠ AdamWPrune shows significant degradation")

        # Training time comparison
        if 'training_time' in spam and 'training_time' in prune:
            time_diff = ((prune['training_time'] - spam['training_time']) / spam['training_time']) * 100
            print(f"Training time difference: {time_diff:+.2f}%")

        # Memory comparison if available
        if 'peak_memory_mb' in spam and 'peak_memory_mb' in prune:
            mem_diff = ((prune['peak_memory_mb'] - spam['peak_memory_mb']) / spam['peak_memory_mb']) * 100
            print(f"Peak memory difference: {mem_diff:+.2f}%")
            if mem_diff < 0:
                print(f"✓ AdamWPrune saves {abs(mem_diff):.1f}% memory!")

        # Sparsity achieved
        if 'final_sparsity' in prune:
            print(f"\nFinal sparsity achieved: {prune['final_sparsity']*100:.1f}%")
    else:
        print("\nMissing results:")
        if spam_key not in results:
            print(f"  - {spam_key} (AdamWSpam baseline)")
        if prune_key not in results:
            print(f"  - {prune_key} (AdamWPrune with 50% pruning)")
        print("\nAvailable results:", list(results.keys()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gpt2/compare_ab_results.py test_matrix_results_*/")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_metrics(results_dir)
    compare_results(results)

    print("\n" + "="*60)
    print("For detailed graphs, check:", Path(results_dir) / "graphs/")
