#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Quick leaderboard view for sweep results.

Usage:
    python scripts/sweep_leaderboard.py [sweep_dir] [--top N]
"""

import sys
import json
import glob
from pathlib import Path
import tabulate


def find_latest_sweep():
    """Find most recent sweep directory."""
    sweeps = sorted(glob.glob("test_matrix_results/sweep_*"))
    return sweeps[-1] if sweeps else None


def load_sweep_results(sweep_dir):
    """Load results from sweep_summary.json or reconstruct from test dirs."""
    sweep_path = Path(sweep_dir)
    summary_file = sweep_path / "sweep_summary.json"
    
    if summary_file.exists():
        # Use existing summary
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    # Reconstruct from individual tests
    results = []
    for test_dir in sorted(sweep_path.iterdir()):
        if not test_dir.is_dir() or not test_dir.name.startswith('config_'):
            continue
        
        config_file = test_dir / "config.txt"
        
        # Find the actual test output directory (it's a subdirectory)
        test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if test_subdirs:
            # Use the first subdirectory (should only be one)
            actual_test_dir = test_subdirs[0]
            metrics_file = actual_test_dir / "training_metrics.json"
        else:
            # Fallback to direct path if no subdirectory
            metrics_file = test_dir / "training_metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Extract config parameters
                params = {}
                with open(config_file, 'r') as f:
                    for line in f:
                        if 'ADAMWPRUNE_WEIGHT_DECAY=' in line:
                            params['wd'] = line.split('=')[1].strip().strip('"')
                        elif 'ADAMWPRUNE_BETA1=' in line:
                            params['b1'] = line.split('=')[1].strip().strip('"')
                        elif 'PRUNING_WARMUP=' in line:
                            warmup = int(line.split('=')[1].strip())
                            params['warmup'] = warmup // 391  # Convert to epochs
                        elif 'PRUNING_RAMP_END_EPOCH=' in line:
                            params['ramp'] = int(line.split('=')[1].strip())
                
                # Get final metrics (handle both test_acc and test_accuracy)
                test_acc_key = 'test_accuracy' if 'test_accuracy' in data else 'test_acc'
                
                if data.get(test_acc_key) or data.get('final_accuracy'):
                    if data.get(test_acc_key):
                        test_acc = data[test_acc_key][-1]
                        best_acc = max(data[test_acc_key])
                        best_epoch = data[test_acc_key].index(best_acc) + 1
                    else:
                        test_acc = data.get('final_accuracy', 0)
                        best_acc = data.get('best_accuracy', test_acc)
                        best_epoch = 0
                    
                    results.append({
                        'config': test_dir.name.split('_')[1],
                        'test_acc': test_acc,
                        'test_loss': data.get('test_loss', [0])[-1] if data.get('test_loss') else 0,
                        'final_sparsity': data.get('final_sparsity', data.get('sparsity', [0])[-1] if data.get('sparsity') else 0),
                        'params': params,
                        'best_epoch': best_epoch,
                        'best_acc': best_acc,
                    })
            except:
                continue
    
    return results


def display_leaderboard(results, top_n=10):
    """Display formatted leaderboard."""
    if not results:
        print("No completed tests found")
        return
    
    # Sort by test accuracy
    sorted_results = sorted(results, key=lambda x: x.get('test_acc', 0), reverse=True)
    
    # Prepare table data
    table_data = []
    for i, r in enumerate(sorted_results[:top_n], 1):
        p = r.get('params', {})
        
        # Format row
        row = [
            i,  # Rank
            f"cfg_{r.get('config', '?')}",
            f"{r.get('test_acc', 0):.2f}%",
            f"{r.get('final_sparsity', 0)*100:.0f}%" if r.get('final_sparsity', 0) > 0 else "0%",
            p.get('wd', '?'),
            p.get('b1', '?'),
            f"e{p.get('warmup', '?')}" if p.get('warmup') else '?',
            f"e{p.get('ramp', '?')}" if p.get('ramp') else '?',
            r.get('best_epoch', '?'),
            f"{r.get('best_acc', 0):.2f}%" if r.get('best_acc') else f"{r.get('test_acc', 0):.2f}%",
        ]
        table_data.append(row)
    
    # Display table
    headers = ['Rank', 'Config', 'Final Acc', 'Sparsity', 'WD', 'Beta1', 'Warmup', 'Ramp', 'Best@', 'Best Acc']
    print(tabulate.tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Summary statistics
    print(f"\nTotal completed: {len(results)}")
    if results:
        avg_acc = sum(r.get('test_acc', 0) for r in results) / len(results)
        print(f"Average accuracy: {avg_acc:.2f}%")
        
        # Best configuration details
        best = sorted_results[0]
        print(f"\n{'='*60}")
        print("BEST CONFIGURATION:")
        print(f"{'='*60}")
        print(f"Config: cfg_{best.get('config', '?')}")
        print(f"Final Test Accuracy: {best.get('test_acc', 0):.2f}%")
        print(f"Best Test Accuracy: {best.get('best_acc', best.get('test_acc', 0)):.2f}% @ epoch {best.get('best_epoch', '?')}")
        print(f"Final Sparsity: {best.get('final_sparsity', 0)*100:.0f}%")
        print("\nHyperparameters:")
        p = best.get('params', {})
        print(f"  Weight Decay: {p.get('wd', '?')}")
        print(f"  Beta1: {p.get('b1', '?')}")
        print(f"  Warmup: epoch {p.get('warmup', '?')}")
        print(f"  Ramp End: epoch {p.get('ramp', '?')}")


def main():
    # Parse arguments
    sweep_dir = None
    top_n = 10
    
    for arg in sys.argv[1:]:
        if arg.startswith('--top='):
            try:
                top_n = int(arg.split('=')[1])
            except:
                pass
        elif not arg.startswith('-'):
            sweep_dir = arg
    
    # Find sweep directory if not specified
    if not sweep_dir:
        sweep_dir = find_latest_sweep()
        if not sweep_dir:
            print("No sweep directories found")
            print("\nUsage: python scripts/sweep_leaderboard.py [sweep_dir] [--top=N]")
            sys.exit(1)
        print(f"Using latest sweep: {sweep_dir}\n")
    
    # Load and display results
    print(f"{'='*60}")
    print(f"SWEEP LEADERBOARD - {Path(sweep_dir).name}")
    print(f"{'='*60}\n")
    
    results = load_sweep_results(sweep_dir)
    display_leaderboard(results, top_n)


if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate for better table formatting...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        import tabulate
    
    main()