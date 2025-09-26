#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Monitor hyperparameter sweep progress without interfering with running tests.

Usage:
    python scripts/monitor_sweep.py [sweep_directory]

    If no directory specified, finds the latest sweep_* directory.
"""

import sys
import os
import json
import glob
from pathlib import Path
from datetime import datetime
import time


def find_latest_sweep_dir():
    """Find the most recent sweep directory."""
    sweep_dirs = sorted(glob.glob("test_matrix_results/sweep_*"))
    if not sweep_dirs:
        return None
    return sweep_dirs[-1]


def parse_training_metrics(metrics_file):
    """Extract key metrics from training_metrics.json."""
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Handle both formats (test_acc and test_accuracy)
        test_acc_key = 'test_accuracy' if 'test_accuracy' in data else 'test_acc'
        test_loss_key = 'test_loss'
        sparsity_key = 'sparsity'

        # Try to get from arrays or use final values
        if data.get('epochs'):
            final_idx = -1
            return {
                'final_epoch': data['epochs'][final_idx] if data.get('epochs') else 0,
                'test_acc': data[test_acc_key][final_idx] if data.get(test_acc_key) else data.get('final_accuracy', 0),
                'test_loss': data[test_loss_key][final_idx] if data.get(test_loss_key) else 0,
                'sparsity': data[sparsity_key][final_idx] if data.get(sparsity_key) else data.get('final_sparsity', 0),
                'total_epochs': len(data.get('epochs', [])),
            }
        else:
            # Use final values directly
            return {
                'final_epoch': 100,
                'test_acc': data.get('final_accuracy', 0),
                'test_loss': 0,
                'sparsity': data.get('final_sparsity', 0),
                'total_epochs': 100,
            }
    except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError) as e:
        return None


def get_current_epoch_from_log(log_file):
    """Get current epoch from output.log by reading last epoch line."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Search from end for epoch progress
        for line in reversed(lines):
            if 'Epoch [' in line and '/' in line:
                # Parse: "Epoch [78/100] Train Loss: ..."
                epoch_part = line.split('[')[1].split('/')[0]
                return int(epoch_part)
    except:
        pass
    return 0


def parse_config_params(config_file):
    """Extract key hyperparameters from config file."""
    params = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')

                    # Extract key parameters
                    if 'ADAMWPRUNE_WEIGHT_DECAY' in key:
                        params['wd'] = value
                    elif 'ADAMWPRUNE_BETA1' in key:
                        params['b1'] = value
                    elif 'PRUNING_WARMUP' in key:
                        # Convert to epoch
                        try:
                            warmup = int(value)
                            params['warmup'] = f"e{warmup//391}"  # 391 steps per epoch
                        except:
                            params['warmup'] = value
                    elif 'PRUNING_RAMP_END_EPOCH' in key:
                        params['ramp'] = f"e{value}"
                    elif 'TARGET_SPARSITY' in key:
                        params['sparsity'] = f"{float(value)*100:.0f}%"
    except:
        pass
    return params


def format_time_elapsed(start_time):
    """Format elapsed time as HH:MM:SS."""
    if not start_time:
        return "??:??:??"

    try:
        # Parse timestamp from directory name or file
        if isinstance(start_time, str):
            # Try to parse from timestamp
            dt = datetime.strptime(start_time, "%Y%m%d_%H%M%S")
            elapsed = datetime.now() - dt
        else:
            elapsed = datetime.now() - start_time

        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except:
        return "??:??:??"


def monitor_sweep(sweep_dir):
    """Monitor sweep progress and display leaderboard."""
    sweep_path = Path(sweep_dir)

    if not sweep_path.exists():
        print(f"Error: Directory '{sweep_dir}' not found")
        return

    # Get all test directories
    test_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir() and d.name.startswith('config_')])

    if not test_dirs:
        print("No test directories found")
        return

    # Collect results
    results = []
    running = []
    pending = []
    failed = []

    for test_dir in test_dirs:
        # Extract config number (e.g., config_001_resnet18_adamwprune_state_70 -> 001)
        dir_parts = test_dir.name.split('_')
        config_name = dir_parts[1] if len(dir_parts) > 1 else test_dir.name
        config_file = test_dir / "config.txt"

        # Find the actual test output directory (it's a subdirectory)
        test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if test_subdirs:
            # Use the first subdirectory (should only be one)
            actual_test_dir = test_subdirs[0]
            metrics_file = actual_test_dir / "training_metrics.json"
            log_file = actual_test_dir / "output.log"
        else:
            # Fallback to direct path if no subdirectory
            metrics_file = test_dir / "training_metrics.json"
            log_file = test_dir / "output.log"

        # Get hyperparameters
        params = parse_config_params(config_file)

        # Check status
        if metrics_file.exists():
            # Completed
            metrics = parse_training_metrics(metrics_file)
            if metrics:
                results.append({
                    'config': f"cfg_{config_name}",
                    'test_acc': metrics['test_acc'],
                    'test_loss': metrics['test_loss'],
                    'sparsity': metrics['sparsity'],
                    'params': params,
                    'epochs': metrics['total_epochs'],
                })
        elif log_file.exists():
            # Check if running or failed
            current_epoch = get_current_epoch_from_log(log_file)
            if current_epoch > 0:
                # Running
                running.append({
                    'config': f"cfg_{config_name}",
                    'current_epoch': current_epoch,
                    'params': params,
                })
            else:
                # Check if failed
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    if 'error' in log_content.lower() or 'failed' in log_content.lower():
                        failed.append({
                            'config': f"cfg_{config_name}",
                            'params': params,
                        })
                    else:
                        # Just started or pending
                        pending.append({
                            'config': f"cfg_{config_name}",
                            'params': params,
                        })
        else:
            # Not started yet
            pending.append({
                'config': f"cfg_{config_name}",
                'params': params,
            })

    # Clear screen for clean display
    os.system('clear' if os.name == 'posix' else 'cls')

    # Display header
    print("=" * 80)
    print(f"HYPERPARAMETER SWEEP MONITOR - {sweep_path.name}")
    print("=" * 80)

    # Summary
    total = len(test_dirs)
    completed = len(results)
    print(f"\nProgress: {completed}/{total} completed, {len(running)} running, {len(pending)} pending, {len(failed)} failed")

    # Progress bar
    progress = completed / total if total > 0 else 0
    bar_length = 50
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"[{bar}] {progress*100:.1f}%")

    # Currently running
    if running:
        print("\n" + "=" * 80)
        print("CURRENTLY RUNNING:")
        print("-" * 80)
        print(f"{'Config':<10} {'Epoch':<12} {'WD':<8} {'Beta1':<8} {'Warmup':<10} {'Ramp':<8}")
        print("-" * 80)
        for r in running[:5]:  # Show max 5 running
            p = r['params']
            print(f"{r['config']:<10} {r['current_epoch']:>3}/100     {p.get('wd', '?'):<8} {p.get('b1', '?'):<8} {p.get('warmup', '?'):<10} {p.get('ramp', '?'):<8}")

    # Leaderboard
    if results:
        print("\n" + "=" * 80)
        print("LEADERBOARD (Top 10):")
        print("-" * 80)

        # Sort by test accuracy
        sorted_results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

        print(f"{'Rank':<6} {'Config':<10} {'Test Acc':<10} {'Sparsity':<10} {'WD':<8} {'Beta1':<8} {'Warmup':<10} {'Ramp':<8}")
        print("-" * 80)

        for i, r in enumerate(sorted_results[:10], 1):
            p = r['params']
            print(f"{i:<6} {r['config']:<10} {r['test_acc']:>6.2f}%    {r['sparsity']:>6.1f}%    {p.get('wd', '?'):<8} {p.get('b1', '?'):<8} {p.get('warmup', '?'):<10} {p.get('ramp', '?'):<8}")

        # Best configuration details
        if sorted_results:
            best = sorted_results[0]
            print("\n" + "=" * 80)
            print("BEST CONFIGURATION SO FAR:")
            print("-" * 80)
            print(f"Config: {best['config']}")
            print(f"Test Accuracy: {best['test_acc']:.2f}%")
            print(f"Final Sparsity: {best['sparsity']:.1f}%")
            print("\nHyperparameters:")
            for key, value in best['params'].items():
                param_name = {
                    'wd': 'Weight Decay',
                    'b1': 'Beta1',
                    'warmup': 'Pruning Warmup',
                    'ramp': 'Ramp End',
                    'sparsity': 'Target Sparsity'
                }.get(key, key)
                print(f"  {param_name:.<20} {value}")

    # Failed tests
    if failed:
        print("\n" + "=" * 80)
        print(f"FAILED TESTS ({len(failed)}):")
        print("-" * 80)
        for f in failed[:5]:  # Show max 5 failed
            print(f"  {f['config']}: wd={f['params'].get('wd', '?')}, b1={f['params'].get('b1', '?')}")

    # Timestamp
    print("\n" + "=" * 80)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if sweep is complete
    if completed == total:
        print("\n🎉 SWEEP COMPLETE! 🎉")
        print(f"Check {sweep_path}/sweep_analysis.txt for full analysis")


def watch_mode(sweep_dir, refresh_interval=30):
    """Continuously monitor sweep progress."""
    print(f"Watching {sweep_dir} (refresh every {refresh_interval}s, Ctrl+C to stop)")

    try:
        while True:
            monitor_sweep(sweep_dir)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")


def main():
    # Check for watch mode first
    watch = '--watch' in sys.argv or '-w' in sys.argv

    # Remove flags from argv to get directory
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

    # Get sweep directory
    if args:
        sweep_dir = args[0]
    else:
        # Find latest sweep directory
        sweep_dir = find_latest_sweep_dir()
        if not sweep_dir:
            print("No sweep directories found in test_matrix_results/")
            print("\nUsage: python scripts/monitor_sweep.py [sweep_directory]")
            sys.exit(1)
        print(f"Using latest sweep: {sweep_dir}")

    if watch:
        # Extract refresh interval if provided
        interval = 30
        for arg in sys.argv:
            if arg.startswith('--interval='):
                try:
                    interval = int(arg.split('=')[1])
                except:
                    pass
        watch_mode(sweep_dir, interval)
    else:
        # Single run
        monitor_sweep(sweep_dir)
        print("\n(Use --watch or -w to continuously monitor)")


if __name__ == "__main__":
    main()
