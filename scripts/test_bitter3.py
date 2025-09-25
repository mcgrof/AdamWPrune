#!/usr/bin/env python3
"""Test script for bitter3 variant of AdamWPrune optimizer."""

import sys
import os
import subprocess
from datetime import datetime


def run_gpt2_test():
    """Run GPT-2 test with bitter3 variant."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test configuration
    cmd = [
        "python",
        "train.py",
        "--optimizer",
        "adamwprune",
        "--adamwprune-variant",
        "bitter3",
        "--pruning-method",
        "state",
        "--target-sparsity",
        "0.5",
        "--pruning-warmup",
        "1000",
        "--batch-size",
        "16",
        "--max-iters",
        "1000",  # Short test run
        "--eval-interval",
        "100",
        "--device",
        "cuda",
        "--seed",
        "42",
    ]

    print(f"Testing bitter3 variant at {timestamp}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)

    # Change to gpt2 directory
    original_dir = os.getcwd()
    os.chdir("/data/AdamWPrune/gpt2")

    try:
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Print output
        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        # Check for success
        if result.returncode == 0:
            print("\n✓ Test completed successfully!")

            # Extract perplexity from output
            lines = result.stdout.split("\n")
            for line in lines:
                if "val_perplexity" in line:
                    print(f"Found: {line}")
        else:
            print(f"\n✗ Test failed with return code {result.returncode}")

    except Exception as e:
        print(f"\n✗ Error running test: {e}")

    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    run_gpt2_test()
