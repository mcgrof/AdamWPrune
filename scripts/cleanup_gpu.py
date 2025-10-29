#!/usr/bin/env python3
"""
GPU memory cleanup script for AMD ROCm GPUs.

This script aggressively cleans up GPU memory between test runs to prevent
out-of-memory errors during sequential training runs.
"""

import gc
import sys
import os


def cleanup_gpu():
    """Perform aggressive GPU memory cleanup."""
    try:
        import torch

        # Check if ROCm/CUDA is available
        if not torch.cuda.is_available():
            print("No GPU available, skipping cleanup")
            return True

        # Get current memory stats before cleanup
        device = torch.device("cuda:0")
        mem_allocated_before = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved_before = torch.cuda.memory_reserved(device) / 1024**3

        print(
            f"Before cleanup: {mem_allocated_before:.2f} GiB allocated, {mem_reserved_before:.2f} GiB reserved"
        )

        # Clear PyTorch's GPU cache
        torch.cuda.empty_cache()

        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

        # Clear cache again after gc
        torch.cuda.empty_cache()

        # Get memory stats after cleanup
        mem_allocated_after = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved_after = torch.cuda.memory_reserved(device) / 1024**3

        print(
            f"After cleanup: {mem_allocated_after:.2f} GiB allocated, {mem_reserved_after:.2f} GiB reserved"
        )
        print(
            f"Freed: {mem_allocated_before - mem_allocated_after:.2f} GiB allocated, {mem_reserved_before - mem_reserved_after:.2f} GiB reserved"
        )

        return True

    except ImportError:
        print("PyTorch not available, skipping GPU cleanup")
        return False
    except Exception as e:
        print(f"Error during GPU cleanup: {e}")
        return False


if __name__ == "__main__":
    success = cleanup_gpu()
    sys.exit(0 if success else 1)
