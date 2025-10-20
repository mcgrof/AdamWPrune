#!/usr/bin/env python3
"""
Test script for flash attention support in PyTorch.

This script tests whether PyTorch was built with memory efficient attention
(flash attention) support. Once PyTorch is rebuilt with flash attention enabled,
this test should pass without warnings.

Usage:
    python scripts/test_flash_attention.py

Exit codes:
    0: Flash attention is available and working
    1: Flash attention test failed or warnings detected
"""

import os
import sys
import warnings

# CRITICAL: Set environment variables before importing torch
# Read from config.py if available, otherwise use defaults
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from config import Config

    config = Config()
    if (
        hasattr(config, "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
        and config.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
    ):
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
except (ImportError, AttributeError):
    # Fallback to safe default if config.py doesn't exist or doesn't have the setting
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def test_flash_attention():
    """Test if flash attention is available and working."""
    print("Testing flash attention support...")
    print("-" * 60)

    # Show environment variables
    aotriton_enabled = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")
    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: {aotriton_enabled}")
    print()

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("\n❌ CUDA not available - cannot test flash attention")
            return False

        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print()

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Test flash attention as provided by user
            print("Running scaled_dot_product_attention test...")
            x = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
            out = torch.nn.functional.scaled_dot_product_attention(x, x, x)

            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            print()

            # Check for warnings
            if len(w) > 0:
                print(f"⚠️  {len(w)} warning(s) detected:")
                for warning in w:
                    print(f"  - {warning.category.__name__}: {warning.message}")
                print()
                print("❌ Flash attention is NOT available (warnings detected)")
                print("   PyTorch needs to be rebuilt with flash attention support")
                return False
            else:
                print("✅ No warnings detected")
                print("✅ Flash attention is available and working!")
                return True

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    success = test_flash_attention()

    print("-" * 60)
    if success:
        print("Result: PASS")
        sys.exit(0)
    else:
        print("Result: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
