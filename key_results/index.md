# Key Test Results Index

This directory contains key test matrix results that demonstrate important findings.

## Latest Results (September 2025)

### ResNet-50 ImageNet
- **[test_matrix_results_20250912_023452](test_matrix_results_20250912_023452/report.md)** - **NEW SOTA**: AdamWPrune with AdamW base achieves 74.54% at 50% sparsity!
  - AdamWPrune dominates with AdamW as base optimizer
  - Consistent 12,602.5 MB memory usage across all sparsity levels
  - Outperforms AdamWSPAM by 1.32% at 50% sparsity

### Historical Results

#### ResNet-18 CIFAR-10
- [test_matrix_results_20250903_180836](test_matrix_results_20250903_180836/report.md) - Baseline results showing AdamWPrune matches movement pruning at 90.69%

#### ResNet-50 CIFAR-100  
- [test_matrix_results_20250908_121537](test_matrix_results_20250908_121537/report.md) - Initial state pruning implementation
- [test_matrix_results_20250908_190856](test_matrix_results_20250908_190856/report.md) - Extended tests across multiple sparsity levels

## Key Findings Summary

1. **AdamWPrune with AdamW base** achieves state-of-the-art 74.54% accuracy at 50% sparsity on ResNet-50 ImageNet
2. **Memory efficiency**: Consistently uses least GPU memory (12,602.5 MB) across all configurations
3. **Base optimizer matters**: AdamW base outperforms other bases at moderate sparsity (50%)
4. **Stability**: AdamWPrune shows excellent stability with only 0.22% std deviation

*Generated: 2025-09-13*
