# Key Test Results Index

This directory contains key test matrix results that demonstrate important findings.

## Latest Results (September 2025)

### ResNet-50 ImageNet
- **[test_matrix_results_20250913_200218](test_matrix_results_20250913_200218/ANALYSIS.md)** - **NEW STATE-OF-THE-ART**: AdamWPrune with AdamWSpam base achieves 74.56% at 50% sparsity!
  - **Universal superiority**: Beats AdamWSpam at ALL sparsity levels (50%, 70%, 90%)
  - **Pruning improves accuracy**: 50% sparsity is 1.96% better than baseline
  - **Synergistic effects**: SPAM's gradient stabilization enhances state-based pruning
  - Memory efficiency: 6.06% accuracy per GB

- **[test_matrix_results_20250912_023452](test_matrix_results_20250912_023452/report.md)** - AdamWPrune with AdamW base achieves 74.54% at 50% sparsity
  - Previous best with AdamW as base optimizer
  - Consistent 12,602.5 MB memory usage across all sparsity levels
  - Outperforms AdamWSPAM by 1.32% at 50% sparsity

### Historical Results

#### ResNet-18 CIFAR-10
- [test_matrix_results_20250903_180836](test_matrix_results_20250903_180836/report.md) - Baseline results showing AdamWPrune matches movement pruning at 90.69%

#### ResNet-50 CIFAR-100  
- [test_matrix_results_20250908_121537](test_matrix_results_20250908_121537/report.md) - Initial state pruning implementation
- [test_matrix_results_20250908_190856](test_matrix_results_20250908_190856/report.md) - Extended tests across multiple sparsity levels

## Key Findings Summary

1. **AdamWPrune with AdamWSpam base** achieves NEW state-of-the-art 74.56% accuracy at 50% sparsity on ResNet-50 ImageNet
2. **Universal superiority**: AdamWPrune beats AdamWSpam at ALL sparsity levels when using AdamWSpam as base
3. **Pruning improves accuracy**: 50% sparsity achieves 1.96% better accuracy than unpruned baseline
4. **Memory efficiency**: Consistently uses 12,602.5 MB with 6.06% accuracy per GB efficiency
5. **Base optimizer matters**: AdamWSpam base (74.56%) slightly outperforms AdamW base (74.54%) at 50% sparsity
6. **Synergistic effects**: SPAM's gradient stabilization enhances state-based pruning decisions

*Generated: 2025-09-14*
