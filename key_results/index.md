# Key Test Results Index

This directory contains key test matrix results that demonstrate important findings.

## Available Results

### ResNet-50 ImageNet Results

- **[test_matrix_results_20250908_121537](test_matrix_results_20250908_121537/summary_report.txt)** - September 2025
  - **Model**: ResNet-50 (25.6M parameters)
  - **Dataset**: ImageNet
  - **Key Finding**: AdamWPrune achieves 72.92% accuracy with lowest GPU memory (12,270 MiB)
  - **Optimizers Tested**: SGD, Adam, AdamW, AdamWAdv, AdamWSPAM, AdamWPrune
  - **Hardware**: AMD Radeon Pro W7900 (48GB)

### ResNet-18 CIFAR-10 Results

- **[test_matrix_results_20250903_180836](test_matrix_results_20250903_180836/report.md)** - September 2025
  - **Model**: ResNet-18 (11.2M parameters)
  - **Dataset**: CIFAR-10
  - **Key Finding**: AdamWPrune achieves 90.66% accuracy at 70% sparsity
  - **Memory Usage**: 1489.2 MiB (most efficient)

*Last Updated: 2025-09-10*
