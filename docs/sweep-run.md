# Hyperparameter Sweep Guide

This guide explains how to run systematic hyperparameter sweeps for AdamWPrune optimization.

## Overview

The hyperparameter sweep system allows you to:
- Define parameter ranges and choices in a single configuration file
- Automatically generate all possible combinations
- Run tests sequentially with progress tracking
- Analyze results to find the best configuration

## Quick Start

```bash
# Generate and run a sweep with one command
make sweep RANGE_CONFIG=resnet18/defconfigs/resnet18-state-pruning-compare-range
```

This will:
1. Generate all configuration combinations
2. Run each configuration
3. Produce analysis report with best parameters

## Configuration Format

Create a range-based configuration file with the following format:

```bash
# Fixed parameters
CONFIG_MODEL_MODE_SINGLE=y
CONFIG_TARGET_SPARSITY=0.7

# Range parameters (start:step:end)
CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.03:0.02:0.09

# Choice parameters (option1|option2|option3)
CONFIG_ADAMWPRUNE_BETA1=0.85|0.87|0.89
CONFIG_ADAMWPRUNE_AMSGRAD=y|n

# Single value
CONFIG_ADAMWPRUNE_BETA2=0.999
```

### Parameter Specification Formats

1. **Single Value**: `CONFIG_PARAM=value`
   - Example: `CONFIG_ADAMWPRUNE_BETA2=0.999`

2. **Choices**: `CONFIG_PARAM=option1|option2|option3`
   - Example: `CONFIG_ADAMWPRUNE_AMSGRAD=y|n`
   - Example: `CONFIG_PRUNING_WARMUP=7800|11700|15600`

3. **Numeric Range**: `CONFIG_PARAM=start:step:end`
   - Example: `CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.03:0.02:0.09`
   - Generates: 0.03, 0.05, 0.07, 0.09

## Step-by-Step Process

### 1. Create Range Configuration

Example configuration for state pruning optimization:

```bash
# resnet18/defconfigs/resnet18-state-pruning-compare-range

# Fixed model settings
CONFIG_MODEL_SELECT_RESNET18=y
CONFIG_OPTIMIZER_SELECT_ADAMWPRUNE=y
CONFIG_PRUNING_METHOD="state"
CONFIG_TARGET_SPARSITY=0.7

# Tunable parameters
CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.03|0.05|0.07
CONFIG_ADAMWPRUNE_BETA1=0.85|0.87|0.89
CONFIG_PRUNING_WARMUP=7800|11700  # Epoch 20 or 30
CONFIG_PRUNING_RAMP_END_EPOCH=70|80
```

### 2. Generate Configurations

```bash
make sweep-generate RANGE_CONFIG=resnet18/defconfigs/resnet18-state-pruning-compare-range
```

This creates `sweep_configs/` directory with:
- Individual config files (`config_001`, `config_002`, etc.)
- `combinations_summary.txt` listing all combinations

### 3. Run the Sweep

```bash
make sweep-run
```

Or run with custom settings:

```bash
python scripts/run_test_matrix.py --config-dir sweep_configs --output-dir my_sweep_results
```

### 4. Analyze Results

Results are saved in `test_matrix_results/sweep_YYYYMMDD_HHMMSS/` with:

- **sweep_summary.json**: All test results in JSON format
- **sweep_analysis.txt**: Human-readable analysis report showing:
  - Best configuration with full parameters
  - Top 10 configurations ranked by test accuracy
  - Failed tests summary
- **Individual test directories**: Each containing:
  - `config.txt`: Exact configuration used
  - `training_metrics.json`: Training history
  - `output.log`: Full training log

## Example Sweep Analysis Output

```
Hyperparameter Sweep Analysis
============================================================

Total configurations tested: 36

BEST CONFIGURATION:
----------------------------------------
Config file: config_017
Test Accuracy: 88.51%
Final Sparsity: 70.0%
Training Time: 1842.3s

Hyperparameters:
  CONFIG_ADAMWPRUNE_BETA1=0.87
  CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.05
  CONFIG_PRUNING_WARMUP=7800
  CONFIG_PRUNING_RAMP_END_EPOCH=80
  CONFIG_TARGET_SPARSITY=0.7

============================================================
TOP 10 CONFIGURATIONS:
----------------------------------------

1. config_017
   Test Acc: 88.51% | Sparsity: 70.0% | Time: 1842.3s
2. config_023
   Test Acc: 88.43% | Sparsity: 70.0% | Time: 1838.7s
...
```

## Estimating Runtime

Calculate total runtime before starting:

```python
num_combinations = weight_decay_options * beta1_options * warmup_options * ramp_options
total_hours = (num_combinations * 32) / 60  # Assuming 32 minutes per test
```

Example:
- 3 weight decay × 3 beta1 × 2 warmup × 2 ramp = 36 tests
- 36 tests × 32 minutes = 19.2 hours

## Tips for Efficient Sweeps

### 1. Start Small
Begin with a coarse grid to identify promising regions:
```bash
CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.01|0.05|0.10
CONFIG_ADAMWPRUNE_BETA1=0.80|0.85|0.90
```

### 2. Refine Around Best Values
After finding good parameters, do a finer sweep:
```bash
# If 0.87 worked well for beta1
CONFIG_ADAMWPRUNE_BETA1=0.86|0.87|0.88
```

### 3. Fix Less Important Parameters
Reduce combinations by fixing parameters that have minimal impact:
```bash
# Fix AMSGrad if it consistently performs worse
CONFIG_ADAMWPRUNE_AMSGRAD=n
```

### 4. Use Shorter Epochs for Quick Tests
For initial exploration, reduce epochs:
```bash
CONFIG_NUM_EPOCHS=20  # Quick test to eliminate bad configs
```

## Common Parameter Ranges

### AdamWPrune Optimizer
- **Weight Decay**: 0.01 to 0.1 (higher for more regularization)
- **Beta1**: 0.85 to 0.95 (momentum coefficient)
- **Beta2**: 0.999 (usually fixed)
- **AMSGrad**: y or n

### Pruning Schedule
- **Warmup**: 
  - Epoch 10: 3900 steps
  - Epoch 20: 7800 steps
  - Epoch 30: 11700 steps
  - Epoch 40: 15600 steps
- **Ramp End**: Epochs 60-90
- **Target Sparsity**: 0.5 to 0.9

## Cleaning Up

```bash
# Remove generated configurations
make sweep-clean

# Remove test results (be careful!)
rm -rf test_matrix_results/sweep_*
```

## Advanced Usage

### Parallel Execution (Coming Soon)

For systems with multiple GPUs, parallel execution can reduce sweep time:
```bash
# Future feature
make sweep-parallel RANGE_CONFIG=config_file PARALLEL=4
```

### Custom Analysis Scripts

Create custom analysis by processing `sweep_summary.json`:

```python
import json
import pandas as pd

with open('sweep_summary.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
best = df.nlargest(5, 'test_acc')
print(best[['config_file', 'test_acc', 'final_sparsity']])
```

## Troubleshooting

### Out of Memory
If tests fail due to OOM:
- Reduce batch size in the range config
- Run fewer parallel jobs
- Monitor GPU memory usage

### Long Runtime
To reduce sweep time:
- Decrease number of epochs for initial exploration
- Use coarser parameter grids
- Focus on most impactful parameters

### Failed Tests
Check individual test logs:
```bash
cat test_matrix_results/sweep_*/config_XXX_*/output.log
```

Common issues:
- Learning rate too high/low
- Pruning warmup longer than total epochs
- Invalid parameter combinations

## Example Configurations

### Quick Test (2-3 hours)
```bash
CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.03|0.07
CONFIG_ADAMWPRUNE_BETA1=0.85|0.89
CONFIG_PRUNING_WARMUP=7800
CONFIG_PRUNING_RAMP_END_EPOCH=70
# Total: 4 configurations
```

### Comprehensive (24 hours)
```bash
CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.03|0.05|0.07
CONFIG_ADAMWPRUNE_BETA1=0.85|0.87|0.89
CONFIG_PRUNING_WARMUP=7800|11700
CONFIG_PRUNING_RAMP_END_EPOCH=70|80
# Total: 36 configurations
```

### Fine-tuning (6 hours)
```bash
CONFIG_ADAMWPRUNE_WEIGHT_DECAY=0.04|0.05|0.06
CONFIG_ADAMWPRUNE_BETA1=0.86|0.87|0.88
CONFIG_PRUNING_WARMUP=7800
CONFIG_PRUNING_RAMP_END_EPOCH=80
# Total: 9 configurations
```