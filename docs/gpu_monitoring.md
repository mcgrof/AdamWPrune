# GPU Monitoring Integration

This project integrates GPU monitoring using [gputop](https://github.com/mcgrof/gputop), a vendor-agnostic GPU monitoring tool that provides consistent memory and performance tracking across NVIDIA, AMD, and Intel GPUs.

## Overview

The GPU monitoring system provides:
- Real-time GPU performance tracking during training
- Vendor-agnostic monitoring (works with NVIDIA, AMD, Intel GPUs)
- JSON data export for analysis
- Automatic graph generation
- A/B testing comparison between different configurations
- Memory usage, utilization, temperature, and power monitoring

## Why gputop?

Traditional GPU monitoring relies on vendor-specific tools:
- NVIDIA: nvidia-smi
- AMD: rocm-smi
- Intel: intel_gpu_top

[gputop](https://github.com/mcgrof/gputop) provides a unified interface across all vendors, enabling:
- Consistent data collection regardless of GPU vendor
- Simplified integration without vendor-specific code
- Portable monitoring scripts that work on any GPU
- Standardized output format for analysis

## Installation

The `scripts/gputop.py` file is included from the [gputop project](https://github.com/mcgrof/gputop). No additional installation is required as it's bundled with this repository.

## Usage

### 1. Direct Training with Monitoring

```bash
# Train LeNet-5 with AdamW and magnitude pruning at 70% sparsity
python3 scripts/train_with_monitoring.py \
    --model lenet5 \
    --config-name "lenet5_adamw_magnitude_70" \
    --generate-graphs \
    -- \
    --optimizer adamw \
    --pruning-method magnitude \
    --target-sparsity 0.7
```

### 2. Test Matrix with GPU Monitoring

The test matrix runner automatically uses GPU monitoring for all training runs:

```bash
# Run test matrix with GPU monitoring for all tests
make test-matrix

# Run parallel test matrix with GPU monitoring
make parallel
```

### 3. A/B Testing Comparison

Compare different optimizer configurations:

```bash
# Run two different configurations
python3 scripts/train_with_monitoring.py --model lenet5 --config-name "run1" -- --optimizer adamw
python3 scripts/train_with_monitoring.py --model lenet5 --config-name "run2" --compare-with results/gpu_stats_run1_*.json -- --optimizer adamwspam

# Manual comparison using gputop
python3 scripts/gputop.py --compare results/gpu_stats_run1_*.json results/gpu_stats_run2_*.json
```

### 4. Visualization

Generate graphs from collected GPU statistics:

```bash
# Generate comparison graphs
python3 scripts/visualize_gpu_comparison.py results/gpu_stats_run1_*.json results/gpu_stats_run2_*.json

# Generate memory timeline
python3 scripts/generate_gpu_memory_comparison.py test_matrix_results_20250829_040425
```

## Output Files

Each monitored training run generates:
- `gpu_stats_<config>_<timestamp>.json` - Raw GPU statistics
- `gpu_stats_<config>_<timestamp>_summary.txt` - Human-readable summary
- `gpu_stats_<config>_<timestamp>_plot.png` - GPU usage graphs (if --generate-graphs)

## Data Format

The JSON output contains timestamped GPU metrics:

```json
{
  "timestamp": "2025-08-29T04:35:11.660455",
  "gpu_name": "AMD Radeon Pro W7900",
  "gpu_type": "AMD",
  "utilization": 95.0,
  "memory_used": 1424.3,
  "memory_total": 49152.0,
  "memory_percent": 2.9,
  "power": 250.0,
  "temperature": 75.0,
  "elapsed_seconds": 10.5,
  "training_metadata": {
    "epoch": 1,
    "batch": 100,
    "phase": "training"
  }
}
```

## Integration with Test Matrix

The test matrix system automatically:
1. Monitors GPU usage for each test configuration
2. Saves statistics to the test results directory
3. Generates comparative visualizations
4. Includes GPU memory analysis in summary reports

Example test matrix results:
```
test_matrix_results_20250829_040425/
├── resnet18_adamwprune_state_70/
│   ├── gpu_stats_*.json         # Raw GPU data
│   ├── gpu_stats_*_summary.txt  # Summary statistics
│   └── gpu_stats_*_plot.png     # GPU usage graph
└── graphs/
    ├── gpu_memory_timeline.png   # Combined timeline
    └── training_memory_comparison.png  # Comparative analysis
```

## Key Findings from GPU Monitoring

Using gputop monitoring, we've measured:
- **AdamWPrune**: 1381.5 MiB mean GPU memory (ResNet-18, 70% sparsity)
- **Other optimizers**: ~1494 MiB average
- **Memory savings**: 7.5% reduction with AdamWPrune

These measurements were collected consistently across different GPU vendors using the unified gputop interface.

## Contributing

For improvements to the GPU monitoring tool itself, please contribute to the upstream [gputop project](https://github.com/mcgrof/gputop).

## License

The gputop.py script maintains its original license from the [gputop project](https://github.com/mcgrof/gputop). See the script header for details.