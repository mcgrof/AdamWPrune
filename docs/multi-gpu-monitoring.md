# Multi-GPU Monitoring and Analysis

## Overview

AdamWPrune provides comprehensive monitoring and analysis capabilities for multi-GPU training setups. The system automatically detects multi-GPU configurations (2+ GPUs) and dynamically adapts to any number of GPUs, providing both aggregate and per-GPU analysis. Tested and optimized for AWS g5.12xlarge instances with 4x NVIDIA A10G GPUs, but supports any multi-GPU configuration.

## Architecture

### GPU Monitoring Components

1. **GPU Monitor (`scripts/gputop.py`)**
   - Real-time monitoring of all GPUs simultaneously
   - Automatic detection of multi-GPU setups
   - Dual data format support (single + multi-GPU)

2. **Memory Comparison (`scripts/generate_gpu_memory_comparison.py`)**
   - Per-GPU memory breakdown visualization
   - Load balancing analysis across GPUs
   - Multi-GPU aware graph generation

3. **Consolidated Summary (`scripts/generate_consolidated_gpu_summary.py`)**
   - Aggregate statistics across all GPUs
   - Per-GPU performance breakdowns
   - Load balance quality assessment

## Data Formats

### Single GPU Format (Legacy)
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "gpu_index": 0,
  "gpu_name": "NVIDIA A10G",
  "utilization": 85,
  "memory_used": 12800,
  "memory_total": 24576,
  "power": 150
}
```

### Multi-GPU Format (New)
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "multi_gpu_data": [
    {
      "gpu_index": 0,
      "gpu_name": "NVIDIA A10G",
      "utilization": 85,
      "memory_used": 12800,
      "memory_total": 24576,
      "power": 150
    },
    // ... GPU 1, 2, 3 data
  ],
  "aggregate_stats": {
    "total_memory_used": 51200,
    "total_memory_total": 98304,
    "total_memory_percent": 52.1,
    "average_utilization": 82.5,
    "total_power": 600,
    "max_temperature": 78,
    "gpu_count": 4
  }
}
```

## GPU Load Balancing Analysis

### Load Balance Metrics

The system calculates Coefficient of Variation (CV) to measure load distribution:

```
CV = (standard_deviation / mean) × 100%
```

### Quality Classifications

| CV Range | Quality | Description |
|----------|---------|-------------|
| < 5%     | Excellent | Very well balanced across GPUs |
| 5-10%    | Good      | Well balanced distribution |
| 10-20%   | Fair      | Acceptable but could improve |
| ≥ 20%    | Poor      | Imbalanced - investigate issues |

### Example Analysis Output

```
ADAMWPRUNE Training - Per-GPU Breakdown:
    GPU 0: 12800.0 MB (±45.2)
    GPU 1: 12750.0 MB (±42.8)
    GPU 2: 12825.0 MB (±47.1)
    GPU 3: 12780.0 MB (±44.6)
    Total:  51155.0 MB
    Balance: CV = 0.3% (Excellent)
```

## Generated Visualizations

### 1. Per-GPU Memory Breakdown
- **File**: `per_gpu_memory_breakdown.png`
- **Content**: Dynamic grid layout based on GPU count (1x2 for 2 GPUs, 2x2 for 4 GPUs, 2x3 for 6 GPUs, etc.)
- **Purpose**: Identify individual GPU performance patterns

### 2. GPU Load Balance Analysis
- **File**: `gpu_load_balance_analysis.png`
- **Content**:
  - Left: Memory distribution across GPUs
  - Right: Load balance coefficient with quality indicators
- **Purpose**: Assess workload distribution efficiency

### 3. Enhanced Memory Comparison
- **File**: `gpu_memory_comparison.png`
- **Content**: Total memory usage with multi-GPU annotations
- **Purpose**: Compare optimizer efficiency in multi-GPU context

## Usage Examples

### Real-time Multi-GPU Monitoring

```bash
# Monitor all 4 GPUs with logging
python scripts/gputop.py --stats-file gpu_monitoring.json

# This generates:
# - gpu_monitoring.json (selected GPU data)
# - gpu_monitoring_multi_gpu.json (all GPU data)
```

### Generate Multi-GPU Analysis

```bash
# Generate comprehensive GPU analysis
python scripts/generate_gpu_memory_comparison.py test_matrix_results_20250115_103000/

# Auto-generates additional plots for multi-GPU:
# - per_gpu_memory_breakdown.png
# - gpu_load_balance_analysis.png
```

### Consolidated Multi-GPU Report

```bash
# Generate text report with multi-GPU analysis
python scripts/generate_consolidated_gpu_summary.py

# Output includes:
# - Per-optimizer aggregate statistics
# - Per-GPU breakdown for each optimizer
# - Load balance assessment
```

## Configuration for Multi-GPU

### Dynamic GPU Detection

The system automatically detects the number of available GPUs and adapts accordingly. No manual configuration of GPU count is required.

### GPT-2 Training Configuration

```bash
# Use the optimized defconfig with experiment tracking
make defconfig-gpt2-finewebedu-a10gx4 TRACKER=wandb,trackio
```

Key multi-GPU settings in defconfig:
```
CONFIG_GPT2_USE_DDP=y
CONFIG_GPT2_NUM_GPUS=4  # This is a hint, actual count auto-detected
CONFIG_GPT2_DDP_BACKEND="nccl"
CONFIG_GPU_MONITOR=y  # Enable monitoring
```

### Supported GPU Configurations

| GPU Count | Layout | Example Use Cases |
|-----------|--------|------------------|
| 2 GPUs    | 1x2 grid | Development, small models |
| 4 GPUs    | 2x2 grid | AWS g5.12xlarge, production training |
| 6 GPUs    | 2x3 grid | High-memory models |
| 8 GPUs    | 2x4 grid | Large-scale training |
| 8+ GPUs   | Dynamic grid | Research clusters |

### Test Matrix with Multi-GPU

```bash
# Run test matrix with GPU monitoring
make test_matrix GPU_MONITOR=y

# This automatically:
# 1. Enables GPU monitoring during training
# 2. Collects multi-GPU data
# 3. Generates multi-GPU analysis graphs
```

## Advanced Features

### Automatic Detection

The system automatically detects multi-GPU setups:
- Checks for multiple GPU devices
- Enables appropriate data collection
- Generates multi-GPU specific visualizations

### Backward Compatibility

All existing functionality works unchanged:
- Single-GPU setups continue to work
- Legacy JSON format maintained
- Existing scripts auto-detect data format

### Performance Impact

Multi-GPU monitoring overhead:
- **CPU**: < 1% additional overhead per GPU
- **Memory**: ~10MB additional for 4-GPU monitoring
- **Storage**: ~2x JSON file size for multi-GPU data

## Troubleshooting

### Common Issues

1. **Missing Multi-GPU Data**
   ```bash
   # Check GPU detection
   python -c "import scripts.gputop; m=scripts.gputop.GPUMonitor(); m.initialize(); print(f'GPUs: {m.gpu_count}')"
   ```

2. **Load Balance Issues (CV > 20%)**
   - Check DDP configuration
   - Verify batch size distribution
   - Monitor for GPU memory fragmentation
   - Ensure proper CUDA context management

3. **Incomplete Graphs**
   ```bash
   # Verify multi-GPU data files exist
   find . -name "*_multi_gpu.json" -ls
   ```

### Performance Optimization

For optimal 4x A10G performance:
- Use batch sizes divisible by 4
- Enable `CONFIG_GPT2_DDP_FIND_UNUSED_PARAMS=n` for efficiency
- Monitor load balance regularly (target CV < 5%)
- Use `CONFIG_PARALLEL_BATCH_SIZE=256` for memory efficiency

## Integration with Experiment Tracking

### WandB Integration

Multi-GPU metrics automatically logged to WandB:
```python
wandb.log({
    "gpu_total_memory": aggregate_stats["total_memory_used"],
    "gpu_avg_utilization": aggregate_stats["average_utilization"],
    "gpu_load_balance_cv": calculated_cv,
    "gpu_0_memory": per_gpu_stats[0]["memory_used"],
    # ... per-GPU metrics
})
```

### Trackio Integration

Local tracking includes multi-GPU dashboards:
- Real-time multi-GPU monitoring
- Historical load balance trends
- Per-GPU performance comparison

## File Locations

### Monitoring Scripts
- `scripts/gputop.py` - Real-time GPU monitoring
- `scripts/generate_gpu_memory_comparison.py` - Memory analysis
- `scripts/generate_consolidated_gpu_summary.py` - Summary reports

### Configuration Files
- `gpt2/defconfigs/gpt2-finewebedu-a10gx4` - 4x A10G optimized config
- `Kconfig` - Multi-GPU configuration options

### Output Files
- `gpu_stats*.json` - Single GPU monitoring data
- `gpu_stats*_multi_gpu.json` - Multi-GPU monitoring data
- `graphs/per_gpu_*.png` - Per-GPU visualization
- `graphs/gpu_load_balance_*.png` - Load balance analysis

## Best Practices

### Monitoring Strategy
1. **Continuous Monitoring**: Enable GPU monitoring for all training runs
2. **Load Balance Tracking**: Monitor CV < 5% for optimal performance
3. **Memory Efficiency**: Track total memory usage vs individual GPU peaks
4. **Performance Correlation**: Compare load balance with training speed

### Analysis Workflow
1. **Real-time**: Use `gputop.py` during development
2. **Post-training**: Generate comprehensive analysis with comparison scripts
3. **Optimization**: Use load balance metrics to tune batch sizes and DDP settings
4. **Reporting**: Include multi-GPU analysis in experiment documentation

### Optimization Guidelines
- **Excellent Balance (CV < 5%)**: Optimal configuration, no changes needed
- **Good Balance (CV 5-10%)**: Monitor trends, minor optimization possible
- **Fair Balance (CV 10-20%)**: Review batch size and data loading
- **Poor Balance (CV > 20%)**: Investigate DDP configuration and data distribution

## Experiment Tracking

### Dual Tracking with WandB and TrackIO

The system supports simultaneous tracking with both WandB (cloud) and TrackIO (local):

```bash
# Configure both trackers
make defconfig-gpt2-finewebedu-a10gx4 TRACKER=wandb,trackio

# Run tests with both tracking systems active
make test-matrix MAX_ITERS=500 TRACKER=wandb,trackio
```

Both trackers will log metrics in parallel:
- **WandB**: Cloud-based, collaborative, web dashboard
- **TrackIO**: Local, console-based, privacy-focused

### TrackIO Local Dashboard

#### Launch TrackIO Web Server
```bash
# Start the TrackIO dashboard server
make trackio-view

# Or specify a project
make trackio-view PROJECT=tracking-11f50
```

This starts a local web server (typically on port 7861) with:
- Real-time metrics updates
- Interactive graphs
- Training progress visualization
- Loss/accuracy trends

Note: TrackIO generates a unique write token for security. The server will display the full URL with token.

#### Get Dashboard URL Info
```bash
# Show URL without starting server
make trackio-web

# Output:
# URL: http://localhost:7860/?project=tracking-11f50
```

### Quick Test Configuration

For rapid iteration and testing:

```bash
# Run quick test with 500 iterations (~15 minutes per test)
make test-matrix MAX_ITERS=500 TRACKER=wandb,trackio

# This overrides the default full epoch training
# Useful for:
# - Validating pipeline functionality
# - Testing new configurations
# - Quick performance comparisons
```

### Live Console Monitoring

Monitor training progress from the console:

```bash
# Watch latest metrics
watch -n 2 'tail -5 test_matrix_results_*/gpt2_*/output.log | grep Iter'

# GPU memory timeline
while true; do
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader
  sleep 5
done | tee gpu_timeline.csv

# Compare multiple runs
for dir in test_matrix_results_*/gpt2_*; do
  echo "$dir:"
  grep "Final" $dir/output.log 2>/dev/null
done
```

### Progress Estimation

Check estimated completion time:

```bash
# Estimate time remaining for current test matrix
make estimate

# Output includes:
# - Current iteration/epoch progress
# - Time remaining per test
# - Total completion estimate
# - Per-GPU memory usage
```