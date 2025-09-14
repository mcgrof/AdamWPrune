# GPT-2 Training Module

## Memory Requirements

### System RAM Requirements
- **GPT-2 124M**: ~8-16GB system RAM
- **GPT-2 350M (medium)**: ~32-64GB system RAM (can cause system lockup!)
- **GPT-2 774M (large)**: ~64-128GB system RAM
- **GPT-2 1.5B (xl)**: 128GB+ system RAM

**WARNING**: The model loading and tokenization process happens in system RAM before being transferred to GPU. Using large models without sufficient system RAM will cause severe system slowdown or crashes.

### GPU VRAM Requirements (approximate)
- **GPT-2 124M**:
  - Batch size 32: ~4GB VRAM
  - Batch size 64: ~8GB VRAM
  - Batch size 96: ~12GB VRAM
  - Batch size 128: ~16GB VRAM

## Data Preparation

Before training, prepare the dataset:
```bash
python3 gpt2/prepare_data.py --dataset shakespeare
```

## Optimizer Recommendations for GPT-2

### Why SPAM Features Are Essential for GPT-2

GPT-2 and other transformer models benefit significantly from SPAM (Spike-Aware Pruning-Adaptive Momentum) features:

1. **Gradient Spikes**: Attention mechanisms and rare token updates cause sudden gradient explosions
2. **Complex Loss Landscape**: Multiple local minima and sharp valleys require advanced exploration
3. **Training Stability**: Language models are prone to instability without proper gradient management

### Recommended SPAM Settings

The following SPAM features are now **enabled by default** for GPT-2:

```bash
# Spike detection and clipping (prevents gradient explosions)
CONFIG_SPAM_ENABLE_CLIP=y          # Detect and clip gradient spikes
CONFIG_SPAM_SPIKE_THRESHOLD="2.0"  # Z-score threshold for spike detection
CONFIG_SPAM_THETA="50.0"           # Clipping strength parameter

# Periodic exploration (escape local minima)
CONFIG_SPAM_PERIODIC_RESET=y       # Reset momentum periodically
CONFIG_SPAM_INTERVAL=1000          # Reset every 1000 steps
CONFIG_SPAM_WARMUP=y               # Smooth recovery after reset
CONFIG_SPAM_WARMUP_STEPS=100       # 100 step cosine warmup
```

These settings provide:
- **Stability**: Prevents training crashes from gradient explosions
- **Better convergence**: Escapes suboptimal local minima
- **Minimal overhead**: <2% training time increase

## Available Configurations

### NVIDIA L4 Optimized (24GB VRAM)
```bash
make defconfig-gpt2-shakespeare-l4
make
```

### AMD W7900 Optimized (48GB VRAM)
```bash
make defconfig-gpt2-shakespeare-w7900
make
```
**Note**: Uses 124M model to avoid system RAM issues. For 350M+ models, ensure you have 64GB+ system RAM.

### Multi-Optimizer Testing
```bash
make defconfig-gpt2-shakespeare-matrix
make test-matrix
```

## Troubleshooting

### System becomes unresponsive during model loading
- **Cause**: Insufficient system RAM for the model size
- **Solution**: Use smaller model (124M) or add more system RAM
- **Emergency**: SSH in and `pkill -9 python` or `pkill -9 train`

### CUDA out of memory
- **Cause**: Batch size too large for GPU VRAM
- **Solution**: Reduce batch_size or use gradient accumulation

### Training is slow
- **Cause**: Too many data workers or inefficient settings
- **Solution**: Reduce num_workers if system RAM is limited

## Model Sizes

| Model | Parameters | Layers | Hidden Size | Heads | System RAM | VRAM (BS=32) |
|-------|------------|--------|-------------|-------|------------|--------------|
| gpt2 | 124M | 12 | 768 | 12 | 8-16GB | 4GB |
| gpt2-medium | 355M | 24 | 1024 | 16 | 32-64GB | 8GB |
| gpt2-large | 774M | 36 | 1280 | 20 | 64-128GB | 16GB |
| gpt2-xl | 1.5B | 48 | 1600 | 25 | 128GB+ | 24GB |

## Performance Tips

1. **System RAM is critical**: Model loading happens in system RAM first
2. **Use mixed precision**: Reduces VRAM usage by ~50%
3. **Gradient accumulation**: Simulate larger batches without more VRAM
4. **Reduce workers**: Each worker duplicates data in RAM
5. **Flash attention**: Reduces memory usage and speeds up training

## Example Training Command

Direct training with custom settings:
```bash
cd gpt2
python3 train.py \
    --batch-size 32 \
    --gradient-accumulation 4 \
    --num-epochs 2 \
    --optimizer adamw \
    --pruning-method none \
    --device cuda
```
