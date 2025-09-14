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

## Available Configurations

### Conservative (16GB System RAM, any GPU)
```bash
make defconfig-gpt2-shakespeare-conservative
make
```

### Standard (24GB+ GPU)
```bash
make defconfig-gpt2-shakespeare-simple
make
```

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
