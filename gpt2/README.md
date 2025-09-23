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

## AdamWPrune State Pruning: Bitter Lesson Variants (Active R&D)

### Background: The Bitter Lesson

Richard Sutton's "Bitter Lesson" teaches us that simple methods leveraging compute scale consistently outperform complex, clever approaches. Our initial AdamWPrune implementation used a sophisticated hybrid momentum-stability scoring system, but empirical results showed a 21% perplexity gap compared to simple magnitude pruning.

### Current Results (GPT-2 124M, 50% sparsity target, finewebedu dataset)

| Method | Perplexity | Actual Sparsity | Training Time | GPU Memory |
|--------|------------|-----------------|---------------|------------|
| AdamWSPAM + Magnitude | **42.36** | 49.9% | 578.6 min | 28.2 GB |
| AdamWSPAM + Movement | 65.71 | 78.1% (!) | 586.3 min | 29.2 GB |
| AdamWPrune + State (bitter0) | 51.39 | 49.7% | **478.8 min** | **24.7 GB** |

Key findings:
- AdamWPrune is **21% faster** and uses **14% less memory**
- However, it has **21% worse perplexity** than magnitude pruning
- The 21% symmetry (speed gain = quality loss) suggests a fundamental trade-off

### Bitter Lesson Variants Under Testing

We're testing three variants to apply the bitter lesson:

#### bitter0 (Original - Complex)
```bash
--adamwprune-variant bitter0
```
- Hybrid momentum-stability scoring: `importance = momentum * stability`
- Results: 51.39 perplexity, fastest training, lowest memory
- Problem: Too clever, doesn't match simple magnitude pruning quality

#### bitter1 (Pure Magnitude - Simple)
```bash
--adamwprune-variant bitter1
```
- Pure magnitude scoring: `importance = |weight|`
- Keeps boolean masks for memory efficiency
- Expected: ~42 perplexity (matching AdamWSPAM+magnitude)
- Hypothesis: Simple scoring + efficient storage = best of both worlds

#### bitter2 (Scale-Aware - Use Saved Resources)
```bash
--adamwprune-variant bitter2
```
- Same as bitter1 but automatically uses saved resources:
  - Increases iterations by 21% (10,000 → 12,100)
  - OR increase batch size by 14% (if memory permits)
- Hypothesis: Let scale compensate for simplicity

### Running Bitter Lesson Experiments

Test all variants:
```bash
make defconfig-gpt2-adamwprune-finewebedu-79000-bitter-lesson
make test-matrix
```

Or test individual variants:
```bash
# Test bitter1 (pure magnitude with boolean masks)
python train.py \
    --optimizer adamwprune \
    --pruning-method state \
    --adamwprune-variant bitter1 \
    --target-sparsity 0.5

# Test bitter2 (with automatic scaling)
python train.py \
    --optimizer adamwprune \
    --pruning-method state \
    --adamwprune-variant bitter2 \
    --target-sparsity 0.5
```

### Ongoing R&D Status

**Current Focus**: Validating if bitter1 (simple magnitude + boolean masks) can match AdamWSPAM+magnitude quality while maintaining the 21% speed and 14% memory advantages.

**Why This Matters**:
- If successful, we get the quality of magnitude pruning with significant efficiency gains
- The memory savings enable larger models or batch sizes
- Proves that clever scoring wasn't necessary - just efficient implementation

**Next Steps**:
1. Complete bitter lesson variant testing
2. If bitter1 succeeds, make it the default
3. Explore using saved memory for architectural improvements
4. Test on larger models (GPT-2 350M+) where memory savings are critical

### Technical Details

The key innovation isn't the scoring method but the **boolean mask storage**:
- Traditional pruning: float32 masks + scores = 2x weight memory overhead
- AdamWPrune: boolean masks = 0.03x weight memory overhead
- This 60x reduction in pruning overhead is the real breakthrough

The bitter lesson teaches us to focus on this efficiency gain rather than clever scoring algorithms.

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
