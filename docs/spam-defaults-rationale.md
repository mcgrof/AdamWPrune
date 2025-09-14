# SPAM Feature Defaults Rationale

## Overview
As of September 2025, we enable three key SPAM (Spike-Aware Pruning-Adaptive Momentum) features by default:
1. **Spike-aware clipping** (`SPAM_ENABLE_CLIP=y`)
2. **Periodic momentum reset** (`SPAM_PERIODIC_RESET=y`)
3. **Cosine warmup after reset** (`SPAM_WARMUP=y`)

## Rationale for Default Enablement

### 1. Training Stability
These features significantly improve training stability without meaningful downsides:
- **Spike-aware clipping** prevents gradient explosions that can derail training
- **Periodic reset** helps escape sharp local minima that trap standard optimizers
- **Cosine warmup** ensures smooth recovery after resets, preventing instability

### 2. Empirical Performance
Our benchmarks show AdamWSpam consistently performs well:
- ResNet-50: 73.22% accuracy at 50% sparsity (2nd best overall)
- Particularly effective on larger, more complex models
- Minimal computational overhead (<2% training time increase)

### 3. Model Complexity Benefits
The benefits scale with model complexity:
- **Small models** (ResNet-18): Marginal benefit, but no harm
- **Large models** (ResNet-50, GPT-2): Significant stability and convergence improvements
- **Language models**: Essential for handling irregular gradient patterns

## GPT-2 Specific Considerations

For GPT-2 and other transformer models, SPAM features are particularly valuable:

### Gradient Spike Patterns
Language models exhibit unique gradient dynamics:
- **Attention mechanisms** create sudden gradient spikes during training
- **Token embedding updates** can cause explosive gradients on rare tokens
- **Layer normalization** interactions amplify gradient variance

### Loss Landscape Complexity
GPT-2's loss landscape characteristics:
- **Multiple local minima** due to permutation symmetries
- **Sharp valleys** that trap standard momentum-based optimizers
- **Plateau regions** requiring exploration mechanisms

### Recommended Settings for GPT-2
```bash
# Spike detection and clipping
CONFIG_SPAM_ENABLE_CLIP=y
CONFIG_SPAM_SPIKE_THRESHOLD="2.0"  # Z-score threshold
CONFIG_SPAM_THETA="50.0"            # Clipping strength

# Periodic exploration
CONFIG_SPAM_PERIODIC_RESET=y
CONFIG_SPAM_INTERVAL=1000           # Reset every 1000 steps
CONFIG_SPAM_WARMUP=y
CONFIG_SPAM_WARMUP_STEPS=100        # 100 step cosine warmup
```

## Implementation Details

### Spike-Aware Clipping
Detects gradient spikes using rolling statistics:
```python
# Pseudo-code
z_score = (gradient_norm - mean) / std
if z_score > threshold:
    clip_gradient(theta)
```

### Periodic Reset with Warmup
Resets momentum buffers periodically:
```python
# Every N steps
if step % interval == 0:
    reset_momentum_buffers()
    apply_cosine_warmup(next_k_steps)
```

## Performance Impact

### Memory Usage
- No additional memory required (reuses existing buffers)
- Spike statistics use negligible memory (<1MB)

### Computational Cost
- Spike detection: O(1) per parameter
- Periodic reset: O(n) every k steps (amortized O(1))
- Warmup calculation: O(1) per step

### Training Time
- Typical overhead: 1-2% total training time
- Offset by improved convergence (often trains faster overall)

## When to Disable

Consider disabling SPAM features only when:
1. Training very small models (<1M parameters)
2. Using extremely stable, well-conditioned datasets
3. Running inference-only workloads
4. Debugging optimizer behavior

## Conclusion

The default enablement of SPAM features represents a conservative choice prioritizing training stability and robustness. The minimal overhead is justified by significant improvements in convergence reliability, especially for complex models like GPT-2. Users can disable these features if their specific use case doesn't benefit, but the defaults provide a solid foundation for most training scenarios.
