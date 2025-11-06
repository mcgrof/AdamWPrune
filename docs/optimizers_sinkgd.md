# SinkGD: Sinkhorn-like Gradient Descent

## Overview

SinkGD is a novel optimizer that applies entropic optimal transport-inspired normalization to gradients before parameter updates. It uses iterative row/column normalization with temperature scaling to encourage structured, balanced gradient updates.

This optimizer is particularly useful in settings where vanilla adaptive methods like Adam/AdamW exhibit noisy or over-adaptive behavior, providing more stable optimization geometry through gradient transport regularization.

## Mathematical Foundation

### The Core Idea

Standard optimizers apply gradients directly or adaptively scale them per parameter. SinkGD instead transforms the gradient tensor itself using a Sinkhorn-like iterative normalization process before applying updates.

The gradient transform applies alternating row and column normalization:

```
For t = 1 to n_iter:
    g ← g / ||g||_row      (row normalization)
    g ← g / ||g||_col      (column normalization)
    g ← tanh(g / τ)        (temperature smoothing)
```

Where:
- `τ` (tau): Temperature parameter controlling smoothing (smaller = sharper)
- `n_iter`: Number of normalization iterations (more = stronger balancing)

After normalization, scale is restored to match original gradient RMS, preserving learning dynamics while regularizing the structure.

### Why "Sinkhorn-like"?

The Sinkhorn algorithm solves optimal transport problems by iteratively normalizing rows and columns of a cost matrix. SinkGD borrows this alternating normalization pattern to regularize gradient structure, encouraging balanced updates across parameter dimensions.

This is inspired by entropic optimal transport, where temperature parameter `τ` acts as an entropy regularizer, smoothing the gradient distribution.

### Decoupled Weight Decay

Like AdamW, SinkGD uses decoupled weight decay:

```
p ← p - λ * lr * p        (weight decay)
p ← p - lr * g_tilde      (gradient update)
```

Where `g_tilde` is the Sinkhorn-transformed gradient. This ensures weight decay is independent of the gradient transform.

## Implementation Details

### Algorithm

```python
class SinkGD:
    def __init__(self, lr=3e-4, weight_decay=0.1, tau=0.1, n_iter=5, eps=1e-8)

    def step(self):
        for p in parameters:
            g = p.grad

            # 1. Decoupled weight decay
            p -= weight_decay * lr * p

            # 2. Sinkhorn-like gradient transform
            g_tilde = sinkhorn_transform(g, tau, n_iter, eps)

            # 3. Parameter update
            p -= lr * g_tilde
```

### Hyperparameters

- **lr** (learning rate): Step size for parameter updates
  - Default: 3e-4
  - Range: 1e-4 to 1e-3 typical
  - Higher = faster convergence but less stable

- **weight_decay**: L2 penalty strength (decoupled)
  - Default: 0.1
  - Range: 0.0 to 0.5
  - Higher = stronger regularization

- **tau** (temperature): Smoothing parameter for tanh activation
  - Default: 0.1
  - Lower (0.05) = sharper, more precise gradients
  - Higher (0.2) = smoother, more diffuse gradients

- **n_iter** (iterations): Number of normalization cycles
  - Default: 5
  - More iterations = stronger balancing effect
  - Typical range: 3-10

- **eps** (epsilon): Numerical stability constant
  - Default: 1e-8
  - Prevents division by zero in normalizations

## Ablation Study: S0-S3

The SinkGD ablation study tests optimizer variants on the Lens L6 architecture (reciprocity + discoverability + K/V compression), comparing SinkGD against AdamWSPAM baseline.

### S0: Lens L6 + AdamWSPAM (control)

Baseline configuration using AdamWSPAM optimizer. Same architecture as L6 (parameter-neutral with K/V compression), but using standard adaptive momentum for comparison.

**Purpose**: Control experiment to isolate SinkGD's effect from architectural changes.

### S1: Lens L6 + SinkGD (balanced, default)

Test SinkGD with default balanced hyperparameters.

**Hyperparameters**:
- tau = 0.1 (moderate smoothing)
- n_iter = 5 (standard balancing)

**Expected behavior**: Balanced gradient structure regularization, potentially smoother loss curves than AdamWSPAM while maintaining convergence speed.

### S2: Lens L6 + SinkGD (sharper, precise)

Test SinkGD with sharper gradient transforms via lower temperature and more iterations.

**Hyperparameters**:
- tau = 0.05 (sharper, less smoothing)
- n_iter = 10 (stronger balancing)

**Expected behavior**: More precise gradient alignment, potentially faster initial convergence but requires careful learning rate tuning. Higher computational cost from more iterations.

### S3: Lens L6 + SinkGD (softer, smooth)

Test SinkGD with softer gradient transforms via higher temperature and fewer iterations.

**Hyperparameters**:
- tau = 0.2 (smoother, more diffuse)
- n_iter = 3 (lighter balancing)

**Expected behavior**: More exploratory optimization with smoother dynamics. May help escape sharp minima. Faster per-step computation from fewer iterations.

## Usage Examples

### Basic Usage

```bash
# S1: Default balanced SinkGD
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step S1 \
  --dataset finewebedu

# S2: Sharper SinkGD for precise optimization
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step S2 \
  --dataset finewebedu

# S3: Softer SinkGD for smoother dynamics
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step S3 \
  --dataset finewebedu
```

### Custom Hyperparameters

```bash
# Custom SinkGD configuration
python gpt2/train_ra_mla.py \
  --optimizer sinkgd \
  --sinkgd-lr 5e-4 \
  --sinkgd-weight-decay 0.15 \
  --sinkgd-tau 0.12 \
  --sinkgd-iters 7 \
  --dataset finewebedu
```

### Dry-Run Validation

Always validate new configurations before GPU training:

```bash
# Quick architecture validation (single step)
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step S1 \
  --dry-run

# Validate all SinkGD steps (~20 seconds total)
for step in S0 S1 S2 S3; do
  python gpt2/train_ra_mla.py \
    --ra-mla-ablation-step $step \
    --dry-run
done
```

## When to Use SinkGD

### Good Use Cases

- **Noisy gradients**: When training exhibits high variance in gradient magnitudes across parameters
- **Large models**: Where gradient structure matters for stable convergence
- **Exploration needed**: When stuck in sharp minima, softer SinkGD (S3) may help escape
- **Parameter imbalance**: When some layers dominate updates, Sinkhorn balancing can help

### When to Stick with AdamW/AdamWSPAM

- **Well-tuned baselines**: If AdamW already works well, switching may not help
- **Computational budget**: SinkGD adds per-step cost from normalization iterations
- **Sparse gradients**: SinkGD's row/col normalization assumes dense structure
- **Proven pipelines**: Production systems benefit from battle-tested optimizers

## Computational Cost

SinkGD adds overhead from iterative normalizations:

- **Per iteration**: 2 reductions (row + col norm) + 1 tanh activation
- **Default (n_iter=5)**: ~5× base gradient processing cost
- **Sharper (n_iter=10)**: ~10× base cost
- **Softer (n_iter=3)**: ~3× base cost

However, overhead is typically <5% of total training time since most cost is in forward/backward passes, not optimizer step.

## Technical Notes

### Numerical Stability

- All normalization done in fp32, even with mixed precision training
- Epsilon clamping prevents division by zero
- Scale restoration preserves original gradient RMS

### Edge Cases

- **1D tensors** (biases): Use simple L1 normalization, skip row/col structure
- **Scalar gradients**: Apply tanh smoothing only
- **Zero gradients**: Epsilon clamping ensures no NaN/Inf

### AMP Compatibility

SinkGD is fully compatible with PyTorch AMP (automatic mixed precision):
- Gradients transformed in fp32 for stability
- Results cast back to original dtype (bfloat16/fp16)
- No special GradScaler handling needed

## References

This implementation is inspired by:

- **Sinkhorn Algorithm**: Optimal transport via alternating projections
  - Sinkhorn & Knopp (1967): "Concerning nonnegative matrices and doubly stochastic matrices"

- **Entropic Regularization**: Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"

- **AdamW**: Decoupled weight decay pattern
  - Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"

## Victory Conditions

A successful SinkGD ablation should demonstrate:

1. **Comparable or better validation loss** vs AdamWSPAM baseline (S0)
2. **Smoother loss curves** with less gradient noise
3. **Stable convergence** across all three variants (S1, S2, S3)
4. **Trade-off clarity**: Understand when sharper (S2) vs softer (S3) helps

If S1/S2/S3 underperform S0, this suggests either:
- Lens architecture doesn't benefit from gradient structure regularization
- Hyperparameters need tuning for this specific task
- AdamWSPAM's momentum is better suited to GPT-2 training dynamics

The ablation study will reveal whether Sinkhorn-like gradient transport provides value for transformer optimization beyond standard adaptive methods.
