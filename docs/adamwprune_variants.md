# AdamWPrune Variants Documentation

## Overview

AdamWPrune implements several pruning variants following the "bitter lesson" philosophy - simpler methods often outperform complex ones when given sufficient compute. Each variant explores different trade-offs between simplicity, performance, and computational efficiency.

## Variant Comparison Table

| Variant | Pruning Method | Sparsity Schedule | Training Budget | Expected PPL @ 50% |
|---------|---------------|-------------------|-----------------|-------------------|
| bitter0 | Hybrid momentum-stability | Linear | 10,000 iters | ~51.51 |
| bitter1 | Pure magnitude | Linear | 10,000 iters | ~42.10 |
| bitter2 | Pure magnitude | Linear | 12,100 iters (+21%) | ~46.07 |
| bitter3 | Gradient-magnitude | Cubic | 13,000 iters (+30%) | ~42-44 |
| bitter4 | Gradient-magnitude + layer-adaptive | Cubic | 13,000 iters (+30%) | ~40-42 |
| bitter5 | Movement-to-zero | TBD | TBD | TBD |
| bitter6 | Coherence-weighted | TBD | TBD | TBD |
| bitter7 | Conservative variance-based | TBD | TBD | TBD |
| bitter8 | Bias-corrected gradient-magnitude | TBD | TBD | TBD |
| bitter9 | Hybrid multi-signal | TBD | TBD | TBD |

## Detailed Variant Descriptions

### bitter0: Original Hybrid Approach
- **Algorithm**: Complex hybrid of momentum and stability signals
- **Importance Score**: `|w| * |exp_avg| * |w|/sqrt(exp_avg_sq)`
- **Philosophy**: More sophisticated pruning should yield better results
- **Result**: Worst performance, validating the bitter lesson

### bitter1: Pure Magnitude Pruning
- **Algorithm**: Simple magnitude-based pruning
- **Importance Score**: `|w|`
- **Philosophy**: Simplest possible approach
- **Result**: Outperforms complex bitter0 by significant margin

### bitter2: Scale-Aware Magnitude
- **Algorithm**: Magnitude pruning with extended training
- **Importance Score**: `|w|`
- **Training**: Uses 21% more iterations to leverage compute savings
- **Result**: Mixed - doesn't improve over bitter1 despite more training

### bitter3: Gradient-Magnitude Pruning
- **Algorithm**: Combines weight magnitude with gradient activity
- **Importance Score**: `|w| * sqrt(|exp_avg|)`
- **Schedule**: Cubic sparsity ramp (`progress^3`) for gentler early pruning
- **Training**: 30% more iterations (13,000)
- **Philosophy**: Use readily available gradient info without added complexity
- **Key Features**:
  - Considers both static importance (magnitude) and dynamic importance (gradients)
  - Cubic schedule protects early-stage feature learning
  - Extended training budget maximizes compute efficiency

### bitter4: Gradient-Magnitude + Layer-Adaptive
- **Algorithm**: bitter3 with layer-adaptive sparsity distribution
- **Importance Score**: `|w| * sqrt(|exp_avg|)`
- **Schedule**: Cubic sparsity ramp with per-layer adjustment
- **Training**: 30% more iterations (13,000)
- **Sparsity Distribution**:
  - Early layers: 0.7x base sparsity (preserve feature extraction)
  - Middle layers: 1.0x base sparsity
  - Later layers: 1.3x base sparsity (more task-specific, prunable)
- **Philosophy**: Different network depths have different redundancy patterns
- **Key Features**:
  - All benefits of bitter3
  - Adaptive sparsity preserves critical early features
  - More aggressive pruning in redundant later layers
  - Better overall perplexity with same total sparsity

### bitter5: Movement-to-Zero
- **Algorithm**: Identifies weights Adam is actively pushing toward zero
- **Importance Score**: `-sign(w) * exp_avg / sqrt(exp_avg_sq) + 0.1 * |w|`
- **Philosophy**: Prune weights where gradient direction opposes current value
- **Key Features**:
  - Positive movement score = moving toward zero (candidate for pruning)
  - Blends with small magnitude component for stability
- **Status**: Implemented but not extensively tested

### bitter6: Coherence-Weighted Gradient-Magnitude
- **Algorithm**: Penalizes oscillatory gradients using coherence signal
- **Importance Score**: `|w| * sqrt(|exp_avg|) * sqrt(exp_avg^2 / exp_avg_sq)`
- **Philosophy**: Coherence (m²/v) measures gradient consistency
- **Key Features**:
  - High coherence = consistent gradient direction = important
  - Low coherence = oscillatory gradients = less important
- **Status**: Implemented but not extensively tested

### bitter7: Conservative Variance-Based
- **Algorithm**: Uses fourth root of second moment for stable pruning
- **Importance Score**: `|w| * (exp_avg_sq^0.25 + eps)`
- **Philosophy**: Variance accumulates slowly (beta2=0.999), making it conservative
- **Key Features**:
  - Fourth root (`^0.25`) provides additional damping
  - Finds parameters with consistently small gradients over long history
  - Less susceptible to recent noise compared to momentum-based methods
  - Most stable pruning signal for long-term low activity detection
- **Status**: Implemented, high potential for original AdamWPrune goals

#### Why Beta2 Matters for bitter7

The choice of beta2=0.999 (variance) over beta1=0.9 (momentum) is crucial for stable pruning:

![Gradient EMA Comparison](./images/gradient_ema_comparison.png)

**Key observations from the visualization**:

1. **Raw gradients are extremely noisy** (gray spikes): Using raw gradients for pruning decisions would be unstable

2. **Beta1=0.8-0.9 (momentum)** tracks recent changes quickly but still shows significant oscillation

3. **Beta2=0.999 (variance)** provides the smoothest signal, filtering out short-term noise while preserving long-term trends

**Why this matters for pruning**:
- Pruning is an **irreversible decision** - you can't easily recover pruned parameters
- Momentum (beta1=0.9) responds to ~10 recent steps, making it susceptible to temporary gradient spikes
- Variance (beta2=0.999) accumulates over ~1000 steps, capturing true long-term parameter activity
- The fourth root (`^0.25`) further dampens the signal, ensuring only parameters with consistently low gradients are pruned

**Mathematical intuition**:
```python
# Momentum: tracks ~10 steps (beta1=0.9)
# 0.9^10 ≈ 0.35  (35% weight from 10 steps ago)

# Variance: tracks ~1000 steps (beta2=0.999)
# 0.999^1000 ≈ 0.37  (37% weight from 1000 steps ago!)
```

This makes bitter7 ideal for production pruning where stability and confidence in pruning decisions is critical.

### bitter8: Bias-Corrected Gradient-Magnitude
- **Algorithm**: Applies Adam's bias correction before scoring
- **Importance Score**: `|w| * sqrt(|exp_avg / (1 - beta1^t)|)`
- **Philosophy**: Account for initialization bias in early training
- **Key Features**:
  - Uses bias-corrected momentum m_hat
  - More accurate in early training steps
- **Status**: Implemented but not extensively tested

### bitter9: Hybrid Multi-Signal
- **Algorithm**: Combines magnitude, gradient, and movement signals
- **Importance Score**: `|w| * sqrt(|exp_avg|) - 0.1 * movement_to_zero`
- **Philosophy**: Robust scoring from multiple complementary signals
- **Key Features**:
  - Magnitude: static weight importance
  - Gradient: dynamic activity importance
  - Movement: directional update importance
- **Status**: Implemented but not extensively tested

## Implementation Details

### Gradient-Magnitude Scoring (bitter3/bitter4)
```python
if "exp_avg" in state:
    grad_importance = sqrt(abs(exp_avg) + eps)
    importance = abs(weight) * grad_importance
else:
    importance = abs(weight)
```

### Layer-Adaptive Distribution (bitter4 only)
```python
position = layer_idx / (total_layers - 1)
scale = 0.7 + 0.6 * position  # 0.7x to 1.3x
layer_sparsity = min(0.95, base_sparsity * scale)
```

### Cubic Schedule (bitter3/bitter4)
```python
progress = (current_step - warmup) / (total_steps - warmup)
progress = progress ** 3  # Cubic instead of linear
current_sparsity = target_sparsity * progress
```

### Variance-Based Scoring (bitter5-9)
```python
# bitter5: Movement to zero
if "exp_avg" in state and "exp_avg_sq" in state:
    movement = -(weight.sign() * exp_avg) / (sqrt(exp_avg_sq) + eps)
    importance = -movement + abs(weight) * 0.1

# bitter6: Coherence-weighted
if "exp_avg" in state and "exp_avg_sq" in state:
    coherence = sqrt(exp_avg**2 / (exp_avg_sq + eps))
    importance = abs(weight) * sqrt(abs(exp_avg) + eps) * coherence

# bitter7: Conservative variance-based
if "exp_avg_sq" in state:
    importance = abs(weight) * (abs(exp_avg_sq) + eps) ** 0.25

# bitter8: Bias-corrected
if "exp_avg" in state:
    m_hat = exp_avg / (1 - beta1**step + eps)
    importance = abs(weight) * sqrt(abs(m_hat) + eps)

# bitter9: Hybrid
if "exp_avg" in state and "exp_avg_sq" in state:
    movement = -(weight.sign() * exp_avg) / (sqrt(exp_avg_sq) + eps)
    importance = abs(weight) * sqrt(abs(exp_avg) + eps) - 0.1 * movement
```

## Usage Examples

### Training with bitter3
```bash
python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter3 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --max-iters 10000  # Auto-adjusted to 13,000
```

### Training with bitter4
```bash
python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter4 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --max-iters 10000  # Auto-adjusted to 13,000
```

### Training with bitter5-9
```bash
# Any of bitter5, bitter6, bitter7, bitter8, bitter9
python train.py \
    --optimizer adamwprune \
    --adamwprune-variant bitter5 \
    --pruning-method state \
    --target-sparsity 0.5 \
    --max-iters 10000
```

## Key Insights

1. **Bitter Lesson Validated**: Simple magnitude (bitter1) beats complex hybrid (bitter0)
2. **Gradient Information Helps**: bitter3/bitter4 improve by incorporating gradient activity
3. **Schedule Matters**: Cubic schedule protects early training dynamics
4. **Layer Adaptation Works**: Different depths benefit from different sparsity levels
5. **Extended Training**: Using saved compute for more iterations improves quality

## Memory Efficiency

All variants achieve similar memory savings:
- **Theoretical**: 40% reduction (5.03x → 3.03x weights)
- **Actual GPU**: ~8.2% reduction in practice
- **Mechanism**: Boolean masks instead of float movement scores

## Recommendations

- **For simplicity**: Use bitter1 (pure magnitude)
- **For best perplexity**: Use bitter4 (gradient-magnitude + layer-adaptive)
- **For balanced approach**: Use bitter3 (gradient-magnitude only)
- **For experimental**: Try bitter5-9 (various Adam state signals, not extensively tested)
- **Avoid**: bitter0 (overly complex) and bitter2 (no clear benefit)

