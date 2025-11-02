# RATIO Pruning Variants

## Overview

RATIO pruning combines **structural importance** (golden ratio 1:2.5) with **dynamic importance** (from Adam optimizer states) to make pruning decisions that preserve the attention:MLP ratio.

### Key Insight: Respect Adam's Beta Values

Adam's default hyperparameters encode important information:
- `beta1 = 0.9` (momentum): 90% of past gradients
- `beta2 = 0.999` (variance): 99.9% of past gradients

Variance (`exp_avg_sq`) accumulates much more slowly than momentum (`exp_avg`), making it a more **conservative signal** for pruning. Using variance requires careful handling to avoid over-aggressive pruning.

---

## Pruning Variants

### Variant bitter7: Conservative Variance-Based
**Signal**: `|w| * (exp_avg_sq^0.25 + eps)`

**Rationale**:
- Uses second moment (variance) which accumulates slowly (beta2=0.999)
- Fourth root (`^0.25`) makes it even more conservative
- Finds parameters with consistently small gradients over long history
- Less susceptible to recent noise

**When to use**: When you want very stable pruning that only removes parameters with long-term low activity.

```python
importance = torch.abs(param) * (exp_avg_sq.abs()**0.25 + 1e-8)
```

---

### Variant bitter8: RATIO Structure-Aware (Momentum)
**Signal**: `structural_weight * |w| * sqrt(|exp_avg| + eps)`

**Rationale**:
- Combines golden ratio structural importance with momentum-based dynamic importance
- Attention params weighted 2.5× higher (they're scarce)
- MLP params weighted 1.0× (they're abundant)
- Coupling params weighted 3.0× (critical bidirectional flow)
- Uses `bitter3` momentum signal for dynamic component

**Structural weights**:
```python
structural_importance = {
    'attention': 2.5,    # 1/2.5 of total capacity → more valuable per-param
    'mlp_base': 1.0,     # 2.5/2.5 of capacity → baseline importance
    'coupling': 3.0      # Critical for RA↔MLP information flow
}
```

**When to use**: For RATIO models where you want to preserve the golden ratio during pruning.

```python
# Classify parameter by structure
if 'attn' in param_name:
    structural_weight = 2.5
elif 'mlp' in param_name and 'cross' not in param_name:
    structural_weight = 1.0
elif 'cross' in param_name or 'coupler' in param_name:
    structural_weight = 3.0

# Compute importance
momentum_signal = torch.sqrt(exp_avg.abs() + 1e-8)
importance = structural_weight * torch.abs(param) * momentum_signal
```

---

### Variant bitter9: RATIO Structure-Aware (Conservative Variance)
**Signal**: `structural_weight * |w| * (exp_avg_sq^0.25 + eps)`

**Rationale**:
- Combines golden ratio structural importance with conservative variance signal
- Most conservative variant: respects both structure AND long-term stability
- Best for finding parameters that are structurally less important AND dynamically stable

**When to use**: For final RATIO deployment where you want maximum confidence in pruning decisions.

```python
variance_signal = exp_avg_sq.abs()**0.25 + 1e-8
importance = structural_weight * torch.abs(param) * variance_signal
```

---

## Ratio-Preserving Pruning Algorithm

Goal: Prune while maintaining golden ratio 1:2.5.

**Constraint equations**:
```
1. Total sparsity: α·A + μ·M = S·(A + M)
2. Ratio preservation: M·(1-μ) / [A·(1-α)] = 2.5

Where:
- A = total attention params
- M = total MLP params
- S = target sparsity (e.g., 0.5 = 50%)
- α = fraction to prune from attention
- μ = fraction to prune from MLP
```

**Solution**:
```python
def calculate_pruning_budgets(A, M, S, ratio=2.5):
    """Calculate pruning fractions that preserve golden ratio."""
    # Solve system of equations
    alpha = (S*(A+M) + A*(ratio-1) - M) / (A*(1+ratio))
    mu = (S*(A+M) - alpha*A) / M
    return alpha, mu

# Example: A=1.57M (MLA attention), M=3.93M (golden ratio MLP), S=0.5
alpha, mu = calculate_pruning_budgets(1.57e6, 3.93e6, 0.5, 2.5)
# Result: alpha ≈ 0.56 (prune 56% of attention)
#         mu ≈ 0.48 (prune 48% of MLP)
# After pruning: ratio = (3.93M × 0.52) / (1.57M × 0.44) ≈ 2.5 ✓
```

**Algorithm**:
1. Calculate pruning budgets (α, μ) to preserve ratio
2. Within attention budget: prune by importance score (lowest first)
3. Within MLP budget: prune by importance score (lowest first)
4. Verify ratio is maintained within tolerance (2.3 ≤ ratio ≤ 2.7)

---

## Comparison with Standard Pruning

### Standard Pruning (Structure-Blind)
```python
# Compute importance for ALL parameters
all_importance = compute_importance(all_params)

# Sort and prune bottom 50%
sorted_params = sort_by_importance(all_importance)
prune_bottom_50_percent(sorted_params)

# Problem: No control over attention:MLP ratio!
# Result: May prune 80% of attention, 10% of MLP → ratio destroyed
```

### RATIO Pruning (Structure-Aware)
```python
# Classify parameters by structure
attn_params, mlp_params = classify_by_structure(model)

# Calculate budgets to preserve ratio
alpha, mu = calculate_pruning_budgets(...)

# Compute importance with structural weighting
attn_importance = [structural_weight_attn * dynamic_importance(p)
                   for p in attn_params]
mlp_importance = [structural_weight_mlp * dynamic_importance(p)
                  for p in mlp_params]

# Prune within budgets
prune_bottom_fraction(attn_params, alpha, by=attn_importance)
prune_bottom_fraction(mlp_params, mu, by=mlp_importance)

# Result: Golden ratio preserved!
```

---

## Implementation Notes

### Identifying Parameter Structures

```python
def classify_parameter_structure(name, param):
    """Classify parameter by its structural role in RATIO."""
    if any(k in name for k in ['attn', 'attention', 'q_proj', 'k_proj',
                                 'v_proj', 'o_proj', 'q_latent', 'v_up']):
        return 'attention'
    elif any(k in name for k in ['cross', 'coupler', 'gate_to_heads',
                                   'recip']):
        return 'coupling'
    elif any(k in name for k in ['mlp', 'c_fc', 'c_proj']):
        return 'mlp_base'
    else:
        return 'other'
```

### Accessing Optimizer States

```python
# In AdamW optimizer step
for group in self.param_groups:
    for p in group['params']:
        if p.grad is None:
            continue

        state = self.state[p]
        exp_avg = state['exp_avg']          # Momentum (beta1=0.9)
        exp_avg_sq = state['exp_avg_sq']    # Variance (beta2=0.999)

        # Compute importance
        if variant == 'bitter7':
            importance = p.abs() * (exp_avg_sq.abs()**0.25 + 1e-8)
        elif variant == 'bitter8':
            struct_weight = get_structural_weight(p)
            importance = struct_weight * p.abs() * torch.sqrt(exp_avg.abs() + 1e-8)
        elif variant == 'bitter9':
            struct_weight = get_structural_weight(p)
            importance = struct_weight * p.abs() * (exp_avg_sq.abs()**0.25 + 1e-8)

        # Store for pruning decision
        self.importance_scores[p] = importance
```

---

## Recommended Workflow

### Step 1: Train with Structure Tracking
```bash
make defconfig-gpt2-finewebedu-a10gx4-ra-mla-full
make  # Trains all 15 ablation steps
```

### Step 2: Apply RATIO Pruning (Step 14)
Use `bitter9` (most conservative) for final deployment:
```python
pruning_plan = optimizer.get_ratio_preserving_pruning_plan(
    target_sparsity=0.5,
    variant='bitter9',  # Conservative variance + structural
    ratio=2.5,
    tolerance=0.2
)
apply_pruning(model, pruning_plan)
```

### Step 3: Verify
```python
# Check ratio is preserved
attn_params = count_params(model, structure='attention')
mlp_params = count_params(model, structure='mlp')
ratio = mlp_params / attn_params
assert 2.3 <= ratio <= 2.7, f"Ratio {ratio:.2f} outside tolerance!"
print(f"Final ratio: 1:{ratio:.2f} ✓")
```

---

## Future Work

1. **Adaptive structural weights**: Learn optimal weights during training
2. **Layer-wise ratios**: Different ratios for early/mid/late layers
3. **Gradual pruning**: Slowly increase sparsity while maintaining ratio
4. **Quantization-aware**: Adjust effective ratio for INT8/INT4 deployment

---

**Last Updated**: 2025-11-02
**Status**: Design document for implementation
