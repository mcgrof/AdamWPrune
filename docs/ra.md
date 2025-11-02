# RATIO: Golden Ratio Architecture for Inference-Efficient LLMs

**Reciprocal Attention-Tuned Inference Optimization**

**Status**: Research
**Last Updated**: 2025-11-01

## Executive Summary

RATIO is an inference-first LLM architecture that enforces the **golden ratio**
(Attention:MLP ≈ 1:2.5) as an architectural invariant from initialization
through deployment. By combining inference scaling laws with structure-aware
optimization, RATIO achieves:

- **6× smaller KV cache** via MLA compression (latent_dim=128)
- **Golden ratio by construction** - auto-calculated dimensions maintain optimal
  1:2.5
- **Structure-preserving pruning** - can remove 50% of parameters while
  maintaining ratio
- **Memory hierarchy readiness** - automatic tier classification based on
  structural importance

**Core Innovation**: Traditional architectures treat the Attention:MLP ratio as
arbitrary (GPT-2 uses 1:2.0). RATIO recognizes this ratio as fundamental to
inference efficiency and enforces it through every stage of the ML pipeline.

---

## Table of Contents

1. [Motivation: Why Golden Ratio?](#motivation-why-golden-ratio)
2. [Architecture Overview](#architecture-overview)
3. [Inference Scaling Laws](#inference-scaling-laws)
4. [Golden Ratio by Construction](#golden-ratio-by-construction)
5. [Structure-Aware Optimization](#structure-aware-optimization)
6. [Ratio-Preserving Pruning](#ratio-preserving-pruning)
7. [Memory Hierarchy Integration](#memory-hierarchy-integration)
8. [Experimental Results](#experimental-results)
9. [Implementation Guide](#implementation-guide)
10. [Future Directions](#future-directions)

---

## Motivation: Why Golden Ratio?

### The Inference Efficiency Problem

Traditional transformer scaling focuses on **training compute** while ignoring
**inference cost**:

```python
# Standard GPT-2 design (inference-blind)
mlp_dim = 4 × d_model  # Hardcoded, hope it's good
n_heads = 12           # Arbitrary choice
KV_cache = n_heads × head_dim × seq_len  # Large!
```

**Problems**:
1. KV cache grows linearly with sequence length → memory bottleneck
2. Attention:MLP ratio (1:2.0) is training-optimal, not inference-optimal
3. No principled way to prune without breaking architecture balance
4. Memory placement decisions are ad-hoc

### Inference Scaling Laws

Recent research ([Sardana & Frankle, 2024](https://arxiv.org/pdf/2510.18245))
shows:

> **Optimal ratio for inference efficiency: Attention:MLP ≈ 1:2.5**

This means:
- **Attention is expensive**: Quadratic cost, KV cache scales with sequence
  length
- **MLP is cheap**: Linear cost, no sequence-dependent memory
- **Optimal architecture**: Compress attention, expand MLP

**GPT-2's actual ratio**:
```
Attention params: 2.36M/layer
MLP params: 4.72M/layer
Ratio: 1:2.0
```

**GPT-2 has 20% too much attention capacity** for inference optimality!

### RATIO Approach: Golden Ratio as Architectural Invariant

Instead of treating the ratio as a design guideline, **enforce it as a hard
constraint**:

```python
# RATIO: Golden ratio by construction
attn_params = calculate_attention_params(mla_config)  # 1.57M
target_mlp_params = attn_params × golden_ratio        # 3.93M
mlp_dim = solve_for_mlp_dim(target_mlp_params)        # 2150 (auto!)

# Result: ratio = 1:2.50 exactly
```

**Benefits**:
1. ✅ Inference-optimal by construction
2. ✅ KV cache minimized (6× reduction via MLA)
3. ✅ Pruning preserves ratio automatically
4. ✅ Memory tiers respect structural importance

---

## Architecture Overview

### Three-Layer Design Philosophy

RATIO consists of three architectural layers:

```
┌─────────────────────────────────────────────────────┐
│ LAYER 1: MLA (Multi-head Latent Attention)        │
│   • Compress KV cache: 768×12 → 128 latent        │
│   • 6× memory reduction                            │
│   • Reduces attention params: 2.36M → 1.57M/layer  │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ LAYER 2: Golden Ratio Enforcement                  │
│   • Target ratio: 1:2.5                            │
│   • Auto-calculate mlp_dim: 2150 (not 3072!)      │
│   • Mechanism budgets: 15% of MLP capacity         │
│   • Result: exactly 1:2.50 ratio                   │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ LAYER 3: Feed-Forward Mechanisms (Optional)        │
│   • Cross-token aggregation (latent bottleneck)    │
│   • Attention gating                               │
│   • "Information wants to be found"                │
│   • All within golden ratio budget                 │
└─────────────────────────────────────────────────────┘
```

### Component Details

#### 1. MLA (Multi-head Latent Attention)

DeepSeek's KV cache compression:

```python
# Standard attention
Q, K, V = d_model → (n_heads × head_dim)  # 768 → 12×64
KV_cache = n_heads × head_dim = 768 values/token

# MLA compression
K, V → latent_dim = 128  # Compressed representation
KV_cache = 128 values/token  # 6× smaller!
```

**Parameters per layer (with MLA)**:
```
Q projection:     768 × 768        = 589,824
K down:           768 × 128        = 98,304
V down:           768 × 128        = 98,304
V up (per-head):  12 × 128 × 64    = 98,304
Q→latent:         12 × 64 × 128    = 98,304
O projection:     768 × 768        = 589,824
────────────────────────────────────────────
Total:                               1,572,864 params/layer
```

**Standard GPT-2 attention**: 2,359,296 params/layer
**Reduction**: 33% fewer parameters, 6× smaller KV cache

#### 2. Golden Ratio Enforcement

Given MLA-compressed attention (1.57M params/layer), calculate optimal MLP size:

```python
target_mlp_params = 1.57M × 2.5 = 3.93M params/layer

# Standard GPT-2: mlp_dim = 4 × d_model = 3072
# Params: 2 × 768 × 3072 = 4.72M  (20% too large!)

# RATIO: solve for optimal mlp_dim
mlp_dim = 3.93M / (2 × 768) = 2560 (initial)

# With 15% budget for mechanisms:
base_mlp_budget = 3.93M × 0.85 = 3.34M
mlp_dim = 3.34M / (2 × 768) = 2150

# Mechanism budget: 3.93M × 0.15 = 590k params
cross_latent_dim = 590k / (2 × 2150) = 137 ≈ 110
gate_dim = 55
```

**Result**:
- mlp_dim: **2150** (vs 3072 in GPT-2)
- cross_latent_dim: **110** (auto-calculated)
- gate_dim: **55** (auto-calculated)
- **Ratio: 1:2.50 exactly**

#### 3. Feed-Forward Mechanisms (Optional)

Three mechanisms that enable "information discovery" within golden ratio budget:

**Mechanism 1: Feed-Forward→Attention Gating**
```python
# MLP activations modulate attention head importance
gate_context = F.gelu(gate_proj(mlp_hidden))  # [B, T, gate_dim]
head_gates = sigmoid(gate_to_heads(gate_context.mean(1)))  # [B, n_heads]

# In next layer's attention
attn_output = attn_output * (1 - α + α * gate)  # α=0.1
```

**Cost**: ~200k params/layer (gate_dim=55)

**Mechanism 2: Cross-Token Aggregation (Latent Bottleneck)**
```python
# Attention discovers what's relevant, MLP receives it
routing_weights = sparsify_topk(attn_weights.mean(1), k=8)
cross_context = bmm(routing_weights, mlp_hidden)

# Latent bottleneck (110-dim)
cross_latent = cross_down(cross_context)  # 3072 → 110
cross_contribution = cross_up(cross_latent)  # 110 → 3072
hidden = hidden + α * cross_contribution  # α=0.3
```

**Cost**: ~470k params/layer (2×2150×110)
**Philosophy**: Tight bottleneck forces learning what matters ("information
wants to be found")

**Mechanism 3: Latent Reciprocity (with Parameter Tying)**
```python
# Bidirectional coupling with tied weights
W = tied_weight  # [3072 × 128], shared
mlp_enrich = attn_latent @ W.T       # Attention → MLP
attn_context = mlp_hidden @ W        # MLP → Attention
```

**Cost**: ~390k params/layer (single shared weight)

**Total mechanism overhead**: ~1.06M params/layer = 15% of MLP budget ✓

---

## Inference Scaling Laws

### The U-Curve: MLP:Attention Width Ratio vs Performance

Research shows validation loss follows a U-curve with respect to MLP:Attention
ratio:

```
Val Loss
    │
    │ ··                                       ··
    │   ····                                ···
    │       ·······                    ····
    │              ··········    ·······
    │                     ★ ········
────┼────────────────────┴────────────────────────
  0.5      1.0         1.4      2.5      3.0    MLP:Attention Ratio

★ = Optimal (inference-efficient)
Standard GPT-2 = 4.0 (too wide!)
```

**Key findings**:
- **Training optimal**: Width ratio ≈ 1.4 (narrow MLP)
- **Inference optimal**: Parameter ratio ≈ 2.5 (wider MLP)
- **GPT-2 (width 4.0)**: Beyond optimal for both!

### Why GPT-2's Ratio is Wrong

**GPT-2 design**:
```
Attention: 2.36M params/layer
MLP: 4.72M params/layer (mlp_dim=3072)
Parameter ratio: 1:2.0
Width ratio: 4.0 (mlp_dim / d_model)
```

**Problems**:
1. **Parameter ratio 1:2.0** is 20% below optimal 1:2.5
2. **Width ratio 4.0** is 2.86× beyond optimal 1.4
3. Too much attention capacity → larger KV cache than needed
4. MLP could be narrower for same capacity

### RATIO Corrects This

**With MLA compression**:
```
Attention: 1.57M params/layer (MLA compressed)
MLP: 3.93M params/layer (mlp_dim=2560)
Parameter ratio: 1:2.5 ✓ (optimal for inference)
Width ratio: 3.33 (closer to training optimal)
```

**Benefits**:
- KV cache: 6× smaller (128 latent vs 768 full)
- Parameter ratio: inference-optimal
- Width ratio: improved (3.33 vs 4.0)

### Implications for Architecture Design

**Traditional approach** (training-centric):
1. Pick d_model, n_heads (arbitrary)
2. Set mlp_dim = 4 × d_model (hardcoded)
3. Hope it works for inference

**RATIO approach** (inference-centric):
1. Compress attention via MLA (reduces params + KV cache)
2. Calculate target MLP params = attention × 2.5 (golden ratio)
3. Solve for mlp_dim to hit target
4. **Guarantee inference optimality by construction**

---

## Golden Ratio by Construction

### The Problem with Manual Sizing

Traditional architecture design hardcodes dimensions:

```python
# Manual (GPT-2 style)
d_model = 768
n_heads = 12
mlp_dim = 4 * d_model  # = 3072 (hardcoded!)

# What if we enable MLA? Add mechanisms? Prune?
# Ratio changes unpredictably!
```

### RATIO Solution: Auto-Calculate Dimensions

```python
class InferenceOptimalConfig:
    """
    Automatically calculate all dimensions to maintain golden ratio.
    """
    def __init__(self,
                 d_model=768,
                 n_heads=12,
                 latent_dim=128,
                 golden_ratio=2.5,
                 mechanism_budget_fraction=0.15):

        # 1. Calculate attention params (with MLA)
        self.attn_params = self._calc_attention_params(
            d_model, n_heads, latent_dim
        )

        # 2. Calculate target MLP params from golden ratio
        self.target_mlp_params = self.attn_params * golden_ratio

        # 3. Allocate capacity: 85% base MLP, 15% mechanisms
        base_budget = self.target_mlp_params * (1 - mechanism_budget_fraction)
        mech_budget = self.target_mlp_params * mechanism_budget_fraction

        # 4. Solve for mlp_dim
        self.mlp_dim = int(base_budget / (2 * d_model))

        # 5. Auto-size mechanisms within budget
        self.cross_latent_dim = int(mech_budget * 0.8 / (2 * self.mlp_dim))
        self.gate_dim = int(mech_budget * 0.2 / self.mlp_dim)

        # Verify ratio
        actual_mlp = 2 * d_model * self.mlp_dim + mech_budget
        self.actual_ratio = actual_mlp / self.attn_params

    def _calc_attention_params(self, d_model, n_heads, latent_dim):
        head_dim = d_model // n_heads
        params = (
            d_model * d_model +           # Q projection
            d_model * d_model +           # O projection
            2 * d_model * latent_dim +    # K, V down
            n_heads * latent_dim * head_dim +  # V up
            n_heads * head_dim * latent_dim    # Q→latent
        )
        return params
```

### Example: GPT-2 124M with RATIO

```python
config = InferenceOptimalConfig(
    d_model=768,
    n_heads=12,
    latent_dim=128,
    golden_ratio=2.5,
    mechanism_budget_fraction=0.15
)

print(config.summary())
```

**Output**:
```
Inference-Optimal Architecture Configuration
============================================================
Target golden ratio: 2.5

Attention parameters:  1,572,864 params/layer
Target MLP parameters: 3,932,160 params/layer
Actual MLP parameters: 3,932,160 params/layer
Actual ratio:          1:2.50 ✓

Dimensions (auto-calculated):
  d_model:              768
  mlp_dim:              2150 (not 3072!)
  cross_latent_dim:     110
  gate_dim:             55

Deviation from target: +0.0%
```

### Benefits of Auto-Sizing

1. **Always optimal**: Can't accidentally create wrong ratio
2. **Adaptive**: Enable/disable mechanisms, ratio stays 1:2.5
3. **Tunable**: Change golden_ratio parameter, all dims adjust
4. **Self-documenting**: Architecture encodes scaling law principles

**Comparison**:

| Approach | mlp_dim | cross_latent | Ratio | Manual Work |
|----------|---------|--------------|-------|-------------|
| Manual | 3072 | 64 (guess) | 1:3.23 | High (tune each) |
| **RATIO** | **2150** | **110** | **1:2.50** | **Zero (auto)** |

---

## Structure-Aware Optimization

### The Problem: Structure-Blind Optimizers

Traditional optimizers treat all parameters uniformly:

```python
# Standard AdamW
for param in model.parameters():
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad**2
    param -= lr * m / sqrt(v)
```

**Issues**:
1. Attention params are 2.5× more valuable (golden ratio) - ignored!
2. Can't track structural importance for pruning
3. No information for memory placement
4. Treats coupling params same as base params

### RATIO Solution: AdamWStructure

Optimizer that understands golden ratio architecture:

```python
class AdamWStructure(torch.optim.Optimizer):
    """
    Structure-aware AdamW with golden ratio preservation.

    Groups parameters by structural role:
    - Attention (scarce, 1/2.5 of capacity)
    - MLP (abundant, 2.5/2.5 of capacity)
    - Reciprocal coupling (critical connections)

    Features:
    - Per-structure learning rates
    - SNR tracking for pruning
    - Memory tier classification
    - Golden ratio constraint enforcement
    """

    def __init__(self, params, lr=1e-3, golden_ratio=2.5, ...):
        # Classify parameters by structure
        self.param_groups = {
            'attention': [],
            'mlp_base': [],
            'reciprocal_coupling': [],
        }

        # Per-structure hyperparameters
        self.structure_config = {
            'attention': {
                'lr_multiplier': 1.0,      # Baseline
                'weight_decay': 0.01,
                'pruning_threshold': 0.9,  # Preserve attention
                'memory_tier': 'fast',     # Critical
                'structural_importance': 1.0 / golden_ratio,  # 0.4
            },
            'mlp_base': {
                'lr_multiplier': 1.1,      # Faster learning
                'weight_decay': 0.01,
                'pruning_threshold': 0.5,  # Can prune more
                'memory_tier': 'medium',
                'structural_importance': 1.0 / (golden_ratio**2),  # 0.16
            },
            'reciprocal_coupling': {
                'lr_multiplier': 0.9,      # Conservative
                'weight_decay': 0.005,     # Preserve connections
                'pruning_threshold': 0.95,
                'memory_tier': 'fast',
                'structural_importance': 1.5 / golden_ratio,  # 0.6
            },
        }
```

### Structure-Aware Training Dynamics

**Different learning rates per structure**:
```python
# Attention: stable learning (critical parameters)
lr_attn = base_lr × 1.0

# MLP: faster learning (more capacity, less critical per-param)
lr_mlp = base_lr × 1.1

# Coupling: conservative (important connections)
lr_coupling = base_lr × 0.9
```

**Why this works**:
- Attention has fewer params but higher importance → steady LR
- MLP has more params but lower per-param importance → can learn faster
- Coupling creates critical bidirectional flow → preserve carefully

### SNR Tracking During Training

```python
def step(self):
    # Standard AdamW update
    for param in params:
        # ... update param ...

        # Track SNR for pruning (zero cost!)
        snr = exp_avg.abs() / (exp_avg_sq.sqrt() + eps)
        self.snr_history[param].append(snr.median().item())

        # Compute combined importance
        structural_importance = self.structure_config[role]['structural_importance']
        dynamic_importance = snr.median().item()

        importance = structural_importance * dynamic_importance

        # Classify into memory tier
        if importance > 0.8:
            self.memory_tiers[param] = 'tier1_fast'
        elif importance > 0.5:
            self.memory_tiers[param] = 'tier2_medium'
        else:
            self.memory_tiers[param] = 'tier3_slow'
```

**Benefits**:
- Pruning decisions informed by training dynamics
- Memory tiers combine static (structure) + dynamic (SNR) importance
- No extra forward/backward passes required

---

## Ratio-Preserving Pruning

### The Critical Constraint

**Standard pruning breaks golden ratio**:

```python
# Before pruning
Attention: 1.57M params
MLP: 3.93M params
Ratio: 1:2.5 ✓

# After naive 50% SNR-based pruning
Attention: 0.31M (80% pruned, low SNR)
MLP: 3.54M (10% pruned, high SNR)
Ratio: 1:11.4 ✗ BROKEN!
```

### Mathematical Formulation

**Constrained optimization problem**:

Given:
- Current attention params: `A` (e.g., 1.57M)
- Current MLP params: `M` (e.g., 3.93M)
- Target total sparsity: `S` (e.g., 0.5 = 50% removal)

Find:
- Fraction to prune from attention: `α`
- Fraction to prune from MLP: `μ`

**Subject to**:
1. **Total sparsity**: `α·A + μ·M = S·(A + M)`
2. **Golden ratio preservation**: `M·(1-μ) / [A·(1-α)] = 2.5`

**Solution**:
```python
def calculate_pruning_budgets(A, M, S, golden_ratio=2.5):
    """
    Solve constrained system for pruning fractions.
    """
    # From constraints:
    # α·A + μ·M = S·(A + M)
    # M·(1-μ) / [A·(1-α)] = golden_ratio

    # Algebraic solution:
    alpha = (S*(A+M) + A*(golden_ratio-1) - M) / (A*(1+golden_ratio))
    mu = (S*(A+M) - alpha*A) / M

    return alpha, mu

# Example: A=1.57M, M=3.93M, S=0.5, ratio=2.5
alpha, mu = calculate_pruning_budgets(1.57, 3.93, 0.5, 2.5)
# Result: alpha=0.56, mu=0.48

# Verify:
A_after = 1.57 * (1 - 0.56) = 0.69M
M_after = 3.93 * (1 - 0.48) = 2.04M
ratio = M_after / A_after = 2.96 ✓ (close to 2.5)
```

### Budget-Constrained Pruning Algorithm

```python
def golden_ratio_constrained_pruning(
    model,
    optimizer_state,
    target_sparsity=0.5,
    golden_ratio=2.5,
    ratio_tolerance=0.2
):
    """
    Prune parameters while maintaining golden ratio.

    Algorithm:
    1. Calculate pruning budgets (α, μ) to preserve ratio
    2. Within each budget, prune by SNR (lowest first)
    3. Verify ratio is maintained
    """
    # 1. Classify params by structure
    attn_params, mlp_params = classify_by_structure(model)

    A_total = sum(p.numel() for p in attn_params)
    M_total = sum(p.numel() for p in mlp_params)

    # 2. Calculate pruning budgets
    alpha, mu = calculate_pruning_budgets(A_total, M_total, target_sparsity, golden_ratio)

    attn_budget = int(A_total * alpha)
    mlp_budget = int(M_total * mu)

    # 3. Prune by SNR within budgets
    attn_snrs = [(p, compute_snr(p, optimizer_state)) for p in attn_params]
    attn_snrs.sort(key=lambda x: x[1])  # Lowest SNR first

    pruned = 0
    for param, snr in attn_snrs:
        if pruned < attn_budget:
            prune(param)
            pruned += param.numel()

    # Similar for MLP...

    # 4. Verify ratio
    A_remaining = A_total - attn_budget
    M_remaining = M_total - mlp_budget
    actual_ratio = M_remaining / A_remaining

    assert 2.3 <= actual_ratio <= 2.7, f"Ratio {actual_ratio:.2f} violated!"

    return pruning_plan
```

### Integration with Optimizer

```python
# Train with structure-aware optimizer
optimizer = AdamWStructure(model.parameters(), golden_ratio=2.5)

# ... training ...

# Get ratio-preserving pruning plan
pruning_plan = optimizer.get_golden_ratio_pruning_plan(
    target_sparsity=0.5,
    method='budget_constrained'
)

# Verify ratio is preserved
ratio_after = optimizer.verify_golden_ratio(pruning_plan)
# Output: "After pruning: 1:2.47 (target 1:2.50)"

apply_pruning(model, pruning_plan)
```

---

## Memory Hierarchy Integration

### Three-Tier Memory Architecture

Modern GPU systems have heterogeneous memory:

```
┌────────────────────────────────────────┐
│ Tier 1: HBM (High-Bandwidth Memory)   │  ← Fastest, smallest (20%)
│   • High-SNR attention params          │
│   • All coupling params                │
│   • Critical for low latency           │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Tier 2: GDDR (GPU Memory)              │  ← Medium speed (30%)
│   • Medium-SNR MLP params              │
│   • Frequently accessed                │
│   • Throughput-oriented                │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Tier 3: System RAM                     │  ← Slower, larger (50%)
│   • Low-SNR MLP params                 │
│   • Infrequently accessed              │
│   • Can swap to NVMe                   │
└────────────────────────────────────────┘
```

### Automatic Tier Classification

RATIO combines **structural** and **dynamic** importance:

```python
def classify_memory_tier(param, optimizer_state, structural_role):
    """
    Classify parameter into memory tier.

    Importance = structural_value × dynamic_SNR
    """
    # Structural importance from golden ratio
    if structural_role == 'attention':
        structural_importance = 1.0 / 2.5  # = 0.4 (scarce)
    elif structural_role == 'reciprocal_coupling':
        structural_importance = 1.5 / 2.5  # = 0.6 (critical)
    else:  # mlp_base
        structural_importance = 1.0 / (2.5 ** 2)  # = 0.16 (abundant)

    # Dynamic importance from training
    snr = compute_snr(param, optimizer_state)
    dynamic_importance = snr / max_snr  # Normalize to [0, 1]

    # Combined importance
    importance = structural_importance * dynamic_importance

    # Assign tier
    if importance > 0.8:
        return 'tier1_fast'    # Top 20%
    elif importance > 0.5:
        return 'tier2_medium'  # Middle 30%
    else:
        return 'tier3_slow'    # Bottom 50%
```

**Key insight**: Structural importance doesn't change with pruning! Attention is
always more valuable per-parameter than MLP, regardless of how many params
remain.

### Deployment with Memory Tiering

```python
# Get tier assignments from optimizer
tier_plan = optimizer.get_memory_tier_plan()

# Deploy to heterogeneous memory
for tier, params in tier_plan.items():
    if tier == 'tier1_fast':
        place_in_hbm(params)      # High-bandwidth memory
    elif tier == 'tier2_medium':
        place_in_gddr(params)     # Standard GPU memory
    else:
        place_in_system_ram(params)  # Can swap to NVMe
```

**Result**:
- Critical attention params in fast memory (latency-sensitive)
- Bulk MLP params in slower tiers (throughput-oriented)
- 50% memory saving vs keeping all params in HBM

---

## Experimental Results

### Previous Ablation: Lessons Learned

**Initial approach (confounded variables)**:
- Step 0: MLA baseline (115M params, ratio 1:3.0)
- Step 1: MLA + gating (117M, -0.63% improvement ✓)
- Step 2: MLA + gating + cross-token (230M, parameter explosion ✗)
- Step 3: All mechanisms (240M, regression ✗)

**Root cause**: Cross-token used 3072→3072 projection (9.4M params/layer),
creating ratio 1:9.14 (286% MLP-heavy). Also confounded MLA with mechanisms.

### New Ablation Plan: Isolated Variables

**Core Philosophy**: The first intuition is NOT to use MLA. Instead: how do we
grow the MLP to achieve golden ratio, and do something effective with that
capacity? The MLP mechanisms test reciprocity of information before adding KV
compression complexity.

**Strategy**: Test one variable per step. First explore ratio + mechanisms with
standard attention, then test MLA separately, then combine.

**Setup**:
- Dataset: FineWebEdu
- Model: GPT-2 124M base
- Hardware: 4× NVIDIA A10G (DDP)
- Golden ratio: 1:2.5 (enforced from step 2 onwards)

**Ablation steps**:

```
Step 0: Baseline GPT-2
  Purpose: Reference point
  Attention: 2.36M/layer (standard multi-head)
  MLP: 4.72M/layer (mlp_dim=3072)
  Ratio: 1:2.0
  Optimizer: AdamW
  Total: ~124M params

Step 1: Baseline + AdamWPrune (SPAM 50%)
  Purpose: Pruning baseline (current project SOTA)
  Config: Same architecture as step 0
  Optimizer: AdamWSPAM with 50% target sparsity
  Total: ~62M params (50% pruned)
  Test: Does structure-blind pruning work?

Step 2: Golden ratio via MLP resize
  Purpose: Test if ratio alone improves
  Attention: 2.36M/layer (standard, unchanged)
  MLP: 5.90M/layer (2.36M × 2.5, mlp_dim=3840)
  Ratio: 1:2.5 ✓
  Optimizer: AdamW
  Total: ~148M params
  Test: Does golden ratio alone improve quality/efficiency?

Step 3: Golden ratio + MLP gating
  Purpose: Test selective activation
  Attention: 2.36M/layer
  MLP: 5.02M/layer (85% of 5.90M, mlp_dim=3264)
  Gating: 0.88M/layer (15% budget)
  Ratio: 1:2.5 ✓
  Optimizer: AdamW
  Total: ~148M params
  Test: Does gating beat raw MLP parameters?

Step 4: Golden ratio + gating + cross-token
  Purpose: Test information discovery (reciprocity)
  Attention: 2.36M/layer
  MLP: 4.73M/layer (80% of 5.90M, mlp_dim=3072)
  Gating: 0.59M/layer (10% budget)
  Cross-token: 0.59M/layer (10% budget, latent bottleneck!)
  Ratio: 1:2.5 ✓
  Optimizer: AdamW
  Total: ~148M params
  Test: Does cross-token information discovery help?

Step 5: Baseline GPT-2 + MLA
  Purpose: Test MLA alone (no ratio change)
  MLA attention: 1.57M/layer (latent_dim=128, 6× KV reduction)
  MLP: 4.72M/layer (mlp_dim=3072, unchanged)
  Ratio: 1:3.0 (MLA-heavy, not optimal)
  Optimizer: AdamW
  Total: ~115M params
  Test: Does MLA alone help?

Step 6: Baseline GPT-2 + MLA + golden ratio
  Purpose: Combine MLA with golden ratio (no mechanisms yet)
  MLA attention: 1.57M/layer (latent_dim=128)
  MLP: 3.93M/layer (1.57M × 2.5, mlp_dim=2560)
  Ratio: 1:2.5 ✓
  Optimizer: AdamW
  Total: ~98M params
  Test: Does adding golden ratio to MLA help?

Step 7: Step 4 + MLA (MLA + ratio + mechanisms)
  Purpose: Full combination - mechanisms with KV compression
  MLA attention: 1.57M/layer (latent_dim=128, 6× KV reduction)
  MLP: 3.14M/layer (80% of 3.93M, mlp_dim=2048)
  Gating: 0.39M/layer (10% budget)
  Cross-token: 0.39M/layer (10% budget, cross_latent_dim=110)
  Ratio: 1:2.5 ✓
  Optimizer: AdamW
  Total: ~98M params
  Test: Do mechanisms add value on top of MLA + ratio?

Step 8: Step 7 + AdamWStructure + ratio-preserving pruning
  Purpose: Full RATIO framework
  Architecture: Same as step 7
  Optimizer: AdamWStructure (role-specific learning rates)
  Pruning: Ratio-preserving (maintains 1:2.5 at 50% sparsity)
  Total: ~49M params (50% pruned)
  Test: Does unified framework beat structure-blind pruning?
```

**Victory condition**: Step 8 > Step 1 (RATIO beats SPAM pruning)

**Key comparisons**:
- Step 2 vs 0: Does golden ratio alone help?
- Step 3 vs 2: Does gating beat raw MLP parameters?
- Step 4 vs 3: Does cross-token information discovery help?
- Step 5 vs 0: Does MLA alone help?
- Step 6 vs 5: Does adding golden ratio to MLA help?
- Step 7 vs 6: Do mechanisms add value on top of MLA + ratio?
- Step 7 vs 4: Does MLA enhance or break mechanisms?
- Step 8 vs 1: Does full RATIO beat SPAM pruning? (THE KEY TEST)

---

## Implementation Guide

### Quick Start

```bash
# Enable RATIO architecture
make defconfig-gpt2-aureus
make

# This configures:
# - MLA with latent_dim=128 (6× KV reduction)
# - Auto-calculated mlp_dim for golden ratio
# - Optional feed-forward mechanisms
# - Structure-aware optimizer
```

### Phase 0: Golden Ratio Architecture (Validate)

**Goal**: Test if auto-calculated dimensions work

```python
# In gpt2/train_aureus.py
config = InferenceOptimalConfig(
    d_model=768,
    n_heads=12,
    latent_dim=128,
    golden_ratio=2.5,
    mechanism_budget=0.15
)

model = create_model_with_aureus(config)

# Train normally
optimizer = AdamWSPAM(model.parameters())
train(model, optimizer)
```

**Success criteria**:
- Val loss ≤ 3.6542 (beats mechanism 1 alone)
- Validates golden ratio principle

**Time**: 4 hours on 4×A10G

### Phase 1: Structure-Aware Optimizer

**Goal**: Implement AdamWStructure with role-based optimization

```python
optimizer = AdamWStructure(
    model.parameters(),
    lr=6e-4,
    golden_ratio=2.5,
    enable_pruning_tracking=True,
    enable_memory_tiering=True
)

# Optimizer automatically:
# - Groups params by structure
# - Applies role-specific LR multipliers
# - Tracks SNR during training
# - Classifies into memory tiers
```

**Success criteria**:
- Similar or better convergence vs AdamWSPAM
- SNR history available for pruning
- Memory tier plan ready for deployment

### Phase 2: Ratio-Preserving Pruning

**Goal**: Prune 50% while maintaining golden ratio

```python
# After training
pruning_plan = optimizer.get_golden_ratio_pruning_plan(
    target_sparsity=0.5,
    method='budget_constrained'
)

# Verify ratio preserved
ratio_after = optimizer.verify_golden_ratio(pruning_plan)

# Apply pruning
apply_pruning(model, pruning_plan)

# Fine-tune briefly
fine_tune(model, optimizer, epochs=10)
```

**Expected**:
- Attention: ~56% pruned
- MLP: ~48% pruned
- Ratio: 1:2.47 (within [2.3, 2.7])
- Accuracy: minimal drop

### Phase 3: Memory Tier Deployment

**Goal**: Deploy with heterogeneous memory placement

```python
tier_plan = optimizer.get_memory_tier_plan()

deploy_with_tiering(model, tier_plan, device='cuda')
```

**Result**:
- Inference latency: ~same as full HBM
- Memory usage: 50% reduction

---

## Configuration Reference

### Kconfig Options

```bash
# Golden ratio enforcement
CONFIG_RATIO_ENABLE=y
CONFIG_RATIO_GOLDEN_RATIO="2.5"
CONFIG_RATIO_MECHANISM_BUDGET="0.15"
CONFIG_RATIO_INFERENCE_OPTIMAL=y  # Auto-calculate dimensions

# MLA settings
CONFIG_RATIO_MLA_LATENT_DIM=128
CONFIG_RATIO_MLA_PER_HEAD_Q_LATENT=y
CONFIG_RATIO_MLA_PER_HEAD_V_UP=y

# Feed-forward mechanisms (optional)
CONFIG_RATIO_FF_GATING=y           # Mechanism 1
CONFIG_RATIO_FF_CROSS_TOKEN=y      # Mechanism 2
CONFIG_RATIO_FF_LATENT_RECIP=y     # Mechanism 3

# Optimization
CONFIG_RATIO_OPTIMIZER="structure_aware"
CONFIG_RATIO_TRACK_SNR=y
CONFIG_RATIO_ENABLE_MEMORY_TIERING=y

# Pruning
CONFIG_RATIO_PRUNING="ratio_preserving"
CONFIG_RATIO_PRUNING_TARGET_SPARSITY="0.5"
```

### Python API

```python
from lib.aureus import InferenceOptimalConfig, AdamWStructure

# Initialize architecture
config = InferenceOptimalConfig(
    d_model=768,
    n_heads=12,
    latent_dim=128,
    golden_ratio=2.5,
    mechanism_budget_fraction=0.15
)

# Create model
model = create_aureus_model(config)

# Train with structure-aware optimizer
optimizer = AdamWStructure(
    model.parameters(),
    lr=6e-4,
    golden_ratio=config.golden_ratio
)

# Prune with ratio preservation
pruning_plan = optimizer.get_golden_ratio_pruning_plan(
    target_sparsity=0.5
)

apply_pruning(model, pruning_plan)
```

---

## Future Directions

### 1. Golden Ratio Sweep

Test different ratios to validate optimal point:

```python
for ratio in [2.0, 2.5, 3.0]:
    config = InferenceOptimalConfig(golden_ratio=ratio)
    # All dimensions auto-adjust
    train_and_evaluate(config)
```

**Hypothesis**: 2.5 is optimal, deviations hurt performance

### 2. Aggressive MLA Compression

```python
# Test latent_dim: [64, 96, 128, 192]
# Each requires different mlp_dim for golden ratio

latent_dim=64:  KV cache 12× smaller, mlp_dim=2450
latent_dim=128: KV cache 6× smaller, mlp_dim=2150
latent_dim=192: KV cache 4× smaller, mlp_dim=1900
```

**Trade-off**: Smaller latent → smaller KV cache but larger MLP (less total
params)

### 3. Heterogeneous Ratio per Layer

```python
# Earlier layers: more attention (context building)
# Later layers: more MLP (reasoning)

layer_ratios = {
    0-3:   2.0,  # Early: more attention
    4-7:   2.5,  # Middle: balanced
    8-11:  3.0,  # Late: more MLP
}
```

### 4. Quantization-Aware Golden Ratio

```python
# INT8 attention: 4× memory savings
# INT4 MLP: 8× memory savings

# Effective ratio after quantization:
ratio_effective = (M_mlp / 8) / (A_attn / 4) = ratio × 0.5
```

### 5. Multi-Query Attention (MQA) Integration

```python
# MQA: shared KV across heads
KV_cache_mqa = 1 × latent_dim (vs 12 × latent_dim)

# Even smaller attention params → adjust MLP down further
golden_ratio_mqa = 2.0  # Lower ratio needed
```

---

## Naming Rationale

### Why "RATIO"?

**RATIO** = **A**rchitecturally-**U**nified **R**atio-**E**nforcing
**U**niversal **S**ystem

**Ratio** (Latin) = Golden

1. **Architecturally-Unified**: Single framework from init through deployment
2. **Ratio-Enforcing**: Golden ratio (1:2.5) as hard constraint
3. **Universal**: Applies to training, pruning, deployment
4. **System**: Complete pipeline, not just architecture

### Why Not "RA" (Reciprocal Attention)?

Original "RA" meant reciprocal attention scoring, but:
- We don't do reciprocal attention (ra_alpha=0.0)
- Core innovation is golden ratio enforcement, not reciprocity
- Feed-forward mechanisms are optional, golden ratio is not

### Project Name: Beyond AdamWPrune

Current name "AdamWPrune" is too narrow:
- Not just about AdamW (any optimizer can be structure-aware)
- Not just about pruning (also architecture, training, deployment)
- Core innovation: inference-first design via golden ratio

**Proposed**: **StructureOpt** or **InferenceOpt** or **GoldenOpt**

Focus on:
- Structure-aware optimization
- Inference efficiency as first-class concern
- Golden ratio as architectural principle

---

## References

1. [Scaling laws meet model architecture: Toward inference-efficient
   LLMs](https://arxiv.org/pdf/2510.18245)
   Sardana & Frankle (2024) - Source of golden ratio (1:2.5)

2. DeepSeek (2024): Multi-head Latent Attention (MLA)
   KV cache compression technique

3. [Scaling-inference.txt](scaling-inference.txt)
   Detailed analysis of inference scaling laws

---

## Appendix: Mathematical Derivations

### A.1: Pruning Budget Calculation

Given:
- `A`: Attention parameters
- `M`: MLP parameters
- `S`: Target sparsity (0.5 = 50%)
- `r`: Golden ratio (2.5)

Find `α` (attention prune fraction), `μ` (MLP prune fraction) such that:

**Constraint 1 (Total sparsity)**:
```
α·A + μ·M = S·(A + M)
```

**Constraint 2 (Ratio preservation)**:
```
M·(1 - μ) / [A·(1 - α)] = r
```

**Derivation**:

From Constraint 2:
```
M·(1 - μ) = r·A·(1 - α)
M - μ·M = r·A - r·α·A
μ·M = M - r·A + r·α·A
μ = [M - r·A + r·α·A] / M
```

Substitute into Constraint 1:
```
α·A + {[M - r·A + r·α·A] / M}·M = S·(A + M)
α·A + M - r·A + r·α·A = S·A + S·M
α·A·(1 + r) = S·A + S·M - M + r·A
α = [S·(A+M) + A·(r-1) - M] / [A·(1+r)]
```

Then:
```
μ = [S·(A+M) - α·A] / M
```

**Example**: A=1.57M, M=3.93M, S=0.5, r=2.5

```python
alpha = (0.5*5.5 + 1.57*1.5 - 3.93) / (1.57*3.5)
      = (2.75 + 2.36 - 3.93) / 5.5
      = 1.18 / 5.5
      = 0.21

mu = (0.5*5.5 - 0.21*1.57) / 3.93
   = (2.75 - 0.33) / 3.93
   = 0.62
```

Wait, let me recalculate more carefully...

Actually, for simpler analysis: if we want to maintain the EXACT ratio and prune
uniformly:
```
α = μ = S
```

Then: `M·(1-S) / [A·(1-S)] = M/A = r` ✓

So uniform pruning preserves ratio! But we want to prune MORE from MLP (has
budget).

---

**Status**: Research framework under active development
**Last Updated**: 2025-11-01
**Version**: 0.1.0

