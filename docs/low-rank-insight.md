# Low-Rank MLP Context: No Bloat!

## The Problem

When giving MLP access to attention's cross-token context, naive approach adds bloat:

```python
# WRONG: Full-rank projection (BLOAT!)
ctx_proj = nn.Linear(E, mult*E, bias=False)  # 768 × 3072 = 2.36M params
ctx_h = ctx_proj(ctx_summary)  # [B, T, mult*E]
```

**Cost per layer**: 2.36M params
**Total (12 layers)**: +28.3M params (33% bloat!)

This is stupid. We're just letting MLP see what attention computed - why does that need 2.36M parameters?

## The Solution: Low-Rank Factorization

MLP doesn't need full-rank projection of attention output. It needs a **compressed summary**.

```python
# RIGHT: Low-rank bottleneck (EFFICIENT!)
R = 128  # Bottleneck dimension

ctx_down = nn.Linear(E, R, bias=False)     # 768 × 128 = 98K params
ctx_up = nn.Linear(R, mult*E, bias=False)  # 128 × 3072 = 393K params

# Two-stage projection: E → R → mult*E
ctx_h = ctx_down(ctx_summary)  # [B, T, R] - compress
ctx_h = ctx_up(ctx_h)           # [B, T, mult*E] - expand
```

**Cost per layer**: 98K + 393K = 491K params
**Total (12 layers)**: +5.9M params (7% increase)

### Savings: **5× parameter reduction!**

## Why This Works

Attention output is already a **aggregated** representation:
```python
attn_out = softmax(Q @ K^T) @ V  # Already compressed cross-token info
```

MLP only needs the "gist" of what attention computed, not every detail. Low-rank captures this:

- **Full rank (2.36M)**: Can represent any E → mult*E transformation
- **Low-rank (491K)**: Can represent "attention computed X, so MLP should shift toward Y"

The second is sufficient and **5× cheaper**.

## Conductor Mode (Optional)

Even smarter: only use context when MLP-heavy (route_gate < 0.5):

```python
use_context = route_gate < 0.5  # Only when MLP needs to compete

if use_context:
    ctx_h = ctx_down(ctx_summary)
    ctx_h = ctx_up(ctx_h)
    h = (1-alpha)*h + alpha*ctx_h  # Blend
```

**Rationale**:
- When `g ≈ 0.9` (attention-heavy): MLP doesn't need context help
- When `g ≈ 0.3` (MLP-heavy): MLP needs context to compete with attention

**Benefit**: Gradual transition. As route gate anneals, MLP "learns" to use context.

## Parameter Count Comparison

### Baseline GPT-2

```
Attention: 4*E² = 4 × 768² = 2.36M params/layer
MLP: 2*E*(mult*E) = 2 × 768 × 3072 = 4.72M params/layer
Total: 7.08M params/layer × 12 = 85M total
```

### Lens-Gated (Full-Rank Context) - BLOATED

```
Attention: 2.36M/layer (unchanged)
MLP: 4.72M/layer (unchanged)
+ ctx_proj: 2.36M/layer (BLOAT!)
+ Reciprocity/Disc/Gates: ~10K/layer (negligible)
Total: 9.44M/layer × 12 = 113.3M (+33% BLOAT)
```

### Lens-Gated (Low-Rank Context) - EFFICIENT

```
Attention: 2.36M/layer (unchanged)
MLP: 4.72M/layer (unchanged)
+ ctx_down: 98K/layer (low-rank)
+ ctx_up: 393K/layer (low-rank)
+ Reciprocity/Disc/Gates: ~10K/layer (negligible)
Total: 7.57M/layer × 12 = 90.9M (+7% efficient increase)
```

## Visual Comparison

```
Full-Rank Path (BLOAT):
attention_out [768] ──[2.36M params]──> mlp_hidden [3072]
                    └─ Full rank matrix

Low-Rank Path (EFFICIENT):
attention_out [768] ──[98K]──> bottleneck [128] ──[393K]──> mlp_hidden [3072]
                    └─ Compress  └─ Decompress

Captures "gist" of attention output, 5× cheaper!
```

## Configuration

```python
# Enable low-rank context (default)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    mlp_use_ctx_summary=True,
    mlp_ctx_rank=128,  # Bottleneck (higher = more capacity, more params)
    mlp_ctx_conductor=False,  # Always use context
)

# Enable conductor mode (context only when MLP-heavy)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    mlp_use_ctx_summary=True,
    mlp_ctx_rank=128,
    mlp_ctx_conductor=True,  # Adaptive based on route gate
)

# Disable context entirely (pure MLP + reciprocity/disc)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    mlp_use_ctx_summary=False,  # No context at all
)
```

## Ablation Studies

Test different bottleneck sizes:

```python
# Aggressive compression (R=64)
mlp_ctx_rank=64  # 49K + 196K = 245K params/layer (10× reduction!)

# Standard (R=128)
mlp_ctx_rank=128  # 98K + 393K = 491K params/layer (5× reduction)

# Higher capacity (R=256)
mlp_ctx_rank=256  # 196K + 786K = 982K params/layer (2.5× reduction)

# Full rank (R=3072, no bottleneck)
mlp_ctx_rank=3072  # Back to 2.36M params/layer (NO reduction, BLOAT)
```

## Key Insight

> **Don't add bloat. Split computation, don't multiply it.**

MLP needs a **hint** from attention, not a full copy. Low-rank factorization gives the hint cheaply.

Route gate + annealing + low-rank context = efficient attention→MLP shift without parameter explosion.

---

**Final params with low-rank R=128**:
- Mechanisms 1-4 (reciprocity, disc, lens, route): 9,444 params (0.01%)
- Mechanism 5 (low-rank context): 5.9M params (7%)
- **Total**: 90.9M vs 85M baseline (7% increase for full lens-gating)

Compare to naive full-rank: 113.3M (33% bloat!)

**We saved 22.4M parameters** with low-rank factorization. That's not bloat - that's efficiency.
