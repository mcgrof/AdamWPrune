# MLA + Reciprocal MLP: Inference-Efficient Transformer Architecture

**Status**: Experimental
**Last Updated**: 2025-10-31

## Quick Start

```bash
make defconfig-gpt2-ra-mla-full
make
```

This runs an ablation study with 6 steps:
- 4 unique configurations testing different reciprocal MLP mechanism combinations
- 2 reproducibility checks (steps 4-5 duplicate steps 2-3 for verification)

All tests use MLA (Multi-head Latent Attention) with memory optimizations (parameter tying and topk sparsification) enabled.

## Overview

This architecture combines DeepSeek's MLA for KV cache compression with three novel reciprocal MLP mechanisms that enable bidirectional information flow between attention and MLP layers. The focus is on inference efficiency while improving model capacity.

**Why not regular Reciprocal Attention (RA)?** Initial exploration showed that reciprocal attention scoring (where tokens mutually attend to each other) adds learned parameters and computational cost during training without clear benefits. We pivoted to reciprocal MLP mechanisms instead, which provide bidirectional flow at lower cost.

## Architecture Components

### Multi-head Latent Attention (MLA)

DeepSeek's approach: compress KV cache from O(n·D·H) to O(n·L) where L << D.

- **Standard attention**: 768 dims × 12 heads = full KV cache
- **MLA**: 768 → 128 latent dims (6× compression)
- **Benefit**: Massive inference memory savings, scales with sequence length

Our implementation: `latent_dim=128`, `ra_window=64`, `ra_alpha=0.0` (no reciprocal scoring overhead)

### Three Reciprocal MLP Mechanisms

These add bidirectional information flow between attention and MLP layers without O(n²) cost:

#### 1. MLP-to-Attention Gating (α=0.1)

MLP activations modulate attention head importance in the next layer:

```python
# In MLP layer L:
gate_context = gate_proj(mlp_hidden)
head_gates = sigmoid(gate_to_heads(gate_context))  # [B, H]

# In attention layer L+1:
gated = (1 - α) * attn_output + α * (attn_output * gate)
```

Cost: Nearly free (small gating network: hidden_dim → gate_dim → n_heads)

#### 2. Cross-Token MLP Aggregation (α=0.3)

MLP receives weighted sum of other tokens' MLP activations using attention weights:

```python
# Reuse attention weights from layer L:
routing_weights = attn_weights.mean(dim=1)  # [B, T, T]
cross_context = bmm(routing_weights, mlp_hidden)
hidden = hidden + α * cross_proj(cross_context)
```

Cost: One aggregation per MLP layer (linear, no extra attention)

Key insight: Creates "attention mass" in MLP space without O(n²) cost.

#### 3. MLP Latent Reciprocity (α=0.2)

Bidirectional pathways between attention and MLP latent spaces:

```python
# In MLP layer L:
mlp_latent = mlp_down(hidden)
attn_contribution = attn_to_mlp(attn_latent_L)
hidden = hidden + α * attn_contribution

# Store for attention layer L+1:
mlp_to_attn_context = mlp_to_attn(mlp_latent)
```

Cost: Small latent projections (hidden_dim ↔ latent_dim)

### Memory Optimization Enhancements

To address the 34% memory overhead observed in initial experiments, we implemented two optimization strategies:

#### Parameter Tying for MLP-Attention Coupling (Mechanism 3)

Three tying modes for bidirectional projections between MLP and attention latent spaces:

**1. Untied (Baseline)**
- Two independent linear maps: attn→mlp (W_a2m) and mlp→attn (W_m2a)
- Most parameters, most expressive
- Memory: 2× projection weights

**2. Tied Transpose (Recommended)**
- One weight W: attn→mlp uses W^T, mlp→attn uses W
- Reduces parameters by 50% for coupling weights
- Memory: 1× projection weight
- Improves training stability via parameter sharing
- Configuration: `CONFIG_RA_MLA_MLP_TYING_MODE="tied_transpose"`

**3. Per-Head Scalar (Experimental)**
- Minimal parameters: only n_heads×2 scalars
- Lowest memory footprint
- May sacrifice expressiveness
- Configuration: `CONFIG_RA_MLA_MLP_TYING_MODE="per_head_scalar"`

```python
# Tied transpose mode: single weight matrix
W = self.W  # [hidden_dim, attn_latent_dim]
mlp_enrich = attn_latent @ W.T      # attn→mlp
attn_context = mlp_hidden @ W       # mlp→attn (transpose)
```

Parameter reduction: ~50% for coupling weights, ~3-4% overall model parameters.

#### Sparsification for Cross-Token MLP (Mechanism 2)

Reduces MLP token broadcasting overhead by keeping only the most important cross-token connections:

**1. Top-K Sparsification (Recommended)**
- Keep only top-k attention weights per token
- Typical k=8 provides good accuracy/efficiency tradeoff
- Expected: 50-75% reduction in aggregation overhead
- Configuration: `CONFIG_RA_MLA_MLP_SPARSE_MODE="topk"`, `CONFIG_RA_MLA_MLP_SPARSE_K=8`

**2. RMS Threshold**
- Keep weights above tau × RMS(row)
- Adaptive sparsity per token
- More flexible than top-k but less predictable
- Configuration: `CONFIG_RA_MLA_MLP_SPARSE_MODE="rms"`, `CONFIG_RA_MLA_MLP_SPARSE_RMS_THRESHOLD="0.5"`

**3. None (Baseline)**
- Use full attention weights
- No sparsification
- Configuration: `CONFIG_RA_MLA_MLP_SPARSE_MODE="none"`

```python
# Top-k sparsification example
routing_weights = attn_weights.mean(dim=1)  # [B, T, T]
vals, idx = torch.topk(routing_weights, k=8, dim=-1)
sparse_weights = torch.zeros_like(routing_weights)
sparse_weights.scatter_(-1, idx, vals)
sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True)
```

Combined impact:
- Parameter tying: ~50% reduction in coupling weights
- Sparsification: 50-75% reduction in MLP aggregation bandwidth
- Expected total memory reduction: addresses the 34% overhead from initial experiments

## Preliminary Results (30 iterations)

Initial ablation study on FineWebEdu dataset (too early for conclusions, but shows direction):

| Step | Mechanisms | Parameters | Val Loss | Improvement | GPU Memory |
|------|-----------|------------|----------|-------------|------------|
| 0 | Baseline (MLA only) | 115.0M | 9.5776 | - | 33.8 GiB |
| 1 | MLP→Attn Gate | 117.4M | 9.5359 | -4.35% | 45.2 GiB |
| 2 | Gate + Cross-Token | 230.6M | 9.5572 | -2.13% | 45.1 GiB |
| 3 | All three mechanisms | 240.3M | 9.5161 | **-6.42%** | 45.2 GiB |

All tests use AdamWSPAM optimizer.

**Key observations** (30 iterations only, not conclusive):
- All reciprocal mechanisms trend positively
- Best configuration: all three mechanisms together (-6.42%)
- Memory overhead: 34% increase (33.8 → 45.2 GiB) mostly from gating mechanism
- Full evaluation requires 10,000+ iterations (CONFIG_GPT2_MAX_ITERS=10400)

## Motivation: Inference Scaling Laws

Traditional scaling laws optimize for training compute but ignore inference cost. Key insights from [Scaling laws meet model architecture: Toward inference-efficient LLMs](https://arxiv.org/pdf/2510.18245) (Sardana & Frankle, 2024):

- MLP expansion is cheap: linear inference cost
- Attention compression gives massive wins: KV cache scales with sequence length
- Optimal ratio: Attention params : MLP params ≈ 1 : 2.5
- Design principle: Shift 20-30% compute from attention → MLP

Standard transformers: Attention → MLP (one-way flow)
Reciprocal MLP: Attention ⇄ MLP (bidirectional flow)

Benefits:
1. More expressive without quadratic attention cost
2. MLP gains cross-token context at linear cost
3. Aligns with inference scaling laws (favor MLP over attention)
4. Flexible: can trade attention compression for MLP capacity

## Configuration System

The defconfig system allows easy testing of different configurations:

```bash
# Full reciprocal MLP (all mechanisms)
make defconfig-gpt2-ra-mla-full
make

# Individual mechanisms can be controlled via CONFIG_ variables
# See defconfigs/gpt2-ra-mla-full for full configuration
```

Key configuration parameters:

**MLA Settings:**
- `CONFIG_GPT2_MAX_ITERS=10400`: Training iterations (good for proper evaluation)
- `CONFIG_RA_MLA_LATENT_DIM=128`: MLA compression dimension
- `CONFIG_RA_MLA_RA_WINDOW=64`: Window size (unused with ra_alpha=0.0)
- `CONFIG_RA_MLA_RA_ALPHA=0.0`: No reciprocal attention scoring overhead

**Reciprocal MLP Mechanisms:**
- `CONFIG_RA_MLA_MLP_ATTN_GATE=y`: Enable mechanism 1 (MLP→Attn gating)
- `CONFIG_RA_MLA_MLP_CROSS_TOKEN=y`: Enable mechanism 2 (cross-token aggregation)
- `CONFIG_RA_MLA_MLP_LATENT_RECIP=y`: Enable mechanism 3 (latent reciprocity)

**Memory Optimizations:**
- `CONFIG_RA_MLA_MLP_TYING_MODE="tied_transpose"`: Parameter tying for mechanism 3
  - Options: "untied", "tied_transpose", "per_head_scalar"
- `CONFIG_RA_MLA_MLP_SPARSE_MODE="topk"`: Sparsification for mechanism 2
  - Options: "none", "topk", "rms"
- `CONFIG_RA_MLA_MLP_SPARSE_K=8`: Top-k value for topk mode
- `CONFIG_RA_MLA_MLP_SPARSE_RMS_THRESHOLD="0.5"`: RMS threshold for rms mode
- `CONFIG_RA_MLA_MLP_SPARSE_NORMALIZE=y`: Re-normalize weights after sparsification
- `CONFIG_RA_MLA_MLP_SPARSE_HEAD_AVERAGE=y`: Average attention weights across heads

## Implementation

**Core files**:
- `gpt2/train_ra_mla.py`: Training script with ablation support via `--ra-mla-ablation-step`
- `gpt2/ra_mla_gpt2.py`: Architecture implementation with MLA + reciprocal MLP

**Ablation control**: Use `--ra-mla-ablation-step N`:
- Step 0: Baseline (MLA only, no reciprocal mechanisms)
- Step 1: Mechanism 1 only (MLP→Attn Gate)
- Step 2: Mechanisms 1+2 (Gate + Cross-Token)
- Step 3: All three mechanisms
- Step 4: Mechanisms 1+2 (reproducibility check, identical to step 2)
- Step 5: All three mechanisms (reproducibility check, identical to step 3)

**Note on steps 4-5**: These are reproducibility checks that run the same configurations as steps 2-3. They were originally intended to compare different optimizers (AdamW vs AdamWSPAM), but the default configuration uses AdamWSPAM for all steps, making them pure reproducibility runs. To test unique configurations only, use `CONFIG_RA_MLA_ABLATION_STEPS="0,1,2,3"`.

## Next Steps

1. **Complete full training**: Run 10,400 iterations for proper evaluation (4 unique configurations + 2 reproducibility checks)
2. **Evaluate memory optimizations**: Test parameter tying and sparsification impact on 34% memory overhead
3. **Ablation study with optimizations**: Compare tied_transpose vs untied, topk vs rms sparsification
4. **Scaling law experiments**: Test aggressive attention compression (latent_dim 64→32) with MLP expansion
5. **Bitter scaling integration**: Combine with pruning methods
6. **Long-context evaluation**: Test on sequences > 2048 tokens to measure KV cache savings

## Why MLP Reciprocity Instead of Attention Reciprocity?

Initial exploration of reciprocal attention scoring (RA) showed:
- Adds learned parameters for reciprocal scoring computation
- Training cost increase without clear inference benefits
- Complexity doesn't align with inference efficiency goals

Reciprocal MLP mechanisms offer:
- Cheaper computational cost (reuse attention weights, linear projections)
- Better alignment with inference scaling laws (favor MLP pathway)
- Bidirectional information flow without O(n²) overhead
- Modular design: can enable/disable mechanisms independently

## References

- [Scaling laws meet model architecture: Toward inference-efficient LLMs](https://arxiv.org/pdf/2510.18245) - Sardana & Frankle (2024)
- DeepSeek (2024): Multi-head Latent Attention (MLA)
- [scaling-inference.txt](scaling-inference.txt): Detailed scaling law analysis
