# MLA + Reciprocal MLP: Inference-Efficient Transformer Architecture

**Status**: Experimental
**Last Updated**: 2025-10-29

## Quick Start

```bash
make defconfig-gpt2-ra-mla-full
make
```

This runs MLA (Multi-head Latent Attention) with reciprocal MLP mechanisms enabled.

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
- `CONFIG_GPT2_MAX_ITERS=10400`: Training iterations (good for proper evaluation)
- `CONFIG_RA_MLA_LATENT_DIM=128`: MLA compression dimension
- `CONFIG_RA_MLA_RA_WINDOW=64`: Window size (unused with ra_alpha=0.0)
- `CONFIG_RA_MLA_RA_ALPHA=0.0`: No reciprocal attention scoring overhead
- `CONFIG_RA_MLA_MLP_ATTN_GATE=y`: Enable mechanism 1
- `CONFIG_RA_MLA_MLP_CROSS_TOKEN=y`: Enable mechanism 2
- `CONFIG_RA_MLA_MLP_LATENT_RECIP=y`: Enable mechanism 3

## Implementation

**Core files**:
- `gpt2/train_ra_mla.py`: Training script with ablation support via `--ra-mla-ablation-step`
- `gpt2/ra_mla_gpt2.py`: Architecture implementation with MLA + reciprocal MLP

**Ablation control**: Use `--ra-mla-ablation-step N`:
- Step 0: Baseline (MLA only, no reciprocal mechanisms)
- Step 1: Mechanism 1 only (MLP→Attn Gate)
- Step 2: Mechanisms 1+2 (Gate + Cross-Token)
- Step 3: All three mechanisms (full solution)

## Next Steps

1. **Complete full training**: Run 10,400 iterations for proper evaluation (~36 hours for 6-test ablation)
2. **Memory optimization**: Analyze 34% memory overhead, consider optimizations
3. **Scaling law experiments**: Test aggressive attention compression (latent_dim 64→32) with MLP expansion
4. **Bitter scaling integration**: Combine with pruning methods
5. **Long-context evaluation**: Test on sequences > 2048 tokens to measure KV cache savings

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
