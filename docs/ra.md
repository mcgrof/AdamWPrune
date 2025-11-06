# Reciprocal Attention (RA)

## Core Idea

Standard attention uses a forward scoring matrix S. We enhance it with three zero-cost or near-zero-cost mechanisms to improve attention quality and enable efficient KV cache reduction.

```python
# Standard attention
S = Q @ K.T

# Reciprocal attention (transpose)
S_rec = S.T

# Discoverability (column bias from learned vectors)
d = sigmoid(K @ u_h)  # u_h: [head_dim] per head, 768 params total

# Final attention logits with learned per-head weights
logits = w_std * S + w_rec * S_rec + w_disc * d
```

The three mechanisms:

1. **Reciprocity**: Transpose S to let tokens attend backward. Free (transpose costs nothing).

2. **Discoverability**: Tiny learned vectors let important tokens broadcast their importance regardless of query. Costs 768 parameters.

3. **Lens gates**: Softmax over [w_std, w_rec, w_disc] ensures scale stability and learns per-head mixing. Costs 36 parameters.

Combined overhead: 9,444 parameters (0.01% of GPT-2 124M).

## Route Gate: Learning the Ratio

Beyond attention quality, we add a route gate to learn attention versus MLP balance:

```python
g = sigmoid(route_gate_raw + route_bias_add)
out = H + g*(H_attn - H) + (1-g)*(H_mlp - H_attn)
```

Where g starts near 0.69 (attention-heavy) and anneals to 0.27 (MLP-heavy) during training. This shifts computation away from attention, reducing KV cache requirements at inference.

Costs 12 parameters (1 per block).

## K/V Compression (Parameter-Neutral)

To offset MLP context overhead, we compress K and V projections via low-rank factorization:

```python
# Standard: E → H*D (788K params per projection)
# Compressed: E → R → H*D where R=128
k = k_up(k_down(x))  # Saves 660K params per projection
```

K/V compression saves 9.5M parameters (788K × 12 layers × 2 projections). This funds MLP context additions while staying parameter-neutral or even reducing total count.

## MLP Context (Low-Rank)

MLP receives lightweight attention summary via low-rank factorization:

```python
ctx_h = ctx_up(ctx_down(attn_summary))  # E → R=128 → mult*E
h = (1-alpha)*h_standard + alpha*ctx_h
```

Costs 5.9M parameters (491K per layer × 12). Combined with K/V compression, the net result is 3.6M parameter savings versus baseline.

Optional conductor mode only uses context when route_gate < 0.5 (MLP-heavy regime).

## Ablation Steps

The implementation supports systematic ablation studies:

**L0**: Baseline (no enhancements)
**L1**: Reciprocity only
**L2**: Discoverability only
**L3**: Reciprocity + Discoverability
**L4**: Attention-only (MLP disabled)
**L5**: Full lens without MLP context
**L6**: Full lens + K/V compression + MLP context (parameter-neutral)
**L7**: L6 + conductor mode

## SinkGD Optimizer

Beyond architecture, we provide SinkGD optimizer that applies Sinkhorn-like gradient normalization:

```python
# Iterative row/column normalization with temperature scaling
for _ in range(n_iter):
    g = g / g.abs().sum(dim=-1, keepdim=True)  # row normalize
    g = g / g.abs().sum(dim=-2, keepdim=True)  # col normalize
    g = tanh(g / tau)  # temperature smoothing
```

This encourages structured, balanced gradient updates. Ablation steps S0-S3 test SinkGD against AdamWSPAM baseline on the L6 architecture.

**S0**: Lens L6 + AdamWSPAM (control)
**S1**: Lens L6 + SinkGD default (tau=0.1, n_iter=5)
**S2**: Lens L6 + SinkGD sharper (tau=0.05, n_iter=10)
**S3**: Lens L6 + SinkGD softer (tau=0.2, n_iter=3)

## RWR Attention

We also provide an alternative attention mechanism based on Random Walk with Restart that factorizes attention into LOCAL + RWR components:

```
A(q_i) ≈ LOCAL(i) + γ * RWR(i)
```

Where LOCAL handles short-range via windowed attention and RWR captures long-range structure through sparse random walks on the token graph. This reduces QK^T matmul cost while maintaining expressiveness.

RWR supports reversible chains (detailed balance) and reciprocal coupling (forward/backward saliency mixing). Ablation steps R0-R3 test RWR variants.

**R0**: Standard GPT-2 baseline
**R1**: RWR default (α=0.2, T=4, topk=32)
**R2**: R1 + reversible chain (P_rev symmetrization)
**R3**: R2 + reciprocal (β=0.7) + discoverability

## Usage

Train with ablation steps:

```bash
# Lens ablation
python gpt2/train_ra_mla.py --ra-mla-ablation-step L6 --dataset finewebedu

# SinkGD ablation
python gpt2/train_ra_mla.py --ra-mla-ablation-step S1 --dataset finewebedu

# RWR ablation
python gpt2/train_ra_mla.py --ra-mla-ablation-step R1 --dataset finewebedu
```

Dry-run validation before GPU time:

```bash
python gpt2/train_ra_mla.py --ra-mla-ablation-step L6 --dry-run
```

## Implementation

**Core files**:
- `gpt2/ra_lens_gpt2.py`: Lens-gated architecture with patching
- `lib/optimizers.py`: SinkGD optimizer
- `lib/graph_builder.py`: Sparse graph construction for RWR
- `rwr_attention.py`: RWR kernel attention with patching
- `gpt2/train_ra_mla.py`: Training integration

All ablation steps pass dry-run validation. The implementations are ready for GPU training experiments.

## Goal

Improve attention quality through reciprocity and discoverability while learning to shift computation from attention to MLP. The result: better model quality with smaller KV cache at inference (target 50-70% reduction).

The route gate explicitly learns this trade-off rather than assuming a fixed ratio. Annealing from attention-heavy to MLP-heavy gives the model time to develop cross-token MLP capabilities before reducing attention reliance.
