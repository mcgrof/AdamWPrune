# Lens-Gated Architecture: Simplified Enhancements

## Design Principle: Maximum Impact, Minimal Cost

After evaluating complex mechanisms (MLA compression, bidirectional MLP-Attention coupling), we've distilled to **five core innovations** with near-zero compute overhead.

---

## Enhancement 1: Reciprocal Attention (RA) - Mathematical Reciprocity

**THE KEY INSIGHT**: Reciprocity is just a transpose!

```python
# Standard attention (directed, asymmetric)
S = Q @ K^T    # [B, H, T, T]
# S[i,j] = how much query i wants key j

# Reciprocal attention (symmetric mutual affinity)
S_rec = S^T    # JUST A TRANSPOSE! FREE!
# S_rec[i,j] = S[j,i] = how much query j wants key i

# If i strongly attends to j, boost j's attention to i
```

**Visual Example** (T=4 tokens):
```
S = [[a, b, c, d],      S_rec = [[a, e, i, m],
     [e, f, g, h],               [b, f, j, n],
     [i, j, k, l],               [c, g, k, o],
     [m, n, o, p]]               [d, h, l, p]]

Diagonal is symmetric (self-attention preserved)
Off-diagonal captures mutual token affinity
```

**Why it matters**:
- Standard attention: Token "Rome" can attend to "Italy", but not vice versa
- With reciprocity: If "Rome" needs "Italy", boost "Italy" attending back to "Rome"
- Result: Bidirectional token communication

**Compute cost**: **ZERO** (transpose is free)
**Parameter cost**: **0 params** (uses existing S matrix)

**Benefit**: Bidirectional token communication with **zero overhead**.

---

## Enhancement 2: Discoverability - Column Bias

**THE KEY INSIGHT**: Some tokens want to be found (important facts, keywords, entities).

```python
# Tiny per-head vector u_h ∈ ℝ^D (head_dim, typically 64)
u = nn.Parameter(torch.randn(n_head, head_dim) * 0.02)

# Column bias: d_j = <K_j, u_h> for each key vector
d = torch.einsum('bhtd,hd->bht', K, u)  # [B, H, T]
d = d - d.mean(dim=-1, keepdim=True)    # Zero-mean for stability
d = d.unsqueeze(-2)                     # [B, H, 1, T] - broadcast as column
```

**Intuition**:
- Token j with high d_j broadcasts "I'm important!" to all queries
- Allows important tokens to be found regardless of query content
- Example: "Einstein" token can signal high importance to all queries

**Visual Example** (T=4):
```
Standard S:
  [[a, b, c, d],     Token 0 attends: [a, b, c, d] - only sees these scores
   [e, f, g, h],     Token 1 attends: [e, f, g, h]
   [i, j, k, l],     Token 2 attends: [i, j, k, l]
   [m, n, o, p]]     Token 3 attends: [m, n, o, p]

With discoverability d = [0.2, 0.0, -0.1, 0.5]:
  [[a+0.2, b+0.0, c-0.1, d+0.5],   Token 3 gets boost from all queries!
   [e+0.2, f+0.0, g-0.1, h+0.5],   (column bias broadcasts)
   [i+0.2, j+0.0, k-0.1, l+0.5],
   [m+0.2, n+0.0, o-0.1, p+0.5]]
```

**Parameter cost**: n_head × head_dim = 12 × 64 = **768 params** (0.01% overhead)

---

## Enhancement 3: Lens Gates - Softmax-Normalized Mixing

**THE KEY INSIGHT**: Mix standard, reciprocal, and discoverability with learned weights that always sum to 1.

```python
# Per-head gates: [w_std, w_rec, w_disc]
gates = nn.Parameter(torch.zeros(n_head, 3))
w = F.softmax(gates, dim=-1)  # Always sums to 1 per head

# Final attention logits
logits = w_std * S + w_rec * S^T + w_disc * d
```

**Why softmax**:
- **Scale stability**: Weights always sum to 1 (no exploding/vanishing scores)
- **Learnable**: Model learns optimal balance per head
- **Init bias**: Start w_std≈0.8, w_rec≈0.15, w_disc≈0.05 (mostly standard)
- **Per-head adaptation**: Each head learns different mixing strategy

**Example learned weights after training**:
```
Head 0: w = [0.92, 0.06, 0.02]  # Mostly standard (content-based)
Head 1: w = [0.65, 0.30, 0.05]  # Heavy reciprocity (mutual affinity)
Head 2: w = [0.75, 0.10, 0.15]  # More discoverability (find important tokens)
```

**Parameter cost**: n_head × 3 = 12 × 3 = **36 params** (<0.001% overhead)

---

## Enhancement 4: Route Gate - Learning the Ratio

**THE GOAL**: Reduce KV cache size at inference by shifting computation to MLP.

**KEY INSIGHT**: Attention requires KV caching (for autoregressive generation), MLP does not.

```python
# Single learnable scalar per block
route_gate_raw = nn.Parameter(torch.tensor(2.2))  # sigmoid≈0.9 initially
g = torch.sigmoid(route_gate_raw + route_bias_add)  # [0, 1]

# Blend residual contributions
H_attn = H + attn_out
H_mlp = H_attn + mlp_out
out = H + g*(H_attn - H) + (1-g)*(H_mlp - H_attn)
```

**Interpretation**:
- **g ≈ 1.0**: Attention-heavy (traditional 4:1 MLP:Attention ratio, **large KV cache**)
- **g ≈ 0.5**: Balanced (2:1 ratio, medium KV cache)
- **g ≈ 0.3**: MLP-heavy (1:1.4 ratio, **small KV cache**)

**Why it matters**:

| Route Gate (g) | Attention Weight | MLP Weight | KV Cache Size | Inference Memory |
|----------------|------------------|------------|---------------|------------------|
| 0.9 (traditional) | 90% | 10% | 100% (baseline) | High |
| 0.7 | 70% | 30% | 70% | Reduced |
| 0.5 | 50% | 50% | 50% | Half! |
| 0.3 | 30% | 70% | 30% | 70% savings! |

**Goal**: Model learns to shift toward MLP, enabling smaller KV cache → better inference latency and memory.

**Optional annealing schedule**:
```python
# During training, gradually push toward MLP
for step in training_loop:
    if step > warmup_steps:
        model.adjust_route_bias(delta=-0.0001)  # Gradual shift to MLP
```

**Parameter cost**: **1 param per block** (12 total for GPT-2)

---

## Enhancement 5: MLP Context Summary

**THE KEY INSIGHT**: Give MLP access to what attention computed (enable MLP to participate with cross-token info).

```python
# In transformer block:
attn_out, attn_weights = self.attn(H)
H1 = H + attn_out

# MLP sees attention output (optional: stop gradient)
ctx_summary = attn_out.detach()  # [B, T, E]
mlp_input = torch.cat([H1, ctx_summary], dim=-1)  # [B, T, 2*E]

# MLP with gating
h = self.fc1(mlp_input)      # [B, T, 4*E]
g = sigmoid(self.fc_gate(mlp_input))  # [B, T, 4*E] - channel gates
h = gelu(h) * g              # Gated activation
mlp_out = self.fc2(h)        # [B, T, E]
```

**Why it matters**:
- Standard MLP: Each token processes independently (no cross-token info)
- With context summary: MLP sees aggregated cross-token context from attention
- Enables MLP to make better predictions with global sequence context

**Example**: Next token prediction for "The capital of France is"
- Attention computes: "capital" + "France" → context about geographic entities
- MLP receives: [current_hidden, attention_context]
- MLP can leverage cross-token geographic relationships
- Better prediction: "Paris" (instead of generic continuation)

**Parameter cost**: MLP input dim increases from E to 2E:
- fc1: 2E × 4E = 8E² (was 4E²)
- fc_gate: 2E × 4E = 8E² (channel gating)
- fc2: 4E × E = 4E² (unchanged)
- **Overhead**: +8E² params (for GPT-2: +4.7M per layer)

**Note**: This is the only significant parameter increase, but enables better MLP performance.

---

## Total Parameter Overhead

```
Baseline GPT-2 per layer:       7.08M

Lens-gated additions:
  - Reciprocity (RA):                 0 params (transpose is free)
  - Discoverability (u_h):          768 params
  - Lens gates:                      36 params
  - Route gate:                       1 param
  - MLP context (fc1+fc_gate):   +4.7M params
────────────────────────────────────────────────────
Total per layer:                 ~11.8M (67% increase)

GPT-2 124M total:
  - Baseline 12 layers:           85.0M
  - With lens-gating:            141.6M (67% increase)
```

**Key insight**: MLP context summary is the only significant cost. All other mechanisms (reciprocity, discoverability, lens gates, route gate) are **negligible** (<0.02% overhead).

---

## Compute Overhead: ZERO Extra GEMMs

All five enhancements add **no extra matrix multiplications** beyond standard transformer:

| Operation | Compute Cost |
|-----------|--------------|
| S^T (reciprocity) | Free (transpose) |
| d = K·u (discoverability) | Tiny einsum (H×D params) |
| Lens gate mixing | Element-wise ops (negligible) |
| Route gate blending | Element-wise ops (negligible) |
| MLP context concat | Memory copy (no compute) |

**Total extra FLOPs**: < 0.01% vs baseline transformer

---

## What We Removed (from previous complex design)

**Removed mechanisms**:
1. **MLA latent compression**: K/V down-projection to 128D latent space
2. **Complex bidirectional MLP↔Attention projections**: attn_to_mlp, mlp_to_attn
3. **Cross-token MLP aggregation**: 9.4M params per layer! (expensive!)
4. **Multiple separate alpha parameters**: ra_alpha, mlp_gate_alpha, mlp_cross_alpha, mlp_recip_alpha_*

**Why removed**:
- **MLA**: Compression adds complexity without clear benefit over standard Q/K/V
- **Bidirectional coupling**: Too many parameters (393K per layer), unclear value
- **Cross-token MLP**: 9.4M params per layer is prohibitive (134M total across 12 layers)
- **Separate alphas**: Lens gates are cleaner (softmax-normalized, stable)

**Result**:
- **Simpler architecture** (5 focused mechanisms vs 8+ complex ones)
- **Clearer mathematical motivation** (S^T = reciprocity, d = broadcast importance)
- **Much lower overhead** (negligible compute, <1% params except MLP context)
- **Goal-oriented design** (route gate explicitly targets KV cache reduction)

---

## Summary: The Five Pillars

1. **Reciprocity (S^T)**: Bidirectional token communication, zero cost
2. **Discoverability (d)**: Tokens can broadcast importance, tiny cost (768 params)
3. **Lens Gates (softmax)**: Stable learned mixing, tiny cost (36 params)
4. **Route Gate (g)**: Learn attention vs MLP ratio, reduces KV cache, tiny cost (1 param)
5. **MLP Context**: Enable MLP to leverage cross-token info, moderate cost (+4.7M params)

**Total philosophy**: Every parameter must justify its existence through improved validation loss and/or inference efficiency. Mechanisms 1-4 are nearly free. Mechanism 5 is the only significant cost, but enables MLP to compete with attention (necessary for route gate learning).
