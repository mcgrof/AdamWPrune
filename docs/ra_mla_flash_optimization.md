# FlashAttention-Style Optimization for RA+MLA

## Executive Summary

RA+MLA trades KV computation from training to inference, but does so inefficiently in the naive implementation. By applying FlashAttention's tiling and fusion principles, we can:

- **Avoid materializing** O(n²·H) attention matrices (same as FlashAttention)
- **Fuse** low-rank projections with attention computation
- **Tile** reciprocal band computation for SRAM locality
- **Expected speedup**: 3-5× over naive RA+MLA, competitive with standard FlashAttention

## FlashAttention Core Principles

### 1. **Tiling** (SRAM Utilization)
Break computation into blocks that fit in fast SRAM (~20 MB on A100), avoiding slow HBM roundtrips.

**Key insight**: SRAM is ~10-20× faster than HBM. Minimize HBM reads/writes.

### 2. **Kernel Fusion**
Fuse multiple operations into a single kernel to avoid materializing intermediate tensors.

**Example**: Instead of:
```python
Q_lat = Q @ W_q_lat  # Materialize [B,T,H,L]
scores = Q_lat @ K_lat.T  # Materialize [B,H,T,T]
attn = softmax(scores)  # Materialize [B,H,T,T]
out = attn @ V  # Final output
```

Do:
```python
# Fused: never materialize full scores/attn
for block in tiles:
    q_block = Q[block] @ W_q_lat  # Small block
    scores_block = q_block @ K_lat.T
    attn_block = softmax(scores_block)  # Small block
    out[block] += attn_block @ V  # Accumulate
```

### 3. **Online Softmax**
Compute softmax incrementally without materializing full attention matrix.

**Algorithm** (Milakov & Gimelshein, 2018):
```
For each query block:
    m_i = -∞  # running max
    l_i = 0   # running sum
    o_i = 0   # running output

    For each key/value block:
        # Compute block scores
        S_ij = Q_i @ K_j^T

        # Update running statistics
        m_new = max(m_i, rowmax(S_ij))
        l_new = exp(m_i - m_new) * l_i + rowsum(exp(S_ij - m_new))
        o_new = exp(m_i - m_new) * o_i + exp(S_ij - m_new) @ V_j

        m_i, l_i, o_i = m_new, l_new, o_new

    # Final normalization
    O_i = o_i / l_i
```

This never materializes the full O(n²) attention matrix!

## RA+MLA Memory Complexity Analysis

### Naive Implementation (Current)

**Forward pass materializes:**
1. `Q_latent`: [B, T, H, L] - query-to-latent projection
2. `logits`: [B, H, T, T_tot] - attention scores (**the bottleneck**)
3. `logits_recip`: [B, H, T, T_tot] - reciprocal scores (if RA enabled)
4. `attn`: [B, H, T, T_tot] - softmax output
5. `V_expanded`: [B, T_tot, H, D] - expanded values

**Memory**: O(B·H·T² + B·T·H·L + B·T·H·D)

The **O(B·H·T²)** term dominates for long sequences!

### FlashAttention-Style (Optimized)

**Materialize only:**
1. Small tiles: [B, H, B_q, B_k] where B_q, B_k ≪ T
2. Running statistics: [B, H, T] for max/sum
3. Latent K/V: [B, T, L] (already compressed!)

**Memory**: O(B·T·H·L + B·T·H·D + B·H·T)

For L = 64, D = 64, H = 12, this is **~20× less memory** than naive!

## Tiled Algorithm for RA+MLA

### Block Sizes

```python
# Tuned for A100 (40GB HBM, 20MB SRAM per SM)
B_q = 128  # Query block size (rows of attention matrix)
B_k = 128  # Key block size (columns of attention matrix)
L = 64     # Latent dimension (given)

# SRAM budget per block:
# - Q_block: [B_q, H, D] ≈ 128 * 12 * 64 * 2 bytes = 196 KB
# - K_lat_block: [B_k, L] ≈ 128 * 64 * 2 bytes = 16 KB
# - V_lat_block: [B_k, L] ≈ 16 KB
# - Scores_block: [H, B_q, B_k] ≈ 12 * 128 * 128 * 4 bytes = 786 KB
# - Total: ~1 MB ≪ 20 MB ✓
```

### Algorithm Pseudocode

```python
def flash_ra_mla_attention(
    Q: [B, T, H, D],          # Per-head queries
    K_lat: [B, T_tot, L],     # Latent keys (shared across heads)
    V_lat: [B, T_tot, L],     # Latent values (shared across heads)
    W_q_lat: [H, D, L],       # Q-to-latent projection
    W_v_up: [H, L, D],        # V up-projection
    ra_alpha: float,          # Reciprocal weight
    ra_window: int,           # Reciprocal band width
) -> [B, T, H, D]:

    B, T, H, D = Q.shape
    T_tot, L = K_lat.shape[1], K_lat.shape[2]

    # Output and running statistics (in HBM, but small)
    O = torch.zeros(B, T, H, D)
    m = torch.full((B, H, T), -inf)  # running max
    l = torch.zeros(B, H, T)         # running sum

    # Divide into blocks
    num_q_blocks = (T + B_q - 1) // B_q
    num_k_blocks = (T_tot + B_k - 1) // B_k

    # Outer loop: iterate over query blocks (rows)
    for q_idx in range(num_q_blocks):
        q_start = q_idx * B_q
        q_end = min(q_start + B_q, T)

        # Load query block into SRAM
        Q_block = Q[:, q_start:q_end, :, :]  # [B, B_q, H, D]

        # Project to latent space (FUSED)
        Q_lat_block = einsum('bqhd,hdl->bqhl', Q_block, W_q_lat)  # [B, B_q, H, L]

        # Initialize block accumulator (in SRAM)
        O_block = torch.zeros(B, q_end - q_start, H, D)
        m_block = torch.full((B, H, q_end - q_start), -inf)
        l_block = torch.zeros(B, H, q_end - q_start)

        # Inner loop: iterate over key/value blocks (columns)
        for k_idx in range(num_k_blocks):
            k_start = k_idx * B_k
            k_end = min(k_start + B_k, T_tot)

            # Load K/V latent block into SRAM
            K_lat_block = K_lat[:, k_start:k_end, :]  # [B, B_k, L]
            V_lat_block = V_lat[:, k_start:k_end, :]  # [B, B_k, L]

            # === Compute attention scores (FUSED) ===

            # Standard attention: Q_lat @ K_lat^T
            scores_block = einsum('bqhl,bkl->bhqk', Q_lat_block, K_lat_block) / sqrt(L)
            # Shape: [B, H, B_q, B_k]

            # === Reciprocal Attention (if enabled) ===
            if ra_alpha > 0:
                # Compute reciprocal term: K_lat @ Q_lat^T (transposed direction)
                # But we need Q from the KEY positions, not current query block
                # This requires caching or recomputation - see note below
                scores_recip = compute_reciprocal_scores(
                    Q_lat_block, K_lat_block, q_start, k_start, ra_window
                )

                # Apply band mask
                band_mask = create_band_mask(q_start, q_end, k_start, k_end, ra_window)
                scores_block = torch.where(
                    band_mask,
                    scores_block + ra_alpha * scores_recip,
                    scores_block
                )

            # === Causal masking ===
            causal_mask = create_causal_mask(q_start, q_end, k_start, k_end)
            scores_block = scores_block.masked_fill(~causal_mask, -inf)

            # === Online softmax update ===

            # Block max
            m_block_k = scores_block.max(dim=-1).values  # [B, H, B_q]

            # Update running max
            m_new = torch.maximum(m_block, m_block_k)

            # Update running sum (with correction for max change)
            alpha = torch.exp(m_block - m_new)
            beta = torch.exp(scores_block - m_new.unsqueeze(-1))  # [B, H, B_q, B_k]

            l_new = alpha * l_block + beta.sum(dim=-1)

            # === Expand V and compute weighted output ===

            # V up-projection (FUSED)
            V_block = einsum('bkl,hld->bkhd', V_lat_block, W_v_up)  # [B, B_k, H, D]

            # Weighted sum (with correction for max change)
            O_new = alpha.unsqueeze(-1) * O_block + einsum('bhqk,bkhd->bqhd', beta, V_block)

            # Update accumulators
            m_block = m_new
            l_block = l_new
            O_block = O_new

        # Normalize and write back to HBM
        O[:, q_start:q_end, :, :] = O_block / l_block.unsqueeze(-1)
        m[:, :, q_start:q_end] = m_block
        l[:, :, q_start:q_end] = l_block

    return O
```

### Reciprocal Computation Challenge

The reciprocal term needs `Q_lat[k_positions]` when computing attention from `q_positions`. This creates a dependency:

**Option 1: Cache Q_lat in HBM** (memory trade-off)
- Store `Q_lat_all`: [B, T_tot, H, L] in HBM
- Load `Q_lat_block` corresponding to key positions
- Memory cost: O(B·T·H·L) - still much better than O(B·H·T²)!

**Option 2: Recompute Q_lat on-the-fly** (compute trade-off)
- When processing key block [k_start:k_end], recompute `Q_lat` for those positions
- Requires storing hidden states [B, T_tot, E] or reloading them
- 2× FLOP cost for Q-to-latent projection, but no extra memory

**Option 3: Hybrid - cache only within window** (best of both)
- Cache `Q_lat` only for positions within `ra_window` of current query block
- Memory: O(B·H·W·L) where W = ra_window
- For W = 64, this is **64× smaller** than caching all Q_lat!

**Recommendation**: Option 3 (hybrid) gives the best trade-off.

## Memory Access Analysis

### Naive Implementation

**Per iteration over T query positions:**
- Read Q: [B, T, H, D] from HBM (B·T·H·D reads)
- Write Q_lat: [B, T, H, L] to HBM (B·T·H·L writes)
- Read Q_lat: [B, T, H, L] from HBM (B·T·H·L reads)
- Read K_lat: [B, T, L] from HBM × H heads (B·T·L·H reads)
- Write scores: [B, H, T, T] to HBM (B·H·T² writes) **← bottleneck**
- Read scores: [B, H, T, T] from HBM (B·H·T² reads) **← bottleneck**
- Write attn: [B, H, T, T] to HBM (B·H·T² writes) **← bottleneck**

**Total HBM traffic**: O(B·H·T²) - dominated by attention matrix

### Tiled Implementation

**Per query block [B_q rows]:**
- Read Q_block: [B, B_q, H, D] (once per q_block)
- Compute Q_lat_block in SRAM (no HBM write!)
- For each key block:
  - Read K_lat_block: [B, B_k, L] (once per k_block, shared across heads!)
  - Read V_lat_block: [B, B_k, L]
  - Compute scores_block in SRAM (no HBM write!)
  - Compute attn_block in SRAM (no HBM write!)
  - Update O_block in SRAM
- Write O_block: [B, B_q, H, D] to HBM

**Total HBM traffic**: O(B·T·H·D + B·T·L·(T/B_k))

For B_k = 128, this is **~128× less** HBM traffic for the K_lat reads!

### Speedup Estimation

**Arithmetic intensity** (FLOPs per byte):

- Naive: ~1-2 (memory-bound, wasting bandwidth)
- Tiled: ~10-20 (compute-bound, utilizing tensor cores)

**Expected speedup over naive RA+MLA**: 3-5×

**Comparison to standard FlashAttention**:
- FlashAttention: O(n²·D) FLOPs, O(n·D) HBM traffic
- Tiled RA+MLA: O(n²·L) FLOPs, O(n·L) HBM traffic
- If L = D/4, RA+MLA is **4× faster** than FlashAttention (in theory)

## Implementation Roadmap

### Phase 1: PyTorch Prototype (1-2 days)
```python
# File: gpt2/ra_mla_flash_pytorch.py

def flash_ra_mla_pytorch(Q, K_lat, V_lat, cfg):
    """
    Pure PyTorch tiled implementation.
    Won't match CUDA performance but validates algorithm.
    """
    # Implement tiled algorithm from pseudocode above
    # Use torch.utils.checkpoint for memory efficiency
    ...
```

**Goal**: Verify correctness, measure memory savings

### Phase 2: Triton Kernel (3-5 days)
```python
# File: gpt2/ra_mla_triton.py

import triton
import triton.language as tl

@triton.jit
def ra_mla_fwd_kernel(
    Q, K_lat, V_lat, W_q_lat, W_v_up,
    O, m, l,  # outputs
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kl,
    # ... more strides
    B_q: tl.constexpr, B_k: tl.constexpr,
    L: tl.constexpr, ra_alpha: tl.constexpr,
):
    """
    Triton kernel for RA+MLA forward pass.
    Advantages: Python-like syntax, automatic SRAM management
    """
    # Block indexing
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Load Q block into SRAM
    q_offsets = pid_q * B_q + tl.arange(0, B_q)
    Q_block = tl.load(Q + q_offsets, mask=q_offsets < T)

    # Project to latent (SRAM only)
    Q_lat_block = tl.dot(Q_block, W_q_lat[pid_h])

    # Loop over K blocks
    for k_block_idx in range(0, T_tot, B_k):
        # Load K_lat, V_lat blocks
        k_offsets = k_block_idx + tl.arange(0, B_k)
        K_lat_block = tl.load(K_lat + k_offsets, mask=k_offsets < T_tot)
        V_lat_block = tl.load(V_lat + k_offsets, mask=k_offsets < T_tot)

        # Compute scores (SRAM)
        scores_block = tl.dot(Q_lat_block, tl.trans(K_lat_block)) / tl.sqrt(L)

        # Reciprocal term (if enabled)
        if ra_alpha > 0:
            # Load cached Q_lat for reciprocal
            ...

        # Online softmax update
        ...

    # Write output to HBM
    tl.store(O + q_offsets, O_block)
```

**Advantages of Triton**:
- Python-like syntax (easier than CUDA)
- Automatic SRAM management
- Performance competitive with hand-written CUDA (90-95%)
- Integrates directly with PyTorch

**Timeline**: 3-5 days for forward + backward pass

### Phase 3: CUDA Kernel (7-10 days, optional)
Only if you need the last 5-10% performance or Triton limitations.

**When to use CUDA**:
- Need precise control over SRAM usage
- Targeting older GPUs without good Triton support
- Production deployment where every % matters

## Backward Pass Considerations

FlashAttention saves memory in backward pass too! The key insight:

**Standard backward**: Need to store full attention matrix for gradient computation
**Flash backward**: Recompute attention on-the-fly from Q, K, V

For RA+MLA:
- Recompute `scores` from `Q_lat` and `K_lat` (cheap: O(n²·L))
- Recompute `attn` via online softmax
- Compute gradients in tiles

**Memory**: O(n·L) instead of O(n²·H) - even more critical for training!

## Practical Next Steps

1. **Validate naive implementation** (current `ra_mla_gpt2.py`)
   - Train small model (1K iters) to verify correctness
   - Profile memory and time

2. **Implement PyTorch tiled version**
   - Write `flash_ra_mla_pytorch.py`
   - Verify output matches naive (within numerical tolerance)
   - Measure memory savings

3. **Benchmark PyTorch tiled vs naive**
   - Sequence lengths: 512, 1024, 2048, 4096
   - Measure: time, memory, perplexity
   - Expect: ~30-50% memory reduction even in PyTorch

4. **Implement Triton kernel**
   - Start with forward pass only
   - Use `torch.autograd.Function` wrapper
   - Backward pass can use PyTorch autograd initially

5. **Full Triton forward + backward**
   - Implement recomputation-based backward
   - Should match FlashAttention performance (within 10%)

## Expected Performance

**Sequence length: 2048, Batch: 8, GPT-2 (124M)**

| Implementation | Memory (GB) | Time/iter (ms) | Throughput (tok/s) |
|----------------|-------------|----------------|---------------------|
| Naive RA+MLA | 22 | 450 | 36K |
| PyTorch Tiled | 14 | 350 | 47K |
| Triton Kernel | 12 | 180 | 91K |
| Standard Flash | 11 | 200 | 82K |

**Key takeaway**: Triton RA+MLA should be **10% faster** than FlashAttention due to lower FLOP count (L < D)!

## References

1. **FlashAttention**: Dao et al. (2022) - https://arxiv.org/abs/2205.14135
2. **FlashAttention-2**: Dao (2023) - https://arxiv.org/abs/2307.08691
3. **Online Softmax**: Milakov & Gimelshein (2018) - https://arxiv.org/abs/1805.02867
4. **Triton**: Tillet et al. (2019) - https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

## Conclusion

RA+MLA is a perfect candidate for FlashAttention-style optimization:
- Already doing custom computation (low-rank projections)
- Latent space L << D reduces FLOP count
- Tiling eliminates O(n²) memory bottleneck
- Expected 3-5× speedup over naive, competitive with standard FlashAttention

**The combination of algorithmic efficiency (low-rank) + systems optimization (tiling) could make RA+MLA the fastest attention mechanism for long contexts!**
