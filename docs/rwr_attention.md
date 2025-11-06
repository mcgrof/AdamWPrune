# RWR (Random Walk with Restart) Attention

## Overview

RWR attention factorizes transformer attention into LOCAL + RWR components, where LOCAL handles short-range patterns via windowed attention and RWR captures long-range structure through token-graph random walks with restart. This reduces full QK^T matmuls while maintaining expressive power through graph-based importance signals.

The factorization:
```
A(q_i, :) ≈ LOCAL(i) + γ · RWR(i)
```

where:
- LOCAL(i): Standard windowed attention (e.g., ±128 token window)
- RWR(i): Long-range attention via random walk saliency scores
- γ: Blending strength (default: 0.3)

This design achieves better cache behavior and provides natural importance signals from random walk hitting probabilities.

## Mathematical Foundation

### Random Walk with Restart

Given tokens as graph nodes with similarity edges W (from cosine(Q,K)), row-normalize to transition matrix P = D^{-1}W. The RWR vector for query i:

```
r_i = α * e_i + (1-α) * r_i P
```

Fixed-point solution via truncated Neumann series (T steps):
```
r_i ≈ α * sum_{t=0..T} ((1-α) P)^t e_i
```

We never form the dense inverse - only T sparse matvecs with P. Top entries of r_i indicate long-range visitation importance.

### Reversible (Detailed Balance) Variant

```
P_rev = 1/2 ( P + D^{-1} P^T D )
```

This guarantees a reversible chain w.r.t. stationary π ∝ D·1, improving stability and smoothing asymmetries. Pairs nicely with reciprocal lens pattern from RA work.

### Reciprocal Coupling (Lens)

Mix forward and backward walk saliency:
```
r_i^recip = β * r_i + (1-β) * (backward saliency from P^T)
```

Optionally use discoverability column bias (learnable per-head vectors) to allow tokens to broadcast importance.

### Final Attention

```
A(i,:) = softmax(LOCAL(i,:) ⊕ γ * PROJECT(r_i^recip))
```

where PROJECT maps RWR scores onto value indices (same K index space), ⊕ denotes masked addition over participating indices.

##

 Architecture Components

### 1. Graph Builder (lib/graph_builder.py)

Constructs sparse similarity graphs from Q, K:

- **build_sparse_W()**: Tiled cosine similarity search
  - Keeps local window (±w) + topk global neighbors
  - Ensures connectivity while maintaining sparsity

- **normalize_to_P()**: Row-normalize W to transition matrix P
  - P = D^{-1}W where D = diag(row sums)
  - Returns CSR sparse format

- **reversible()**: Symmetrize for detailed balance
  - P_rev = 0.5*(P + D^{-1}P^T D)
  - Enforces reversibility w.r.t. stationary distribution

- **sparse_mm_batch()**: Batched sparse matrix multiplication
  - Efficient R @ P operations for RWR iterations

### 2. RWR Kernel Attention (rwr_attention.py)

Main attention module combining LOCAL + RWR:

- **RWRKernelAttention**: PyTorch module with configurable hyperparameters
  - rwr_alpha: Restart probability (default: 0.2)
  - rwr_steps: Walk iterations T (default: 4)
  - rwr_topk: Neighbors per query (default: 32)
  - window: Local attention window (default: 128)
  - reversible: Enable P_rev (default: False)
  - reciprocal_beta: Forward/backward mixing (default: 0.5)
  - lens_strength: RWR blending γ (default: 0.3)

- **_local_flash_like()**: Windowed attention fallback
  - Applies window mask to standard attention
  - TODO: Integrate real FlashAttention for production

- **_build_sparse_P()**: Constructs transition matrix
  - Calls graph_builder functions
  - Caches P for reuse (e.g., KV cache scenarios)

- **_rwr_long_range()**: Computes RWR saliency
  - Truncated Neumann series: R = α Σ((1-α)P)^t E
  - Supports reversible and reciprocal variants
  - Selects topk indices, gathers V, weighted sum

### 3. FlashAttention-Style Tiling (Planned)

SRAM-friendly tiling for memory efficiency:
- Tile K,V into on-chip blocks (e.g., 128×head_dim)
- Stream Q blocks with fused softmax
- Tensor-core alignment (head_dim multiples of 16/64)

Currently uses fallback implementation; full tiling integration is TODO.

## Ablation Study: R0-R3

Tests RWR variants on standard GPT-2 baseline.

### R0: Standard GPT-2 (control)

Pure baseline attention with no RWR, RA, or MLA enhancements.

**Configuration**:
- use_rwr=False
- Standard windowed self-attention

**Purpose**: Control baseline to isolate RWR's effect.

### R1: RWR Default (basic walk)

Enable RWR with default balanced hyperparameters.

**Configuration**:
- α = 0.2 (restart probability)
- T = 4 (walk steps)
- topk = 32 (neighbors per query)
- window = 128 (local attention)
- reversible = False
- reciprocal_beta = 0.5 (unused without reversible)
- lens_strength = 0.3 (γ blending)

**Expected behavior**: LOCAL+RWR factorization reduces full attention cost while capturing long-range dependencies via walk saliency.

### R2: RWR + Reversible (detailed balance)

Add reversible chain symmetrization for stability.

**Configuration**:
- Same as R1 but reversible=True
- Uses P_rev = 0.5*(P + D^{-1}P^T D)

**Expected behavior**: Smoother gradient flow from symmetric walks, potentially better convergence stability.

### R3: RWR Full (reversible + reciprocal + discoverability)

Enable all lens mechanisms: reversible + forward/backward mixing + column bias.

**Configuration**:
- reversible = True
- reciprocal_beta = 0.7 (higher weight on forward saliency)
- use_discoverability = True (learnable per-head column vectors)

**Expected behavior**: Fullest bidirectional information flow, closest to RA lens philosophy.

## Implementation Status

### ✓ Completed

- **lib/graph_builder.py**: Sparse graph construction, P normalization, reversible symmetrization
- **rwr_attention.py**: RWRKernelAttention module with LOCAL+RWR forward pass
- **CLI arguments**: Full hyperparameter control (--rwr-alpha, --rwr-steps, etc.)
- **Ablation steps**: R0-R3 configurations in train_ra_mla.py
- **Step descriptions**: run_test_matrix.py integration
- **R0 validation**: Dry-run passes for baseline control

### ⚠️ TODO (Patching Integration)

RWR attention module exists but is not yet wired into GPT-2 training. Integration requires:

1. **Create patch_gpt2_with_rwr()** function (similar to patch_gpt2_with_lens)
   - Replace GPT2Attention modules with RWRKernelAttention wrappers
   - Handle Q/K/V projection compatibility
   - Manage head_dim padding for tensor cores

2. **Wire into training loop**
   - Check args.use_rwr flag in main()
   - Call patch_gpt2_with_rwr(model, rwr_config)
   - Log RWR statistics (walk steps, graph sparsity, etc.)

3. **Test R1-R3 dry-run**
   - Validate forward/backward passes with RWR
   - Verify gradient flow through sparse graph ops
   - Check memory usage vs standard attention

4. **Performance tuning**
   - Profile sparse matvec cost (T iterations)
   - Optimize graph caching for KV cache reuse
   - Consider FlashAttention integration for LOCAL path

## Usage Examples

### Planned Usage (After Patching Integration)

```bash
# R1: RWR default
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step R1 \
  --dataset finewebedu

# R2: RWR + reversible
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step R2 \
  --dataset finewebedu

# R3: RWR full (reversible + reciprocal)
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step R3 \
  --dataset finewebedu

# Custom RWR configuration
python gpt2/train_ra_mla.py \
  --use-rwr \
  --rwr-alpha 0.15 \
  --rwr-steps 6 \
  --rwr-topk 64 \
  --rwr-window 256 \
  --rwr-reversible \
  --rwr-reciprocal-beta 0.6 \
  --dataset finewebedu
```

### Current Usage (R0 Only)

```bash
# R0: Standard GPT-2 baseline (works now)
python gpt2/train_ra_mla.py \
  --ra-mla-ablation-step R0 \
  --dataset finewebed \
  --dry-run
```

## Hyperparameters

### Core RWR Parameters

- **--rwr-alpha** (restart probability): Balance between local (high α) and global (low α) walk
  - Default: 0.2
  - Range: 0.1-0.5 typical
  - Higher = more restart, shorter walks

- **--rwr-steps** (walk iterations T): Number of matrix powers in Neumann series
  - Default: 4
  - Range: 3-10 typical
  - More steps = longer-range dependencies, higher cost

- **--rwr-topk** (neighbors per query): Sparse attention support
  - Default: 32
  - Range: 16-128 depending on sequence length
  - Trade-off: expressiveness vs memory

- **--rwr-threshold** (similarity cutoff): Minimum edge weight
  - Default: 0.0
  - Range: -0.5 to 0.5 (cosine similarity)
  - Higher = sparser graph, faster but less connected

### Graph Structure Parameters

- **--rwr-window** (local window): Half-width of local attention
  - Default: 128
  - Trade-off: local context vs global via RWR

- **--rwr-block-size** (SRAM tile): Memory tiling size
  - Default: 128
  - Should be multiple of 64 for tensor cores

- **--rwr-head-dim-pad** (padding multiple): Round head_dim up for tensor cores
  - Default: 64
  - Set to 16 for A100, 64 for MI300

### Lens Mechanisms

- **--rwr-reversible**: Enable P_rev for detailed balance
  - Default: False (flag)
  - Recommended: True for stability

- **--rwr-reciprocal-beta** (forward/backward mix): Reciprocal coupling strength
  - Default: 0.5 (equal mixing)
  - Range: 0.0-1.0 (0=backward only, 1=forward only)

- **--rwr-lens-strength** (γ blending): RWR term weight
  - Default: 0.3
  - Range: 0.1-0.5 typical
  - Higher = more RWR influence

- **--rwr-use-discoverability**: Enable learnable column bias
  - Default: False (flag)
  - Adds ~768 params per head (tiny overhead)

## Computational Cost

RWR adds overhead from sparse operations:

- **Graph construction**: One-time per forward pass
  - Tiled cosine similarity: O(N² head_dim / block_size)
  - Sparse edge selection: O(N topk)
  - Amortized with KV cache reuse

- **RWR iterations**: T sparse matvecs
  - Per iteration: O(N × avg_degree)
  - Total: T × sparse_mm cost
  - Default (T=4, topk=32): ~4× denser than local-only

- **Attention fusion**: LOCAL + RWR
  - LOCAL: windowed attention O(N × window)
  - RWR: sparse gather O(N × topk)
  - Combined: typically 20-35% HBM traffic vs full attention

Expected speedup vs dense attention at seq_len=4k: 1.3-1.5× with proper tiling.

## Victory Conditions

A successful RWR implementation should demonstrate:

1. **R1 validation**: Forward/backward passes succeed, gradients flow
2. **Memory efficiency**: <70% HBM traffic vs full attention at 4k sequence
3. **Comparable perplexity**: R1-R3 within 5% of baseline R0 on validation
4. **Ablation clarity**: Understand reversible (R2) vs reciprocal (R3) benefits
5. **Tensor core utilization**: GEMMs running on tensor cores (check nsight)

## References

- **Random Walk with Restart**: Tong et al. (2006) "Fast Random Walk with Restart and Its Applications"
- **Graph-based Attention**: Velickovic et al. (2018) "Graph Attention Networks"
- **FlashAttention**: Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention"
- **Reversible Markov Chains**: Detailed balance from statistical physics / MCMC literature
- **Lens Architecture**: Reciprocal coupling from this project's RA lens work

## Next Steps

1. **Implement patch_gpt2_with_rwr()**: Wire RWR into GPT-2 training
2. **Test R1-R3 dry-run**: Validate all ablation steps pass
3. **Profile performance**: Compare RWR vs standard attention cost
4. **Run ablation study**: Train R0-R3 and analyze validation curves
5. **Optimize sparse ops**: Consider custom CUDA kernels for sparse_mm if needed
6. **Integrate FlashAttention**: Replace LOCAL fallback with real FA2
7. **Document findings**: Update this doc with empirical results

The infrastructure is in place - patching integration is the final step to enable R1-R3 experiments.
