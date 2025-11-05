# Lens-Gated Architecture Usage Examples

## Basic Usage

```python
from transformers import GPT2LMHeadModel, GPT2Config
from ra_lens_gpt2 import patch_gpt2_with_lens_attention

# Load or create model
config = GPT2Config(n_layer=12, n_embd=768, n_head=12)
model = GPT2LMHeadModel(config)

# Patch with lens-gated attention (all features enabled)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=True,        # Enable S^T mixing
    use_discoverability=True,     # Enable column bias
    use_route_gate=True,          # Enable attention/MLP learning
    mlp_use_ctx_summary=True,     # MLP sees attention output
)

# Train as usual
# ...
```

## Ablation Studies

### 1. Baseline (No Enhancements)

```python
# Standard GPT-2 (no lens-gating)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=False,
    use_discoverability=False,
    use_route_gate=False,
    mlp_use_ctx_summary=False,
)
```

### 2. Reciprocity Only

```python
# Test pure reciprocity (S + S^T mixing)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=True,         # ✓ Enable
    use_discoverability=False,    # ✗ Disable
    use_route_gate=False,
    mlp_use_ctx_summary=False,
)
```

### 3. Discoverability Only

```python
# Test pure discoverability (column bias)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=False,
    use_discoverability=True,     # ✓ Enable
    use_route_gate=False,
    mlp_use_ctx_summary=False,
)
```

### 4. Attention-Only (No MLP)

```python
# Critical ablation: Can attention alone work?
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=True,
    use_discoverability=True,
    use_route_gate=False,  # No need for route gate
    mlp_use_ctx_summary=False,
    mlp_disabled=True,     # ✓ Disable MLP entirely
)
```

### 5. Full Lens-Gated (All Features)

```python
# All enhancements enabled
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=True,
    use_discoverability=True,
    use_route_gate=True,
    mlp_use_ctx_summary=True,
)
```

## Route Gate Annealing (Key Feature!)

### Goal: Learn to shift from attention-heavy to MLP-heavy

```python
from ra_lens_gpt2 import (
    patch_gpt2_with_lens_attention,
    apply_route_annealing,
    get_mean_route_gate,
)

# Patch with annealing config
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_route_gate=True,
    init_route_gate=0.8,  # Start balanced: sigmoid(0.8)≈0.69
    # Annealing schedule embedded in cfg:
    # route_anneal_start: 2000   # Start annealing at step 2000
    # route_anneal_end: 10000    # Finish at step 10000
    # route_anneal_target: -1.0  # Target: sigmoid(-1.0)≈0.27 (MLP-heavy)
)

# Training loop with annealing
for step in range(max_steps):
    # Apply annealing at each step
    apply_route_annealing(model, step, cfg)

    # Forward pass
    loss = model(input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()

    # Log route gate progress
    if step % 100 == 0:
        mean_gate = get_mean_route_gate(model)
        print(f"Step {step}: route_gate={mean_gate:.3f}")
        # Step 0: route_gate=0.689 (attention-heavy)
        # Step 2000: route_gate=0.689 (annealing starts)
        # Step 6000: route_gate=0.500 (halfway)
        # Step 10000: route_gate=0.269 (MLP-heavy, annealing done)
```

### Annealing Schedule Visualization

```
Route Gate Evolution:

Step 0-2000:     g ≈ 0.69 (69% attention, 31% MLP)  [warmup, stable]
                  ┃
                  ┃ Annealing starts
                  ▼
Step 2000-10000: g: 0.69 → 0.27  [linear interpolation]
                  ┃
                  ┃ Gradually shift to MLP
                  ▼
Step 10000+:     g ≈ 0.27 (27% attention, 73% MLP)  [MLP-heavy]

Result: Smaller KV cache needed! (73% savings in attention reliance)
```

## Custom Annealing Strategies

### Strategy 1: Start Balanced, Shift to MLP

```python
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_route_gate=True,
    init_route_gate=0.0,   # sigmoid(0.0)=0.5 (perfectly balanced)
    route_anneal_start=2000,
    route_anneal_end=8000,
    route_anneal_target=-1.5,  # sigmoid(-1.5)≈0.18 (aggressive MLP shift)
)
```

### Strategy 2: Start Attention-Heavy, Gradual Shift

```python
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_route_gate=True,
    init_route_gate=2.2,   # sigmoid(2.2)≈0.90 (traditional, attention-heavy)
    route_anneal_start=5000,  # Later start (more warmup)
    route_anneal_end=15000,
    route_anneal_target=0.0,  # sigmoid(0.0)=0.50 (balanced target)
)
```

### Strategy 3: No Annealing (Let Model Learn)

```python
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_route_gate=True,
    init_route_gate=0.0,   # Start balanced
    route_anneal_start=999999,  # Never anneal
    route_anneal_end=999999,
    # Let route_gate_raw learn freely via gradients
)
```

## MLP Expansion Ratio Experiments

```python
# Standard 4:1 ratio
model, cfg = patch_gpt2_with_lens_attention(
    model,
    mlp_expansion_ratio=4.0,  # 768 → 3072 → 768
)

# Golden ratio 2.5:1 (parameter efficient)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    mlp_expansion_ratio=2.5,  # 768 → 1920 → 768
)

# Larger MLP 5:1 (more MLP capacity)
model, cfg = patch_gpt2_with_lens_attention(
    model,
    mlp_expansion_ratio=5.0,  # 768 → 3840 → 768
)
```

## Monitoring Lens Gates During Training

```python
from ra_lens_gpt2 import analyze_lens_gates, analyze_route_gates

# After training
lens_stats = analyze_lens_gates(model)
print(f"Mean w_std: {lens_stats['mean_w_std']:.3f}")    # Standard attention
print(f"Mean w_rec: {lens_stats['mean_w_rec']:.3f}")    # Reciprocity
print(f"Mean w_disc: {lens_stats['mean_w_disc']:.3f}")  # Discoverability

route_stats = analyze_route_gates(model)
print(f"Mean route gate: {route_stats['mean_route_gate']:.3f}")
print(f"Range: [{route_stats['min_route_gate']:.3f}, {route_stats['max_route_gate']:.3f}]")

# Example output after training:
# Mean w_std: 0.825 (mostly standard attention)
# Mean w_rec: 0.140 (some reciprocity learned)
# Mean w_disc: 0.035 (little discoverability needed)
# Mean route gate: 0.315 (MLP-heavy learned!)
# Range: [0.280, 0.350] (consistent across layers)
```

## Complete Training Example

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Config, AdamW
from ra_lens_gpt2 import (
    patch_gpt2_with_lens_attention,
    apply_route_annealing,
    get_mean_route_gate,
    print_architecture_summary,
)

# 1. Create model
config = GPT2Config(n_layer=12, n_embd=768, n_head=12)
model = GPT2LMHeadModel(config)

# 2. Patch with full lens-gated features
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=True,
    use_discoverability=True,
    use_route_gate=True,
    mlp_use_ctx_summary=True,
    init_route_gate=0.0,  # Start balanced
    route_anneal_start=2000,
    route_anneal_end=10000,
    route_anneal_target=-1.0,  # Target MLP-heavy
)

# 3. Print architecture summary
print_architecture_summary(model)

# 4. Setup optimizer
optimizer = AdamW(model.parameters(), lr=6e-4)

# 5. Training loop
max_steps = 15000
for step in range(max_steps):
    # Apply route gate annealing
    apply_route_annealing(model, step, cfg)

    # Get batch (pseudo-code)
    input_ids, labels = get_batch()

    # Forward + backward
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Logging
    if step % 500 == 0:
        mean_gate = get_mean_route_gate(model)
        print(f"Step {step:5d} | Loss: {loss.item():.4f} | Route: {mean_gate:.3f}")

# 6. Final analysis
print("\n=== Training Complete ===")
print_architecture_summary(model)
```

## Key Insights

### Compute Redistribution (Not Addition!)

```python
# WRONG: Adding 2E input to MLP (expensive!)
# mlp_input = concat([hidden, attn_out])  # [B, T, 2*E]

# RIGHT: Blending context into hidden space (lightweight)
h = fc1(hidden)  # [B, T, mult*E]
if ctx_summary:
    ctx_h = ctx_proj(ctx_summary)  # [B, T, mult*E]
    h = (1-alpha)*h + alpha*ctx_h  # Blend, not concat!
```

### Route Gate Learning Goal

**Goal**: Reduce KV cache size at inference by shifting to MLP.

- Attention requires KV caching (memory heavy)
- MLP requires no caching (memory light)
- Route gate learns: "How much can MLP handle?"
- Result: Smaller cache → better inference latency

### Annealing Strategy

**Why anneal instead of just learning?**

- Early training: Attention is powerful, well-understood
- MLP needs time to learn cross-token patterns
- Annealing gives MLP a "ramp up" period
- By end of training: MLP has learned to compete
- Alternative: Let route_gate_raw learn freely (no annealing)

Both approaches valid! Annealing is more directed, free learning is more exploratory.
