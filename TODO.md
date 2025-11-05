# TODO List - Reciprocal Attention and MLP Enhancements

## Completed

- [x] Fix per_head_scalar zeros bug in ReciprocalCoupler
  - Was returning zeros instead of actual projections
  - Now uses linear layers with per-head scalar gating
- [x] Make mixing parameters learnable
  - `ra_alpha`: Reciprocal attention mixing weight
  - `mlp_gate_alpha`: MLP-to-attention gating mixing weight
  - `mlp_cross_alpha`: Cross-token MLP aggregation mixing weight
  - `mlp_recip_alpha_mlp`: Attention→MLP enrichment mixing weight
  - `mlp_recip_alpha_attn`: MLP→Attention context mixing weight
  - All initialized at config defaults, now learnable during training
- [x] Add positive bias initialization to gate layers
  - `gate_to_heads` bias initialized to 2.0
  - sigmoid(2.0) ≈ 0.88, so gates start mostly open
  - Allows gradual learning of gating behavior
- [x] Document parameter tying modes in ReciprocalCoupler
  - Added detailed docstring explaining untied, tied_transpose, per_head_scalar
  - Included parameter counts and tradeoffs

## Immediate Actions Required

- [ ] Run code formatter on modified files
  ```bash
  black gpt2/ra_mla_gpt2.py
  python3 scripts/fix_whitespace_issues.py gpt2/ra_mla_gpt2.py
  ```

- [ ] Validate changes with dry-run
  ```bash
  make check
  ```

- [ ] Commit changes with detailed message
  - Explain per_head_scalar bug fix
  - Explain learnable mixing parameters rationale
  - Explain positive bias initialization for gates
  - Include testing requirements

- [ ] Sync code to remote machine for failed tests
  ```bash
  # On remote: /opt/dlami/nvme/AdamWPrune
  git pull
  ```

- [ ] Re-run failed reciprocal MLP tests
  ```bash
  make defconfig-gpt2-ratio-ablation-reciprocal
  make
  ```

## Items to Evaluate

### Gradient Norm Logging

**Status**: Needs evaluation - requires training loop modifications

Add logging of gradient norms for key learnable parameters to diagnose if mechanisms receive meaningful gradients:
- `ra_alpha` (reciprocal attention)
- `mlp_gate_alpha` (MLP gating)
- `mlp_cross_alpha` (cross-token)
- `mlp_recip_alpha_*` (bidirectional reciprocity)
- `gate_proj`, `gate_to_heads` (gating projections)
- `q_to_latent` (query-to-latent projection)

**Questions to answer**:
- Do these parameters receive non-vanishing gradients?
- Are gradients meaningful (not near-zero)?
- Should this be always-on or conditional logging?
- Should we log to wandb/trackio or just stderr?
- Performance impact of gradient norm computation?

**Implementation approach**:
- Add gradient logging hooks in training loop
- Conditional on flag to avoid overhead
- Log every N iterations, not every step
- Use `param.grad.norm()` after backward pass

### Learnable MLP Expansion Ratio

**Status**: Needs evaluation - architectural complexity

Make MLP expansion ratio (e.g., 4.0 for standard, 5.0 for golden ratio) a learnable parameter instead of fixed.

**Potential approaches**:
1. **Continuous interpolation**: Maintain multiple MLP layers at different sizes, interpolate outputs
2. **Discrete gating**: Keep current approach but learn which "width" to use per layer
3. **Parameter masking**: Single large MLP with learnable masks to effectively reduce width

**Questions to answer**:
- Is this worth the complexity vs just testing multiple fixed ratios?
- Would golden ratio (1:2.5) emerge naturally from learning?
- How to handle pretrained weight initialization if ratio changes?
- Memory implications of multiple MLP sizes?
- Does ratio need to be per-layer or global?

**Current observation**:
- Step 2 (golden ratio 1:2.5) had WORSE validation loss than baseline
- Suggests fixed golden ratio may not be optimal for this task
- Or needs more training iterations to converge

### Ablation: Gates Forced to 1.0

**Status**: Simple to implement - evaluate if worthwhile

Add ablation steps that test mechanisms with gates permanently set to 1.0 (fully open, no modulation).

**Purpose**:
- Isolate whether gate learning helps or hurts
- Test if mechanisms are useful without adaptive gating
- Current gates may learn to close/attenuate signal

**Implementation**:
- Add config flag `FORCE_GATES_OPEN=y`
- In code: replace sigmoid gates with constant 1.0
- Run same steps (3,4,7,10,12,16) with forced gates
- Compare to learned gating version

**New ablation steps** (if implemented):
- Step 19: Step 3 with gates=1.0 (golden ratio + gating, no learning)
- Step 20: Step 4 with gates=1.0 (+ cross-token, no learning)
- Step 21: Step 7 with gates=1.0 (RA + mechanisms, no learning)
- Step 22: Step 10 with gates=1.0 (MLA + mechanisms, no learning)
- Step 23: Step 12 with gates=1.0 (RA+MLA + mechanisms, no learning)
- Step 24: Step 16 with gates=1.0 (Full RATIO, no learning)

**Evaluation needed**:
- Is this worth 6 more training runs?
- Or just test on 1-2 representative steps?
- Requires modifying train_ra_mla.py to add --force-gates-open flag

## Current Test Status

### Completed Tests (test_matrix_results_20251103_153403)
- Step 0 (Baseline): val_loss = 3.5740
- Step 1 (+ SPAM pruning): val_loss = 3.5716 (↓ 0.07% - better!)
- Step 2 (Golden ratio): val_loss = 3.6238 (↑ 1.39% - worse)
- Step 5 (+ RA): val_loss = 3.6367 (↑ 1.75% - worse)
- Step 6 (RA + golden ratio): Currently running

### Failed Tests (outdated remote code - need re-run)
- Step 3: Golden ratio + MLP gating
- Step 4: Step 3 + cross-token
- Step 7: RA + golden ratio + mechanisms
- Step 10: MLA + golden ratio + mechanisms
- Step 12: RA+MLA + golden ratio + mechanisms
- Step 16: Full RATIO (all mechanisms)

**Failure cause**: Assertion error from commit fb765bd (already fixed in commit 3c40bb2)

**Resolution**: Git pull on remote machine, run defconfig-gpt2-ratio-ablation-reciprocal

## Observations and Questions

### Golden Ratio Performance
- Step 2 (25% more MLP parameters) performed WORSE than baseline
- Possible causes:
  - Undertrained (10K iters insufficient for larger model)
  - Bad weight initialization (copying pretrained + random for new dims)
  - Golden ratio assumption doesn't hold for this task/dataset

### RA Performance
- Step 5 (reciprocal attention) performed WORSE than baseline
- Fixed ra_alpha=0.3 may be suboptimal
- Now learnable - will model learn to adjust or disable?

### Mechanism Effectiveness
- Neither golden ratio nor RA helped in isolation
- Question: Will combinations help?
- Question: Will learnable mixing help? (gates can learn to close if harmful)

### Next Analysis
After failed tests complete:
- Compare mechanisms ON vs OFF
- Compare learned mixing weights vs fixed defaults
- Track mixing weight evolution during training (wandb)
- Determine if mechanisms need longer training or are fundamentally unhelpful
