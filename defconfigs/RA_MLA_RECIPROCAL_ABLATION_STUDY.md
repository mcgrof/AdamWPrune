# RA+MLA Reciprocal MLP Ablation Study

This directory contains defconfigs for a systematic ablation study of the RA+MLA+Reciprocal MLP architecture.

## Experiment Overview

The ablation study progressively enables reciprocal MLP mechanisms to isolate their individual contributions. All experiments share the same baseline configuration (MLA with latent_dim=128, RA alpha=0.0) to ensure fair comparison.

## Defconfigs

### Baseline
**File**: `gpt2-ra-mla-baseline`
**Description**: Pure MLA baseline (no RA, no reciprocal MLP)
- MLA latent compression: latent_dim=128 (6× compression)
- RA disabled: alpha=0.0
- Reciprocal MLP: All mechanisms disabled
- Optimizer: AdamWSPAM
- **Expected**: val_loss ~3.61, ppl ~37.2 (36% better than vanilla)

### Step 1: Mechanism 1 Only (MLP-to-Attention Gating)
**File**: `gpt2-ra-mla-reciprocal-step1`
**Description**: Enable mechanism 1: MLP gates attention heads
- Baseline: MLA latent_dim=128, RA alpha=0.0
- **New**: CONFIG_RA_MLA_MLP_ATTN_GATE=y
  - gate_alpha=0.1
  - gate_dim=64
- Mechanisms 2 & 3: Disabled
- Optimizer: AdamWSPAM
- **Expected**: val_loss ~3.56-3.59 (-0.02 to -0.05 improvement)
- **Hypothesis**: MLP learns to steer attention focus based on learned features

### Step 2: Mechanisms 1+2 (+ Cross-Token MLP Aggregation)
**File**: `gpt2-ra-mla-reciprocal-step2`
**Description**: Enable mechanisms 1 and 2
- Baseline: MLA latent_dim=128, RA alpha=0.0
- **Mechanism 1**: CONFIG_RA_MLA_MLP_ATTN_GATE=y
- **Mechanism 2**: CONFIG_RA_MLA_MLP_CROSS_TOKEN=y
  - cross_alpha=0.3
- Mechanism 3: Disabled
- Optimizer: AdamWSPAM
- **Expected**: val_loss ~3.51-3.56 (-0.05 to -0.10 improvement over baseline)
- **Hypothesis**: Cross-token MLP aggregation provides strongest contribution (most parameter-efficient way to get global context)

### Step 3: Full Reciprocal MLP (All Three Mechanisms)
**File**: `gpt2-ra-mla-reciprocal-step3`
**Description**: Enable all three mechanisms
- Baseline: MLA latent_dim=128, RA alpha=0.0
- **Mechanism 1**: CONFIG_RA_MLA_MLP_ATTN_GATE=y
- **Mechanism 2**: CONFIG_RA_MLA_MLP_CROSS_TOKEN=y
- **Mechanism 3**: CONFIG_RA_MLA_MLP_LATENT_RECIP=y
  - recip_alpha=0.2
  - mlp_latent_dim=128
- Optimizer: AdamWSPAM
- **Expected**: val_loss ~3.46-3.51 (-0.10 to -0.15 improvement over baseline)
- **Hypothesis**: Full bidirectional information flow throughout network

### Step 4: Mechanisms 1+2 with AdamWSPAM (Sanity Check)
**File**: `gpt2-ra-mla-reciprocal-step4-adamwspam`
**Description**: Test mechanisms 1+2 without mechanism 3, confirming AdamWSPAM compatibility
- Baseline: MLA latent_dim=128, RA alpha=0.0
- **Mechanism 1**: CONFIG_RA_MLA_MLP_ATTN_GATE=y
- **Mechanism 2**: CONFIG_RA_MLA_MLP_CROSS_TOKEN=y
- Mechanism 3: Disabled
- Optimizer: AdamWSPAM (explicit)
- **Expected**: Should match step2 results (val_loss ~3.51-3.56)
- **Purpose**: Verify that mechanism 2 (cross-token aggregation) works well with SPAM optimizer

### Step 5: Full Solution with AdamWSPAM
**File**: `gpt2-ra-mla-reciprocal-step5-full`
**Description**: Complete reciprocal MLP implementation
- Baseline: MLA latent_dim=128, RA alpha=0.0
- **All three mechanisms enabled**
- Optimizer: AdamWSPAM (optimal for reciprocal MLP)
- **Expected**: val_loss ~3.40-3.50 (-0.11 to -0.21 improvement over baseline)
- **Target**: Approach or beat val_loss 3.40 (ppl ~30), demonstrating 45% improvement over vanilla GPT-2

## Running the Ablation Study

### Sequential Execution (Recommended)
```bash
# Step 0: Baseline (already exists)
make KCONFIG_CONFIG=defconfigs/gpt2-ra-mla-baseline

# Step 1: Mechanism 1 only
make KCONFIG_CONFIG=defconfigs/gpt2-ra-mla-reciprocal-step1

# Step 2: Mechanisms 1+2
make KCONFIG_CONFIG=defconfigs/gpt2-ra-mla-reciprocal-step2

# Step 3: Full reciprocal MLP (all three)
make KCONFIG_CONFIG=defconfigs/gpt2-ra-mla-reciprocal-step3

# Step 4: Mechanisms 1+2 with AdamWSPAM (sanity check)
make KCONFIG_CONFIG=defconfigs/gpt2-ra-mla-reciprocal-step4-adamwspam

# Step 5: Full solution with AdamWSPAM
make KCONFIG_CONFIG=defconfigs/gpt2-ra-mla-reciprocal-step5-full
```

### Batch Execution (Parallel)
```bash
# Run all ablations in parallel (requires multiple GPUs or sequential queueing)
for config in gpt2-ra-mla-baseline \
              gpt2-ra-mla-reciprocal-step1 \
              gpt2-ra-mla-reciprocal-step2 \
              gpt2-ra-mla-reciprocal-step3 \
              gpt2-ra-mla-reciprocal-step4-adamwspam \
              gpt2-ra-mla-reciprocal-step5-full; do
    KCONFIG_CONFIG=defconfigs/$config make &
done
wait
```

## Expected Results Summary

| Config | Mechanisms | Expected val_loss | Expected ppl | Improvement over baseline |
|--------|-----------|------------------|--------------|---------------------------|
| **Baseline** | None | 3.6154 | 37.17 | - (baseline) |
| **Step 1** | 1 only | 3.56-3.59 | 35.2-36.3 | -0.02 to -0.05 |
| **Step 2** | 1+2 | 3.51-3.56 | 33.4-35.2 | -0.05 to -0.10 |
| **Step 3** | 1+2+3 | 3.46-3.51 | 31.8-33.4 | -0.10 to -0.15 |
| **Step 4** | 1+2 (SPAM) | 3.51-3.56 | 33.4-35.2 | (same as step 2) |
| **Step 5** | 1+2+3 (SPAM) | 3.40-3.50 | 30.0-33.1 | -0.11 to -0.21 |

For reference:
- **Vanilla GPT-2**: val_loss 4.0655, ppl 58.29
- **MLA baseline improvement**: 36% better than vanilla
- **Target with full reciprocal MLP**: 45-48% better than vanilla

## Key Hypotheses

1. **Mechanism 2 is strongest**: Cross-token MLP aggregation provides the most improvement per parameter because it enables global context at linear cost (reusing attention weights).

2. **Mechanisms are additive**: Each mechanism contributes independently, with cumulative benefits when combined.

3. **AdamWSPAM is optimal**: The SPAM optimizer's spike detection and periodic reset work well with reciprocal MLP's bidirectional information flow.

4. **Mechanism 3 adds polish**: MLP latent space reciprocity provides smaller but consistent improvement by enabling latent-space information exchange.

## Analysis After Running

After completing the ablation study, analyze:

1. **Individual mechanism contributions**:
   - Mechanism 1: step1 - baseline
   - Mechanism 2: step2 - step1
   - Mechanism 3: step3 - step2

2. **Optimizer interaction**:
   - Compare step2 vs step4 (should be identical)
   - Compare step3 vs step5 (should be similar, step5 may be slightly better)

3. **Cumulative effect**:
   - Plot val_loss across steps
   - Check if improvements are monotonic
   - Verify cumulative benefit matches prediction (-0.11 to -0.21)

4. **Attention patterns**:
   - Examine attention entropy across steps
   - Check reciprocity scores (if logged)
   - Verify MLP gating effectiveness

## Troubleshooting

**If step1 is worse than baseline:**
- Check that mechanism 1 is properly initialized
- Verify gate_alpha is small enough (0.1 should be safe)
- May need lower initial alpha for stability

**If step2 doesn't improve over step1:**
- Verify attention weights are being reused correctly
- Check cross_alpha tuning (0.3 may be too high/low)
- Ensure cross-token aggregation is not creating gradient flow issues

**If step3 is worse than step2:**
- Mechanism 3 may interfere with mechanisms 1+2
- Check latent dimension compatibility
- May need to tune recip_alpha lower (0.1 instead of 0.2)

**If step5 doesn't improve over step3:**
- AdamWSPAM may not provide additional benefit
- Verify SPAM configuration is correct
- Check if SPAM's periodic reset is disrupting reciprocal pathways

## Next Steps After Ablation

1. **If results are promising (val_loss < 3.50)**:
   - Run longer training (20K-30K iterations)
   - Test with aggressive latent compression (latent_dim=64 or 32)
   - Combine with bitter-scale pruning

2. **If results are mixed (val_loss 3.50-3.60)**:
   - Tune alpha parameters
   - Try different latent dimensions
   - Experiment with layer-specific mechanism enabling

3. **If results are disappointing (val_loss > 3.60)**:
   - Analyze failure modes
   - Check gradient flow and attention patterns
   - Consider architectural modifications

## References

- `docs/ra.md`: Full documentation of RA+MLA+Reciprocal MLP
- `Kconfig.ra_mla`: Configuration options and help text
- `gpt2/ra_mla_gpt2.py`: Core implementation
- `gpt2/train_ra_mla.py`: Training script

## Contact

For questions or issues with the ablation study, refer to the main project documentation or open an issue.
