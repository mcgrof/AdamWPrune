# Claude AI Assistant Preferences

## Git Commit Practices

### Commit Structure
- Make small, atomic commits - one logical change per commit
- Each commit should be functional and not break the build
- Run code formatter (black for Python) after each change
- Run scripts/fix_whitespace_issues.py always on all files
- Test that code runs successfully before committing

### Commit Messages
- **MANDATORY**: Always use this exact format for ALL commits:
  ```
  file.py: brief description of change

  Detailed explanation of what was changed and why.
  Include technical details about the implementation.

  Generated-by: Claude AI
  Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>
  ```

- **CRITICAL**: Never use "🤖 Generated with [Claude Code]" or "Co-Authored-By: Claude"
- **REQUIRED**: Every commit MUST have both "Generated-by: Claude AI" and "Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>"
- **NO EXCEPTIONS**: This format is mandatory for ALL commits, no matter how small
- **STYLE**: Be terse and to the point. NO shopping-list style bullet points. Write in paragraphs explaining the change, rationale, and technical details concisely. Avoid verbose enumeration unless absolutely necessary for clarity.

### Development Workflow
1. Make a single focused change
2. Run `black` formatter on Python files
3. Test that the code runs without errors
4. Commit with detailed message
5. Repeat for next change

## Code Style

### Python
- Use `black` formatter for all Python code
- Follow PEP 8 conventions (handled by black)
- No manual formatting - always use black

### Defconfig Files
- **CRITICAL**: Defconfig files must use exact Kconfig syntax: `CONFIG_XXX=y` (no spaces around `=`)
- **DO NOT** apply `black` formatter to defconfig files or `.config` files
- Kconfig parser silently ignores lines with spaces around equals signs
- After any edit to defconfigs, verify syntax: `grep " = " defconfigs/*` should return nothing

## GPU Optimization Preferences

### Training Optimizations
When optimizing PyTorch training for AMD GPUs:
- Increase batch size to utilize GPU memory
- Enable cuDNN benchmark mode
- Use mixed precision training (AMP)
- Add multiple data loader workers with pinned memory
- Include GPU warmup routine
- Use torch.compile() for graph optimization
- Enable TensorFloat32 for matrix operations
- Add comprehensive timing and metrics
- Save trained models after completion

### Performance Monitoring
- Display GPU info at startup
- Show per-epoch timing
- Track test accuracy after each epoch
- Report total training time and average per epoch

## Hardware
- Primary GPU: AMD Radeon Pro W7900 (48GB)
- Optimize for maximum GPU utilization

## Testing Requirements
- Always verify code runs before committing
- Check for linting/formatting issues
- Ensure no syntax errors

## Experiment Workflow

The standard workflow for running experiments:

1. **Load configuration**: `make defconfig-<name>`
   - Example: `make defconfig-gpt2-ratio-ablation`
   - This loads the defconfig and generates config.py

2. **Build and run**: `make`
   - The build system automatically runs the configured experiments
   - For test matrix mode, this runs all ablation steps
   - Results are saved to the configured output directory

3. **Never manually invoke scripts/run_test_matrix.py**
   - The Makefile handles test execution automatically
   - Manual script invocation is for debugging only

Example complete workflow:
```bash
make defconfig-gpt2-ratio-ablation
make
# Results appear in test_matrix_results_ratio_ablation/
```

## Configuration System Internals

### Type Handling
- `.config` files use string values: `"y"`, `"n"`, `"value"`
- `config.py` converts to Python types: `True`, `False`, integers, floats
- When checking config values in Python code, handle both types:
  ```python
  # Good - handles both string and boolean
  if value in ("y", True):

  # Bad - only works with one type
  if value == "y":
  ```

### Test Matrix vs Ablation Mode
- **Mutually exclusive**: Cannot enable both `CONFIG_TEST_MATRIX_MODE` and `CONFIG_RA_MLA_ABLATION_MODE`
- Test matrix mode: Tests optimizer/pruning combinations
- Ablation mode: Tests architectural variations (RA, MLA, RA-CT, etc.)
- Always verify which mode is active when debugging unexpected test counts

## Ablation Study Requirements

### Multi-File Synchronization
When extending ablation studies with new steps, **THREE** files must be updated in sync:

1. **defconfigs/gpt2-ratio-ablation**: Add step descriptions in comments
2. **gpt2/train_ra_mla.py**: Add step configurations (elif step == "N" blocks)
3. **scripts/run_test_matrix.py**: Update `step_descriptions` dictionary

Missing any of these causes:
- Defconfig only: Steps run but have no description
- train_ra_mla.py only: Steps fail to execute
- run_test_matrix.py only: Descriptions show but steps don't run

### Ablation Step Checklist
When adding a new ablation step:
- [ ] Add step config block to train_ra_mla.py (around line 500+)
- [ ] Update step_descriptions dict in run_test_matrix.py (around line 2095)
- [ ] Document step in defconfig comments
- [ ] Update CONFIG_RA_MLA_ABLATION_STEPS string to include new step number
- [ ] Verify with dry-run: `python scripts/run_test_matrix.py --dry-run`

## Defensive Programming

### Assertions for Optional Features
When implementing optional/conditional features that depend on data flow:

- **Always add assertions** to catch when features are enabled but required data is None
- Silent failures are worse than crashes - they waste GPU time and produce invalid results
- Pattern:
  ```python
  if self.cfg.feature_enabled:
      assert required_data is not None, "feature_enabled but no required_data"
  ```

Examples from RA+MLA:
- MLP-to-Attention gating requires `mlp_gate_context` from previous block
- MLP latent reciprocity requires `mlp_latent_context` from previous block
- Cross-token MLP requires `attn_weights` from attention layer

### Context Flow for Multi-Block Architectures
When implementing bidirectional information flow between transformer blocks:

- Use wrapper classes (e.g., `RA_MLA_Block`) to manage context state across blocks
- Store contexts in instance variable (e.g., `self._ctx = {}`)
- Pass contexts as keyword arguments (enables detection of missing connections)
- Produce contexts for the **next** block at the end of forward pass
- Never assume contexts exist - always check with assertions when used

## Architectural Pattern Guidelines

### Feature Independence and Composability
When adding new attention/MLP mechanisms:

- **Keep features orthogonal**: RA-CT (attention-only gating) vs MLP mechanisms (cross-layer flow)
- **Use clear naming**: `ra_cross_token` for attention features, `mlp_attn_gate` for MLP features
- **Enable ablation**: Each feature should be independently testable
- **Avoid coupling**: RA-CT doesn't require MLA/RA, can be tested on baseline GPT-2

### Per-Head Learnable Parameters
For per-head gating mechanisms:

- Initialize to near-identity: `bias ≈ 2.0` for sigmoid gates (pass-through initially)
- Use affine transforms: `sigmoid(stat * scale + bias)` for numerical stability
- Shape: `[n_head]` for per-head parameters, expandable to `[B,H,T]` when needed
- Consider `head_average=True` option for cheaper computation

### Statistics-Based Gating
When implementing gating based on attention statistics:

- Support multiple modes: `topk`, `max`, `entropy`, `rms`
- Provide `detach_stats` option to compute under `no_grad()` for memory savings
- Apply gate at multiple points: `weights` (pre-softmax) or `output` (post-aggregation)
- Use `alpha` mixing parameter for smooth interpolation: `(1-α)·x + α·(x⊙gate)`

## GPU Memory Management

### Memory Optimization Strategies
- Enable expandable segments: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`
- Disable expensive metrics logging during training (e.g., entropy computation on attention weights)
- Use `@torch.no_grad()` for statistics computation that doesn't need gradients
- Monitor for OOM errors in attention mechanisms - often caused by extra allocations for metrics
- Batch size × gradient accumulation = effective batch size (keep constant when adjusting for memory)

### A10G-Specific Considerations
- 24GB VRAM per GPU requires careful batch size tuning
- For GPT-2 124M with RA+MLA: batch_size=8, gradient_accumulation=8 (effective=64)
- Tensor dimensions should be multiples of 64 for optimal tensor core utilization
- Disable metrics logging for attention mechanisms to prevent OOM during entropy computation

## Documentation
- Keep changes well-documented in commit messages
- Explain technical rationale for optimizations
- Include performance impact where applicable

## Avoid silly language

You are not allowed to use the word "comprehensive". It is overused
and does not explain anything. We prefer to be terse and to the point.

# Memory

I want you to remember most of our conversations about this project.
