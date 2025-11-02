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

## Documentation
- Keep changes well-documented in commit messages
- Explain technical rationale for optimizations
- Include performance impact where applicable

## Avoid silly language

You are not allowed to use the word "comprehensive". It is overused
and does not explain anything. We prefer to be terse and to the point.

# Memory

I want you to remember most of our conversations about this project.
