# Claude AI Assistant Preferences

## Git Commit Practices

### Commit Structure
- Make small, atomic commits - one logical change per commit
- Each commit should be functional and not break the build
- Run code formatter (black for Python) after each change
- Run scripts/fix_whitespace_issues.py always on all files
- Test that code runs successfully before committing

### Commit Messages
- Use descriptive commit messages with the format:
  ```
  file.py: brief description of change

  Detailed explanation of what was changed and why.
  Include technical details about the implementation.

  Generated-by: Claude AI
  Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>
  ```

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

## Documentation
- Keep changes well-documented in commit messages
- Explain technical rationale for optimizations
- Include performance impact where applicable
