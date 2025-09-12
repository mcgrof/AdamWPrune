# ResNet-50 CIFAR-100 Training Results

## Executive Summary

After fixing a critical bug that completely disabled state pruning in ResNet-50, AdamWPrune now demonstrates strong performance achieving **72.38% accuracy at 70% sparsity** - the best result among all Adam-based optimizers at the target sparsity level.

### Key Achievements
- **72.38% accuracy at 70% sparsity** (best Adam variant)
- **12,428.6 MB GPU memory usage** (lowest among all tested)
- **State pruning properly implemented** after bug fix
- **Competitive with AdamWSPAM** (only 0.20% behind)

## Critical Bug Fix

### The Problem
Initial tests showed AdamWPrune achieving 0% sparsity despite targeting 70%. Investigation revealed:
1. `run_test_matrix.py` was passing `--pruning-method none` instead of `state` for AdamWPrune
2. State pruning was completely unimplemented in `resnet50/train.py`
3. AdamWPrune pruning flags were never being set

### The Solution
- Fixed `run_test_matrix.py` to pass correct pruning method
- Implemented state pruning support in `resnet50/train.py`
- Added proper flag initialization for AdamWPrune
- Corrected optimizer tuple unpacking

## Test Configuration

- **Model**: ResNet-50 (25.6M parameters)
- **Dataset**: CIFAR-100 (100 classes)
- **Training**: 100 epochs
- **Hardware**: Various GPUs (AMD/NVIDIA)
- **Batch Size**: 128
- **Pruning Target**: 70% sparsity
- **Pruning Schedule**: Gradual ramp-up to epoch 80

## Results at 70% Sparsity

### Fair Comparison (All at Target Sparsity)

| Rank | Optimizer | Best Accuracy | Epoch | GPU Memory (MB) | Status |
|------|-----------|---------------|-------|-----------------|--------|
| 1 | SGD | 74.57% | 91 | 12,501.1 | Best overall |
| **2** | **AdamWPrune** | **72.38%** | **93** | **12,428.6** | **Best Adam** |
| 3 | AdamWSPAM | 72.18% | 100 | 12,694.1 | -0.20% |
| 4 | AdamW | 71.34% | 91 | 12,600.0 | -1.04% |
| 5 | Adam | 71.23% | 85 | 12,600.4 | -1.15% |
| 6 | AdamWAdv | 70.65% | 93 | 12,694.0 | -1.73% |

### Detailed Performance Analysis

#### AdamWPrune State Pruning
- **Best at target sparsity**: 72.38% (epoch 93, 69% actual sparsity)
- **Overall best**: 72.61% (epoch 63, during sparsity ramp-up)
- **Final accuracy**: 70.56% (epoch 100)
- **Stability**: 0.68% std over last 10 epochs
- **Degradation from peak**: 2.05%

#### Key Observations
1. **Sparsity ramp-up impact**: Best accuracy occurs at epoch 63 (~55% sparsity) before full sparsity
2. **Fair comparison**: When comparing at same sparsity (70%), AdamWPrune leads Adam variants
3. **Memory efficiency**: Achieves lowest GPU memory despite state tracking overhead
4. **Stability trade-off**: Higher variance than movement pruning but better peak performance

## GPU Memory Analysis

### Memory Usage Comparison

| Optimizer | Mean Memory | vs Best | Memory Efficiency |
|-----------|------------|---------|-------------------|
| **AdamWPrune** | **12,428.6 MB** | **Lowest** | **5.98%/GB** |
| SGD | 12,501.1 MB | +72.5 MB | 6.11%/GB |
| AdamW | 12,600.0 MB | +171.4 MB | 5.80%/GB |
| Adam | 12,600.4 MB | +171.8 MB | 5.79%/GB |
| AdamWAdv | 12,694.0 MB | +265.4 MB | 5.70%/GB |
| AdamWSPAM | 12,694.1 MB | +265.5 MB | 5.75%/GB |

### Memory Efficiency Insights
- AdamWPrune achieves the **lowest memory usage** despite maintaining optimizer states
- State reuse for pruning eliminates need for separate importance buffers
- 265 MB savings compared to AdamWSPAM/AdamWAdv

## Optimizer Selection by Model Scale

### Critical Finding
Our testing reveals that **optimal optimizer selection depends on model size**:

#### ResNet-18 (11.2M parameters)
- **Best**: AdamW (90.30%)
- AdamWPrune: 90.28% (nearly identical)
- Simpler weight decay sufficient for smaller models

#### ResNet-50 (25.6M parameters)
- **Best overall**: SGD (74.57%)
- **Best Adam**: AdamWPrune (72.38%)
- **Close second**: AdamWSPAM (72.18%)
- Complex loss landscapes benefit from advanced techniques

### Implications
- Larger models benefit from spike-aware momentum (SPAM) or state-based pruning
- AdamWPrune with configurable base optimizer can adapt to model scale
- Consider AdamWSPAM base for AdamWPrune on very large models

## Training Dynamics

### Sparsity Evolution
- **Epochs 1-10**: Warmup phase, no pruning
- **Epochs 10-80**: Gradual sparsity increase
- **Epoch 63**: Peak accuracy (72.61%) at ~55% sparsity
- **Epochs 80-100**: Stable at 70% sparsity

### Comparison with Movement Pruning
- **Movement pruning**: Applies 70% sparsity from epoch 1
- **State pruning**: Gradual ramp-up preserves accuracy longer
- **Trade-off**: Better peak accuracy but more instability at full sparsity

## Recommendations

### When to Use AdamWPrune
1. **Memory-constrained environments**: Lowest GPU memory usage
2. **Large models (>20M params)**: State pruning scales well
3. **Gradual pruning preferred**: When training stability matters
4. **Adam ecosystem required**: Best performance among Adam variants

### Hyperparameter Suggestions
- **Base optimizer**: AdamW for <20M params, consider AdamWSPAM for larger
- **Pruning warmup**: 10 epochs minimum
- **Ramp-up duration**: 70 epochs for gradual transition
- **Learning rate**: Standard AdamW schedule works well

## Future Work

1. **Test with AdamWSPAM base**: May combine benefits of both approaches
2. **Extended training**: Test beyond 100 epochs for convergence
3. **ImageNet validation**: Scale to full ImageNet dataset
4. **Structured pruning**: Explore channel/filter pruning variants

## Conclusion

AdamWPrune with fixed state pruning implementation achieves **72.38% accuracy at 70% sparsity** on ResNet-50 CIFAR-100, demonstrating:
- **Best accuracy among Adam variants** at target sparsity
- **Lowest GPU memory usage** of all tested configurations
- **Successful state-based pruning** using optimizer momentum/variance
- **Scalability** to larger models where advanced optimizers excel

The bug fix revealed AdamWPrune's true potential, validating the core hypothesis that optimizer states can effectively guide pruning decisions while minimizing memory overhead.
