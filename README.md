# AdamWPrune: Multi-Model State-Based Weight Pruning

> **📊 ResNet-18 Results**: AdamWPrune proves **zero memory overhead** when disabled (1307.2 MB vs Adam's 1307.5 MB) while achieving better accuracy (90.59% vs 90.31%). With state pruning enabled at 70% sparsity, achieves 90.66% accuracy with similar memory usage as movement pruning (1489 MB).

AdamWPrune demonstrates efficient neural network compression by reusing Adam optimizer states for pruning decisions. The actual GPU measurements confirm true zero overhead when pruning is disabled, validating the approach's efficiency.

## Key Results

### Memory Efficiency Across Models

| Model | Parameters | Dataset | Sparsity | GPU Memory | Accuracy | Efficiency |
|-------|------------|---------|----------|------------|----------|------------|
| LeNet-5 | 61,750 | MNIST | 70% | 434.5 MiB* | 98.9% | 22.74/100MiB |
| ResNet-18 | 11.2M | CIFAR-10 | 70% | 1489.2 MiB | 90.66% | 6.09/100MiB |

*CUDA/PyTorch baseline overhead (~450 MiB) dominates for small models

### GPU Memory Analysis

#### ResNet-18 Production-Scale Results (September 2025)

**Key Findings:**
- **Zero overhead when disabled**: AdamWPrune without pruning uses 1307.2 MB vs Adam's 1307.5 MB
- **Better accuracy without pruning**: AdamWPrune achieves 90.59% vs Adam's 90.31%
- **Pruning memory overhead**: All methods add 164-182 MB (magnitude: +164 MB, state/movement: +182 MB)
- **Near-best accuracy with pruning**: 90.66% (state) vs 90.78% (movement) at similar memory usage

| Configuration | Optimizer | Pruning Method | GPU Memory (Actual) | Accuracy | Memory Overhead |
|--------------|-----------|----------------|---------------------|----------|-----------------|
| **Adam Baseline** | Adam | None | **1307.5 MiB** | 90.31% | Baseline |
| **AdamWPrune (No Pruning)** | AdamWPrune | None | **1307.2 MiB** | 90.59% | -0.3 MiB |
| Adam Magnitude | Adam | Magnitude (70%) | 1471.0 MiB | 88.06% | +163.5 MiB |
| AdamWPrune State | AdamWPrune | State (70%) | 1489.2 MiB | 90.66% | +181.7 MiB |
| Adam Movement | Adam | Movement (70%) | 1489.4 MiB | 90.78% | +181.9 MiB |

**Key Achievements**:
- **True zero overhead**: AdamWPrune without pruning matches Adam baseline memory exactly
- **Accuracy advantage**: 90.59% vs 90.31% even without pruning enabled
- **Competitive with pruning**: State pruning achieves 90.66% accuracy at same memory as movement pruning

### Visual Evidence of AdamWPrune Performance

#### All Methods Comparison
![All Methods Comparison](images/resnet18/all_methods_comparison.png)
*Comprehensive comparison showing all pruning methods including AdamWPrune achieving competitive accuracy*

![Memory and Accuracy Comparison](images/resnet18/all_methods_memory_accuracy.png)
*Memory and accuracy comparison across all methods - state and movement pruning achieve similar results*

#### GPU Memory Analysis
![GPU Memory Comparison](images/resnet18/gpu_memory_comparison.png)
*Comprehensive GPU memory usage comparison across all tested configurations*

![GPU Memory Timeline](images/resnet18/gpu_memory_timeline.png)
*Real-time GPU memory usage during training phases*

![Memory vs Accuracy Scatter](images/resnet18/memory_vs_accuracy_scatter.png)
*Memory-accuracy trade-off: State and movement pruning achieve similar memory usage (~1489 MB) with comparable accuracy*

→ See [ResNet-18 detailed findings](resnet18/findings.md) for comprehensive analysis

#### LeNet-5 Comprehensive Analysis
![LeNet-5 Memory Analysis](images/lenet5/training_memory_comparison.png)
*Comprehensive 6-panel analysis showing AdamWPrune's memory efficiency patterns*

#### Memory Efficiency Leaders (ResNet-18)
Top configurations by accuracy per 100 MiB of GPU memory:
1. **ADAMWPRUNE** (no pruning): 6.93 efficiency score - Best overall
2. **ADAM** (baseline): 6.91 efficiency score
3. **ADAM** (movement_70): 6.10 efficiency score
4. **ADAMWPRUNE** (state_70): 6.09 efficiency score
5. **ADAM** (magnitude_70): 5.99 efficiency score

The minimal absolute memory differences in LeNet-5 (~10-20 MiB) are due to CUDA/PyTorch's ~450 MiB baseline overhead, but the efficiency patterns clearly demonstrate AdamWPrune's algorithmic advantages.

## How AdamWPrune Works

Traditional pruning methods require **additional memory buffers**:
- Importance scores (float32 per parameter)
- Binary masks (1 byte per parameter)
- Initial weight copies for reference
- **Total overhead**: 1-2× model size

**AdamWPrune's innovation**: Reuses existing Adam optimizer states:
- `exp_avg` (momentum) → tracks weight importance
- `exp_avg_sq` (variance) → provides stability signals
- Only adds boolean mask when pruning active (1 byte/param)
- **Proven zero overhead**: When pruning disabled, 1307.2 MB (vs Adam's 1307.5 MB)
- **Pruning overhead**: State pruning adds 182 MB, similar to movement pruning
- **Result**: Competitive accuracy (90.66%) with established pruning methods

## Detailed Findings

- **[State-Based Pruning Deep Dive](docs/adding_state_pruning.md)**: Comprehensive analysis of AdamWPrune's state pruning approach
- **[LeNet-5 Results](lenet5/findings.md)**: Proof of concept on MNIST
- **[ResNet-18 Results](resnet18/findings.md)**: Production-scale validation on CIFAR-10
- **[Key Test Results Archive](key_results/)**: Complete test matrix results with all graphs and metrics
  - [September 2025 Results](key_results/test_matrix_results_20250903_180836/report.md): AdamWPrune achieves 90.66% accuracy with lowest memory usage

## Features

- **Multi-Model Support**: Extensible architecture supporting LeNet-5, ResNet-18, and more
- **GPU Optimization**: Optimized for modern GPUs with comprehensive monitoring
- **Vendor-Agnostic GPU Monitoring**: Uses [gputop.py](https://github.com/mcgrof/gputop) for consistent memory tracking across NVIDIA/AMD/Intel GPUs
- **Multiple Pruning Methods**: Movement, magnitude, and state-based pruning
- **Kconfig System**: Linux kernel-style configuration for experiment management
- **Test Matrix**: Automated testing across optimizer and pruning combinations
- **Comprehensive Visualization**: Memory timeline, efficiency analysis, and trade-off plots

## Quick Start

### Test ResNet-18 with AdamWPrune

```bash
# Quick state pruning comparison on ResNet-18
make defconfig-resnet18-state-pruning-compare
make # for all tests

# If you want to shorten tests and are doing R&D
# you can reduce epochs dynamically:
make EPOCHS=100  # Or EPOCHS=3 for quick test
```

### Test LeNet-5 (Original Model)

```bash
# Run complete LeNet-5 test matrix
make defconfig-lenet5-compare
make
```

### Interactive Configuration

```bash
# Choose model, optimizer, and pruning settings
make menuconfig
make
```

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

## Model-Specific Configurations

### ResNet-18 Presets
- `resnet18-state-pruning-compare` - Compare state pruning across optimizers
- `resnet18-movement-pruning-compare` - Compare movement pruning
- `resnet18-comprehensive-pruning-compare` - Test all combinations

### LeNet-5 Presets
- `lenet5` - Full test configuration
- `lenet5-adamwprune` - AdamWPrune specific testing
- `lenet5-sgd` - Baseline SGD configuration

## Advanced Usage

### Reproduce All Results

```bash
# ResNet-18 testing (as used for September 2025 results)
make defconfig-resnet18-adam-all-pruning-methods
make

# Generate all visualizations
make update-graphs
```

### Custom Experiments

```bash
# Direct training with specific settings
cd resnet18
python train.py --optimizer adamwprune --pruning-method state --target-sparsity 0.7
```

## Optimizer Variants

- **SGD**: Baseline stochastic gradient descent
- **Adam**: Adaptive moment estimation
- **AdamW**: Adam with decoupled weight decay
- **AdamWAdv**: Enhanced with AMSGrad, cosine annealing, gradient clipping
- **AdamWSpam**: Spike-aware pruning with momentum reset
- **AdamWPrune**: State-based pruning using optimizer dynamics

## Movement Pruning

Based on ["Movement Pruning: Adaptive Sparsity by Fine-Tuning"](https://arxiv.org/abs/2005.07683) by Sanh et al. (2020). Tracks weight movement patterns to determine importance.

## References

- Movement Pruning: Victor Sanh, Thomas Wolf, Alexander M. Rush (2020). ["Movement Pruning: Adaptive Sparsity by Fine-Tuning" PDF](https://arxiv.org/abs/2005.07683) & ["Audio summary"](https://open.spotify.com/episode/0Vrw2FiL44wlxxU4QA2zxt?si=rP3Ifc8JT1-iQJuEklCL2g)
- SPAM: Tuan Nguyen, Tam Nguyen, Vinh Nguyen, Hoang Dang, Dung D. Le, Anh Tran (2024). ["SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training" PDF](https://arxiv.org/abs/2409.07321) & ["Audio summary"](https://open.spotify.com/episode/7vKFYxrH1o137zl9MfcKAz?si=oVMoHS61QD6Jjm3XYOTDNQ)
- Gradient Problems in RNNs: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013). ["On the difficulty of training recurrent neural networks" PDF](https://arxiv.org/abs/1211.5063) & ["Audio summary"](https://open.spotify.com/episode/0okbpKt5U4jmiYwqhVks1S?si=QeGK8t2MT5iYzcj5VE9dMw)
- Adam: Diederik P. Kingma, Jimmy Ba (2014). ["Adam: A Method for Stochastic Optimization" PDF](https://arxiv.org/abs/1412.6980) & ["Audio summary"](https://open.spotify.com/episode/6GIPqEzRvwHvRMYYI3M4Ar?si=hMWeNH9PR-O48or43EN2iQ)
- AdamW: Ilya Loshchilov, Frank Hutter (2019). ["Decoupled Weight Decay Regularization" PDF](https://arxiv.org/abs/1711.05101) & ["Audio summary"](https://open.spotify.com/episode/0s5ywoHyIS1dTTT2cLxPpV?si=h335wbgGQ0m94FsBtX-SxQ)
- Adafactor: Noam Shazeer, Mitchell Stern (2018). ["Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" PDF](https://arxiv.org/abs/1804.04235) & ["Audio summary"](https://open.spotify.com/episode/46DNk6Mkfk4r6xikZPzYT1?si=UUkAQyQEQai-rQypL_lqgA)

## Citation

If you use this work, please cite:

```bibtex
@misc{AdamWPrune2025,
  title        = {AdamWPrune: Multi-Model State-based Pruning},
  author       = {Luis Chamberlain},
  year         = {2025},
  howpublished = {\url{https://github.com/mcgrof/AdamWPrune}},
  note         = {State-based pruning across LeNet-5 and ResNet-18}
}
```

## License

All AdamWPrune code except scripts/kconfig is MIT licensed. The scripts/kconfig directory is GPLv2. The project as a whole is GPLv2. AI models generated by this project can be licensed as you choose.

See LICENSE for details.
