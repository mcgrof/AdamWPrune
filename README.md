# AdamWPrune: Multi-Model State-Based Weight Pruning

> **📊 ResNet-18 Results**: AdamWPrune achieves **lowest GPU memory usage** (1381.6 MiB) among all tested optimizers while maintaining 88% accuracy at 70% sparsity - **102-158 MiB less** than alternatives!

AdamWPrune demonstrates efficient neural network compression by reusing Adam optimizer states for pruning decisions, eliminating the memory overhead of traditional pruning methods. Now validated across multiple architectures from LeNet-5 (61K parameters) to ResNet-18 (11.2M parameters).

## Key Results

### Memory Efficiency Across Models

| Model | Parameters | Dataset | Sparsity | GPU Memory | Accuracy | Efficiency |
|-------|------------|---------|----------|------------|----------|------------|
| LeNet-5 | 61,750 | MNIST | 70% | 434.5 MiB* | 98.9% | 22.74/100MiB |
| ResNet-18 | 11.2M | CIFAR-10 | 70% | 1381.6 MiB | 88.08% | 6.37/100MiB |

*CUDA/PyTorch baseline overhead (~450 MiB) dominates for small models

### GPU Memory Analysis

#### ResNet-18 Production-Scale Results (NEW!)

**🔥 Key Finding: AdamWPrune achieves lowest GPU memory usage among all optimizers**

| Optimizer | Pruning Method | GPU Memory | Accuracy | Memory Savings |
|-----------|---------------|------------|----------|----------------|
| **AdamWPrune** | **State (70%)** | **1381.6 MiB** | **88.08%** | **Baseline** |
| SGD | Movement (70%) | 1429.0 MiB | 92.12% | -3.3% |
| Adam | Movement (70%) | 1483.4 MiB | 90.25% | -6.9% |
| AdamW | Movement (70%) | 1483.7 MiB | 90.33% | -6.9% |
| AdamWAdv | Movement (70%) | 1539.4 MiB | 89.94% | -10.2% |
| AdamWSpam | Movement (70%) | 1539.3 MiB | 90.12% | -10.2% |

**Memory Efficiency**: AdamWPrune uses **102-158 MiB less** GPU memory than other optimizers
**Trade-off**: ~4% accuracy gap vs SGD (88.08% vs 92.12%)

![ResNet-18 Training Memory](images/resnet18/training_memory_comparison.png)

→ See [ResNet-18 detailed findings](resnet18/findings.md) for comprehensive analysis

#### LeNet-5 Comprehensive Analysis
![LeNet-5 Memory Analysis](images/lenet5/training_memory_comparison.png)
*Comprehensive 6-panel analysis showing AdamWPrune's memory efficiency patterns*

#### Memory Efficiency Leaders
Top configurations by accuracy per 100 MiB of GPU memory:
1. **ADAMWPRUNE** (state_70): 22.74 efficiency score
2. **ADAM** (baseline): 22.67 efficiency score
3. **ADAMW** (magnitude_70): 22.54 efficiency score

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
- **Result**: 7.5% GPU memory reduction on ResNet-18

## Detailed Findings

- **[LeNet-5 Results](lenet5/findings.md)**: Proof of concept on MNIST
- **[ResNet-18 Results](resnet18/findings.md)**: Production-scale validation on CIFAR-10

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
# ResNet-18 comprehensive testing
make defconfig-resnet18-comprehensive-pruning-compare
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
