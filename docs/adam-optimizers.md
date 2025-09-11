# Understanding Adam Optimizers: Evolution and Variants

## What is an Optimizer?

Before diving into Adam, let's understand what an optimizer does. When training neural networks, we need to adjust millions of parameters (weights) to minimize errors. An optimizer is the algorithm that decides **how much** and **in which direction** to adjust each parameter after seeing training examples.

Think of it like navigating a foggy mountain landscape while blindfolded, trying to reach the lowest valley. The optimizer is your strategy for taking steps - how big, which direction, and whether to trust your current path or try something new.

## The Adam Optimizer: A Breakthrough in 2014

### The Problem Adam Solved

Before Adam, we had two main approaches:
- **SGD (Stochastic Gradient Descent)**: Like taking consistent steps downhill, but uses the same step size for all parameters. Works well but requires careful tuning.
- **AdaGrad/RMSprop**: Adapts step sizes per parameter, but can become too conservative over time.

### Adam's Innovation

Adam ([Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)) combined the best of both worlds:
- **Momentum** (from SGD): Remembers the direction you've been moving, like a ball rolling downhill that builds up speed
- **Adaptive learning rates** (from RMSprop): Different step sizes for different parameters

📎 [Listen to the Adam paper explained](https://open.spotify.com/episode/6GIPqEzRvwHvRMYYI3M4Ar?si=hMWeNH9PR-O48or43EN2iQ)

The name "Adam" comes from "**Ada**ptive **M**oment estimation" - it adapts based on both:
1. **First moment** (mean/momentum): Average of recent gradients - "which way have we been going?"
2. **Second moment** (variance): Average of squared gradients - "how bumpy is this path?"

### Why Adam Became Popular

Adam quickly became the default optimizer because:
- **Less tuning required**: Works well with default settings across many problems
- **Fast convergence**: Often reaches good solutions quicker than SGD
- **Handles sparse gradients**: Works well when only some parameters update frequently

## The Evolution: AdamW (2019)

### The Weight Decay Problem

Researchers discovered that Adam had a subtle but important flaw in how it handled **weight decay** (L2 regularization) - a technique to prevent overfitting by penalizing large weights.

In traditional optimizers like SGD, weight decay directly shrinks weights toward zero. But Adam was inadvertently coupling weight decay with the adaptive learning rate, making it less effective for some parameters.

### AdamW's Solution

AdamW ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) **decoupled** weight decay from the gradient-based updates:
- Weight decay is applied directly to weights, not mixed with gradients
- This simple change significantly improved performance, especially for transformer models

📎 [Listen to the AdamW paper explained](https://open.spotify.com/episode/0s5ywoHyIS1dTTT2cLxPpV?si=h335wbgGQ0m94FsBtX-SxQ)

**Result**: AdamW became the new standard, particularly for language models. It's now the default optimizer in PyTorch for transformers.

## Modern Variants: The AdamW Family

### Why "AdamW" Prefix?

We use "AdamW" as a prefix for modern variants because:
1. They build on AdamW's decoupled weight decay foundation
2. It signals they're production-ready (AdamW is battle-tested)
3. It indicates compatibility with modern best practices

### AdamWAdv (Advanced)

Our enhanced version that adds three improvements:
- **AMSGrad**: Fixes a theoretical issue where Adam might not converge by keeping track of maximum historical variance
- **Cosine Annealing**: Gradually reduces learning rate following a cosine curve
- **Gradient Clipping**: Prevents exploding gradients by capping their magnitude

Best for: Stable training when you have time for careful tuning.

### AdamWSPAM (Spike-Aware Pruning Adaptive Momentum)

Based on [Nguyen et al., 2024](https://arxiv.org/abs/2409.07321), AdamWSPAM addresses a specific problem in large model training: **gradient spikes**.

📎 [Listen to the SPAM paper explained](https://open.spotify.com/episode/7vKFYxrH1o137zl9MfcKAz?si=oVMoHS61QD6Jjm3XYOTDNQ)

**The Problem**: During training, especially for large models, gradients can suddenly spike - like hitting an unexpected cliff while descending. These spikes can throw off Adam's momentum, sending training in wrong directions.

**SPAM's Solution**:
1. **Monitors gradient patterns** to detect unusual spikes (>2 standard deviations)
2. **Resets momentum** when spikes occur, preventing bad directions from persisting
3. **Adapts dynamically** based on a temperature parameter θ (theta)

**Why it works better for larger models**:
- Larger models have more complex loss landscapes with more "cliffs" and sudden changes
- The momentum reset mechanism prevents these disruptions from derailing training
- Our tests confirm: AdamWSPAM beats AdamW on ResNet-50 but not on smaller ResNet-18

### AdamWPrune (State-Based Pruning)

Our innovation that reuses Adam's internal states for pruning decisions:
- **Traditional pruning**: Needs extra memory to track weight importance
- **AdamWPrune**: Uses existing momentum and variance estimates to identify important weights
- **Result**: Achieves pruning with minimal memory overhead

The key insight: Adam is already tracking which weights are important (high momentum, low variance) - we just use that information for pruning.

## Practical Guidelines

### Choosing an Optimizer

Based on our empirical findings:

**For smaller models (≤15M parameters):**
- Start with **AdamW** - best balance of performance and simplicity
- Consider **Adam** if you don't need weight decay

**For larger models (>15M parameters):**
- Try **AdamWSPAM** - the spike detection helps with complex loss landscapes
- Use **AdamW** as fallback if SPAM's overhead isn't justified

**For pruning tasks:**
- **AdamWPrune** for memory-efficient pruning
- Configure base optimizer based on model size (AdamW for small, AdamWSPAM for large)

**When stability is critical:**
- **AdamWAdv** with its additional safety features
- **SGD** still wins for some computer vision tasks requiring ultimate accuracy

### Key Hyperparameters

**Learning Rate (lr)**:
- Start with 1e-3 for Adam variants
- Scale down for larger models (5e-4 or 1e-4)

**Betas (β₁, β₂)**:
- Default (0.9, 0.999) works for most cases
- Increase β₁ to 0.95 for noisy gradients
- Decrease β₂ to 0.99 for better responsiveness

**Weight Decay**:
- 0.01-0.1 for AdamW variants
- 0 for vanilla Adam (use AdamW if you need decay)

**SPAM Theta (θ)**:
- Higher values (50-100) for aggressive spike handling
- Lower values (10-30) for gentle momentum adjustments

## The Gradient Problem Context

Understanding why we need these optimizers requires knowing about gradient problems:

📎 [Listen to gradient problems in RNNs explained](https://open.spotify.com/episode/0okbpKt5U4jmiYwqhVks1S?si=QeGK8t2MT5iYzcj5VE9dMw)

**Vanishing Gradients**: Signals become too weak to update early layers
**Exploding Gradients**: Signals become too strong, causing instability

Adam and its variants help manage these issues through adaptive scaling and momentum, making deep learning practical for complex models.

## Summary

The Adam optimizer family represents an evolution in training neural networks:
1. **Adam** (2014): Combined momentum with adaptive learning rates
2. **AdamW** (2019): Fixed weight decay coupling, became the new standard
3. **Modern variants**: Address specific challenges (spikes, memory, stability)

The key lesson from our testing: **No single optimizer is best for all models**. Model size, task complexity, and training constraints all influence which optimizer performs best. The Adam family gives us a toolbox where each tool has its optimal use case.
