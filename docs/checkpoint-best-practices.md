# Model Checkpoint Best Practices: Industry Standards and Case Studies

## Executive Summary

Model checkpointing is a critical practice in production ML systems. Our AdamWPrune experiments demonstrate why: models often achieve peak performance mid-training, with significant accuracy degradation in later epochs. This guide covers industry-standard checkpointing strategies with real examples from our ResNet-50 experiments.

## Table of Contents
1. [Why Checkpointing Matters](#why-checkpointing-matters)
2. [Industry Standard Practices](#industry-standard-practices)
3. [Case Studies from AdamWPrune](#case-studies-from-adamwprune)
4. [Implementation Strategies](#implementation-strategies)
5. [Storage and Management](#storage-and-management)
6. [Production Deployment](#production-deployment)

## Why Checkpointing Matters

### The Peak vs. Final Problem

Our experiments reveal a critical pattern across all optimizers:

| Optimizer | Best Accuracy | Best Epoch | Final Accuracy | Final Epoch | Degradation |
|-----------|--------------|------------|----------------|-------------|-------------|
| AdamWPrune | 74.68% | 50 | 70.56% | 100 | -4.12% |
| AdamWPrune (70% sparsity) | 72.61% | 63 | 70.56% | 100 | -2.05% |
| AdamWSPAM | 72.18% | 100 | 72.18% | 100 | 0% |
| AdamW | 71.34% | 91 | 70.04% | 100 | -1.30% |

**Key Insight**: Without checkpointing, you would deploy a model that's 2-4% worse than your best achievement!

### Real-World Implications

In production scenarios, a 4% accuracy difference can mean:
- **E-commerce**: Millions in lost revenue from poor recommendations
- **Medical AI**: Critical misdiagnoses that proper checkpointing would prevent
- **Autonomous vehicles**: Safety-critical decisions with suboptimal models
- **NLP systems**: Degraded user experience in chatbots and assistants

## Industry Standard Practices

### 1. Best Model Checkpointing (Most Common)

Save the model whenever validation metrics improve:

```python
class BestModelCheckpoint:
    def __init__(self, monitor='val_accuracy', mode='max'):
        self.monitor = monitor
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        
    def save_if_best(self, model, optimizer, epoch, metrics):
        current = metrics[self.monitor]
        is_best = (self.mode == 'max' and current > self.best) or \
                  (self.mode == 'min' and current < self.best)
        
        if is_best:
            self.best = current
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': self.best,
                'metrics': metrics
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f"✅ Saved best model: {self.monitor}={current:.4f} at epoch {epoch}")
            return True
        return False
```

### 2. Regular Interval Checkpointing

Save at fixed intervals for training recovery and analysis:

```python
def save_checkpoint(model, optimizer, epoch, interval=10):
    if epoch % interval == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

### 3. Multi-Metric Checkpointing

Track multiple metrics simultaneously:

```python
checkpoints = {
    'best_accuracy': BestModelCheckpoint('accuracy', 'max'),
    'best_loss': BestModelCheckpoint('loss', 'min'),
    'best_f1': BestModelCheckpoint('f1_score', 'max'),
    'best_sparsity_accuracy': BestModelCheckpoint('accuracy_at_target_sparsity', 'max')
}
```

## Case Studies from AdamWPrune

### Case Study 1: AdamWPrune Peak Performance

**Scenario**: ResNet-50 with AdamWPrune on CIFAR-100

```python
# Training progression
epochs = range(1, 101)
accuracies = [...]  # Actual training data

# Peak at epoch 63
best_epoch = 63
best_accuracy = 74.68  # At 50% sparsity

# Final epoch
final_epoch = 100
final_accuracy = 70.56  # Degraded by 4.12%

# Without checkpointing: Deploy 70.56% model ❌
# With checkpointing: Deploy 74.68% model ✅
```

**Lesson**: The model achieved peak performance at 50% sparsity during gradual pruning ramp-up, then degraded as sparsity increased.

### Case Study 2: Optimizer-Specific Patterns

Different optimizers peak at different times:

```python
optimizer_peaks = {
    'SGD': {'epoch': 91, 'accuracy': 74.57},
    'AdamWPrune': {'epoch': 63, 'accuracy': 72.61},
    'AdamWSPAM': {'epoch': 100, 'accuracy': 72.18},  # Stable to end
    'AdamW': {'epoch': 91, 'accuracy': 71.34},
    'Adam': {'epoch': 85, 'accuracy': 71.23},
}

# Implication: Need optimizer-aware checkpointing strategies
```

### Case Study 3: Sparsity-Aware Checkpointing

For pruning experiments, track accuracy at different sparsity levels:

```python
class SparsityAwareCheckpoint:
    def __init__(self, target_sparsities=[0.5, 0.7, 0.9]):
        self.target_sparsities = target_sparsities
        self.best_at_sparsity = {s: {'accuracy': 0, 'epoch': 0} 
                                 for s in target_sparsities}
    
    def update(self, epoch, sparsity, accuracy):
        # Find closest target sparsity
        for target in self.target_sparsities:
            if abs(sparsity - target) < 0.01:  # Within 1%
                if accuracy > self.best_at_sparsity[target]['accuracy']:
                    self.best_at_sparsity[target] = {
                        'accuracy': accuracy,
                        'epoch': epoch,
                        'actual_sparsity': sparsity
                    }
                    # Save checkpoint for this sparsity level
                    torch.save(model.state_dict(), 
                              f'best_model_sparsity_{int(target*100)}.pth')
```

## Implementation Strategies

### 1. Early Stopping with Patience

Prevent overfitting while keeping best model:

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
```

### 2. Ensemble Checkpointing

Save top-K models for ensemble:

```python
class TopKCheckpoints:
    def __init__(self, k=3):
        self.k = k
        self.checkpoints = []  # List of (score, path) tuples
        
    def update(self, score, model, epoch):
        checkpoint_path = f'top_model_epoch_{epoch}.pth'
        
        # Save model
        torch.save(model.state_dict(), checkpoint_path)
        
        # Update top-k list
        self.checkpoints.append((score, checkpoint_path))
        self.checkpoints.sort(reverse=True)
        
        # Remove models outside top-k
        if len(self.checkpoints) > self.k:
            _, path_to_remove = self.checkpoints.pop()
            os.remove(path_to_remove)
```

### 3. Training Resume Support

Enable seamless training continuation:

```python
def save_full_checkpoint(model, optimizer, scheduler, epoch, global_step):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

def resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint['cuda_rng_state'] is not None:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    return checkpoint['epoch'], checkpoint['global_step']
```

## Storage and Management

### Storage Strategies

1. **Local Fast Storage**: Keep recent checkpoints on NVMe/SSD
2. **Cloud Backup**: Sync best models to S3/GCS
3. **Compression**: Use `torch.save(..., pickle_protocol=4)` for smaller files
4. **Cleanup Policy**: Delete intermediate checkpoints after training

### Typical Storage Requirements

| Model | Checkpoint Size | 100 Epochs Storage |
|-------|----------------|-------------------|
| LeNet-5 | ~1 MB | ~10 MB (every 10) |
| ResNet-18 | ~45 MB | ~450 MB |
| ResNet-50 | ~100 MB | ~1 GB |
| GPT-2 | ~500 MB | ~5 GB |
| GPT-3 175B | ~350 GB | ~3.5 TB |

### Checkpoint Naming Convention

```python
checkpoint_name = f"{model_name}_{optimizer}_{dataset}_epoch{epoch}_acc{accuracy:.2f}.pth"
# Example: resnet50_adamwprune_cifar100_epoch63_acc74.68.pth
```

## Production Deployment

### Model Selection Strategy

```python
def select_production_model(checkpoint_dir):
    """Select best model for production deployment."""
    
    # 1. Load all checkpoint metadata
    checkpoints = []
    for ckpt_file in glob.glob(f"{checkpoint_dir}/*.pth"):
        ckpt = torch.load(ckpt_file, map_location='cpu')
        checkpoints.append({
            'path': ckpt_file,
            'accuracy': ckpt['metrics']['accuracy'],
            'epoch': ckpt['epoch'],
            'sparsity': ckpt['metrics'].get('sparsity', 0)
        })
    
    # 2. Apply business rules
    candidates = [c for c in checkpoints 
                  if c['accuracy'] > 0.70  # Minimum accuracy threshold
                  and c['sparsity'] >= 0.5  # Minimum compression
                  and c['epoch'] > 20]      # Sufficient training
    
    # 3. Select best candidate
    best = max(candidates, key=lambda x: x['accuracy'])
    
    # 4. Validate before deployment
    model = load_model(best['path'])
    if validate_model(model):
        return best['path']
    else:
        raise ValueError("Best model failed validation")
```

### A/B Testing with Checkpoints

Deploy multiple checkpoints for comparison:

```python
models = {
    'champion': load_checkpoint('best_model_overall.pth'),
    'challenger_1': load_checkpoint('best_model_sparsity_70.pth'),
    'challenger_2': load_checkpoint('best_model_epoch_90.pth')
}

# Route traffic for A/B testing
def route_request(user_id):
    bucket = hash(user_id) % 100
    if bucket < 80:
        return models['champion']
    elif bucket < 90:
        return models['challenger_1']
    else:
        return models['challenger_2']
```

## Recommendations

### For Research Projects
1. Save checkpoints every 10 epochs
2. Always save best validation accuracy
3. Keep final 3 checkpoints for analysis
4. Use descriptive names with metrics

### For Production Training
1. Implement best model checkpointing (mandatory)
2. Save hourly checkpoints for long runs
3. Validate checkpoints before deployment
4. Maintain checkpoint changelog

### For Pruning/Compression
1. Save at each sparsity level
2. Track accuracy-sparsity trade-off
3. Keep best model per sparsity target
4. Compare compressed vs. original

## Conclusion

Our AdamWPrune experiments demonstrate that proper checkpointing can mean the difference between deploying a 70.56% accuracy model versus a 74.68% model - a massive 4.12% improvement that comes free with proper checkpointing.

**Remember**: The best model is rarely the final model. Always checkpoint, always validate, always deploy the best.

## Code Examples

Complete checkpointing implementation used in AdamWPrune:

```python
# From resnet50/train.py
if args.save_checkpoint and epoch % args.checkpoint_interval == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'current_accuracy': test_accuracy,
        'sparsity': current_sparsity,
        'args': args
    }
    checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if test_accuracy > best_accuracy:
        torch.save(checkpoint, 'best_model.pth')
        print(f"New best model saved: {test_accuracy:.2f}% at epoch {epoch}")
```

This approach has proven invaluable in identifying that AdamWPrune achieves its best performance at intermediate sparsity levels, a finding that would have been lost without proper checkpointing.