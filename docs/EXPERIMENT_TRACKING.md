# Experiment Tracking

The GPT-2 training script now supports experiment tracking with both Trackio and Weights & Biases (WandB).

## Installation

### Trackio (Lightweight, Local-First)
```bash
pip install trackio
```

### WandB (Full-Featured, Cloud-Based)
```bash
pip install wandb
```

## Usage

### Kconfig Configuration

Use `make menuconfig` to configure tracking through the menu system:
1. Navigate to "Experiment Tracking"
2. Select your preferred backend (None/Trackio/WandB)
3. Optionally set a custom project name (auto-generated if empty)

### Auto-Generated Project Names

If no project name is specified, it's automatically generated as:
```
{directory_name}-{checksum}
```
For example: `AdamWPrune-a1d53c54`

This ensures unique project names even when working in different clones of the same repository.

### Command-Line Arguments

- `--tracker`: Choose tracking backend (`none`, `trackio`, or `wandb`)
- `--tracker-project`: Project name (auto-generated as `{dir}-{checksum}` if not provided)
- `--tracker-run-name`: Optional custom run name (auto-generated if not provided)

### Examples

#### Using Trackio
```bash
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam \
    --tracker trackio \
    --tracker-project my-experiments \
    --tracker-run-name experiment-1
```

After training, view results with:
```bash
trackio show
```

#### Using WandB
```bash
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam \
    --tracker wandb \
    --tracker-project my-experiments \
    --tracker-run-name experiment-1
```

For offline mode (no login required):
```bash
export WANDB_MODE=offline
python gpt2/train.py --tracker wandb ...
```

Later sync offline runs:
```bash
wandb sync
```

#### No Tracking (Default)
```bash
python gpt2/train.py \
    --dataset finewebedu \
    --optimizer adamwspam \
    --tracker none
```

## Tracked Metrics

Both trackers log the following metrics:
- **Training Loss**: Loss at each logging interval
- **Validation Loss**: Loss at each evaluation interval
- **Learning Rate**: Current learning rate
- **Sparsity**: Current model sparsity (for pruning experiments)
- **Final Metrics**: Best validation loss, total training time

## Testing

Run the test script to verify both integrations work:
```bash
python test_tracking.py
```

## Comparison

| Feature | Trackio | WandB |
|---------|---------|-------|
| Installation Size | Lightweight | Heavy |
| Cloud Dependency | No (local SQLite) | Yes (cloud storage) |
| Dashboard | Local web UI | Cloud web UI |
| Team Collaboration | Limited | Full support |
| Cost | Free | Free tier + paid plans |
| Offline Mode | Always offline | Requires configuration |
| Advanced Features | Basic | Extensive (sweeps, reports, etc.) |

## Recommendations

- **Use Trackio** when:
  - You want simple, local experiment tracking
  - You're concerned about data privacy
  - You don't need cloud collaboration features
  - You want minimal dependencies

- **Use WandB** when:
  - You need team collaboration
  - You want advanced features (hyperparameter sweeps, reports)
  - You need cloud backup of experiments
  - You want integration with other ML tools

## Troubleshooting

### Trackio Issues
- If dashboard doesn't open: `trackio show --port 8080`
- Database location: `~/.trackio/trackio.db`

### WandB Issues
- Login issues: `wandb login` or use offline mode
- Sync offline runs: `wandb sync ./wandb/offline-run-*`