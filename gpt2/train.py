"""
GPT-2 training script with AdamWPrune support.
Adapted from nanoGPT by Andrej Karpathy: https://github.com/karpathy/nanoGPT

Integrates with the AdamWPrune optimizer for state-based pruning experiments.
"""

import os
import sys
import time
import math
import pickle
import argparse
from contextlib import nullcontext
from datetime import datetime
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig

# Import from parent directory's lib
try:
    from lib.optimizers import create_optimizer
    from lib.movement_pruning import MovementPruning
    from lib.magnitude_pruning import MagnitudePruning
except ImportError:
    # If direct import fails, try adding parent to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lib.optimizers import create_optimizer
    from lib.movement_pruning import MovementPruning
    from lib.magnitude_pruning import MagnitudePruning

# -----------------------------------------------------------------------------
# Argument parsing
parser = argparse.ArgumentParser(description="GPT-2 training with AdamWPrune")

# Model configuration
parser.add_argument(
    "--model-name",
    type=str,
    default="gpt2",
    choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    help="GPT-2 model size",
)
parser.add_argument("--block-size", type=int, default=1024, help="Context length")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--bias", action="store_true", default=True, help="Use bias in Linear/LayerNorm"
)

# Dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="shakespeare",
    choices=["shakespeare", "finewebedu", "openwebtext"],
    help="Dataset to use",
)
parser.add_argument("--data-dir", type=str, default="gpt2/data", help="Data directory")

# Training
parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
parser.add_argument(
    "--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps"
)
parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--max-iters", type=int, default=5000, help="Maximum iterations")
parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
parser.add_argument(
    "--decay-lr", action="store_true", default=True, help="Use learning rate decay"
)
parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")

# Optimizer
parser.add_argument(
    "--optimizer",
    type=str,
    default="adamw",
    choices=["adamw", "adamwprune", "adamwspam", "adamwadv", "sgd"],
    help="Optimizer to use",
)

# Pruning
parser.add_argument(
    "--pruning-method",
    type=str,
    default="none",
    choices=["none", "magnitude", "movement", "state"],
    help="Pruning method",
)
parser.add_argument(
    "--target-sparsity", type=float, default=0.5, help="Target sparsity"
)
parser.add_argument(
    "--pruning-warmup", type=int, default=1000, help="Pruning warmup steps"
)

# AdamWPrune specific
parser.add_argument(
    "--adamwprune-base-optimizer-name",
    type=str,
    default="adamw",
    help="Base optimizer for AdamWPrune",
)
parser.add_argument(
    "--adamwprune-beta1", type=float, default=0.9, help="AdamWPrune beta1"
)
parser.add_argument(
    "--adamwprune-beta2", type=float, default=0.999, help="AdamWPrune beta2"
)
parser.add_argument(
    "--adamwprune-weight-decay", type=float, default=0.1, help="AdamWPrune weight decay"
)

# Evaluation
parser.add_argument(
    "--eval-interval", type=int, default=100, help="Evaluation interval"
)
parser.add_argument(
    "--eval-samples", type=int, default=200, help="Number of evaluation samples"
)
parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

# System
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["float32", "float16", "bfloat16"],
    help="Data type for training",
)
parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
parser.add_argument(
    "--flash-attention", action="store_true", default=True, help="Use Flash Attention"
)

# Output
parser.add_argument(
    "--output-dir", type=str, default="gpt2/outputs", help="Output directory"
)
parser.add_argument(
    "--json-output", type=str, default=None, help="JSON output file for metrics"
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Device setup
device = args.device
dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[args.dtype]
ptdtype = dtype
ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device, dtype=ptdtype)
)

# Fix random seeds
torch.manual_seed(1337)
np.random.seed(1337)

# -----------------------------------------------------------------------------
# Data loading


def get_batch(split, data_dir, dataset, block_size, batch_size, device):
    """Get a batch of data."""
    # Load data
    if split == "train":
        data_path = os.path.join(data_dir, dataset, "train.bin")
    else:
        data_path = os.path.join(data_dir, dataset, "val.bin")

    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Generate random positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Create batch
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if device == "cuda":
        # Pin arrays for async transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# -----------------------------------------------------------------------------
# Model initialization

print(f"Initializing GPT-2 model: {args.model_name}")
config = GPTConfig.from_name(args.model_name)
config.block_size = args.block_size
config.dropout = args.dropout
config.bias = args.bias

model = GPT(config)
model.to(device)

# Compile model if requested
if args.compile and hasattr(torch, "compile"):
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Optimizer setup

print(f"Setting up {args.optimizer} optimizer...")

# Create optimizer using our unified interface
if args.optimizer in ["adamwprune", "adamwspam", "adamwadv"]:
    # Use our custom optimizers (they handle parameter grouping internally)
    optimizer, scheduler, _, _, adamwprune_state = create_optimizer(
        model=model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        args=args,
        model_type="gpt2",
    )
else:
    # Use model's configure_optimizers for standard PyTorch optimizers
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.999),
        device_type=device,
    )
    scheduler = None
    adamwprune_state = None

# Pruning setup
pruner = None
if args.pruning_method != "none" and args.pruning_method != "state":
    print(f"Setting up {args.pruning_method} pruning...")
    if args.pruning_method == "magnitude":
        pruner = MagnitudePruning(
            model=model,
            target_sparsity=args.target_sparsity,
            warmup_steps=args.pruning_warmup,
            ramp_end_step=args.max_iters,
        )
    elif args.pruning_method == "movement":
        pruner = MovementPruning(
            model=model,
            target_sparsity=args.target_sparsity,
            warmup_steps=args.pruning_warmup,
            ramp_end_step=args.max_iters,
        )

# -----------------------------------------------------------------------------
# Training loop


@torch.no_grad()
def evaluate(
    model, data_dir, dataset, block_size, batch_size, device, eval_samples=200
):
    """Evaluate the model."""
    model.eval()
    losses = []

    for _ in range(eval_samples):
        x, y = get_batch("val", data_dir, dataset, block_size, batch_size, device)
        with ctx:
            logits, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return np.mean(losses)


def get_lr(it, warmup_steps, learning_rate, min_lr, max_iters):
    """Learning rate schedule with warmup and cosine decay."""
    # Warmup
    if it < warmup_steps:
        return learning_rate * it / warmup_steps
    # Cosine decay
    if it > max_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Training metrics
metrics = {
    "config": vars(args),
    "model_params": model.get_num_params(),
    "train_losses": [],
    "val_losses": [],
    "learning_rates": [],
    "sparsities": [],
    "timestamps": [],
    "iterations": [],
}

# Initialize gradient scaler for mixed precision
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

print(f"\nStarting training...")
print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
print(f"Device: {device}, dtype: {dtype}")
print(
    f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation}"
)
print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
print("-" * 50)

# Training loop
model.train()
optimizer.zero_grad(set_to_none=True)

t0 = time.time()
running_loss = 0.0
best_val_loss = float("inf")

for iter_num in range(args.max_iters):

    # Determine learning rate
    if args.decay_lr:
        lr = get_lr(
            iter_num, args.warmup_steps, args.learning_rate, args.min_lr, args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        lr = args.learning_rate

    # Accumulate gradients
    for micro_step in range(args.gradient_accumulation):
        x, y = get_batch(
            "train",
            args.data_dir,
            args.dataset,
            args.block_size,
            args.batch_size,
            device,
        )

        with ctx:
            logits, loss = model(x, y)
            loss = loss / args.gradient_accumulation

        # Backward pass
        scaler.scale(loss).backward()
        running_loss += loss.item()

    # Clip gradients
    if args.optimizer != "sgd":
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Update pruning
    if pruner is not None:
        pruner.update_masks(iter_num)

    # Logging
    if iter_num % args.log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        avg_loss = running_loss / args.log_interval

        # Calculate sparsity
        if pruner is not None:
            sparsity = pruner.get_sparsity()
        elif args.optimizer == "adamwprune" and args.pruning_method == "state":
            # Get sparsity from AdamWPrune
            sparsity = 0.0  # TODO: implement sparsity calculation
        else:
            sparsity = 0.0

        print(
            f"Iter {iter_num:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
            f"sparsity {sparsity:.1%} | {dt*1000/args.log_interval:.1f}ms/iter"
        )

        metrics["train_losses"].append(avg_loss)
        metrics["learning_rates"].append(lr)
        metrics["sparsities"].append(sparsity)
        metrics["iterations"].append(iter_num)
        metrics["timestamps"].append(time.time())

        running_loss = 0.0

    # Evaluation
    if iter_num % args.eval_interval == 0:
        val_loss = evaluate(
            model,
            args.data_dir,
            args.dataset,
            args.block_size,
            args.batch_size,
            device,
            args.eval_samples,
        )

        print(f"Validation loss: {val_loss:.4f}")
        metrics["val_losses"].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
                "args": args,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved best model (val_loss: {val_loss:.4f})")

# -----------------------------------------------------------------------------
# Final evaluation and saving

print("\n" + "=" * 50)
print("Training complete!")

# Final evaluation
final_val_loss = evaluate(
    model,
    args.data_dir,
    args.dataset,
    args.block_size,
    args.batch_size,
    device,
    args.eval_samples * 2,
)

print(f"Final validation loss: {final_val_loss:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Save final model
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iter_num": args.max_iters,
    "val_loss": final_val_loss,
    "best_val_loss": best_val_loss,
    "config": config,
    "args": args,
}
torch.save(checkpoint, os.path.join(args.output_dir, "final_model.pt"))

# Save metrics
metrics["final_val_loss"] = final_val_loss
metrics["best_val_loss"] = best_val_loss
metrics["total_time"] = (
    time.time() - metrics["timestamps"][0] if metrics["timestamps"] else 0
)

if args.json_output:
    with open(args.json_output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.json_output}")

# Save detailed metrics
metrics_path = os.path.join(args.output_dir, "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved detailed metrics to {metrics_path}")

print("\nTraining complete!")
