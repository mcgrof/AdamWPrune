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
    from lib.optimizers import (
        create_optimizer,
        apply_spam_gradient_processing,
        apply_periodic_spam_reset,
        apply_adamprune_masking,
        update_adamprune_masks,
    )
    from lib.movement_pruning import MovementPruning
    from lib.magnitude_pruning import MagnitudePruning
except ImportError:
    # If direct import fails, try adding parent to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lib.optimizers import (
        create_optimizer,
        apply_spam_gradient_processing,
        apply_periodic_spam_reset,
        apply_adamprune_masking,
        update_adamprune_masks,
    )
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
parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

# Training
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size (default optimized for 24GB+ GPUs)",
)
parser.add_argument(
    "--gradient-accumulation",
    type=int,
    default=4,
    help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
)
parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override for number of epochs (alias for num-epochs)",
)
parser.add_argument(
    "--max-iters",
    type=int,
    default=10000,
    help="Maximum iterations (for better convergence)",
)
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
    choices=["sgd", "adam", "adamw", "adamwadv", "adamwspam", "adamwprune"],
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
parser.add_argument(
    "--adamwprune-amsgrad", type=str, default="false", help="Use AMSGrad for AdamWPrune"
)

# SPAM configuration
parser.add_argument("--spam-theta", type=float, default=50.0, help="SPAM theta")
parser.add_argument(
    "--spam-interval", type=int, default=1000, help="SPAM reset interval"
)
parser.add_argument(
    "--spam-warmup-steps", type=int, default=100, help="SPAM warmup steps"
)
parser.add_argument(
    "--spam-enable-clip", action="store_true", help="Enable SPAM clipping"
)

# Evaluation
parser.add_argument(
    "--eval-interval", type=int, default=100, help="Evaluation interval"
)
parser.add_argument(
    "--eval-samples", type=int, default=200, help="Number of evaluation samples"
)
parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

# Output
parser.add_argument(
    "--json-output", type=str, default=None, help="Path to save training metrics JSON"
)

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

# Experiment tracking
parser.add_argument(
    "--tracker",
    type=str,
    default="none",
    choices=["none", "trackio", "wandb"],
    help="Experiment tracker to use (none, trackio, or wandb)",
)
parser.add_argument(
    "--tracker-project",
    type=str,
    default="adamwprune-gpt2",
    help="Project name for experiment tracker",
)
parser.add_argument(
    "--tracker-run-name",
    type=str,
    default=None,
    help="Run name for experiment tracker (auto-generated if not provided)",
)

def main():
    """Main training function."""
    args = parser.parse_args()

    # Handle epochs alias
    if args.epochs is not None:
        args.num_epochs = args.epochs

    # -----------------------------------------------------------------------------
    # Setup

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize experiment tracker
    tracker = None
    if args.tracker == "trackio":
        try:
            import trackio
            run_name = args.tracker_run_name or f"gpt2_{args.optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trackio.init(
                project=args.tracker_project,
                config=vars(args),
                name=run_name,
            )
            tracker = "trackio"
            print(f"Initialized Trackio tracking for project: {args.tracker_project}", flush=True)
        except ImportError:
            print("Warning: trackio not installed. Install with: pip install trackio", flush=True)
            tracker = None
    elif args.tracker == "wandb":
        try:
            import wandb
            run_name = args.tracker_run_name or f"gpt2_{args.optimizer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.tracker_project,
                config=vars(args),
                name=run_name,
            )
            tracker = "wandb"
            print(f"Initialized WandB tracking for project: {args.tracker_project}", flush=True)
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb", flush=True)
            tracker = None

    # Device setup - auto-detect if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", flush=True)
        device = "cpu"
    else:
        device = args.device

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    ptdtype = dtype

    # Only use autocast if on CUDA and not using float32
    if device == "cpu" or args.dtype == "float32":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

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
    # DDP setup (if enabled)
    ddp = False
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    # Check if DDP is enabled via config
    try:
        import config as cfg
        if hasattr(cfg, 'config') and hasattr(cfg.config, 'GPT2_USE_DDP'):
            use_ddp = cfg.config.GPT2_USE_DDP == 'y'
            ddp_backend = getattr(cfg.config, 'GPT2_DDP_BACKEND', 'nccl')
            ddp_find_unused = getattr(cfg.config, 'GPT2_DDP_FIND_UNUSED_PARAMS', 'y') == 'y'
        else:
            use_ddp = False
            ddp_backend = 'nccl'
            ddp_find_unused = True
    except ImportError:
        use_ddp = False
        ddp_backend = 'nccl'
        ddp_find_unused = True

    # Initialize DDP if enabled and environment variables are set
    if use_ddp and 'RANK' in os.environ:
        ddp = True
        init_process_group(backend=ddp_backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        print(f"DDP initialized: rank {ddp_rank}/{ddp_world_size}, local rank {ddp_local_rank}, device {device}", flush=True)
    else:
        master_process = True
        seed_offset = 0
        if use_ddp:
            print("DDP enabled in config but RANK environment variable not set. Running in single GPU mode.", flush=True)

    # -----------------------------------------------------------------------------
    # Model initialization

    if master_process:
        print(f"Initializing GPT-2 model: {args.model_name}", flush=True)
    config = GPTConfig.from_name(args.model_name)
    config.block_size = args.block_size
    config.dropout = args.dropout
    config.bias = args.bias

    model = GPT(config)
    model.to(device)

    # Wrap model in DDP if enabled
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=ddp_find_unused)

    # Compile model if requested (only compile the base model, not DDP wrapper)
    if args.compile and hasattr(torch, "compile") and not ddp:
        print("Compiling model with torch.compile()...", flush=True)
        model = torch.compile(model)

    # -----------------------------------------------------------------------------
    # Optimizer setup

    print(f"Setting up {args.optimizer} optimizer...", flush=True)

    # Enable state pruning for AdamWPrune when requested
    if args.optimizer == "adamwprune" and args.pruning_method == "state":
        args.adamwprune_enable_pruning = True
        args.adamwprune_target_sparsity = args.target_sparsity
        args.adamwprune_warmup_steps = args.pruning_warmup
        args.adamwprune_ramp_end_epoch = min(8, args.num_epochs - 1)

    # Create optimizer using the library function
    optimizer, scheduler, gradient_clip_norm, spam_state, adamwprune_state = (
        create_optimizer(
            model=model,
            optimizer_type=args.optimizer,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            args=args,
            model_type="gpt2",
        )
    )

    # Pruning setup
    pruner = None
    if args.pruning_method != "none" and args.pruning_method != "state":
        print(f"Setting up {args.pruning_method} pruning...", flush=True)
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

    # Initialize gradient scaler for mixed precision (only for CUDA)
    if device == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    else:
        # CPU doesn't support GradScaler, use a dummy that just passes through
        class DummyScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        scaler = DummyScaler()

    print(f"\nStarting training...", flush=True)
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M", flush=True)
    print(f"Device: {device}, dtype: {dtype}", flush=True)
    print(
        f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation}",
        flush=True
    )
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}", flush=True)
    print("-" * 50, flush=True)

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

        # Gradient processing for special optimizers
        if args.optimizer != "sgd":
            # Only unscale if using actual CUDA scaler
            if device == "cuda":
                scaler.unscale_(optimizer)

            # Apply AdamWPrune gradient masking
            apply_adamprune_masking(optimizer, adamwprune_state)

            # Apply SPAM gradient processing
            apply_spam_gradient_processing(optimizer, model, spam_state, gradient_clip_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Periodic SPAM momentum reset with optional warmup
        apply_periodic_spam_reset(optimizer, spam_state)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Update AdamWPrune state-based pruning
        # Calculate current epoch based on iterations (approximate)
        iters_per_epoch = (
            args.max_iters // args.num_epochs if args.num_epochs > 0 else args.max_iters
        )
        current_epoch = iter_num // iters_per_epoch
        update_adamprune_masks(optimizer, adamwprune_state, None, current_epoch)

        # Update pruning for external pruners
        if pruner is not None:
            pruner.update_masks(iter_num)

        # Logging (only on master process)
        if iter_num % args.log_interval == 0 and master_process:
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
                f"sparsity {sparsity:.1%} | {dt*1000/args.log_interval:.1f}ms/iter",
                flush=True
            )

            metrics["train_losses"].append(avg_loss)
            metrics["learning_rates"].append(lr)
            metrics["sparsities"].append(sparsity)
            metrics["iterations"].append(iter_num)
            metrics["timestamps"].append(time.time())

            # Log to experiment tracker
            if tracker == "trackio":
                import trackio
                trackio.log({
                    "iteration": iter_num,
                    "train_loss": avg_loss,
                    "learning_rate": lr,
                    "sparsity": sparsity,
                })
            elif tracker == "wandb":
                import wandb
                wandb.log({
                    "iteration": iter_num,
                    "train_loss": avg_loss,
                    "learning_rate": lr,
                    "sparsity": sparsity,
                })

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

            print(f"Validation loss: {val_loss:.4f}", flush=True)
            metrics["val_losses"].append(val_loss)

            # Log validation to experiment tracker
            if tracker == "trackio":
                import trackio
                trackio.log({
                    "iteration": iter_num,
                    "val_loss": val_loss,
                })
            elif tracker == "wandb":
                import wandb
                wandb.log({
                    "iteration": iter_num,
                    "val_loss": val_loss,
                })

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
                print(f"Saved best model (val_loss: {val_loss:.4f})", flush=True)

    # -----------------------------------------------------------------------------
    # Final evaluation and saving

    print("\n" + "=" * 50, flush=True)
    print("Training complete!", flush=True)

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

    print(f"Final validation loss: {final_val_loss:.4f}", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)

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
        print(f"Saved metrics to {args.json_output}", flush=True)

    # Save detailed metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved detailed metrics to {metrics_path}", flush=True)

    print("\nTraining complete!", flush=True)

    # Finish experiment tracking
    if tracker == "trackio":
        import trackio
        trackio.log({
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "total_time": metrics["total_time"],
        })
        trackio.finish()
        print("Trackio tracking finished. Run 'trackio show' to view results.", flush=True)
    elif tracker == "wandb":
        import wandb
        wandb.log({
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "total_time": metrics["total_time"],
        })
        wandb.finish()
        print("WandB tracking finished. Check your WandB dashboard for results.", flush=True)

    # Cleanup DDP
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
