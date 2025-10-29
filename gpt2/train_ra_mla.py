"""
Training script for GPT-2 with Reciprocal Attention (RA) + MLA.

This script extends the base GPT-2 training with:
- RA+MLA attention mechanism
- Attention entropy and reciprocity logging
- Empirical complexity measurement (actual FLOPs and memory)
- Integration with AdamWPrune pruning hooks

Usage:
    python train_ra_mla.py --dataset finewebedu \
                           --latent-dim 64 \
                           --ra-window 64 \
                           --ra-alpha 0.5 \
                           --max-iters 10000

Ablation studies:
    # Pure MLA (no reciprocal)
    python train_ra_mla.py --ra-alpha 0.0 --latent-dim 64

    # Different latent dimensions
    python train_ra_mla.py --latent-dim 32  # more compression
    python train_ra_mla.py --latent-dim 128  # less compression

    # Different reciprocal windows
    python train_ra_mla.py --ra-window 32  # narrow local context
    python train_ra_mla.py --ra-window 128  # wider local context
"""

import os
import sys

# CRITICAL: Set environment variables before importing torch
# Read from config.py if available, otherwise use defaults
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from config import Config

    config = Config()
    if hasattr(config, "PYTORCH_CUDA_ALLOC_CONF"):
        # Set both old and new variable names for compatibility
        alloc_conf = config.PYTORCH_CUDA_ALLOC_CONF
        os.environ.setdefault("PYTORCH_ALLOC_CONF", alloc_conf)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)

    # Enable experimental flash attention for Navi31/32/33 when explicitly requested
    # Only set environment variable if CONFIG_GPT2_FLASH_ATTENTION=y AND we detect Navi3x
    flash_attention_enabled = (
        hasattr(config, "GPT2_FLASH_ATTENTION")
        and config.GPT2_FLASH_ATTENTION
        and config.GPT2_FLASH_ATTENTION != "n"
    )

    if flash_attention_enabled:
        # Detect Navi31/32/33 before importing torch using rocm-smi
        try:
            import subprocess

            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.lower()
                is_navi3x = any(
                    x in gpu_name for x in ["navi31", "navi32", "navi33", "w7900"]
                )
                if is_navi3x:
                    os.environ.setdefault(
                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # rocm-smi not available or timed out
except (ImportError, AttributeError):
    # Fallback to safe defaults if config.py doesn't exist or doesn't have the settings
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import math
import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np

# Suppress wandb weave warning
try:
    import weave
except ImportError:
    pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from ra_mla_gpt2 import patch_gpt2_with_ra_mla, score_heads_for_prune_gpt2

# Import training utilities from base train.py
try:
    from lib.optimizers import create_optimizer
    from lib.scaling_curves import show_scaling_curves
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lib.optimizers import create_optimizer
    from lib.scaling_curves import show_scaling_curves


# -----------------------------------------------------------------------------
# Helper functions
def supports_tensorcore_fp32():
    """
    Detect if the current device supports tensor cores (NVIDIA) or WMMA (AMD).
    Works on both CUDA and ROCm backends.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check device capability for NVIDIA GPUs
        cap = torch.cuda.get_device_capability(0)
        # Tensor cores require SM >= 7.0 (Volta and newer)
        if cap and cap[0] >= 7:
            return True
    except Exception:
        pass

    # ROCm heuristic: look for RDNA3 / gfx11+ which have WMMA ISA
    if torch.version.hip:
        try:
            arch = torch.cuda.get_device_name(0).lower()
            # Check for RDNA3 (gfx110x, Navi3x, W7900) or MI series
            return any(
                x in arch
                for x in [
                    "gfx110",
                    "navi31",
                    "navi32",
                    "navi33",
                    "w7900",
                    "mi300",
                    "mi250",
                ]
            )
        except Exception:
            pass

    return False


# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description="GPT-2 RA+MLA training")

# Model configuration
parser.add_argument("--model-name", type=str, default="gpt2", help="GPT-2 model size")
parser.add_argument("--block-size", type=int, default=1024, help="Context length")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

# RA+MLA configuration
parser.add_argument(
    "--latent-dim", type=int, default=64, help="Latent dimension for K/V compression"
)
parser.add_argument(
    "--ra-window", type=int, default=64, help="Reciprocal attention band width"
)
parser.add_argument(
    "--ra-alpha",
    type=float,
    default=0.5,
    help="Reciprocal attention weight (0.0 = pure MLA)",
)
parser.add_argument(
    "--per-head-q-latent",
    action="store_true",
    default=True,
    help="Per-head Q-to-latent projection",
)
parser.add_argument(
    "--per-head-v-up",
    action="store_true",
    default=True,
    help="Per-head V up-projection",
)
parser.add_argument(
    "--use-flash",
    action="store_true",
    default=True,
    help="Use FlashAttention if available",
)

# Reciprocal MLP configuration
parser.add_argument(
    "--mlp-attn-gate",
    action="store_true",
    default=False,
    help="Enable mechanism 1: MLP gates attention heads",
)
parser.add_argument(
    "--no-mlp-attn-gate",
    dest="mlp_attn_gate",
    action="store_false",
    help="Disable MLP attention gating",
)
parser.add_argument(
    "--mlp-cross-token",
    action="store_true",
    default=False,
    help="Enable mechanism 2: Cross-token MLP aggregation",
)
parser.add_argument(
    "--no-mlp-cross-token",
    dest="mlp_cross_token",
    action="store_false",
    help="Disable cross-token MLP",
)
parser.add_argument(
    "--mlp-latent-recip",
    action="store_true",
    default=False,
    help="Enable mechanism 3: MLP-attention latent reciprocity",
)
parser.add_argument(
    "--no-mlp-latent-recip",
    dest="mlp_latent_recip",
    action="store_false",
    help="Disable MLP latent reciprocity",
)
parser.add_argument(
    "--mlp-gate-alpha",
    type=float,
    default=0.1,
    help="Mixing weight for MLP attention gating",
)
parser.add_argument(
    "--mlp-cross-alpha",
    type=float,
    default=0.3,
    help="Mixing weight for cross-token MLP",
)
parser.add_argument(
    "--mlp-recip-alpha",
    type=float,
    default=0.2,
    help="Mixing weight for MLP latent reciprocity",
)
parser.add_argument(
    "--mlp-gate-dim",
    type=int,
    default=64,
    help="Context vector dimension for gating",
)
parser.add_argument(
    "--mlp-latent-dim",
    type=int,
    default=128,
    help="Dimension for MLP latent space",
)
parser.add_argument(
    "--ra-mla-ablation-step",
    type=str,
    default=None,
    help="Ablation study step (0-5). Overrides mechanism flags to enable specific combinations.",
)

# Dataset
parser.add_argument("--dataset", type=str, default="shakespeare", help="Dataset to use")
parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

# Training
parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
parser.add_argument(
    "--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps"
)
parser.add_argument("--max-iters", type=int, default=10000, help="Maximum iterations")
parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")

# Optimizer
parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use")

# SPAM configuration (for AdamWSPAM optimizer)
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

# Test matrix compatibility arguments (accepted but ignored for RA+MLA)
parser.add_argument(
    "--decay-lr",
    action="store_true",
    default=True,
    help="Use LR decay (always enabled for RA+MLA)",
)
parser.add_argument(
    "--pruning-method",
    type=str,
    default="none",
    help="Pruning method (ignored for RA+MLA)",
)

# Evaluation and logging
parser.add_argument(
    "--eval-interval", type=int, default=100, help="Evaluation interval"
)
parser.add_argument(
    "--eval-samples", type=int, default=200, help="Number of evaluation samples"
)
parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
parser.add_argument(
    "--log-metrics", action="store_true", default=True, help="Log attention metrics"
)

# Experiment tracking
parser.add_argument(
    "--tracker",
    type=str,
    default="none",
    help="Experiment tracker(s): none, trackio, wandb, or comma-separated",
)
parser.add_argument(
    "--tracker-project",
    type=str,
    default=None,
    help="Project name for experiment tracking",
)
parser.add_argument(
    "--tracker-run-name",
    type=str,
    default=None,
    help="Run name for experiment tracking",
)

# Output
parser.add_argument(
    "--json-output", type=str, default=None, help="Path to save metrics JSON"
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="checkpoints_ra_mla",
    help="Checkpoint directory",
)

# System
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    help="Data type (bfloat16, float16, float32)",
)
parser.add_argument(
    "--compile", action="store_true", default=False, help="Use torch.compile"
)

args = parser.parse_args()

# Override max_iters from environment if set (takes precedence over config and command-line)
if os.environ.get("GPT2_MAX_ITERS"):
    args.max_iters = int(os.environ.get("GPT2_MAX_ITERS"))

# Handle ablation study step if specified
# This overrides the mechanism flags to enable specific combinations for ablation study
if args.ra_mla_ablation_step is not None:
    step = args.ra_mla_ablation_step
    if step == "0":
        # Baseline: MLA only, no reciprocal MLP mechanisms
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
    elif step == "1":
        # Step 1: Mechanism 1 only (MLP-to-Attention Gating)
        args.mlp_attn_gate = True
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
    elif step == "2":
        # Step 2: Mechanisms 1+2 (+ Cross-Token MLP Aggregation)
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
    elif step == "3":
        # Step 3: All three mechanisms
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = True
    elif step == "4":
        # Step 4: Mechanisms 1+2 (AdamWSPAM sanity check)
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
    elif step == "5":
        # Step 5: Full solution (all three mechanisms with AdamWSPAM)
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = True
    else:
        raise ValueError(f"Invalid ablation step: {step}. Must be 0-5.")

# Override RA+MLA config from config.py if available (for test matrix integration)
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_dir, "config.py")
    if os.path.exists(config_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        if hasattr(config_module, "config"):
            cfg = config_module.config
            # Override RA+MLA parameters from config if they exist
            if hasattr(cfg, "RA_MLA_LATENT_DIM"):
                args.latent_dim = int(cfg.RA_MLA_LATENT_DIM)
            if hasattr(cfg, "RA_MLA_RA_WINDOW"):
                args.ra_window = int(cfg.RA_MLA_RA_WINDOW)
            if hasattr(cfg, "RA_MLA_RA_ALPHA"):
                args.ra_alpha = float(cfg.RA_MLA_RA_ALPHA)
            if hasattr(cfg, "RA_MLA_PER_HEAD_Q_LATENT"):
                args.per_head_q_latent = (
                    cfg.RA_MLA_PER_HEAD_Q_LATENT == "y"
                    or cfg.RA_MLA_PER_HEAD_Q_LATENT is True
                )
            if hasattr(cfg, "RA_MLA_PER_HEAD_V_UP"):
                args.per_head_v_up = (
                    cfg.RA_MLA_PER_HEAD_V_UP == "y" or cfg.RA_MLA_PER_HEAD_V_UP is True
                )
            if hasattr(cfg, "RA_MLA_USE_FLASH"):
                args.use_flash = (
                    cfg.RA_MLA_USE_FLASH == "y" or cfg.RA_MLA_USE_FLASH is True
                )
            if hasattr(cfg, "RA_MLA_LOG_METRICS"):
                args.log_metrics = (
                    cfg.RA_MLA_LOG_METRICS == "y" or cfg.RA_MLA_LOG_METRICS is True
                )
            # Override training parameters only if not set via environment variable
            # (test matrix passes GPT2_MAX_ITERS via environment which overrides config)
            if hasattr(cfg, "GPT2_MAX_ITERS") and not os.environ.get("GPT2_MAX_ITERS"):
                args.max_iters = int(cfg.GPT2_MAX_ITERS)
except Exception as e:
    # If config.py doesn't exist or can't be loaded, just use command line args
    pass


# ============================================================================
# Data Loading
# ============================================================================


def get_batch(
    split: str, batch_size: int, block_size: int, device: str, data_dir: str = "data"
):
    """Load a batch of data."""
    # This is a placeholder - you should replace with actual data loading
    # For now, generate random data for testing
    data_path = os.path.join(data_dir, args.dataset)

    if not os.path.exists(data_path):
        print(
            f"Warning: Data path {data_path} not found, using random data for testing"
        )
        # Generate random tokens
        x = torch.randint(0, 50257, (batch_size, block_size), device=device)
        y = torch.randint(0, 50257, (batch_size, block_size), device=device)
        return x, y

    # Load actual data (implement based on your data format)
    # For FineWebEdu, you'd load from preprocessed .bin files
    train_data = np.memmap(
        os.path.join(data_path, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_path, "val.bin"), dtype=np.uint16, mode="r")

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )

    if device != "cpu":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )

    return x, y


# ============================================================================
# Metrics Logging
# ============================================================================


class RAMLAMetrics:
    """Track and log RA+MLA specific metrics."""

    def __init__(self):
        self.attention_entropy = []
        self.reciprocity_score = []
        self.forward_time = []
        self.memory_allocated = []
        self.iteration = []

    def log(self, iter: int, model: GPT, forward_time_ms: float):
        """Log metrics from the model."""
        # Collect attention entropy and reciprocity from all layers
        entropy_vals = []
        reciprocity_vals = []

        for block in model.transformer.h:
            if hasattr(block.attn, "core"):  # RA_MLA_Attention
                attn = block.attn.core
                if attn.attention_entropy is not None:
                    entropy_vals.append(attn.attention_entropy)
                if attn.reciprocity_score is not None:
                    reciprocity_vals.append(attn.reciprocity_score)

        # Average across layers
        if entropy_vals:
            self.attention_entropy.append(np.mean(entropy_vals))
        if reciprocity_vals:
            self.reciprocity_score.append(np.mean(reciprocity_vals))

        self.forward_time.append(forward_time_ms)
        self.memory_allocated.append(
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        )
        self.iteration.append(iter)

    def save(self, path: str):
        """Save metrics to JSON."""
        data = {
            "iteration": self.iteration,
            "attention_entropy": self.attention_entropy,
            "reciprocity_score": self.reciprocity_score,
            "forward_time_ms": self.forward_time,
            "memory_mb": self.memory_allocated,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved metrics to {path}")


# ============================================================================
# Learning Rate Schedule
# ============================================================================


def get_lr(
    iter: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float
) -> float:
    """Cosine learning rate schedule with warmup."""
    # Linear warmup
    if iter < warmup_iters:
        return lr * iter / warmup_iters
    # Cosine decay
    if iter > max_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def estimate_loss(
    model, eval_samples: int, batch_size: int, block_size: int, device: str
):
    """Estimate loss on train and val sets."""
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(eval_samples):
            x, y = get_batch(split, batch_size, block_size, device, args.data_dir)
            logits, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = np.mean(batch_losses)

    model.train()
    return losses


# ============================================================================
# Main Training Loop
# ============================================================================


def main():
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Validate device - handle CUDA/ROCm availability
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA/ROCm not available - falling back to CPU")
            print("For AMD GPUs (W7900), install PyTorch with ROCm support:")
            print(
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1"
            )
            device = "cpu"
        else:
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

            # Enable TensorFloat32 for matmul if supported (WMMA on AMD, Tensor Cores on NVIDIA)
            if supports_tensorcore_fp32():
                torch.set_float32_matmul_precision("high")
                print("  Enabled TensorFloat32 matmul precision (WMMA/Tensor Cores)")
            else:
                print("  Tensor cores/WMMA not detected, using default precision")
    else:
        print(f"Using device: {device}")

    # Configure torch.compile() for better performance
    if args.compile:
        # Allow .item() calls in compiled graphs (for metrics logging)
        torch._dynamo.config.capture_scalar_outputs = True
        # Increase recompilation limit to handle attention variations
        torch._dynamo.config.cache_size_limit = 128
        print("  torch.compile enabled with scalar output capture")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create model
    print(f"Creating GPT-2 model: {args.model_name}")
    model_config = GPTConfig.from_name(args.model_name)
    model_config.block_size = args.block_size
    model_config.dropout = args.dropout
    model = GPT(model_config)

    # Apply RA+MLA patch
    print(f"Applying RA+MLA patch:")
    print(
        f"  latent_dim={args.latent_dim}, ra_window={args.ra_window}, ra_alpha={args.ra_alpha}"
    )
    print(
        f"  per_head_q_latent={args.per_head_q_latent}, per_head_v_up={args.per_head_v_up}"
    )
    if args.mlp_attn_gate or args.mlp_cross_token or args.mlp_latent_recip:
        print("Reciprocal MLP mechanisms:")
        if args.mlp_attn_gate:
            print(f"  [1] MLP-to-Attention Gating: α={args.mlp_gate_alpha}")
        if args.mlp_cross_token:
            print(f"  [2] Cross-Token MLP Aggregation: α={args.mlp_cross_alpha}")
        if args.mlp_latent_recip:
            print(f"  [3] MLP Latent Reciprocity: α={args.mlp_recip_alpha}")

    model, ra_cfg = patch_gpt2_with_ra_mla(
        model,
        latent_dim=args.latent_dim,
        ra_window=args.ra_window,
        ra_alpha=args.ra_alpha,
        per_head_q_latent=args.per_head_q_latent,
        per_head_v_up=args.per_head_v_up,
        use_flash=args.use_flash,
        log_metrics=args.log_metrics,
        # Reciprocal MLP parameters
        mlp_attn_gate=args.mlp_attn_gate,
        mlp_cross_token=args.mlp_cross_token,
        mlp_latent_recip=args.mlp_latent_recip,
        mlp_gate_alpha=args.mlp_gate_alpha,
        mlp_cross_alpha=args.mlp_cross_alpha,
        mlp_recip_alpha=args.mlp_recip_alpha,
        mlp_gate_dim=args.mlp_gate_dim,
        mlp_latent_dim=args.mlp_latent_dim,
    )

    model = model.to(device)

    # Compile if requested
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # Create optimizer
    optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state = (
        create_optimizer(
            model,
            args.optimizer,
            args.learning_rate,
            num_epochs=None,
            args=args,
            model_type="gpt2",
        )
    )

    print(
        f"Optimizer: {args.optimizer}, LR: {args.learning_rate}, weight_decay: {args.weight_decay}"
    )

    # Metrics tracking
    metrics = RAMLAMetrics()

    # Initialize experiment tracking
    tracker_names = [
        t.strip()
        for t in args.tracker.split(",")
        if t.strip() and t.strip().lower() != "none"
    ]

    if tracker_names:
        # Generate run name if not provided
        run_name = args.tracker_run_name
        if not run_name:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ra_mla_{args.model_name}_{args.optimizer}_L{args.latent_dim}_a{args.ra_alpha}_{timestamp}"

        # Default project name
        project_name = args.tracker_project if args.tracker_project else "gpt2-ra-mla"

        print(f"\nInitializing experiment tracking: {', '.join(tracker_names)}")
        print(f"  Project: {project_name}")
        print(f"  Run: {run_name}")

        if "trackio" in tracker_names:
            try:
                import trackio

                trackio.init(project=project_name, config=vars(args), name=run_name)
                print("  ✓ Trackio initialized")

                # Upload .config file if it exists
                config_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".config",
                )
                if os.path.exists(config_file):
                    try:
                        # Trackio may have a save method, but if not, we'll just log the path
                        if hasattr(trackio, "save"):
                            trackio.save(config_file)
                            print(f"  ✓ Uploaded .config to Trackio")
                        else:
                            trackio.log({"config_file": config_file})
                            print(f"  ✓ Logged .config path to Trackio")
                    except Exception as e:
                        print(f"  ⚠ Could not upload .config to Trackio: {e}")
            except ImportError:
                print("  ✗ Trackio not available (install with: pip install trackio)")
                tracker_names.remove("trackio")

        if "wandb" in tracker_names:
            try:
                import wandb

                wandb.init(
                    project=project_name,
                    config=vars(args),
                    name=run_name,
                )
                print("  ✓ WandB initialized")

                # Upload .config file if it exists
                config_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".config",
                )
                if os.path.exists(config_file):
                    wandb.save(config_file, base_path=os.path.dirname(config_file))
                    print(f"  ✓ Uploaded .config to WandB")
            except ImportError:
                print("  ✗ WandB not available (install with: pip install wandb)")
                tracker_names.remove("wandb")

    # Training loop - Print detailed configuration
    print(f"\nStarting training for {args.max_iters} iterations...")
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(
        f"Batch size: {args.batch_size}, gradient accumulation: {args.gradient_accumulation}"
    )
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Optimizer: {args.optimizer}")

    # Show pruning configuration
    if (
        hasattr(args, "pruning_method")
        and args.pruning_method
        and args.pruning_method != "none"
    ):
        print(
            f"Pruning: {args.pruning_method} @ {float(args.target_sparsity)*100:.0f}% target sparsity"
        )
    else:
        print("Pruning: None")

    # Show AdamWPrune variant if applicable
    if args.optimizer == "adamwprune" and hasattr(args, "adamwprune_variant"):
        variant = getattr(args, "adamwprune_variant", "bitter0")
        print(f"AdamWPrune variant: {variant}")
        if hasattr(args, "adamwprune_base_optimizer_name"):
            base_opt = getattr(args, "adamwprune_base_optimizer_name", "adamw")
            print(f"Base optimizer: {base_opt}")

    # Show RA+MLA ablation step if specified
    if args.ra_mla_ablation_step is not None:
        step_names = {
            "0": "Baseline (MLA only, no reciprocal MLP)",
            "1": "Mechanism 1 (MLP-to-Attention Gating)",
            "2": "Mechanisms 1+2 (+ Cross-Token Aggregation)",
            "3": "All three mechanisms",
            "4": "Mechanisms 1+2 (AdamWSPAM check)",
            "5": "Full solution (all three + AdamWSPAM)",
        }
        step_desc = step_names.get(
            args.ra_mla_ablation_step, f"Step {args.ra_mla_ablation_step}"
        )
        print(f"RA+MLA Ablation: Step {args.ra_mla_ablation_step} - {step_desc}")

    # Show reciprocal MLP mechanisms
    if args.mlp_attn_gate or args.mlp_cross_token or args.mlp_latent_recip:
        mechanisms = []
        if args.mlp_attn_gate:
            mechanisms.append("MLP-Attn-Gate")
        if args.mlp_cross_token:
            mechanisms.append("Cross-Token")
        if args.mlp_latent_recip:
            mechanisms.append("Latent-Recip")
        print(f"Reciprocal MLP: {', '.join(mechanisms)}")

    print("=" * 60 + "\n")

    # Show inference scaling law U-curves
    try:
        # Get model configuration parameters from model_config
        n_layers = model_config.n_layer
        n_embd = model_config.n_embd

        # Calculate MLP dimension
        # Standard GPT-2 uses 4x expansion (n_embd -> 4*n_embd)
        mlp_dim = 4 * n_embd

        # Count total parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Display the U-curves
        print(show_scaling_curves(n_layers, n_embd, mlp_dim, param_count))
    except Exception as e:
        # If scaling curves fail, just continue - this is informational only
        print(f"Note: Could not display scaling curves: {e}\n")

    model.train()
    iter_num = 0
    best_val_loss = float("inf")

    while iter_num < args.max_iters:
        # Update learning rate
        lr = get_lr(
            iter_num, args.warmup_steps, args.max_iters, args.learning_rate, args.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(
                model, args.eval_samples, args.batch_size, args.block_size, device
            )
            print(
                f"Iter {iter_num:6d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}"
            )

            # Log evaluation metrics to trackers
            if tracker_names:
                eval_metrics = {
                    "iteration": iter_num,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "learning_rate": lr,
                }

                if "trackio" in tracker_names:
                    import trackio

                    trackio.log(eval_metrics)

                if "wandb" in tracker_names:
                    import wandb

                    wandb.log(eval_metrics)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "iteration": iter_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": vars(args),
                    },
                    checkpoint_path,
                )
                print(f"Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Forward-backward
        t0 = time.time()
        total_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.gradient_accumulation):
            x, y = get_batch(
                "train", args.batch_size, args.block_size, device, args.data_dir
            )

            with torch.amp.autocast(
                device_type="cuda" if "cuda" in device else "cpu", dtype=dtype
            ):
                logits, loss = model(x, y)
                loss = loss / args.gradient_accumulation

            total_loss += loss.item()
            loss.backward()

        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        # Optimizer step
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms

        # Log metrics
        if args.log_metrics and iter_num % args.log_interval == 0:
            metrics.log(iter_num, model, dt)

            # Log RA-specific metrics to trackers
            if tracker_names and len(metrics.attention_entropy) > 0:
                ra_metrics = {
                    "iteration": iter_num,
                    "train_loss_step": total_loss,
                    "learning_rate": lr,
                    "forward_time_ms": dt,
                    "memory_mb": (
                        metrics.memory_allocated[-1] if metrics.memory_allocated else 0
                    ),
                }

                # Add RA-specific metrics if available
                if metrics.attention_entropy:
                    ra_metrics["attention_entropy"] = metrics.attention_entropy[-1]
                if metrics.reciprocity_score:
                    ra_metrics["reciprocity_score"] = metrics.reciprocity_score[-1]

                if "trackio" in tracker_names:
                    import trackio

                    trackio.log(ra_metrics)

                if "wandb" in tracker_names:
                    import wandb

                    wandb.log(ra_metrics)

        # Print progress
        if iter_num % args.log_interval == 0:
            print(
                f"Iter {iter_num:6d} | loss {total_loss:.4f} | time {dt:.1f}ms | lr {lr:.2e}"
            )

        iter_num += 1

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "iteration": iter_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        },
        final_checkpoint_path,
    )
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")

    # Save metrics
    if args.json_output:
        metrics.save(args.json_output)
    else:
        default_metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
        metrics.save(default_metrics_path)

    # Final evaluation
    print("\nFinal evaluation:")
    final_losses = estimate_loss(
        model, args.eval_samples, args.batch_size, args.block_size, device
    )
    print(f"Train loss: {final_losses['train']:.4f}")
    print(f"Val loss: {final_losses['val']:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Print metrics summary
    if metrics.attention_entropy:
        print(f"\nAttention Metrics:")
        print(
            f"  Entropy: {np.mean(metrics.attention_entropy):.3f} ± {np.std(metrics.attention_entropy):.3f}"
        )
    if metrics.reciprocity_score:
        print(
            f"  Reciprocity: {np.mean(metrics.reciprocity_score):.3f} ± {np.std(metrics.reciprocity_score):.3f}"
        )
    if metrics.forward_time:
        print(f"  Avg iteration time: {np.mean(metrics.forward_time):.1f}ms")

    # Log final summary to trackers
    if tracker_names:
        final_summary = {
            "final_train_loss": final_losses["train"],
            "final_val_loss": final_losses["val"],
            "best_val_loss": best_val_loss,
        }

        if metrics.attention_entropy:
            final_summary["avg_attention_entropy"] = float(
                np.mean(metrics.attention_entropy)
            )
            final_summary["std_attention_entropy"] = float(
                np.std(metrics.attention_entropy)
            )

        if metrics.reciprocity_score:
            final_summary["avg_reciprocity_score"] = float(
                np.mean(metrics.reciprocity_score)
            )
            final_summary["std_reciprocity_score"] = float(
                np.std(metrics.reciprocity_score)
            )

        if metrics.forward_time:
            final_summary["avg_forward_time_ms"] = float(np.mean(metrics.forward_time))

        if "trackio" in tracker_names:
            import trackio

            trackio.log(final_summary)
            trackio.finish()

        if "wandb" in tracker_names:
            import wandb

            wandb.log(final_summary)
            wandb.finish()


if __name__ == "__main__":
    main()
