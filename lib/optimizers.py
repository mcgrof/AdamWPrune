# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model,
    optimizer_type,
    learning_rate,
    num_epochs=None,
    args=None,
):
    """
    Create an optimizer based on the specified type.

    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer (sgd, adam, adamw, adamwadv, adamwspam, adamwprune)
        learning_rate: Learning rate for the optimizer
        num_epochs: Number of epochs (needed for some schedulers)
        args: Additional arguments containing SPAM parameters

    Returns:
        optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state
    """
    scheduler = None
    gradient_clip_norm = None
    spam_state = None
    adamprune_state = None

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info("Using Adam optimizer")

    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        logger.info("Using AdamW optimizer with weight decay=0.0001")

    elif optimizer_type == "adamwadv":
        # AdamW Advanced with all enhancements
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,  # Stronger weight decay for better regularization
            amsgrad=True,  # AMSGrad variant for better convergence
        )
        # Cosine annealing learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        # Gradient clipping for stability
        gradient_clip_norm = 1.0
        logger.info("Using AdamW Advanced optimizer with:")
        logger.info("  - AMSGrad enabled for better convergence")
        logger.info("  - Weight decay=0.01 for stronger regularization")
        logger.info("  - Cosine annealing LR schedule")
        logger.info("  - Gradient clipping norm=1.0")

    elif optimizer_type == "adamwspam":
        # AdamW with SPAM (Spike-Aware Pruning-Adaptive Momentum)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=True,
        )
        # Initialize SPAM state tracking
        spam_state = {
            "gradient_history": deque(maxlen=100),
            "gradient_norms": deque(maxlen=100),
            "spike_threshold": 2.0,  # Detect spikes > 2 std deviations
            "spike_events": [],
            "last_reset_step": 0,
            "global_step": 0,
            "momentum_reset_count": 0,
            # SPAM fidelity options
            "theta": args.spam_theta if args else 50.0,
            "interval": args.spam_interval if args else 0,
            "warmup_steps": args.spam_warmup_steps if args else 0,
            "enable_clip": args.spam_enable_clip if args else False,
            "warmup_until": 0,
        }
        # Cosine annealing with SPAM awareness
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        gradient_clip_norm = 1.0
        logger.info("Using AdamW SPAM optimizer with:")
        logger.info("  - Spike detection (threshold=2.0 std)")
        logger.info("  - Automatic momentum reset on spikes")
        logger.info("  - AMSGrad for stability")
        logger.info("  - Adaptive gradient clipping")
        logger.info("  - Cosine annealing LR schedule")
        if spam_state["interval"] > 0:
            logger.info(
                f"  - Periodic momentum reset interval: {spam_state['interval']} steps"
            )
            if spam_state["warmup_steps"] > 0:
                logger.info(
                    f"  - Cosine warmup after reset: {spam_state['warmup_steps']} steps"
                )
        if spam_state["enable_clip"]:
            logger.info(
                f"  - Spike-aware clipping enabled (theta={spam_state['theta']})"
            )

    elif optimizer_type == "adamwprune":
        # AdamWPrune - Experimental: All enhancements + state-based pruning

        # Get tuning parameters from args or use defaults
        beta1 = float(getattr(args, 'adamwprune_beta1', 0.9) if args else 0.9)
        beta2 = float(getattr(args, 'adamwprune_beta2', 0.999) if args else 0.999)
        weight_decay = float(getattr(args, 'adamwprune_weight_decay', 0.01) if args else 0.01)
        amsgrad = bool(getattr(args, 'adamwprune_amsgrad', True) if args else True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=1e-8,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        logger.info(f"Using AdamWPrune with tuned parameters:")
        logger.info(f"  - Beta1: {beta1} (momentum coefficient)")
        logger.info(f"  - Beta2: {beta2} (variance coefficient)")
        logger.info(f"  - Weight decay: {weight_decay}")
        logger.info(f"  - AMSGrad: {amsgrad}")

        # Initialize both SPAM and pruning state
        spam_state = {
            "gradient_history": deque(maxlen=100),
            "gradient_norms": deque(maxlen=100),
            "spike_threshold": 2.0,
            "spike_events": [],
            "last_reset_step": 0,
            "global_step": 0,
            "momentum_reset_count": 0,
            # SPAM fidelity options
            "theta": args.spam_theta if args else 50.0,
            "interval": args.spam_interval if args else 0,
            "warmup_steps": args.spam_warmup_steps if args else 0,
            "enable_clip": args.spam_enable_clip if args else False,
            "warmup_until": 0,
        }

        # AdamWPrune specific state for state-based pruning
        # Only enable built-in state-based pruning when explicitly using "movement" or "state" method
        # For "magnitude" pruning, use external MagnitudePruning class instead
        adamprune_state = {
            "pruning_enabled": (
                args.pruning_method in ["movement", "state", "adamwprune"]
                and args.target_sparsity > 0
                if args
                else False
            ),
            "target_sparsity": (
                args.target_sparsity
                if args and args.pruning_method in ["movement", "state", "adamwprune"]
                else 0
            ),
            "warmup_steps": args.pruning_warmup if args else 100,
            "pruning_frequency": 50,
            "ramp_end_epoch": getattr(args, 'pruning_ramp_end_epoch', 75) if args else 75,
            "step_count": 0,
            "masks": {},  # module -> bool mask buffer
            "pruning_strategy": "hybrid",  # hybrid of momentum and stability
        }

        # Initialize masks for prunable layers as boolean buffers
        if adamprune_state["pruning_enabled"]:
            for _, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    mask = torch.ones_like(module.weight.data, dtype=torch.bool)
                    module.register_buffer("adamprune_mask", mask)
                    adamprune_state["masks"][module] = module.adamprune_mask

        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        gradient_clip_norm = 1.0

        logger.info("Using AdamWPrune (Experimental) optimizer with:")
        logger.info("  - All AdamWAdv + SPAM features")
        logger.info("  - State-based pruning using Adam momentum/variance")
        logger.info("  - Hybrid pruning strategy (momentum × stability)")
        if adamprune_state["pruning_enabled"]:
            logger.info(
                f"  - Target sparsity: {adamprune_state['target_sparsity']:.1%}"
            )
            logger.info(f"  - Pruning warmup: {adamprune_state['warmup_steps']} steps")

    else:  # Default to SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        logger.info("Using SGD optimizer with momentum=0.9")

    return optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state


def apply_spam_gradient_processing(
    optimizer,
    model,
    spam_state,
    gradient_clip_norm,
):
    """Apply SPAM-specific gradient processing including spike detection and clipping."""
    if spam_state is None:
        return

    # SPAM-inspired spike-aware clipping using Adam's second moment
    if spam_state.get("enable_clip", False):
        theta = float(spam_state.get("theta", 50.0))
        eps = 1e-12
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state.get(p, {})
                v = state.get("exp_avg_sq", None)
                if v is None:
                    continue
                # threshold per-parameter: sqrt(theta * v)
                thr = (theta * (v + eps)).sqrt()
                g = p.grad
                # clip per-parameter exceeding threshold
                over = g.abs() > thr
                if over.any():
                    g.data[over] = g.data[over].sign() * thr.data[over]

    # Compute gradient norm for spike detection
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    # Track gradient norms (only if finite)
    if np.isfinite(total_norm):
        spam_state["gradient_norms"].append(total_norm)
    spam_state["global_step"] += 1

    # Detect spikes after warmup period
    if len(spam_state["gradient_norms"]) >= 10:
        history = np.array(list(spam_state["gradient_norms"])[:-1])
        # Filter out NaN and infinite values
        history = history[np.isfinite(history)]

        if len(history) > 0:
            mean_norm = np.mean(history)
            std_norm = np.std(history)

            if std_norm > 0 and np.isfinite(mean_norm) and np.isfinite(std_norm):
                z_score = (total_norm - mean_norm) / std_norm

                # Spike detected - reset momentum
                if z_score > spam_state["spike_threshold"] and np.isfinite(z_score):
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            state = optimizer.state.get(p, {})
                            if "exp_avg" in state:
                                state["exp_avg"].mul_(0.5)  # Soft reset
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"].mul_(0.9)  # Gentle reset

                    spam_state["spike_events"].append(spam_state["global_step"])
                    spam_state["momentum_reset_count"] += 1
                    spam_state["last_reset_step"] = spam_state["global_step"]

                    logger.debug(
                        f"SPAM: Spike detected (z={z_score:.2f}), momentum reset #{spam_state['momentum_reset_count']}"
                    )


def apply_periodic_spam_reset(optimizer, spam_state):
    """Apply periodic SPAM momentum reset with optional warmup."""
    if spam_state is None:
        return

    spam_state["global_step"] += 1
    interval = int(spam_state.get("interval", 0))

    if interval > 0 and spam_state["global_step"] % interval == 0:
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
        spam_state["momentum_reset_count"] += 1
        spam_state["last_reset_step"] = spam_state["global_step"]
        if int(spam_state.get("warmup_steps", 0)) > 0:
            spam_state["warmup_until"] = spam_state["global_step"] + int(
                spam_state["warmup_steps"]
            )

    # Apply cosine warmup scaling to param group lrs if within warmup window
    warmup_until = int(spam_state.get("warmup_until", 0))
    if warmup_until > spam_state["global_step"]:
        remaining = warmup_until - spam_state["global_step"]
        total = int(spam_state.get("warmup_steps", 0))
        # cosine from small to 1.0
        t = (total - remaining) / max(total, 1)
        scale = 0.5 * (1 - np.cos(np.pi * t))  # 0 -> 1
        scale = max(1e-3, float(scale))
        for group in optimizer.param_groups:
            base_lr = group.get("base_lr", group["lr"])
            group["lr"] = base_lr * scale


def apply_adamprune_masking(optimizer, adamprune_state):
    """Apply AdamWPrune gradient masking."""
    if adamprune_state is None or not adamprune_state["pruning_enabled"]:
        return

    for module, mask in adamprune_state["masks"].items():
        if module.weight.grad is not None:
            module.weight.grad.data.mul_(mask.to(module.weight.grad.dtype))


def update_adamprune_masks(optimizer, adamprune_state, train_loader, epoch):
    """Update AdamWPrune pruning masks based on Adam states."""
    if adamprune_state is None or not adamprune_state["pruning_enabled"]:
        return

    # Always increment step count when called
    adamprune_state["step_count"] += 1

    # Update masks based on Adam states
    if (
        adamprune_state["step_count"] > adamprune_state["warmup_steps"]
        and adamprune_state["step_count"] % adamprune_state["pruning_frequency"] == 0
    ):

        # Calculate current sparsity level (gradual ramp-up)
        # Use ramp_end_epoch from state
        ramp_end_epoch = adamprune_state.get("ramp_end_epoch", 75)
        ramp_end_step = len(train_loader) * ramp_end_epoch
        progress = min(
            1.0,
            (adamprune_state["step_count"] - adamprune_state["warmup_steps"])
            / (ramp_end_step - adamprune_state["warmup_steps"]),
        )
        current_sparsity = adamprune_state["target_sparsity"] * progress

        # Compute importance scores using Adam states
        all_scores = []
        for module in adamprune_state["masks"].keys():
            # Get Adam states for this layer
            state = optimizer.state.get(module.weight, {})
            if "exp_avg" in state and "exp_avg_sq" in state:
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Hybrid strategy: combine momentum and stability signals
                # Momentum signal: weights moving strongly in one direction
                momentum_score = torch.abs(module.weight.data * exp_avg)

                # Stability signal: weights with consistent updates
                stability_score = torch.abs(module.weight.data) / (
                    torch.sqrt(exp_avg_sq) + 1e-8
                )

                # Combine both signals
                importance = momentum_score * stability_score
                all_scores.append(importance.flatten())
            else:
                # Fallback to magnitude if no Adam states yet
                all_scores.append(torch.abs(module.weight.data).flatten())

        if all_scores:
            all_scores = torch.cat(all_scores)

            # Find global threshold for target sparsity
            k = int(current_sparsity * all_scores.numel())
            if k > 0:
                threshold = torch.kthvalue(all_scores, k).values

                # Update masks
                for module in adamprune_state["masks"].keys():
                    state = optimizer.state.get(module.weight, {})
                    if "exp_avg" in state and "exp_avg_sq" in state:
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        momentum_score = torch.abs(module.weight.data * exp_avg)
                        stability_score = torch.abs(module.weight.data) / (
                            torch.sqrt(exp_avg_sq) + 1e-8
                        )
                        importance = momentum_score * stability_score
                    else:
                        importance = torch.abs(module.weight.data)

                    # Update boolean mask buffer
                    new_mask = importance > threshold
                    adamprune_state["masks"][module].data = new_mask.to(torch.bool)

                    # Apply mask to weights immediately
                    module.weight.data.mul_(
                        adamprune_state["masks"][module].to(module.weight.dtype)
                    )

    # Always apply existing masks to maintain sparsity
    elif adamprune_state["step_count"] > adamprune_state["warmup_steps"]:
        for module in adamprune_state["masks"].keys():
            module.weight.data.mul_(
                adamprune_state["masks"][module].to(module.weight.dtype)
            )

    # Also mask optimizer states to keep pruned weights inactive
    for module, mask in adamprune_state["masks"].items():
        state = optimizer.state.get(module.weight, {})
        if "exp_avg" in state:
            state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
