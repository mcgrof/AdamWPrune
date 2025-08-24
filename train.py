# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import logging
import json
from datetime import datetime
import os
import argparse
import numpy as np
from collections import deque

# Parse command line arguments
parser = argparse.ArgumentParser(description="LeNet-5 training with optional pruning")
parser.add_argument(
    "--pruning-method",
    type=str,
    default="none",
    choices=["none", "movement"],
    help="Pruning method to use (default: none)",
)
parser.add_argument(
    "--target-sparsity",
    type=float,
    default=0.9,
    help="Target sparsity for pruning (default: 0.9)",
)
parser.add_argument(
    "--pruning-warmup",
    type=int,
    default=100,
    help="Number of warmup steps before pruning starts (default: 100)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adam", "adamw", "adamwadv", "adamwspam", "adamwprune"],
    help="Optimizer to use for training (default: sgd)",
)
parser.add_argument(
    "--spam-theta",
    type=float,
    default=50.0,
    help="SPAM spike threshold theta for approximate GSS (default: 50.0)",
)
parser.add_argument(
    "--spam-interval",
    type=int,
    default=0,
    help="SPAM periodic momentum reset interval in steps (0 disables)",
)
parser.add_argument(
    "--spam-warmup-steps",
    type=int,
    default=0,
    help="SPAM cosine warmup steps after each reset (default: 0)",
)
parser.add_argument(
    "--spam-enable-clip",
    action="store_true",
    help="Enable SPAM spike-aware clipping using Adam's second moment",
)
parser.add_argument(
    "--json-output",
    type=str,
    default="training_metrics.json",
    help="json output file to use for stats, deafult is training_metrics.json",
)
args = parser.parse_args()

# Conditionally import pruning module
if args.pruning_method == "movement":
    from movement_pruning import MovementPruning

# Define relevant variables for the ML task
batch_size = 512
num_classes = 10
learning_rate = 0.001
num_epochs = 10
num_workers = 16  # Use multiple workers for data loading

# Movement pruning hyperparameters (when enabled)
enable_pruning = args.pruning_method != "none"
initial_sparsity = 0.0  # Start with no pruning
target_sparsity = args.target_sparsity
warmup_steps = args.pruning_warmup
pruning_frequency = 50  # Update masks every 50 steps
ramp_end_epoch = 8  # Reach target sparsity by epoch 8

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
log_filename = "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)

# Initialize metrics tracking
training_metrics = {
    "start_time": datetime.now().isoformat(),
    "config": {
        "batch_size": batch_size,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "pruning_method": args.pruning_method,
        "target_sparsity": target_sparsity if enable_pruning else None,
        "pruning_warmup": warmup_steps if enable_pruning else None,
        "optimizer": args.optimizer,
    },
    "epochs": [],
}

if device.type == "cuda":
    # Enable TensorFloat32 for faster matrix multiplication on GPUs that support it
    torch.set_float32_matmul_precision("high")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"Using GPU: {gpu_name}")
    logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    training_metrics["device"] = {
        "type": "cuda",
        "name": gpu_name,
        "memory_gb": gpu_memory,
    }
else:
    logger.info("Using CPU")
    training_metrics["device"] = {"type": "cpu"}

# Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    ),
    download=True,
)


test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
        ]
    ),
    download=True,
)


train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Pin memory for faster GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2,  # Prefetch batches
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size * 2,  # Larger batch for testing since no gradients
    shuffle=False,  # No need to shuffle test data
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = LeNet5(num_classes).to(device)

# Compile the model for faster execution (PyTorch 2.0+)
if torch.__version__ >= "2.0.0" and device.type == "cuda":
    logger.info("Compiling model with torch.compile()...")
    compile_start_time = time.time()
    model = torch.compile(model, mode="reduce-overhead")
    training_metrics["compile_time"] = time.time() - compile_start_time
    logger.info(
        "Model compilation completed after %.2fs", training_metrics["compile_time"]
    )
    training_metrics["model_compiled"] = True
else:
    training_metrics["model_compiled"] = False

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
scheduler = None
gradient_clip_norm = None
spam_state = None  # For SPAM optimizer state tracking
adamprune_state = None  # For AdamWPrune state-based pruning

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Using Adam optimizer")
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    logger.info("Using AdamW optimizer with weight decay=0.0001")
elif args.optimizer == "adamwadv":
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
elif args.optimizer == "adamwspam":
    # AdamW with SPAM (Spike-Aware Pruning-Adaptive Momentum)
    # Reference (for future fidelity alignment):
    # SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training
    # https://arxiv.org/abs/2501.06842
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
        "spike_threshold": 2.0,  # Detect spikes > 2 std deviations (heuristic)
        "spike_events": [],
        "last_reset_step": 0,
        "global_step": 0,
        "momentum_reset_count": 0,
        # SPAM fidelity options
        "theta": args.spam_theta,
        "interval": args.spam_interval,
        "warmup_steps": args.spam_warmup_steps,
        "enable_clip": args.spam_enable_clip,
        "warmup_until": 0,
    }
    # Cosine annealing with SPAM awareness
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    gradient_clip_norm = 1.0
    logger.info("Using AdamW SPAM optimizer (heuristic + SPAM-inspired options) with:")
    logger.info("  - Spike detection (threshold=2.0 std)")
    logger.info("  - Automatic momentum reset on spikes")
    logger.info("  - AMSGrad for stability")
    logger.info("  - Adaptive gradient clipping")
    logger.info("  - Cosine annealing LR schedule")
    if spam_state["interval"] > 0:
        logger.info(f"  - Periodic momentum reset interval: {spam_state['interval']} steps")
        if spam_state["warmup_steps"] > 0:
            logger.info(f"  - Cosine warmup after reset: {spam_state['warmup_steps']} steps")
    if spam_state["enable_clip"]:
        logger.info(f"  - Spike-aware clipping enabled (theta={spam_state['theta']})")
elif args.optimizer == "adamwprune":
    # AdamWPrune - Experimental: All enhancements + state-based pruning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=True,
    )

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
        "theta": args.spam_theta,
        "interval": args.spam_interval,
        "warmup_steps": args.spam_warmup_steps,
        "enable_clip": args.spam_enable_clip,
        "warmup_until": 0,
    }

    # AdamWPrune specific state for state-based pruning
    adamprune_state = {
        "pruning_enabled": args.pruning_method == "movement",
        "target_sparsity": (
            args.target_sparsity if args.pruning_method == "movement" else 0
        ),
        "warmup_steps": args.pruning_warmup,
        "pruning_frequency": 50,
        "step_count": 0,
        "masks": {},  # module -> bool mask buffer
        "pruning_strategy": "hybrid",  # hybrid of momentum and stability
    }

    # Initialize masks for prunable layers as boolean buffers to minimize memory
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
        logger.info(f"  - Target sparsity: {adamprune_state['target_sparsity']:.1%}")
        logger.info(f"  - Pruning warmup: {adamprune_state['warmup_steps']} steps")
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    logger.info("Using SGD optimizer with momentum=0.9")

# Initialize GradScaler for mixed precision training
scaler = GradScaler("cuda")

# Initialize pruning if enabled
pruner = None
if enable_pruning and args.pruning_method == "movement" and args.optimizer != "adamwprune":
    # Calculate total training steps for pruning schedule
    total_training_steps = len(train_loader) * ramp_end_epoch
    pruner = MovementPruning(
        model=model,
        initial_sparsity=initial_sparsity,
        target_sparsity=target_sparsity,
        warmup_steps=warmup_steps,
        pruning_frequency=pruning_frequency,
        ramp_end_step=total_training_steps,
    )
    logger.info(f"Movement pruning enabled with target sparsity: {target_sparsity}")
    logger.info(
        f"Pruning warmup steps: {warmup_steps}, ramp end: {total_training_steps}"
    )
    training_metrics["pruning"] = {
        "method": "movement",
        "initial_sparsity": initial_sparsity,
        "target_sparsity": target_sparsity,
        "warmup_steps": warmup_steps,
        "ramp_end_step": total_training_steps,
    }

# this is defined to print how many steps are remaining when training
total_step = len(train_loader)

# GPU warmup
if device.type == "cuda":
    logger.info("Warming up GPU...")
    gpu_warmup_start_time = time.time()
    dummy_input = torch.randn(batch_size, 1, 32, 32, dtype=torch.float).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    training_metrics["gpu_warmup_time"] = time.time() - gpu_warmup_start_time
    logger.info("GPU warmed up for %.2fs", training_metrics["gpu_warmup_time"])

logger.info(f"Starting training with batch size: {batch_size}")
logger.info(f"Total training samples: {len(train_dataset)}")
logger.info(f"Total test samples: {len(test_dataset)}")
training_metrics["dataset"] = {
    "train_samples": len(train_dataset),
    "test_samples": len(test_dataset),
}

total_step = len(train_loader)
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    running_loss = 0.0
    epoch_metrics = {
        "epoch": epoch + 1,
        "losses": [],
        "batch_times": [],
    }

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass (use autocast only on CUDA)
        if device.type == "cuda":
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = cost(outputs, labels)
        else:
            outputs = model(images)
            loss = cost(outputs, labels)

        # Backward and optimize with gradient scaling
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        scaler.scale(loss).backward()

        # Apply gradient masking (AdamWPrune) and gradient clipping if enabled
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)

            # SPAM-inspired spike-aware clipping using Adam's second moment (approximate GSS)
            if spam_state is not None and spam_state.get("enable_clip", False):
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

            # If using AdamWPrune, zero gradients on pruned weights before clipping
            if adamprune_state is not None and adamprune_state["pruning_enabled"]:
                for module, mask in adamprune_state["masks"].items():
                    if module.weight.grad is not None:
                        module.weight.grad.data.mul_(mask.to(module.weight.grad.dtype))

            # SPAM spike detection and momentum reset
            if spam_state is not None:
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

                        if (
                            std_norm > 0
                            and np.isfinite(mean_norm)
                            and np.isfinite(std_norm)
                        ):
                            z_score = (total_norm - mean_norm) / std_norm

                            # Spike detected - reset momentum
                            if z_score > spam_state["spike_threshold"] and np.isfinite(
                                z_score
                            ):
                                for group in optimizer.param_groups:
                                    for p in group["params"]:
                                        state = optimizer.state.get(p, {})
                                        if "exp_avg" in state:
                                            state["exp_avg"].mul_(0.5)  # Soft reset
                                        if "exp_avg_sq" in state:
                                            state["exp_avg_sq"].mul_(
                                                0.9
                                            )  # Gentle reset

                                spam_state["spike_events"].append(
                                    spam_state["global_step"]
                                )
                                spam_state["momentum_reset_count"] += 1
                                spam_state["last_reset_step"] = spam_state[
                                    "global_step"
                                ]

                                if (i + 1) % 10 == 0:  # Log significant spikes
                                    logger.info(
                                        f"SPAM: Spike detected (z={z_score:.2f}), momentum reset #{spam_state['momentum_reset_count']}"
                                    )

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        # Periodic SPAM momentum reset (first/second moments), with optional warmup
        if spam_state is not None:
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

        scaler.step(optimizer)
        scaler.update()

        # Apply AdamWPrune state-based pruning
        if adamprune_state is not None and adamprune_state["pruning_enabled"]:
            adamprune_state["step_count"] += 1

            # Update masks based on Adam states
            if (
                adamprune_state["step_count"] > adamprune_state["warmup_steps"]
                and adamprune_state["step_count"] % adamprune_state["pruning_frequency"]
                == 0
            ):

                # Calculate current sparsity level (gradual ramp-up)
                ramp_end_step = len(train_loader) * 8  # Ramp to target by epoch 8
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
                            new_mask = (importance > threshold)
                            adamprune_state["masks"][module].data = new_mask.to(torch.bool)

                            # Apply mask to weights immediately
                            module.weight.data.mul_(adamprune_state["masks"][module].to(module.weight.dtype))

            # Always apply existing masks to maintain sparsity
            elif adamprune_state["step_count"] > adamprune_state["warmup_steps"]:
                for module in adamprune_state["masks"].keys():
                    module.weight.data.mul_(adamprune_state["masks"][module].to(module.weight.dtype))

            # Also mask optimizer states to keep pruned weights inactive
            for module, mask in adamprune_state["masks"].items():
                state = optimizer.state.get(module.weight, {})
                if "exp_avg" in state:
                    state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))

        # Apply movement pruning if enabled (for non-AdamWPrune optimizers)
        elif pruner is not None:
            pruner.step_pruning()

        running_loss += loss.item()

        # Let's see the hockey stick.
        #
        # We want to be able to track the first epoch well. The ideal
        # loss will depend on the number of classes. For Lenet-5 we have
        # 10 classes, so we have a uniform probability (1/10 chance for each
        # class), the expected the cross-entropy loss would be:
        #
        # -log(1/10) = -log(0.1) = 2.303
        #
        # So that's the worst case los swe expect on initialization.
        if epoch == 0:
            if i < 10:
                print_at_steps = 1
            else:
                print_at_steps = 10
        else:
            print_at_steps = 118

        if (i + 1) % print_at_steps == 0:
            avg_loss = running_loss / print_at_steps
            batch_time = time.time() - epoch_start
            logger.info(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f}s".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    total_step,
                    avg_loss,
                    batch_time,
                )
            )
            epoch_metrics["losses"].append(avg_loss)
            epoch_metrics["batch_times"].append(batch_time)
            running_loss = 0.0

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start

        # Log AdamWPrune sparsity if enabled
        if adamprune_state is not None and adamprune_state["pruning_enabled"]:
            total_params = 0
            zero_params = 0
            for mask in adamprune_state["masks"].values():
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()

            global_sparsity = zero_params / total_params if total_params > 0 else 0
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, "
                f"Test Accuracy: {accuracy:.2f}%, Sparsity: {global_sparsity:.2%}"
            )
            epoch_metrics["sparsity"] = global_sparsity
            epoch_metrics["sparsity_stats"] = {"global": {"sparsity": global_sparsity}}
        # Log pruning statistics if enabled
        elif pruner is not None:
            sparsity_stats = pruner.get_sparsity_stats()
            global_sparsity = sparsity_stats["global"]["sparsity"]
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, "
                f"Test Accuracy: {accuracy:.2f}%, Sparsity: {global_sparsity:.2%}"
            )
            epoch_metrics["sparsity"] = global_sparsity
            epoch_metrics["sparsity_stats"] = sparsity_stats
        else:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s, Test Accuracy: {accuracy:.2f}%"
            )

        # Store epoch metrics
        epoch_metrics["accuracy"] = accuracy
        epoch_metrics["epoch_time"] = epoch_time
        epoch_metrics["avg_loss"] = (
            sum(epoch_metrics["losses"]) / len(epoch_metrics["losses"])
            if epoch_metrics["losses"]
            else 0
        )
        training_metrics["epochs"].append(epoch_metrics)

        # Step the learning rate scheduler if using AdamWAdv
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Learning rate adjusted to: {current_lr:.6f}")

total_time = time.time() - start_time
logger.info(f"Training completed in {total_time:.2f} seconds")
logger.info(f"Average time per epoch: {total_time/num_epochs:.2f} seconds")

# Log SPAM statistics if using SPAM optimizer
if spam_state is not None:
    logger.info(f"SPAM Statistics:")
    logger.info(f"  - Total momentum resets: {spam_state['momentum_reset_count']}")
    logger.info(f"  - Spike events: {len(spam_state['spike_events'])}")
    training_metrics["spam_stats"] = {
        "momentum_resets": spam_state["momentum_reset_count"],
        "spike_events": len(spam_state["spike_events"]),
        "spike_steps": spam_state["spike_events"][:10],  # First 10 spikes for analysis
    }

# Log final AdamWPrune statistics
if adamprune_state is not None and adamprune_state["pruning_enabled"]:
    # Apply final pruning
    total_params = 0
    zero_params = 0
    for module, mask in adamprune_state["masks"].items():
        module.weight.data.mul_(mask.to(module.weight.dtype))
        total_params += mask.numel()
        zero_params += (mask == 0).sum().item()

    final_sparsity = zero_params / total_params if total_params > 0 else 0
    logger.info(f"Final AdamWPrune sparsity: {final_sparsity:.2%}")

    # Count remaining parameters
    total_model_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    logger.info(f"Total parameters: {total_model_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info(f"Compression ratio: {total_model_params/non_zero_params:.2f}x")

    training_metrics["final_sparsity"] = final_sparsity
    training_metrics["total_params"] = total_model_params
    training_metrics["non_zero_params"] = non_zero_params
# Log final pruning statistics
elif pruner is not None:
    final_sparsity = pruner.prune_model_final()
    logger.info(f"Final model sparsity: {final_sparsity:.2%}")

    # Count remaining parameters
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info(f"Compression ratio: {total_params/non_zero_params:.2f}x")

    training_metrics["final_sparsity"] = final_sparsity
    training_metrics["total_params"] = total_params
    training_metrics["non_zero_params"] = non_zero_params

# Final metrics
training_metrics["total_training_time"] = total_time
training_metrics["avg_time_per_epoch"] = total_time / num_epochs
training_metrics["final_accuracy"] = accuracy
training_metrics["end_time"] = datetime.now().isoformat()

# Save the model. This uses buffered IO, so we flush to measure how long
# it takes to save with writeback incurred to the filesystem.
model_filename = "lenet5_pruned.pth" if enable_pruning else "lenet5_optimized.pth"
save_start_time = time.time()
with open(model_filename, "wb") as f:
    torch.save(model.state_dict(), f)
    f.flush()
    os.fsync(f.fileno())
training_metrics["save_time"] = time.time() - save_start_time
training_metrics["save_size"] = os.path.getsize(model_filename)

logger.info(
    "Model saved as %s (%.2f MB) in %.2fs",
    model_filename,
    training_metrics["save_size"] / (1024 * 1024),
    training_metrics["save_time"],
)

# Save metrics to JSON for plotting
with open(args.json_output, "w") as f:
    json.dump(training_metrics, f, indent=2)
logger.info(f"Training metrics saved to {args.json_output}")

# Training complete
logger.info("Training script finished successfully")

# We use persistent_workers=True to keep workers accross epochs, so to not
# have to spawn  new ones. When we finish triajning we just need to clean up
# persistent workers explicitly, otherwise this will delay the exit.
del train_loader
del test_loader
