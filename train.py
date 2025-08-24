# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
import time
import logging
import json
from datetime import datetime
import os
import argparse

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize GradScaler for mixed precision training
scaler = GradScaler("cuda")

# Initialize pruning if enabled
pruner = None
if enable_pruning and args.pruning_method == "movement":
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
        scaler.step(optimizer)
        scaler.update()

        # Apply movement pruning if enabled
        if pruner is not None:
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

        # Log pruning statistics if enabled
        if pruner is not None:
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

total_time = time.time() - start_time
logger.info(f"Training completed in {total_time:.2f} seconds")
logger.info(f"Average time per epoch: {total_time/num_epochs:.2f} seconds")

# Log final pruning statistics
if pruner is not None:
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
with open("training_metrics.json", "w") as f:
    json.dump(training_metrics, f, indent=2)
logger.info("Training metrics saved to training_metrics.json")

# Training complete
logger.info("Training script finished successfully")

# We use persistent_workers=True to keep workers accross epochs, so to not
# have to spawn  new ones. When we finish triajning we just need to clean up
# persistent workers explicitly, otherwise this will delay the exit.
del train_loader
del test_loader
