#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Simple WandB integration test for Kconfig.
Submits fake training data for a dummy model to verify WandB integration.
"""

import sys
import time
import random
import math
import numpy as np

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

# Try to import config from Kconfig
try:
    sys.path.append(".")
    from config import config

    has_config = True
except ImportError:
    print("Warning: config.py not found, using defaults")
    has_config = False

    class DummyConfig:
        WANDB_PROJECT = "adamwprune-test"
        WANDB_ENTITY = None
        LEARNING_RATE = "0.001"
        BATCH_SIZE = 32
        NUM_EPOCHS = 10
        OPTIMIZER = "adamw"
        MODEL = "dummy_model"
        PRUNING_METHOD = "magnitude"
        TARGET_SPARSITY = "0.5"

    config = DummyConfig()


def generate_fake_metrics(epoch, step):
    """Generate realistic-looking fake metrics."""
    # Simulate learning curves
    progress = (epoch * 100 + step) / 1000.0

    # Loss decreases over time with noise
    loss = 2.5 * math.exp(-progress * 2) + random.gauss(0.1, 0.02)

    # Accuracy increases over time
    accuracy = min(0.95, 0.1 + 0.85 * (1 - math.exp(-progress * 1.5)))
    accuracy += random.gauss(0, 0.01)

    # Sparsity increases gradually
    if hasattr(config, "TARGET_SPARSITY"):
        target = float(config.TARGET_SPARSITY)
    else:
        target = 0.5
    sparsity = min(target, target * (1 - math.exp(-progress * 3)))

    # Learning rate with decay
    base_lr = float(getattr(config, "LEARNING_RATE", "0.001"))
    lr = base_lr * math.exp(-progress * 0.5)

    # Perplexity for language models
    perplexity = math.exp(loss)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "sparsity": sparsity,
        "learning_rate": lr,
        "perplexity": perplexity,
        "epoch": epoch,
        "step": step,
    }


def main():
    """Run WandB integration test."""
    print("=" * 60)
    print("WandB Integration Test")
    print("=" * 60)

    # Get configuration
    project = getattr(config, "WANDB_PROJECT", "adamwprune-test")
    entity = getattr(config, "WANDB_ENTITY", None)

    # Initialize WandB
    run_config = {
        "model": getattr(config, "MODEL", "dummy_model"),
        "optimizer": getattr(config, "OPTIMIZER", "adamw"),
        "learning_rate": float(getattr(config, "LEARNING_RATE", "0.001")),
        "batch_size": getattr(config, "BATCH_SIZE", 32),
        "num_epochs": getattr(config, "NUM_EPOCHS", 10),
        "pruning_method": getattr(config, "PRUNING_METHOD", "magnitude"),
        "target_sparsity": float(getattr(config, "TARGET_SPARSITY", "0.5")),
        "test_mode": True,
        "fake_data": True,
    }

    print(f"Project: {project}")
    print(f"Entity: {entity or 'default'}")
    print(f"Config: {run_config}")
    print()

    try:
        wandb.init(
            project=project,
            entity=entity,
            config=run_config,
            name=f"test-run-{int(time.time())}",
            tags=["test", "fake-data", "kconfig-integration"],
        )
        print("✓ WandB initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize WandB: {e}")
        sys.exit(1)

    # Simulate training
    print("\nSimulating training...")
    print("-" * 40)

    num_epochs = run_config["num_epochs"]
    steps_per_epoch = 100

    for epoch in range(num_epochs):
        epoch_metrics = []

        for step in range(steps_per_epoch):
            # Generate fake metrics
            metrics = generate_fake_metrics(epoch, step)
            epoch_metrics.append(metrics)

            # Log to WandB
            wandb.log(metrics)

            # Print progress every 20 steps
            if step % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {step+1}/{steps_per_epoch}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"sparsity={metrics['sparsity']:.2%}"
                )

            # Small delay to simulate training
            time.sleep(0.01)

        # Log epoch summary
        epoch_loss = np.mean([m["loss"] for m in epoch_metrics])
        epoch_acc = np.mean([m["accuracy"] for m in epoch_metrics])

        wandb.log(
            {
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_acc,
                "epoch": epoch + 1,
            }
        )

        print(f"Epoch {epoch+1} complete: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    # Log final metrics
    final_metrics = {
        "final_loss": epoch_metrics[-1]["loss"],
        "final_accuracy": epoch_metrics[-1]["accuracy"],
        "final_sparsity": epoch_metrics[-1]["sparsity"],
        "best_accuracy": max(m["accuracy"] for m in epoch_metrics),
        "total_steps": num_epochs * steps_per_epoch,
    }

    wandb.summary.update(final_metrics)

    print("\n" + "=" * 40)
    print("Final Metrics:")
    for key, value in final_metrics.items():
        if "sparsity" in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.4f}")

    # Finish WandB run
    wandb.finish()

    print("\n✓ WandB test completed successfully!")
    print(f"View run at: {wandb.run.get_url() if wandb.run else 'N/A'}")


if __name__ == "__main__":
    main()
