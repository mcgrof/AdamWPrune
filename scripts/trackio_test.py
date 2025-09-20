#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Simple Trackio integration test for Kconfig.
Submits fake training data for a dummy model to verify Trackio integration.
"""

import sys
import time
import random
import math
import numpy as np

try:
    import trackio
except ImportError:
    print("Error: trackio not installed. Install with: pip install trackio")
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
        TRACKER_PROJECT = "adamwprune-test"
        TRACKIO_PORT = 7860
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

    # Memory usage (simulated)
    memory_mb = 2000 + random.randint(-100, 100)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "sparsity": sparsity,
        "learning_rate": lr,
        "perplexity": perplexity,
        "memory_mb": memory_mb,
        "epoch": epoch,
        "step": step,
    }


def main():
    """Run Trackio integration test."""
    print("=" * 60)
    print("Trackio Integration Test")
    print("=" * 60)

    # Get configuration
    project = getattr(config, "TRACKER_PROJECT", None)
    if not project:
        # Try to get auto-generated name
        project = getattr(config, "AUTO_PROJECT_NAME", None)
    if not project:
        # Fallback to default
        project = "adamwprune-test"
    port = getattr(config, "TRACKIO_PORT", 7860)

    # Initialize Trackio
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
    print(f"Dashboard port: {port}")
    print(f"Config: {run_config}")
    print()

    try:
        # Initialize Trackio with project and config
        trackio.init(
            project=project,
            config=run_config,
            name=f"test-run-{int(time.time())}",
            tags=["test", "fake-data", "kconfig-integration"],
        )
        print("✓ Trackio initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Trackio: {e}")
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

            # Log to Trackio
            trackio.log(metrics)

            # Print progress every 20 steps
            if step % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {step+1}/{steps_per_epoch}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"sparsity={metrics['sparsity']:.2%}, "
                    f"mem={metrics['memory_mb']:.0f}MB"
                )

            # Small delay to simulate training
            time.sleep(0.01)

        # Log epoch summary
        epoch_loss = np.mean([m["loss"] for m in epoch_metrics])
        epoch_acc = np.mean([m["accuracy"] for m in epoch_metrics])
        epoch_memory = np.mean([m["memory_mb"] for m in epoch_metrics])

        trackio.log(
            {
                "epoch_loss": epoch_loss,
                "epoch_accuracy": epoch_acc,
                "epoch_memory_mb": epoch_memory,
                "epoch": epoch + 1,
            }
        )

        print(
            f"Epoch {epoch+1} complete: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, mem={epoch_memory:.0f}MB"
        )

    # Log final metrics
    final_metrics = {
        "final_loss": epoch_metrics[-1]["loss"],
        "final_accuracy": epoch_metrics[-1]["accuracy"],
        "final_sparsity": epoch_metrics[-1]["sparsity"],
        "best_accuracy": max(m["accuracy"] for m in epoch_metrics),
        "min_loss": min(m["loss"] for m in epoch_metrics),
        "total_steps": num_epochs * steps_per_epoch,
    }

    # Log final summary to Trackio
    for key, value in final_metrics.items():
        trackio.log({key: value})

    print("\n" + "=" * 40)
    print("Final Metrics:")
    for key, value in final_metrics.items():
        if "sparsity" in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.4f}")

    # Finish Trackio run
    trackio.finish()

    print("\n✓ Trackio test completed successfully!")
    print(f"\nTo view the results, run:")
    print(f"  trackio show --port {port}")
    print(f"\nThen open: http://localhost:{port}")
    print("\nNote: The dashboard will show all tracked runs in the project.")
    print("      Look for the run named 'test-run-<timestamp>' with tag 'test'.")


if __name__ == "__main__":
    main()
