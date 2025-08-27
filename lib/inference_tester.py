#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Inference testing and memory measurement module.
Measures actual GPU memory usage during inference with pruned models.
"""

import torch
import torch.nn as nn
import time
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import os
import signal
import atexit
import threading

logger = logging.getLogger(__name__)


class GPUMonitor:
    """GPU monitoring wrapper for gputop during inference testing."""

    def __init__(self, output_file: str, model_name: str = "inference"):
        self.output_file = output_file
        self.model_name = model_name
        self.process = None
        self.monitor_thread = None
        self.should_stop = False

    def start(self):
        """Start GPU monitoring."""
        try:
            # First try gputop if available
            try:
                subprocess.run(["which", "gputop"], check=True, capture_output=True)
                cmd = [
                    "gputop",
                    "--json",
                    self.output_file,
                    "--interval",
                    "100",  # 100ms intervals
                    "--model",
                    self.model_name,
                ]
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to our monitor_gpu.py script
                monitor_script = (
                    Path(__file__).parent.parent / "scripts" / "monitor_gpu.py"
                )
                if monitor_script.exists():
                    cmd = [
                        "python",
                        str(monitor_script),
                        self.output_file,
                        "100",  # 100ms intervals
                        self.model_name,
                    ]
                else:
                    logger.warning("No GPU monitoring tool available")
                    return False

            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            logger.info(
                f"Started GPU monitoring (PID: {self.process.pid}, output: {self.output_file})"
            )
            return True

        except Exception as e:
            logger.warning(f"GPU monitoring disabled: {e}")
            return False

    def stop(self):
        """Stop GPU monitoring."""
        if self.process:
            try:
                # Send SIGTERM for graceful shutdown
                self.process.terminate()
                self.process.wait(timeout=2)
                logger.info(f"Stopped GPU monitoring (PID: {self.process.pid})")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.warning(f"Error stopping GPU monitor: {e}")
            finally:
                self.process = None


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.5)


def count_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """Count model parameters and sparsity."""
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0

    return {
        "total_params": total_params,
        "zero_params": zero_params,
        "dense_params": total_params - zero_params,
        "sparsity": sparsity,
        "compression_ratio": 1.0 / (1.0 - sparsity) if sparsity < 1.0 else float("inf"),
    }


def measure_inference_memory(
    model: nn.Module,
    input_shape: tuple,
    batch_sizes: List[int] = [1, 32, 128],
    warmup_runs: int = 3,
    measure_runs: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Measure GPU memory usage during inference.

    Args:
        model: The model to test
        input_shape: Shape of single input (e.g., (3, 32, 32) for CIFAR)
        batch_sizes: List of batch sizes to test
        warmup_runs: Number of warmup iterations
        measure_runs: Number of measurement iterations
        device: Device to run on (default: cuda if available)

    Returns:
        Dictionary with inference memory measurements
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Count sparsity
    sparsity_info = count_model_sparsity(model)

    results = {"model_info": sparsity_info, "measurements": [], "device": str(device)}

    for batch_size in batch_sizes:
        logger.info(f"Testing batch size {batch_size}...")

        # Clear memory
        clear_gpu_memory()

        # Measure baseline memory (model only)
        mem_baseline = get_gpu_memory_mb()

        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        mem_with_input = get_gpu_memory_mb()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Measure inference memory
        memory_measurements = []
        inference_times = []

        with torch.no_grad():
            for _ in range(measure_runs):
                clear_gpu_memory()
                dummy_input = torch.randn(batch_size, *input_shape).to(device)

                mem_before = get_gpu_memory_mb()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                output = model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                mem_after = get_gpu_memory_mb()

                memory_measurements.append(mem_after)
                inference_times.append((end_time - start_time) * 1000)  # ms

        # Calculate statistics
        import numpy as np

        mem_mean = np.mean(memory_measurements)
        mem_std = np.std(memory_measurements)
        mem_max = np.max(memory_measurements)
        mem_min = np.min(memory_measurements)

        time_mean = np.mean(inference_times)
        time_std = np.std(inference_times)

        batch_results = {
            "batch_size": batch_size,
            "memory_mb": {
                "baseline": mem_baseline,
                "with_input": mem_with_input,
                "mean": mem_mean,
                "std": mem_std,
                "max": mem_max,
                "min": mem_min,
                "overhead": mem_mean - mem_baseline,
            },
            "inference_time_ms": {
                "mean": time_mean,
                "std": time_std,
                "throughput": (batch_size / time_mean) * 1000,  # samples/sec
            },
        }

        results["measurements"].append(batch_results)

        logger.info(
            f"  Memory: {mem_mean:.1f}±{mem_std:.1f} MB (overhead: {mem_mean - mem_baseline:.1f} MB)"
        )
        logger.info(
            f"  Time: {time_mean:.2f}±{time_std:.2f} ms (throughput: {batch_results['inference_time_ms']['throughput']:.1f} samples/sec)"
        )

    return results


def compare_inference_models(
    models: Dict[str, nn.Module],
    input_shape: tuple,
    batch_sizes: List[int] = [1, 32, 128],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare inference memory across multiple models.

    Args:
        models: Dictionary of {name: model} to compare
        input_shape: Input shape for the models
        batch_sizes: Batch sizes to test
        output_path: Optional path to save results JSON

    Returns:
        Comparison results dictionary
    """
    comparison_results = {}

    for name, model in models.items():
        logger.info(f"\nTesting model: {name}")
        results = measure_inference_memory(model, input_shape, batch_sizes)
        comparison_results[name] = results

        # Print summary
        sparsity = results["model_info"]["sparsity"]
        logger.info(f"  Sparsity: {sparsity:.1%}")
        logger.info(f"  Total params: {results['model_info']['total_params']:,}")
        logger.info(f"  Zero params: {results['model_info']['zero_params']:,}")

    # Save results if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"\nSaved inference comparison to {output_path}")

    return comparison_results


def run_inference_test(
    model: nn.Module,
    model_name: str,
    input_shape: tuple,
    batch_sizes: Optional[str] = None,
    save_path: Optional[Path] = None,
    enable_gpu_monitor: bool = True,
) -> Dict[str, Any]:
    """
    Run inference test on a single model (for integration with training scripts).

    Args:
        model: Trained model to test
        model_name: Name of the model
        input_shape: Input shape (e.g., (3, 32, 32))
        batch_sizes: Comma-separated string of batch sizes (e.g., "1,32,128")
        save_path: Optional path to save results

    Returns:
        Inference test results
    """
    # Parse batch sizes
    if batch_sizes:
        batch_list = [int(x.strip()) for x in batch_sizes.split(",")]
    else:
        batch_list = [1, 32, 128]

    logger.info(f"\n{'='*60}")
    logger.info(f"Running inference test for {model_name}")
    logger.info(f"{'='*60}")

    # Start GPU monitoring if enabled
    gpu_monitor = None
    gpu_monitor_file = None
    if enable_gpu_monitor:
        # Create monitoring output file path
        if save_path:
            base_path = (
                Path(save_path).parent
                if isinstance(save_path, (str, Path))
                else Path(".")
            )
            gpu_monitor_file = base_path / f"gpu_inference_{model_name}.json"
        else:
            gpu_monitor_file = Path(f"gpu_inference_{model_name}.json")

        gpu_monitor = GPUMonitor(str(gpu_monitor_file), model_name)
        monitor_started = gpu_monitor.start()

        if monitor_started:
            # Give monitor time to initialize
            time.sleep(0.5)

    # Run inference memory measurements
    results = measure_inference_memory(model, input_shape, batch_list)

    # Stop GPU monitoring
    if gpu_monitor:
        time.sleep(0.5)  # Ensure final measurements are captured
        gpu_monitor.stop()

        # Add GPU monitor file path to results
        results["gpu_monitor_file"] = str(gpu_monitor_file)
        logger.info(f"GPU monitoring data saved to {gpu_monitor_file}")

    # Add model name to results
    results["model_name"] = model_name

    # Print summary
    logger.info(f"\nInference Test Summary:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Sparsity: {results['model_info']['sparsity']:.1%}")
    logger.info(f"  Parameters: {results['model_info']['total_params']:,}")
    logger.info(f"  Dense parameters: {results['model_info']['dense_params']:,}")

    logger.info(f"\nMemory Usage by Batch Size:")
    for measurement in results["measurements"]:
        bs = measurement["batch_size"]
        mem = measurement["memory_mb"]["mean"]
        overhead = measurement["memory_mb"]["overhead"]
        throughput = measurement["inference_time_ms"]["throughput"]
        logger.info(
            f"  Batch {bs:3d}: {mem:6.1f} MB (overhead: {overhead:5.1f} MB, "
            f"throughput: {throughput:6.1f} samples/sec)"
        )

    # Save if requested
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved inference results to {save_path}")

    return results


# Example usage
if __name__ == "__main__":
    # Test with a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.fc = nn.Linear(128 * 28 * 28, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    logging.basicConfig(level=logging.INFO)

    model = SimpleModel()

    # Apply some pruning (set 50% weights to zero)
    with torch.no_grad():
        for param in model.parameters():
            mask = torch.rand_like(param) > 0.5
            param.mul_(mask.float())

    results = run_inference_test(
        model,
        "SimpleModel",
        input_shape=(3, 32, 32),
        batch_sizes="1,16,32",
        save_path=Path("test_inference_results.json"),
    )
