#!/usr/bin/env python3
"""
Unified metrics extraction for both CNN and GPT-2 models.
Handles accuracy for CNNs and perplexity for language models.
"""

import json
import os
import math


def extract_metrics(test_dir):
    """Extract metrics from a test directory, handling both CNN and GPT-2 formats."""
    metrics_file = os.path.join(test_dir, "training_metrics.json")

    if not os.path.exists(metrics_file):
        return None

    with open(metrics_file, "r") as f:
        data = json.load(f)

    result = {
        "success": True,
        "model": data.get("config", {}).get("model", "unknown"),
    }

    # Determine if this is a language model (GPT-2) or CNN
    is_gpt2 = (
        "gpt2" in test_dir.lower() or data.get("config", {}).get("model") == "gpt2"
    )

    if is_gpt2:
        # GPT-2 metrics: use perplexity
        result["metric_type"] = "perplexity"

        # Get perplexity values
        if "best_perplexity" in data:
            result["best_metric"] = data["best_perplexity"]
            result["final_metric"] = data.get(
                "final_perplexity", data["best_perplexity"]
            )
        elif "best_val_loss" in data:
            # Calculate perplexity from loss
            best_loss = data["best_val_loss"]
            final_loss = data.get("final_val_loss", best_loss)
            result["best_metric"] = math.exp(min(best_loss, 20))
            result["final_metric"] = math.exp(min(final_loss, 20))
        else:
            result["best_metric"] = float("inf")
            result["final_metric"] = float("inf")

        # Get delta PPL
        result["delta_ppl"] = data.get("delta_ppl", 0.0)

        # Get latency measurements
        result["latency_seq512_p50"] = data.get("latency_seq512_p50", -1)
        result["latency_seq512_p95"] = data.get("latency_seq512_p95", -1)
        result["latency_seq1024_p50"] = data.get("latency_seq1024_p50", -1)
        result["latency_seq1024_p95"] = data.get("latency_seq1024_p95", -1)

        # Memory measurements
        result["gpu_memory_allocated_mb"] = data.get("gpu_memory_allocated_mb", 0)
        result["gpu_memory_reserved_mb"] = data.get("gpu_memory_reserved_mb", 0)

    else:
        # CNN metrics: use accuracy
        result["metric_type"] = "accuracy"

        # Try different field names for accuracy
        if "test_accuracy" in data and data["test_accuracy"]:
            result["best_metric"] = max(data["test_accuracy"]) * 100
            result["final_metric"] = data["test_accuracy"][-1] * 100
        elif "final_accuracy" in data:
            result["best_metric"] = data.get("best_accuracy", data["final_accuracy"])
            result["final_metric"] = data["final_accuracy"]
        else:
            # Try to extract from epochs data
            epochs = data.get("epochs", [])
            if epochs and isinstance(epochs[0], dict):
                accs = [
                    e.get("test_accuracy", 0) for e in epochs if "test_accuracy" in e
                ]
                if accs:
                    result["best_metric"] = max(accs) * 100
                    result["final_metric"] = accs[-1] * 100
                else:
                    result["best_metric"] = 0.0
                    result["final_metric"] = 0.0
            else:
                result["best_metric"] = 0.0
                result["final_metric"] = 0.0

    # Get sparsity
    if "sparsities" in data and data["sparsities"]:
        result["final_sparsity"] = data["sparsities"][-1]
    else:
        result["final_sparsity"] = 0.0

    # Get training time
    result["total_time"] = data.get("total_time", 0)

    return result


def format_metric(value, metric_type):
    """Format metric value based on type."""
    if metric_type == "accuracy":
        return f"{value:.2f}%"
    elif metric_type == "perplexity":
        if value == float("inf"):
            return "∞"
        return f"{value:.2f}"
    else:
        return f"{value:.2f}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: extract_metrics.py <test_directory>")
        sys.exit(1)

    test_dir = sys.argv[1]
    metrics = extract_metrics(test_dir)

    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("No metrics found")
