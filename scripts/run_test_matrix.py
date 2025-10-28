#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Test matrix runner for AdamWPrune neural network training.
Parses Kconfig settings and runs all combinations of selected models, optimizers, and pruning methods.
"""

import os
import sys
import subprocess
import json
import itertools
import argparse
from pathlib import Path
import time
from datetime import datetime
import shutil
import yaml
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global lock for thread-safe printing and result handling
print_lock = threading.Lock()


def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_time_estimates(completed_results, remaining_tests):
    """Calculate time estimates based on completed tests, accounting for variants."""
    if not completed_results:
        return {}

    # Calculate average time per test from completed results
    successful_times = [
        r["elapsed_time"]
        for r in completed_results
        if r.get("success", False) and "elapsed_time" in r
    ]

    if not successful_times:
        return {}

    avg_time_per_test = sum(successful_times) / len(successful_times)

    return {
        "per_test": avg_time_per_test,
        "total_remaining": avg_time_per_test * remaining_tests,
        "completed_count": len(successful_times),
    }


def calculate_variant_aware_time_estimates(completed_results, remaining_combinations):
    """Calculate time estimates with awareness of AdamWPrune variants and iterations."""
    if not completed_results:
        # No completed tests yet, use rough estimates
        base_estimate = 8 * 3600  # 8 hours as baseline
        total_estimate = 0

        for combo in remaining_combinations:
            if (
                combo.get("optimizer") == "adamwprune"
                and combo.get("variant") == "bitter2"
            ):
                # Bitter2 runs 21% more iterations
                total_estimate += base_estimate * 1.21
            else:
                total_estimate += base_estimate

        return {
            "per_test": base_estimate,
            "total_remaining": total_estimate,
            "completed_count": 0,
        }

    # Group completed results by optimizer type for better estimates
    optimizer_times = {}
    for r in completed_results:
        if r.get("success", False) and "elapsed_time" in r:
            opt_key = r.get("optimizer", "unknown")
            # Include variant in key for AdamWPrune
            if opt_key == "adamwprune" and "variant" in r:
                opt_key = f"{opt_key}_{r['variant']}"

            if opt_key not in optimizer_times:
                optimizer_times[opt_key] = []
            optimizer_times[opt_key].append(r["elapsed_time"])

    # Calculate averages per optimizer/variant
    avg_times = {}
    for key, times in optimizer_times.items():
        avg_times[key] = sum(times) / len(times) if times else 0

    # Estimate remaining time
    total_remaining = 0
    for combo in remaining_combinations:
        opt = combo.get("optimizer", "unknown")
        variant = combo.get("variant", "")

        # Build key to look up timing
        if opt == "adamwprune" and variant:
            lookup_key = f"{opt}_{variant}"
        else:
            lookup_key = opt

        if lookup_key in avg_times:
            estimated_time = avg_times[lookup_key]
        elif opt in avg_times:
            # Fall back to optimizer without variant
            estimated_time = avg_times[opt]
        elif avg_times:
            # Fall back to overall average
            estimated_time = sum(avg_times.values()) / len(avg_times)
        else:
            estimated_time = 8 * 3600  # Default 8 hours

        # Adjust for bitter2's extended iterations
        if opt == "adamwprune" and variant == "bitter2":
            # If we have timing from bitter0 or bitter1, adjust it
            if "adamwprune_bitter0" in avg_times or "adamwprune_bitter1" in avg_times:
                base_time = avg_times.get(
                    "adamwprune_bitter0",
                    avg_times.get("adamwprune_bitter1", estimated_time),
                )
                estimated_time = base_time * 1.21  # 21% more iterations
            elif estimated_time > 0:
                estimated_time *= 1.21

        total_remaining += estimated_time

    # Calculate average for display
    avg_per_test = (
        total_remaining / len(remaining_combinations) if remaining_combinations else 0
    )

    return {
        "per_test": avg_per_test,
        "total_remaining": total_remaining,
        "completed_count": len(completed_results),
        "optimizer_times": avg_times,  # Include for debugging
    }


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
        subprocess.TimeoutExpired,
    ):
        return 0  # GPU monitoring failed, assume no memory usage


def get_gpu_memory_total():
    """Get total GPU memory in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
        subprocess.TimeoutExpired,
    ):
        return 48000  # Default for W7900, fallback if monitoring fails


def should_start_job(max_memory_percent=90):
    """Check if we should start another job based on GPU memory usage."""
    total_memory = get_gpu_memory_total()
    used_memory = get_gpu_memory_usage()
    memory_percent = (used_memory / total_memory) * 100
    return memory_percent < max_memory_percent


def run_single_test_wrapper(args):
    """Wrapper for run_single_test to work with ThreadPoolExecutor."""
    (
        combo,
        config,
        output_dir,
        test_num,
        total_tests,
        max_memory_percent,
        override_epochs,
        time_estimates,
    ) = args

    # Wait for GPU memory to be available
    while not should_start_job(max_memory_percent):
        time.sleep(5)  # Wait 5 seconds before checking again

    with print_lock:
        print(
            f"\n[{test_num}/{total_tests}] Starting parallel job: {combo['model']}_{combo['optimizer']}_{combo['pruning']}"
        )

    result = run_single_test(
        combo,
        config,
        output_dir,
        test_num,
        total_tests,
        parallel_mode=True,
        override_epochs=override_epochs,
        time_estimates=time_estimates,
    )

    with print_lock:
        if result and result.get("success", False):
            print(
                f"✓ [{test_num}/{total_tests}] Completed: {result.get('test_id', 'unknown')}"
            )
        else:
            print(
                f"✗ [{test_num}/{total_tests}] Failed: {combo['model']}_{combo['optimizer']}_{combo['pruning']}"
            )

    return result


def parse_kconfig(config_path=".config"):
    """Parse Kconfig .config file and extract test matrix settings."""
    config = {}

    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Run 'make menuconfig' first.")
        sys.exit(1)

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse CONFIG_KEY=value or CONFIG_KEY="value"
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove CONFIG_ prefix
                if key.startswith("CONFIG_"):
                    key = key[7:]
                # Remove quotes from value
                value = value.strip('"')
                config[key] = value

    # Also try to load the generated Python config for derived values
    try:
        import config as pyconfig

        if hasattr(pyconfig.config, "ADAMWPRUNE_BASE_OPTIMIZER_NAME"):
            config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"] = (
                pyconfig.config.ADAMWPRUNE_BASE_OPTIMIZER_NAME
            )
    except ImportError:
        pass  # config.py might not exist yet

    return config


def parse_yaml_config(yaml_path="test-matrix.yaml"):
    """Parse YAML configuration file and convert to Kconfig-like format."""
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Convert YAML to Kconfig-like format
    config = {}

    # Test matrix settings
    if "test_matrix" in yaml_config:
        tm = yaml_config["test_matrix"]
        config["TEST_MATRIX_MODE"] = "y"

        if "models" in tm:
            config["TEST_MODELS"] = ",".join(tm["models"])
        if "optimizers" in tm:
            config["TEST_OPTIMIZERS"] = ",".join(tm["optimizers"])
        if "pruning_methods" in tm:
            config["TEST_PRUNING_METHODS"] = ",".join(tm["pruning_methods"])

        # Handle sparsity levels from YAML
        if "sparsity_levels" in tm:
            # Map common sparsity levels to the TEST_SPARSITY_XX format
            for level in tm["sparsity_levels"]:
                level_str = str(level)
                if level_str == "0.5" or level_str == "50":
                    config["TEST_SPARSITY_50"] = "y"
                elif level_str == "0.7" or level_str == "70":
                    config["TEST_SPARSITY_70"] = "y"
                elif level_str == "0.9" or level_str == "90":
                    config["TEST_SPARSITY_90"] = "y"
                elif level_str == "0.95" or level_str == "95":
                    config["TEST_SPARSITY_95"] = "y"
                elif level_str == "0.99" or level_str == "99":
                    config["TEST_SPARSITY_99"] = "y"

    # Common configuration
    if "common_config" in yaml_config:
        cc = yaml_config["common_config"]
        # Only override BATCH_SIZE if not already set in .config
        if "BATCH_SIZE" not in config:
            config["BATCH_SIZE"] = str(cc.get("batch_size", 512))
        config["NUM_EPOCHS"] = str(cc.get("num_epochs", 10))
        config["LEARNING_RATE"] = str(cc.get("learning_rate", 0.001))
        config["NUM_WORKERS"] = str(cc.get("num_workers", 16))
        config["DEVICE"] = cc.get("device", "cuda")

        # Pruning config
        if "pruning" in cc:
            pc = cc["pruning"]
            config["TARGET_SPARSITY"] = str(pc.get("target_sparsity", 0.9))
            config["PRUNING_WARMUP"] = str(pc.get("warmup_steps", 100))
            config["PRUNING_FREQUENCY"] = str(pc.get("frequency", 50))
            if "ramp_end_epoch" in pc:
                config["PRUNING_RAMP_END_EPOCH"] = str(pc["ramp_end_epoch"])

    # Advanced options
    if "advanced" in yaml_config:
        adv = yaml_config["advanced"]
        config["COMPILE_MODEL"] = "y" if adv.get("compile_model", True) else "n"
        config["MIXED_PRECISION"] = "y" if adv.get("mixed_precision", True) else "n"

    return config


def get_test_matrix(config):
    """Extract test matrix components from config."""
    matrix = {
        "models": [],
        "optimizers": [],
        "pruning_methods": [],
        "sparsity_levels": [],
    }

    # Check if we should use test matrix framework
    # Use it for both multiple modes AND single mode (for consistent logging)
    use_test_framework = (
        config.get("TEST_MATRIX_MODE") == "y"
        or config.get("OPTIMIZER_MODE_MULTIPLE") == "y"
        or config.get("MODEL_MODE_MULTIPLE") == "y"
        or config.get("PRUNING_MODE_MULTIPLE") == "y"
        or config.get("OPTIMIZER_MODE_SINGLE") == "y"
        or config.get("MODEL_MODE_SINGLE") == "y"
        or config.get("PRUNING_MODE_SINGLE") == "y"
    )

    # For backward compatibility, keep multiple_mode check
    multiple_mode = use_test_framework

    if not multiple_mode:
        # Single mode - use single selections
        if "MODEL" in config:
            matrix["models"] = [config["MODEL"]]
        if "OPTIMIZER" in config and config["OPTIMIZER"]:
            matrix["optimizers"] = [config["OPTIMIZER"]]
        if config.get("ENABLE_PRUNING") == "y" and "PRUNING_METHOD" in config:
            matrix["pruning_methods"] = [config["PRUNING_METHOD"]]
        elif config.get("ENABLE_PRUNING") != "y":
            matrix["pruning_methods"] = ["none"]
        return matrix

    # Test matrix mode - parse comma-separated lists
    models = []

    # Check new Kconfig model system first
    if config.get("MODEL_MODE_MULTIPLE") == "y":
        # Multiple model mode - check individual enables
        if (
            config.get("MODEL_ENABLE_LENET5") == "y"
            or config.get("TEST_MODEL_LENET5") == "y"
        ):
            models.append("lenet5")
        if (
            config.get("MODEL_ENABLE_RESNET18") == "y"
            or config.get("TEST_MODEL_RESNET18") == "y"
        ):
            models.append("resnet18")
    elif config.get("MODEL_MODE_SINGLE") == "y":
        # Single model mode - use the selected model
        if config.get("MODEL_SELECT_LENET5") == "y":
            models = ["lenet5"]
        elif config.get("MODEL_SELECT_RESNET18") == "y":
            models = ["resnet18"]
        elif config.get("MODEL_SELECT_RESNET50") == "y":
            models = ["resnet50"]
        elif config.get("MODEL_SELECT_GPT2") == "y":
            models = ["gpt2"]

    # Fall back to legacy TEST_MODELS if new system not configured
    if not models and "TEST_MODELS" in config and config["TEST_MODELS"]:
        models = [m.strip() for m in config["TEST_MODELS"].split(",")]

    # Default to lenet5 if nothing found
    if not models:
        models = ["lenet5"]

    matrix["models"] = models

    # Build optimizer list
    optimizers = []

    # Check new Kconfig optimizer system first
    if config.get("OPTIMIZER_MODE_MULTIPLE") == "y":
        # Multiple optimizer mode - check individual enables
        for config_key, value in config.items():
            if config_key.startswith("OPTIMIZER_ENABLE_") and value == "y":
                optimizer_name = config_key.replace("OPTIMIZER_ENABLE_", "").lower()
                optimizers.append(optimizer_name)
    elif config.get("OPTIMIZER_MODE_SINGLE") == "y":
        # Single optimizer mode - use the selected optimizer
        if config.get("OPTIMIZER_SELECT_SGD") == "y":
            optimizers = ["sgd"]
        elif config.get("OPTIMIZER_SELECT_ADAM") == "y":
            optimizers = ["adam"]
        elif config.get("OPTIMIZER_SELECT_ADAMW") == "y":
            optimizers = ["adamw"]
        elif config.get("OPTIMIZER_SELECT_ADAMWADV") == "y":
            optimizers = ["adamwadv"]
        elif config.get("OPTIMIZER_SELECT_ADAMWSPAM") == "y":
            optimizers = ["adamwspam"]
        elif config.get("OPTIMIZER_SELECT_ADAMWPRUNE") == "y":
            optimizers = ["adamwprune"]

    # Fall back to legacy TEST_OPTIMIZER_ENABLED_* flags
    if not optimizers:
        for config_key, value in config.items():
            if config_key.startswith("TEST_OPTIMIZER_ENABLED_") and value == "y":
                # Extract optimizer name from config key (e.g., TEST_OPTIMIZER_ENABLED_SGD -> sgd)
                optimizer_name = config_key.replace(
                    "TEST_OPTIMIZER_ENABLED_", ""
                ).lower()
                optimizers.append(optimizer_name)

    if optimizers:
        matrix["optimizers"] = optimizers

    # For pruning, check if any pruning methods are selected
    pruning_methods = []

    # Check new Kconfig pruning system first
    if config.get("PRUNING_MODE_MULTIPLE") == "y":
        # Multiple pruning mode - check individual enables
        if config.get("PRUNING_ENABLED_NONE") == "y":
            pruning_methods.append("none")
        if config.get("PRUNING_ENABLED_MAGNITUDE") == "y":
            pruning_methods.append("magnitude")
        if config.get("PRUNING_ENABLED_MOVEMENT") == "y":
            pruning_methods.append("movement")
        if config.get("PRUNING_ENABLED_STATE") == "y":
            pruning_methods.append("state")
    elif config.get("PRUNING_MODE_SINGLE") == "y":
        # Single pruning mode - use the selected method
        if config.get("PRUNING_SELECT_NONE") == "y":
            pruning_methods = ["none"]
        elif config.get("PRUNING_SELECT_MAGNITUDE") == "y":
            pruning_methods = ["magnitude"]
        elif config.get("PRUNING_SELECT_MOVEMENT") == "y":
            pruning_methods = ["movement"]
        elif config.get("PRUNING_SELECT_STATE") == "y":
            pruning_methods = ["state"]
    elif config.get("PRUNING_MODE_NONE") == "y":
        # No pruning mode
        pruning_methods = ["none"]

    # Fall back to legacy TEST_PRUNING_METHODS if new system not configured
    if (
        not pruning_methods
        and "TEST_PRUNING_METHODS" in config
        and config["TEST_PRUNING_METHODS"]
    ):
        pruning_methods = [p.strip() for p in config["TEST_PRUNING_METHODS"].split(",")]

    # Default to none if nothing found
    if not pruning_methods:
        pruning_methods = ["none"]

    matrix["pruning_methods"] = pruning_methods

    # Special case: AdamWPrune uses state-based pruning only when its pruning is enabled
    adamwprune_pruning_enabled = getattr(config, "ADAMWPRUNE_ENABLE_PRUNING", False)
    if (
        "adamwprune" in matrix["optimizers"]
        and adamwprune_pruning_enabled
        and "state" not in matrix["pruning_methods"]
    ):
        matrix["pruning_methods"].append("state")

    # Check for AdamWPrune variants (bitter0-9)
    adamwprune_variants = []
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER0") == "y":
        adamwprune_variants.append("bitter0")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER1") == "y":
        adamwprune_variants.append("bitter1")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER2") == "y":
        adamwprune_variants.append("bitter2")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER3") == "y":
        adamwprune_variants.append("bitter3")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER4") == "y":
        adamwprune_variants.append("bitter4")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER5") == "y":
        adamwprune_variants.append("bitter5")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER6") == "y":
        adamwprune_variants.append("bitter6")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER7") == "y":
        adamwprune_variants.append("bitter7")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER8") == "y":
        adamwprune_variants.append("bitter8")
    if config.get("GPT2_ADAMWPRUNE_VARIANT_BITTER9") == "y":
        adamwprune_variants.append("bitter9")

    # Store variants in matrix for later use
    matrix["adamwprune_variants"] = (
        adamwprune_variants if adamwprune_variants else ["bitter0"]
    )

    # Check for RA_MLA ablation mode
    ra_mla_ablation_steps = []
    if config.get("RA_MLA_ABLATION_MODE") == "y" and config.get("ENABLE_RA_MLA") == "y":
        # Parse the ablation steps string (e.g., "0,1,2,3,4,5")
        ablation_steps_str = config.get("RA_MLA_ABLATION_STEPS", "").strip('"')
        if ablation_steps_str:
            ra_mla_ablation_steps = [s.strip() for s in ablation_steps_str.split(",")]
        else:
            # If ablation mode is enabled but no steps specified, check individual flags
            if config.get("RA_MLA_ABLATION_BASELINE") == "y":
                ra_mla_ablation_steps.append("0")
            if config.get("RA_MLA_ABLATION_STEP1") == "y":
                ra_mla_ablation_steps.append("1")
            if config.get("RA_MLA_ABLATION_STEP2") == "y":
                ra_mla_ablation_steps.append("2")
            if config.get("RA_MLA_ABLATION_STEP3") == "y":
                ra_mla_ablation_steps.append("3")
            if config.get("RA_MLA_ABLATION_STEP4") == "y":
                ra_mla_ablation_steps.append("4")
            if config.get("RA_MLA_ABLATION_STEP5") == "y":
                ra_mla_ablation_steps.append("5")

    matrix["ra_mla_ablation_steps"] = (
        ra_mla_ablation_steps if ra_mla_ablation_steps else None
    )

    # Get sparsity levels from individual TEST_SPARSITY_* configs
    matrix["sparsity_levels"] = []

    # Check for each possible sparsity level config
    sparsity_configs = [
        ("TEST_SPARSITY_50", "0.5"),
        ("TEST_SPARSITY_70", "0.7"),
        ("TEST_SPARSITY_90", "0.9"),
        ("TEST_SPARSITY_95", "0.95"),
        ("TEST_SPARSITY_99", "0.99"),
    ]

    for config_name, sparsity_value in sparsity_configs:
        if config.get(config_name) == "y":
            matrix["sparsity_levels"].append(sparsity_value)

    # If no sparsity levels selected from TEST_SPARSITY_* but we have TARGET_SPARSITY
    if not matrix["sparsity_levels"] and config.get("TARGET_SPARSITY"):
        target_sparsity = config["TARGET_SPARSITY"].strip('"')
        matrix["sparsity_levels"] = [target_sparsity]

    # If no sparsity levels selected but pruning is enabled, use default
    if (
        not matrix["sparsity_levels"]
        and matrix["pruning_methods"]
        and matrix["pruning_methods"] != ["none"]
    ):
        # Default to 90% if nothing specified
        matrix["sparsity_levels"] = ["0.9"]

    return matrix


def generate_combinations(matrix):
    """Generate all combinations of test matrix."""
    combinations = []

    # Get sparsity levels from config
    sparsity_levels = matrix.get(
        "sparsity_levels", ["0.9"]
    )  # Default to 90% if not specified

    # Get AdamWPrune variants
    adamwprune_variants = matrix.get("adamwprune_variants", ["bitter0"])

    # Get RA_MLA ablation steps
    ra_mla_ablation_steps = matrix.get("ra_mla_ablation_steps", None)

    for model, optimizer, pruning in itertools.product(
        matrix["models"], matrix["optimizers"], matrix["pruning_methods"]
    ):
        # Skip invalid combinations
        if optimizer == "adamwprune" and pruning not in ["none", "state"]:
            continue  # AdamWPrune only works with state-based pruning
        if optimizer != "adamwprune" and pruning == "state":
            continue  # State-based pruning is only for AdamWPrune (built-in)

        # For no pruning, sparsity is always 0
        if pruning == "none":
            # Check if we should generate RA_MLA ablation steps
            if ra_mla_ablation_steps and model == "gpt2":
                # Generate one combination for each ablation step
                for ablation_step in ra_mla_ablation_steps:
                    combo = {
                        "model": model,
                        "optimizer": optimizer,
                        "pruning": pruning,
                        "sparsity": "0.0",
                        "ra_mla_ablation_step": ablation_step,
                    }
                    # Include AdamWPrune variant if applicable
                    if optimizer == "adamwprune":
                        for variant in adamwprune_variants:
                            combo_with_variant = combo.copy()
                            combo_with_variant["variant"] = variant
                            combinations.append(combo_with_variant)
                    else:
                        combinations.append(combo)
            # For AdamWPrune without RA_MLA ablation, generate combinations for each variant
            elif optimizer == "adamwprune":
                for variant in adamwprune_variants:
                    combinations.append(
                        {
                            "model": model,
                            "optimizer": optimizer,
                            "pruning": pruning,
                            "sparsity": "0.0",
                            "variant": variant,
                        }
                    )
            else:
                combinations.append(
                    {
                        "model": model,
                        "optimizer": optimizer,
                        "pruning": pruning,
                        "sparsity": "0.0",
                    }
                )
        else:
            # For pruning methods, test each sparsity level
            for sparsity in sparsity_levels:
                # For AdamWPrune, generate combinations for each variant
                if optimizer == "adamwprune":
                    for variant in adamwprune_variants:
                        combinations.append(
                            {
                                "model": model,
                                "optimizer": optimizer,
                                "pruning": pruning,
                                "sparsity": sparsity,
                                "variant": variant,
                            }
                        )
                else:
                    combinations.append(
                        {
                            "model": model,
                            "optimizer": optimizer,
                            "pruning": pruning,
                            "sparsity": sparsity,
                        }
                    )

    return combinations


def run_single_test(
    combination,
    config,
    output_dir,
    test_num,
    total_tests,
    parallel_mode=False,
    override_epochs=None,
    time_estimates=None,
):
    """Run a single test combination."""
    import os  # Explicit import to fix UnboundLocalError

    model = combination["model"]
    optimizer = combination["optimizer"]
    pruning = combination["pruning"]
    sparsity = combination.get("sparsity", "0.0")
    variant = combination.get("variant", None)
    ra_mla_ablation_step = combination.get("ra_mla_ablation_step", None)

    # Create test identifier including sparsity level, variant, and ablation step if applicable
    if pruning == "none":
        if ra_mla_ablation_step:
            # Include RA_MLA ablation step in test ID (e.g., gpt2_adamwspam_ramla_step2)
            test_id = f"{model}_{optimizer}_ramla_step{ra_mla_ablation_step}"
            if variant:
                test_id = (
                    f"{model}_{optimizer}_{variant}_ramla_step{ra_mla_ablation_step}"
                )
        elif variant:
            test_id = f"{model}_{optimizer}_{variant}_{pruning}"
        else:
            test_id = f"{model}_{optimizer}_{pruning}"
    else:
        # Include sparsity in the test ID (e.g., sgd_movement_90)
        sparsity_pct = int(float(sparsity) * 100)
        if variant:
            test_id = f"{model}_{optimizer}_{variant}_{pruning}_{sparsity_pct}"
        else:
            test_id = f"{model}_{optimizer}_{pruning}_{sparsity_pct}"
    test_output_dir = os.path.join(output_dir, test_id)
    os.makedirs(test_output_dir, exist_ok=True)

    if not parallel_mode:
        print(f"\n{'='*60}")
        print(f"Test {test_num}/{total_tests}: {test_id}")
        print(f"{'='*60}")
    # In parallel mode, printing is handled by the wrapper function
    if not parallel_mode:
        print(f"Model: {model}")
        print(f"Optimizer: {optimizer}")
        if variant:
            print(f"Variant: {variant}")
        if ra_mla_ablation_step:
            print(f"RA_MLA Ablation Step: {ra_mla_ablation_step}")
        print(f"Pruning: {pruning}")
        if pruning != "none":
            print(f"Sparsity: {float(sparsity)*100:.0f}%")
        print(f"Output: {test_output_dir}")

        # Show time estimates if available
        if time_estimates:
            if "per_test" in time_estimates:
                print(
                    f"Estimated time for this test: {format_time(time_estimates['per_test'])}"
                )
            remaining_tests = total_tests - test_num + 1
            if "per_test" in time_estimates:
                remaining_time = time_estimates["per_test"] * remaining_tests
                print(f"Estimated time remaining: {format_time(remaining_time)}")

        print(f"{'='*60}")

    # Build command using GPU monitoring wrapper
    cmd = ["python3", "scripts/train_with_monitoring.py"]
    cmd.extend(["--model", model])
    cmd.extend(["--config-name", test_id])
    cmd.extend(["--output-dir", test_output_dir])
    cmd.append("--generate-graphs")

    # Set working directory to parent (where scripts/ is located)
    working_dir = "."

    # Add separator to indicate start of training script arguments
    cmd.append("--")

    # Add configuration arguments that train.py actually accepts
    cmd.extend(["--optimizer", optimizer])

    # Add GPT2 dataset configuration if applicable
    if model == "gpt2" and "GPT2_DATASET_NAME" in config:
        cmd.extend(["--dataset", config["GPT2_DATASET_NAME"]])

    # Add batch size for GPT-2 (CNN models get it from config.py)
    if model == "gpt2" and "BATCH_SIZE" in config:
        cmd.extend(["--batch-size", config["BATCH_SIZE"]])

    # Add block size for GPT-2
    if model == "gpt2" and "GPT2_BLOCK_SIZE" in config:
        cmd.extend(["--block-size", config["GPT2_BLOCK_SIZE"]])

    # Add gradient accumulation for GPT-2
    if model == "gpt2" and "GPT2_GRADIENT_ACCUMULATION" in config:
        cmd.extend(["--gradient-accumulation", config["GPT2_GRADIENT_ACCUMULATION"]])

    # Add other GPT-2 specific settings
    if model == "gpt2":
        if config.get("GPT2_COMPILE") == "y" or config.get("COMPILE_MODEL") == "y":
            cmd.append("--compile")
        if config.get("GPT2_FLASH_ATTENTION") == "y":
            cmd.append("--flash-attention")
        if "GPT2_WARMUP_STEPS" in config:
            cmd.extend(["--warmup-steps", config["GPT2_WARMUP_STEPS"]])
        if "GPT2_EVAL_INTERVAL" in config:
            cmd.extend(["--eval-interval", config["GPT2_EVAL_INTERVAL"]])
        if "GPT2_EVAL_SAMPLES" in config:
            cmd.extend(["--eval-samples", config["GPT2_EVAL_SAMPLES"]])
        # Support MAX_ITERS from environment or config
        max_iters = os.environ.get("GPT2_MAX_ITERS") or config.get("GPT2_MAX_ITERS")
        if max_iters:
            cmd.extend(["--max-iters", str(max_iters)])
        if config.get("GPT2_DECAY_LR") == "y":
            cmd.append("--decay-lr")
        if "GPT2_MIN_LR" in config:
            cmd.extend(["--min-lr", config["GPT2_MIN_LR"]])
        if "LEARNING_RATE" in config:
            cmd.extend(["--learning-rate", config["LEARNING_RATE"]])
        if "GPT2_WEIGHT_DECAY" in config:
            cmd.extend(["--weight-decay", config["GPT2_WEIGHT_DECAY"]])
        elif "ADAMWPRUNE_WEIGHT_DECAY" in config:
            cmd.extend(["--weight-decay", config["ADAMWPRUNE_WEIGHT_DECAY"]])

    # AdamWPrune uses state-based pruning
    if optimizer == "adamwprune" and pruning == "state":
        # For AdamWPrune with state pruning, pass "state" as the method
        cmd.extend(["--pruning-method", "state"])
        cmd.extend(["--target-sparsity", sparsity])
        if "PRUNING_WARMUP" in config:
            cmd.extend(["--pruning-warmup", config["PRUNING_WARMUP"]])
    elif pruning and pruning != "none":
        # For other optimizers, pass the pruning method
        cmd.extend(["--pruning-method", pruning])
        # Use the specific sparsity for this test
        cmd.extend(["--target-sparsity", sparsity])
        if "PRUNING_WARMUP" in config:
            cmd.extend(["--pruning-warmup", config["PRUNING_WARMUP"]])
        # Note: train.py doesn't accept pruning-frequency as an argument
    else:
        # No pruning case
        cmd.extend(["--pruning-method", "none"])

    # Add SPAM configuration if applicable
    if optimizer in ["adamwspam", "adamwprune"]:
        if "SPAM_THETA" in config:
            cmd.extend(["--spam-theta", config["SPAM_THETA"]])
        if "SPAM_INTERVAL" in config:
            cmd.extend(["--spam-interval", config["SPAM_INTERVAL"]])
        if "SPAM_WARMUP_STEPS" in config:
            cmd.extend(["--spam-warmup-steps", config["SPAM_WARMUP_STEPS"]])
        if config.get("SPAM_ENABLE_CLIP") == "y":
            cmd.append("--spam-enable-clip")

    # Add AdamWPrune tuning parameters if present
    if optimizer == "adamwprune":
        if "ADAMWPRUNE_BASE_OPTIMIZER_NAME" in config:
            cmd.extend(
                [
                    "--adamwprune-base-optimizer-name",
                    config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"],
                ]
            )
        if "ADAMWPRUNE_BETA1" in config:
            cmd.extend(["--adamwprune-beta1", config["ADAMWPRUNE_BETA1"]])
        if "ADAMWPRUNE_BETA2" in config:
            cmd.extend(["--adamwprune-beta2", config["ADAMWPRUNE_BETA2"]])
        if "ADAMWPRUNE_WEIGHT_DECAY" in config:
            cmd.extend(["--adamwprune-weight-decay", config["ADAMWPRUNE_WEIGHT_DECAY"]])
        if "ADAMWPRUNE_AMSGRAD" in config:
            amsgrad_val = "true" if config["ADAMWPRUNE_AMSGRAD"] == "y" else "false"
            cmd.extend(["--adamwprune-amsgrad", amsgrad_val])
        # Add variant parameter if specified
        if variant:
            cmd.extend(["--adamwprune-variant", variant])

    # Add RA_MLA ablation step parameter if specified
    if ra_mla_ablation_step:
        cmd.extend(["--ra-mla-ablation-step", ra_mla_ablation_step])

    # Note: batch size is configured via config.py, not command line arguments

    # JSON output for metrics - use absolute path since training runs from model dir
    json_output = os.path.abspath(
        os.path.join(test_output_dir, "training_metrics.json")
    )
    cmd.extend(["--json-output", json_output])

    # Add experiment tracking configuration if specified
    tracker = None
    if "EXPERIMENT_TRACKER" in config and config["EXPERIMENT_TRACKER"] != "none":
        tracker = config["EXPERIMENT_TRACKER"]
    elif "TRACKER_CLI_VALUE" in config and config["TRACKER_CLI_VALUE"]:
        # Pass all trackers as comma-separated value
        tracker = config["TRACKER_CLI_VALUE"]

    if tracker and tracker != "none":
        cmd.extend(["--tracker", tracker])

        # Handle project name - auto-generate if empty
        if "TRACKER_PROJECT" in config and config["TRACKER_PROJECT"]:
            cmd.extend(["--tracker-project", config["TRACKER_PROJECT"]])
        else:
            # Auto-generate project name based on directory and checksum
            import hashlib

            cwd = os.getcwd()
            dir_name = os.path.basename(cwd)
            # Create a short checksum of the full path for uniqueness
            path_hash = hashlib.md5(cwd.encode()).hexdigest()[:8]
            auto_project = f"{dir_name}-{path_hash}"
            cmd.extend(["--tracker-project", auto_project])

        if "TRACKER_RUN_NAME" in config and config["TRACKER_RUN_NAME"]:
            cmd.extend(["--tracker-run-name", config["TRACKER_RUN_NAME"]])
        else:
            # Use the test ID as the run name for better clarity in wandb/trackio
            cmd.extend(["--tracker-run-name", test_id])
        # Set WANDB offline mode if configured
        if tracker == "wandb" and config.get("WANDB_OFFLINE") == "y":
            import os

            os.environ["WANDB_MODE"] = "offline"

    # Override epochs if requested (for quick testing)
    if override_epochs is not None:
        cmd.extend(["--epochs", str(override_epochs)])

    # Capture timing
    start_time = time.time()

    # Run the training
    if not parallel_mode:
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

    # Open log file for this test
    log_file = os.path.join(test_output_dir, "output.log")

    try:
        # Run with real-time output to console AND capture to file
        with open(log_file, "w") as f_log:
            # Use Popen for real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=working_dir,  # Run from the model directory
            )

            # Stream output line by line (suppress in parallel mode)
            for line in process.stdout:
                if not parallel_mode:
                    print(line, end="")  # Print to console (only in serial mode)
                f_log.write(line)  # Write to file
                f_log.flush()  # Ensure it's written

            # Wait for process to complete
            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        elapsed_time = time.time() - start_time

        # Parse results from JSON if available
        metrics_file = os.path.join(test_output_dir, "training_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Extract GPU memory stats from gpu_stats files
        gpu_stats_files = list(Path(test_output_dir).glob("gpu_stats*.json"))
        if gpu_stats_files:
            # Read the most recent gpu_stats file
            gpu_stats_file = sorted(gpu_stats_files)[-1]
            try:
                with open(gpu_stats_file, "r") as f:
                    gpu_data = json.load(f)
                    if gpu_data and isinstance(gpu_data, list):
                        # Calculate mean and max memory usage
                        memory_values = [
                            d.get("memory_used", 0)
                            for d in gpu_data
                            if "memory_used" in d
                        ]
                        if memory_values:
                            metrics["gpu_memory_mean"] = sum(memory_values) / len(
                                memory_values
                            )
                            metrics["gpu_memory_max"] = max(memory_values)
            except Exception as e:
                print(f"  Warning: Could not extract GPU memory data: {e}")

        # Add test metadata
        metrics["test_id"] = test_id
        metrics["model"] = model
        metrics["optimizer"] = optimizer
        metrics["pruning"] = pruning
        metrics["target_sparsity"] = float(sparsity) if pruning != "none" else 0
        metrics["elapsed_time"] = elapsed_time
        metrics["success"] = True
        # Add variant information if present
        if variant:
            metrics["variant"] = variant

        if not parallel_mode:
            print(f"✓ Test completed in {elapsed_time:.2f} seconds")
            if "final_accuracy" in metrics:
                print(f"  Final accuracy: {metrics['final_accuracy']:.2f}%")
            elif "best_accuracy" in metrics:
                print(f"  Best accuracy: {metrics['best_accuracy']:.2f}%")
            if "final_sparsity" in metrics:
                print(f"  Final sparsity: {metrics['final_sparsity']:.4f}")

        return metrics

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        if not parallel_mode:
            print(f"\n✗ Test failed with return code {e.returncode}")
            print(f"  Check log at: {log_file}")

        # Save error information
        with open(os.path.join(test_output_dir, "error.txt"), "w") as f:
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write(f"Time elapsed: {elapsed_time:.2f} seconds\n")

        return {
            "test_id": test_id,
            "model": model,
            "optimizer": optimizer,
            "pruning": pruning,
            "target_sparsity": float(sparsity) if pruning != "none" else 0,
            "elapsed_time": time.time() - start_time,
            "success": False,
            "error": str(e),
        }


def fix_all_results_json(results_dir):
    """Fix all_results.json by rebuilding from individual training_metrics.json files."""
    results_path = Path(results_dir)
    all_results_file = results_path / "all_results.json"

    # Collect all results from individual test directories
    all_results = []
    fixed_count = 0

    for test_dir in sorted(results_path.glob("resnet*")):
        if not test_dir.is_dir():
            continue

        metrics_file = test_dir / "training_metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)

            # Extract GPU memory stats if not already present
            if "gpu_memory_mean" not in data or "gpu_memory_max" not in data:
                gpu_stats_files = list(test_dir.glob("gpu_stats*.json"))
                if gpu_stats_files:
                    # Read the most recent gpu_stats file
                    gpu_stats_file = sorted(gpu_stats_files)[-1]
                    try:
                        with open(gpu_stats_file, "r") as gf:
                            gpu_data = json.load(gf)
                            if gpu_data and isinstance(gpu_data, list):
                                # Calculate mean and max memory usage
                                memory_values = [
                                    d.get("memory_used", 0)
                                    for d in gpu_data
                                    if "memory_used" in d
                                ]
                                if memory_values:
                                    data["gpu_memory_mean"] = sum(memory_values) / len(
                                        memory_values
                                    )
                                    data["gpu_memory_max"] = max(memory_values)
                                    fixed_count += 1
                    except Exception:
                        pass  # Silently skip if GPU data can't be read

            # Ensure test_id is set
            if "test_id" not in data or not data["test_id"]:
                data["test_id"] = test_dir.name
                fixed_count += 1

            # Parse test directory name to get model, optimizer, pruning, sparsity
            test_name = test_dir.name
            parts = test_name.split("_")

            # Expected format: model_optimizer_pruning_sparsity
            if "model" not in data or not data["model"]:
                data["model"] = parts[0] if len(parts) > 0 else "unknown"
            if "optimizer" not in data or not data["optimizer"]:
                data["optimizer"] = parts[1] if len(parts) > 1 else "unknown"
            if "pruning" not in data or not data["pruning"]:
                data["pruning"] = parts[2] if len(parts) > 2 else "none"

            # Get sparsity from name or from data
            if "target_sparsity" not in data:
                if len(parts) > 3:
                    data["target_sparsity"] = float(parts[3]) / 100.0
                elif "final_sparsity" in data:
                    data["target_sparsity"] = data["final_sparsity"]
                else:
                    data["target_sparsity"] = 0.0

            # Ensure required fields exist
            if "success" not in data:
                data["success"] = True
            if "total_time" not in data and "elapsed_time" in data:
                data["total_time"] = data["elapsed_time"]

            all_results.append(data)

        except Exception as e:
            print(f"Warning: Could not process {test_dir.name}: {e}")

    # Check if we need to update the file
    needs_update = False
    if all_results_file.exists():
        try:
            with open(all_results_file, "r") as f:
                existing_data = json.load(f)
            if len(existing_data) != len(all_results) or fixed_count > 0:
                needs_update = True
        except:
            needs_update = True
    else:
        needs_update = True

    if needs_update:
        # Save the fixed all_results.json
        with open(all_results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Fixed all_results.json: {len(all_results)} tests found")
        if fixed_count > 0:
            print(f"  Repaired {fixed_count} test entries with missing metadata")

    return all_results


def create_summary_report(results, output_dir):
    """Create a summary report of all test results."""
    json_file = os.path.join(output_dir, "all_results.json")

    # Load existing results if they exist (for continuation mode)
    existing_results = []
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_results = existing_data
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Merge existing and new results, avoiding duplicates
    merged_results = existing_results.copy()
    existing_test_ids = {r.get("test_id") for r in existing_results if "test_id" in r}

    for result in results:
        if result.get("test_id") not in existing_test_ids:
            merged_results.append(result)

    # Save merged JSON results
    with open(json_file, "w") as f:
        json.dump(merged_results, f, indent=2)

    # Import and use the regenerate_summary module for consistent reporting
    try:
        # Dynamically import regenerate_summary module
        spec = importlib.util.spec_from_file_location(
            "regenerate_summary",
            os.path.join(os.path.dirname(__file__), "regenerate_summary.py"),
        )
        regenerate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regenerate_module)

        # Use the regenerate_summary function
        regenerate_module.regenerate_summary(output_dir)
    except Exception as e:
        print(f"Warning: Could not use enhanced summary generation: {e}")
        # Fall back to basic summary
        report_file = os.path.join(output_dir, "summary_report.txt")
        with open(report_file, "w") as f:
            f.write("Test Matrix Summary Report (Basic)\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            successful = sum(1 for r in results if r.get("success", False))
            failed = len(results) - successful
            f.write(f"Total tests: {len(results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")

            print(f"\nBasic summary report saved to: {report_file}")
            print(f"JSON results saved to: {json_file}")


def generate_sweep_analysis(output_dir, results):
    """Generate analysis report for hyperparameter sweep."""
    if not results:
        return

    # Sort results by test accuracy
    sorted_results = sorted(results, key=lambda x: x.get("test_acc", 0), reverse=True)

    # Write analysis report
    report_file = output_dir / "sweep_analysis.txt"
    with open(report_file, "w") as f:
        f.write("Hyperparameter Sweep Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total configurations tested: {len(results)}\n\n")

        # Best configuration
        if sorted_results:
            best = sorted_results[0]
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Config file: {best.get('config_file', 'unknown')}\n")
            f.write(f"Test Accuracy: {best.get('test_acc', 0):.2f}%\n")
            f.write(f"Final Sparsity: {best.get('final_sparsity', 0):.1%}\n")
            f.write(f"Training Time: {best.get('total_time', 0):.1f}s\n")

            # Extract hyperparameters from config
            if "config_file" in best:
                f.write("\nHyperparameters:\n")
                # Parse the config file to extract key parameters
                # The test_name might not be in the result, so construct it from config_file
                config_num = best["config_file"].replace("config_", "").split("_")[0]
                test_dirs = [
                    d
                    for d in output_dir.iterdir()
                    if d.is_dir() and config_num in d.name
                ]
                if test_dirs:
                    config_path = test_dirs[0] / "config.txt"
                else:
                    config_path = output_dir / best["config_file"] / "config.txt"
                if config_path.exists():
                    with open(config_path, "r") as cf:
                        for line in cf:
                            if (
                                "ADAMWPRUNE" in line
                                or "PRUNING_WARMUP" in line
                                or "TARGET_SPARSITY" in line
                            ):
                                if "=" in line and not line.startswith("#"):
                                    f.write(f"  {line.strip()}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("TOP 10 CONFIGURATIONS:\n")
        f.write("-" * 40 + "\n\n")

        # Top 10 results
        for i, result in enumerate(sorted_results[:10], 1):
            f.write(f"{i}. {result.get('config_file', 'unknown')}\n")
            f.write(f"   Test Acc: {result.get('test_acc', 0):.2f}% | ")
            f.write(f"Sparsity: {result.get('final_sparsity', 0):.1%} | ")
            f.write(f"Time: {result.get('total_time', 0):.1f}s\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("FAILED TESTS:\n")
        f.write("-" * 40 + "\n")

        failed_count = len([r for r in results if r.get("status") == "failed"])
        f.write(f"Failed: {failed_count}/{len(results)}\n")

    print(f"Analysis report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run test matrix for AdamWPrune training"
    )
    parser.add_argument(
        "--config",
        default=".config",
        help="Path to Kconfig .config file (default: .config)",
    )
    parser.add_argument(
        "--config-yaml", help="Path to YAML configuration file (overrides --config)"
    )
    parser.add_argument(
        "--config-dir",
        help="Path to directory containing multiple config files to test",
    )
    parser.add_argument(
        "--output-dir",
        default="test_matrix_results",
        help="Output directory for results (default: test_matrix_results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel tests to run (default: 1)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop execution on first test failure",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output during execution"
    )
    parser.add_argument(
        "--rerun-dir",
        help="Existing results directory to update (skips timestamp creation)",
    )
    parser.add_argument(
        "--filter-optimizer",
        help="Only run tests for specific optimizer(s), comma-separated (e.g., adamwprune)",
    )
    parser.add_argument(
        "--override-epochs",
        type=int,
        help="Override number of epochs for quick testing (e.g., 1 for smoke test)",
    )
    parser.add_argument(
        "--continue-dir",
        help="Continue from an existing test matrix directory (clean and resume incomplete tests)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically answer yes to all prompts",
    )
    args = parser.parse_args()

    # Handle continuation mode
    if args.continue_dir:
        # First, fix any broken all_results.json by rebuilding from training_metrics.json files
        fix_all_results_json(args.continue_dir)

    if args.continue_dir:
        from pathlib import Path

        continue_dir = Path(args.continue_dir)
        if not continue_dir.exists():
            print(f"Error: Directory '{continue_dir}' does not exist")
            sys.exit(1)

        print(f"Continuing test matrix from: {continue_dir}")
        print("=" * 60)

        # First, identify incomplete runs (use --dry-run to avoid prompts)
        clean_cmd = [
            "python3",
            "scripts/clean_incomplete_runs.py",
            str(continue_dir),
            "--dry-run",  # Don't prompt or remove anything yet
            "--json-output",
            "/tmp/incomplete_runs.json",
        ]
        result = subprocess.run(clean_cmd, capture_output=True, text=True)
        print(result.stdout)

        # Load the results
        incomplete_info = None
        if os.path.exists("/tmp/incomplete_runs.json"):
            with open("/tmp/incomplete_runs.json", "r") as f:
                incomplete_info = json.load(f)

        # Parse config to get expected test combinations
        config = parse_kconfig()
        matrix = get_test_matrix(config)
        expected_combinations = generate_combinations(matrix)

        # Generate expected test directory names
        expected_tests = set()
        for combo in expected_combinations:
            model = combo["model"]
            optimizer = combo["optimizer"]
            pruning = combo["pruning"]
            sparsity = combo.get("sparsity", "0.0")
            variant = combo.get("variant", None)

            if pruning == "none":
                if variant:
                    test_id = f"{model}_{optimizer}_{variant}_{pruning}"
                else:
                    test_id = f"{model}_{optimizer}_{pruning}"
            else:
                sparsity_pct = int(float(sparsity) * 100)
                if variant:
                    test_id = f"{model}_{optimizer}_{variant}_{pruning}_{sparsity_pct}"
                else:
                    test_id = f"{model}_{optimizer}_{pruning}_{sparsity_pct}"
            expected_tests.add(test_id)

        # Check what tests actually exist
        existing_tests = set()
        for test_dir in continue_dir.glob(f"{matrix['models'][0]}_*"):
            if test_dir.is_dir():
                existing_tests.add(test_dir.name)

        # Find missing tests
        missing_tests = expected_tests - existing_tests

        # Check if we have failed tests but no incomplete ones (special case)
        incomplete_runs = incomplete_info.get("incomplete_runs", [])
        failed_runs = incomplete_info.get("failed_runs", [])

        # Report on missing tests
        if missing_tests:
            print(
                f"\n⚠️  Found {len(missing_tests)} MISSING test(s) from expected configuration:"
            )
            for missing_test in sorted(missing_tests):
                print(f"   ❌ {missing_test}")

            print(f"\nThese tests are configured but have not been run.")
            print("Would you like to run these missing tests?")
            if not args.yes:
                response = input("Continue? (y/N): ").strip().lower()
                if response != "y":
                    print("Aborted.")
                    sys.exit(0)

            # Add missing tests to the list of tests to run
            tests_to_run = []
            for combo in expected_combinations:
                model = combo["model"]
                optimizer = combo["optimizer"]
                pruning = combo["pruning"]
                sparsity = combo.get("sparsity", "0.0")
                variant = combo.get("variant", None)

                if pruning == "none":
                    if variant:
                        test_id = f"{model}_{optimizer}_{variant}_{pruning}"
                    else:
                        test_id = f"{model}_{optimizer}_{pruning}"
                else:
                    sparsity_pct = int(float(sparsity) * 100)
                    if variant:
                        test_id = (
                            f"{model}_{optimizer}_{variant}_{pruning}_{sparsity_pct}"
                        )
                    else:
                        test_id = f"{model}_{optimizer}_{pruning}_{sparsity_pct}"

                if test_id in missing_tests:
                    tests_to_run.append(combo)

            if tests_to_run:
                print(f"\nRunning {len(tests_to_run)} missing test(s)...")

                # Run the missing tests
                all_results = []
                for i, combo in enumerate(tests_to_run, 1):
                    print(f"\n{'='*60}")
                    result = run_single_test(
                        combo,
                        config,
                        str(continue_dir),
                        i,
                        len(tests_to_run),
                        parallel_mode=False,
                        override_epochs=None,
                        time_estimates=None,
                    )
                    if result:
                        all_results.append(result)

                # Update summary
                if all_results:
                    # Reload existing results
                    json_file = os.path.join(str(continue_dir), "all_results.json")
                    existing_results = []
                    if os.path.exists(json_file):
                        with open(json_file, "r") as f:
                            existing_data = json.load(f)
                            if isinstance(existing_data, list):
                                existing_results = existing_data

                    # Merge results
                    for result in all_results:
                        existing_results.append(result)

                    # Save updated results
                    with open(json_file, "w") as f:
                        json.dump(existing_results, f, indent=2)

                    create_summary_report(existing_results, str(continue_dir))
                print(f"\n✓ Completed {len(tests_to_run)} missing test(s)")
                sys.exit(0)

        if not incomplete_runs and failed_runs:
            # All tests completed but some failed - offer to re-run failed tests only
            print(f"\n⚠️  All tests completed but {len(failed_runs)} test(s) FAILED:")
            for failed_test in failed_runs:
                print(f"   ✗ {failed_test}")

            print(f"\nWould you like to remove and re-run only the failed tests?")
            if not args.yes:
                response = input("Continue? (y/N): ").strip().lower()
                if response != "y":
                    print("Aborted.")
                    sys.exit(0)
            else:
                print("Auto-accepting due to --yes flag")

            # Remove failed tests
            for failed_test in failed_runs:
                failed_dir = continue_dir / failed_test
                if failed_dir.exists():
                    shutil.rmtree(failed_dir)
                    print(f"Removed: {failed_test}")

            # Set incomplete_runs to failed_runs for processing below
            incomplete_runs = failed_runs
            incomplete_info["incomplete_runs"] = failed_runs
        elif not incomplete_runs and not failed_runs:
            print("No incomplete or failed runs found. Test matrix is complete!")
            sys.exit(0)
        elif not incomplete_runs:
            print("No incomplete runs found. Test matrix is complete.")
            sys.exit(0)

        # Get configuration from the original directory
        config_file = continue_dir / "config.txt"
        if not config_file.exists():
            config_file = continue_dir / "config.yaml"
            if not config_file.exists():
                print(f"Error: No config file found in {continue_dir}")
                sys.exit(1)

        # Get run information
        incomplete_runs = incomplete_info.get("incomplete_runs", [])
        complete_runs = incomplete_info.get("complete_runs", [])
        failed_runs_from_info = incomplete_info.get("failed_runs", [])

        print(f"\nTest Matrix Continuation Plan:")
        print(f"  Complete runs: {len(complete_runs)}")
        if failed_runs_from_info and failed_runs == failed_runs_from_info:
            print(f"  Failed runs to re-run: {len(failed_runs)}")
        else:
            print(f"  Incomplete runs to remove: {len(incomplete_runs)}")
            if failed_runs_from_info:
                print(f"  Failed runs: {len(failed_runs_from_info)} (not being re-run)")

        # Parse config to get total expected tests
        if config_file.suffix == ".yaml":
            config = parse_yaml_config(str(config_file))
        else:
            config = parse_kconfig(str(config_file))

        matrix = get_test_matrix(config)
        all_combinations = generate_combinations(matrix)

        # Determine which tests need to be run
        complete_test_ids = set(complete_runs)
        tests_to_run = []

        # Complete the full matrix - run all tests not yet complete
        for combo in all_combinations:
            if combo["pruning"] == "none":
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}"
            else:
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}_{sparsity_pct}"

            if test_id not in complete_test_ids:
                tests_to_run.append(combo)

        # Show clear breakdown of what will be run
        total_expected = len(all_combinations)
        never_started = len(tests_to_run) - len(incomplete_runs)
        print(f"  Tests to run: {len(tests_to_run)}")
        if never_started > 0:
            print(f"    - Re-running incomplete: {len(incomplete_runs)}")
            print(f"    - Running never-started: {never_started}")
        print(f"  Total tests in matrix: {total_expected}")

        # Calculate time estimate based on completed tests
        if complete_runs:
            # Try to get timing from completed runs
            total_time = 0
            count = 0
            for run_name in complete_runs:
                metrics_file = continue_dir / run_name / "training_metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)
                            if "elapsed_time" in metrics:
                                total_time += metrics["elapsed_time"]
                                count += 1
                    except:
                        pass

            if count > 0:
                avg_time = total_time / count
                estimated_time = avg_time * len(tests_to_run)
                print(
                    f"  Estimated time: {format_time(estimated_time)} (based on {count} completed runs)"
                )

        print("\nTests to be run:")
        for combo in tests_to_run[:5]:  # Show first 5
            if combo["pruning"] == "none":
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}"
            else:
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}_{sparsity_pct}"
            print(f"  - {test_id}")
        if len(tests_to_run) > 5:
            print(f"  ... and {len(tests_to_run) - 5} more")

        print("\n" + "=" * 60)

        # Simple, clear prompt
        response = (
            input(f"Continue with {len(tests_to_run)} tests? (y/N): ").strip().lower()
        )
        if response != "y":
            print("Aborted.")
            sys.exit(0)

        # Clean incomplete runs
        print("\nCleaning incomplete runs...")
        for run_name in incomplete_runs:
            run_path = continue_dir / run_name
            if run_path.exists():
                shutil.rmtree(run_path)
                print(f"  Removed: {run_name}")

        # Set up to continue with the remaining tests
        args.rerun_dir = str(continue_dir)
        combinations = tests_to_run
        output_dir = str(continue_dir)

        # Load existing results for the report
        existing_results = []
        json_file = continue_dir / "all_results.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                existing_data = json.load(f)
                existing_results = (
                    existing_data if isinstance(existing_data, list) else []
                )

        # Skip the normal flow and jump to test execution
        print(f"\nContinuing with {len(tests_to_run)} remaining tests...")
        total_tests = len(tests_to_run)

    # Handle config directory mode
    elif args.config_dir:
        # Get all config files from directory
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            print(f"Error: Config directory '{config_dir}' does not exist")
            sys.exit(1)

        config_files = sorted(config_dir.glob("config_*"))
        if not config_files:
            print(f"Error: No config files found in '{config_dir}'")
            sys.exit(1)

        print(f"Found {len(config_files)} configuration files to test")

        # For config_dir mode, we'll run each config as a separate test
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"sweep_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nOutput directory: {output_dir}")

        # Run each configuration
        all_results = []
        for i, config_file in enumerate(config_files, 1):
            print(f"\n{'='*60}")
            print(f"Testing configuration {i}/{len(config_files)}: {config_file.name}")
            print("=" * 60)

            # Parse this config
            config = parse_kconfig(str(config_file))

            # Get single combination from this config (it's a fixed config)
            matrix = get_test_matrix(config)
            combinations = generate_combinations(matrix)

            if len(combinations) != 1:
                print(
                    f"Warning: Config {config_file.name} generated {len(combinations)} combinations, expected 1"
                )

            # Run the test
            combo = combinations[0]
            test_name = f"{config_file.stem}_{combo['model']}_{combo['optimizer']}"
            if combo["pruning"] != "none":
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_name += f"_{combo['pruning']}_{sparsity_pct}"

            test_output_dir = output_dir / test_name

            # Copy config file to test directory for reference
            test_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, test_output_dir / "config.txt")

            result = run_single_test(
                combo,
                config,
                test_output_dir,
                i,
                len(config_files),
                override_epochs=args.override_epochs,
            )

            if result:
                result["config_file"] = config_file.name
                result["test_name"] = test_name  # Add test_name for analysis
                all_results.append(result)

        # Save combined results
        summary_file = output_dir / "sweep_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Generate sweep analysis
        generate_sweep_analysis(output_dir, all_results)

        print(f"\n{'='*60}")
        print(f"Sweep completed: {len(all_results)} successful tests")
        print(f"Results saved to: {output_dir}")
        return

    # Skip configuration parsing if we're in continuation mode (already done above)
    if not args.continue_dir:
        # Parse configuration (normal mode)
        if args.config_yaml:
            config = parse_yaml_config(args.config_yaml)
            config_source = args.config_yaml
        else:
            config = parse_kconfig(args.config)
            config_source = args.config

        # Get test matrix
        matrix = get_test_matrix(config)

        # Apply optimizer filter if specified
        if args.filter_optimizer:
            filter_optimizers = [o.strip() for o in args.filter_optimizer.split(",")]
            matrix["optimizers"] = [
                o for o in matrix["optimizers"] if o in filter_optimizers
            ]
            print(f"Filtering to optimizers: {', '.join(filter_optimizers)}")

        print("Test Matrix Configuration:")
        print(f"  Models: {', '.join(matrix['models'])}")
        print(f"  Optimizers: {', '.join(matrix['optimizers'])}")
        print(f"  Pruning methods: {', '.join(matrix['pruning_methods'])}")

        # Generate combinations
        combinations = generate_combinations(matrix)
        print(f"\nTotal test combinations: {len(combinations)}")

    if args.dry_run:
        print("\nDry run - would execute:")
        for i, combo in enumerate(combinations, 1):
            variant_part = f"_{combo['variant']}" if combo.get("variant") else ""
            if combo["pruning"] == "none":
                test_id = f"{combo['model']}_{combo['optimizer']}{variant_part}_{combo['pruning']}"
            else:
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_id = f"{combo['model']}_{combo['optimizer']}{variant_part}_{combo['pruning']}_{sparsity_pct}"
            print(f"  {i}. {test_id}")
        return

    # Create or use existing output directory
    # Note: args.continue_dir sets args.rerun_dir internally
    if args.rerun_dir or args.continue_dir:
        # Use existing directory
        output_dir = (
            args.rerun_dir if not args.continue_dir else output_dir
        )  # output_dir already set in continue mode
        if not os.path.exists(output_dir):
            print(f"Error: Rerun directory '{output_dir}' does not exist")
            sys.exit(1)
        if not args.continue_dir:  # Only print this in normal rerun mode
            print(f"\nReusing existing directory: {output_dir}")

        # Load existing results if available
        if not args.continue_dir:  # In continue mode, existing_results already loaded
            existing_results = []
            json_file = os.path.join(output_dir, "all_results.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    existing_data = json.load(f)
                    existing_results = (
                        existing_data if isinstance(existing_data, list) else []
                    )
                print(f"Found {len(existing_results)} existing test results")
    else:
        # Create new directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        existing_results = []
        print(f"\nOutput directory: {output_dir}")

    # Save configuration (only for new runs, not rerun or continue)
    if not args.rerun_dir and not args.continue_dir:
        if args.config_yaml:
            shutil.copy2(config_source, os.path.join(output_dir, "config.yaml"))
        else:
            shutil.copy2(config_source, os.path.join(output_dir, "config.txt"))

    # Run all combinations
    if args.rerun_dir:
        # When re-running, keep existing results for tests not being re-run
        # Filter out results that will be re-run
        rerun_test_ids = set()
        for combo in combinations:
            if combo["pruning"] == "none":
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}"
            else:
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}_{sparsity_pct}"
            rerun_test_ids.add(test_id)

        # Keep existing results that are not being re-run
        results = [
            r for r in existing_results if r.get("test_id") not in rerun_test_ids
        ]
        print(
            f"Keeping {len(results)} existing results, re-running {len(rerun_test_ids)} tests"
        )
    else:
        results = []

    if not args.continue_dir:  # In continue mode, total_tests already set
        total_tests = len(combinations)

    # Show test plan preview before starting (skip for continuation mode)
    if not args.continue_dir:
        print("\n" + "=" * 60)
        if total_tests == 1:
            print("SINGLE TEST EXECUTION (using test matrix framework)")
        else:
            print("TEST MATRIX EXECUTION PLAN")
        print("=" * 60)
        print(f"Total tests to run: {total_tests}")
        print(f"Parallel jobs: {args.parallel}")
        print(f"Output directory: {output_dir}")
        print("\nTests that will be executed:")

    # Group tests by optimizer for clearer display (skip for continuation)
    if not args.continue_dir:
        tests_by_optimizer = {}
        for combo in combinations:
            opt = combo["optimizer"]
            if opt not in tests_by_optimizer:
                tests_by_optimizer[opt] = []

            variant_str = f"{combo.get('variant', '')} " if combo.get("variant") else ""
            if combo["pruning"] == "none":
                test_desc = f"  - {variant_str}{combo['pruning']} (no sparsity)"
            else:
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_desc = (
                    f"  - {variant_str}{combo['pruning']} @ {sparsity_pct}% sparsity"
                )
            tests_by_optimizer[opt].append(test_desc)

        for optimizer in sorted(tests_by_optimizer.keys()):
            # Check if AdamWPrune has a base optimizer configured
            optimizer_display = optimizer.upper()
            if optimizer == "adamwprune" and "ADAMWPRUNE_BASE_OPTIMIZER_NAME" in config:
                base_opt = config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"]
                optimizer_display = f"{optimizer.upper()} (base: {base_opt.upper()})"

            print(
                f"\n{optimizer_display} ({len(tests_by_optimizer[optimizer])} tests):"
            )
            for test in tests_by_optimizer[optimizer]:
                print(test)

        print("\n" + "=" * 60)

    if not args.dry_run:
        # Check if YES=1 environment variable is set to skip confirmation
        auto_yes = os.environ.get("YES", "").strip() == "1"

        # Always ask for confirmation unless YES=1 is set or in continuation mode (already asked)
        if not auto_yes and not args.continue_dir:
            print(
                f"\nAbout to run {total_tests} training job{'s' if total_tests != 1 else ''}."
            )
            response = input("Continue? (y/N): ").strip().lower()
            if response != "y":
                print("Aborted by user.")
                sys.exit(0)
        elif auto_yes and not args.continue_dir:
            print(
                f"\nAuto-confirmed (YES=1): Running {total_tests} training job{'s' if total_tests != 1 else ''}."
            )

        if not args.continue_dir:
            print("\nStarting test matrix execution...")

    # Get parallel settings from config or args
    parallel_jobs = args.parallel
    if parallel_jobs == 1 and config.get("PARALLEL_JOBS"):
        parallel_jobs = int(config.get("PARALLEL_JOBS", 1))

    max_memory_percent = int(config.get("MAX_GPU_MEMORY_PERCENT", 90))

    if parallel_jobs > 1:
        print(
            f"\nRunning {parallel_jobs} parallel jobs with {max_memory_percent}% max GPU memory"
        )
        print(f"Total GPU memory: {get_gpu_memory_total()}MB")
        print("Using parallel execution with GPU memory monitoring...")

        # Prepare arguments for parallel execution
        job_args = []
        for i, combo in enumerate(combinations, 1):
            job_args.append(
                (
                    combo,
                    config,
                    output_dir,
                    i,
                    total_tests,
                    max_memory_percent,
                    args.override_epochs,
                    None,  # time_estimates - will be calculated dynamically
                )
            )

        # Run tests in parallel with thread pool
        completed_results = []
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            # Submit all jobs
            future_to_combo = {
                executor.submit(run_single_test_wrapper, args): args
                for args in job_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_combo):
                try:
                    result = future.result()
                    if result:
                        completed_results.append(result)

                        # Stop on failure if requested
                        if args.stop_on_failure and not result.get("success", False):
                            print(f"\n{'='*60}")
                            print("Stopping due to test failure (--stop-on-failure)")
                            print(f"{'='*60}")
                            # Cancel remaining futures
                            for f in future_to_combo:
                                f.cancel()
                            break

                except Exception as e:
                    print(f"Error in parallel job: {e}")

        results.extend(completed_results)
    else:
        # Serial execution (original behavior)
        for i, combo in enumerate(combinations, 1):
            # Calculate time estimates based on completed tests
            time_estimates = None
            if i > 1:  # After first test completes
                # Get remaining combinations for variant-aware estimates
                remaining_combinations = combinations[
                    i - 1 :
                ]  # Current and future tests
                time_estimates = calculate_variant_aware_time_estimates(
                    results, remaining_combinations
                )
                if (
                    i == 2 and time_estimates
                ):  # After first test, show full test plan estimate
                    print(f"\n{'='*60}")
                    print(
                        f"Time Estimate: Total remaining time approximately {format_time(time_estimates['total_remaining'])}"
                    )
                    if "optimizer_times" in time_estimates:
                        print("Per-optimizer/variant estimates:")
                        for key, time in time_estimates["optimizer_times"].items():
                            print(f"  {key}: {format_time(time)}")
                    print(f"{'='*60}")

            result = run_single_test(
                combo,
                config,
                output_dir,
                i,
                total_tests,
                override_epochs=args.override_epochs,
                time_estimates=time_estimates,
            )
            if result:
                results.append(result)

                # Stop on failure if requested
                if args.stop_on_failure and not result.get("success", False):
                    print(f"\n{'='*60}")
                    print("Stopping due to test failure (--stop-on-failure)")
                    print(f"{'='*60}")
                    break

    # Create summary report
    create_summary_report(results, output_dir)

    # Generate graphs if configured
    if config.get("AUTO_GENERATE_GRAPHS") == "y":
        print("\nGenerating comparison graphs...")
        try:
            graph_dir = os.path.join(output_dir, "graphs")
            cmd = [
                "python3",
                "scripts/generate_optimizer_graphs.py",
                output_dir,
                graph_dir,
            ]
            subprocess.run(cmd, check=True)
            print(f"Graphs saved to: {graph_dir}")
        except Exception as e:
            print(f"Warning: Could not generate graphs: {e}")

    print("\n" + "=" * 60)
    print("Test matrix execution completed!")
    print("=" * 60)
    print(f"Results directory: {output_dir}")
    print("To generate/update graphs, run: make update-graphs")


if __name__ == "__main__":
    main()
