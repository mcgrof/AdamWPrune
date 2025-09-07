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
):
    """Run a single test combination."""
    model = combination["model"]
    optimizer = combination["optimizer"]
    pruning = combination["pruning"]
    sparsity = combination.get("sparsity", "0.0")

    # Create test identifier including sparsity level
    if pruning == "none":
        test_id = f"{model}_{optimizer}_{pruning}"
    else:
        # Include sparsity in the test ID (e.g., sgd_movement_90)
        sparsity_pct = int(float(sparsity) * 100)
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
        print(f"Pruning: {pruning}")
        if pruning != "none":
            print(f"Sparsity: {float(sparsity)*100:.0f}%")
        print(f"Output: {test_output_dir}")
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

    # AdamWPrune has built-in state-based pruning
    if optimizer == "adamwprune" and pruning == "state":
        # For AdamWPrune with state pruning, we don't pass pruning-method
        # as it's handled internally by the optimizer
        cmd.extend(["--pruning-method", "none"])  # No external pruning
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
        if "ADAMWPRUNE_BETA1" in config:
            cmd.extend(["--adamwprune-beta1", config["ADAMWPRUNE_BETA1"]])
        if "ADAMWPRUNE_BETA2" in config:
            cmd.extend(["--adamwprune-beta2", config["ADAMWPRUNE_BETA2"]])
        if "ADAMWPRUNE_WEIGHT_DECAY" in config:
            cmd.extend(["--adamwprune-weight-decay", config["ADAMWPRUNE_WEIGHT_DECAY"]])
        if "ADAMWPRUNE_AMSGRAD" in config:
            amsgrad_val = "true" if config["ADAMWPRUNE_AMSGRAD"] == "y" else "false"
            cmd.extend(["--adamwprune-amsgrad", amsgrad_val])

    # Note: batch size is configured via config.py, not command line arguments

    # JSON output for metrics - use absolute path since training runs from model dir
    json_output = os.path.abspath(
        os.path.join(test_output_dir, "training_metrics.json")
    )
    cmd.extend(["--json-output", json_output])

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

        # Add test metadata
        metrics["test_id"] = test_id
        metrics["model"] = model
        metrics["optimizer"] = optimizer
        metrics["pruning"] = pruning
        metrics["target_sparsity"] = float(sparsity) if pruning != "none" else 0
        metrics["elapsed_time"] = elapsed_time
        metrics["success"] = True

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


def create_summary_report(results, output_dir):
    """Create a summary report of all test results."""
    json_file = os.path.join(output_dir, "all_results.json")

    # Save JSON results
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

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
    args = parser.parse_args()

    # Handle config directory mode
    if args.config_dir:
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
            if combo["pruning"] == "none":
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}"
            else:
                sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
                test_id = f"{combo['model']}_{combo['optimizer']}_{combo['pruning']}_{sparsity_pct}"
            print(f"  {i}. {test_id}")
        return

    # Create or use existing output directory
    if args.rerun_dir:
        # Use existing directory
        output_dir = args.rerun_dir
        if not os.path.exists(output_dir):
            print(f"Error: Rerun directory '{output_dir}' does not exist")
            sys.exit(1)
        print(f"\nReusing existing directory: {output_dir}")

        # Load existing results if available
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

    # Save configuration (only for new runs)
    if not args.rerun_dir:
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

    total_tests = len(combinations)

    # Show test plan preview before starting
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

    # Group tests by optimizer for clearer display
    tests_by_optimizer = {}
    for combo in combinations:
        opt = combo["optimizer"]
        if opt not in tests_by_optimizer:
            tests_by_optimizer[opt] = []

        if combo["pruning"] == "none":
            test_desc = f"  - {combo['pruning']} (no sparsity)"
        else:
            sparsity_pct = int(float(combo.get("sparsity", "0")) * 100)
            test_desc = f"  - {combo['pruning']} @ {sparsity_pct}% sparsity"
        tests_by_optimizer[opt].append(test_desc)

    for optimizer in sorted(tests_by_optimizer.keys()):
        # Check if AdamWPrune has a base optimizer configured
        optimizer_display = optimizer.upper()
        if optimizer == "adamwprune" and "ADAMWPRUNE_BASE_OPTIMIZER_NAME" in config:
            base_opt = config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"]
            optimizer_display = f"{optimizer.upper()} (base: {base_opt.upper()})"

        print(f"\n{optimizer_display} ({len(tests_by_optimizer[optimizer])} tests):")
        for test in tests_by_optimizer[optimizer]:
            print(test)

    print("\n" + "=" * 60)

    if not args.dry_run:
        # Check if YES=1 environment variable is set to skip confirmation
        auto_yes = os.environ.get("YES", "").strip() == "1"

        # Always ask for confirmation unless YES=1 is set
        if not auto_yes:
            print(
                f"\nAbout to run {total_tests} training job{'s' if total_tests != 1 else ''}."
            )
            response = input("Continue? (y/N): ").strip().lower()
            if response != "y":
                print("Aborted by user.")
                sys.exit(0)
        else:
            print(
                f"\nAuto-confirmed (YES=1): Running {total_tests} training job{'s' if total_tests != 1 else ''}."
            )

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
            result = run_single_test(
                combo,
                config,
                output_dir,
                i,
                total_tests,
                override_epochs=args.override_epochs,
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
