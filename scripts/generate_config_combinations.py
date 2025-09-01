#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate all configuration combinations from a range-based config file.

Usage:
    python generate_config_combinations.py <input_config> <output_dir>
"""

import sys
import os
from pathlib import Path
import itertools
from typing import List, Dict, Any, Tuple


def parse_range_value(value: str) -> List[str]:
    """
    Parse a config value that can be:
    - Single value: "0.9"
    - Range: "0.1:0.1:0.5" (start:step:end)
    - Choices: "y|n" or "option1|option2|option3"
    """
    value = value.strip().strip('"')

    # Check for choices (pipe-separated)
    if "|" in value:
        return value.split("|")

    # Check for range (colon-separated with 3 parts)
    if ":" in value and value.count(":") == 2:
        parts = value.split(":")
        try:
            start = float(parts[0])
            step = float(parts[1])
            end = float(parts[2])

            # Generate range values
            values = []
            current = start
            while current <= end + 1e-9:  # Small epsilon for floating point
                # Format based on whether it's an integer or float
                if step == int(step) and current == int(current):
                    values.append(str(int(current)))
                else:
                    values.append(f"{current:.2f}" if current < 1 else str(current))
                current += step
            return values
        except ValueError:
            # If it's not numeric, treat as single value
            return [value]

    # Single value
    return [value]


def parse_config_file(filepath: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Parse a config file and return:
    - variable_options: dict of variable_name -> list of possible values
    - fixed_values: dict of variable_name -> single value
    """
    variable_options = {}
    fixed_values = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse CONFIG_NAME=value lines
            if "=" in line:
                # Split on first = only
                key, value = line.split("=", 1)
                key = key.strip()

                # Remove inline comments (everything after # that's not in quotes)
                if "#" in value:
                    # Simple approach: if # appears outside quotes, remove from there
                    in_quotes = False
                    for i, char in enumerate(value):
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == "#" and not in_quotes:
                            value = value[:i]
                            break

                value = value.strip()

                # Parse the value to get options
                options = parse_range_value(value)

                if len(options) > 1:
                    variable_options[key] = options
                else:
                    fixed_values[key] = options[0]

    return variable_options, fixed_values


def generate_combinations(
    variable_options: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    """Generate all combinations of variable options."""
    if not variable_options:
        return [{}]

    # Get keys and values in consistent order
    keys = sorted(variable_options.keys())
    value_lists = [variable_options[key] for key in keys]

    # Generate all combinations
    combinations = []
    for values in itertools.product(*value_lists):
        combo = dict(zip(keys, values))
        combinations.append(combo)

    return combinations


def write_config_file(
    filepath: str,
    fixed_values: Dict[str, str],
    variable_values: Dict[str, str],
    combo_index: int,
):
    """Write a single configuration file."""
    with open(filepath, "w") as f:
        # Write header
        f.write(f"# SPDX-License-Identifier: MIT\n")
        f.write(f"# Auto-generated configuration {combo_index}\n")
        f.write(f"# Variable parameters for this run:\n")

        # Write variable parameters as comments
        for key, value in sorted(variable_values.items()):
            f.write(f"# {key} = {value}\n")
        f.write("\n")

        # Write all configuration values
        all_values = {**fixed_values, **variable_values}
        for key in sorted(all_values.keys()):
            value = all_values[key]
            # Add quotes if needed for string values
            if not value.replace(".", "").replace("-", "").isdigit() and value not in [
                "y",
                "n",
            ]:
                if not (value.startswith('"') and value.endswith('"')):
                    value = f'"{value}"'
            f.write(f"{key}={value}\n")


def create_summary_file(
    output_dir: Path,
    combinations: List[Dict[str, str]],
    variable_options: Dict[str, List[str]],
):
    """Create a summary file listing all combinations."""
    summary_file = output_dir / "combinations_summary.txt"

    with open(summary_file, "w") as f:
        f.write("Configuration Combinations Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total combinations: {len(combinations)}\n\n")

        f.write("Variable parameters:\n")
        for key, values in sorted(variable_options.items()):
            f.write(f"  {key}: {', '.join(values)}\n")
        f.write("\n")

        f.write("Combinations:\n")
        f.write("-" * 50 + "\n")

        for i, combo in enumerate(combinations, 1):
            f.write(f"\nConfiguration {i:03d}:\n")
            for key, value in sorted(combo.items()):
                # Extract just the parameter name for readability
                param_name = (
                    key.replace("CONFIG_", "")
                    .replace("ADAMWPRUNE_", "")
                    .replace("PRUNING_", "")
                )
                f.write(f"  {param_name}: {value}\n")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python generate_config_combinations.py <input_config> <output_dir>"
        )
        sys.exit(1)

    input_config = sys.argv[1]
    output_dir = Path(sys.argv[2])

    # Check input file exists
    if not os.path.exists(input_config):
        print(f"Error: Input file '{input_config}' not found")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse the config file
    print(f"Parsing {input_config}...")
    variable_options, fixed_values = parse_config_file(input_config)

    print(f"Found {len(variable_options)} variable parameters:")
    for key, values in variable_options.items():
        print(f"  {key}: {len(values)} options")

    # Generate all combinations
    combinations = generate_combinations(variable_options)
    print(f"\nGenerating {len(combinations)} configurations...")

    # Write each configuration
    for i, combo in enumerate(combinations, 1):
        config_name = f"config_{i:03d}"
        config_path = output_dir / config_name
        write_config_file(config_path, fixed_values, combo, i)

        if i % 10 == 0:
            print(f"  Generated {i}/{len(combinations)} configurations...")

    # Create summary file
    create_summary_file(output_dir, combinations, variable_options)

    print(
        f"\nSuccessfully generated {len(combinations)} configurations in {output_dir}"
    )
    print(f"Summary written to {output_dir}/combinations_summary.txt")

    # Show a sample of what will be tested
    if len(combinations) > 0:
        print("\nSample configuration (config_001):")
        for key, value in sorted(combinations[0].items()):
            param_name = (
                key.replace("CONFIG_", "")
                .replace("ADAMWPRUNE_", "")
                .replace("PRUNING_", "")
            )
            print(f"  {param_name}: {value}")


if __name__ == "__main__":
    main()
