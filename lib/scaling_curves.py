"""
Inference Scaling Law U-Curves Visualization

Based on research showing two key U-shaped trade-offs:
1. Training: Validation loss vs MLP:Attention width ratio
2. Inference: Latency vs Depth (at fixed parameter budget)

This module provides ASCII visualization to show where current
hyperparameters fall on these curves.
"""

import math


def plot_ascii_u_curve(
    x_min, x_max, x_opt, x_current, x_label, y_label, title, width=60, height=15
):
    """
    Plot a U-shaped curve in ASCII with current position marked.

    Args:
        x_min: Minimum x value for the curve
        x_max: Maximum x value for the curve
        x_opt: Optimal x value (minimum of U-curve)
        x_current: Current x value to mark on the curve
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Title of the plot
        width: Width of the plot in characters
        height: Height of the plot in characters

    Returns:
        String containing the ASCII plot
    """
    lines = []

    # Title
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    # Create the grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Define U-curve function: y = a*(x - x_opt)^2 + b
    # Normalize to fit in height
    a = 1.0
    b = 0.0

    def u_curve(x):
        return a * (x - x_opt) ** 2 + b

    # Calculate y range for normalization
    y_values = [u_curve(x) for x in [x_min, x_max, x_opt]]
    y_min_val = min(y_values)
    y_max_val = max(y_values)

    # Plot the curve
    for col in range(width):
        # Map column to x value
        x = x_min + (x_max - x_min) * col / (width - 1)
        y_val = u_curve(x)

        # Map y value to row (inverted, 0 is top)
        if y_max_val > y_min_val:
            norm_y = (y_val - y_min_val) / (y_max_val - y_min_val)
        else:
            norm_y = 0.5
        row = int((1.0 - norm_y) * (height - 1))
        row = max(0, min(height - 1, row))

        grid[row][col] = "·"

    # Mark optimal point
    opt_col = int((x_opt - x_min) / (x_max - x_min) * (width - 1))
    opt_col = max(0, min(width - 1, opt_col))
    opt_y = u_curve(x_opt)
    if y_max_val > y_min_val:
        norm_opt_y = (opt_y - y_min_val) / (y_max_val - y_min_val)
    else:
        norm_opt_y = 0.5
    opt_row = int((1.0 - norm_opt_y) * (height - 1))
    opt_row = max(0, min(height - 1, opt_row))
    grid[opt_row][opt_col] = "★"

    # Mark current point if within range
    if x_min <= x_current <= x_max:
        cur_col = int((x_current - x_min) / (x_max - x_min) * (width - 1))
        cur_col = max(0, min(width - 1, cur_col))
        cur_y = u_curve(x_current)
        if y_max_val > y_min_val:
            norm_cur_y = (cur_y - y_min_val) / (y_max_val - y_min_val)
        else:
            norm_cur_y = 0.5
        cur_row = int((1.0 - norm_cur_y) * (height - 1))
        cur_row = max(0, min(height - 1, cur_row))
        grid[cur_row][cur_col] = "●"

    # Add y-axis with fixed-width label (10 chars for alignment)
    # Pad y_label to fit
    label_width = 10

    for i, row in enumerate(grid):
        if i == 0:
            label = f"{y_label:>{label_width}s} │"
        elif i == height - 1:
            label = f"{'(good)':>{label_width}s} │"
        else:
            label = " " * label_width + " │"
        lines.append(label + "".join(row))

    # Add x-axis
    lines.append(" " * label_width + " └" + "─" * width)

    # Add x-axis labels (aligned with the left margin)
    lines.append(" " * (label_width + 2) + f"{x_label}")
    lines.append(
        " " * (label_width + 1)
        + f"{x_min:.1f}"
        + " " * (width - 20)
        + f"{x_opt:.1f}★"
        + " " * 8
        + f"{x_max:.1f}"
    )

    # Add legend
    lines.append("")
    lines.append("  ★ = Optimal    ● = Current configuration")
    lines.append("")

    # Add assessment
    if x_min <= x_current <= x_max:
        distance = abs(x_current - x_opt)
        relative_distance = distance / (x_max - x_min)

        if relative_distance < 0.1:
            assessment = "✓ Excellent - Very close to optimal"
        elif relative_distance < 0.2:
            assessment = "✓ Good - Near optimal range"
        elif relative_distance < 0.35:
            assessment = "⚠ Moderate - Consider adjusting towards optimal"
        else:
            assessment = "⚠ Suboptimal - Significant deviation from optimal"

        direction = "increase" if x_current < x_opt else "decrease"
        lines.append(f"  Assessment: {assessment}")
        lines.append(f"  Current: {x_current:.2f} | Optimal: {x_opt:.2f}")
        if relative_distance > 0.1:
            lines.append(
                f"  Suggestion: {direction.capitalize()} {x_label.lower()} towards {x_opt:.2f}"
            )
    else:
        lines.append(
            f"  Current value ({x_current:.2f}) is outside plot range [{x_min:.1f}, {x_max:.1f}]"
        )

    return "\n".join(lines)


def show_mlp_attention_ratio_curve(mlp_width, attn_width):
    """
    Show where the current MLP:Attention width ratio falls on the training U-curve.

    Based on inference scaling law research showing optimal ratio around 1.4.

    Args:
        mlp_width: MLP hidden dimension
        attn_width: Attention dimension (d_model)

    Returns:
        ASCII plot string
    """
    if attn_width == 0:
        ratio = 0.0
    else:
        ratio = mlp_width / attn_width

    return plot_ascii_u_curve(
        x_min=0.5,
        x_max=3.0,
        x_opt=1.4,
        x_current=ratio,
        x_label="MLP:Attention Width Ratio",
        y_label="Val Loss",
        title="Training Efficiency: MLP:Attention Width Ratio",
        width=60,
        height=12,
    )


def show_depth_latency_curve(n_layers, param_budget_gb=None):
    """
    Show where the current depth falls on the inference latency U-curve.

    Based on inference scaling law research showing optimal depth around 28 layers
    at fixed parameter budget.

    Args:
        n_layers: Number of transformer layers
        param_budget_gb: (Optional) Parameter budget in GB for context

    Returns:
        ASCII plot string
    """
    title = "Inference Efficiency: Depth vs Latency"
    if param_budget_gb:
        title += f" ({param_budget_gb:.1f}GB param budget)"

    return plot_ascii_u_curve(
        x_min=8,
        x_max=64,
        x_opt=28,
        x_current=n_layers,
        x_label="Transformer Depth (layers)",
        y_label="Latency",
        title=title,
        width=60,
        height=12,
    )


def show_scaling_curves(n_layers, d_model, mlp_dim, param_count=None):
    """
    Show both U-curves for the current model configuration.

    Args:
        n_layers: Number of transformer layers
        d_model: Model dimension (attention width)
        mlp_dim: MLP hidden dimension
        param_count: (Optional) Total parameter count

    Returns:
        String with both ASCII plots
    """
    output = []

    output.append("\n" + "=" * 70)
    output.append("INFERENCE SCALING LAW ANALYSIS")
    output.append("=" * 70)
    output.append("")

    # Show model configuration
    output.append("Model Configuration:")
    output.append(f"  Layers: {n_layers}")
    output.append(f"  d_model: {d_model}")
    output.append(f"  MLP dim: {mlp_dim}")
    output.append(f"  MLP:Attn ratio: {mlp_dim/d_model:.2f}")
    if param_count:
        param_gb = param_count / 1e9
        output.append(f"  Parameters: {param_count/1e6:.1f}M ({param_gb:.2f}GB)")
    output.append("")

    # Show training U-curve
    output.append(show_mlp_attention_ratio_curve(mlp_dim, d_model))
    output.append("")

    # Show inference U-curve
    param_gb = param_count / 1e9 if param_count else None
    output.append(show_depth_latency_curve(n_layers, param_gb))
    output.append("")

    output.append("=" * 70)
    output.append("Note: Curves based on inference scaling law research")
    output.append("      Optimal values may vary based on specific use case")
    output.append("=" * 70)
    output.append("")

    return "\n".join(output)
