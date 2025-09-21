#!/usr/bin/env python3
"""
TrackIO View - Terminal-based metrics viewer for TrackIO projects.

This provides a console UI for viewing TrackIO metrics in real-time without needing a web browser.
Perfect for monitoring training progress on remote servers or in terminal-only environments.

Ported from the official TrackIO implementation for better terminal handling and UI.
"""

import argparse
import json
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import sqlite3

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text
    from rich import box
    from rich.columns import Columns

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Term:
    """Terminal control codes and variables"""

    width: int = 80
    height: int = 24
    resized: bool = True
    _w: int = 0
    _h: int = 0

    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"
    alt_screen = "\033[?1049h"
    normal_screen = "\033[?1049l"
    clear = "\033[2J\033[0;0f"
    normal = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    underline = "\033[4m"

    @classmethod
    def refresh(cls):
        """Get terminal dimensions"""
        try:
            cls._w, cls._h = os.get_terminal_size()
        except:
            cls._w, cls._h = 80, 24

        if cls._w != cls.width or cls._h != cls.height:
            cls.width = cls._w
            cls.height = cls._h
            cls.resized = True


class Color:
    """Color management for terminal output"""

    @staticmethod
    def fg(r: int, g: int, b: int) -> str:
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def bg(r: int, g: int, b: int) -> str:
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def gradient(value: float, colors: List[Tuple[int, int, int]]) -> str:
        """Generate color based on value (0.0-1.0) across gradient"""
        if value <= 0:
            return Color.fg(*colors[0])
        if value >= 1:
            return Color.fg(*colors[-1])

        segment_size = 1.0 / (len(colors) - 1)
        segment = int(value / segment_size)
        segment_pos = (value % segment_size) / segment_size

        if segment >= len(colors) - 1:
            return Color.fg(*colors[-1])

        c1 = colors[segment]
        c2 = colors[segment + 1]

        r = int(c1[0] + (c2[0] - c1[0]) * segment_pos)
        g = int(c1[1] + (c2[1] - c1[1]) * segment_pos)
        b = int(c1[2] + (c2[2] - c1[2]) * segment_pos)

        return Color.fg(r, g, b)


class Theme:
    """Color theme definitions"""

    # Loss gradient (blue -> green -> yellow -> red)
    loss_gradient = [
        (0, 100, 200),  # Blue (low loss)
        (0, 200, 100),  # Green
        (200, 200, 0),  # Yellow
        (255, 100, 0),  # Orange
        (255, 0, 0),  # Red (high loss)
    ]

    # Learning rate gradient
    lr_gradient = [
        (100, 100, 255),  # Light blue
        (200, 100, 255),  # Purple
        (255, 100, 200),  # Pink
    ]

    # Sparsity gradient (red -> yellow -> green)
    sparsity_gradient = [
        (255, 0, 0),  # Red (0% sparse)
        (255, 200, 0),  # Orange
        (255, 255, 0),  # Yellow
        (100, 255, 0),  # Light green
        (0, 255, 0),  # Green (100% sparse)
    ]

    main_fg = Color.fg(200, 200, 200)
    title = Color.fg(255, 255, 255)
    border = Color.fg(100, 100, 100)
    text = Color.fg(180, 180, 180)
    success = Color.fg(0, 255, 100)
    warning = Color.fg(255, 200, 0)
    error = Color.fg(255, 50, 50)


class Graph:
    """Enhanced graph with gradient colors and smooth rendering"""

    def __init__(
        self, width: int, height: int, min_value: float = 0, max_value: float = 100
    ):
        self.width = width
        self.height = height
        self.min_value = min_value
        self.max_value = max_value
        self.data = deque(maxlen=width)
        self.markers = []  # For marking special points

    def add_value(self, value: float):
        """Add a new value to the graph"""
        self.data.append(value)

    def add_marker(self, position: int, label: str):
        """Add a marker at a specific position"""
        self.markers.append((position, label))

    def draw(
        self, gradient: List[Tuple[int, int, int]], show_values: bool = True
    ) -> List[str]:
        """Draw the graph with gradient colors"""
        lines = []

        if not self.data:
            return [" " * self.width] * self.height

        # Normalize data
        data_list = list(self.data)
        while len(data_list) < self.width:
            data_list.insert(0, self.min_value)

        # Auto-adjust range if needed
        actual_min = min(data_list)
        actual_max = max(data_list)
        if actual_min < self.min_value:
            self.min_value = actual_min * 0.95
        if actual_max > self.max_value:
            self.max_value = actual_max * 1.05

        range_val = self.max_value - self.min_value
        if range_val == 0:
            range_val = 1

        # Create graph matrix
        matrix = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Plot data points with smoothing
        for col in range(len(data_list)):
            value = data_list[col]
            norm_value = (value - self.min_value) / range_val
            row = int((1 - norm_value) * (self.height - 1))

            if 0 <= row < self.height:
                # Use different characters for visual variety
                if col > 0:
                    prev_value = data_list[col - 1]
                    prev_norm = (prev_value - self.min_value) / range_val
                    prev_row = int((1 - prev_norm) * (self.height - 1))

                    # Draw connecting lines for smooth graph
                    if abs(prev_row - row) > 1:
                        step = 1 if row > prev_row else -1
                        for r in range(prev_row, row, step):
                            if 0 <= r < self.height:
                                matrix[r][col] = "│"

                # Use dots for actual data points
                matrix[row][col] = (
                    "●" if norm_value > 0.8 else "○" if norm_value > 0.5 else "·"
                )

        # Render with colors
        for row in range(self.height):
            line_chars = []
            for col in range(self.width):
                if matrix[row][col] != " ":
                    # Calculate color based on height position
                    height_ratio = 1 - (row / (self.height - 1))
                    color = Color.gradient(height_ratio, gradient)
                    line_chars.append(color + matrix[row][col] + Term.normal)
                else:
                    line_chars.append(" ")
            lines.append("".join(line_chars))

        # Add scale on the left if requested
        if show_values:
            for i, line in enumerate(lines):
                scale_val = self.max_value - (i / (self.height - 1)) * range_val
                scale_str = f"{scale_val:7.2f} "
                lines[i] = scale_str + line

        return lines


class TrackIOConsole:
    """Console UI for TrackIO metrics visualization."""

    def __init__(self, project: str = None):
        """Initialize the console."""
        self.project = project or self._detect_project()
        self.data_dir = None
        self.db_path = None
        self.use_advanced_ui = True

        # Find TrackIO database or data directory
        self._find_trackio_data()

    def _detect_project(self) -> str:
        """Try to detect project name from environment or default."""
        # Check for environment variable
        project = os.environ.get("TRACKIO_PROJECT")
        if project:
            return project

        # Try to detect from current directory name
        cwd = Path.cwd()

        # Generate project name from directory
        import hashlib

        dir_name = cwd.name
        path_hash = hashlib.md5(str(cwd).encode()).hexdigest()[:8]
        return f"{dir_name}-{path_hash}"

    def _find_trackio_data(self):
        """Find TrackIO database or data directory."""
        # Check for database in standard locations
        self.trackio_dirs = [
            Path.home() / ".cache" / "huggingface" / "trackio" / f"{self.project}.db",
            Path.home() / ".cache" / "trackio" / f"{self.project}.db",
            Path.home() / ".trackio" / f"{self.project}.db",
            Path.cwd() / "trackio.db",
            Path.cwd() / ".trackio" / "trackio.db",
        ]

        for db_path in self.trackio_dirs:
            if db_path.exists():
                self.db_path = db_path
                print(f"Found database: {db_path}")
                return

        # Check for data directories
        data_dirs = [
            Path.home() / ".trackio" / self.project,
            Path.cwd() / "trackio_data",
            Path.cwd() / ".trackio",
        ]

        for data_dir in data_dirs:
            if data_dir.exists():
                self.data_dir = data_dir
                print(f"Found data directory: {data_dir}")
                return

        print(f"Warning: No TrackIO data found for project '{self.project}'")

    def find_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Find and read the latest metrics from database or files."""
        if self.db_path:
            return self._read_from_database()
        elif self.data_dir:
            return self._read_from_json()
        return None

    def _read_from_database(self) -> Optional[Dict[str, Any]]:
        """Read metrics from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get latest metrics
            cursor.execute(
                """
                SELECT id, timestamp, run_name, step, metrics
                FROM metrics
                ORDER BY timestamp DESC
                LIMIT 100
            """
            )

            rows = cursor.fetchall()
            if not rows:
                return None

            metrics = {"data": []}

            for row in rows:
                try:
                    # row[4] is the metrics JSON string
                    metric_data = json.loads(row[4]) if row[4] else {}

                    # Flatten the structure for easier display
                    entry = {
                        "timestamp": row[1],
                        "run_name": row[2],
                        "step": row[3],
                    }

                    # Merge in the actual metrics
                    entry.update(metric_data)
                    metrics["data"].append(entry)
                except json.JSONDecodeError:
                    continue

            conn.close()
            return metrics
        except Exception as e:
            print(f"Error reading database: {e}")
            return None

    def _read_from_json(self) -> Optional[Dict[str, Any]]:
        """Read metrics from JSON files."""
        metrics = {}

        # Find all run directories
        run_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not run_dirs:
            return None

        # Get latest run
        latest_run = run_dirs[-1]
        metrics_file = latest_run / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                metrics["run_name"] = latest_run.name
                metrics["data"] = data

        return metrics

    def parse_training_log(self, log_path: Path) -> Dict[str, Any]:
        """Parse a training output.log file for metrics."""
        metrics = {
            "iterations": [],
            "losses": [],
            "perplexities": [],
            "learning_rates": [],
            "sparsities": [],
            "times": [],
        }

        if not log_path.exists():
            return metrics

        with open(log_path) as f:
            for line in f:
                if "Iter" in line and "loss" in line and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        try:
                            metrics["iterations"].append(int(parts[0].split()[-1]))
                            metrics["losses"].append(float(parts[1].split()[-1]))
                            metrics["perplexities"].append(float(parts[2].split()[-1]))

                            lr_str = parts[3].split()[-1]
                            # Convert scientific notation
                            if "e" in lr_str:
                                metrics["learning_rates"].append(float(lr_str))
                            else:
                                metrics["learning_rates"].append(float(lr_str))

                            sparsity_str = parts[4].split()[1].rstrip("%")
                            metrics["sparsities"].append(float(sparsity_str))

                            # Extract time if present
                            if "ms/iter" in parts[-1]:
                                time_str = parts[-1].split()[-1].replace("ms/iter", "")
                                metrics["times"].append(float(time_str))
                        except (ValueError, IndexError):
                            continue

        return metrics

    def create_ascii_graph(
        self, values: List[float], width: int = 50, height: int = 10, title: str = ""
    ) -> str:
        """Create an ASCII graph of values with gradient coloring."""
        if not values:
            return "No data"

        min_val = min(values)
        max_val = max(values)

        # Use the enhanced Graph class
        graph = Graph(width, height, min_val, max_val)

        # Sample if too many values
        if len(values) > width:
            step = len(values) // width
            sampled_values = values[::step]
        else:
            sampled_values = values

        # Add all values to graph
        for v in sampled_values:
            graph.add_value(v)

        # Draw with gradient colors
        lines = graph.draw(Theme.loss_gradient, show_values=True)

        if title:
            lines.insert(0, "")
            lines.insert(0, f"  {Theme.title}{title}{Term.normal}")

        # Add x-axis
        lines.append(f"{'':7s} +{'-' * width}")

        return "\n".join(lines)

    def display_simple(self, metrics: Dict[str, Any]):
        """Display metrics in simple ASCII format (no rich library)."""
        print(f"{Term.clear}{Term.hide_cursor}")

        # Header with colors
        print(f"{Theme.border}" + "═" * min(Term.width - 2, 80) + f"{Term.normal}")
        print(
            f"  {Theme.title}TrackIO Console Dashboard - Project: {self.project}{Term.normal}"
        )
        print(f"{Theme.border}" + "═" * min(Term.width - 2, 80) + f"{Term.normal}")

        if not metrics:
            print("\n  No metrics found. Is training running?")
            return

        # If we have parsed training metrics
        if "iterations" in metrics and metrics["iterations"]:
            iters = metrics["iterations"]
            losses = metrics["losses"]

            # Color code the metrics
            print(
                f"\n  {Theme.text}Latest Iteration: {Theme.success}{iters[-1]}{Term.normal}"
            )

            # Color code loss based on value
            loss_color = (
                Theme.success
                if losses[-1] < 2
                else Theme.warning if losses[-1] < 4 else Theme.error
            )
            print(
                f"  {Theme.text}Latest Loss: {loss_color}{losses[-1]:.4f}{Term.normal}"
            )

            if len(losses) > 1:
                change = losses[-1] - losses[0]
                change_color = Theme.success if change < 0 else Theme.error
                print(
                    f"  {Theme.text}Loss Change: {change_color}{change:+.4f}{Term.normal}"
                )

            # ASCII graph of loss - use available terminal width
            graph_width = min(Term.width - 10, 80)
            graph_height = min(Term.height - 20, 12)

            if len(losses) > 5:
                print(
                    "\n"
                    + self.create_ascii_graph(
                        losses[-graph_width:] if len(losses) > graph_width else losses,
                        width=graph_width,
                        height=graph_height,
                        title="Loss Trend",
                    )
                )

            # Show learning rate if available with color
            if "learning_rates" in metrics and metrics["learning_rates"]:
                lr = metrics["learning_rates"][-1]
                lr_color = Color.gradient(lr / 1e-2, Theme.lr_gradient)
                print(f"\n  {Theme.text}Learning Rate: {lr_color}{lr:.2e}{Term.normal}")

            # Show sparsity if available with gradient color
            if "sparsities" in metrics and metrics["sparsities"]:
                sparsity = metrics["sparsities"][-1]
                sparsity_color = Color.gradient(sparsity / 100, Theme.sparsity_gradient)
                print(
                    f"  {Theme.text}Sparsity: {sparsity_color}{sparsity:.1f}%{Term.normal}"
                )

            # Estimate time remaining
            if "times" in metrics and metrics["times"] and len(metrics["times"]) > 1:
                avg_time = sum(metrics["times"][-10:]) / len(metrics["times"][-10:])
                if iters[-1] < 10000:  # Assume 10000 iterations total
                    remaining = (10000 - iters[-1]) * avg_time / 1000 / 60
                    print(f"\n  Estimated time remaining: {remaining:.1f} minutes")

    def display_advanced(self, metrics: Dict[str, Any], iteration: int):
        """Display with advanced terminal UI features."""
        print(Term.clear)

        # Draw border box that adapts to terminal size
        self._draw_box(
            1, 1, Term.width - 2, Term.height - 2, f"TrackIO Dashboard - {self.project}"
        )

        if not metrics:
            self._center_text(
                Term.height // 2,
                "No metrics found. Is training running?",
                Theme.warning,
            )
            return

        # If we have data from database
        if "data" in metrics and metrics["data"]:
            # Convert database format to display format
            display_metrics = self._convert_db_to_display_format(metrics["data"])
            self._display_training_metrics(display_metrics)
        # If we have parsed training metrics
        elif "iterations" in metrics and metrics["iterations"]:
            self._display_training_metrics(metrics)

    def _convert_db_to_display_format(self, data: list) -> Dict[str, Any]:
        """Convert database format to display format."""
        # Data comes in reverse chronological order, so reverse it for proper display
        data_reversed = list(reversed(data))

        # Extract iterations, losses, and other metrics
        iterations = []
        losses = []
        learning_rates = []
        sparsities = []

        for entry in data_reversed:
            if "iteration" in entry:
                iterations.append(entry["iteration"])
                # Only append to other lists if we have the corresponding data
                if "train_loss" in entry:
                    losses.append(entry["train_loss"])
                if "learning_rate" in entry:
                    learning_rates.append(entry["learning_rate"])
                if "sparsity" in entry:
                    sparsities.append(entry["sparsity"])

        # Ensure we have data to display
        if not iterations or not losses:
            return {}

        # Return in expected format
        result = {
            "iterations": iterations,
            "losses": losses,
        }

        if learning_rates:
            result["learning_rates"] = learning_rates
        if sparsities:
            result["sparsities"] = sparsities

        # Add latest values for display (from original data[0] which is most recent)
        if data:
            latest = data[0]  # Most recent entry
            result["current_iter"] = latest.get("iteration", 0)
            result["current_loss"] = latest.get("train_loss", 0)
            result["current_lr"] = latest.get("learning_rate", 0)
            result["current_sparsity"] = latest.get("sparsity", 0)

        return result

    def _draw_box(self, x: int, y: int, w: int, h: int, title: str = ""):
        """Draw a box with optional title."""
        print(f"\033[{y};{x}f{Theme.border}┌", end="")
        if title:
            title_str = f"─┤ {Theme.title}{title}{Theme.border} ├"
            print(title_str, end="")
            remaining = w - len(title) - 6
            print("─" * remaining, end="")
        else:
            print("─" * (w - 2), end="")
        print("┐")

        for i in range(1, h - 1):
            print(f"\033[{y + i};{x}f│", end="")
            print(f"\033[{y + i};{x + w - 1}f│")

        print(f"\033[{y + h - 1};{x}f└" + "─" * (w - 2) + "┘" + Term.normal)

    def _center_text(self, y: int, text: str, color: str = ""):
        """Center text on a line."""
        x = (Term.width - len(text)) // 2
        print(f"\033[{y};{x}f{color}{text}{Term.normal}")

    def _display_training_metrics(self, metrics: Dict[str, Any]):
        """Display training metrics with graphs."""
        if not metrics or "iterations" not in metrics or "losses" not in metrics:
            self._center_text(
                Term.height // 2,
                "Waiting for training data...",
                Theme.warning,
            )
            return

        iters = metrics["iterations"]
        losses = metrics["losses"]

        # Stats panel - use left side of screen
        stats_y = 3
        stats_x = 3
        stats_width = min(30, Term.width // 3)

        print(f"\033[{stats_y};{stats_x}f{Theme.text}Current Statistics:{Term.normal}")
        stats_y += 2

        # Iteration
        print(
            f"\033[{stats_y};{stats_x}f{Theme.text}Iteration: {Theme.success}{iters[-1]}{Term.normal}"
        )
        stats_y += 1

        # Loss with color coding
        loss_color = (
            Theme.success
            if losses[-1] < 2
            else Theme.warning if losses[-1] < 4 else Theme.error
        )
        print(
            f"\033[{stats_y};{stats_x}f{Theme.text}Loss: {loss_color}{losses[-1]:.4f}{Term.normal}"
        )
        stats_y += 1

        # Learning rate if available
        if "learning_rates" in metrics and metrics["learning_rates"]:
            lr = metrics["learning_rates"][-1]
            lr_color = Color.gradient(lr / 1e-2, Theme.lr_gradient)
            print(
                f"\033[{stats_y};{stats_x}f{Theme.text}LR: {lr_color}{lr:.2e}{Term.normal}"
            )
            stats_y += 1

        # Sparsity if available
        if "sparsities" in metrics and metrics["sparsities"]:
            sparsity = metrics["sparsities"][-1]
            sparsity_color = Color.gradient(sparsity / 100, Theme.sparsity_gradient)
            print(
                f"\033[{stats_y};{stats_x}f{Theme.text}Sparsity: {sparsity_color}{sparsity:.1%}{Term.normal}"
            )
            stats_y += 1

        # Graph panel - use remaining space
        graph_x = stats_x + stats_width + 5
        graph_y = 3
        graph_width = Term.width - graph_x - 3
        graph_height = min(Term.height - 10, 15)

        # Create and display loss graph
        if len(losses) > 2 and graph_width > 20:
            graph = Graph(graph_width, graph_height, min(losses), max(losses))

            # Add recent values
            recent_losses = (
                losses[-graph_width:] if len(losses) > graph_width else losses
            )
            for loss in recent_losses:
                graph.add_value(loss)

            # Draw graph
            graph_lines = graph.draw(Theme.loss_gradient, show_values=True)

            # Title
            print(f"\033[{graph_y};{graph_x}f{Theme.title}Loss Trend{Term.normal}")

            # Display graph lines
            for i, line in enumerate(graph_lines):
                print(f"\033[{graph_y + i + 2};{graph_x}f{line}")

    def run(self, interval: float = 1.0, log_path: Path = None):
        """Run the dashboard with automatic updates."""

        # Setup signal handlers
        def signal_handler(sig, frame):
            print(f"{Term.normal_screen}{Term.show_cursor}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Handle terminal resize (if supported)
        try:

            def resize_handler(sig, frame):
                Term.resized = True

            signal.signal(signal.SIGWINCH, resize_handler)
        except (AttributeError, ValueError):
            # SIGWINCH not available on this platform
            pass

        try:
            iteration = 0
            while True:
                Term.refresh()

                if log_path and log_path.exists():
                    metrics = self.parse_training_log(log_path)
                else:
                    metrics = self.find_latest_metrics()

                if RICH_AVAILABLE and not self.use_advanced_ui:
                    self.display_rich(metrics)
                else:
                    self.display_advanced(metrics, iteration)

                time.sleep(interval)
                iteration += 1

        except KeyboardInterrupt:
            print(f"{Term.normal_screen}{Term.show_cursor}")
            print("\n\nDashboard closed.")
            sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TrackIO Console Dashboard")
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        help="Project name to track (auto-detected if not provided)",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        help="Path to output.log file to parse instead of database",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--simple",
        "-s",
        action="store_true",
        help="Use simple UI (no rich library)",
    )

    args = parser.parse_args()

    # Create console
    console = TrackIOConsole(project=args.project)

    # Set UI mode
    if args.simple:
        console.use_advanced_ui = False

    # Run dashboard
    log_path = Path(args.log) if args.log else None

    # Enter alternate screen
    print(Term.alt_screen, end="")

    try:
        console.run(interval=args.interval, log_path=log_path)
    finally:
        # Always restore normal screen on exit
        print(Term.normal_screen + Term.show_cursor, end="")


if __name__ == "__main__":
    main()
