#!/usr/bin/env python3
"""
TrackIO Console Dashboard - A terminal-based metrics viewer for TrackIO projects.

This provides a console UI for viewing TrackIO metrics without needing a web browser.
Could be contributed back to the TrackIO project.
"""

import argparse
import json
import os
import sys
import time
import signal
import fcntl
import termios
import tty
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from select import select
import sqlite3

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
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
        (0, 100, 200),   # Blue (low loss)
        (0, 200, 100),   # Green
        (200, 200, 0),   # Yellow
        (255, 100, 0),   # Orange
        (255, 0, 0),     # Red (high loss)
    ]

    # Learning rate gradient
    lr_gradient = [
        (100, 100, 255), # Light blue
        (200, 100, 255), # Purple
        (255, 100, 200), # Pink
    ]

    # Sparsity gradient (red -> yellow -> green)
    sparsity_gradient = [
        (255, 0, 0),     # Red (0% sparse)
        (255, 200, 0),   # Orange
        (255, 255, 0),   # Yellow
        (100, 255, 0),   # Light green
        (0, 255, 0),     # Green (100% sparse)
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

    def __init__(self, width: int, height: int, min_value: float = 0, max_value: float = 100):
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

    def draw(self, gradient: List[Tuple[int, int, int]], show_values: bool = True) -> List[str]:
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
                matrix[row][col] = "●" if norm_value > 0.8 else "○" if norm_value > 0.5 else "·"

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


class MultiLineGraph:
    """Graph supporting multiple overlaid metrics"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.metrics = {}  # Dict of metric_name: Graph
        self.colors = {}

    def add_metric(self, name: str, color: Tuple[int, int, int], min_val: float = 0, max_val: float = 100):
        """Add a metric to track"""
        self.metrics[name] = Graph(self.width, self.height, min_val, max_val)
        self.colors[name] = color

    def add_value(self, name: str, value: float):
        """Add value to specific metric"""
        if name in self.metrics:
            self.metrics[name].add_value(value)

    def draw(self) -> List[str]:
        """Draw all metrics overlaid"""
        lines = [" " * self.width for _ in range(self.height)]

        # Draw each metric with its own color
        for name, graph in self.metrics.items():
            if not graph.data:
                continue

            color = self.colors[name]
            color_str = Color.fg(*color)

            data_list = list(graph.data)
            while len(data_list) < self.width:
                data_list.insert(0, graph.min_value)

            range_val = graph.max_value - graph.min_value
            if range_val == 0:
                range_val = 1

            for col, value in enumerate(data_list):
                norm_value = (value - graph.min_value) / range_val
                row = int((1 - norm_value) * (self.height - 1))

                if 0 <= row < self.height and col < self.width:
                    # Only draw if position is empty or we're overlaying
                    char = "·" if len(self.metrics) > 1 else "●"
                    line_list = list(lines[row])
                    line_list[col] = char
                    lines[row] = "".join(line_list)

        # Add colors to final output
        colored_lines = []
        for line in lines:
            colored_line = ""
            for char in line:
                if char != " ":
                    # Use first metric's color for now
                    first_color = next(iter(self.colors.values()))
                    colored_line += Color.fg(*first_color) + char + Term.normal
                else:
                    colored_line += char
            colored_lines.append(colored_line)

        return colored_lines


class TrackIOConsole:
    """Console viewer for TrackIO metrics."""

    def __init__(self, project: str):
        self.project = project
        self.console = Console() if RICH_AVAILABLE else None
        self.graphs = {}
        self.multi_graph = None
        self.use_advanced_ui = True  # Use advanced terminal UI by default

        # Find TrackIO data location
        self.trackio_dirs = [
            Path.home() / '.trackio' / project,
            Path.home() / '.cache' / 'trackio' / project,
            Path.home() / '.cache' / 'huggingface' / 'trackio' / project,
            Path.cwd() / '.trackio' / project,
            Path.cwd() / f'trackio_{project}.db',
        ]

        self.data_dir = None
        self.db_path = None

        for path in self.trackio_dirs:
            if path.exists():
                if path.suffix == '.db':
                    self.db_path = path
                else:
                    self.data_dir = path
                break

    def find_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Find and load the latest metrics from TrackIO storage."""
        if self.db_path and self.db_path.exists():
            return self._read_from_db()
        elif self.data_dir and self.data_dir.exists():
            return self._read_from_json()
        return None

    def _read_from_db(self) -> Optional[Dict[str, Any]]:
        """Read metrics from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Try to find metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]

            metrics = {}
            if 'metrics' in tables:
                cursor.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 100")
                rows = cursor.fetchall()
                if rows:
                    # Get column names
                    columns = [d[0] for d in cursor.description]
                    metrics['data'] = [dict(zip(columns, row)) for row in rows]

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
        metrics_file = latest_run / 'metrics.json'

        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                metrics['run_name'] = latest_run.name
                metrics['data'] = data

        return metrics

    def parse_training_log(self, log_path: Path) -> Dict[str, Any]:
        """Parse a training output.log file for metrics."""
        metrics = {
            'iterations': [],
            'losses': [],
            'perplexities': [],
            'learning_rates': [],
            'sparsities': [],
            'times': []
        }

        if not log_path.exists():
            return metrics

        with open(log_path) as f:
            for line in f:
                if 'Iter' in line and 'loss' in line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        try:
                            metrics['iterations'].append(int(parts[0].split()[-1]))
                            metrics['losses'].append(float(parts[1].split()[-1]))
                            metrics['perplexities'].append(float(parts[2].split()[-1]))

                            lr_str = parts[3].split()[-1]
                            # Convert scientific notation
                            if 'e' in lr_str:
                                metrics['learning_rates'].append(float(lr_str))
                            else:
                                metrics['learning_rates'].append(float(lr_str))

                            sparsity_str = parts[4].split()[1].rstrip('%')
                            metrics['sparsities'].append(float(sparsity_str))

                            # Extract time if present
                            if 'ms/iter' in parts[-1]:
                                time_str = parts[-1].split()[-1].replace('ms/iter', '')
                                metrics['times'].append(float(time_str))
                        except (ValueError, IndexError):
                            continue

        return metrics

    def create_ascii_graph(self, values: List[float], width: int = 50, height: int = 10,
                           title: str = "") -> str:
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
        print(f"{Theme.border}" + "═"*60 + f"{Term.normal}")
        print(f"  {Theme.title}TrackIO Console Dashboard - Project: {self.project}{Term.normal}")
        print(f"{Theme.border}" + "═"*60 + f"{Term.normal}")

        if not metrics:
            print("\n  No metrics found. Is training running?")
            return

        # If we have parsed training metrics
        if 'iterations' in metrics and metrics['iterations']:
            iters = metrics['iterations']
            losses = metrics['losses']

            # Color code the metrics
            print(f"\n  {Theme.text}Latest Iteration: {Theme.success}{iters[-1]}{Term.normal}")

            # Color code loss based on value
            loss_color = Theme.success if losses[-1] < 2 else Theme.warning if losses[-1] < 4 else Theme.error
            print(f"  {Theme.text}Latest Loss: {loss_color}{losses[-1]:.4f}{Term.normal}")

            if len(losses) > 1:
                change = losses[-1] - losses[0]
                change_color = Theme.success if change < 0 else Theme.error
                print(f"  {Theme.text}Loss Change: {change_color}{change:+.4f}{Term.normal}")

            # ASCII graph of loss
            if len(losses) > 5:
                print("\n" + self.create_ascii_graph(
                    losses[-50:] if len(losses) > 50 else losses,
                    width=50, height=8, title="Loss Trend"
                ))

            # Show learning rate if available with color
            if 'learning_rates' in metrics and metrics['learning_rates']:
                lr = metrics['learning_rates'][-1]
                lr_color = Color.gradient(lr / 1e-2, Theme.lr_gradient)
                print(f"\n  {Theme.text}Learning Rate: {lr_color}{lr:.2e}{Term.normal}")

            # Show sparsity if available with gradient color
            if 'sparsities' in metrics and metrics['sparsities']:
                sparsity = metrics['sparsities'][-1]
                sparsity_color = Color.gradient(sparsity / 100, Theme.sparsity_gradient)
                print(f"  {Theme.text}Sparsity: {sparsity_color}{sparsity:.1f}%{Term.normal}")

            # Estimate time remaining
            if 'times' in metrics and metrics['times'] and len(metrics['times']) > 1:
                avg_time = sum(metrics['times'][-10:]) / len(metrics['times'][-10:])
                if iters[-1] < 500:  # Assume 500 iterations total
                    remaining = (500 - iters[-1]) * avg_time / 1000 / 60
                    print(f"\n  Estimated time remaining: {remaining:.1f} minutes")

    def display_rich(self, metrics: Dict[str, Any]):
        """Display metrics with rich formatting."""
        if not RICH_AVAILABLE:
            return self.display_simple(metrics)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Header
        header_text = Text(f"TrackIO Console Dashboard - {self.project}", justify="center")
        header_text.stylize("bold magenta")
        layout["header"].update(Panel(header_text))

        # Body content
        if not metrics:
            layout["body"].update(Panel("No metrics found. Is training running?"))
        else:
            body_layout = Layout()
            body_layout.split_row(
                Layout(name="metrics", ratio=1),
                Layout(name="graph", ratio=2)
            )

            # Metrics table
            table = Table(box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            if 'iterations' in metrics and metrics['iterations']:
                table.add_row("Iteration", str(metrics['iterations'][-1]))
                table.add_row("Loss", f"{metrics['losses'][-1]:.4f}")

                if 'perplexities' in metrics:
                    table.add_row("Perplexity", f"{metrics['perplexities'][-1]:.1f}")
                if 'learning_rates' in metrics:
                    table.add_row("Learning Rate", f"{metrics['learning_rates'][-1]:.2e}")
                if 'sparsities' in metrics:
                    table.add_row("Sparsity", f"{metrics['sparsities'][-1]:.1f}%")

            body_layout["metrics"].update(Panel(table, title="Current Metrics"))

            # Graph
            if 'losses' in metrics and len(metrics['losses']) > 1:
                graph_text = self.create_ascii_graph(
                    metrics['losses'][-50:] if len(metrics['losses']) > 50 else metrics['losses'],
                    width=40, height=10, title="Loss"
                )
                body_layout["graph"].update(Panel(graph_text, title="Training Progress"))

            layout["body"].update(body_layout)

        # Footer
        layout["footer"].update(Panel(
            Text("Press Ctrl+C to exit | Updates every 2 seconds", justify="center")
        ))

        self.console.print(layout)

    def monitor_live(self, log_path: Optional[Path] = None, interval: int = 2):
        """Monitor metrics live with updates."""
        # Set up terminal
        print(f"{Term.alt_screen}{Term.hide_cursor}")

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print(f"{Term.normal_screen}{Term.show_cursor}")
            print("\n\nDashboard closed.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

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

    def display_advanced(self, metrics: Dict[str, Any], iteration: int):
        """Display with advanced terminal UI features."""
        print(Term.clear)

        # Draw border box
        self._draw_box(1, 1, Term.width - 2, Term.height - 2,
                      f"TrackIO Dashboard - {self.project}")

        if not metrics:
            self._center_text(Term.height // 2, "No metrics found. Is training running?", Theme.warning)
            return

        # If we have parsed training metrics
        if 'iterations' in metrics and metrics['iterations']:
            self._display_training_metrics(metrics)

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

        print(f"\033[{y + h - 1};{x}f└" + "─" * (w - 2) + "┘{Term.normal}")

    def _center_text(self, y: int, text: str, color: str = ""):
        """Center text on a line."""
        x = (Term.width - len(text)) // 2
        print(f"\033[{y};{x}f{color}{text}{Term.normal}")

    def _display_training_metrics(self, metrics: Dict[str, Any]):
        """Display training metrics with graphs."""
        iters = metrics['iterations']
        losses = metrics['losses']

        # Stats panel
        stats_y = 3
        stats_x = 3

        print(f"\033[{stats_y};{stats_x}f{Theme.text}Current Statistics:{Term.normal}")
        stats_y += 2

        # Iteration
        print(f"\033[{stats_y};{stats_x}f{Theme.text}Iteration: {Theme.success}{iters[-1]}{Term.normal}")
        stats_y += 1

        # Loss with color coding
        loss_color = Theme.success if losses[-1] < 2 else Theme.warning if losses[-1] < 4 else Theme.error
        print(f"\033[{stats_y};{stats_x}f{Theme.text}Loss: {loss_color}{losses[-1]:.4f}{Term.normal}")
        stats_y += 1

        # Loss change
        if len(losses) > 1:
            change = losses[-1] - losses[0]
            change_color = Theme.success if change < 0 else Theme.error
            print(f"\033[{stats_y};{stats_x}f{Theme.text}Change: {change_color}{change:+.4f}{Term.normal}")
            stats_y += 1

        # Additional metrics
        if 'perplexities' in metrics and metrics['perplexities']:
            ppl = metrics['perplexities'][-1]
            ppl_color = Theme.success if ppl < 100 else Theme.warning if ppl < 200 else Theme.error
            print(f"\033[{stats_y};{stats_x}f{Theme.text}Perplexity: {ppl_color}{ppl:.1f}{Term.normal}")
            stats_y += 1

        if 'learning_rates' in metrics and metrics['learning_rates']:
            lr = metrics['learning_rates'][-1]
            print(f"\033[{stats_y};{stats_x}f{Theme.text}LR: {Theme.warning}{lr:.2e}{Term.normal}")
            stats_y += 1

        if 'sparsities' in metrics and metrics['sparsities']:
            sparsity = metrics['sparsities'][-1]
            sparsity_color = Color.gradient(sparsity / 100, Theme.sparsity_gradient)
            print(f"\033[{stats_y};{stats_x}f{Theme.text}Sparsity: {sparsity_color}{sparsity:.1f}%{Term.normal}")
            stats_y += 1

        # Graph panel - Loss over time
        graph_width = min(60, Term.width - 10)
        graph_height = min(15, Term.height - stats_y - 5)
        graph_x = Term.width - graph_width - 5
        graph_y = 3

        if len(losses) > 1:
            # Create and draw loss graph
            loss_graph = Graph(graph_width, graph_height, min(losses), max(losses))
            for loss in losses[-graph_width:]:
                loss_graph.add_value(loss)

            graph_lines = loss_graph.draw(Theme.loss_gradient, show_values=True)

            print(f"\033[{graph_y};{graph_x}f{Theme.text}Loss Trend:{Term.normal}")
            for i, line in enumerate(graph_lines):
                print(f"\033[{graph_y + i + 1};{graph_x - 8}f{line}")

        # Time estimate
        if 'times' in metrics and metrics['times'] and len(metrics['times']) > 1:
            avg_time = sum(metrics['times'][-10:]) / len(metrics['times'][-10:])
            if iters[-1] < 500:  # Assume 500 iterations total
                remaining = (500 - iters[-1]) * avg_time / 1000 / 60
                time_y = Term.height - 3
                print(f"\033[{time_y};{stats_x}f{Theme.text}Est. remaining: {Theme.warning}{remaining:.1f} min{Term.normal}")


def main():
    parser = argparse.ArgumentParser(description="TrackIO Console Dashboard")
    parser.add_argument(
        "--project", "-p",
        help="TrackIO project name",
        default=None
    )
    parser.add_argument(
        "--log", "-l",
        help="Path to training output.log file",
        default=None
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=2,
        help="Update interval in seconds (default: 2)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Display once and exit (no live monitoring)"
    )

    args = parser.parse_args()

    # Try to auto-detect project from .config if not specified
    if not args.project and not args.log:
        config_file = Path(".config")
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    if "CONFIG_TRACKER_PROJECT" in line:
                        args.project = line.split('"')[1]
                        break

    # Try to find latest log file if not specified
    if not args.log and not args.project:
        # Look for latest test_matrix_results
        test_dirs = sorted(Path(".").glob("test_matrix_results_*"))
        if test_dirs:
            latest_test = test_dirs[-1]
            log_files = list(latest_test.glob("*/output.log"))
            if log_files:
                args.log = log_files[0]

    if not args.project and not args.log:
        print("Error: Please specify --project or --log")
        print("\nExamples:")
        print("  trackio-console --project tracking-11f50")
        print("  trackio-console --log test_matrix_results_*/gpt2_*/output.log")
        print("  trackio-console  # Auto-detect from .config")
        sys.exit(1)

    # Create dashboard
    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            print(f"Error: Log file not found: {log_path}")
            sys.exit(1)

        dashboard = TrackIOConsole(args.project or "training")
        if args.once:
            metrics = dashboard.parse_training_log(log_path)
            dashboard.display_simple(metrics)
        else:
            dashboard.monitor_live(log_path, args.interval)
    else:
        dashboard = TrackIOConsole(args.project)
        if args.once:
            metrics = dashboard.find_latest_metrics()
            dashboard.display_simple(metrics)
        else:
            dashboard.monitor_live(None, args.interval)


if __name__ == "__main__":
    main()