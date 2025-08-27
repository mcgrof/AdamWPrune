#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
GPU monitoring integration using gputop.py for training metrics and AB testing.
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add scripts directory to path for gputop import
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

try:
    from gputop import GPUMonitor, GPUTop

    GPUTOP_AVAILABLE = True
except ImportError:
    GPUTOP_AVAILABLE = False

# Always import simple monitor as fallback
try:
    from .gpu_monitor_simple import SimpleGPUMonitorContext

    SIMPLE_MONITOR_AVAILABLE = True
except ImportError:
    try:
        from gpu_monitor_simple import SimpleGPUMonitorContext

        SIMPLE_MONITOR_AVAILABLE = True
    except ImportError:
        SIMPLE_MONITOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingGPUMonitor:
    """
    GPU monitoring wrapper for training sessions with AB testing support.
    """

    def __init__(self, config_name: str = None, output_dir: str = "results"):
        self.config_name = config_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Generate unique filename based on config and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config_name:
            self.stats_file = (
                self.output_dir / f"gpu_stats_{config_name}_{timestamp}.json"
            )
        else:
            self.stats_file = self.output_dir / f"gpu_stats_{timestamp}.json"

        self.monitor = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.training_start_time = None
        self.training_end_time = None

        if not GPUTOP_AVAILABLE:
            logger.warning("gputop module not available - GPU monitoring disabled")

    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        if not GPUTOP_AVAILABLE:
            logger.info("GPU monitoring not available - skipping")
            return

        try:
            self.monitor = GPUTop(stats_file=str(self.stats_file))
            self.training_start_time = time.time()

            # Start monitoring in background thread
            self.monitoring_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitoring_thread.start()

            logger.info(
                f"Started GPU monitoring - stats will be saved to {self.stats_file}"
            )

        except Exception as e:
            logger.error(f"Failed to start GPU monitoring: {e}")
            self.monitor = None

    def _monitor_loop(self):
        """Background monitoring loop."""
        if not self.monitor:
            return

        try:
            while not self.stop_event.is_set():
                stats = self.monitor.gpu_monitor.get_stats()
                if stats:
                    # Add training context to stats
                    stats["training_active"] = True
                    stats["training_elapsed"] = (
                        time.time() - self.training_start_time
                        if self.training_start_time
                        else 0
                    )

                    self.monitor.log_stats_to_json(stats)

                # Sleep for 1 second between measurements
                time.sleep(1)

        except Exception as e:
            logger.error(f"GPU monitoring error: {e}")

    def stop_monitoring(self):
        """Stop GPU monitoring and save final stats."""
        if not self.monitor:
            return

        self.training_end_time = time.time()
        self.stop_event.set()

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        # Save final summary
        if self.monitor and hasattr(self.monitor, "save_summary_stats"):
            try:
                self.monitor.save_summary_stats()
                logger.info(f"GPU monitoring summary saved alongside {self.stats_file}")
            except Exception as e:
                logger.error(f"Failed to save GPU monitoring summary: {e}")

        logger.info("GPU monitoring stopped")

    def get_stats_file(self) -> Optional[Path]:
        """Get the path to the GPU stats JSON file."""
        if self.stats_file and self.stats_file.exists():
            return self.stats_file
        return None

    def generate_graphs(self) -> Optional[Path]:
        """Generate graphs from GPU stats data."""
        if not GPUTOP_AVAILABLE or not self.stats_file or not self.stats_file.exists():
            return None

        try:
            # Use gputop to generate graphs
            graph_file = str(self.stats_file).replace(".json", "_plot.png")
            cmd = [
                sys.executable,
                str(scripts_dir / "gputop.py"),
                "--gen-graph",
                str(self.stats_file),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"GPU performance graphs saved to {graph_file}")
                return Path(graph_file)
            else:
                logger.error(f"Failed to generate graphs: {result.stderr}")

        except Exception as e:
            logger.error(f"Error generating GPU graphs: {e}")

        return None

    @staticmethod
    def compare_runs(
        stats_file1: Path, stats_file2: Path, label1: str = None, label2: str = None
    ):
        """
        Compare two GPU monitoring runs using gputop's built-in comparison.

        Args:
            stats_file1: First GPU stats JSON file
            stats_file2: Second GPU stats JSON file
            label1: Label for first run (optional)
            label2: Label for second run (optional)
        """
        if not GPUTOP_AVAILABLE:
            logger.warning("gputop not available - cannot compare runs")
            return

        if not stats_file1.exists() or not stats_file2.exists():
            logger.error("One or both stats files do not exist")
            return

        try:
            # Use gputop's comparison feature
            cmd = [
                sys.executable,
                str(scripts_dir / "gputop.py"),
                "--compare",
                str(stats_file1),
                str(stats_file2),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                logger.info("GPU runs comparison completed")
            else:
                logger.error(f"Comparison failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Error comparing GPU runs: {e}")

    def add_training_metadata(self, metadata: Dict[str, Any]):
        """
        Add training-specific metadata to the GPU stats file.
        This helps with AB testing analysis.
        """
        if not self.stats_file or not self.stats_file.exists():
            return

        try:
            # Load existing stats
            with open(self.stats_file, "r") as f:
                stats_data = json.load(f)

            # Add metadata to the first entry if it exists
            if stats_data and len(stats_data) > 0:
                stats_data[0]["training_metadata"] = metadata

                # Save back
                with open(self.stats_file, "w") as f:
                    json.dump(stats_data, f, indent=2)

                logger.info("Training metadata added to GPU stats")

        except Exception as e:
            logger.error(f"Failed to add training metadata: {e}")


def create_training_monitor(
    config_name: str = None, output_dir: str = "results"
) -> TrainingGPUMonitor:
    """
    Factory function to create a training GPU monitor.

    Args:
        config_name: Configuration name for file naming
        output_dir: Directory to save GPU stats

    Returns:
        TrainingGPUMonitor instance
    """
    return TrainingGPUMonitor(config_name=config_name, output_dir=output_dir)


# Context manager for easy use
class GPUMonitoringContext:
    """Context manager for GPU monitoring during training."""

    def __init__(
        self,
        config_name: str = None,
        output_dir: str = "results",
        training_metadata: Dict[str, Any] = None,
    ):
        self.config_name = config_name
        self.output_dir = Path(output_dir)
        self.training_metadata = training_metadata or {}

        # Generate stats file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config_name:
            self.stats_file = (
                self.output_dir / f"gpu_stats_{config_name}_{timestamp}.json"
            )
        else:
            self.stats_file = self.output_dir / f"gpu_stats_{timestamp}.json"

        self.use_simple_monitor = True  # Prefer simple monitor for training
        self.monitor = None

    def __enter__(self):
        """Start GPU monitoring."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_simple_monitor and SIMPLE_MONITOR_AVAILABLE:
            # Use simple monitor (quieter, focused on data collection)
            self.simple_context = SimpleGPUMonitorContext(
                str(self.stats_file), self.training_metadata
            )
            logger.info(
                f"Started simple GPU monitoring - stats will be saved to {self.stats_file}"
            )
            return self.simple_context.__enter__()

        elif GPUTOP_AVAILABLE:
            # Fall back to full gputop monitor
            self.monitor = create_training_monitor(
                self.config_name, str(self.output_dir)
            )
            self.monitor.start_monitoring()
            logger.info(
                f"Started full GPU monitoring - stats will be saved to {self.monitor.stats_file}"
            )
            return self.monitor

        else:
            logger.warning(
                "No GPU monitoring available - training will proceed without monitoring"
            )
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop GPU monitoring and generate final outputs."""
        if hasattr(self, "simple_context") and self.simple_context:
            # Simple monitor handles everything in its context manager
            self.simple_context.__exit__(exc_type, exc_val, exc_tb)
            logger.info("Simple GPU monitoring completed")

            # Generate graphs using gputop if available
            if GPUTOP_AVAILABLE and self.stats_file.exists():
                try:
                    self._generate_graphs_from_stats()
                except Exception as e:
                    logger.warning(f"Failed to generate graphs: {e}")

        elif self.monitor:
            # Full monitor cleanup
            self.monitor.stop_monitoring()

            if self.training_metadata:
                self.monitor.add_training_metadata(self.training_metadata)

            self.monitor.generate_graphs()

    def _generate_graphs_from_stats(self):
        """Generate graphs using gputop from saved stats."""
        if not self.stats_file.exists():
            return

        try:
            cmd = [
                sys.executable,
                str(scripts_dir / "gputop.py"),
                "--gen-graph",
                str(self.stats_file),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                graph_file = str(self.stats_file).replace(".json", "_plot.png")
                logger.info(f"GPU performance graphs generated: {graph_file}")
            else:
                logger.warning(f"Graph generation failed: {result.stderr}")

        except Exception as e:
            logger.warning(f"Error generating graphs: {e}")


if __name__ == "__main__":
    # Test the monitoring system
    print("Testing GPU monitoring integration...")

    with GPUMonitoringContext("test_config") as monitor:
        print("Monitoring started - sleeping for 5 seconds...")
        time.sleep(5)

    print("Monitoring completed")

    stats_file = monitor.get_stats_file()
    if stats_file:
        print(f"Stats saved to: {stats_file}")
    else:
        print("No stats file generated")
