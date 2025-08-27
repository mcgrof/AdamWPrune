#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Simple GPU monitoring script for inference testing.
Uses nvidia-smi to capture memory usage during inference.
"""

import subprocess
import json
import time
import sys
import signal
import threading
from datetime import datetime
from pathlib import Path


class GPUMonitor:
    def __init__(self, output_file, interval_ms=100, model_name="inference"):
        self.output_file = Path(output_file)
        self.interval = interval_ms / 1000.0  # Convert to seconds
        self.model_name = model_name
        self.stop_flag = threading.Event()
        self.thread = None
        self.data = []

    def _monitor_loop(self):
        """Monitor GPU memory in a loop."""
        start_time = time.time()

        while not self.stop_flag.is_set():
            try:
                # First try rocm-smi for AMD GPUs
                if (
                    subprocess.run(
                        ["which", "rocm-smi"], capture_output=True
                    ).returncode
                    == 0
                ):
                    # AMD GPU monitoring with rocm-smi
                    result = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram", "--json"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    # Parse JSON output
                    import json as js

                    data = js.loads(result.stdout)

                    # Extract memory info for each GPU
                    for gpu_id_str, gpu_info in data.items():
                        if gpu_id_str.startswith("card"):
                            gpu_id = int(gpu_id_str.replace("card", ""))
                            memory_used = float(
                                gpu_info.get("VRAM Total Used Memory (B)", 0)
                            ) / (1024 * 1024)
                            memory_total = float(
                                gpu_info.get("VRAM Total Memory (B)", 0)
                            ) / (1024 * 1024)

                            # Get GPU utilization (rocm-smi doesn't provide this easily)
                            gpu_util = 0  # Placeholder

                            # Record data point
                            self.data.append(
                                {
                                    "timestamp": time.time() - start_time,
                                    "gpu_id": gpu_id,
                                    "memory_mb": memory_used,
                                    "memory_total_mb": memory_total,
                                    "gpu_utilization": gpu_util,
                                    "model": self.model_name,
                                }
                            )
                else:
                    # NVIDIA GPU monitoring with nvidia-smi
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    # Parse output
                    lines = result.stdout.strip().split("\n")
                    for gpu_id, line in enumerate(lines):
                        parts = line.split(", ")
                        if len(parts) >= 3:
                            memory_used = float(parts[0])
                            memory_total = float(parts[1])
                            gpu_util = float(parts[2])

                            # Record data point
                            self.data.append(
                                {
                                    "timestamp": time.time() - start_time,
                                    "gpu_id": gpu_id,
                                    "memory_mb": memory_used,
                                    "memory_total_mb": memory_total,
                                    "gpu_utilization": gpu_util,
                                    "model": self.model_name,
                                }
                            )

            except (subprocess.CalledProcessError, ValueError) as e:
                # Silently continue on error
                pass

            # Sleep for interval
            time.sleep(self.interval)

    def start(self):
        """Start monitoring in background thread."""
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop monitoring and save data."""
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1)

        # Save data to JSON
        if self.data:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w") as f:
                json.dump(
                    {
                        "model": self.model_name,
                        "samples": self.data,
                        "interval_ms": int(self.interval * 1000),
                        "total_samples": len(self.data),
                    },
                    f,
                    indent=2,
                )

            print(
                f"Saved {len(self.data)} GPU monitoring samples to {self.output_file}"
            )

        return len(self.data)


def main():
    """Command-line interface for GPU monitoring."""
    if len(sys.argv) < 2:
        print("Usage: python monitor_gpu.py <output_file> [interval_ms] [model_name]")
        sys.exit(1)

    output_file = sys.argv[1]
    interval_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    model_name = sys.argv[3] if len(sys.argv) > 3 else "inference"

    monitor = GPUMonitor(output_file, interval_ms, model_name)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping GPU monitor...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Starting GPU monitor (output: {output_file}, interval: {interval_ms}ms)")
    monitor.start()

    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    monitor.stop()


if __name__ == "__main__":
    main()
