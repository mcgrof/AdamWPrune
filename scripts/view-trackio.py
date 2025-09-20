#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Helper script to view Trackio dashboard when trackio command is not in PATH.
"""

import sys
import argparse

try:
    import trackio
except ImportError:
    print("Error: trackio not installed. Install with: pip install trackio")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='View Trackio dashboard')
    parser.add_argument('--project', '-p', help='Project name', default=None)
    parser.add_argument('--port', type=int, default=7860, help='Dashboard port (default: 7860)')

    args = parser.parse_args()

    print(f"Opening Trackio dashboard on port {args.port}...")
    if args.project:
        print(f"Project: {args.project}")
        trackio.show(project=args.project, port=args.port)
    else:
        print("Showing all projects")
        trackio.show(port=args.port)

    print(f"\nDashboard available at: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the dashboard")

if __name__ == "__main__":
    main()