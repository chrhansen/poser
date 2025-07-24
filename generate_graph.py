#!/usr/bin/env python3
"""
Standalone script to generate graphs from distance CSV files.
"""

import argparse
from pathlib import Path

from src.visualization.plotter import MetricsPlotter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate graphs from pose distance measurements"
    )
    parser.add_argument("csv_file", type=str, help="Path to the _distances.csv file")
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the graph image (default: auto-generated)",
    )
    return parser.parse_args()


def main():
    """Generate graph from CSV file."""
    args = parse_args()

    # Check if CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return 1

    if not str(csv_path).endswith("_distances.csv"):
        print("Warning: Expected a file ending with '_distances.csv'")

    # Create plotter and generate graph
    plotter = MetricsPlotter()

    try:
        plotter.generate_offline_graph(str(csv_path), args.output)
        return 0
    except Exception as e:
        print(f"Error generating graph: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
