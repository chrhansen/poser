#!/usr/bin/env python3
"""
Metrics visualization module for plotting knee and ankle distances over time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricsPlotter:
    """Create graphs from distance measurements."""

    def __init__(self):
        """Initialize the plotter."""
        self.fig = None
        self.ax = None
        self.knee_line = None
        self.ankle_line = None
        self.knee_data = []
        self.ankle_data = []
        self.time_data = []
        self.update_interval = 5  # Update plot every 5 frames
        self.frame_count = 0

    def generate_offline_graph(self, csv_path: str, output_path: str | None = None):
        """
        Generate a static graph from a distances CSV file.

        Args:
            csv_path: Path to the _distances.csv file
            output_path: Optional output path for the graph image
        """
        # Read CSV data
        df = pd.read_csv(csv_path)

        # Handle missing values
        df["knee_distance"] = pd.to_numeric(df["knee_distance"], errors="coerce")
        df["ankle_distance"] = pd.to_numeric(df["ankle_distance"], errors="coerce")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot data
        plt.plot(
            df["timestamp_ms"],
            df["knee_distance"],
            label="Knee Distance",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            df["timestamp_ms"],
            df["ankle_distance"],
            label="Ankle Distance",
            color="red",
            linewidth=2,
        )

        # Customize plot
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Distance", fontsize=12)
        plt.title("Knee and Ankle Distances Over Time", fontsize=14, fontweight="bold")
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Set y-axis to start from 0
        plt.ylim(bottom=0)

        # Save or show
        if output_path is None:
            # Generate output path from input
            csv_path = Path(csv_path)
            output_path = csv_path.parent / f"{csv_path.stem}_graph.png"

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Graph saved to: {output_path}")

    def init_realtime_plot(self):
        """Initialize the real-time plot window."""
        plt.ion()  # Interactive mode

        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        # Initialize empty lines
        (self.knee_line,) = self.ax.plot(
            [], [], "b-", label="Knee Distance", linewidth=2
        )
        (self.ankle_line,) = self.ax.plot(
            [], [], "r-", label="Ankle Distance", linewidth=2
        )

        # Setup plot
        self.ax.set_xlabel("Time (ms)", fontsize=12)
        self.ax.set_ylabel("Distance", fontsize=12)
        self.ax.set_title(
            "Real-time Knee and Ankle Distances", fontsize=14, fontweight="bold"
        )
        self.ax.legend(loc="upper right", fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # Initial axis limits
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 100)

        # Show the plot window
        plt.show(block=False)

    def update_realtime_plot(
        self,
        timestamp_ms: float,
        knee_distance: float | None,
        ankle_distance: float | None,
    ):
        """
        Update the real-time plot with new data.

        Args:
            timestamp_ms: Current timestamp in milliseconds
            knee_distance: Knee distance measurement
            ankle_distance: Ankle distance measurement
        """
        self.frame_count += 1

        # Append data
        self.time_data.append(timestamp_ms)
        self.knee_data.append(knee_distance if knee_distance is not None else np.nan)
        self.ankle_data.append(ankle_distance if ankle_distance is not None else np.nan)

        # Only update plot at specified interval
        if self.frame_count % self.update_interval != 0:
            return

        # Update line data
        self.knee_line.set_data(self.time_data, self.knee_data)
        self.ankle_line.set_data(self.time_data, self.ankle_data)

        # Adjust axis limits
        if len(self.time_data) > 0:
            # X-axis: show last 5 seconds of data
            x_min = max(0, timestamp_ms - 5000)
            x_max = timestamp_ms + 500
            self.ax.set_xlim(x_min, x_max)

            # Y-axis: auto-scale with some padding
            valid_knee = [d for d in self.knee_data if not np.isnan(d)]
            valid_ankle = [d for d in self.ankle_data if not np.isnan(d)]

            if valid_knee or valid_ankle:
                all_valid = valid_knee + valid_ankle
                y_max = max(all_valid) * 1.1 if all_valid else 100
                self.ax.set_ylim(0, y_max)

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the real-time plot window."""
        if self.fig is not None:
            plt.close(self.fig)

    def save_current_plot(self, output_path: str):
        """Save the current real-time plot to a file."""
        if self.fig is not None:
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Real-time plot saved to: {output_path}")
