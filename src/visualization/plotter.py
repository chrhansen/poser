#!/usr/bin/env python3
"""
Metrics visualization module for plotting shin angles over time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricsPlotter:
    """Create graphs from shin angle measurements."""

    def __init__(self):
        """Initialize the plotter."""
        self.fig = None
        self.ax = None
        self.angle_3d_raw_line = None
        self.angle_3d_smooth_line = None
        self.angle_3d_raw_data = []
        self.angle_3d_smooth_data = []
        self.time_data = []
        self.update_interval = 5  # Update plot every 5 frames
        self.frame_count = 0

    def generate_offline_graph(self, csv_path: str, output_path: str | None = None):
        """
        Generate a static graph from an angles CSV file.

        Args:
            csv_path: Path to the _angles.csv file
            output_path: Optional output path for the graph image
        """
        # Read CSV data
        df = pd.read_csv(csv_path)

        # Handle missing values
        # shin_angle_3d_ma contains raw values, shin_angle_3d contains smoothed values
        df["shin_angle_3d"] = pd.to_numeric(df["shin_angle_3d"], errors="coerce")
        df["shin_angle_3d_ma"] = pd.to_numeric(df["shin_angle_3d_ma"], errors="coerce")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot raw 3D shin angle (stored in shin_angle_3d_ma column)
        has_raw = df["shin_angle_3d_ma"].notna().any()
        if has_raw:
            plt.plot(
                df["timestamp_ms"],
                df["shin_angle_3d_ma"],
                label="Shin Angle 3D Raw (degrees)",
                color="red",
                linewidth=2,
                alpha=0.7,
            )

        # Plot smoothed 3D shin angle (stored in shin_angle_3d column)
        has_smooth = df["shin_angle_3d"].notna().any()
        if has_smooth:
            plt.plot(
                df["timestamp_ms"],
                df["shin_angle_3d"],
                label="Shin Angle 3D Smoothed (degrees)",
                color="green",
                linewidth=2,
                alpha=0.9,
            )

        # Customize plot
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Angle (degrees)", fontsize=12)
        plt.title("Shin Angles Over Time", fontsize=14, fontweight="bold")
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Set y-axis to show angle range (0-180 degrees)
        plt.ylim(0, 180)

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
        (self.angle_3d_raw_line,) = self.ax.plot(
            [], [], "r-", label="Shin Angle 3D Raw", linewidth=2, alpha=0.7
        )
        (self.angle_3d_smooth_line,) = self.ax.plot(
            [], [], "g-", label="Shin Angle 3D Smoothed", linewidth=2, alpha=0.9
        )

        # Setup plot
        self.ax.set_xlabel("Time (ms)", fontsize=12)
        self.ax.set_ylabel("Angle (degrees)", fontsize=12)
        self.ax.set_title("Real-time Shin Angles", fontsize=14, fontweight="bold")
        self.ax.legend(loc="upper right", fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # Initial axis limits
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 180)

        # Show the plot window
        plt.show(block=False)

    def update_realtime_plot(
        self,
        timestamp_ms: float,
        shin_angle_3d_raw: float | None = None,
        shin_angle_3d_smooth: float | None = None,
    ):
        """
        Update the real-time plot with new data.

        Args:
            timestamp_ms: Current timestamp in milliseconds
            shin_angle_3d_raw: Raw shin angle in 3D (world coordinates) in degrees
            shin_angle_3d_smooth: Smoothed shin angle in 3D (world coordinates) in degrees
        """
        self.frame_count += 1

        # Append data
        self.time_data.append(timestamp_ms)
        self.angle_3d_raw_data.append(
            shin_angle_3d_raw if shin_angle_3d_raw is not None else np.nan
        )
        self.angle_3d_smooth_data.append(
            shin_angle_3d_smooth if shin_angle_3d_smooth is not None else np.nan
        )

        # Only update plot at specified interval
        if self.frame_count % self.update_interval != 0:
            return

        # Update line data
        self.angle_3d_raw_line.set_data(self.time_data, self.angle_3d_raw_data)
        self.angle_3d_smooth_line.set_data(self.time_data, self.angle_3d_smooth_data)

        # Adjust axis limits
        if len(self.time_data) > 0:
            # X-axis: show last 5 seconds of data
            x_min = max(0, timestamp_ms - 5000)
            x_max = timestamp_ms + 500
            self.ax.set_xlim(x_min, x_max)

            # Y-axis: auto-scale with some padding
            valid_raw = [d for d in self.angle_3d_raw_data if not np.isnan(d)]
            valid_smooth = [d for d in self.angle_3d_smooth_data if not np.isnan(d)]

            all_valid = valid_raw + valid_smooth
            if all_valid:
                y_min = max(0, min(all_valid) * 0.9)
                y_max = min(180, max(all_valid) * 1.1)
                self.ax.set_ylim(y_min, y_max)

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
