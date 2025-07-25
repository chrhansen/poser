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
        self.angle_2d_line = None
        self.angle_2d_ma_line = None
        self.angle_3d_line = None
        self.angle_3d_ma_line = None
        self.angle_2d_data = []
        self.angle_2d_ma_data = []
        self.angle_3d_data = []
        self.angle_3d_ma_data = []
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
        df["shin_angle_2d"] = pd.to_numeric(df["shin_angle_2d"], errors="coerce")
        df["shin_angle_3d"] = pd.to_numeric(df["shin_angle_3d"], errors="coerce")

        # Check if moving average columns exist
        has_ma = "shin_angle_2d_ma" in df.columns and "shin_angle_3d_ma" in df.columns
        if has_ma:
            df["shin_angle_2d_ma"] = pd.to_numeric(
                df["shin_angle_2d_ma"], errors="coerce"
            )
            df["shin_angle_3d_ma"] = pd.to_numeric(
                df["shin_angle_3d_ma"], errors="coerce"
            )

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot 2D shin angle
        plt.plot(
            df["timestamp_ms"],
            df["shin_angle_2d"],
            label="Shin Angle 2D (degrees)",
            color="blue",
            linewidth=2,
            alpha=0.7,
        )
        
        # Plot 3D shin angle if available
        has_3d = df["shin_angle_3d"].notna().any()
        if has_3d:
            plt.plot(
                df["timestamp_ms"],
                df["shin_angle_3d"],
                label="Shin Angle 3D (degrees)",
                color="green",
                linewidth=2,
                alpha=0.7,
            )

        # Plot moving averages if available
        if has_ma:
            plt.plot(
                df["timestamp_ms"],
                df["shin_angle_2d_ma"],
                label="Shin Angle 2D (MA)",
                color="darkblue",
                linewidth=2,
                linestyle="--",
            )
            if has_3d:
                plt.plot(
                    df["timestamp_ms"],
                    df["shin_angle_3d_ma"],
                    label="Shin Angle 3D (MA)",
                    color="darkgreen",
                    linewidth=2,
                    linestyle="--",
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
        (self.angle_2d_line,) = self.ax.plot(
            [], [], "b-", label="Shin Angle 2D", linewidth=2, alpha=0.7
        )
        (self.angle_3d_line,) = self.ax.plot(
            [], [], "g-", label="Shin Angle 3D", linewidth=2, alpha=0.7
        )
        (self.angle_2d_ma_line,) = self.ax.plot(
            [], [], "b--", label="Shin Angle 2D (MA)", linewidth=2
        )
        (self.angle_3d_ma_line,) = self.ax.plot(
            [], [], "g--", label="Shin Angle 3D (MA)", linewidth=2
        )

        # Setup plot
        self.ax.set_xlabel("Time (ms)", fontsize=12)
        self.ax.set_ylabel("Angle (degrees)", fontsize=12)
        self.ax.set_title(
            "Real-time Shin Angles", fontsize=14, fontweight="bold"
        )
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
        shin_angle_2d: float | None,
        shin_angle_2d_ma: float | None = None,
        shin_angle_3d: float | None = None,
        shin_angle_3d_ma: float | None = None,
    ):
        """
        Update the real-time plot with new data.

        Args:
            timestamp_ms: Current timestamp in milliseconds
            shin_angle_2d: Shin angle in 2D (frame coordinates) in degrees
            shin_angle_2d_ma: Moving average of 2D shin angle
            shin_angle_3d: Shin angle in 3D (world coordinates) in degrees
            shin_angle_3d_ma: Moving average of 3D shin angle
        """
        self.frame_count += 1

        # Append data
        self.time_data.append(timestamp_ms)
        self.angle_2d_data.append(shin_angle_2d if shin_angle_2d is not None else np.nan)
        self.angle_3d_data.append(shin_angle_3d if shin_angle_3d is not None else np.nan)
        self.angle_2d_ma_data.append(
            shin_angle_2d_ma if shin_angle_2d_ma is not None else np.nan
        )
        self.angle_3d_ma_data.append(
            shin_angle_3d_ma if shin_angle_3d_ma is not None else np.nan
        )

        # Only update plot at specified interval
        if self.frame_count % self.update_interval != 0:
            return

        # Update line data
        self.angle_2d_line.set_data(self.time_data, self.angle_2d_data)
        self.angle_3d_line.set_data(self.time_data, self.angle_3d_data)
        self.angle_2d_ma_line.set_data(self.time_data, self.angle_2d_ma_data)
        self.angle_3d_ma_line.set_data(self.time_data, self.angle_3d_ma_data)

        # Adjust axis limits
        if len(self.time_data) > 0:
            # X-axis: show last 5 seconds of data
            x_min = max(0, timestamp_ms - 5000)
            x_max = timestamp_ms + 500
            self.ax.set_xlim(x_min, x_max)

            # Y-axis: auto-scale with some padding
            valid_2d = [d for d in self.angle_2d_data if not np.isnan(d)]
            valid_3d = [d for d in self.angle_3d_data if not np.isnan(d)]
            valid_2d_ma = [d for d in self.angle_2d_ma_data if not np.isnan(d)]
            valid_3d_ma = [d for d in self.angle_3d_ma_data if not np.isnan(d)]

            all_valid = valid_2d + valid_3d + valid_2d_ma + valid_3d_ma
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
