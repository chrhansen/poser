#!/usr/bin/env python3
"""
Metrics storage module for saving pose landmark data and distances to CSV files.
"""

import csv
from pathlib import Path


class MetricsLogger:
    """Handle writing of landmark and distance data to CSV files."""

    def __init__(self, video_filename: str, output_dir: str = "output"):
        """
        Initialize the metrics logger.

        Args:
            video_filename: Name of the input video file
            output_dir: Directory to save output files
        """
        self.video_name = Path(video_filename).stem
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

        # Setup file paths
        self.distances_file = self.output_dir / f"{self.video_name}_distances.csv"
        self.landmarks_file = self.output_dir / f"{self.video_name}_landmarks.csv"
        self.world_landmarks_file = self.output_dir / f"{self.video_name}_world_landmarks.csv"

        # Open files and create writers
        self._init_files()

    def _init_files(self):
        """Initialize CSV files with headers."""
        # Distances file
        self.distances_fp = open(self.distances_file, "w", newline="")
        self.distances_writer = csv.writer(self.distances_fp)
        self.distances_writer.writerow(
            [
                "frame_number",
                "timestamp_ms",
                "knee_distance",
                "ankle_distance",
                "knee_distance_ma",
                "ankle_distance_ma",
            ]
        )

        # Landmarks file
        self.landmarks_fp = open(self.landmarks_file, "w", newline="")
        self.landmarks_writer = csv.writer(self.landmarks_fp)
        self.landmarks_writer.writerow(
            [
                "frame_number",
                "timestamp_ms",
                "landmark_index",
                "landmark_name",
                "x",
                "y",
                "z",
                "visibility",
            ]
        )
        
        # World landmarks file (only created if world landmarks are available)
        self.world_landmarks_fp = None
        self.world_landmarks_writer = None
        self.has_world_landmarks = False

    def log_distances(
        self,
        frame_number: int,
        timestamp_ms: float,
        knee_distance: float | None,
        ankle_distance: float | None,
        knee_distance_ma: float | None = None,
        ankle_distance_ma: float | None = None,
    ):
        """
        Log distance measurements for a frame.

        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            knee_distance: Distance between knees (None if not detected)
            ankle_distance: Distance between ankles (None if not detected)
            knee_distance_ma: Moving average of knee distance (None if not available)
            ankle_distance_ma: Moving average of ankle distance (None if not available)
        """
        # Convert None to empty string for CSV
        knee_dist_str = "" if knee_distance is None else f"{knee_distance:.6f}"
        ankle_dist_str = "" if ankle_distance is None else f"{ankle_distance:.6f}"
        knee_ma_str = "" if knee_distance_ma is None else f"{knee_distance_ma:.6f}"
        ankle_ma_str = "" if ankle_distance_ma is None else f"{ankle_distance_ma:.6f}"

        self.distances_writer.writerow(
            [
                frame_number,
                f"{timestamp_ms:.2f}",
                knee_dist_str,
                ankle_dist_str,
                knee_ma_str,
                ankle_ma_str,
            ]
        )

        # Flush to ensure data is written
        self.distances_fp.flush()

    def log_all_landmarks(
        self,
        frame_number: int,
        timestamp_ms: float,
        landmarks: dict[str, tuple[float, float, float, float]],
        landmark_indices: dict[str, int],
    ):
        """
        Log all landmark positions for a frame.

        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            landmarks: Dictionary mapping landmark names to (x, y, z, visibility) tuples
            landmark_indices: Dictionary mapping landmark names to their indices
        """
        for name, (x, y, z, visibility) in landmarks.items():
            idx = landmark_indices.get(name, -1)

            self.landmarks_writer.writerow(
                [
                    frame_number,
                    f"{timestamp_ms:.2f}",
                    idx,
                    name,
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                    f"{visibility:.6f}",
                ]
            )

        # Flush to ensure data is written
        self.landmarks_fp.flush()

    def log_world_landmarks(
        self,
        frame_number: int,
        timestamp_ms: float,
        landmarks: dict[str, tuple[float, float, float, float]],
        landmark_indices: dict[str, int],
    ):
        """
        Log world landmark positions for all tracked landmarks.
        
        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            landmarks: Dictionary mapping landmark names to (x, y, z, visibility) tuples
                      where x, y, z are in meters (world coordinates)
            landmark_indices: Dictionary mapping landmark names to their indices
        """
        # Initialize world landmarks file on first call
        if not self.has_world_landmarks and not self.world_landmarks_fp:
            self.world_landmarks_fp = open(self.world_landmarks_file, "w", newline="")
            self.world_landmarks_writer = csv.writer(self.world_landmarks_fp)
            self.world_landmarks_writer.writerow(
                [
                    "frame_number",
                    "timestamp_ms",
                    "landmark_index",
                    "landmark_name",
                    "x_meters",
                    "y_meters",
                    "z_meters",
                    "visibility",
                ]
            )
            self.has_world_landmarks = True
            
        if self.world_landmarks_writer is None:
            return
            
        # Write each landmark's world coordinates
        for name, (x, y, z, visibility) in landmarks.items():
            idx = landmark_indices.get(name, -1)
            self.world_landmarks_writer.writerow(
                [
                    frame_number,
                    f"{timestamp_ms:.2f}",
                    idx,
                    name,
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                    f"{visibility:.6f}",
                ]
            )
            
        # Flush to ensure data is written
        self.world_landmarks_fp.flush()

    def close(self):
        """Close all open files."""
        self.distances_fp.close()
        self.landmarks_fp.close()
        
        if self.world_landmarks_fp:
            self.world_landmarks_fp.close()

        print("Metrics saved to:")
        print(f"  - Distances: {self.distances_file}")
        print(f"  - Landmarks: {self.landmarks_file}")
        
        if self.has_world_landmarks:
            print(f"  - World landmarks: {self.world_landmarks_file}")
