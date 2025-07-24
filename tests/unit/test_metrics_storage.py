"""Unit tests for MetricsLogger."""

import csv
from pathlib import Path

import pytest

from src.metrics.storage import MetricsLogger


class TestMetricsLogger:
    """Test the MetricsLogger class."""

    def test_init_creates_files(self, temp_output_dir):
        """Test that initialization creates the correct files."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))
        
        assert logger.video_name == "test_video"
        assert logger.output_dir == temp_output_dir
        assert logger.distances_file.exists()
        assert logger.landmarks_file.exists()
        
        # Close files for cleanup
        logger.close()

    def test_log_distances(self, temp_output_dir):
        """Test logging distance measurements."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))
        
        # Log some test data
        logger.log_distances(0, 0.0, 40.5, 50.3)
        logger.log_distances(1, 33.3, 42.1, 51.7)
        logger.log_distances(2, 66.6, None, None)  # Test None values
        
        logger.close()
        
        # Read and verify the CSV
        with open(logger.distances_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        assert len(rows) == 4  # Header + 3 data rows
        assert rows[0] == ['frame_number', 'timestamp_ms', 'knee_distance', 'ankle_distance']
        assert rows[1] == ['0', '0.00', '40.500000', '50.300000']
        assert rows[2] == ['1', '33.30', '42.100000', '51.700000']
        assert rows[3] == ['2', '66.60', '', '']  # None values become empty strings

    def test_log_all_landmarks(self, temp_output_dir):
        """Test logging all landmark positions."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))
        
        landmarks = {
            'LEFT_KNEE': (80.0, 450.0, 0.0, 0.85),
            'RIGHT_KNEE': (120.0, 450.0, 0.0, 0.85),
        }
        
        landmark_indices = {
            'LEFT_KNEE': 13,
            'RIGHT_KNEE': 14,
        }
        
        logger.log_all_landmarks(0, 0.0, landmarks, landmark_indices)
        logger.close()
        
        # Read and verify the CSV
        with open(logger.landmarks_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        assert len(rows) == 3  # Header + 2 data rows
        assert rows[0] == [
            'frame_number', 'timestamp_ms', 'landmark_index', 'landmark_name',
            'x', 'y', 'z', 'visibility'
        ]
        
        # Check one landmark row
        left_knee_row = next(r for r in rows if r[3] == 'LEFT_KNEE')
        assert left_knee_row[2] == '13'  # landmark index
        assert left_knee_row[4] == '80.000000'  # x
        assert left_knee_row[5] == '450.000000'  # y
        assert left_knee_row[6] == '0.000000'  # z
        assert left_knee_row[7] == '0.850000'  # visibility

    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()
        
        logger = MetricsLogger("test.mp4", str(output_dir))
        assert output_dir.exists()
        
        logger.close()

    def test_file_paths_from_video_name(self, temp_output_dir):
        """Test that file paths are correctly derived from video name."""
        logger = MetricsLogger("/path/to/my_video.mp4", str(temp_output_dir))
        
        assert logger.video_name == "my_video"
        assert logger.distances_file.name == "my_video_distances.csv"
        assert logger.landmarks_file.name == "my_video_landmarks.csv"
        
        logger.close()

    def test_close_prints_paths(self, temp_output_dir, capsys):
        """Test that close() prints the file paths."""
        logger = MetricsLogger("test.mp4", str(temp_output_dir))
        logger.close()
        
        captured = capsys.readouterr()
        assert "Metrics saved to:" in captured.out
        assert "Distances:" in captured.out
        assert "Landmarks:" in captured.out
        assert str(logger.distances_file) in captured.out
        assert str(logger.landmarks_file) in captured.out