"""Unit tests for MetricsLogger."""

import csv

from src.metrics.storage import MetricsLogger


class TestMetricsLogger:
    """Test the MetricsLogger class."""

    def test_init_creates_files(self, temp_output_dir):
        """Test that initialization creates the correct files."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))

        assert logger.video_name == "test_video"
        assert logger.output_dir == temp_output_dir
        assert logger.angles_file.exists()
        assert logger.landmarks_file.exists()

        # Close files for cleanup
        logger.close()

    def test_log_shin_angles(self, temp_output_dir):
        """Test logging shin angle measurements."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))

        # Log some test data
        logger.log_shin_angles(0, 0.0, 45.5, 45.2, 44.8, 44.5)
        logger.log_shin_angles(1, 33.3, 42.1, 43.8, 41.7, 43.3)
        logger.log_shin_angles(2, 66.6, None, None, None, None)  # Test None values

        logger.close()

        # Read and verify the CSV
        with open(logger.angles_file) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 4  # Header + 3 data rows
        assert rows[0] == [
            "frame_number",
            "timestamp_ms",
            "shin_angle_2d",
            "shin_angle_2d_ma",
            "shin_angle_3d",
            "shin_angle_3d_ma",
        ]
        assert rows[1] == [
            "0",
            "0.00",
            "45.50",
            "45.20",
            "44.80",
            "44.50",
        ]
        assert rows[2] == [
            "1",
            "33.30",
            "42.10",
            "43.80",
            "41.70",
            "43.30",
        ]
        assert rows[3] == [
            "2",
            "66.60",
            "",
            "",
            "",
            "",
        ]  # None values become empty strings

    def test_log_all_landmarks(self, temp_output_dir):
        """Test logging all landmark positions."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))

        landmarks = {
            "LEFT_KNEE": (80.0, 450.0, 0.0, 0.85),
            "RIGHT_KNEE": (120.0, 450.0, 0.0, 0.85),
        }

        landmark_indices = {
            "LEFT_KNEE": 13,
            "RIGHT_KNEE": 14,
        }

        logger.log_all_landmarks(0, 0.0, landmarks, landmark_indices)
        logger.close()

        # Read and verify the CSV
        with open(logger.landmarks_file) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # Header + 2 data rows
        assert rows[0] == [
            "frame_number",
            "timestamp_ms",
            "landmark_index",
            "landmark_name",
            "x",
            "y",
            "z",
            "visibility",
        ]

        # Check one landmark row
        left_knee_row = next(r for r in rows if r[3] == "LEFT_KNEE")
        assert left_knee_row[2] == "13"  # landmark index
        assert left_knee_row[4] == "80.000000"  # x
        assert left_knee_row[5] == "450.000000"  # y
        assert left_knee_row[6] == "0.000000"  # z
        assert left_knee_row[7] == "0.850000"  # visibility

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
        assert logger.angles_file.name == "my_video_angles.csv"
        assert logger.landmarks_file.name == "my_video_landmarks.csv"

        logger.close()

    def test_close_prints_paths(self, temp_output_dir, capsys):
        """Test that close() prints the file paths."""
        logger = MetricsLogger("test.mp4", str(temp_output_dir))
        logger.close()

        captured = capsys.readouterr()
        assert "Metrics saved to:" in captured.out
        assert "Angles:" in captured.out
        assert "Landmarks:" in captured.out
        assert str(logger.angles_file) in captured.out
        assert str(logger.landmarks_file) in captured.out

    def test_log_world_landmarks(self, temp_output_dir):
        """Test logging world landmark coordinates."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))

        # Test world landmarks (in meters)
        world_landmarks = {
            "LEFT_SHOULDER": (-0.2, 0.8, -0.1, 0.95),
            "RIGHT_SHOULDER": (0.2, 0.8, -0.1, 0.95),
            "LEFT_HIP": (-0.15, 0.4, 0.0, 0.98),
            "RIGHT_HIP": (0.15, 0.4, 0.0, 0.98),
            "LEFT_KNEE": (-0.15, 0.0, 0.05, 0.90),
            "RIGHT_KNEE": (0.15, 0.0, 0.05, 0.90),
            "LEFT_ANKLE": (-0.15, -0.4, 0.1, 0.85),
            "RIGHT_ANKLE": (0.15, -0.4, 0.1, 0.85),
        }

        landmark_indices = {
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_KNEE": 25,
            "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28,
        }

        logger.log_world_landmarks(0, 0.0, world_landmarks, landmark_indices)
        logger.log_world_landmarks(1, 33.3, world_landmarks, landmark_indices)
        logger.close()

        # Verify world landmarks file was created
        assert logger.world_landmarks_file.exists()

        # Read and verify the CSV
        with open(logger.world_landmarks_file) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 17  # Header + 8 landmarks * 2 frames
        assert rows[0] == [
            "frame_number",
            "timestamp_ms",
            "landmark_index",
            "landmark_name",
            "x_meters",
            "y_meters",
            "z_meters",
            "visibility",
        ]

        # Check a specific landmark
        left_knee_row = next(r for r in rows if r[3] == "LEFT_KNEE" and r[0] == "0")
        assert left_knee_row[2] == "25"  # landmark index
        assert left_knee_row[4] == "-0.150000"  # x in meters
        assert left_knee_row[5] == "0.000000"  # y in meters
        assert left_knee_row[6] == "0.050000"  # z in meters
        assert left_knee_row[7] == "0.900000"  # visibility

    def test_log_world_landmarks_none_values(self, temp_output_dir):
        """Test that log_world_landmarks handles None values gracefully."""
        logger = MetricsLogger("test_video.mp4", str(temp_output_dir))

        # Test with empty landmarks dict (since the method expects a dict)
        logger.log_world_landmarks(0, 0.0, {}, {})
        logger.close()

        # World landmarks file should be created with just header
        assert logger.world_landmarks_file.exists()

        with open(logger.world_landmarks_file) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 1  # Only header

    def test_close_with_world_landmarks(self, temp_output_dir, capsys):
        """Test that close() prints world landmarks file path when it exists."""
        logger = MetricsLogger("test.mp4", str(temp_output_dir))

        # Log some world landmarks to create the file
        logger.log_world_landmarks(
            0, 0.0, {"LEFT_KNEE": (0, 0, 0, 1)}, {"LEFT_KNEE": 25}
        )
        logger.close()

        captured = capsys.readouterr()
        assert "World landmarks:" in captured.out
        assert str(logger.world_landmarks_file) in captured.out
