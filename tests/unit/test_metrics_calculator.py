"""Unit tests for PoseMetricsCalculator."""

import numpy as np
import pytest

from src.metrics.calculator import PoseMetricsCalculator


class TestPoseMetricsCalculator:
    """Test the PoseMetricsCalculator class."""

    def test_init_yolo(self):
        """Test initialization with YOLO detector."""
        calc = PoseMetricsCalculator("yolo")
        assert calc.detector_type == "yolo"
        assert calc.landmarks == calc.YOLO_LANDMARKS

    def test_init_mediapipe(self):
        """Test initialization with MediaPipe detector."""
        calc = PoseMetricsCalculator("mediapipe")
        assert calc.detector_type == "mediapipe"
        assert calc.landmarks == calc.MEDIAPIPE_LANDMARKS

    def test_init_invalid_detector(self):
        """Test initialization with invalid detector type."""
        with pytest.raises(ValueError, match="Unknown detector type"):
            PoseMetricsCalculator("invalid")

    def test_calculate_distances_yolo(self, sample_keypoints_yolo):
        """Test distance calculation with YOLO keypoints."""
        calc = PoseMetricsCalculator("yolo")
        distances = calc.calculate_distances(sample_keypoints_yolo)

        # Expected distances based on sample data
        # Knees: (80, 450) to (120, 450) = 40 pixels
        # Ankles: (75, 520) to (125, 520) = 50 pixels
        assert distances["knee_distance"] == pytest.approx(40.0)
        assert distances["ankle_distance"] == pytest.approx(50.0)

    def test_calculate_distances_mediapipe(self, sample_keypoints_mediapipe):
        """Test distance calculation with MediaPipe keypoints."""
        calc = PoseMetricsCalculator("mediapipe")
        distances = calc.calculate_distances(sample_keypoints_mediapipe)

        assert distances["knee_distance"] == pytest.approx(40.0)
        assert distances["ankle_distance"] == pytest.approx(50.0)

    def test_calculate_distances_empty_keypoints(self):
        """Test distance calculation with empty keypoints."""
        calc = PoseMetricsCalculator("yolo")
        distances = calc.calculate_distances(np.array([]))

        assert distances["knee_distance"] is None
        assert distances["ankle_distance"] is None

    def test_calculate_distances_none_keypoints(self):
        """Test distance calculation with None keypoints."""
        calc = PoseMetricsCalculator("yolo")
        distances = calc.calculate_distances(None)

        assert distances["knee_distance"] is None
        assert distances["ankle_distance"] is None

    def test_low_confidence_landmarks(self, sample_keypoints_yolo):
        """Test that low confidence landmarks are ignored."""
        calc = PoseMetricsCalculator("yolo")

        # Set knee confidence to very low
        keypoints = sample_keypoints_yolo.copy()
        keypoints[13, 2] = 0.05  # Left knee confidence

        distances = calc.calculate_distances(keypoints)
        assert (
            distances["knee_distance"] is None
        )  # Should be None due to low confidence

    def test_get_all_landmark_positions(self, sample_keypoints_yolo):
        """Test getting all landmark positions."""
        calc = PoseMetricsCalculator("yolo")
        positions = calc.get_all_landmark_positions(sample_keypoints_yolo)

        assert "LEFT_KNEE" in positions
        assert "RIGHT_KNEE" in positions
        assert "LEFT_ANKLE" in positions
        assert "RIGHT_ANKLE" in positions

        # Check specific position
        x, y, z, vis = positions["LEFT_KNEE"]
        assert x == 80.0
        assert y == 450.0
        assert z == 0.0  # YOLO doesn't have z coordinate
        assert vis == 0.85

    def test_euclidean_distance_2d(self):
        """Test Euclidean distance calculation for 2D points."""
        calc = PoseMetricsCalculator("yolo")

        point1 = np.array([0, 0])
        point2 = np.array([3, 4])

        distance = calc._calculate_euclidean_distance(point1, point2)
        assert distance == pytest.approx(5.0)  # 3-4-5 triangle

    def test_euclidean_distance_3d(self):
        """Test Euclidean distance calculation for 3D points."""
        calc = PoseMetricsCalculator("yolo")

        point1 = np.array([0, 0, 0])
        point2 = np.array([1, 1, 1])

        distance = calc._calculate_euclidean_distance(point1, point2)
        assert distance == pytest.approx(np.sqrt(3))

    def test_euclidean_distance_mixed_dimensions(self):
        """Test Euclidean distance with mixed 2D/3D points."""
        calc = PoseMetricsCalculator("yolo")

        point1 = np.array([0, 0])  # 2D
        point2 = np.array([3, 4, 0])  # 3D

        distance = calc._calculate_euclidean_distance(point1, point2)
        assert distance == pytest.approx(5.0)

    def test_euclidean_distance_none_points(self):
        """Test Euclidean distance with None points."""
        calc = PoseMetricsCalculator("yolo")

        assert calc._calculate_euclidean_distance(None, np.array([0, 0])) is None
        assert calc._calculate_euclidean_distance(np.array([0, 0]), None) is None
        assert calc._calculate_euclidean_distance(None, None) is None
