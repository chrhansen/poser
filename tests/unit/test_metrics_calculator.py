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

    def test_calculate_shin_angles_yolo(self, sample_keypoints_yolo):
        """Test shin angle calculation with YOLO keypoints."""
        calc = PoseMetricsCalculator("yolo")
        angles = calc.calculate_shin_angles(sample_keypoints_yolo)

        # Expected angle: parallel shins should have near-zero angle
        # Left shin: (80, 450) to (75, 520)
        # Right shin: (120, 450) to (125, 520)
        # Both pointing roughly downward
        assert angles["shin_angle"] is not None
        assert angles["shin_angle"] < 10  # Should be nearly parallel

    def test_calculate_shin_angles_mediapipe(self, sample_keypoints_mediapipe):
        """Test shin angle calculation with MediaPipe keypoints."""
        calc = PoseMetricsCalculator("mediapipe")
        angles = calc.calculate_shin_angles(sample_keypoints_mediapipe)

        assert angles["shin_angle"] is not None
        assert angles["shin_angle"] < 10  # Should be nearly parallel

    def test_calculate_shin_angles_empty_keypoints(self):
        """Test shin angle calculation with empty keypoints."""
        calc = PoseMetricsCalculator("yolo")
        angles = calc.calculate_shin_angles(np.array([]))

        assert angles["shin_angle"] is None
        assert angles["shin_angle_ma"] is None

    def test_calculate_shin_angles_none_keypoints(self):
        """Test shin angle calculation with None keypoints."""
        calc = PoseMetricsCalculator("yolo")
        angles = calc.calculate_shin_angles(None)

        assert angles["shin_angle"] is None
        assert angles["shin_angle_ma"] is None

    def test_low_confidence_landmarks(self, sample_keypoints_yolo):
        """Test that low confidence landmarks are ignored."""
        calc = PoseMetricsCalculator("yolo")

        # Set knee confidence to very low
        keypoints = sample_keypoints_yolo.copy()
        keypoints[13, 2] = 0.05  # Left knee confidence

        angles = calc.calculate_shin_angles(keypoints)
        assert angles["shin_angle"] is None  # Should be None due to low confidence

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

    def test_shin_angle_calculation_2d(self):
        """Test shin angle calculation for 2D vectors."""
        calc = PoseMetricsCalculator("yolo")

        # Parallel shins
        knee_L = np.array([0, 0])
        ankle_L = np.array([0, 10])
        knee_R = np.array([5, 0])
        ankle_R = np.array([5, 10])

        angle = calc._calculate_shin_angle(knee_L, ankle_L, knee_R, ankle_R)
        assert angle == pytest.approx(0.0)  # Parallel

    def test_shin_angle_calculation_perpendicular(self):
        """Test shin angle calculation for perpendicular vectors."""
        calc = PoseMetricsCalculator("yolo")

        # Perpendicular shins
        knee_L = np.array([0, 0])
        ankle_L = np.array([0, 10])
        knee_R = np.array([5, 0])
        ankle_R = np.array([15, 0])

        angle = calc._calculate_shin_angle(knee_L, ankle_L, knee_R, ankle_R)
        assert angle == pytest.approx(90.0)  # Perpendicular

    def test_shin_angle_none_points(self):
        """Test shin angle with None points."""
        calc = PoseMetricsCalculator("yolo")

        assert (
            calc._calculate_shin_angle(
                None, np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
            )
            is None
        )
        assert (
            calc._calculate_shin_angle(
                np.array([0, 0]), None, np.array([0, 0]), np.array([0, 0])
            )
            is None
        )
        assert (
            calc._calculate_shin_angle(
                np.array([0, 0]), np.array([0, 0]), None, np.array([0, 0])
            )
            is None
        )
        assert (
            calc._calculate_shin_angle(
                np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), None
            )
            is None
        )
