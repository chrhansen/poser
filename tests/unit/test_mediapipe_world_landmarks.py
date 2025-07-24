"""Unit tests for MediaPipe world landmarks functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.pose.mediapipe_pose_detector import MediaPipePoseDetector


class TestMediaPipeWorldLandmarks:
    """Test MediaPipe world landmarks functionality."""

    @pytest.fixture
    def mock_mediapipe_detector(self):
        """Create a mock MediaPipe detector with world landmarks."""
        cfg = {"mediapipe_model_path": "models/pose_landmarker_heavy.task"}

        with patch("src.pose.mediapipe_pose_detector.vision.PoseLandmarker"):
            detector = MediaPipePoseDetector(cfg)
            detector.detector = Mock()
            return detector

    def test_get_world_landmarks_returns_correct_shape(
        self, mock_mediapipe_detector, sample_frame, sample_bbox
    ):
        """Test that get_world_landmarks returns correct shape array."""
        # Create mock world landmarks
        mock_world_landmark = Mock()
        mock_world_landmark.x = 0.5
        mock_world_landmark.y = 1.0
        mock_world_landmark.z = -0.2
        mock_world_landmark.visibility = 0.95

        # Create mock detection result
        mock_result = Mock()
        mock_result.pose_world_landmarks = [[mock_world_landmark] * 33]  # 33 landmarks

        mock_mediapipe_detector.detector.detect_for_video.return_value = mock_result

        # Test
        world_landmarks = mock_mediapipe_detector.get_world_landmarks(
            sample_frame, sample_bbox
        )

        assert world_landmarks is not None
        assert world_landmarks.shape == (33, 4)  # 33 landmarks with x, y, z, visibility
        assert world_landmarks[0, 0] == 0.5  # x in meters
        assert world_landmarks[0, 1] == 1.0  # y in meters
        assert world_landmarks[0, 2] == -0.2  # z in meters
        assert world_landmarks[0, 3] == 0.95  # visibility

    def test_get_world_landmarks_none_when_no_detection(
        self, mock_mediapipe_detector, sample_frame, sample_bbox
    ):
        """Test that get_world_landmarks returns None when no pose detected."""
        # Create mock result with no landmarks
        mock_result = Mock()
        mock_result.pose_world_landmarks = []

        mock_mediapipe_detector.detector.detect_for_video.return_value = mock_result

        # Test
        world_landmarks = mock_mediapipe_detector.get_world_landmarks(
            sample_frame, sample_bbox
        )

        assert world_landmarks is None

    def test_get_world_landmarks_uses_different_timestamp(
        self, mock_mediapipe_detector, sample_frame, sample_bbox
    ):
        """Test that get_world_landmarks uses a different timestamp than detect()."""
        # Track timestamps used
        timestamps_used = []

        def capture_timestamp(image, timestamp):
            timestamps_used.append(timestamp)
            result = Mock()
            result.pose_landmarks = []
            result.pose_world_landmarks = []
            return result

        mock_mediapipe_detector.detector.detect_for_video.side_effect = (
            capture_timestamp
        )
        mock_mediapipe_detector.last_timestamp_ms = 100

        # Call detect first
        mock_mediapipe_detector.detect(sample_frame, sample_bbox)

        # Then call get_world_landmarks
        mock_mediapipe_detector.get_world_landmarks(sample_frame, sample_bbox)

        # Should have used different timestamps
        assert len(timestamps_used) == 2
        assert timestamps_used[0] != timestamps_used[1]
        assert timestamps_used[1] == timestamps_used[0] + 1  # Should increment by 1

    def test_detect_returns_z_coordinates(
        self, mock_mediapipe_detector, sample_frame, sample_bbox
    ):
        """Test that detect() now returns z-coordinates from MediaPipe landmarks."""
        # Create mock landmark with z-coordinate
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.6
        mock_landmark.z = 0.1  # Z-coordinate in normalized space
        mock_landmark.visibility = 0.9

        # Create mock detection result
        mock_result = Mock()
        mock_result.pose_landmarks = [[mock_landmark] * 33]
        mock_result.pose_world_landmarks = []

        mock_mediapipe_detector.detector.detect_for_video.return_value = mock_result

        # Test
        keypoints = mock_mediapipe_detector.detect(sample_frame, sample_bbox)

        assert keypoints is not None
        assert keypoints.shape == (33, 4)  # x, y, z, conf
        # Z should be scaled to target_size like x and y
        expected_z = 0.1 * mock_mediapipe_detector.target_size
        assert keypoints[0, 2] == expected_z

    def test_world_landmarks_in_meters(
        self, mock_mediapipe_detector, sample_frame, sample_bbox
    ):
        """Test that world landmarks are in meters (not pixels)."""
        # Create realistic world landmarks in meters
        mock_landmarks = []
        # Typical human proportions in meters
        positions = [
            (0.0, 1.6, 0.0),  # Head ~1.6m from ground
            (-0.2, 1.2, 0.0),  # Left shoulder
            (0.2, 1.2, 0.0),  # Right shoulder
            (-0.15, 0.6, 0.0),  # Left hip
            (0.15, 0.6, 0.0),  # Right hip
            (-0.15, 0.3, 0.0),  # Left knee
            (0.15, 0.3, 0.0),  # Right knee
            (-0.15, 0.0, 0.0),  # Left ankle at ground
            (0.15, 0.0, 0.0),  # Right ankle at ground
        ]

        for i in range(33):
            mock_landmark = Mock()
            if i < len(positions):
                mock_landmark.x, mock_landmark.y, mock_landmark.z = positions[i]
            else:
                mock_landmark.x = 0.0
                mock_landmark.y = 1.0
                mock_landmark.z = 0.0
            mock_landmark.visibility = 0.9
            mock_landmarks.append(mock_landmark)

        mock_result = Mock()
        mock_result.pose_world_landmarks = [mock_landmarks]

        mock_mediapipe_detector.detector.detect_for_video.return_value = mock_result

        # Test
        world_landmarks = mock_mediapipe_detector.get_world_landmarks(
            sample_frame, sample_bbox
        )

        # Verify landmarks are in reasonable meter ranges
        assert np.all(np.abs(world_landmarks[:, 0]) < 1.0)  # X within ±1 meter
        assert np.all(
            world_landmarks[:, 1] >= -0.1
        )  # Y above ground (-0.1 for tolerance)
        assert np.all(world_landmarks[:, 1] <= 2.0)  # Y below 2 meters
        assert np.all(np.abs(world_landmarks[:, 2]) < 1.0)  # Z within ±1 meter
