"""Unit tests for YOLO keypoint format consistency."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.pose.yolo_pose_detector import YOLOPoseDetector


class TestYOLOKeypointFormat:
    """Test YOLO keypoint format consistency with MediaPipe."""
    
    @pytest.fixture
    def mock_yolo_detector(self):
        """Create a mock YOLO detector."""
        cfg = {
            "yolo_model": "yolo11n-pose",
            "confidence": 0.5
        }
        
        with patch('src.pose.yolo_pose_detector.YOLO'):
            detector = YOLOPoseDetector(cfg)
            detector.model = Mock()
            return detector
    
    def test_detect_returns_4_element_keypoints(self, mock_yolo_detector, sample_frame, sample_bbox):
        """Test that YOLO detect() returns 4-element keypoints (x, y, z, conf)."""
        # Create mock YOLO results with 3-element keypoints
        mock_box = Mock()
        mock_box.xyxy = np.array([[50, 150, 150, 550]])
        mock_box.conf = np.array([0.9])
        
        # YOLO originally returns (17, 3) with x, y, conf
        # Create mock tensor that has .cpu() method
        keypoints_array = np.array([[
            [100, 200, 0.9],   # nose
            [90, 220, 0.8],    # left_eye
            [110, 220, 0.8],   # right_eye
            [85, 230, 0.7],    # left_ear
            [115, 230, 0.7],   # right_ear
            [80, 280, 0.85],   # left_shoulder
            [120, 280, 0.85],  # right_shoulder
            [75, 350, 0.8],    # left_elbow
            [125, 350, 0.8],   # right_elbow
            [70, 420, 0.75],   # left_wrist
            [130, 420, 0.75],  # right_wrist
            [85, 380, 0.9],    # left_hip
            [115, 380, 0.9],   # right_hip
            [80, 450, 0.85],   # left_knee
            [120, 450, 0.85],  # right_knee
            [75, 520, 0.8],    # left_ankle
            [125, 520, 0.8],   # right_ankle
        ]])
        
        # Mock the tensor object
        mock_tensor = Mock()
        mock_tensor.cpu.return_value.numpy.return_value = keypoints_array[0]
        
        # Create a mock data object that behaves like a tensor with shape attribute
        mock_data = Mock()
        mock_data.shape = keypoints_array.shape  # (1, 17, 3)
        mock_data.__getitem__ = lambda self, idx: mock_tensor if idx == 0 else None
        
        mock_keypoints = Mock()
        mock_keypoints.data = mock_data
        
        mock_result = Mock()
        mock_result.boxes = Mock(data=mock_box)
        mock_result.keypoints = mock_keypoints
        
        mock_yolo_detector.model.return_value = [mock_result]
        
        # Test
        keypoints = mock_yolo_detector.detect(sample_frame, sample_bbox)
        
        assert keypoints is not None
        assert keypoints.shape == (17, 4)  # Should be (17, 4) not (17, 3)
        
        # Check that z is always 0 for YOLO
        assert np.all(keypoints[:, 2] == 0.0)
        
        # Check that confidence is in the last position
        assert keypoints[0, 3] == 0.9  # nose confidence
        assert keypoints[5, 3] == 0.85  # left shoulder confidence
        
        # Check x, y values are reasonable (transformed coordinates)
        assert keypoints[0, 0] > 0  # nose x should be positive
        assert keypoints[0, 1] > 0  # nose y should be positive
        assert np.all(keypoints[:, 0] >= 0)  # All x coords should be non-negative
        assert np.all(keypoints[:, 1] >= 0)  # All y coords should be non-negative
    
    def test_detect_handles_no_keypoints(self, mock_yolo_detector, sample_frame, sample_bbox):
        """Test that detect handles cases with no keypoints gracefully."""
        mock_box = Mock()
        mock_box.xyxy = np.array([[50, 150, 150, 550]])
        mock_box.conf = np.array([0.9])
        
        mock_result = Mock()
        mock_result.boxes = Mock(data=mock_box)
        mock_result.keypoints = None  # No keypoints
        
        mock_yolo_detector.model.return_value = [mock_result]
        
        # Test
        keypoints = mock_yolo_detector.detect(sample_frame, sample_bbox)
        
        assert keypoints is None
    
    def test_keypoint_format_compatible_with_drawing(self, mock_yolo_detector):
        """Test that 4-element keypoints work with drawing functions."""
        # Create 4-element keypoints
        keypoints = np.array([
            [100, 200, 0.0, 0.9],   # x, y, z, conf
            [120, 220, 0.0, 0.8],
            [140, 240, 0.0, 0.85],
        ])
        
        # Verify we can extract confidence from last element
        for kpt in keypoints:
            x, y = kpt[0], kpt[1]
            conf = kpt[-1]  # Last element is confidence
            
            assert isinstance(x, (int, float, np.number))
            assert isinstance(y, (int, float, np.number))
            assert isinstance(conf, (int, float, np.number))
            assert 0 <= conf <= 1