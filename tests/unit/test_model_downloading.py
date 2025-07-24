#!/usr/bin/env python3
"""
Test automatic model downloading functionality.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pose.mediapipe_pose_detector import MediaPipePoseDetector
from src.pose.yolo_pose_detector import YOLOPoseDetector


class TestModelDownloading(unittest.TestCase):
    """Test automatic model downloading for pose detectors."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.chdir(self.original_cwd)

    def test_mediapipe_model_download(self):
        """Test that MediaPipe models are downloaded automatically."""
        # Create a test config with model path in temp directory
        test_model_path = os.path.join(self.temp_dir, "models", "pose_landmarker_heavy.task")
        cfg = {
            "mediapipe_model_path": test_model_path,
            "smoothing": {"kind": "none"},
            "conf_min": 0.2,
            "pad_ratio": 0.5
        }
        
        # Ensure model doesn't exist
        self.assertFalse(os.path.exists(test_model_path))
        
        # Create detector
        detector = MediaPipePoseDetector(cfg)
        
        # Mock both the download and MediaPipe's PoseLandmarker
        with patch.object(detector, '_download_model') as mock_download:
            with patch('src.pose.mediapipe_pose_detector.vision.PoseLandmarker') as mock_landmarker:
                # Configure mock to avoid actual MediaPipe initialization
                mock_landmarker.create_from_options.return_value = MagicMock()
                
                # Create the file after "download" to simulate successful download
                def side_effect():
                    os.makedirs(os.path.dirname(test_model_path), exist_ok=True)
                    Path(test_model_path).touch()
                
                mock_download.side_effect = side_effect
                
                # This should trigger download since file doesn't exist
                detector.load_model(cfg)
                
                # Verify download was called
                mock_download.assert_called_once()

    def test_mediapipe_model_no_download_if_exists(self):
        """Test that MediaPipe models are not downloaded if they already exist."""
        # Create a test config with model path in temp directory
        test_model_path = os.path.join(self.temp_dir, "models", "pose_landmarker_heavy.task")
        cfg = {
            "mediapipe_model_path": test_model_path,
            "smoothing": {"kind": "none"},
            "conf_min": 0.2,
            "pad_ratio": 0.5
        }
        
        # Create the model file
        os.makedirs(os.path.dirname(test_model_path), exist_ok=True)
        Path(test_model_path).touch()
        
        # Ensure model exists
        self.assertTrue(os.path.exists(test_model_path))
        
        # Create detector
        detector = MediaPipePoseDetector(cfg)
        
        # Mock the download method and MediaPipe
        with patch.object(detector, '_download_model') as mock_download:
            with patch('src.pose.mediapipe_pose_detector.vision.PoseLandmarker') as mock_landmarker:
                # Configure mock to avoid actual MediaPipe initialization
                mock_landmarker.create_from_options.return_value = MagicMock()
                
                # This should NOT trigger download since file exists
                detector.load_model(cfg)
                
                # Verify download was NOT called
                mock_download.assert_not_called()

    def test_mediapipe_download_creates_directory(self):
        """Test that MediaPipe download creates the models directory if needed."""
        # Use a path that doesn't exist
        test_model_path = os.path.join(self.temp_dir, "new_models_dir", "pose_landmarker_heavy.task")
        cfg = {
            "mediapipe_model_path": test_model_path,
            "smoothing": {"kind": "none"},
            "conf_min": 0.2,
            "pad_ratio": 0.5
        }
        
        detector = MediaPipePoseDetector(cfg)
        
        # Mock urllib to avoid actual download
        with patch('urllib.request.urlretrieve') as mock_urlretrieve:
            mock_urlretrieve.return_value = (test_model_path, None)
            
            # This should create directory and download
            detector._download_model()
            
            # Verify directory was created
            self.assertTrue(os.path.exists(os.path.dirname(test_model_path)))
            
            # Verify download was attempted
            mock_urlretrieve.assert_called_once()

    def test_mediapipe_unknown_model_error(self):
        """Test that unknown MediaPipe model paths raise an error."""
        cfg = {
            "mediapipe_model_path": "models/unknown_model.task",
            "smoothing": {"kind": "none"},
            "conf_min": 0.2,
            "pad_ratio": 0.5
        }
        
        detector = MediaPipePoseDetector(cfg)
        
        # This should raise ValueError for unknown model
        with self.assertRaises(ValueError) as context:
            detector._download_model()
        
        self.assertIn("Unknown MediaPipe model", str(context.exception))

    @patch('src.pose.yolo_pose_detector.YOLO')
    def test_yolo_model_auto_download(self, mock_yolo_class):
        """Test that YOLO models are downloaded automatically by ultralytics."""
        cfg = {
            "pose_model": "models/yolo11x-pose.pt",
            "smoothing": {"kind": "none"},
            "pad_ratio": 0.5,
            "conf_min": 0.2
        }
        
        # Create detector and load model
        detector = YOLOPoseDetector(cfg)
        detector.load_model(cfg)
        
        # Verify YOLO was instantiated with the model path
        # The YOLO class handles downloading automatically
        mock_yolo_class.assert_called_once_with("models/yolo11x-pose.pt")


if __name__ == "__main__":
    unittest.main()