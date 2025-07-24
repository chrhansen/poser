"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_keypoints_yolo():
    """Sample YOLO keypoints for testing (17 keypoints)."""
    # Format: x, y, confidence
    keypoints = np.array(
        [
            [100, 200, 0.9],  # 0: nose
            [90, 220, 0.8],  # 1: left_eye
            [110, 220, 0.8],  # 2: right_eye
            [85, 230, 0.7],  # 3: left_ear
            [115, 230, 0.7],  # 4: right_ear
            [80, 280, 0.85],  # 5: left_shoulder
            [120, 280, 0.85],  # 6: right_shoulder
            [75, 350, 0.8],  # 7: left_elbow
            [125, 350, 0.8],  # 8: right_elbow
            [70, 420, 0.75],  # 9: left_wrist
            [130, 420, 0.75],  # 10: right_wrist
            [85, 380, 0.9],  # 11: left_hip
            [115, 380, 0.9],  # 12: right_hip
            [80, 450, 0.85],  # 13: left_knee
            [120, 450, 0.85],  # 14: right_knee
            [75, 520, 0.8],  # 15: left_ankle
            [125, 520, 0.8],  # 16: right_ankle
        ]
    )
    return keypoints


@pytest.fixture
def sample_keypoints_mediapipe():
    """Sample MediaPipe keypoints for testing (33 keypoints)."""
    # Create 33 keypoints, focusing on the ones we care about
    keypoints = np.zeros((33, 3))

    # Set confidence for all to 0.8
    keypoints[:, 2] = 0.8

    # Set specific landmarks we use
    keypoints[25] = [80, 450, 0.85]  # LEFT_KNEE
    keypoints[26] = [120, 450, 0.85]  # RIGHT_KNEE
    keypoints[27] = [75, 520, 0.8]  # LEFT_ANKLE
    keypoints[28] = [125, 520, 0.8]  # RIGHT_ANKLE

    return keypoints


@pytest.fixture
def sample_bbox():
    """Sample bounding box for testing."""
    return np.array([50, 150, 150, 550])  # x1, y1, x2, y2


@pytest.fixture
def sample_frame():
    """Create a sample video frame for testing."""
    # Create a simple 640x480 BGR image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some variation
    frame[100:200, 100:200] = [255, 0, 0]  # Blue square
    frame[250:350, 250:350] = [0, 255, 0]  # Green square
    return frame


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir
