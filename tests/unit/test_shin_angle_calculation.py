"""Unit tests for shin angle calculation."""

import numpy as np
import pytest

from src.metrics.calculator import PoseMetricsCalculator


class TestShinAngleCalculation:
    """Test shin angle calculation functionality."""
    
    def test_parallel_shins_zero_angle(self):
        """Test that perfectly parallel shins result in 0 degrees."""
        calculator = PoseMetricsCalculator(detector_type="yolo")
        
        # Create parallel shin vectors (both pointing straight up)
        knee_L = np.array([100, 100, 0])
        ankle_L = np.array([100, 200, 0])
        knee_R = np.array([200, 100, 0])
        ankle_R = np.array([200, 200, 0])
        
        angle = calculator._calculate_shin_angle(knee_L, ankle_L, knee_R, ankle_R)
        
        assert angle is not None
        assert abs(angle) < 0.1  # Should be very close to 0
    
    def test_perpendicular_shins_90_degrees(self):
        """Test that perpendicular shins result in 90 degrees."""
        calculator = PoseMetricsCalculator(detector_type="yolo")
        
        # Create perpendicular shin vectors
        knee_L = np.array([100, 100])  # Left shin points up
        ankle_L = np.array([100, 200])
        knee_R = np.array([200, 100])  # Right shin points right
        ankle_R = np.array([100, 100])
        
        angle = calculator._calculate_shin_angle(knee_L, ankle_L, knee_R, ankle_R)
        
        assert angle is not None
        assert abs(angle - 90) < 0.1  # Should be very close to 90
    
    def test_opposite_shins_180_degrees(self):
        """Test that opposite shins result in 180 degrees."""
        calculator = PoseMetricsCalculator(detector_type="yolo")
        
        # Create opposite shin vectors
        knee_L = np.array([100, 100])  # Left shin points up
        ankle_L = np.array([100, 200])
        knee_R = np.array([200, 200])  # Right shin points down
        ankle_R = np.array([200, 100])
        
        angle = calculator._calculate_shin_angle(knee_L, ankle_L, knee_R, ankle_R)
        
        assert angle is not None
        assert abs(angle - 180) < 0.1  # Should be very close to 180
    
    def test_3d_shin_angle_calculation(self):
        """Test shin angle calculation with 3D coordinates."""
        calculator = PoseMetricsCalculator(detector_type="mediapipe")
        
        # Create 3D shin vectors with some depth
        knee_L = np.array([0.1, 0.3, 0.0])  # In meters
        ankle_L = np.array([0.1, 0.0, 0.0])
        knee_R = np.array([-0.1, 0.3, 0.1])  # Right shin has some forward tilt
        ankle_R = np.array([-0.1, 0.0, 0.0])
        
        angle = calculator._calculate_shin_angle(knee_L, ankle_L, knee_R, ankle_R)
        
        assert angle is not None
        assert 0 < angle < 45  # Should be a small angle due to slight tilt
    
    def test_calculate_shin_angles_with_keypoints(self):
        """Test the full calculate_shin_angles method with keypoints."""
        calculator = PoseMetricsCalculator(detector_type="yolo")
        
        # Create mock keypoints array (17 keypoints for YOLO)
        keypoints = np.zeros((17, 4))  # x, y, z, conf
        
        # Set the relevant landmarks with high confidence
        # Left knee (index 13)
        keypoints[13] = [100, 300, 0, 0.9]
        # Right knee (index 14)
        keypoints[14] = [200, 300, 0, 0.9]
        # Left ankle (index 15)
        keypoints[15] = [100, 400, 0, 0.9]
        # Right ankle (index 16)
        keypoints[16] = [200, 400, 0, 0.9]
        
        result = calculator.calculate_shin_angles(keypoints, is_world_coords=False)
        
        assert result["shin_angle"] is not None
        assert abs(result["shin_angle"]) < 0.1  # Should be close to 0 (parallel)
        assert result["shin_angle_ma"] is not None  # Should have moving average
    
    def test_missing_landmarks_returns_none(self):
        """Test that missing landmarks result in None angle."""
        calculator = PoseMetricsCalculator(detector_type="yolo")
        
        # Create keypoints with low confidence landmarks
        keypoints = np.zeros((17, 4))
        keypoints[:, 3] = 0.05  # All landmarks have very low confidence
        
        result = calculator.calculate_shin_angles(keypoints)
        
        assert result["shin_angle"] is None
        assert result["shin_angle_ma"] is None
    
    def test_moving_average_calculation(self):
        """Test that moving average is calculated correctly."""
        calculator = PoseMetricsCalculator(detector_type="yolo", window_size=3)
        
        # Create keypoints that will produce different angles
        keypoints1 = np.zeros((17, 4))
        keypoints1[[13, 14, 15, 16], :] = [
            [100, 300, 0, 0.9],  # Left knee
            [200, 300, 0, 0.9],  # Right knee
            [100, 400, 0, 0.9],  # Left ankle
            [200, 400, 0, 0.9],  # Right ankle
        ]
        
        keypoints2 = np.zeros((17, 4))
        keypoints2[[13, 14, 15, 16], :] = [
            [100, 300, 0, 0.9],  # Left knee
            [200, 300, 0, 0.9],  # Right knee
            [100, 400, 0, 0.9],  # Left ankle
            [180, 400, 0, 0.9],  # Right ankle moved inward
        ]
        
        # Calculate angles multiple times
        result1 = calculator.calculate_shin_angles(keypoints1)
        result2 = calculator.calculate_shin_angles(keypoints2)
        result3 = calculator.calculate_shin_angles(keypoints1)
        
        # Moving average should be affected by all three measurements
        assert result3["shin_angle_ma"] is not None
        assert result1["shin_angle"] != result2["shin_angle"]  # Different angles
        assert result1["shin_angle_ma"] == result1["shin_angle"]  # First MA equals first value
    
    def test_world_coordinates_vs_frame_coordinates(self):
        """Test that world and frame coordinate calculations use separate buffers."""
        calculator = PoseMetricsCalculator(detector_type="mediapipe")
        
        # Frame coordinates (pixels)
        frame_keypoints = np.zeros((33, 4))
        frame_keypoints[[25, 26, 27, 28], :] = [
            [100, 300, 0, 0.9],  # Left knee
            [200, 300, 0, 0.9],  # Right knee
            [100, 400, 0, 0.9],  # Left ankle
            [200, 400, 0, 0.9],  # Right ankle
        ]
        
        # World coordinates (meters)
        world_keypoints = np.zeros((33, 4))
        world_keypoints[[25, 26, 27, 28], :] = [
            [-0.15, 0.3, 0.0, 0.9],  # Left knee
            [0.15, 0.3, 0.0, 0.9],   # Right knee
            [-0.15, 0.0, 0.0, 0.9],  # Left ankle
            [0.15, 0.0, 0.1, 0.9],   # Right ankle with forward tilt
        ]
        
        # Calculate angles
        frame_result = calculator.calculate_shin_angles(frame_keypoints, is_world_coords=False)
        world_result = calculator.calculate_shin_angles(world_keypoints, is_world_coords=True)
        
        # Should use different buffers
        assert frame_result["shin_angle"] is not None
        assert world_result["shin_angle"] is not None
        assert abs(frame_result["shin_angle"]) < 0.1  # Frame coords are parallel
        assert world_result["shin_angle"] > 5  # World coords have tilt