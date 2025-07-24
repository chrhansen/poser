#!/usr/bin/env python3
"""
Pose metrics calculator for computing distances between landmarks.
"""

from typing import Dict, Optional, Union, Tuple
import numpy as np
import math


class PoseMetricsCalculator:
    """Calculate distances between pose landmarks."""
    
    # MediaPipe landmark indices
    MEDIAPIPE_LANDMARKS = {
        'LEFT_KNEE': 25,
        'RIGHT_KNEE': 26,
        'LEFT_ANKLE': 27,
        'RIGHT_ANKLE': 28
    }
    
    # YOLO/COCO landmark indices
    YOLO_LANDMARKS = {
        'LEFT_KNEE': 13,
        'RIGHT_KNEE': 14,
        'LEFT_ANKLE': 15,
        'RIGHT_ANKLE': 16
    }
    
    def __init__(self, detector_type: str = 'yolo'):
        """
        Initialize the metrics calculator.
        
        Args:
            detector_type: Either 'yolo' or 'mediapipe'
        """
        self.detector_type = detector_type
        if detector_type == 'mediapipe':
            self.landmarks = self.MEDIAPIPE_LANDMARKS
        elif detector_type == 'yolo':
            self.landmarks = self.YOLO_LANDMARKS
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def calculate_distances(self, keypoints: np.ndarray) -> Dict[str, Optional[float]]:
        """
        Calculate knee and ankle distances from keypoints.
        
        Args:
            keypoints: Array of shape (N, 3) with (x, y, conf) or (N, 4) with (x, y, z, conf)
                      For YOLO, coordinates are in pixels (2D)
                      For MediaPipe, coordinates can be 3D world coordinates
        
        Returns:
            Dictionary with 'knee_distance' and 'ankle_distance' in appropriate units
            Returns None for distances if landmarks are not detected
        """
        if keypoints is None or len(keypoints) == 0:
            return {'knee_distance': None, 'ankle_distance': None}
        
        # Extract relevant landmarks
        left_knee = self._get_landmark(keypoints, 'LEFT_KNEE')
        right_knee = self._get_landmark(keypoints, 'RIGHT_KNEE')
        left_ankle = self._get_landmark(keypoints, 'LEFT_ANKLE')
        right_ankle = self._get_landmark(keypoints, 'RIGHT_ANKLE')
        
        # Calculate distances
        knee_distance = self._calculate_euclidean_distance(left_knee, right_knee)
        ankle_distance = self._calculate_euclidean_distance(left_ankle, right_ankle)
        
        return {
            'knee_distance': knee_distance,
            'ankle_distance': ankle_distance
        }
    
    def _get_landmark(self, keypoints: np.ndarray, landmark_name: str) -> Optional[np.ndarray]:
        """
        Extract a specific landmark from keypoints array.
        
        Returns:
            Numpy array with coordinates [x, y] or [x, y, z], or None if not detected
        """
        idx = self.landmarks[landmark_name]
        
        if idx >= len(keypoints):
            return None
        
        point = keypoints[idx]
        
        # Check confidence (last element)
        confidence = point[-1] if len(point) >= 3 else 0
        if confidence < 0.1:  # Minimum confidence threshold
            return None
        
        # Return coordinates without confidence
        if len(point) == 3:  # x, y, conf
            return point[:2]
        elif len(point) == 4:  # x, y, z, conf
            return point[:3]
        else:
            return None
    
    def _calculate_euclidean_distance(self, point1: Optional[np.ndarray], 
                                    point2: Optional[np.ndarray]) -> Optional[float]:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point coordinates [x, y] or [x, y, z]
            point2: Second point coordinates [x, y] or [x, y, z]
        
        Returns:
            Euclidean distance or None if either point is None
        """
        if point1 is None or point2 is None:
            return None
        
        # Ensure both points have the same dimensionality
        if len(point1) != len(point2):
            # If one is 2D and other is 3D, pad the 2D with 0 for z
            if len(point1) == 2 and len(point2) == 3:
                point1 = np.append(point1, 0)
            elif len(point1) == 3 and len(point2) == 2:
                point2 = np.append(point2, 0)
        
        # Calculate distance using numpy
        return float(np.linalg.norm(point2 - point1))
    
    def get_all_landmark_positions(self, keypoints: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get all landmark positions with their coordinates and visibility.
        
        Returns:
            Dictionary mapping landmark names to (x, y, z, visibility) tuples
        """
        positions = {}
        
        for name, idx in self.landmarks.items():
            if idx >= len(keypoints):
                positions[name] = (0.0, 0.0, 0.0, 0.0)
                continue
            
            point = keypoints[idx]
            
            if len(point) >= 3:
                x, y = point[0], point[1]
                z = point[2] if len(point) == 4 else 0.0
                visibility = point[-1]  # Last element is always confidence/visibility
            else:
                x, y, z, visibility = 0.0, 0.0, 0.0, 0.0
            
            positions[name] = (float(x), float(y), float(z), float(visibility))
        
        return positions