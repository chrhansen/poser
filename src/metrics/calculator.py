#!/usr/bin/env python3
"""
Pose metrics calculator for computing distances between landmarks.
"""


import numpy as np


class PoseMetricsCalculator:
    """Calculate distances between pose landmarks."""

    # MediaPipe landmark indices
    MEDIAPIPE_LANDMARKS = {
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
    }

    # YOLO/COCO landmark indices
    YOLO_LANDMARKS = {
        "LEFT_SHOULDER": 5,
        "RIGHT_SHOULDER": 6,
        "LEFT_HIP": 11,
        "RIGHT_HIP": 12,
        "LEFT_KNEE": 13,
        "RIGHT_KNEE": 14,
        "LEFT_ANKLE": 15,
        "RIGHT_ANKLE": 16,
    }

    def __init__(self, detector_type: str = "yolo", window_size: int = 3):
        """
        Initialize the metrics calculator.

        Args:
            detector_type: Either 'yolo' or 'mediapipe'
            window_size: Window size for moving average calculation (default: 3)
        """
        self.detector_type = detector_type
        self.window_size = window_size

        # Buffers for moving average calculation
        self.shin_angle_2d_buffer = []
        self.shin_angle_3d_buffer = []

        if detector_type == "mediapipe":
            self.landmarks = self.MEDIAPIPE_LANDMARKS
        elif detector_type == "yolo":
            self.landmarks = self.YOLO_LANDMARKS
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def calculate_shin_angles(
        self, keypoints: np.ndarray, is_world_coords: bool = False
    ) -> dict[str, float | None]:
        """
        Calculate the angle between the two shin vectors (ankle to knee).

        Args:
            keypoints: Array of shape (N, 3) with (x, y, conf) or (N, 4) with (x, y, z, conf)
            is_world_coords: If True, coordinates are in meters (3D world coordinates)
                           If False, coordinates are in pixels (2D frame coordinates)

        Returns:
            Dictionary with 'shin_angle', 'shin_angle_ma'
            Angle is in degrees, where 0° means perfectly parallel shins
        """
        if keypoints is None or len(keypoints) == 0:
            return {
                "shin_angle": None,
                "shin_angle_ma": None,
            }

        # Extract relevant landmarks
        left_knee = self._get_landmark(keypoints, "LEFT_KNEE")
        right_knee = self._get_landmark(keypoints, "RIGHT_KNEE")
        left_ankle = self._get_landmark(keypoints, "LEFT_ANKLE")
        right_ankle = self._get_landmark(keypoints, "RIGHT_ANKLE")

        # Calculate shin angle
        shin_angle = self._calculate_shin_angle(
            left_knee, left_ankle, right_knee, right_ankle
        )

        # Select appropriate buffer based on coordinate type
        buffer = (
            self.shin_angle_3d_buffer if is_world_coords else self.shin_angle_2d_buffer
        )

        # Update buffer and calculate moving average
        angle_ma = self._update_buffer_and_calculate_ma(buffer, shin_angle)

        return {
            "shin_angle": shin_angle,
            "shin_angle_ma": angle_ma,
        }

    def _get_landmark(
        self, keypoints: np.ndarray, landmark_name: str
    ) -> np.ndarray | None:
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

    def _calculate_shin_angle(
        self,
        knee_L: np.ndarray | None,
        ankle_L: np.ndarray | None,
        knee_R: np.ndarray | None,
        ankle_R: np.ndarray | None,
    ) -> float | None:
        """
        Calculate the angle between the two shin vectors.

        Args:
            knee_L: Left knee coordinates [x, y] or [x, y, z]
            ankle_L: Left ankle coordinates [x, y] or [x, y, z]
            knee_R: Right knee coordinates [x, y] or [x, y, z]
            ankle_R: Right ankle coordinates [x, y] or [x, y, z]

        Returns:
            Angle in degrees between the two shin vectors (0° = parallel)
        """
        if any(p is None for p in [knee_L, ankle_L, knee_R, ankle_R]):
            return None

        # Build shin vectors (from ankle to knee)
        vL = knee_L - ankle_L  # Left shin vector
        vR = knee_R - ankle_R  # Right shin vector

        # Normalize vectors to avoid numerical issues
        vL_norm = np.linalg.norm(vL)
        vR_norm = np.linalg.norm(vR)

        if vL_norm == 0 or vR_norm == 0:
            return None

        vL = vL / vL_norm
        vR = vR / vR_norm

        # Calculate angle using dot product
        # cos(θ) = (vL · vR) / (|vL| |vR|)
        # Since vectors are normalized, cos(θ) = vL · vR
        cos_theta = np.clip(np.dot(vL, vR), -1.0, 1.0)

        # Convert to degrees
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def get_all_landmark_positions(
        self, keypoints: np.ndarray
    ) -> dict[str, tuple[float, float, float, float]]:
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

            if len(point) == 4:
                # Format: (x, y, z, visibility/confidence)
                x, y, z, visibility = point[0], point[1], point[2], point[3]
            elif len(point) == 3:
                # Format: (x, y, visibility/confidence) - no z coordinate
                x, y, visibility = point[0], point[1], point[2]
                z = 0.0
            else:
                x, y, z, visibility = 0.0, 0.0, 0.0, 0.0

            positions[name] = (float(x), float(y), float(z), float(visibility))

        return positions

    def _update_buffer_and_calculate_ma(
        self, buffer: list[float], value: float | None
    ) -> float | None:
        """
        Update buffer with new value and calculate moving average.

        Args:
            buffer: Buffer to update
            value: New value to add (can be None)

        Returns:
            Moving average or None if not enough valid values
        """
        if value is not None:
            buffer.append(value)

            # Keep only the last window_size values
            if len(buffer) > self.window_size:
                buffer.pop(0)

            # Calculate moving average
            if len(buffer) > 0:
                return float(np.mean(buffer))

        return None
