#!/usr/bin/env python3
"""
Pose detection module with factory for different pose detection engines.
"""


import numpy as np

from utils.visual import draw_skeleton

from .mediapipe_pose_detector import MediaPipePoseDetector
from .pose_detector_base import PoseDetectorBase
from .yolo_pose_detector import YOLOPoseDetector


def create_pose_detector(detector_type: str, cfg: dict) -> PoseDetectorBase:
    """
    Factory function to create pose detector based on type.

    Args:
        detector_type: Either 'yolo' or 'mediapipe'
        cfg: Configuration dictionary

    Returns:
        PoseDetectorBase instance
    """
    if detector_type == "mediapipe":
        return MediaPipePoseDetector(cfg)
    elif detector_type == "yolo":
        return YOLOPoseDetector(cfg)
    else:
        raise ValueError(f"Unknown pose detector type: {detector_type}")


class PoseDetector:
    """
    Wrapper class for pose detection that maintains compatibility with existing code.
    This class handles the full pipeline including drawing.
    """

    def __init__(self, cfg: dict):
        """Initialize pose detector."""
        self.cfg = cfg
        self.detector = None
        self.detector_type = None

    def load_model(self, cfg: dict, detector_type: str = "yolo"):
        """Load the specified pose detection model."""
        self.detector_type = detector_type
        self.detector = create_pose_detector(detector_type, cfg)
        self.detector.load_model(cfg)

    def run(
        self,
        frame: np.ndarray,
        bbox: np.ndarray | None,
        dt: float = 1 / 30.0,
        return_keypoints: bool = False,
    ) -> np.ndarray:
        """
        Run pose detection on the given bbox region.

        Args:
            frame: Original video frame
            bbox: Bounding box [x1, y1, x2, y2] of the person
            dt: Time delta (unused, kept for compatibility)
            return_keypoints: If True, return tuple of (frame, keypoints)

        Returns:
            Updated frame with skeleton drawn, or tuple of (frame, keypoints)
        """
        if self.detector is None:
            raise RuntimeError("Pose model not initialized")

        if bbox is None:
            # No bbox provided, return unchanged frame
            if return_keypoints:
                return frame, None
            return frame

        # Detect keypoints
        keypoints = self.detector.detect(frame, bbox)

        if keypoints is not None and len(keypoints) > 0:
            # Draw skeleton
            draw_skeleton(frame, keypoints, self.detector_type, self.detector.conf_min)

        if return_keypoints:
            return frame, keypoints
        return frame

    def get_world_landmarks(
        self, frame: np.ndarray, bbox: np.ndarray | None
    ) -> np.ndarray | None:
        """
        Get world landmarks if available (currently only for MediaPipe).

        Args:
            frame: Original video frame
            bbox: Bounding box [x1, y1, x2, y2] of the person

        Returns:
            World landmarks array of shape (N, 4) with (x, y, z, visibility) in meters,
            or None if not available
        """
        if self.detector is None or bbox is None:
            return None

        # Only MediaPipe supports world landmarks
        if self.detector_type == "mediapipe" and hasattr(
            self.detector, "get_world_landmarks"
        ):
            return self.detector.get_world_landmarks(frame, bbox)

        return None
