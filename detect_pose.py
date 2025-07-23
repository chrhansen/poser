#!/usr/bin/env python3
"""
Pose detection module with factory for different pose detection engines.
"""

from typing import Optional
import numpy as np
from utils.visual import draw_skeleton

from pose_detector_base import PoseDetectorBase
from yolo_pose_detector import YOLOPoseDetector
from mediapipe_pose_detector import MediaPipePoseDetector


def create_pose_detector(detector_type: str, cfg: dict) -> PoseDetectorBase:
    """
    Factory function to create pose detector based on type.
    
    Args:
        detector_type: Either 'yolo' or 'mediapipe'
        cfg: Configuration dictionary
        
    Returns:
        PoseDetectorBase instance
    """
    if detector_type == 'mediapipe':
        return MediaPipePoseDetector(cfg)
    elif detector_type == 'yolo':
        return YOLOPoseDetector(cfg)
    else:
        raise ValueError(f"Unknown pose detector type: {detector_type}")


class PoseDetector:
    """
    Wrapper class for pose detection that maintains compatibility with existing code.
    This class handles the full pipeline including smoothing and drawing.
    """
    
    def __init__(self, cfg: dict):
        """Initialize pose detector."""
        self.cfg = cfg
        self.detector = None
        self.detector_type = None
        
    def load_model(self, cfg: dict, detector_type: str = 'yolo'):
        """Load the specified pose detection model."""
        self.detector_type = detector_type
        self.detector = create_pose_detector(detector_type, cfg)
        self.detector.load_model(cfg)
        
    def run(
        self, 
        frame: np.ndarray, 
        bbox: Optional[np.ndarray], 
        dt: float = 1/30.0
    ) -> np.ndarray:
        """
        Run pose detection on the given bbox region.
        
        Args:
            frame: Original video frame
            bbox: Bounding box [x1, y1, x2, y2] of the person
            dt: Time delta for smoothing
            
        Returns:
            Updated frame with skeleton drawn
        """
        if self.detector is None:
            raise RuntimeError("Pose model not initialized")
            
        if bbox is None:
            # No bbox provided, return unchanged frame
            return frame
            
        # Detect keypoints
        keypoints = self.detector.detect(frame, bbox)
        
        if keypoints is not None and len(keypoints) > 0:
            # Apply smoothing if enabled
            if self.detector.smoothing_cfg.get('kind', 'one_euro') != 'none':
                keypoints = self.detector._smooth_keypoints(keypoints, dt)
            
            # Draw skeleton
            draw_skeleton(frame, keypoints, self.detector_type, self.detector.conf_min)
            
        return frame