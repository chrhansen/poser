#!/usr/bin/env python3
"""
Abstract base class for pose detectors.
"""

from abc import ABC, abstractmethod

import numpy as np


class PoseDetectorBase(ABC):
    """Abstract base class for pose detection engines."""

    def __init__(self, cfg: dict):
        """Initialize with configuration."""
        self.cfg = cfg
        self.model = None
        self.pad_ratio = cfg.get("pad_ratio", 0.5)
        self.conf_min = cfg.get("conf_min", 0.2)

    @abstractmethod
    def load_model(self, cfg: dict):
        """Load the pose detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray, roi_bbox: np.ndarray) -> np.ndarray | None:
        """
        Detect pose keypoints within the given ROI.

        Args:
            frame: Original video frame
            roi_bbox: Bounding box [x1, y1, x2, y2] of the person

        Returns:
            Keypoints array of shape (N, 4) with (x, y, z, conf) in frame coordinates,
            or None if no pose detected
            Note: z may be 0 for 2D-only detectors like YOLO
        """
        pass

    def set_roi_padding(self, p: float):
        """Configure padding ratio (e.g. 0.50)."""
        self.pad_ratio = p
