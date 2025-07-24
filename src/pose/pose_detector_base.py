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
        self.smoothing_cfg = cfg.get("smoothing", {})
        self.smoothers = {}  # joint_id -> smoother instance

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

    def _init_smoother(self, joint_id: int):
        """Initialize smoother for a specific joint."""
        from utils.smoothing import create_smoother

        self.smoothers[joint_id] = create_smoother(self.smoothing_cfg)

    def _smooth_keypoints(self, keypoints: np.ndarray, dt: float) -> np.ndarray:
        """Apply smoothing to keypoints."""
        smoothed = keypoints.copy()

        for i in range(len(keypoints)):
            if (
                keypoints[i, 2] < self.conf_min
                or np.isnan(keypoints[i, 0])
                or np.isnan(keypoints[i, 1])
            ):
                continue

            if i not in self.smoothers:
                self._init_smoother(i)

            # Smooth x and y coordinates
            smoothed[i, 0] = self.smoothers[i].filter(keypoints[i, 0], dt, 0)
            smoothed[i, 1] = self.smoothers[i].filter(keypoints[i, 1], dt, 1)

        return smoothed
