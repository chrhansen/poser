#!/usr/bin/env python3
"""
MediaPipe Pose Landmarker detector implementation.
"""


import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .pose_detector_base import PoseDetectorBase


class MediaPipePoseDetector(PoseDetectorBase):
    """MediaPipe Pose Landmarker detector implementation."""

    def __init__(self, cfg: dict):
        """Initialize MediaPipe Pose detector."""
        super().__init__(cfg)
        self.detector = None
        self.model_path = cfg.get(
            "mediapipe_model_path", "models/pose_landmarker_heavy.task"
        )
        self.target_size = 256  # MediaPipe prefers 256x256 input
        self.last_timestamp_ms = 0

    def load_model(self, cfg: dict):
        """Load MediaPipe Pose Landmarker model."""
        print(f"Loading MediaPipe Pose Landmarker model: {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,  # We're tracking a single skier in the ROI
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, roi_bbox: np.ndarray) -> np.ndarray | None:
        """
        Detect pose keypoints within the given ROI.

        Args:
            frame: Original video frame
            roi_bbox: Bounding box [x1, y1, x2, y2] of the person

        Returns:
            Keypoints array of shape (33, 3) with (x, y, conf) in frame coordinates,
            or None if no pose detected
        """
        if self.detector is None:
            raise RuntimeError("MediaPipe Pose Landmarker not initialized")

        # Step 1: Crop ROI from frame
        roi, transform_info = self._crop_roi(frame, roi_bbox)

        # Step 2: Convert to square by padding
        square_roi, padding_info = self._make_square(roi)

        # Step 3: Resize to 256x256
        resized = cv2.resize(square_roi, (self.target_size, self.target_size))

        # Step 4: Run inference
        keypoints = self._detect_keypoints(resized)

        if keypoints is None:
            return None

        # Step 5: Transform keypoints back to frame coordinates
        keypoints = self._transform_keypoints_to_frame(
            keypoints, transform_info, padding_info, square_roi.shape
        )

        return keypoints

    def _crop_roi(self, frame: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Crop the ROI from frame based on bounding box.
        Returns: (roi_image, transform_info)
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # Ensure bounds are within frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop the ROI
        roi = frame[y1:y2, x1:x2]

        transform_info = {
            "crop_x1": x1,
            "crop_y1": y1,
            "roi_w": x2 - x1,
            "roi_h": y2 - y1,
        }

        return roi, transform_info

    def _make_square(self, roi: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Convert ROI to square by adding padding.
        Returns: (square_image, padding_info)
        """
        h, w = roi.shape[:2]

        if h == w:
            # Already square
            return roi, {"pad_top": 0, "pad_left": 0, "square_size": h}

        # Determine padding needed
        square_size = max(h, w)
        pad_h = square_size - h
        pad_w = square_size - w

        # Add padding evenly on both sides
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding (black borders)
        padded = cv2.copyMakeBorder(
            roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        padding_info = {
            "pad_top": pad_top,
            "pad_left": pad_left,
            "square_size": square_size,
            "original_h": h,
            "original_w": w,
        }

        return padded, padding_info

    def _detect_keypoints(self, image: np.ndarray) -> np.ndarray | None:
        """
        Run MediaPipe pose detection on the image.
        Returns: keypoints array of shape (33, 3) with (x, y, conf) or None
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Increment timestamp for video mode
        self.last_timestamp_ms += 33  # Assuming ~30fps

        # Run detection
        detection_result = self.detector.detect_for_video(
            mp_image, self.last_timestamp_ms
        )

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]
            keypoints = []

            for landmark in landmarks:
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * self.target_size
                y = landmark.y * self.target_size
                # Use visibility as confidence
                conf = landmark.visibility if hasattr(landmark, "visibility") else 1.0
                keypoints.append([x, y, conf])

            return np.array(keypoints)

        return None

    def _transform_keypoints_to_frame(
        self,
        keypoints: np.ndarray,
        transform_info: dict,
        padding_info: dict,
        square_shape: tuple[int, int],
    ) -> np.ndarray:
        """Transform keypoints from model space to original frame space."""
        # Copy keypoints to avoid modifying original
        kpts = keypoints.copy()

        # Step 1: Scale from 256x256 to square ROI size
        square_size = padding_info["square_size"]
        scale_factor = square_size / self.target_size
        kpts[:, 0] *= scale_factor
        kpts[:, 1] *= scale_factor

        # Step 2: Remove padding offset
        kpts[:, 0] -= padding_info["pad_left"]
        kpts[:, 1] -= padding_info["pad_top"]

        # Step 3: Translate to frame coordinates
        kpts[:, 0] += transform_info["crop_x1"]
        kpts[:, 1] += transform_info["crop_y1"]

        return kpts
