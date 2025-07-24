#!/usr/bin/env python3
"""
YOLO-Pose detector implementation.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import cv2
from ultralytics import YOLO

from .pose_detector_base import PoseDetectorBase
from utils.geometry import pad_bbox_to_square


class YOLOPoseDetector(PoseDetectorBase):
    """YOLO-Pose detector implementation."""
    
    def __init__(self, cfg: dict):
        """Initialize YOLO-Pose detector."""
        super().__init__(cfg)
        self.model_path = cfg.get('pose_model', 'models/yolo11x-pose.pt')
        self.target_size = 768  # Optimal size for larger YOLO models
        
    def load_model(self, cfg: dict):
        """Load YOLO-Pose model."""
        print(f"Loading YOLO-Pose model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
    def detect(self, frame: np.ndarray, roi_bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose keypoints within the given ROI.
        
        Args:
            frame: Original video frame
            roi_bbox: Bounding box [x1, y1, x2, y2] of the person
            
        Returns:
            Keypoints array of shape (N, 3) with (x, y, conf) in frame coordinates,
            or None if no pose detected
        """
        if self.model is None:
            raise RuntimeError("YOLO-Pose model not initialized")
            
        # Step 1: Crop and pad ROI to square
        roi, transform_info = self._crop_and_pad_roi(frame, roi_bbox)
        
        # Step 2: Resize for model
        resized, scale_info = self._resize_for_model(roi)
        
        # Step 3: Run inference
        keypoints = self._detect_keypoints(resized)
        
        if keypoints is None:
            return None
            
        # Step 4: Transform keypoints back to frame coordinates
        keypoints = self._transform_keypoints_to_frame(
            keypoints, transform_info, scale_info, resized.shape
        )
        
        return keypoints
        
    def _crop_and_pad_roi(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Crop and pad bbox to square ROI.
        Returns: (roi_image, transform_info)
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Pad the bbox to square
        x1_pad, y1_pad, x2_pad, y2_pad = pad_bbox_to_square(
            x1, y1, x2, y2, w, h, self.pad_ratio
        )
        
        # Crop the ROI
        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        transform_info = {
            'crop_x1': x1_pad,
            'crop_y1': y1_pad,
            'roi_w': x2_pad - x1_pad,
            'roi_h': y2_pad - y1_pad,
            'original_bbox': bbox
        }
        
        return roi, transform_info
        
    def _resize_for_model(self, roi: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Resize ROI for YOLO model input.
        Returns: (resized_image, scale_info)
        """
        h, w = roi.shape[:2]
        max_dim = max(h, w)
        scale = min(self.target_size / max_dim, 1.0)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Make divisible by 32 (YOLO requirement)
        new_h = max(32, (new_h // 32) * 32)
        new_w = max(32, (new_w // 32) * 32)
        
        resized = cv2.resize(roi, (new_w, new_h))
        
        scale_info = {
            'scale': scale,
            'resized_w': new_w,
            'resized_h': new_h,
            'original_w': w,
            'original_h': h
        }
        
        return resized, scale_info
        
    def _detect_keypoints(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run YOLO pose detection on the image.
        Returns: keypoints array of shape (N, 3) with (x, y, conf) or None
        """
        # Run inference with specific parameters
        results = self.model(image, imgsz=self.target_size, conf=0.2, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kpts_data = result.keypoints.data
                if kpts_data.shape[0] > 0:  # At least one person detected
                    # Get first person's keypoints
                    kpts = kpts_data[0].cpu().numpy()  # Shape: (17, 3)
                    # Filter out keypoints with 0 confidence
                    valid_kpts = kpts[kpts[:, 2] > 0]
                    if len(valid_kpts) > 0:
                        return kpts
        return None
        
    def _transform_keypoints_to_frame(
        self, 
        keypoints: np.ndarray, 
        transform_info: Dict,
        scale_info: Dict,
        resized_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Transform keypoints from model space to original frame space."""
        # Copy keypoints to avoid modifying original
        kpts = keypoints.copy()
        
        # Scale from resized image to ROI
        resize_h, resize_w = resized_shape[:2]
        roi_w = transform_info['roi_w']
        roi_h = transform_info['roi_h']
        
        # Avoid division by zero
        if resize_w > 0 and resize_h > 0:
            kpts[:, 0] *= (roi_w / resize_w)
            kpts[:, 1] *= (roi_h / resize_h)
        
        # Translate to frame coordinates
        kpts[:, 0] += transform_info['crop_x1']
        kpts[:, 1] += transform_info['crop_y1']
        
        return kpts