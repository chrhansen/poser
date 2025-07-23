#!/usr/bin/env python3
"""
Pose estimation module using YOLO-Pose or MediaPipe BlazePose.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp


class PoseDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.model_type = None
        self.pad_ratio = cfg.get('pad_ratio', 0.5)
        self.conf_min = cfg.get('conf_min', 0.2)
        self.smoothing_cfg = cfg.get('smoothing', {})
        self.smoothers = {}  # joint_id -> smoother instance
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        
    def load_model(self, cfg: dict):
        """Load either YOLO-Pose or MediaPipe BlazePose per config."""
        pose_model = cfg.get('pose_model', 'yolov8m-pose.pt')
        
        if pose_model == 'mediapipe':
            self.model_type = 'mediapipe'
            print("Loading MediaPipe BlazePose model")
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.model_type = 'yolo'
            print(f"Loading YOLO-Pose model: {pose_model}")
            self.model = YOLO(pose_model)
            
    def set_roi_padding(self, p: float):
        """Configure padding ratio (e.g. 0.50)."""
        self.pad_ratio = p
        
    def _init_smoother(self, joint_id: int):
        """Initialize smoother for a specific joint."""
        from utils.smoothing import create_smoother
        self.smoothers[joint_id] = create_smoother(self.smoothing_cfg)
        
    def _crop_and_pad_roi(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Crop and pad bbox to square ROI.
        Returns: (roi_image, transform_info)
        """
        from utils.geometry import pad_bbox_to_square
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Pad the bbox
        x1_pad, y1_pad, x2_pad, y2_pad = pad_bbox_to_square(
            x1, y1, x2, y2, w, h, self.pad_ratio
        )
        
        # Crop the ROI
        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        transform_info = {
            'crop_x1': x1_pad,
            'crop_y1': y1_pad,
            'roi_w': x2_pad - x1_pad,
            'roi_h': y2_pad - y1_pad
        }
        
        return roi, transform_info
        
    def _resize_for_model(self, roi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize ROI for model input.
        Returns: (resized_image, scale_factor)
        """
        if self.model_type == 'mediapipe':
            # MediaPipe expects 256x256
            resized = cv2.resize(roi, (256, 256))
            scale = 256.0 / max(roi.shape[:2])
        else:  # YOLO
            # For larger models (x-size), use 768px for better limb detection
            target_size = 768
            h, w = roi.shape[:2]
            max_dim = max(h, w)
            scale = min(target_size / max_dim, 1.0)
            new_h = int(h * scale)
            new_w = int(w * scale)
            # Make divisible by 32
            new_h = max(32, (new_h // 32) * 32)
            new_w = max(32, (new_w // 32) * 32)
            resized = cv2.resize(roi, (new_w, new_h))
            
        return resized, scale
        
    def _detect_keypoints(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run pose detection on the image.
        Returns: keypoints array of shape (N, 3) with (x, y, conf) or None
        """
        if self.model_type == 'mediapipe':
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model.process(image_rgb)
            
            if results.pose_landmarks:
                h, w = image.shape[:2]
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    conf = landmark.visibility
                    keypoints.append([x, y, conf])
                return np.array(keypoints)
            return None
        else:  # YOLO
            # Pass imgsz and conf parameters for better detection
            results = self.model(image, imgsz=768, conf=0.2, verbose=False)
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
        resize_scale: float,
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
        
    def _smooth_keypoints(self, keypoints: np.ndarray, dt: float) -> np.ndarray:
        """Apply smoothing to keypoints."""
        smoothed = keypoints.copy()
        
        for i in range(len(keypoints)):
            if keypoints[i, 2] < self.conf_min or np.isnan(keypoints[i, 0]) or np.isnan(keypoints[i, 1]):
                continue
                
            if i not in self.smoothers:
                self._init_smoother(i)
                
            # Smooth x and y coordinates
            smoothed[i, 0] = self.smoothers[i].filter(keypoints[i, 0], dt, 0)
            smoothed[i, 1] = self.smoothers[i].filter(keypoints[i, 1], dt, 1)
            
        return smoothed
        
    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw skeleton on frame (in-place)."""
        from utils.visual import draw_skeleton
        draw_skeleton(frame, keypoints, self.model_type, self.conf_min)
        
    def run(
        self, 
        frame: np.ndarray, 
        bbox: Optional[np.ndarray], 
        dt: float = 1/30.0
    ) -> np.ndarray:
        """
        Run pose detection on the given bbox region.
        - Crop + pad → square ROI
        - Resize for model (256x256 BlazePose or ≤ 640 YOLO-Pose)
        - Inference → keypoints
        - Smooth
        - Draw skeleton on frame (in-place)
        - Return updated frame
        """
        if self.model is None:
            raise RuntimeError("Pose model not initialized")
            
        if bbox is None:
            # No bbox provided, return unchanged frame
            return frame
            
        # Crop and pad ROI
        roi, transform_info = self._crop_and_pad_roi(frame, bbox)
        
        # Resize for model
        resized, scale = self._resize_for_model(roi)
        
        # Detect keypoints
        keypoints = self._detect_keypoints(resized)
        
        if keypoints is not None and len(keypoints) > 0:
            # Transform keypoints to frame coordinates
            keypoints = self._transform_keypoints_to_frame(
                keypoints, transform_info, scale, resized.shape
            )
            
            # Smooth keypoints (skip if smoothing is disabled)
            if self.smoothing_cfg.get('kind', 'one_euro') != 'none':
                keypoints = self._smooth_keypoints(keypoints, dt)
            
            # Draw skeleton
            self._draw_skeleton(frame, keypoints)
            
        return frame