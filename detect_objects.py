#!/usr/bin/env python3
"""
Object detection and tracking module using YOLOv11 and BoT-SORT/OC-SORT.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import BotSort, OcSort
import supervision as sv


Detection = Tuple[np.ndarray, int, float]  # (bbox_xyxy, track_id, conf)


class ObjectDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.tracker = None
        self.device = None
        self.track_lengths: Dict[int, int] = {}  # track_id -> frame_count
        
    def load_model(self, cfg: dict) -> YOLO:
        """Instantiate YOLO v11 with weights from config."""
        model_path = cfg.get('object_model', 'yolo11m.pt')
        print(f"Loading YOLO model: {model_path}")
        model = YOLO(model_path)
        
        # Detect device
        device_str = cfg.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device_str
        
        self.model = model
        return model
    
    def init_tracker(self, cfg: dict, tracker_type: str = 'botsort') -> object:
        """Create BoT-SORT / OC-SORT instance with config parameters."""
        tracker_cfg = cfg.get('tracker', {})
        
        if tracker_type == 'botsort':
            self.tracker = BotSort(
                reid_weights=Path('osnet_x0_25_msmt17.pt'),
                device=self.device,
                half=False,
                track_high_thresh=tracker_cfg.get('track_high_thresh', 0.6),
                track_low_thresh=tracker_cfg.get('track_low_thresh', 0.1),
                new_track_thresh=tracker_cfg.get('new_track_thresh', 0.7),
                track_buffer=30,
                match_thresh=0.8,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=True
            )
            print("Using BotSort tracker")
        else:  # ocsort
            self.tracker = OcSort(
                det_thresh=tracker_cfg.get('new_track_thresh', 0.7),
                max_age=30,
                min_hits=3,
                asso_threshold=0.3,
                delta_t=3,
                asso_func="iou",
                inertia=0.2,
                use_byte=False
            )
            print("Using OcSort tracker")
            
        return self.tracker
    
    def run(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection and tracking on a frame.
        Returns list of (bbox_xyxy, track_id, conf) tuples.
        """
        if self.model is None or self.tracker is None:
            raise RuntimeError("Model or tracker not initialized")
        
        # Run YOLO detection
        results = self.model(frame, device=self.device)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for person class (class_id == 0)
        detections = detections[detections.class_id == 0]
        
        # Convert to tracker format
        if len(detections) > 0:
            dets = []
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = detections.confidence[i] if detections.confidence is not None else 1.0
                dets.append([x1, y1, x2, y2, conf, 0])  # 0 is class_id for person
            dets = np.array(dets)
        else:
            dets = np.empty((0, 6))
        
        # Update tracker
        outputs = self.tracker.update(dets, frame)
        
        # Convert to Detection format and update track lengths
        detection_list = []
        if len(outputs) > 0:
            for i in range(len(outputs)):
                bbox = outputs[i, :4]
                track_id = int(outputs[i, 4])
                conf = outputs[i, 5] if outputs.shape[1] > 5 else 1.0
                
                # Update track lengths
                if track_id not in self.track_lengths:
                    self.track_lengths[track_id] = 0
                self.track_lengths[track_id] += 1
                
                detection_list.append((bbox, track_id, conf))
        
        return detection_list
    
    def get_longest_track_id(self) -> Optional[int]:
        """Return the ID of the longest-lived track."""
        if not self.track_lengths:
            return None
        return max(self.track_lengths.items(), key=lambda x: x[1])[0]