#!/usr/bin/env python3
"""
Track-and-Pose: Modular skier analysis pipeline.
Orchestrates object detection/tracking and pose estimation stages.
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import cv2
import supervision as sv
from typing import List, Optional, Dict, Tuple

from detect_objects import ObjectDetector, Detection
from detect_pose import PoseDetector
from utils.visual import draw_bbox


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Track and analyze skiers with object detection and pose estimation"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--detect",
        type=str,
        default="objects,pose",
        help="Which stages to activate (comma-separated: objects,pose)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "ocsort"],
        help="Tracker backend to use"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file path"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_main_track(
    detections: List[Detection],
    track_lengths: Dict[int, int],
    last_main_id: Optional[int] = None,
    gap_count: int = 0,
    gap_max: int = 10
) -> Tuple[Optional[int], int]:
    """
    Select the main track ID based on longest track history.
    
    Returns:
        Tuple of (main_id, updated_gap_count)
    """
    if not detections or not track_lengths:
        if last_main_id is not None:
            gap_count += 1
            if gap_count < gap_max:
                return last_main_id, gap_count
        return None, 0
    
    # Get current track IDs
    current_ids = [det[1] for det in detections]
    
    # If last main ID is still present, keep it
    if last_main_id in current_ids:
        return last_main_id, 0
    
    # Otherwise, pick the longest-lived track that's currently visible
    visible_tracks = {tid: length for tid, length in track_lengths.items() 
                     if tid in current_ids}
    
    if visible_tracks:
        main_id = max(visible_tracks.items(), key=lambda x: x[1])[0]
        return main_id, 0
    
    return None, gap_count + 1


def main():
    """Main pipeline orchestration."""
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Parse detection stages
    stages = [s.strip() for s in args.detect.split(',')]
    detect_objects = 'objects' in stages
    detect_pose = 'pose' in stages
    
    # Print system information
    print("\n=== System Information ===")
    import platform
    import subprocess
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    # Get more detailed chip info on macOS
    if platform.system() == "Darwin":
        try:
            chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
            print(f"Processor: {chip_info}")
        except:
            print(f"Processor: {platform.processor()}")
    
    # Check PyTorch device availability
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA available: No")
    
    if torch.backends.mps.is_available():
        print(f"Apple Silicon MPS available: Yes")
    else:
        print(f"Apple Silicon MPS available: No")
    
    # Determine which device will be used
    device_config = cfg.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            actual_device = 'cuda'
        elif torch.backends.mps.is_available():
            actual_device = 'mps'
        else:
            actual_device = 'cpu'
    else:
        actual_device = device_config
    
    print(f"\nDevice configuration: {device_config}")
    print(f"Will use device: {actual_device}")
    
    if actual_device == 'cpu':
        print("WARNING: Running on CPU - this will be slower than GPU")
    elif actual_device == 'mps':
        print("Running on Apple Silicon GPU (Metal Performance Shaders)")
    elif actual_device == 'cuda':
        print(f"Running on NVIDIA GPU")
    
    print("========================\n")
    
    # Check source file
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file '{source_path}' does not exist.")
        return
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize detectors
    obj_detector = None
    pose_detector = None
    
    if detect_objects:
        obj_detector = ObjectDetector(cfg)
        obj_detector.load_model(cfg)
        obj_detector.init_tracker(cfg, args.tracker)
    
    if detect_pose:
        pose_detector = PoseDetector(cfg)
        pose_detector.load_model(cfg)
    
    # Setup video info
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    fps = video_info.fps if cfg.get('output_fps') is None else cfg['output_fps']
    
    # Prepare output paths
    output_paths = []
    if detect_objects:
        bbox_path = save_dir / f"{source_path.stem}_with_box{source_path.suffix}"
        output_paths.append(('bbox', bbox_path))
    
    if detect_pose:
        pose_path = save_dir / f"{source_path.stem}_with_pose{source_path.suffix}"
        output_paths.append(('pose', pose_path))
    
    print(f"Processing video: {source_path}")
    print(f"Active stages: {', '.join(stages)}")
    print(f"Output directory: {save_dir}")
    
    # Process video
    cap = cv2.VideoCapture(str(source_path))
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writers
    writers = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for stage, path in output_paths:
        writers[stage] = cv2.VideoWriter(
            str(path), fourcc, fps, (frame_width, frame_height)
        )
    
    # Tracking state
    main_track_id = None
    gap_count = 0
    frame_idx = 0
    
    # Annotators
    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Object detection stage
            detections = []
            if detect_objects:
                detections = obj_detector.run(frame)
            
            # Select main track
            main_track_id, gap_count = select_main_track(
                detections, 
                obj_detector.track_lengths if obj_detector else {},
                main_track_id, 
                gap_count
            )
            
            # Create annotated frames
            bbox_frame = frame.copy() if detect_objects else None
            pose_frame = frame.copy() if detect_pose else None
            
            # Draw bounding boxes if needed
            if detect_objects and detections:
                # Convert to supervision format for drawing
                bboxes = np.array([d[0] for d in detections])
                track_ids = np.array([d[1] for d in detections])
                confidences = np.array([d[2] for d in detections])
                
                sv_detections = sv.Detections(
                    xyxy=bboxes,
                    confidence=confidences,
                    class_id=np.zeros(len(bboxes), dtype=int),
                    tracker_id=track_ids
                )
                
                labels = [f"ID: {tid}" for tid in track_ids]
                
                bbox_frame = bbox_annotator.annotate(
                    scene=bbox_frame,
                    detections=sv_detections
                )
                bbox_frame = label_annotator.annotate(
                    scene=bbox_frame,
                    detections=sv_detections,
                    labels=labels
                )
            
            # Pose estimation stage
            if detect_pose:
                if main_track_id is not None and detections:
                    # Find the main track's bbox
                    main_bbox = None
                    for bbox, track_id, conf in detections:
                        if track_id == main_track_id:
                            main_bbox = bbox
                            break
                    
                    if main_bbox is not None:
                        # Run pose detection
                        dt = 1.0 / fps
                        try:
                            pose_frame = pose_detector.run(pose_frame, main_bbox, dt)
                        except Exception as e:
                            print(f"Pose detection error on frame {frame_idx}: {e}")
                        
                        # Also draw the main track's bbox on pose frame
                        draw_bbox(pose_frame, main_bbox, label=f"ID: {main_track_id}")
            
            # Write frames
            if 'bbox' in writers and bbox_frame is not None:
                writers['bbox'].write(bbox_frame)
            
            if 'pose' in writers and pose_frame is not None:
                writers['pose'].write(pose_frame)
            
            # Show preview if requested
            if args.show:
                preview = pose_frame if detect_pose else bbox_frame
                if preview is not None:
                    cv2.imshow("Track and Pose", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Progress
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    finally:
        # Cleanup
        cap.release()
        for writer in writers.values():
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    
    print("\nProcessing complete!")
    for stage, path in output_paths:
        print(f"  {stage.capitalize()} output: {path}")


if __name__ == "__main__":
    main()