#!/usr/bin/env python3
"""
YOLOv11-based Skier Detection and Tracking Script with BoT-SORT

This script uses YOLOv11 to detect and BoT-SORT to track skiers in video files,
which may provide better tracking for fast-moving objects like skiers.
"""

import argparse
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO
import torch
from boxmot import BotSort, OcSort


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track skiers in video using YOLOv11 with BoT-SORT/OC-SORT"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Full path to the input video file"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "ocsort"],
        help="Tracker to use (default: botsort)"
    )
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=0.6,
        help="High confidence threshold for tracking (default: 0.6)"
    )
    parser.add_argument(
        "--track-low-thresh",
        type=float,
        default=0.1,
        help="Low confidence threshold for tracking (default: 0.1)"
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.7,
        help="Threshold for creating new tracks (default: 0.7)"
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file '{source_path}' does not exist.")
        return

    target_path = source_path.with_stem(f"{source_path.stem}_with_box")

    print(f"Loading YOLOv11 model (will auto-download if needed)...")
    model = YOLO('yolo11m.pt')  # Using medium model for better accuracy

    # Check if GPU is available (CUDA or MPS)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize tracker
    if args.tracker == "botsort":
        tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device=device,
            half=False,  # Disable half precision for compatibility
            track_high_thresh=args.track_high_thresh,
            track_low_thresh=args.track_low_thresh,
            new_track_thresh=args.new_track_thresh,
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=True
        )
        print(f"Using BotSort tracker")
    else:
        tracker = OcSort(
            det_thresh=args.new_track_thresh,
            max_age=30,
            min_hits=3,
            asso_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False
        )
        print(f"Using OcSort tracker")

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(str(source_path))

    print(f"Processing video: {source_path}")
    print(f"Output will be saved to: {target_path}")

    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_path=str(source_path))):
            results = model(frame, device=device)[0]

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
            outputs = tracker.update(dets, frame)

            if len(outputs) > 0:
                # Convert back to supervision format
                tracked_boxes = outputs[:, :4]
                tracked_ids = outputs[:, 4].astype(int)
                tracked_confidences = outputs[:, 5] if outputs.shape[1] > 5 else np.ones(len(outputs))

                tracked_detections = sv.Detections(
                    xyxy=tracked_boxes,
                    confidence=tracked_confidences,
                    class_id=np.zeros(len(tracked_boxes), dtype=int),
                    tracker_id=tracked_ids
                )

                labels = [f"ID: {tracker_id}" for tracker_id in tracked_detections.tracker_id]

                annotated_frame = bounding_box_annotator.annotate(
                    scene=frame.copy(),
                    detections=tracked_detections
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                    labels=labels
                )

                sink.write_frame(annotated_frame)
            else:
                sink.write_frame(frame)

    print(f"\nProcessing complete! Output saved to: {target_path}")


if __name__ == "__main__":
    main()
