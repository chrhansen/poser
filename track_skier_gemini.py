#!/usr/bin/env python3
"""
YOLOv11-based Skier Detection and Tracking Script

This script uses YOLOv11 to detect and track skiers in video files,
drawing persistent bounding boxes around them with unique tracker IDs.
"""

import argparse
from pathlib import Path
import supervision as sv
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track skiers in video using YOLOv11"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Full path to the input video file"
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.is_file():
        print(f"Error: Source file '{source_path}' does not exist or is not a file.")
        return

    target_path = source_path.with_stem(f"{source_path.stem}_with_box")

    print("Loading YOLOv11 model...")
    model = YOLO('yolov11n.pt')

    tracker = sv.ByteTrack()

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path(video_path=str(source_path))

    print(f"Processing video: {source_path}")
    print(f"Output will be saved to: {target_path}")

    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame in sv.get_video_frames_generator(source_path=str(source_path)):
            results = model(frame, verbose=False)[0]

            detections = sv.Detections.from_ultralytics(results)

            detections = detections[detections.class_id == 0]

            tracked_detections = tracker.update_with_detections(detections)

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

            sink.write_frame(frame=annotated_frame)

    print(f"\nProcessing complete! Output saved to: {target_path}")


if __name__ == "__main__":
    main()
