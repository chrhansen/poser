#!/usr/bin/env python3
"""
Track-and-Pose: Modular skier analysis pipeline.
Orchestrates object detection/tracking and pose estimation stages.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

from src.detection.detect_objects import Detection, ObjectDetector
from src.filters.kalman_zero_lag import KalmanRTS
from src.filters.kalman_2d import Kalman2D
from src.metrics.calculator import PoseMetricsCalculator
from src.metrics.storage import MetricsLogger
from src.pose.detect_pose import PoseDetector
from src.projection import (
    estimate_camera_extrinsics,
    initialize_intrinsics,
    project_points_batch,
)
from src.visualization.plotter import MetricsPlotter
from utils.visual import draw_bbox, draw_skeleton
from src.visualization.render_world_debug import render_world_skeletons


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Track and analyze skiers with object detection and pose estimation"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Input video file path"
    )
    parser.add_argument(
        "--detect",
        type=str,
        default="objects,pose",
        help="Which stages to activate (comma-separated: objects,pose)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="output", help="Output directory for all files"
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable live preview window"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file path",
    )
    parser.add_argument(
        "--pose-detector",
        type=str,
        default="mediapipe",
        choices=["yolo", "mediapipe"],
        help="Pose detection engine to use (default: mediapipe)",
    )
    parser.add_argument(
        "--realtime-plot",
        action="store_true",
        help="Show real-time plot of shin angles",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=None,
        help="Override Kalman process noise variance (m/s^2)^2",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=None,
        help="Override Kalman measurement noise variance (m^2)",
    )
    # Deprecated: --metrics-output is no longer used, all outputs go to --save_dir
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_video_with_smoothing(
    source_path: Path,
    save_dir: Path,
    cfg: dict,
    obj_detector: ObjectDetector | None,
    pose_detector: PoseDetector | None,
    args,
) -> None:
    """
    Process video with three-pass Kalman RTS smoothing.

    Pass 1: Detection and buffering
    Pass 2: Calibration and smoothing
    Pass 3: Projection and rendering
    """
    # Setup video info
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    fps = video_info.fps if cfg.get("output_fps") is None else cfg["output_fps"]

    # Open video
    cap = cv2.VideoCapture(str(source_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize metrics components (always enabled)
    metrics_calculator = None
    metrics_logger = None
    if pose_detector:
        metrics_calculator = PoseMetricsCalculator(detector_type=args.pose_detector, window_size=1)
        metrics_logger = MetricsLogger(str(source_path), output_dir=str(save_dir))

    print("\n=== Pass 1: Detection and Buffering ===")

    # Buffers for storing data
    all_frames = []
    all_detections = []
    all_landmarks_2d = []
    all_landmarks_3d = []
    all_visibility = []
    timestamps = []
    main_track_ids = []

    # Tracking state
    main_track_id = None
    gap_count = 0
    frame_idx = 0

    # Pass 1: Read and detect
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            all_frames.append(frame.copy())
            timestamps.append(frame_idx / fps)

            # Object detection
            detections = []
            if obj_detector:
                detections = obj_detector.run(frame)
            all_detections.append(detections)

            # Select main track
            main_track_id, gap_count = select_main_track(
                detections,
                obj_detector.track_lengths if obj_detector else {},
                main_track_id,
                gap_count,
            )
            main_track_ids.append(main_track_id)

            # Pose detection for buffering
            if pose_detector and main_track_id is not None and detections:
                # Find the main track's bbox
                main_bbox = None
                for bbox, track_id, _conf in detections:
                    if track_id == main_track_id:
                        main_bbox = bbox
                        break

                if main_bbox is not None:
                    # Get 2D landmarks
                    _, keypoints_2d = pose_detector.run(
                        frame, main_bbox, 1.0/fps, return_keypoints=True
                    )

                    if keypoints_2d is not None:
                        if args.pose_detector == "mediapipe":
                            world_landmarks = pose_detector.get_world_landmarks(frame, main_bbox)
                            if world_landmarks is not None:
                                all_landmarks_3d.append(world_landmarks[:, :3])
                                vis = world_landmarks[:, 3]
                            else:
                                all_landmarks_3d.append(np.zeros((33, 3)))
                                vis = np.zeros(33)
                        else:
                            all_landmarks_3d.append(np.zeros((33, 3)))
                            vis = keypoints_2d[:, 3]
                        all_landmarks_2d.append(keypoints_2d[:, :2])
                        all_visibility.append(vis)
                    else:
                        # No pose detected
                        all_landmarks_2d.append(np.zeros((33, 2)))
                        all_landmarks_3d.append(np.zeros((33, 3)))
                        all_visibility.append(np.zeros(33))
                else:
                    # No main bbox
                    all_landmarks_2d.append(np.zeros((33, 2)))
                    all_landmarks_3d.append(np.zeros((33, 3)))
                    all_visibility.append(np.zeros(33))
            else:
                # No pose detector or main track
                all_landmarks_2d.append(np.zeros((33, 2)))
                all_landmarks_3d.append(np.zeros((33, 3)))
                all_visibility.append(np.zeros(33))

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Pass 1 Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    finally:
        cap.release()

    # Convert to numpy arrays
    all_landmarks_2d = np.array(all_landmarks_2d)  # (T, N, 2)
    all_landmarks_3d = np.array(all_landmarks_3d)  # (T, N, 3)
    all_visibility = np.array(all_visibility)       # (T, N)
    timestamps = np.array(timestamps)

    # Persist raw landmark data for offline smoothing/analysis
    np.save(save_dir / "raw_landmarks_3d.npy", all_landmarks_3d)
    np.save(save_dir / "visibility.npy", all_visibility)

    # Also store as CSV for easy inspection
    np.savetxt(save_dir / "raw_landmarks_3d.csv", all_landmarks_3d.reshape(all_landmarks_3d.shape[0], -1), delimiter=",")
    np.savetxt(save_dir / "visibility.csv", all_visibility, delimiter=",")

    # Compute time deltas
    dt_seq = np.diff(timestamps)
    dt_seq = np.concatenate([dt_seq, [dt_seq[-1] if len(dt_seq) > 0 else 1.0/fps]])

    # Save dt sequence for offline smoother
    np.save(save_dir / "dt_seq.npy", dt_seq)

    print("\n=== Pass 2: Calibration and Smoothing ===")

    # Initialize outputs
    smooth_landmarks_3d = all_landmarks_3d.copy()
    smooth_landmarks_2d = all_landmarks_2d.copy()

    if args.smooth in ["kalman_rts", "kalman_2d"]:
        # Initialize camera intrinsics
        K = initialize_intrinsics(frame_width, frame_height)

        # Get smoothing config
        smoothing_cfg = cfg.get("smoothing_cfg", {}).copy()
        if args.process_noise is not None:
            smoothing_cfg["process_noise"] = args.process_noise
        if args.measurement_noise is not None:
            smoothing_cfg["measurement_noise"] = args.measurement_noise

        if args.smooth == "kalman_2d":
            # Direct 2D smoothing - simpler and more stable
            print("Smoothing 2D landmarks directly...")
            kalman_2d = Kalman2D(smoothing_cfg, dt_seq)
            smooth_landmarks_2d = kalman_2d.batch_smooth_all(all_landmarks_2d, all_visibility)

            # Save intermediate data if requested
            if cfg.get("save_intermediate", False):
                np.save(save_dir / "raw_landmarks_2d.npy", all_landmarks_2d)
                np.save(save_dir / "smooth_landmarks_2d.npy", smooth_landmarks_2d)

        elif args.smooth == "kalman_rts" and args.pose_detector == "mediapipe":
            # 3D smoothing with projection (original approach)
            projection_cfg = cfg.get("projection_cfg", {})

            # Initialize Kalman smoother
            kalman = KalmanRTS(smoothing_cfg, dt_seq)

            # Smooth 3D landmarks
            print("Smoothing 3D landmarks...")
            smooth_landmarks_3d = kalman.batch_smooth_all(all_landmarks_3d, all_visibility)

            # Render debug world skeletons
            debug_path = save_dir / "world_debug.mp4"
            render_world_skeletons(all_landmarks_3d, smooth_landmarks_3d, debug_path, fps)
            inter_path = save_dir / "world_debug.html"
            try:
                from src.visualization.render_world_interactive import render_world_interactive
                render_world_interactive(all_landmarks_3d, smooth_landmarks_3d, inter_path, fps)
            except Exception as e:
                print(f"Interactive debug generation failed: {e}")

            # Save intermediate data if requested
            if cfg.get("save_intermediate", False):
                np.save(save_dir / "raw_landmarks_3d.npy", all_landmarks_3d)
                np.save(save_dir / "smooth_landmarks_3d.npy", smooth_landmarks_3d)

            # Estimate camera extrinsics
            print("Estimating camera poses...")
            R_seq, t_seq = estimate_camera_extrinsics(
                all_landmarks_2d, all_landmarks_3d, all_visibility, K, projection_cfg
            )

            # Save camera poses if requested
            if cfg.get("save_intermediate", False):
                np.save(save_dir / "camera_R_seq.npy", R_seq)
                np.save(save_dir / "camera_t_seq.npy", t_seq)

            # Project smoothed 3D landmarks to 2D
            print("Projecting smoothed landmarks to 2D...")
            smooth_landmarks_2d = project_points_batch(smooth_landmarks_3d, R_seq, t_seq, K)

    print("\n=== Pass 3: Rendering Output ===")

    # Setup output
    output_path = save_dir / f"{source_path.stem}_with_pose{source_path.suffix}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # Annotators
    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Render frames
    for frame_idx, (frame, detections) in enumerate(zip(all_frames, all_detections)):
        output_frame = frame.copy()

        # Draw bounding boxes if object detection was enabled
        if obj_detector and detections:
            bboxes = np.array([d[0] for d in detections])
            track_ids = np.array([d[1] for d in detections])
            confidences = np.array([d[2] for d in detections])

            sv_detections = sv.Detections(
                xyxy=bboxes,
                confidence=confidences,
                class_id=np.zeros(len(bboxes), dtype=int),
                tracker_id=track_ids,
            )

            labels = [f"ID: {tid}" for tid in track_ids]

            output_frame = bbox_annotator.annotate(
                scene=output_frame, detections=sv_detections
            )
            output_frame = label_annotator.annotate(
                scene=output_frame, detections=sv_detections, labels=labels
            )

        # Draw skeleton
        if pose_detector and main_track_ids[frame_idx] is not None:
            # Prepare keypoints for drawing
            if args.smooth in ["kalman_rts", "kalman_2d"]:
                # Use smoothed 2D landmarks
                keypoints = np.zeros((33, 4))
                keypoints[:, :2] = smooth_landmarks_2d[frame_idx]
                keypoints[:, 3] = all_visibility[frame_idx]
            else:
                # Use raw 2D landmarks
                keypoints = np.zeros((33, 4))
                keypoints[:, :2] = all_landmarks_2d[frame_idx]
                keypoints[:, 3] = all_visibility[frame_idx]

            # Draw skeleton
            draw_skeleton(output_frame, keypoints, args.pose_detector, pose_detector.detector.conf_min)

            # Calculate and log metrics
            if metrics_calculator and metrics_logger:
                timestamp_ms = (frame_idx / fps) * 1000

                # Use the appropriate keypoints based on smoothing
                if args.smooth in ["kalman_rts", "kalman_2d"]:
                    # For smoothed, keypoints already contains the smoothed 2D data
                    pass  # keypoints already set above
                else:
                    # For raw, keypoints already contains raw 2D data
                    pass  # keypoints already set above

                # Calculate shin angles for 2D
                angles_2d = metrics_calculator.calculate_shin_angles(keypoints, is_world_coords=False)

                # Calculate 3D angles if available
                angles_3d_raw = {"shin_angle": None}
                angles_3d_smooth = {"shin_angle": None}

                if args.pose_detector == "mediapipe" and frame_idx < len(all_landmarks_3d):
                    # Raw 3D angles
                    raw_world = all_landmarks_3d[frame_idx]
                    if not np.all(raw_world == 0):
                        raw_world_kpts = np.hstack([raw_world, all_visibility[frame_idx].reshape(-1, 1)])
                        angles_3d_raw = metrics_calculator.calculate_shin_angles(raw_world_kpts, is_world_coords=True)

                    # Smoothed 3D angles
                    if args.smooth == "kalman_rts":
                        smooth_world = smooth_landmarks_3d[frame_idx]
                        smooth_world_kpts = np.hstack([smooth_world, all_visibility[frame_idx].reshape(-1, 1)])
                        angles_3d_smooth = metrics_calculator.calculate_shin_angles(smooth_world_kpts, is_world_coords=True)

                # Log shin angles (raw world goes to shin_angle_3d_ma, smoothed to shin_angle_3d)
                metrics_logger.log_shin_angles(
                    frame_idx,
                    timestamp_ms,
                    angles_2d["shin_angle"],
                    None,  # No 2D MA needed
                    angles_3d_smooth["shin_angle"] if angles_3d_smooth["shin_angle"] is not None else angles_3d_raw["shin_angle"],
                    angles_3d_raw["shin_angle"]
                )

            # Always draw raw skeleton overlay for comparison
            if True:
                # Draw raw skeleton in red color
                raw_keypoints = np.zeros((33, 4))
                raw_keypoints[:, :2] = all_landmarks_2d[frame_idx]
                raw_keypoints[:, 3] = all_visibility[frame_idx]

                # Draw raw skeleton in red
                draw_skeleton(
                    output_frame,
                    raw_keypoints,
                    args.pose_detector,
                    pose_detector.detector.conf_min,
                    point_color=(0, 0, 255),  # Red points
                    line_color=(0, 100, 255)   # Lighter red lines
                )

                # Add legend
                cv2.putText(output_frame, "Raw", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(output_frame, "Smoothed", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(output_frame)

        if frame_idx % 30 == 0:
            print(f"Pass 3 Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

    writer.release()
    print(f"\nOutput saved to: {output_path}")

    # Close metrics logger
    if metrics_logger:
        metrics_logger.close()

    # Generate shin angle plot
    if metrics_logger and pose_detector:
        print("\nGenerating angles graph...")
        plotter = MetricsPlotter()
        graph_path = save_dir / f"{source_path.stem}_angles_graph.png"
        plotter.generate_offline_graph(str(metrics_logger.angles_file), str(graph_path))


def select_main_track(
    detections: list[Detection],
    track_lengths: dict[int, int],
    last_main_id: int | None = None,
    gap_count: int = 0,
    gap_max: int = 10,
) -> tuple[int | None, int]:
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
    visible_tracks = {
        tid: length for tid, length in track_lengths.items() if tid in current_ids
    }

    if visible_tracks:
        main_id = max(visible_tracks.items(), key=lambda x: x[1])[0]
        return main_id, 0

    return None, gap_count + 1


def main():
    """Main pipeline orchestration."""
    args = parse_args()

    # Force smoothing mode to kalman_rts (always-on)
    setattr(args, "smooth", "kalman_rts")

    # Load configuration
    cfg = load_config(args.config)

    # Parse detection stages
    stages = [s.strip() for s in args.detect.split(",")]
    detect_objects = "objects" in stages
    detect_pose = "pose" in stages

    # Print system information
    print("\n=== System Information ===")
    import platform
    import subprocess

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    # Get more detailed chip info on macOS
    if platform.system() == "Darwin":
        try:
            chip_info = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            print(f"Processor: {chip_info}")
        except Exception:
            print(f"Processor: {platform.processor()}")

    # Check PyTorch device availability
    import torch

    print(f"\nPyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: No")

    if torch.backends.mps.is_available():
        print("Apple Silicon MPS available: Yes")
    else:
        print("Apple Silicon MPS available: No")

    # Determine which device will be used
    device_config = cfg.get("device", "auto")
    if device_config == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif torch.backends.mps.is_available():
            actual_device = "mps"
        else:
            actual_device = "cpu"
    else:
        actual_device = device_config

    print(f"\nDevice configuration: {device_config}")
    print(f"Will use device: {actual_device}")

    if actual_device == "cpu":
        print("WARNING: Running on CPU - this will be slower than GPU")
    elif actual_device == "mps":
        print("Running on Apple Silicon GPU (Metal Performance Shaders)")
    elif actual_device == "cuda":
        print("Running on NVIDIA GPU")

    print("========================\n")

    # Check source file
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file '{source_path}' does not exist.")
        return

    # Create output directory with subfolder based on input filename
    base_output_dir = Path(args.save_dir)
    base_output_dir.mkdir(exist_ok=True)

    # Create subfolder based on source filename (replace dots with underscores)
    source_folder_name = source_path.stem.replace(
        ".", "_"
    ) + source_path.suffix.replace(".", "_")
    save_dir = base_output_dir / source_folder_name
    save_dir.mkdir(exist_ok=True)

    # Initialize detectors
    obj_detector = None
    pose_detector = None

    if detect_objects:
        obj_detector = ObjectDetector(cfg)
        obj_detector.load_model(cfg)
        obj_detector.init_tracker(cfg)

    if detect_pose:
        pose_detector = PoseDetector(cfg)
        pose_detector.load_model(cfg, detector_type=args.pose_detector)

    # Initialize metrics components (always enabled for pose detection)
    metrics_calculator = None
    metrics_logger = None
    metrics_plotter = None

    if detect_pose:
        metrics_calculator = PoseMetricsCalculator(detector_type=args.pose_detector, window_size=1)
        metrics_logger = MetricsLogger(str(source_path), output_dir=str(save_dir))

        if args.realtime_plot:
            metrics_plotter = MetricsPlotter()
            metrics_plotter.init_realtime_plot()

    # Setup video info
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    fps = video_info.fps if cfg.get("output_fps") is None else cfg["output_fps"]

    # Prepare output paths
    output_paths = []
    if detect_objects:
        bbox_path = save_dir / f"{source_path.stem}_with_box{source_path.suffix}"
        output_paths.append(("bbox", bbox_path))

    if detect_pose:
        pose_path = save_dir / f"{source_path.stem}_with_pose{source_path.suffix}"
        output_paths.append(("pose", pose_path))

    print(f"Processing video: {source_path}")
    print(f"Active stages: {', '.join(stages)}")
    if detect_pose:
        print(f"Pose detector: {args.pose_detector}")
    print("Smoothing: kalman_rts (always enabled)")
    print(f"Output directory: {save_dir}")
    if not args.no_preview:
        print("\nPress 'q' in the preview window to stop processing early.")

    # Use three-pass processing for Kalman smoothing
    if args.smooth in ["kalman_rts", "kalman_2d"] and detect_pose:
        process_video_with_smoothing(
            source_path, save_dir, cfg, obj_detector, pose_detector, args
        )
        return

    # Process video (standard single-pass mode)
    cap = cv2.VideoCapture(str(source_path))

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writers
    writers = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

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
                gap_count,
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
                    tracker_id=track_ids,
                )

                labels = [f"ID: {tid}" for tid in track_ids]

                bbox_frame = bbox_annotator.annotate(
                    scene=bbox_frame, detections=sv_detections
                )
                bbox_frame = label_annotator.annotate(
                    scene=bbox_frame, detections=sv_detections, labels=labels
                )

            # Pose estimation stage
            keypoints = None
            if detect_pose:
                if main_track_id is not None and detections:
                    # Find the main track's bbox
                    main_bbox = None
                    for bbox, track_id, _conf in detections:
                        if track_id == main_track_id:
                            main_bbox = bbox
                            break

                    if main_bbox is not None:
                        # Run pose detection
                        dt = 1.0 / fps
                        try:
                            if args.metrics:
                                pose_frame, keypoints = pose_detector.run(
                                    pose_frame, main_bbox, dt, return_keypoints=True
                                )
                            else:
                                pose_frame = pose_detector.run(
                                    pose_frame, main_bbox, dt
                                )
                        except Exception as e:
                            print(f"Pose detection error on frame {frame_idx}: {e}")

                        # Also draw the main track's bbox on pose frame
                        draw_bbox(pose_frame, main_bbox, label=f"ID: {main_track_id}")

            # Calculate and log metrics
            if metrics_calculator and metrics_logger:
                timestamp_ms = (frame_idx / fps) * 1000

                if keypoints is not None:
                    # Calculate shin angles for 2D frame coordinates
                    angles_2d = metrics_calculator.calculate_shin_angles(
                        keypoints, is_world_coords=False
                    )

                    # Initialize 3D angles dicts
                    angles_3d_raw = {"shin_angle": None}
                    angles_3d_smooth = {"shin_angle": None}

                    # Log all landmarks
                    landmarks = metrics_calculator.get_all_landmark_positions(keypoints)
                    metrics_logger.log_all_landmarks(
                        frame_idx, timestamp_ms, landmarks, metrics_calculator.landmarks
                    )

                    # World coordinates handling (MediaPipe only)
                    if args.pose_detector == "mediapipe" and main_bbox is not None:
                        world_keypoints_raw = pose_detector.get_world_landmarks(frame, main_bbox)

                        if world_keypoints_raw is not None:
                            angles_3d_raw = metrics_calculator.calculate_shin_angles(world_keypoints_raw, is_world_coords=True)

                            world_landmarks = metrics_calculator.get_all_landmark_positions(world_keypoints_raw)
                            metrics_logger.log_world_landmarks(frame_idx, timestamp_ms, world_landmarks, metrics_calculator.landmarks)

                        # If smoothing active, use smoothed world coords for angle
                        if args.smooth == "kalman_rts" and 'smooth_landmarks_3d' in locals():
                            sm_pts = smooth_landmarks_3d[frame_idx]
                            # Attach fake visibility of 1 to match expected shape
                            sm_kpts = np.hstack([sm_pts, np.ones((sm_pts.shape[0],1))])
                            angles_3d_smooth = metrics_calculator.calculate_shin_angles(sm_kpts, is_world_coords=True)

                    # Log shin angles: raw world in shin_angle_3d_ma column, smoothed in shin_angle_3d
                    metrics_logger.log_shin_angles(
                        frame_idx,
                        timestamp_ms,
                        angles_2d["shin_angle"],
                        None,
                        angles_3d_smooth["shin_angle"],
                        angles_3d_raw["shin_angle"],
                    )

                    # Update real-time plot if enabled
                    if metrics_plotter:
                        metrics_plotter.update_realtime_plot(
                            timestamp_ms,
                            angles_2d["shin_angle"],
                            angles_2d["shin_angle_ma"],
                            angles_3d["shin_angle"],
                            angles_3d["shin_angle_ma"],
                        )
                else:
                    # No keypoints detected, log empty values
                    metrics_logger.log_shin_angles(
                        frame_idx, timestamp_ms, None, None, None, None
                    )

            # Write frames
            if "bbox" in writers and bbox_frame is not None:
                writers["bbox"].write(bbox_frame)

            if "pose" in writers and pose_frame is not None:
                writers["pose"].write(pose_frame)

            # Show preview unless disabled
            if not args.no_preview:
                preview = pose_frame if detect_pose else bbox_frame
                if preview is not None:
                    cv2.imshow("Live Processing Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nProcessing terminated by user.")
                        break

            # Progress
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(
                    f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)"
                )

    finally:
        # Cleanup
        cap.release()
        for writer in writers.values():
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

        # Cleanup metrics components
        if metrics_logger:
            metrics_logger.close()

        if metrics_plotter:
            # Save final plot if real-time plotting was enabled
            if args.realtime_plot:
                plot_path = save_dir / f"{source_path.stem}_realtime_plot.png"
                metrics_plotter.save_current_plot(str(plot_path))
            metrics_plotter.close()

    print("\nProcessing complete!")
    for stage, path in output_paths:
        print(f"  {stage.capitalize()} output: {path}")

    # Generate final graph if metrics were collected
    if metrics_logger:
        print("\nGenerating angles graph...")
        plotter = MetricsPlotter()
        graph_path = save_dir / f"{source_path.stem}_angles_graph.png"
        plotter.generate_offline_graph(str(metrics_logger.angles_file), str(graph_path))


if __name__ == "__main__":
    main()
