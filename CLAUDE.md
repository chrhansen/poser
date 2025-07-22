# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python computer vision project that detects and tracks skiers in video footage using pose detection. The current implementation uses MediaPipe for pose detection with OpenCV tracking and Kalman filtering for motion prediction.

## Key Technologies

- **Python 3.9** with virtual environment at `venv/`
- **MediaPipe** (0.10.14) for pose detection
- **OpenCV** (opencv-contrib-python 4.10.0.82) for video processing and tracking
- **NumPy** (2.0.2) for numerical operations

## Running the Application

The main script is executed as:
```bash
python3 pose_tracker_simple.py --input <video_file> --output <output_file> [--debug]
```

## Code Architecture

The project uses a single-file architecture in `pose_tracker_simple.py` with:
- `Config` dataclass: Configuration parameters for tracking
- `SimpleKalmanFilter`: Motion prediction for smooth tracking
- `SimplePoseTracker`: Main processing engine with two states:
  - SEARCHING: Full-frame pose detection
  - TRACKING: ROI-based tracking with OpenCV tracker

## Alternative Implementation

The `docs/spec-bbox.md` file contains specifications for a different approach using YOLOv11 for skier detection instead of MediaPipe pose detection.

## Development Notes

- No formal testing framework or test files exist
- No linting or formatting tools configured
- Dependencies are installed in the virtual environment but no requirements.txt exists
- The project processes video files frame-by-frame, adding pose overlays and bounding boxes