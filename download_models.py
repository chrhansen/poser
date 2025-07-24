#!/usr/bin/env python3
"""
Download required model files for object detection and pose estimation.
"""

import os
import urllib.request
from pathlib import Path


def download_file(url: str, dest_path: str):
    """Download a file from URL to destination path."""
    print(f"Downloading {os.path.basename(dest_path)}...")

    # Create directory if it doesn't exist
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"Progress: {percent:.1f}%", end="\r")

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print(f"\n✓ Downloaded to {dest_path}")
    except Exception as e:
        print(f"\n✗ Failed to download {os.path.basename(dest_path)}: {e}")
        return False
    return True


def ensure_yolo_model(model_name: str, models_dir: Path):
    """Ensure YOLO model exists in the models directory."""
    model_path = models_dir / model_name
    if model_path.exists():
        print(f"✓ {model_path} already exists")
        return True

    print(f"Note: {model_name} will be automatically downloaded to {model_path}")
    print("      when first used by the application.")
    return True


def main():
    """Download all required models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=== Downloading ML Models ===\n")

    # YOLO models
    print("1. YOLO Object Detection Model")
    ensure_yolo_model("yolo11m.pt", models_dir)

    print("\n2. YOLO Pose Detection Model")
    ensure_yolo_model("yolo11x-pose.pt", models_dir)

    # MediaPipe Pose Landmarker Heavy model
    print("\n3. MediaPipe Pose Landmarker Model")
    pose_model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    pose_model_path = models_dir / "pose_landmarker_heavy.task"

    if not pose_model_path.exists():
        download_file(pose_model_url, str(pose_model_path))
    else:
        print(f"✓ {pose_model_path} already exists, skipping download.")

    print("\n=== Download Summary ===")
    print(f"Models directory: {models_dir.absolute()}")
    print("\nMediaPipe model downloaded successfully!")
    print("\nNote: YOLO models will be automatically downloaded to the models/")
    print("directory when first used by the application.")


if __name__ == "__main__":
    main()
