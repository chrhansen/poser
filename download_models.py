#!/usr/bin/env python3
"""
Download required model files for pose detection.
"""

import os
import urllib.request
from pathlib import Path


def download_file(url: str, dest_path: str):
    """Download a file from URL to destination path."""
    print(f"Downloading {os.path.basename(dest_path)} from {url}...")
    
    # Create directory if it doesn't exist
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"Progress: {percent:.1f}%", end='\r')
    
    urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
    print(f"\nDownloaded to {dest_path}")


def main():
    """Download all required models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # MediaPipe Pose Landmarker Heavy model
    pose_model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    pose_model_path = models_dir / "pose_landmarker_heavy.task"
    
    if not pose_model_path.exists():
        download_file(pose_model_url, str(pose_model_path))
    else:
        print(f"{pose_model_path} already exists, skipping download.")
    
    print("\nAll models downloaded successfully!")


if __name__ == "__main__":
    main()