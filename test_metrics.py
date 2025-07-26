#!/usr/bin/env python3
"""Test that metrics and plots are generated automatically."""

import sys
import subprocess
import os
from pathlib import Path

# Test without smoothing
print("Testing standard mode (should generate metrics automatically)...")
cmd1 = [
    sys.executable, "track.py",
    "--source", "input/00115_trimmed_short.mp4",
    "--detect", "objects,pose",
    "--pose-detector", "mediapipe",
    "--save_dir", "output/test_metrics_standard"
]
subprocess.run(cmd1)

# Test with smoothing
print("\nTesting smoothing mode (should generate metrics automatically)...")
cmd2 = [
    sys.executable, "track.py",
    "--source", "input/00115_trimmed_short.mp4",
    "--detect", "objects,pose",
    "--pose-detector", "mediapipe",
    "--smooth", "kalman_rts",
    "--save_dir", "output/test_metrics_smooth"
]
subprocess.run(cmd2)

# Check for PNG files
print("\nChecking for generated PNG plots...")
for test_dir in ["test_metrics_standard", "test_metrics_smooth"]:
    plot_path = Path(f"output/{test_dir}/00115_trimmed_short_mp4/00115_trimmed_short_angles_graph.png")
    if plot_path.exists():
        print(f"✓ Found plot: {plot_path}")
    else:
        print(f"✗ Missing plot: {plot_path}")

# Cleanup
os.system("rm -rf output/test_metrics_standard output/test_metrics_smooth")