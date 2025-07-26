#!/usr/bin/env python3
"""
Offline smoothing utility: takes previously saved raw 3-D landmarks and regenerates
smoothed landmarks plus debug visualisations without rerunning detection.

Usage:
    python scripts/offline_smooth.py \
        --data-dir output/input_dynamic_short_nz_trimmed_mp4 \
        --fps 30 \
        --process-noise 0.05 \
        --measurement-noise 1e-4

Requires raw_landmarks_3d.npy and visibility.npy inside data-dir. If dt_seq.npy exists
it is used, otherwise constant 1/fps is assumed.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH so that `src` package can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.filters.kalman_zero_lag import KalmanRTS
from src.visualization.render_world_debug import render_world_skeletons
from src.visualization.render_world_interactive import render_world_interactive
from src.metrics.calculator import PoseMetricsCalculator


def parse_args():
    p = argparse.ArgumentParser(description="Offline smoothing and debug visualisation")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory with raw_landmarks_3d.npy and visibility.npy")
    p.add_argument("--fps", type=float, default=30, help="Video FPS used to build dt sequence if dt_seq.npy missing")
    p.add_argument("--process-noise", type=float, default=0.05, help="Kalman process noise variance")
    p.add_argument("--measurement-noise", type=float, default=1e-4, help="Kalman measurement noise variance")
    p.add_argument("--output", type=Path, default=None, help="Optional output directory (default = data-dir)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.output or data_dir
    out_dir.mkdir(exist_ok=True)

    raw_3d_path = data_dir / "raw_landmarks_3d.npy"
    raw_3d_csv = data_dir / "raw_landmarks_3d.csv"
    visibility_path = data_dir / "visibility.npy"

    if raw_3d_path.exists():
        raw_3d = np.load(raw_3d_path)
    elif raw_3d_csv.exists():
        print("Loading raw landmarks from CSV file")
        flat = np.loadtxt(raw_3d_csv, delimiter=",")
        if flat.shape[1] % 3 != 0:
            raise ValueError("CSV column count not divisible by 3")
        N = flat.shape[1] // 3
        raw_3d = flat.reshape(flat.shape[0], N, 3)
    else:
        raise FileNotFoundError(f"Missing {raw_3d_path} or {raw_3d_csv}")

    if visibility_path.exists():
        visibility = np.load(visibility_path)
    else:
        # Fallback: assume all landmarks visible
        print("visibility.npy not found – assuming visibility=1 for all points.")
        visibility = np.ones(raw_3d.shape[:2])

    # dt sequence
    dt_seq_path = data_dir / "dt_seq.npy"
    if dt_seq_path.exists():
        dt_seq = np.load(dt_seq_path)
    else:
        T = raw_3d.shape[0]
        dt_seq = np.full(T - 1, 1.0 / args.fps, dtype=float)
        dt_seq = np.concatenate([dt_seq, [dt_seq[-1]]])

    # Kalman config
    cfg = dict(process_noise=args.process_noise, measurement_noise=args.measurement_noise)

    smoother = KalmanRTS(cfg, dt_seq)
    smooth_3d = smoother.batch_smooth_all(raw_3d, visibility)

    np.save(out_dir / "smooth_landmarks_3d.npy", smooth_3d)

    # Compute shin angles for raw vs smooth
    calc = PoseMetricsCalculator(detector_type="mediapipe", window_size=1)
    rows = []
    for t in range(raw_3d.shape[0]):
        raw_angles = calc.calculate_shin_angles(raw_3d[t], is_world_coords=True)
        smooth_angles = calc.calculate_shin_angles(smooth_3d[t], is_world_coords=True)
        rows.append([
            t, raw_angles["shin_angle"], smooth_angles["shin_angle"]
        ])

    np.savetxt(out_dir / "shin_angles_raw_vs_smooth.csv", rows, delimiter=",", header="frame,raw_angle_deg,smooth_angle_deg", comments="")

    # Debug renders
    render_world_skeletons(raw_3d, smooth_3d, out_dir / "world_debug_offline.mp4", fps=args.fps)
    render_world_interactive(raw_3d, smooth_3d, out_dir / "world_debug_offline.html", fps=args.fps)

    print("Offline smoothing complete. Outputs saved to", out_dir)


if __name__ == "__main__":
    main()
