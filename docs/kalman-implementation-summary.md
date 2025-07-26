# Kalman Filter Implementation Summary

**Date**: 2025-07-26  
**PR**: https://github.com/chrhansen/poser/pull/1  
**Branch**: `feature/kalman-zero-lag-smoothing`

## Overview

Implemented offline Kalman RTS (Rauch-Tung-Striebel) smoother for jitter-free pose tracking, with automatic shin angle metrics generation.

## Key Changes

### 1. Zero-lag Kalman RTS Smoother
- **File**: `src/filters/kalman_zero_lag.py`
- Forward-backward Kalman filtering for 3D world coordinates
- State vector: `[x, y, z, vx, vy, vz]` (position and velocity)
- Constant velocity motion model
- Visibility-adaptive measurement noise

### 2. 2D Kalman Filter Alternative
- **File**: `src/filters/kalman_2d.py`
- Direct smoothing of 2D image coordinates
- More stable than 3D approach (avoids coordinate system issues)
- State vector: `[x, y, vx, vy]` in pixel space

### 3. Three-Pass Processing Pipeline
Modified `track.py` to implement:
- **Pass 1**: Detection and buffering all frames
- **Pass 2**: Calibration and smoothing
- **Pass 3**: Projection and rendering

### 4. Camera Projection
- **File**: `src/projection.py`
- PnP-based camera pose estimation
- 3D-to-2D projection of smoothed landmarks
- Note: Has alignment issues with MediaPipe's hip-centered coordinates

### 5. Automatic Metrics Generation
- Removed `--metrics` flag requirement
- Metrics always generated when pose detection enabled
- Shin angle plots automatically created
- Works in both standard and smoothing modes

### 6. Updated Shin Angle Plotting
- Logs both raw and smoothed world coordinate angles
- Raw angles → `shin_angle_3d_ma` column
- Smoothed angles → `shin_angle_3d` column
- Removed distance metrics (knee-ankle distance)

## Configuration

Added to `configs/default.yaml`:
```yaml
smoothing_cfg:
  kind: kalman_rts
  process_noise: 0.01                 # (m/s^2)^2
  measurement_noise: 1.0e-8           # metres^2
  adapt_R_by_visibility: true
  adapt_R_by_visibility_factor: 1.0
  # 2D smoothing parameters
  process_noise_2d: 100.0             # pixels/s^2
  measurement_noise_2d: 4.0           # pixels^2

projection_cfg:
  pnp_visibility_threshold: 0.8
  pnp_ransac_iterations: 100
  pnp_ransac_reprojection_error: 8.0
```

## Usage

```bash
# Standard mode (no smoothing)
python track.py --source video.mp4 --detect objects,pose --pose-detector mediapipe

# With 3D Kalman smoothing (may have alignment issues)
python track.py --source video.mp4 --detect objects,pose \
  --pose-detector mediapipe --smooth kalman_rts

# With 2D Kalman smoothing (recommended)
python track.py --source video.mp4 --detect objects,pose \
  --pose-detector mediapipe --smooth kalman_2d

# Visual comparison of raw vs smoothed
python track.py --source video.mp4 --detect objects,pose \
  --pose-detector mediapipe --smooth kalman_2d --compare

# Override noise parameters
python track.py --source video.mp4 --detect objects,pose \
  --pose-detector mediapipe --smooth kalman_rts \
  --process-noise 0.1 --measurement-noise 1e-5
```

## Known Issues

1. **3D Projection Misalignment**: The `kalman_rts` mode may show skeleton misalignment due to:
   - MediaPipe's hip-centered coordinate system
   - Inaccurate camera intrinsics estimation
   - PnP instability with body-centric coordinates

2. **Recommendation**: Use `--smooth kalman_2d` for most cases as it's more stable

## Implementation Details

### Files Modified
- `track.py`: Added three-pass processing, automatic metrics
- `src/pose/detect_pose.py`: Removed online smoothing
- `src/pose/pose_detector_base.py`: Removed smoothing methods
- `configs/default.yaml`: Added smoothing configuration
- Deleted: `utils/smoothing.py` (old online smoothing)

### Files Added
- `src/filters/kalman_zero_lag.py`: 3D RTS smoother
- `src/filters/kalman_2d.py`: 2D Kalman filter
- `src/projection.py`: Camera projection utilities
- `docs/spec-smoothing.md`: Original specification

### Metrics Changes
- Always generates `*_angles_graph.png` for shin angles
- Logs raw and smoothed angles separately
- Removed distance-based metrics
- No need for `--metrics` flag anymore

## Testing

Tested on ski videos with MediaPipe pose detection:
- Standard mode generates metrics ✓
- Smoothing modes generate metrics ✓
- Comparison mode shows raw (red) vs smoothed (green) ✓
- PNG plots created automatically ✓

## Notes for Future Sessions

- The 3D smoothing alignment issue stems from MediaPipe's coordinate system
- Consider implementing a coordinate system transformation
- The 2D smoothing is more practical for most use cases
- Metrics are now always generated - no flag needed