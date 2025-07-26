#!/usr/bin/env python3
"""
Camera projection utilities for converting between 3D world coordinates and 2D image coordinates.
Includes PnP-based camera pose estimation and batch point projection.
"""

import cv2
import numpy as np


def estimate_camera_extrinsics(
    landmarks_2d: np.ndarray,
    landmarks_3d: np.ndarray,
    visibility: np.ndarray,
    K: np.ndarray,
    pnp_cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate camera extrinsics for all frames using PnP.

    Args:
        landmarks_2d: (T, N, 2) array of 2D landmark positions in pixels
        landmarks_3d: (T, N, 3) array of 3D landmark positions in meters
        visibility: (T, N) array of visibility scores
        K: 3x3 camera intrinsic matrix
        pnp_cfg: PnP configuration dict with keys:
            - pnp_visibility_threshold: Minimum visibility to use landmark
            - pnp_ransac_iterations: Number of RANSAC iterations
            - pnp_ransac_reprojection_error: RANSAC threshold in pixels

    Returns:
        Tuple of:
            - R_seq: (T, 3, 3) rotation matrices
            - t_seq: (T, 3, 1) translation vectors
    """
    T, N, _ = landmarks_2d.shape

    # Extract config
    vis_thresh = pnp_cfg.get('pnp_visibility_threshold', 0.8)
    ransac_iters = pnp_cfg.get('pnp_ransac_iterations', 100)
    ransac_thresh = pnp_cfg.get('pnp_ransac_reprojection_error', 8.0)

    # Initialize output sequences
    R_seq = np.zeros((T, 3, 3))
    t_seq = np.zeros((T, 3, 1))

    # Process each frame
    for t in range(T):
        # Select high-visibility landmarks
        valid_mask = (visibility[t] > vis_thresh) & (~np.all(landmarks_3d[t] == 0, axis=1))

        if np.sum(valid_mask) < 4:  # Need at least 4 points for PnP
            # Use identity transformation if not enough points
            R_seq[t] = np.eye(3)
            t_seq[t] = np.zeros((3, 1))
            continue

        # Extract valid 2D/3D correspondences
        pts_2d = landmarks_2d[t][valid_mask]
        pts_3d = landmarks_3d[t][valid_mask]

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d.astype(np.float32),
            imagePoints=pts_2d.astype(np.float32),
            cameraMatrix=K,
            distCoeffs=None,  # Assuming no distortion
            iterationsCount=ransac_iters,
            reprojectionError=ransac_thresh,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            R_seq[t] = R
            t_seq[t] = tvec
        else:
            # Fallback to identity if PnP fails
            R_seq[t] = np.eye(3)
            t_seq[t] = np.zeros((3, 1))

    return R_seq, t_seq


def project_points_batch(
    landmarks_3d: np.ndarray,
    R_seq: np.ndarray,
    t_seq: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """
    Project 3D points to 2D for all frames efficiently.

    Args:
        landmarks_3d: (T, N, 3) array of 3D points
        R_seq: (T, 3, 3) rotation matrices
        t_seq: (T, 3, 1) translation vectors
        K: 3x3 camera intrinsic matrix

    Returns:
        landmarks_2d: (T, N, 2) array of 2D pixel coordinates
    """
    T, N, _ = landmarks_3d.shape
    landmarks_2d = np.zeros((T, N, 2))

    for t in range(T):
        # Transform points: P' = R*P + t
        pts_3d = landmarks_3d[t].T  # (3, N)
        pts_cam = R_seq[t] @ pts_3d + t_seq[t]  # (3, N)

        # Project to image plane: p = K*P'
        pts_proj = K @ pts_cam  # (3, N)

        # Normalize by z-coordinate
        pts_2d = pts_proj[:2, :] / pts_proj[2:3, :]  # (2, N)

        # Store result
        landmarks_2d[t] = pts_2d.T

    return landmarks_2d


def initialize_intrinsics(width: int, height: int) -> np.ndarray:
    """
    Initialize camera intrinsics with reasonable defaults.

    Assumes focal length approximately equals image width.
    Principal point at image center.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        K: 3x3 intrinsic matrix
    """
    K = np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    return K
