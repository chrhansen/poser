#!/usr/bin/env python3
"""
2D Kalman filter for direct smoothing of image coordinates.
Simpler and more stable than 3D smoothing + projection.
"""

import numpy as np
from .kalman_zero_lag import KalmanRTS


class Kalman2D(KalmanRTS):
    """
    2D version of Kalman RTS smoother for image coordinates.
    State vector: [x, y, vx, vy] (position and velocity in pixels)
    """
    
    def __init__(self, cfg: dict, dt_seq: np.ndarray):
        """Initialize 2D smoother."""
        super().__init__(cfg, dt_seq)
        
        # Override dimensions for 2D
        self.state_dim = 4  # x, y, vx, vy
        self.obs_dim = 2    # x, y
        
        # Measurement matrix H: [I2, 0] - we only observe position
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[:2, :2] = np.eye(2)
        
        # Scale process noise for pixel coordinates (much larger than meters)
        self.process_noise = cfg.get('process_noise_2d', 100.0)  # pixels/s^2
        self.measurement_noise = cfg.get('measurement_noise_2d', 4.0)  # pixels^2
    
    def _get_F(self, dt: float) -> np.ndarray:
        """State transition for 2D constant velocity model."""
        F = np.eye(self.state_dim)
        F[:2, 2:] = np.eye(2) * dt
        return F
    
    def _get_Q(self, dt: float) -> np.ndarray:
        """Process noise for 2D model."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Position-position block
        Q[:2, :2] = np.eye(2) * (dt4 / 4)
        
        # Position-velocity block
        Q[:2, 2:] = np.eye(2) * (dt3 / 2)
        Q[2:, :2] = np.eye(2) * (dt3 / 2)
        
        # Velocity-velocity block
        Q[2:, 2:] = np.eye(2) * dt2
        
        return Q * self.process_noise
    
    def _get_R(self, visibility: float) -> np.ndarray:
        """Measurement noise for 2D model."""
        R_base = np.eye(self.obs_dim) * self.measurement_noise
        
        if self.adapt_R_by_visibility:
            eps = 1e-6
            scale = 1.0 / (visibility**2 + eps)
            R = R_base * scale * self.adapt_R_by_visibility_factor
        else:
            R = R_base
            
        return R
    
    def batch_smooth_all(self, landmarks_2d: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Smooth 2D landmarks directly.
        
        Args:
            landmarks_2d: Array of shape (T, N, 2) with T frames, N landmarks, 2D positions
            visibility: Array of shape (T, N) with visibility/confidence scores
            
        Returns:
            Smoothed landmarks array of shape (T, N, 2)
        """
        T, N, _ = landmarks_2d.shape
        smoothed = np.zeros_like(landmarks_2d)
        
        # Process each landmark independently
        for landmark_idx in range(N):
            # Extract trajectory for this landmark
            positions = landmarks_2d[:, landmark_idx, :]  # (T, 2)
            vis_scores = visibility[:, landmark_idx]      # (T,)
            
            # Skip if landmark is never visible
            if np.all(vis_scores < 1e-6):
                smoothed[:, landmark_idx, :] = positions
                continue
                
            # Run forward-backward smoother
            smoothed[:, landmark_idx, :] = self._smooth_single_landmark(positions, vis_scores)
            
        return smoothed