#!/usr/bin/env python3
"""
Zero-lag Kalman Rauch-Tung-Striebel (RTS) smoother for MediaPipe 3D landmarks.
Implements offline forward-backward smoothing for jitter removal.
"""

import numpy as np


class KalmanRTS:
    """
    Kalman RTS smoother for 3D landmark sequences.
    
    State vector: [x, y, z, vx, vy, vz] (position and velocity)
    Uses constant velocity motion model.
    """
    
    def __init__(self, cfg: dict, dt_seq: np.ndarray):
        """
        Initialize the smoother with configuration and time-step data.
        
        Args:
            cfg: Smoothing configuration dict with keys:
                - process_noise: Process noise variance (m/s^2)^2
                - measurement_noise: Measurement noise variance (m^2)
                - adapt_R_by_visibility: Whether to adapt measurement noise by visibility
                - adapt_R_by_visibility_factor: Multiplier for visibility adaptation
            dt_seq: Array of time deltas between frames
        """
        self.process_noise = cfg.get('process_noise', 0.01)
        self.measurement_noise = cfg.get('measurement_noise', 4.0e-4)
        self.adapt_R_by_visibility = cfg.get('adapt_R_by_visibility', True)
        self.adapt_R_by_visibility_factor = cfg.get('adapt_R_by_visibility_factor', 1.0)
        self.dt_seq = dt_seq
        
        # State dimension (position + velocity for x,y,z)
        self.state_dim = 6
        self.obs_dim = 3
        
        # Measurement matrix H: [I3, 0] - we only observe position
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[:3, :3] = np.eye(3)
        
    def batch_smooth_all(self, landmarks_3d: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        Main entry point. Smooth all landmarks for all frames.
        
        Args:
            landmarks_3d: Array of shape (T, N, 3) with T frames, N landmarks, 3D positions
            visibility: Array of shape (T, N) with visibility/confidence scores
            
        Returns:
            Smoothed landmarks array of shape (T, N, 3)
        """
        T, N, _ = landmarks_3d.shape
        smoothed = np.zeros_like(landmarks_3d)
        
        # Process each landmark independently
        for landmark_idx in range(N):
            # Extract trajectory for this landmark
            positions = landmarks_3d[:, landmark_idx, :]  # (T, 3)
            vis_scores = visibility[:, landmark_idx]      # (T,)
            
            # Skip if landmark is never visible
            if np.all(vis_scores < 1e-6):
                smoothed[:, landmark_idx, :] = positions
                continue
                
            # Run forward-backward smoother
            smoothed[:, landmark_idx, :] = self._smooth_single_landmark(positions, vis_scores)
            
        return smoothed
    
    def _smooth_single_landmark(self, positions: np.ndarray, vis_scores: np.ndarray) -> np.ndarray:
        """
        Smooth a single landmark trajectory using Kalman RTS.
        
        Args:
            positions: (T, 3) array of 3D positions
            vis_scores: (T,) array of visibility scores
            
        Returns:
            Smoothed positions (T, 3)
        """
        T = len(positions)
        
        # Initialize state sequences
        x_pred = np.zeros((T, self.state_dim))  # Predicted states
        x_filt = np.zeros((T, self.state_dim))  # Filtered states
        P_pred = np.zeros((T, self.state_dim, self.state_dim))  # Predicted covariances
        P_filt = np.zeros((T, self.state_dim, self.state_dim))  # Filtered covariances
        
        # Initialize first state with first valid measurement
        first_valid_idx = np.argmax(vis_scores > 1e-6)
        x_filt[0, :3] = positions[first_valid_idx]
        x_filt[0, 3:] = 0  # Zero initial velocity
        P_filt[0] = np.eye(self.state_dim) * 0.1  # Initial uncertainty
        
        # Forward pass (Kalman filter)
        for k in range(1, T):
            dt = self.dt_seq[k-1]
            
            # State transition matrix F
            F = self._get_F(dt)
            
            # Process noise covariance Q
            Q = self._get_Q(dt)
            
            # Predict
            x_pred[k] = F @ x_filt[k-1]
            P_pred[k] = F @ P_filt[k-1] @ F.T + Q
            
            # Update (if measurement is visible)
            if vis_scores[k] > 1e-6:
                # Measurement
                z = positions[k]
                
                # Measurement noise covariance R (adapted by visibility)
                R = self._get_R(vis_scores[k])
                
                # Innovation
                y = z - self.H @ x_pred[k]
                S = self.H @ P_pred[k] @ self.H.T + R
                
                # Kalman gain
                K = P_pred[k] @ self.H.T @ np.linalg.inv(S)
                
                # Update state and covariance
                x_filt[k] = x_pred[k] + K @ y
                P_filt[k] = (np.eye(self.state_dim) - K @ self.H) @ P_pred[k]
            else:
                # No measurement, use prediction
                x_filt[k] = x_pred[k]
                P_filt[k] = P_pred[k]
        
        # Backward pass (RTS smoother)
        x_smooth = np.zeros_like(x_filt)
        x_smooth[-1] = x_filt[-1]  # Initialize with last filtered state
        
        for k in range(T-2, -1, -1):
            dt = self.dt_seq[k]
            F = self._get_F(dt)
            
            # Smoother gain
            C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k+1])
            
            # Smooth
            x_smooth[k] = x_filt[k] + C @ (x_smooth[k+1] - x_pred[k+1])
        
        # Extract smoothed positions
        return x_smooth[:, :3]
    
    def _get_F(self, dt: float) -> np.ndarray:
        """
        Get state transition matrix for constant velocity model.
        
        F = [[I3, dt*I3],
             [0,  I3   ]]
        """
        F = np.eye(self.state_dim)
        F[:3, 3:] = np.eye(3) * dt
        return F
    
    def _get_Q(self, dt: float) -> np.ndarray:
        """
        Get process noise covariance matrix.
        
        Q = σa² * [[dt⁴/4*I3, dt³/2*I3],
                   [dt³/2*I3, dt²*I3  ]]
        
        where σa² is the process noise parameter (acceleration variance).
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Position-position block
        Q[:3, :3] = np.eye(3) * (dt4 / 4)
        
        # Position-velocity block
        Q[:3, 3:] = np.eye(3) * (dt3 / 2)
        Q[3:, :3] = np.eye(3) * (dt3 / 2)
        
        # Velocity-velocity block
        Q[3:, 3:] = np.eye(3) * dt2
        
        return Q * self.process_noise
    
    def _get_R(self, visibility: float) -> np.ndarray:
        """
        Get measurement noise covariance matrix, adapted by visibility.
        
        R = diag(σr², σr², σr²) / (visibility² + ε)
        """
        R_base = np.eye(self.obs_dim) * self.measurement_noise
        
        if self.adapt_R_by_visibility:
            # Scale measurement noise inversely with visibility
            # Higher visibility = lower noise
            eps = 1e-6
            scale = 1.0 / (visibility**2 + eps)
            R = R_base * scale * self.adapt_R_by_visibility_factor
        else:
            R = R_base
            
        return R