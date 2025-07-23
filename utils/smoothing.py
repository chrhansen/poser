#!/usr/bin/env python3
"""
Smoothing utilities for pose keypoints.
Implements One-Euro filter and Exponential Moving Average (EMA).
"""

import numpy as np
from typing import Dict, Any


class OneEuroFilter:
    """
    One-Euro filter implementation for smoothing noisy signals.
    Based on: https://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        Initialize One-Euro filter.
        
        Args:
            freq: Expected frequency in Hz
            min_cutoff: Minimum cutoff frequency
            beta: Cutoff slope (higher = less lag, more jitter)
            d_cutoff: Cutoff frequency for derivative
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        
    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute alpha value for low-pass filter."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return dt / (dt + tau)
        
    def filter(self, x: float, t: float = None, axis: int = 0) -> float:
        """
        Apply One-Euro filter to input value.
        
        Args:
            x: Input value
            t: Timestamp (if None, use frequency)
            axis: Axis identifier (for multi-dimensional filtering)
            
        Returns:
            Filtered value
        """
        # Check for NaN input
        if np.isnan(x):
            return x
            
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t if t is not None else 0
            return x
            
        # Compute dt
        if t is None:
            dt = 1.0 / self.freq
        else:
            dt = max(0.001, t - self.t_prev)  # Ensure dt is positive
            self.t_prev = t
            
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        dx_hat = self._exponential_smoothing(dx, self.dx_prev, self._alpha(self.d_cutoff, dt))
        self.dx_prev = dx_hat
        
        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the signal
        x_hat = self._exponential_smoothing(x, self.x_prev, self._alpha(cutoff, dt))
        self.x_prev = x_hat
        
        return x_hat
        
    def _exponential_smoothing(self, x_curr: float, x_prev: float, alpha: float) -> float:
        """Apply exponential smoothing."""
        return alpha * x_curr + (1 - alpha) * x_prev


class ExponentialMovingAverage:
    """Simple exponential moving average filter."""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing
        """
        self.alpha = alpha
        self.value = None
        
    def filter(self, x: float, dt: float = None, axis: int = 0) -> float:
        """
        Apply EMA filter to input value.
        
        Args:
            x: Input value
            dt: Time delta (unused, for compatibility)
            axis: Axis identifier (unused, for compatibility)
            
        Returns:
            Filtered value
        """
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class MultiDimensionalFilter:
    """Wrapper for applying filters to multi-dimensional data (e.g., x, y coordinates)."""
    
    def __init__(self, filter_class, **kwargs):
        """
        Initialize multi-dimensional filter.
        
        Args:
            filter_class: Filter class to use (OneEuroFilter or ExponentialMovingAverage)
            **kwargs: Arguments to pass to filter constructor
        """
        self.filter_class = filter_class
        self.filter_args = kwargs
        self.filters = {}  # axis -> filter instance
        
    def filter(self, x: float, dt: float = None, axis: int = 0) -> float:
        """
        Apply filter to specific axis.
        
        Args:
            x: Input value
            dt: Time delta
            axis: Axis identifier (0 for x, 1 for y, etc.)
            
        Returns:
            Filtered value
        """
        if axis not in self.filters:
            self.filters[axis] = self.filter_class(**self.filter_args)
        return self.filters[axis].filter(x, dt, axis)


def create_smoother(cfg: Dict[str, Any]):
    """
    Create a smoother based on configuration.
    
    Args:
        cfg: Smoothing configuration dict with keys:
            - kind: 'one_euro' or 'ema'
            - freq: Frequency in Hz (for one_euro)
            - min_cutoff: Minimum cutoff (for one_euro)
            - beta: Beta parameter (for one_euro)
            - alpha: Alpha parameter (for ema)
            
    Returns:
        MultiDimensionalFilter instance
    """
    kind = cfg.get('kind', 'one_euro')
    
    if kind == 'one_euro':
        return MultiDimensionalFilter(
            OneEuroFilter,
            freq=cfg.get('freq', 30),
            min_cutoff=cfg.get('min_cutoff', 1.0),
            beta=cfg.get('beta', 0.015)
        )
    elif kind == 'ema':
        return MultiDimensionalFilter(
            ExponentialMovingAverage,
            alpha=cfg.get('alpha', 0.3)
        )
    else:
        raise ValueError(f"Unknown smoother kind: {kind}")