#!/usr/bin/env python3
"""
Geometry utilities for bounding box operations.
"""

from typing import Tuple

import numpy as np


def pad_bbox_to_square(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, pad_ratio: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Pad a bounding box to make it square and add extra padding.

    Args:
        x1, y1, x2, y2: Original bounding box coordinates
        img_w, img_h: Image dimensions
        pad_ratio: Padding ratio to add (0.5 = 50% padding)

    Returns:
        Tuple of padded bbox coordinates (x1_pad, y1_pad, x2_pad, y2_pad)
    """
    # Compute center and half-sizes
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # Make square by using max dimension
    size = max(w, h)

    # Add padding
    size_padded = size * (1 + pad_ratio)
    half_size = size_padded / 2.0

    # Compute new bbox
    x1_pad = int(cx - half_size)
    y1_pad = int(cy - half_size)
    x2_pad = int(cx + half_size)
    y2_pad = int(cy + half_size)

    # Clamp to image boundaries
    x1_pad = max(0, x1_pad)
    y1_pad = max(0, y1_pad)
    x2_pad = min(img_w, x2_pad)
    y2_pad = min(img_h, y2_pad)

    return x1_pad, y1_pad, x2_pad, y2_pad


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bboxes.

    Args:
        bbox1: First bbox as [x1, y1, x2, y2]
        bbox2: Second bbox as [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def expand_bbox(bbox: np.ndarray, expansion_ratio: float) -> np.ndarray:
    """
    Expand a bounding box by a given ratio.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        expansion_ratio: Ratio to expand (1.0 = no change, 1.5 = 50% bigger)

    Returns:
        Expanded bbox as [x1, y1, x2, y2]
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    new_w = w * expansion_ratio
    new_h = h * expansion_ratio

    return np.array([cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2])


def center_bbox(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Get the center point of a bounding box.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]

    Returns:
        Tuple of (center_x, center_y)
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cx, cy


def bbox_area(bbox: np.ndarray) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]

    Returns:
        Area of the bbox
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w * h
