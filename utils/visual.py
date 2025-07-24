#!/usr/bin/env python3
"""
Visual utilities for drawing bounding boxes and skeletons.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# COCO pose connections for YOLO models (0-based indexing)
COCO_SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],  # Face
    [5, 11],
    [6, 12],
    [5, 6],  # Torso
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],  # Arms
    [1, 2],
    [0, 1],
    [0, 2],  # Lower body connections
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],  # Legs
]

# MediaPipe pose connections
MP_SKELETON = [
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 7),  # Face
    (4, 5),
    (5, 6),
    (6, 8),  # Face
    (9, 10),  # Mouth
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Upper body
    (11, 23),
    (12, 24),
    (23, 24),  # Torso
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),  # Legs
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),  # Feet
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),  # Hands
    (17, 19),
    (18, 20),  # Hand details
]


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw a bounding box on an image.

    Args:
        image: Input image
        bbox: Bounding box as [x1, y1, x2, y2]
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label to draw
        font_scale: Font scale for label

    Returns:
        Image with bbox drawn
    """
    x1, y1, x2, y2 = bbox.astype(int)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label if provided
    if label:
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    model_type: str = "yolo",
    conf_threshold: float = 0.2,
    point_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 3,
    line_thickness: int = 1,
):
    """
    Draw skeleton on image (in-place).

    Args:
        image: Input image
        keypoints: Array of keypoints with shape (N, 3) where each row is (x, y, conf)
        model_type: 'yolo' or 'mediapipe' to determine skeleton connections
        conf_threshold: Minimum confidence to draw a keypoint
        point_color: BGR color for keypoints
        line_color: BGR color for skeleton lines
        point_radius: Radius for keypoint circles
        line_thickness: Thickness for skeleton lines
    """
    if keypoints is None or len(keypoints) == 0:
        return

    # Select skeleton based on model type
    if model_type == "mediapipe":
        skeleton = MP_SKELETON
    else:  # YOLO
        skeleton = COCO_SKELETON

    # Draw skeleton lines
    for connection in skeleton:
        kpt_a = connection[0]
        kpt_b = connection[1]

        # Check if both keypoints exist and have sufficient confidence
        if kpt_a < len(keypoints) and kpt_b < len(keypoints):
            if (
                keypoints[kpt_a, 2] >= conf_threshold
                and keypoints[kpt_b, 2] >= conf_threshold
                and not np.isnan(keypoints[kpt_a, 0])
                and not np.isnan(keypoints[kpt_a, 1])
                and not np.isnan(keypoints[kpt_b, 0])
                and not np.isnan(keypoints[kpt_b, 1])
            ):

                x1, y1 = int(keypoints[kpt_a, 0]), int(keypoints[kpt_a, 1])
                x2, y2 = int(keypoints[kpt_b, 0]), int(keypoints[kpt_b, 1])

                cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf >= conf_threshold and not np.isnan(x) and not np.isnan(y):
            cv2.circle(image, (int(x), int(y)), point_radius, point_color, -1)


def create_color_palette(n_colors: int) -> List[Tuple[int, int, int]]:
    """
    Create a color palette for tracking visualization.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of BGR color tuples
    """
    # Use HSV color space for better distribution
    colors = []
    for i in range(n_colors):
        hue = int(180 * i / n_colors)
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))

    return colors


def overlay_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
) -> np.ndarray:
    """
    Overlay text on an image with optional background.

    Args:
        image: Input image
        text: Text to overlay
        position: (x, y) position for text
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Optional background color

    Returns:
        Image with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position

    # Draw background if specified
    if bg_color is not None:
        cv2.rectangle(
            image,
            (x - 5, y - text_height - baseline - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1,
        )

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image
