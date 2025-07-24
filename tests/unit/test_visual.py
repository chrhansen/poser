"""Unit tests for visual utilities."""

import numpy as np
import pytest

from utils.visual import create_color_palette, draw_bbox, draw_skeleton


class TestVisualUtils:
    """Test visual utility functions."""

    def test_draw_bbox(self, sample_frame, sample_bbox):
        """Test drawing a bounding box on an image."""
        # Draw bbox without label
        result = draw_bbox(sample_frame.copy(), sample_bbox)
        
        # Check that the image was modified (not all zeros anymore)
        assert not np.array_equal(result, sample_frame)
        
        # Draw bbox with label
        result_with_label = draw_bbox(
            sample_frame.copy(), 
            sample_bbox, 
            label="Test Label"
        )
        assert not np.array_equal(result_with_label, sample_frame)

    def test_draw_skeleton_yolo(self, sample_frame, sample_keypoints_yolo):
        """Test drawing skeleton with YOLO keypoints."""
        frame_copy = sample_frame.copy()
        draw_skeleton(frame_copy, sample_keypoints_yolo, model_type="yolo")
        
        # Check that the image was modified
        assert not np.array_equal(frame_copy, sample_frame)

    def test_draw_skeleton_mediapipe(self, sample_frame, sample_keypoints_mediapipe):
        """Test drawing skeleton with MediaPipe keypoints."""
        frame_copy = sample_frame.copy()
        draw_skeleton(frame_copy, sample_keypoints_mediapipe, model_type="mediapipe")
        
        # Check that the image was modified
        assert not np.array_equal(frame_copy, sample_frame)

    def test_draw_skeleton_empty_keypoints(self, sample_frame):
        """Test drawing skeleton with empty keypoints."""
        frame_copy = sample_frame.copy()
        draw_skeleton(frame_copy, np.array([]), model_type="yolo")
        
        # Image should not be modified
        assert np.array_equal(frame_copy, sample_frame)

    def test_draw_skeleton_none_keypoints(self, sample_frame):
        """Test drawing skeleton with None keypoints."""
        frame_copy = sample_frame.copy()
        draw_skeleton(frame_copy, None, model_type="yolo")
        
        # Image should not be modified
        assert np.array_equal(frame_copy, sample_frame)

    def test_draw_skeleton_low_confidence(self, sample_frame, sample_keypoints_yolo):
        """Test that low confidence keypoints are not drawn."""
        # Set all confidences to very low
        keypoints = sample_keypoints_yolo.copy()
        keypoints[:, 2] = 0.1
        
        frame_copy = sample_frame.copy()
        draw_skeleton(frame_copy, keypoints, model_type="yolo", conf_threshold=0.5)
        
        # Should not draw anything due to low confidence
        assert np.array_equal(frame_copy, sample_frame)

    def test_create_color_palette(self):
        """Test color palette creation."""
        # Test with different numbers of colors
        palette_5 = create_color_palette(5)
        assert len(palette_5) == 5
        assert all(isinstance(color, tuple) for color in palette_5)
        assert all(len(color) == 3 for color in palette_5)
        
        # All values should be in BGR range
        for color in palette_5:
            assert all(0 <= c <= 255 for c in color)
        
        # Colors should be different
        assert len(set(palette_5)) == 5
        
        # Test edge cases
        palette_1 = create_color_palette(1)
        assert len(palette_1) == 1
        
        palette_0 = create_color_palette(0)
        assert len(palette_0) == 0

    def test_draw_skeleton_custom_colors(self, sample_frame, sample_keypoints_yolo):
        """Test drawing skeleton with custom colors."""
        frame_copy = sample_frame.copy()
        
        # Use custom colors
        draw_skeleton(
            frame_copy, 
            sample_keypoints_yolo, 
            model_type="yolo",
            point_color=(255, 0, 0),  # Blue points
            line_color=(0, 255, 0),   # Green lines
            point_radius=10,
            line_thickness=3
        )
        
        # Check that the image was modified
        assert not np.array_equal(frame_copy, sample_frame)

    def test_draw_skeleton_nan_keypoints(self, sample_frame):
        """Test drawing skeleton with NaN keypoints."""
        # Create keypoints with NaN values
        keypoints = np.array([
            [np.nan, 100, 0.9],
            [100, np.nan, 0.9],
            [100, 100, 0.9],
        ])
        
        frame_copy = sample_frame.copy()
        draw_skeleton(frame_copy, keypoints, model_type="yolo")
        
        # Should handle NaN gracefully without crashing
        # Only the valid keypoint should be drawn