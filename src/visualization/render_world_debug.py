import cv2
import numpy as np
from utils.visual import MP_SKELETON

def render_world_skeletons(raw_3d: np.ndarray, smooth_3d: np.ndarray, out_path, fps: int = 30, size: int = 800, scale: int = 300):
    T, N, _ = raw_3d.shape
    origin_x = size // 2
    origin_y = size // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (size, size))
    for t in range(T):
        canvas = np.full((size, size, 3), 255, np.uint8)
        for pts, color in ((raw_3d[t], (0, 0, 255)), (smooth_3d[t], (0, 200, 0))):
            k2d = np.empty((N, 2), dtype=int)
            k2d[:, 0] = (pts[:, 0] * scale + origin_x).astype(int)
            k2d[:, 1] = (-pts[:, 1] * scale + origin_y).astype(int)
            for a, b in MP_SKELETON:
                cv2.line(canvas, (k2d[a, 0], k2d[a, 1]), (k2d[b, 0], k2d[b, 1]), color, 2)
            for p in k2d:
                cv2.circle(canvas, (p[0], p[1]), 3, color, -1)
        writer.write(canvas)
    writer.release()
