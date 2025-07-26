import numpy as np
from pathlib import Path
from utils.visual import MP_SKELETON


def _create_edge_traces(points, color):
    """Return x,y,z for edges for plotly scatter3d"""
    x = []
    y = []
    z = []
    for a, b in MP_SKELETON:
        x += [points[a, 0], points[b, 0], None]
        y += [points[a, 1], points[b, 1], None]
        z += [points[a, 2], points[b, 2], None]
    return x, y, z


def render_world_interactive(raw_3d: np.ndarray, smooth_3d: np.ndarray, out_path: Path, fps: int = 30):
    """Generate interactive HTML animation of raw vs smooth 3-D skeletons."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is required for interactive rendering (pip install plotly). Skipping.")
        return

    T = raw_3d.shape[0]

    # First frame traces
    raw_pts = raw_3d[0]
    sm_pts = smooth_3d[0]

    raw_edges = _create_edge_traces(raw_pts, "red")
    sm_edges = _create_edge_traces(sm_pts, "green")

    traces = [
        go.Scatter3d(x=raw_edges[0], y=raw_edges[1], z=raw_edges[2], mode="lines", line=dict(color="red", width=4), name="raw"),
        go.Scatter3d(x=sm_edges[0], y=sm_edges[1], z=sm_edges[2], mode="lines", line=dict(color="green", width=4), name="smooth"),
    ]

    # Frames
    frames = []
    for t in range(T):
        raw_edges = _create_edge_traces(raw_3d[t], "red")
        sm_edges = _create_edge_traces(smooth_3d[t], "green")
        frames.append(go.Frame(data=[
            go.Scatter3d(x=raw_edges[0], y=raw_edges[1], z=raw_edges[2]),
            go.Scatter3d(x=sm_edges[0], y=sm_edges[1], z=sm_edges[2]),
        ], name=str(t)))

    fig = go.Figure(data=traces, frames=frames)

    # Layout with play button
    fig.update_layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"), aspectmode="data"),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=1000/fps, redraw=True), fromcurrent=True)])]
        )]
    )

    fig.write_html(str(out_path), auto_open=False)
