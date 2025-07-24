## Software Specification

### "Track-and-Pose" – modular skier analysis pipeline

---

### 1. Objective

Deliver a modular Python package that extends an existing object tracker so that object-tracking and pose-estimation stages can be turned on/off from the command line without code edits. The script will produce different output videos depending on the arguments provided.

**Example Usage & Outputs:**

```bash
# default = both stages, outputs two videos
# Outputs: 00115_with_box.mp4, 00115_with_pose.mp4
python track.py --source videos/00115.mp4

# objects only, outputs one video
# Outputs: 00115_with_box.mp4
python track.py --source videos/00115.mp4 --detect objects

# objects + pose (explicit), outputs two videos
# Outputs: 00115_with_box.mp4, 00115_with_pose.mp4
python track.py --source videos/00115.mp4 --detect objects,pose
```

The finished program must:

1.  Preserve the current bounding-box tracker behavior, saving the result to a file like `<source_name>_with_box.mp4`.
2.  Run pose estimation only on the chosen "main" skier's bounding box.
3.  If pose estimation is active, re-project and draw a smoothed skeleton back onto the original frame, saving the result to a separate file like `<source_name>_with_pose.mp4`.

### 2. Project Layout

The project will be a Python package with the following structure:

```python
poser/
├── track.py            # CLI entry-point orchestrating the pipeline
├── detect_objects.py   # Object detection + tracking module
├── detect_pose.py      # Pose estimation + skeleton drawing module
├── utils/
│   ├── smoothing.py    # Exponential / One-Euro filter helpers
│   ├── geometry.py     # Bbox padding, ROI → frame coordinate maps
│   └── visual.py       # Drawing helpers (boxes, labels, skeleton)
├── configs/
│   └── default.yaml
└── requirements.txt
```

All modules must be import-safe (no top-level heavy GPU work).

### 3. Command-line Interface (`track.py`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | str (file/dir/cam) | **required** | Input video or webcam index |
| `--detect` | csv (objects,pose) | "objects,pose" | Which stages to activate |
| `--tracker` | "botsort" \| "ocsort" | "botsort" | Tracker backend (pass-through) |
| `--save_dir` | str | `output/` | Where to save all output files |
| `--show` | store_true | - | Live preview window |
| `--config` | path | `configs/default.yaml` | Override hyper-params |

The command-line argument parser lives in `track.py`; validated values are passed to the two processing modules.

### 4. Module Responsibilities

#### 4.1 `detect_objects.py`

| Function | Purpose |
| :--- | :--- |
| `load_model(cfg)` | Instantiate YOLO v11 with weights from YAML. |
| `init_tracker(cfg)` | Create BoT-SORT / OC-SORT instance (same args as today). |
| `run(frame)` | Return `List[Detection]` where `Detection = (bbox_xyxy, track_id, conf)`. |

The implementation should add a book-keeping dictionary `track_lengths` to expose the “longest-lived” track ID.

#### 4.2 `detect_pose.py`

| Function | Purpose |
| :--- | :--- |
| `load_model(cfg)` | Load either YOLO-Pose or MediaPipe BlazePose per YAML. |
| `set_roi_padding(p)` | Configure padding ratio (e.g. 0.50). |
| `run(frame, bbox, dt)` | • Crop + pad → square ROI<br>• Resize for model (256x256 BlazePose or ≤ 640 YOLO-Pose)<br>• Inference → keypoints<br>• Smooth (`utils.smoothing`)<br>• Draw skeleton on `frame` (in-place)<br>• Return updated frame |

If the pose stage is disabled, `run` is a no-op that returns the untouched frame.

### 5. Algorithmic Details

| Step | Specification |
| :--- | :--- |
| **Main-ID Selection** | At every frame, `main_id = argmax(track_lengths)`. If missing for `gap_max` frames, pick the next best. |
| **Padding & Cropping** | Compute center `(cx,cy)` and half-sizes `(w,h)`, then expand by `pad_ratio`; clamp to image borders (`utils.geometry`). |
| **Resize** | BlazePose → (256, 256). YOLO-Pose → scale so `max(w,h)≤640` and is divisible by 32. |
| **Re-projection** | `x_full = x_roi * (roi_w/resize_w) + crop_x1`; same for `y`. |
| **Smoothing** | One-Euro filter per joint (hyper-parameters in YAML) or EMA with α = 0.3. |
| **Drawing** | Use simple `cv2.circle` + `cv2.line` pairs; joints below `conf_min` are skipped. Colors should be consistent but not hard-coded; accept a BGR tuple from YAML. |

### 6. Configuration (`default.yaml`)

```yaml
object_model: yolov11m.pt
pose_model: YOLO11m-pose.pt       # or "mediapipe"
device: cuda                     # "cpu" | "mps" | "cuda"
pad_ratio: 0.5
conf_min: 0.2
smoothing:
  kind: one_euro
  freq: 30
  min_cutoff: 1.0
  beta: 0.015
tracker:
  track_high_thresh: 0.6
  track_low_thresh: 0.1
  new_track_thresh: 0.7
output_fps: null                 # keep original if null
```

### 7. Dependencies

Versions should be specified for known interoperability.

```makefile
ultralytics==8.2.0
mediapipe==0.10.9
opencv-python-headless==4.11.0
torch>=2.3.0
boxmot>=0.4.2
supervision==0.9.0
numpy>=1.26
pyyaml>=6.0
```

Place these in `requirements.txt` and supply a `pip install -r requirements.txt` line in the `README.md`.

### 8. Deliverables

| File | Description |
| :--- | :--- |
| Source code | All `.py` modules and the `configs/` directory. |
| `README.md` | Setup, quick-start commands, model download links. |
| Sample outputs | GIFs or MP4s showing the object-only vs. object+pose outputs. |
| `requirements.txt` | Exact package versions. |
| `Changelog.md` | A high-level summary of work done. |

### 9. Acceptance Criteria

*   Running the example CLI calls produces (a) no errors, (b) correct overlays, and (c) the correct output videos in the `--save_dir` with the specified naming convention.
*   Documentation allows a new developer to replicate results in under 15 minutes.

### 10. Stretch Goals (Optional)

*   Add a `--detect pose` (pose-only) mode for future use.
*   Export key-points to a CSV or JSON file alongside the output video.
*   Create a real-time GUI with Tk/QT showing FPS and joint trajectories.
