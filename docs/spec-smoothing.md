https://aistudio.google.com/prompts/1uyZHLLloxShF0uclJW-ZTUsERhZyVVxq

---

## **Software Development Specification: Zero-lag Kalman Smoother for MediaPipe 3-D Landmarks**

**Project:** Poser
**Version:** 3.0

### **1. Purpose & Scope**

**Goal:**
The primary objective of this project is to implement an offline, forward-backward Kalman Rauch-Tung-Striebel (RTS) smoother to remove jitter from `pose_world_landmarks` provided by MediaPipe. The resulting smoothed 3D landmarks will then be accurately projected back to 2D image coordinates to render a phase-aligned, stabilized skeleton on the output video.

**Input:**
*   The raw `pose_world_landmarks` (3D, in meters, with a body-centric origin) from MediaPipe.
*   The raw `pose_landmarks` (2D, in normalized image coordinates) from MediaPipe.

**Output:**
*   A video file with the smoothed skeleton overlaid.
*   Optional data files containing the smoothed 3D landmarks for further analysis.

**Non-goals:**
*   Online/live smoothing or rendering.
*   Joint-coupled Unscented Kalman Filter (UKF).
*   Re-training of the underlying pose detection models.

---

### **2. High-level Design**

The implementation will follow a three-stage offline processing pipeline:

**Pass 1: Detection and Buffering**
*   The existing pose detector will run on the input video.
*   For every frame, buffer the following data into NumPy arrays:
    1.  `raw_landmarks_2d`: The (x, y) pixel coordinates, derived from MediaPipe's normalized landmarks and image dimensions.
    2.  `raw_landmarks_3d`: The (X, Y, Z) world coordinates from MediaPipe.
    3.  `visibility_scores`: The per-landmark visibility/confidence scores.

**Pass 2: Calibration and Smoothing (Intermediate Step)**
*   This pass occurs in memory after Pass 1 is complete and processes the entire buffered dataset.
*   **Camera Pose Estimation:** Using the buffered 2D/3D landmark pairs, solve the Perspective-n-Point (PnP) problem for *every frame* to determine the camera's extrinsic parameters (rotation `R` and translation `t`). This results in a sequence of rotation matrices `R_seq` and translation vectors `t_seq`, capturing the camera's pose relative to the subject's hip-centered world origin for each frame.
*   **Kalman Smoothing:** Concurrently, apply the Kalman RTS smoother to the buffered `raw_landmarks_3d` to produce a new sequence of `smooth_landmarks_3d`.

**Pass 3: Projection and Rendering**
*   Iterate through the video frames for a final time to create the output.
*   For each frame `i`:
    1.  Take the `smooth_landmarks_3d[i]`.
    2.  Using the camera intrinsics `K` and the calculated extrinsics for that frame (`R_seq[i]`, `t_seq[i]`), project the smoothed 3D points back into 2D pixel coordinates (`smooth_landmarks_2d[i]`).
    3.  Draw these 2D points onto the video frame to render the final, smoothed skeleton.

---

### **3. Kalman Model (Time-Series Smoothing)**

This section defines the temporal smoothing of the 3D world coordinates.

| Item | Value / Formula | Notes |
| :--- | :--- | :--- |
| **State vector `x`** | `[x, y, z, vx, vy, vz]`ᵀ (6x1) | Position and velocity for a single landmark. |
| **Initial State `x₀`** | `x₀ = [z₀, 0]` | Initialize velocity components (vx, vy, vz) to zero. The initial position will be the first measurement `z₀`. |
| **State Transition `F(Δt)`** | `[[I₃, Δt·I₃], [0₃, I₃]]` | Constant velocity model, where `I₃` is a 3x3 identity matrix. |
| **Measurement Matrix `H`** | `[[I₃, 0₃]]` | We only measure position. |
| **Process Noise `Q(Δt)`** | `Q = σa² * [[Δt⁴/4·I₃, Δt³/2·I₃], [Δt³/2·I₃, Δt²·I₃]]` | Derived from a continuous white noise model for acceleration. `σa²` is the `process_noise` parameter from the config. |
| **Measurement Noise `R`** | `diag(σr², σr², σr²)` | Isotropic measurement noise. `σr²` is the `measurement_noise` parameter from the config. |
| **`Δt` handling**| Per-step calculation | `F` and `Q` must be recomputed for each time step using the actual `Δt_k = t_k - t_{k-1}`. |
| **Visibility-to-`R` Scaling** | `R_new = R / (visibility² + ε)` | De-weight landmarks with low confidence by scaling their measurement noise. `ε` (e.g., 1e-6) must be added for numerical stability. |

---

### **4. Projection Model (World-to-Image Conversion)**

This section defines the geometric projection for rendering.

**Objective:**
To map the filtered world-space coordinates (X, Y, Z) back to image-space pixel coordinates (u, v).

**Methodology:**
The transformation is a standard pinhole camera projection: `(u, v, w) = K @ [R|t] @ (X, Y, Z, 1)`.

*   **Extrinsics (`R_seq`, `t_seq`):** The camera's pose (`R`, `t`) changes each frame. For each frame, it will be estimated by solving the PnP problem with OpenCV's `cv2.solvePnPRansac` using the 2D/3D landmark pairs.
    *   **Input:** Only use landmarks where `visibility > pnp_visibility_threshold` for a stable solution.
    *   **Output:** The process will yield a sequence of rotation vectors (`rvecs`) and translation vectors (`tvecs`) for every frame. The `rvecs` should be converted to 3x3 rotation matrices using `cv2.Rodrigues`.

*   **Intrinsics (`K`):** The camera's intrinsic matrix `K` is assumed to be constant throughout the video. It will be initialized with a reasonable guess based on the video's image dimensions (W, H). A robust approximation for the focal length is the image width.
    ```python
    # Example for K initialization
    K = np.array([[W, 0, W/2],
                  [0, W, H/2],
                  [0, 0,  1 ]], dtype=np.float32)
    ```

---

### **5. Module & API Design**

Two new modules shall be created.

**A. Smoothing Module: `poser/filters/kalman_zero_lag.py`**
*   **`KalmanRTS` Class:**
    *   `__init__(self, cfg: dict, dt_seq: np.ndarray)`: Initializes the smoother with configuration and time-step data.
    *   `batch_smooth_all(self, landmarks_3d: np.ndarray, visibility: np.ndarray) -> np.ndarray`: Main entry point. Takes buffered 3D landmarks and visibility, and returns the smoothed 3D landmark sequence. This method should be vectorized over all 33 joints for performance.

**B. Projection Module: `poser/projection.py`**
*   This module will contain the geometric transformation logic.
*   **`estimate_camera_extrinsics(landmarks_2d, landmarks_3d, visibility, K, pnp_cfg) -> (R_seq, t_seq)`:**
    *   Takes the full sequences of landmark data, the intrinsic matrix, and PnP configuration.
    *   Loops through each frame, selects high-visibility points, and calls `cv2.solvePnPRansac`.
    *   Returns two arrays: `R_seq` (Tx3x3 rotation matrices) and `t_seq` (Tx3x1 translation vectors).
*   **`project_points_batch(landmarks_3d, R_seq, t_seq, K) -> landmarks_2d`:**
    *   Takes sequences of 3D points and corresponding camera poses.
    *   Efficiently projects all points for all frames using vectorized operations.
    *   Returns a `(T, N, 2)` array of pixel coordinates, ready for rendering.

---

### **6. Data-flow & File Artefacts**

The following intermediate NumPy files may be saved to the output directory for debugging and analysis:

*   `raw_landmarks_3d.npy`
*   `smooth_landmarks_3d.npy`
*   `camera_R_seq.npy` (rotation matrices)
*   `camera_t_seq.npy` (translation vectors)

---

### **7. Configuration**

The following configuration block shall be added to `configs/default.yaml`.

```yaml
smoothing_cfg:
  kind: kalman_rts
  # -- Process noise: uncertainty in the motion model (expected acceleration).
  process_noise: 0.01                 # (m/s^2)^2, corresponds to σ=0.1 m/s^2.
  # -- Measurement noise: uncertainty in the MediaPipe detection.
  measurement_noise: 4.0e-4           # metres^2, corresponds to σ=20mm.
  adapt_R_by_visibility: true
  adapt_R_by_visibility_factor: 1.0   # Multiplier for visibility adaptation.

projection_cfg:
  # -- Parameters for Perspective-n-Point (PnP) camera pose estimation.
  pnp_visibility_threshold: 0.8       # Minimum landmark visibility to be used in PnP.
  pnp_ransac_iterations: 100
  pnp_ransac_reprojection_error: 8.0  # RANSAC reprojection error threshold in pixels.```

---

### **8. Testing Strategy**

*   **Kalman Unit Tests:**
    1.  Create a static 100-frame clip with added Gaussian noise (`σ=0.02 m`). Assert that the RMS jitter of the output is reduced by at least 70% compared to the input.
    2.  Process a clip with random `Δt` spacing (±5 ms jitter). Assert that the output is identical within 1% to the output from a fixed-`Δt` variant.

*   **Projection Unit Test:**
    1.  Create a known set of 3D points and camera parameters (`R`, `t`, `K`). Project the points using the `project_points` function and assert that the resulting 2D coordinates match expected values within a small tolerance.

*   **Integration Test:**
    1.  Run the full pipeline (`track.py --smooth kalman_rts`) on the provided `tests/synthetic_turn.mp4`.
    2.  Manually inspect the first, middle, and last frames of the output video to ensure the smoothed skeleton plausibly overlays the subject.

---

### **9. Functional / CLI**

*   `--smooth {none|one_euro|kalman_fwd|kalman_rts}`: Selects the smoothing algorithm.
*   `--compare`: A boolean flag. If present, render both the raw and the smoothed skeletons on the output video for direct, qualitative comparison.

---

### **10. Documentation**

The project's `README.md` file shall be updated in the usage section for this feature with the following notice:

> **Note on Latency**: The `kalman_rts` smoother is a zero-lag *offline* smoother. It adds processing latency because it must process the entire video clip before producing a final result. It is not suitable for live or real-time applications.
