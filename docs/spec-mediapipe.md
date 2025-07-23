From: https://aistudio.google.com/prompts/1kEtDHP6BuRqJun66J2yr3EbRWzTwGLJG

### **Software Specification Document: Dual Pose Detection Engine Integration**

**Project:** Skier Pose Detection
**Repository:** `https://github.com/chrhansen/poser`
**Date:** July 23, 2025


### 1. Introduction

This document specifies the work required to enhance the existing skier pose detection project. The primary objective is to integrate Google's MediaPipe Pose Landmarker as a second, selectable pose detection engine. The existing YOLO-Pose implementation will be retained, creating a dual-engine system. This enhancement aims to leverage the respective strengths of each model and provide flexibility to the user.

The system must intelligently handle the different input requirements and output coordinate systems of both YOLO-Pose and MediaPipe to ensure optimal performance and correct visualization on the original video.

### 2. High-Level Requirements

1.  **Dual Detector Support:** The application must support both the existing YOLO-Pose model and the new MediaPipe Pose Landmarker.
2.  **Runtime Selection:** The user must be able to select which pose detection engine to use via a command-line argument when running the main script (`track.py`).
3.  **Input Optimization:** The code must be adapted to provide each detection engine with its preferred input format, which involves specific image sizing and aspect ratios.
4.  **Coordinate Space Unification:** Pose keypoints detected by either engine must be correctly mapped back to the original video's coordinate space for accurate analysis and visualization.

### 3. Detailed Specifications

#### 3.1. Command-Line Interface (CLI)

The entry point script, `track.py`, must be updated to include a new command-line argument to select the pose detector.

*   **Argument:** `--pose-detector`
*   **Values:**
    *   `yolo`: (Default) Uses the existing YOLO-Pose implementation.
    *   `mediapipe`: Uses the new MediaPipe Pose Landmarker implementation.
*   **Example Usage:**
    ```bash
    # Run with the new MediaPipe detector
    python track.py --source <path_to_video> --pose-detector mediapipe

    # Run with the existing YOLO detector (default behavior)
    python track.py --source <path_to_video>
    ```

#### 3.2. Pose Detector Abstraction Layer

To ensure a clean and maintainable codebase, a software abstraction layer shall be introduced for the pose detectors.

*   A base class or interface should be defined (e.g., `PoseDetector`). This interface must define a common method signature, for example, `detect(frame, roi_bounding_box)`.
*   Two concrete classes will implement this interface:
    1.  `YOLOPoseDetector`: This class will encapsulate the existing YOLO-Pose logic.
    2.  `MediaPipePoseDetector`: This class will encapsulate the new MediaPipe Pose Landmarker logic.
*   In `track.py`, a factory function or conditional block will instantiate the correct detector class based on the `--pose-detector` argument.

#### 3.3. Engine-Specific Input Pre-processing

The core challenge is that each engine has different optimal input requirements. The `detect` method of each respective class must be responsible for preparing the image data accordingly.

*   **For `YOLOPoseDetector`:**
    *   The current implementation, which may resize the input to a preferred width (e.g., 768 pixels), should be encapsulated within this class. The logic must record any scaling or padding applied to the input image.

*   **For `MediaPipePoseDetector`:**
    *   MediaPipe models often perform best on smaller, square images. The implementation should process the input Region of Interest (ROI) as follows:
        1.  From the original video frame, crop the skier ROI provided by the object tracker.
        2.  Convert this ROI into a square image by adding padding (e.g., black bars) to the shorter dimension to make the width and height equal.
        3.  Resize this square ROI to a fixed resolution, such as 256x256 pixels.
        4.  Keep track of all transformation parameters (original ROI coordinates, padding added, resize scale factor) as they are critical for the post-processing step.

#### 3.4. MediaPipe Pose Landmarker Implementation

*   **Model:** Use the `pose_landmarker_heavy.task` model for its high accuracy. This model file should be included in the repository, for instance in a `models/` directory.
*   **Initialization:** The `MediaPipePoseDetector` class will initialize the landmarker in its constructor, configured for video processing.

    ```python
    # Example for MediaPipePoseDetector.__init__
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_heavy.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1 # Assuming we are tracking a single skier in the ROI
    )
    self.detector = vision.PoseLandmarker.create_from_options(options)
    ```

*   **Detection:** The detection call will use the `detect_for_video` method.

    ```python
    # Example for MediaPipePoseDetector.detect
    # roi_image is the pre-processed 256x256 square image
    # timestamp_ms is the video frame timestamp in milliseconds

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_image)
    detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
    ```

#### 3.5. Output Post-processing and Coordinate Mapping

This is a critical step. The keypoints returned by each engine are relative to the *processed input image* (e.g., the 768px wide image for YOLO or the 256x256 image for MediaPipe). They must be transformed back to the coordinate system of the *original, full-size video frame*.

*   Each detector class (`YOLOPoseDetector`, `MediaPipePoseDetector`) must implement the logic to reverse its specific pre-processing transformations.
*   **Example for MediaPipe:**
    1.  The detected landmarks (which are normalized from 0.0 to 1.0) must first be scaled to the 256x256 pixel space.
    2.  These pixel coordinates must then be adjusted to account for the padding that was added to make the ROI square.
    3.  Finally, these coordinates must be scaled and translated back to their correct position within the full original video frame, using the initial ROI bounding box coordinates.
*   **Unified Output:** The `detect` method of both classes must return the final keypoints in a standardized format and in the original frame's coordinate space. This ensures the rest of the application (e.g., the drawing/visualization module) can use the results transparently, without needing to know which engine produced them.

#### 3.6. Dependencies

The project's dependency file (`requirements.txt` or equivalent) must be updated to include `mediapipe`.

```
# requirements.txt
...
mediapipe>=0.10.9
...
```

### 4. Acceptance Criteria

1.  The application runs successfully using both `--pose-detector yolo` and `--pose-detector mediapipe` flags.
2.  When using `--pose-detector mediapipe`, the `pose_landmarker_heavy.task` model is loaded and utilized.
3.  For both detectors, the pre-processing logic correctly prepares the input image to the engine's specifications.
4.  For both detectors, the detected pose landmarks are correctly drawn onto the skier in the final output video, confirming that the coordinate space transformation is accurate.
5.  The code is refactored into the specified `PoseDetector` abstract structure.
6.  The `mediapipe` package is added as a project dependency.
