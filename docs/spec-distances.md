
https://aistudio.google.com/prompts/1ofIB_s2DJu8sjHJlGeQLUvsbUzb4byoH

## **Software Specification: Skier Pose Metrics Extension**

### **1. Overview**

This document outlines the requirements for extending the existing skier pose detection project (`chrhansen/poser`) with functionality to calculate, store, and visualize key performance metrics. The primary goal is to analyze a skier's technique by measuring the distances between their knees and ankles throughout a video.

The new features to be implemented are:
*   **Metric Calculation:** For each frame of a video, calculate the 3D Euclidean distance between the skier's knees and the 3D Euclidean distance between the skier's ankles.
*   **Data Storage:** For each processed video, generate two CSV files in an `output/` directory:
    *   One file containing the calculated knee and ankle distances for each frame.
    *   A second, more detailed file containing the full 3D coordinates and visibility of all detected pose landmarks for each frame.
*   **Data Visualization:**
    *   An offline script to generate a graph from the distances CSV file, plotting knee and ankle distance over time.
    *   A real-time visualization of the same graph, updated and displayed as the video is being processed.

### **2. System Architecture & Module Design**

To maintain a clean and modular codebase, the new functionality will be encapsulated in new modules and integrated into the existing structure.

*   **`metrics/` (New Directory):**
    *   `__init__.py`
    *   **`calculator.py` (New File):** Will contain the `PoseMetricsCalculator` class responsible for all distance calculations.
    *   **`storage.py` (New File):** Will contain the `MetricsLogger` class to handle the writing of landmark and distance data to CSV files.

*   **`visualization/` (New Directory):**
    *   `__init__.py`
    *   **`plotter.py` (New File):** Will contain the `MetricsPlotter` class for both real-time and offline graph generation.

*   **Modified Files:**
    *   **`video_processor.py`:** Will be updated to orchestrate the new modules. It will initialize the calculator, logger, and plotter, and call their respective methods in the main video processing loop.
    *   **`main.py`:** Will be updated with new command-line arguments to control the new features (e.g., enabling/disabling real-time plotting).

### **3. Detailed Functional Requirements**

#### **3.1. Landmark Definitions**

The distance calculations depend on accurately identifying the knee and ankle landmarks from the pose detection models.

*   **MediaPipe Landmarks:** The MediaPipe Pose Landmarker provides 33 3D landmarks. The required landmark indices are:
    *   `LEFT_KNEE`: 25
    *   `RIGHT_KNEE`: 26
    *   `LEFT_ANKLE`: 27
    *   `RIGHT_ANKLE`: 28

*   **YOLO Pose Landmarks:** The YOLOv8-pose model used in the project outputs 17 keypoints in the COCO format. The required landmark indices are:
    *   `LEFT_KNEE`: 13
    *   `RIGHT_KNEE`: 14
    *   `LEFT_ANKLE`: 15
    *   `RIGHT_ANKLE`: 16

#### **3.2. Metric Calculation (`metrics/calculator.py`)**

A `PoseMetricsCalculator` class shall be implemented to compute the distances.

*   **Input:** The method will accept the landmark data structure provided by either the `YoloPose` or `MediapipePose` detector.
*   **3D vs. 2D Coordinates:**
    *   **MediaPipe:** Provides 3D world coordinates in meters (`pose_world_landmarks`). These should be used for the 3D distance calculation.
    *   **YOLO:** Provides 2D normalized image coordinates (`x`, `y`). The distance calculation for YOLO will be in 2D (pixels). The z-coordinate will be recorded as 0. This distinction must be clear in the output data and documentation.
*   **Euclidean Distance:** The calculation will be the standard Euclidean distance. For two points *p1(x1, y1, z1)* and *p2(x2, y2, z2)*, the distance *d* is `sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)`. Python's `math.dist()` or NumPy's `linalg.norm()` can be used for an efficient implementation.
*   **Output:** The class will return a dictionary for each frame, e.g., `{'knee_distance': 0.5, 'ankle_distance': 0.3}`. If landmarks are not detected, the distances should be `None` or `NaN`.

#### **3.3. Data Storage (`metrics/storage.py`)**

A `MetricsLogger` class will handle all file I/O operations.

*   **Initialization:** The logger will be initialized with the input video's filename to create a unique output name. It will also create an `output/` directory if one doesn't exist.
*   **File Formats:**
    1.  **Distances File:**
        *   **Filename:** `output/<video_name>_distances.csv`
        *   **Columns:** `frame_number,timestamp_ms,knee_distance,ankle_distance`
    2.  **Landmarks File:**
        *   **Filename:** `output/<video_name>_landmarks.csv`
        *   **Columns:** `frame_number,timestamp_ms,landmark_index,landmark_name,x,y,z,visibility`
        *   The `landmark_name` (e.g., 'LEFT_KNEE') should be included for readability.
        *   For YOLO data, `z` and `visibility` columns can be left empty or set to 0.
*   **Functionality:** The class will have methods to `log_distances` and `log_all_landmarks` for each frame, appending a new row to the corresponding CSV file. A `close()` method should ensure files are properly saved.

#### **3.4. Visualization (`visualization/plotter.py`)**

A `MetricsPlotter` class will be responsible for creating the graphs. `matplotlib` is the recommended library.

*   **Offline Graph Generation:**
    *   A new standalone script, `generate_graph.py`, will use this class.
    *   **Input:** Path to a `_distances.csv` file.
    *   **Functionality:**
        *   Reads the CSV data using pandas.
        *   Creates a plot with 'Time (ms)' on the x-axis and 'Distance' on the y-axis.
        *   Plots both 'Knee Distance' and 'Ankle Distance' as separate lines with a clear legend.
        *   Saves the plot as a PNG image to `output/<video_name>_distances_graph.png`.

*   **Real-time Graphing:**
    *   **Functionality:** The `MetricsPlotter` will have methods to initialize an interactive plot (`init_realtime_plot`) and update it with new data for each frame (`update_realtime_plot`).
    *   **Implementation:** `matplotlib.pyplot.ion()` can be used for interactive mode. The plot should be displayed in a separate, non-blocking window. To maintain performance, the plot should update at a controlled interval (e.g., every 5 frames) rather than on every single frame.
    *   The real-time plot will be managed within the `VideoProcessor` loop.

### **4. Integration with Existing Code**

#### **4.1. `video_processor.py` Modifications**

The `VideoProcessor` class will be the central point of integration.

```python
# At the top of video_processor.py
from metrics.calculator import PoseMetricsCalculator
from metrics.storage import
