Of course. Here is the complete, self-contained software development specification.

***

## **Software Development Specification: Skier Detection and Tracking Script**

### **1. Introduction and Goals**

The primary goal of this project is to develop a high-performance Python script that utilizes the YOLOv11 object detection model to locate and track a skier within a video file. The script must process a video, identify the skier in each frame, draw a persistent bounding box around them, and save the annotated video as a new file.

The key focus is on achieving the highest possible processing speed and tracking efficiency. The system must be robust enough to handle common visual challenges, such as the skier being temporarily occluded by terrain or other objects, and changes in the skier's size as they move relative to the camera.

This specification outlines the requirements and implementation details for an AI developer to build this script.

### **2. References and Resources**

The following resources provide context and technical background for the project:

*   **Ultralytics YOLO Documentation:** `https://docs.ultralytics.com`
*   **Ultralytics GitHub Repository:** `https://github.com/ultralytics/ultralytics`
*   **Supervision Object Tracking Guide:** `https://supervision.roboflow.com/develop/how_to/track_objects/`
*   **Article on Real-Time Tracking Concepts:** `https://medium.com/@kevinnjagi83/real-time-object-tracking-using-yolov5-kalman-filter-hungarian-algorithm-9bd0e5a94c5a`

### **3. Functional Requirements**

*   **FR-1: Video Input:** The script must accept a single command-line argument (`--source`) that specifies the full path to the input video file.
*   **FR-2: Skier Detection:** The script will use the pre-trained **YOLOv11** model provided by the Ultralytics library to detect objects in each video frame. It will filter these detections to isolate objects of the "person" class (COCO dataset class ID 0), which will serve as the proxy for detecting a "skier".
*   **FR-3: Object Tracking:** An efficient object tracking algorithm (e.g., ByteTrack) must be implemented to assign and maintain a unique ID for each detected person across consecutive frames. This tracking must persist even if the person is temporarily lost from view.
*   **FR-4: Bounding Box Visualization:** A rectangular bounding box shall be drawn around the primary tracked person in every frame they are detected.
*   **FR-5: Tracker ID Annotation:** The bounding box for each tracked person must be annotated with their unique tracking ID (e.g., "ID: 1").
*   **FR-6: Automated Video Output:** The script must produce a new video file containing the original footage overlaid with the bounding boxes and annotations. The output filename must be derived automatically from the input filename by appending a `_with_box` suffix before the file extension. For example, an input file named `ski_run.mov` will result in an output file named `ski_run_with_box.mov`.

### **4. Non-Functional Requirements**

*   **NFR-1: Performance:** The script must be optimized for maximum processing speed (frames per second). Any operations not critical to the core detection and tracking task, such as pose estimation, are explicitly excluded.
*   **NFR-2: Operating System:** The script must be developed and tested to run on macOS Sequoia (v15.5).
*   **NFR-3: Environment Isolation:** To prevent conflicts with other Python installations, all project dependencies must be managed within a dedicated virtual environment created using `venv`.
*   **NFR-4: Python Execution:** The script must be executable using the `python3` command.

### **5. Technical Stack**

*   **Programming Language:** Python 3.9+
*   **Object Detection Model:** Ultralytics YOLOv11 Nano (`yolov11n.pt`). This model is chosen for its balance of high speed and reasonable accuracy.
*   **Core Frameworks:**
    *   `ultralytics`: For loading the YOLOv11 model and running inference.
    *   `supervision`: For handling video processing, tracking (ByteTrack), and annotation (drawing boxes and labels).
*   **Core Libraries:**
    *   `numpy`: For underlying numerical data manipulation.
    *   `pathlib`: For robust and platform-agnostic file path operations.
*   **Environment Management:** `venv`
*   **Package Installation:** `pip` using a `requirements.txt` file.

### **6. Implementation Guide**

#### **6.1. Project Structure and Setup**

The final delivery should include the following files:
```
/project-root
├── track_skier.py
├── requirements.txt
└── README.md
```

The `README.md` file must contain the following setup instructions for the end-user:
```markdown
# 1. Create a Python virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

#### **6.2. Dependency File (`requirements.txt`)**

The `requirements.txt` file must contain:
```
ultralytics
supervision[video]
numpy
```

#### **6.3. Main Script Logic (`track_skier.py`)**

The script should adhere to the following logical flow:

1.  **Imports:** Import necessary modules: `argparse`, `pathlib`, `supervision as sv`, and `ultralytics.YOLO`.
2.  **Argument Parsing:** Use `argparse` to define and parse a single required command-line argument: `--source`.
3.  **Path Management:**
    *   Use `pathlib.Path` to create a Path object from the `--source` argument.
    *   Validate that the source file exists.
    *   Construct the target output path by using `source_path.with_stem(f"{source_path.stem}_with_box")`.
4.  **Model and Tracker Initialization:**
    *   Load the YOLOv11 model: `model = YOLO('yolov11n.pt')`. The library will handle automatic download on the first run.
    *   Initialize the tracker: `tracker = sv.ByteTrack()`.
5.  **Annotator Initialization:**
    *   Set up annotators for visualization: `bounding_box_annotator = sv.BoundingBoxAnnotator()` and `label_annotator = sv.LabelAnnotator()`.
6.  **Video Processing Loop:**
    *   Use `sv.get_video_info()` to retrieve properties (width, height, fps) from the source video.
    *   Use a `with` statement to manage a `sv.VideoSink` object, providing it the target path and video info.
    *   Iterate through the video frames using `sv.get_video_frames_generator(source_path=...)`.
    *   **Inside the loop (for each frame):**
        *   Run model inference on the frame: `results = model(frame)[0]`.
        *   Convert the inference output into the `supervision` format: `detections = sv.Detections.from_ultralytics(results)`.
        *   Filter the detections to keep only the 'person' class (`detections = detections[detections.class_id == 0]`).
        *   Update the tracker with the filtered detections: `tracked_detections = tracker.update(with_detections=detections)`.
        *   Generate labels for the annotations in the format `"ID: {tracker_id}"`.
        *   Use the annotators to draw the bounding boxes and labels onto the frame.
        *   Write the annotated frame to the output video using the `VideoSink` object.
7.  **Completion Message:** After the loop finishes, print a confirmation message to the console indicating that processing is complete and where the output file has been saved.

### **7. Deliverables**

1.  **`track_skier.py`**: The fully functional and commented Python script.
2.  **`requirements.txt`**: The file listing all project dependencies.
3.  **`README.md`**: A documentation file containing clear instructions for setup and execution.

### **8. Execution Example**

The script should be run from the command line as follows:

```bash
python3 track_skier.py --source /path/to/my_skier_video.mov
```
