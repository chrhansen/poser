# Skier Detection and Tracking with YOLOv11

This project provides a high-performance Python script that uses YOLOv11 object detection to locate and track skiers in video files. The script processes videos frame-by-frame, identifies skiers, and draws persistent bounding boxes with unique tracker IDs.

## Visual Example

<table>
<tr>
<td align="center"><b>Before</b></td>
<td align="center"><b>After</b></td>
</tr>
<tr>
<td><img src="frame_before.png" alt="Original frame" width="400"/></td>
<td><img src="frame_after.png" alt="Frame with tracking" width="400"/></td>
</tr>
</table>

## Features

- YOLOv11 Nano model for fast and accurate person detection
- ByteTrack algorithm for robust multi-object tracking
- Automatic output file naming with `_with_box` suffix
- Optimized for macOS Sequoia (v15.5)

## Installation

### 1. Clone the repository and navigate to the project directory
```bash
cd /path/to/poser
```

### 2. Create a Python virtual environment
```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install the required dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `ultralytics` - YOLOv11 implementation
- `supervision` - Video processing and tracking
- `numpy` - Numerical operations
- `opencv-python` - Computer vision operations (installed as dependency)
- `torch` and `torchvision` - Deep learning framework (installed as dependencies)

## First Run - Model Download

The YOLOv11 model will be automatically downloaded on the first run. The script uses `yolo11n.pt` (YOLOv11 Nano), which is approximately 5.35MB and will be saved in the project root directory.

## Usage

### Basic Command
```bash
python3 track_skier.py --source /path/to/your/video.mp4
```

### Example
```bash
# Process a video named "ski_run.mp4"
python3 track_skier.py --source ski_run.mp4

# Process a video with full path
python3 track_skier.py --source /Users/username/Videos/ski_competition.mov
```

### Output File

The output video will be saved in the **same directory** as the input file with `_with_box` added before the file extension.

**Examples:**
- Input: `ski_run.mp4` → Output: `ski_run_with_box.mp4`
- Input: `test-ski-video2.mov` → Output: `test-ski-video2_with_box.mov`
- Input: `/path/to/videos/alpine.mp4` → Output: `/path/to/videos/alpine_with_box.mp4`

## What to Expect

When you run the script:

1. **First run**: The YOLOv11 model will download automatically (one-time download)
2. **Processing**: You'll see frame-by-frame detection results in the terminal
3. **Output**: Each detected person will have:
   - A bounding box drawn around them
   - A unique tracker ID (e.g., "ID: 1", "ID: 2")
   - Persistent tracking even if temporarily occluded

### Terminal Output Example
```
Loading YOLOv11 model (will auto-download if needed)...
Processing video: ski_run.mp4
Output will be saved to: ski_run_with_box.mp4

0: 640x544 1 person, 1 skis, 45.2ms
0: 640x544 1 person, 1 skis, 43.6ms
...
Processing complete! Output saved to: ski_run_with_box.mp4
```

## Supported Video Formats

The script supports all common video formats including:
- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`

## Performance Notes

The script is optimized for maximum processing speed by:
- Using the lightweight YOLOv11 Nano model
- Focusing solely on detection and tracking (no pose estimation)
- Efficient frame-by-frame processing with supervision library

Processing speed depends on:
- Video resolution
- Number of people in the frame
- Hardware capabilities (GPU acceleration is used if available)

## Troubleshooting

### Virtual Environment Not Activated
If you see "ModuleNotFoundError", ensure the virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
```

### Model Download Issues
If the model fails to download automatically, you can manually download it:
```bash
python3 -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt')"
```

### Video Not Found
Ensure you're providing the correct path to your video file. Use absolute paths if relative paths don't work:
```bash
python3 track_skier.py --source /absolute/path/to/video.mp4
```