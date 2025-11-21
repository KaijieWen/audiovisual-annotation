# audiovisual-annotation

A complete Python pipeline for processing video and audio to extract semantic information(transcriptions and object detections) and format data for CVAT integration

## Overview

This pipeline processes raw video and audio files to:
1. **Transcribe speech** using OpenAI Whisper
2. **Detect objects** using YOLOv8
3. **Export annotations** in YOLO format for CVAT

## Directory Structure

```
audiovisual-annotation/
├── inputs/              # Place your raw video and audio files here
├── outputs/
│   ├── transcripts/    # CSV outputs from Whisper
│   ├── detections/     # Annotated video & CSV detection logs
│   └── cvat_prep/      # Formatted files for annotation tools
├── src/                # Python scripts
│   ├── transcribe.py
│   ├── detect_objects.py
│   └── export_for_cvat.py
├── main.py             # Master orchestrator script
├── setup_placeholders.py  # Generate dummy media for testing
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Section 1: Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Navigate to the pipeline directory:
   ```bash
   cd audiovisual-annotation
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This will install:
   - `ultralytics` (for YOLOv8)
   - `openai-whisper` (for audio transcription)
   - `pandas` (for data processing)
   - `opencv-python-headless` (for video processing)
   - `ffmpeg-python` (for audio/video handling)
   - `torch` (PyTorch - CPU version is acceptable if CUDA is unavailable)

3. (Optional) Generate placeholder media files for testing:
   ```bash
   python setup_placeholders.py
   ```

   This creates dummy 5-second video and audio files if they don't exist.

## Section 2: Usage

### Preparing Your Media Files

1. **Replace the dummy files** in the `inputs/` directory with your actual media:
   - Place your video file as `inputs/input_video.mp4`
   - Place your audio file as `inputs/input_audio.mp3` (or `input_audio.wav`)

   **Note:** Whisper supports both MP3 and WAV formats. If you have a WAV file, name it `input_audio.wav`.

### Running the Pipeline

Execute the main orchestrator script:

```bash
python main.py
```

This will run the complete pipeline:
1. **Transcription** - Processes audio and generates `outputs/transcripts/transcription.csv`
2. **Object Detection** - Processes video and generates:
   - `outputs/detections/annotated_output.mp4` (annotated video)
   - `outputs/detections/detections.csv` (detection log)
3. **CVAT Export** - Converts detections to YOLO format:
   - `outputs/cvat_prep/cvat_annotations.zip` (ready for CVAT upload)

### Running Individual Modules

You can also run each module independently:

```bash
# Transcription only
python src/transcribe.py

# Object detection only
python src/detect_objects.py

# CVAT export only (requires detections.csv)
python src/export_for_cvat.py
```

## Section 3: CVAT Integration Guide

This section provides step-by-step instructions for using the exported annotations with CVAT (Computer Vision Annotation Tool).

### Step 1: Access CVAT

1. Go to [CVAT.ai](https://cvat.ai) or your CVAT instance
2. Sign in or create an account

### Step 2: Create a New Task

1. Click **"Tasks"** in the navigation menu
2. Click **"Create new task"** button
3. Fill in the task details:
   - **Name**: Give your task a descriptive name
   - **Labels**: You can add labels here, but they will be imported from the annotation file
4. In the **"Files"** section:
   - Click **"Select files"** or drag and drop
   - Upload your video file: `inputs/input_video.mp4`
5. Click **"Submit"** to create the task

### Step 3: Upload Annotations

1. Open the task you just created
2. Click on the **"Actions"** menu (three dots) → **"Upload annotations"**
3. Select **"YOLO 1.1"** format from the dropdown
4. Click **"Select file"** and upload: `outputs/cvat_prep/cvat_annotations.zip`
5. Click **"Upload"**

The annotations will be imported and displayed on your video frames.

### Step 4: Manual Correction

1. Review the imported annotations frame by frame
2. **Correct 3-5 frames** as needed:
   - Click on a bounding box to select it
   - Drag to move or resize
   - Right-click for options (delete, change class, etc.)
   - Use the annotation tools to add missing detections
3. Save your corrections (they are auto-saved in CVAT)

### Step 5: Export Annotations

1. After making corrections, click **"Actions"** → **"Dump annotations"**
2. Select your preferred format (YOLO 1.1, COCO, Pascal VOC, etc.)
3. Click **"Download"** to save the corrected annotations

## Output File Formats

### Transcription CSV (`outputs/transcripts/transcription.csv`)

Columns:
- `start_time` (float): Start time in seconds
- `end_time` (float): End time in seconds
- `text_content` (string): Transcribed text

### Detections CSV (`outputs/detections/detections.csv`)

Columns:
- `frame_number` (int): Frame number in video
- `timestamp` (float): Timestamp in seconds (calculated from FPS)
- `object_class_id` (int): YOLO class ID
- `object_class_name` (string): Object class name
- `confidence_score` (float): Detection confidence (0.0-1.0)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` (int): Bounding box coordinates

### CVAT Export (`outputs/cvat_prep/cvat_annotations.zip`)

Contains:
- `obj.names`: List of class names (one per line)
- `obj_train_data/`: Directory with one `.txt` file per frame
  - Each file contains YOLO format annotations: `class x_center y_center width height` (all normalized 0-1)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure you've installed all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Video/Audio file not found**: 
   - Ensure your files are in the `inputs/` directory
   - Check file names match exactly: `input_video.mp4` and `input_audio.mp3` (or `.wav`)

3. **CUDA/GPU issues**: 
   - The pipeline works with CPU-only PyTorch
   - For faster processing, install CUDA-enabled PyTorch if you have a compatible GPU

4. **Whisper model download**: 
   - First run will download the Whisper base model (~150MB)
   - Ensure you have internet connection

5. **YOLOv8 model download**: 
   - First run will download yolov8n.pt (~6MB)
   - Ensure you have internet connection

## License

This pipeline is provided as-is for research and development purposes.

