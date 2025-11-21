#!/usr/bin/env python3
"""
Visual Object Detection Module using YOLOv8.
Detects objects in video frames and exports annotated video and detection logs.
"""

import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def detect_objects_in_video(video_path, model_path="yolov8n.pt", 
                            output_video_path=None, output_csv_path=None,
                            conf_threshold=0.25, iou_threshold=0.45,
                            frame_skip=1):
    """
    Detect objects in video using YOLOv8 and save annotated video and CSV log.
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLOv8 model file (default: "yolov8n.pt")
                     Options: yolov8n.pt (nano, fastest), yolov8s.pt (small),
                     yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge, most accurate)
        output_video_path: Path to output annotated video
        output_csv_path: Path to output CSV detection log
        conf_threshold: Confidence threshold (0.0-1.0). Higher = fewer but more confident detections
        iou_threshold: IoU threshold for NMS (0.0-1.0). Higher = more overlapping boxes allowed
        frame_skip: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
    
    Returns:
        DataFrame with detection results
    """
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {model_path}...")
    model = YOLO(model_path)
    
    # Open video
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Detection settings: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, frame_skip={frame_skip}")
    
    # Setup video writer for annotated output
    if output_video_path:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process frames
    detections = []
    frame_number = 0
    frames_processed = 0
    
    print("Processing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        # Skip frames if frame_skip > 1
        if (frame_number - 1) % frame_skip != 0:
            # Still write the frame to output video even if we skip detection
            if output_video_path:
                out.write(frame)
            continue
        
        frames_processed += 1
        timestamp = frame_number / fps if fps > 0 else 0.0
        
        # Run detection with confidence and IoU thresholds
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                confidence = float(box.conf[0].cpu().numpy())
                
                # Store detection (already filtered by conf_threshold in model call)
                detections.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "object_class_id": class_id,
                    "object_class_name": class_name,
                    "confidence_score": confidence,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2
                })
                
                # Draw bounding box with color based on confidence
                # High confidence (>=0.7): green, Medium (0.5-0.7): yellow, Low (<0.5): red
                if confidence >= 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence >= 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f"{class_name} {confidence:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), 
                            (x1 + label_size[0], label_y), color, -1)
                cv2.putText(frame, label, (x1, label_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Write annotated frame
        if output_video_path:
            out.write(frame)
        
        # Progress update
        if frames_processed % 30 == 0:
            print(f"Processed {frames_processed} frames (frame {frame_number}/{total_frames})...")
    
    # Cleanup
    cap.release()
    if output_video_path:
        out.release()
        print(f"Annotated video saved to: {output_video_path}")
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(detections)
    
    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Detection log saved to: {output_csv_path}")
        print(f"Total detections: {len(df)}")
    
    return df


def main():
    """Main function to run object detection."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    video_path = base_dir / "inputs" / "input_video.mp4"
    output_video_path = base_dir / "outputs" / "detections" / "annotated_output.mp4"
    output_csv_path = base_dir / "outputs" / "detections" / "detections.csv"
    
    # Check if video file exists
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "Please run setup_placeholders.py first or provide your video file."
        )
    
    # Detect objects with improved settings
    # Model options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)
    detect_objects_in_video(
        video_path,
        model_path="yolov8n.pt",  # Change to yolov8s.pt, yolov8m.pt, etc. for better accuracy
        output_video_path=output_video_path,
        output_csv_path=output_csv_path,
        conf_threshold=0.4,  # Higher = fewer but more confident detections (default: 0.25)
        iou_threshold=0.45,  # IoU threshold for NMS (default: 0.45)
        frame_skip=1  # Process every frame (increase for faster processing)
    )


if __name__ == "__main__":
    main()

