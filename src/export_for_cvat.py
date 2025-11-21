#!/usr/bin/env python3
"""
CVAT Export Module.
Converts detection CSV to YOLO format for CVAT annotation tool.
"""

import pandas as pd
import zipfile
from pathlib import Path
import cv2


def normalize_bbox(x1, y1, x2, y2, img_width, img_height):
    """
    Convert bounding box from absolute coordinates to YOLO normalized format.
    
    YOLO format: class x_center y_center width height (all normalized 0-1)
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        img_width, img_height: Image dimensions
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized
    """
    # Calculate center and dimensions
    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def export_to_yolo_format(detections_csv_path, video_path, output_dir, 
                          min_confidence=None):
    """
    Convert detection CSV to YOLO format for CVAT.
    
    Args:
        detections_csv_path: Path to detections.csv
        video_path: Path to original video (needed for image dimensions)
        output_dir: Output directory for YOLO format files
        min_confidence: Optional minimum confidence threshold to filter detections (0.0-1.0)
                       If None, uses all detections from CSV
    """
    # Read detections
    print(f"Reading detections from: {detections_csv_path}")
    try:
        df = pd.read_csv(detections_csv_path)
        
        # Apply confidence filter if specified
        if min_confidence is not None:
            initial_count = len(df)
            df = df[df['confidence_score'] >= min_confidence].copy()
            filtered_count = len(df)
            print(f"Filtered detections: {initial_count} -> {filtered_count} (confidence >= {min_confidence})")
    except pd.errors.EmptyDataError:
        print("Warning: Detections CSV file is empty. No detections to export.")
        # Create empty structure for consistency
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        names_file = output_dir / "obj.names"
        with open(names_file, 'w') as f:
            pass  # Create empty file
        print(f"Created empty obj.names file: {names_file}")
        # Create empty zip file
        zip_path = output_dir / "cvat_annotations.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(names_file, names_file.name)
        print(f"Created empty CVAT zip file: {zip_path}")
        return
    
    if df.empty:
        print("Warning: No detections found in CSV file.")
        # Create empty structure for consistency
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        names_file = output_dir / "obj.names"
        with open(names_file, 'w') as f:
            pass  # Create empty file
        print(f"Created empty obj.names file: {names_file}")
        # Create empty zip file
        zip_path = output_dir / "cvat_annotations.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(names_file, names_file.name)
        print(f"Created empty CVAT zip file: {zip_path}")
        return
    
    # Get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video dimensions: {img_width}x{img_height}")
    
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    obj_train_data_dir = output_dir / "obj_train_data"
    obj_train_data_dir.mkdir(exist_ok=True)
    
    # Get unique class names and create obj.names file
    unique_classes = sorted(df['object_class_name'].unique())
    class_to_id = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    
    names_file = output_dir / "obj.names"
    with open(names_file, 'w') as f:
        for class_name in unique_classes:
            f.write(f"{class_name}\n")
    print(f"Created class names file: {names_file} ({len(unique_classes)} classes)")
    
    # Group detections by frame
    grouped = df.groupby('frame_number')
    
    print("Converting detections to YOLO format...")
    for frame_number, frame_detections in grouped:
        # Create frame text file
        frame_file = obj_train_data_dir / f"frame_{frame_number:06d}.txt"
        
        with open(frame_file, 'w') as f:
            for _, detection in frame_detections.iterrows():
                # Get class ID (YOLO format uses index in obj.names)
                class_id = class_to_id[detection['object_class_name']]
                
                # Normalize bounding box
                x_center, y_center, width, height = normalize_bbox(
                    detection['bbox_x1'],
                    detection['bbox_y1'],
                    detection['bbox_x2'],
                    detection['bbox_y2'],
                    img_width,
                    img_height
                )
                
                # Write YOLO format line: class x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created {len(grouped)} frame annotation files in {obj_train_data_dir}")
    
    # Create zip file for easy CVAT upload
    zip_path = output_dir / "cvat_annotations.zip"
    print(f"Creating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add obj.names
        zipf.write(names_file, names_file.name)
        
        # Add all frame annotation files
        for frame_file in sorted(obj_train_data_dir.glob("*.txt")):
            zipf.write(frame_file, f"obj_train_data/{frame_file.name}")
    
    print(f"CVAT export complete! Zip file ready: {zip_path}")


def main():
    """Main function to run CVAT export."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    detections_csv_path = base_dir / "outputs" / "detections" / "detections.csv"
    video_path = base_dir / "inputs" / "input_video.mp4"
    output_dir = base_dir / "outputs" / "cvat_prep"
    
    # Check if detections CSV exists
    if not detections_csv_path.exists():
        raise FileNotFoundError(
            f"Detections CSV not found: {detections_csv_path}\n"
            "Please run detect_objects.py first."
        )
    
    # Check if video exists
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "Required for getting image dimensions."
        )
    
    # Export to YOLO format
    # Optional: Add min_confidence parameter to further filter detections
    # Example: min_confidence=0.5 to only export high-confidence detections
    export_to_yolo_format(detections_csv_path, video_path, output_dir, 
                          min_confidence=None)  # Set to 0.4, 0.5, etc. to filter


if __name__ == "__main__":
    main()

