#!/usr/bin/env python3
"""
Run pipeline with different YOLOv8 models for comparison.
Runs with yolov8s.pt (small) and yolov8x.pt (xlarge).
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcribe import main as transcribe_main
from detect_objects import detect_objects_in_video
from export_for_cvat import main as export_cvat_main


def run_with_model(model_name, model_size):
    """Run detection with a specific model."""
    print("=" * 70)
    print(f"Running with {model_name} ({model_size})")
    print("=" * 70)
    print()
    
    base_dir = Path(__file__).parent
    video_path = base_dir / "inputs" / "input_video.mp4"
    
    # Create model-specific output paths
    output_video = base_dir / "outputs" / "detections" / f"annotated_output_{model_size}.mp4"
    output_csv = base_dir / "outputs" / "detections" / f"detections_{model_size}.csv"
    
    # Run detection
    detect_objects_in_video(
        video_path,
        model_path=f"{model_name}.pt",
        output_video_path=output_video,
        output_csv_path=output_csv,
        conf_threshold=0.4,
        iou_threshold=0.45,
        frame_skip=1
    )
    
    return output_csv


def main():
    """Run pipeline with transcription, then both YOLO models."""
    print("=" * 70)
    print("Multi-Modal Perception Pipeline - Model Comparison")
    print("=" * 70)
    print()
    
    try:
        # Phase 2: Audio Transcription (only once)
        print("Phase 2: Starting Transcription...")
        print("-" * 70)
        transcribe_main()
        print("✓ Transcription completed successfully")
        print()
        
        # Phase 3: Object Detection with YOLOv8 Small
        print("Phase 3a: Starting Object Detection with YOLOv8 Small...")
        print("-" * 70)
        csv_small = run_with_model("yolov8s", "small")
        print("✓ YOLOv8 Small detection completed")
        print()
        
        # Phase 3: Object Detection with YOLOv8 XLarge
        print("Phase 3b: Starting Object Detection with YOLOv8 XLarge...")
        print("-" * 70)
        csv_xlarge = run_with_model("yolov8x", "xlarge")
        print("✓ YOLOv8 XLarge detection completed")
        print()
        
        # Phase 4: CVAT Export for both models
        print("Phase 4a: Starting CVAT Export for YOLOv8 Small...")
        print("-" * 70)
        # Temporarily rename CSV for export
        base_dir = Path(__file__).parent
        original_csv = base_dir / "outputs" / "detections" / "detections.csv"
        cvat_dir_small = base_dir / "outputs" / "cvat_prep" / "small"
        
        # Create a temporary symlink or copy for CVAT export
        import shutil
        shutil.copy(csv_small, original_csv)
        
        # Modify export_for_cvat to use custom output dir
        from export_for_cvat import export_to_yolo_format
        video_path = base_dir / "inputs" / "input_video.mp4"
        export_to_yolo_format(original_csv, video_path, cvat_dir_small)
        print("✓ CVAT export for Small completed")
        print()
        
        print("Phase 4b: Starting CVAT Export for YOLOv8 XLarge...")
        print("-" * 70)
        cvat_dir_xlarge = base_dir / "outputs" / "cvat_prep" / "xlarge"
        shutil.copy(csv_xlarge, original_csv)
        export_to_yolo_format(original_csv, video_path, cvat_dir_xlarge)
        print("✓ CVAT export for XLarge completed")
        print()
        
        # Restore original CSV name
        original_csv.unlink()
        
        print("=" * 70)
        print("Pipeline execution completed successfully!")
        print("=" * 70)
        print()
        print("Output files:")
        print("  - Transcripts: outputs/transcripts/transcription.csv")
        print("  - Detections (Small): outputs/detections/detections_small.csv")
        print("  - Detections (XLarge): outputs/detections/detections_xlarge.csv")
        print("  - Annotated Video (Small): outputs/detections/annotated_output_small.mp4")
        print("  - Annotated Video (XLarge): outputs/detections/annotated_output_xlarge.mp4")
        print("  - CVAT Export (Small): outputs/cvat_prep/small/cvat_annotations.zip")
        print("  - CVAT Export (XLarge): outputs/cvat_prep/xlarge/cvat_annotations.zip")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

