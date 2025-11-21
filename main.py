#!/usr/bin/env python3
"""
Main Orchestrator Script for Audiovisual Annotation Pipeline.
Runs transcription, object detection, and CVAT export sequentially.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transcribe import main as transcribe_main
from detect_objects import main as detect_main
from export_for_cvat import main as export_cvat_main


def main():
    """Run the complete pipeline: transcription → detection → CVAT export."""
    print("=" * 60)
    print("Audiovisual Annotation Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Phase 2: Audio Transcription
        print("Phase 2: Starting Transcription...")
        print("-" * 60)
        transcribe_main()
        print("✓ Transcription completed successfully")
        print()
        
        # Phase 3: Object Detection
        print("Phase 3: Starting Object Detection...")
        print("-" * 60)
        detect_main()
        print("✓ Object detection completed successfully")
        print()
        
        # Phase 4: CVAT Export
        print("Phase 4: Starting CVAT Export...")
        print("-" * 60)
        export_cvat_main()
        print("✓ CVAT export completed successfully")
        print()
        
        print("=" * 60)
        print("Pipeline execution completed successfully!")
        print("=" * 60)
        print()
        print("Output files:")
        print("  - Transcripts: outputs/transcripts/transcription.csv")
        print("  - Detections: outputs/detections/detections.csv")
        print("  - Annotated Video: outputs/detections/annotated_output.mp4")
        print("  - CVAT Export: outputs/cvat_prep/cvat_annotations.zip")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

