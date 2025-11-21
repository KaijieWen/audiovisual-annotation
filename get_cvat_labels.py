#!/usr/bin/env python3
"""
Helper script to extract labels from obj.names file for easy copy-paste into CVAT.

Usage:
    python get_cvat_labels.py [small|xlarge]
    
If no argument is provided, it will show labels for both models.
"""

import sys
from pathlib import Path

def get_labels(model_type):
    """Extract labels from obj.names file."""
    base_dir = Path(__file__).parent
    names_file = base_dir / "outputs" / "cvat_prep" / model_type / "obj.names"
    
    if not names_file.exists():
        return None
    
    with open(names_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    return labels

def print_labels(labels, model_type):
    """Print labels in a CVAT-friendly format."""
    print(f"\n{'='*60}")
    print(f"Labels for {model_type.upper()} model ({len(labels)} labels):")
    print(f"{'='*60}")
    print("\nCopy these labels one by one into CVAT:")
    print("-" * 60)
    for i, label in enumerate(labels, 1):
        print(f"{i:2d}. {label}")
    print("-" * 60)
    print(f"\nTotal: {len(labels)} labels")
    print("\nIn CVAT, add each label with:")
    print("  - Name: (copy from above)")
    print("  - Shape: Rectangle")
    print("="*60)

def main():
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type not in ['small', 'xlarge']:
            print(f"Error: Model type must be 'small' or 'xlarge', got '{model_type}'")
            sys.exit(1)
        
        labels = get_labels(model_type)
        if labels is None:
            print(f"Error: Could not find obj.names file for {model_type} model.")
            print(f"Expected: outputs/cvat_prep/{model_type}/obj.names")
            sys.exit(1)
        
        print_labels(labels, model_type)
    else:
        # Show both
        for model_type in ['small', 'xlarge']:
            labels = get_labels(model_type)
            if labels is not None:
                print_labels(labels, model_type)
            else:
                print(f"\n⚠️  Warning: Could not find labels for {model_type} model")
                print(f"   Expected: outputs/cvat_prep/{model_type}/obj.names")

if __name__ == "__main__":
    main()


