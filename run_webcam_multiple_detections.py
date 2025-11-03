#!/usr/bin/env python3
"""
Run webcam detection with multiple object detections display
"""

import sys
import os
sys.path.append('.')

from ultimate_small_object_detector import UltimateSmallObjectTrainer

def main():
    """Run webcam detection with enhanced multiple detections display"""

    print("Starting Webcam Detection with Multiple Object Support")
    print("=" * 60)

    # Initialize trainer
    trainer = UltimateSmallObjectTrainer(model_path='yolov8x.pt', device_id=0)


    model_path = 'yolov8x.pt'  # Default pre-trained model

    # Run webcam detection with multiple detections enabled
    print("Starting real-time detection...")
    print("Features:")
    print("- Multiple object detection")
    print("- Color-coded by size (Red: small, Green: large)")
    print("- Detection numbers and confidence scores")
    print("- Real-time statistics")
    print("- Press 'q' to quit")
    print()

    trainer.detect_from_webcam(
        model_path=model_path,
        camera_id=0,
        conf_threshold=0.1,  # Much lower threshold to ensure detections
        iou_threshold=0.45,
        max_det=1000  # Allow many detections
    )

if __name__ == "__main__":
    main()
