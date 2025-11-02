#!/usr/bin/env python3
"""
Simple script to run the combined YOLO training system
Usage: python run_combined_training.py
"""

import os
import sys
import torch

def main():
    """Run the combined YOLO training system"""
    print("Starting Combined YOLO Training System...")
    print("="*50)

    # Check if combined_yolo_training.py exists
    if not os.path.exists('combined_yolo_training.py'):
        print("Error: combined_yolo_training.py not found!")
        sys.exit(1)

    # Run the training system
    print("Launching training interface...")
    print("Available datasets:")
    print("1. PASCAL VOC 2012 - General object detection")
    print("2. TinyPerson - Small person detection")
    print("3. VisDrone Dataset - Drone-based object detection")
    print("4. VisDrone Dataset 2 - Additional drone detection")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Execute the training script on GPU
    os.system('python combined_yolo_training.py --device cuda')

if __name__ == "__main__":
    main()
