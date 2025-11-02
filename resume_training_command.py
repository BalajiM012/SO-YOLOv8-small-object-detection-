#!/usr/bin/env python3
"""
Command-line interface for resuming YOLO training and saving checkpoints every 10 epochs.

Usage examples:
1. Resume from checkpoint and train for 50 epochs:
   python resume_training_command.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 50

2. Fresh training with custom parameters:
   python resume_training_command.py --epochs 100 --dataset voc2012.yaml --device cpu

3. Use GPU-optimized training:
   python gpu_optimized_training.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 100

4. Use custom enhanced training:
   python custom_yolo_training.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 100 --test

Available checkpoints:
- models/checkpoints/test_checkpoint_epoch_5
- models/checkpoints/resume_test_checkpoint
"""

import os
import sys
import argparse

def main():
    print("YOLO Training Resume Command Interface")
    print("=" * 50)
    print("This script provides examples of how to resume training with checkpoint saving.")
    print("Use the training scripts directly with command-line arguments.")
    print()

    # Parse arguments
    parser = argparse.ArgumentParser(description='YOLO Training Resume Command Interface')
    parser.add_argument('--list-checkpoints', action='store_true', help='List available checkpoints')
    parser.add_argument('--example', type=str, help='Show example command for specific training script')

    args = parser.parse_args()

    if args.list_checkpoints:
        print("Available checkpoints:")
        checkpoints_dir = "models/checkpoints"
        if os.path.exists(checkpoints_dir):
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.pt'):
                    print(f"  - {checkpoints_dir}/{file}")
        else:
            print("  No checkpoints directory found")
        return

    if args.example:
        examples = {
            'gpu': """
GPU-Optimized Training Examples:
1. Resume from checkpoint:
   python gpu_optimized_training.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 100

2. Fresh training:
   python gpu_optimized_training.py --epochs 50 --dataset voc2012.yaml

3. CPU training:
   python gpu_optimized_training.py --epochs 30 --device cpu
""",
            'custom': """
Custom Enhanced Training Examples:
1. Resume with testing:
   python custom_yolo_training.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 100 --test

2. Custom dataset:
   python custom_yolo_training.py --epochs 50 --dataset custom.yaml --model yolov8x.pt

3. CPU training with custom test images:
   python custom_yolo_training.py --epochs 30 --device cpu --test --test-images ./custom_test_images --max-test-images 100
"""
        }

        if args.example in examples:
            print(examples[args.example])
        else:
            print(f"Available examples: {', '.join(examples.keys())}")
        return

    # Default help
    print("Usage examples:")
    print()
    print("1. List available checkpoints:")
    print("   python resume_training_command.py --list-checkpoints")
    print()
    print("2. Show GPU training examples:")
    print("   python resume_training_command.py --example gpu")
    print()
    print("3. Show custom training examples:")
    print("   python resume_training_command.py --example custom")
    print()
    print("Direct usage examples:")
    print("   python gpu_optimized_training.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 100")
    print("   python custom_yolo_training.py --resume models/checkpoints/test_checkpoint_epoch_5 --epochs 100 --test")

if __name__ == "__main__":
    main()
