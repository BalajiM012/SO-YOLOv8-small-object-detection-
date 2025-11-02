#!/usr/bin/env python3
"""
Resume YOLO Training Script
Resumes training from a checkpoint and saves checkpoints every 10 epochs
"""

import os
import sys
from gpu_optimized_training import OptimizedTrainer

def main():
    """Main function to resume training"""
    print("Resume YOLO Training Script")
    print("="*50)

    # Configuration
    checkpoint_path = 'models/checkpoints/test_checkpoint_epoch_5'
    dataset_config = 'voc2012.yaml'
    additional_epochs = 10  # Train for 10 more epochs

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Check if dataset config exists
    if not os.path.exists(dataset_config):
        print(f"Dataset config not found: {dataset_config}")
        sys.exit(1)

    # Check for GPU
    import torch
    if torch.cuda.is_available():
        device_id = 0
        print("GPU available, using GPU for training")
    else:
        device_id = 'cpu'
        print("No GPU available, using CPU for training")

    # Initialize trainer
    trainer = OptimizedTrainer(model_path='yolov8x.pt', device_id=device_id)

    # Check system requirements (skip GPU check for CPU)
    if torch.cuda.is_available() and not trainer.check_system_requirements():
        print("System requirements not met. Exiting.")
        sys.exit(1)
    elif not torch.cuda.is_available():
        print("Using CPU - proceeding with training...")
        # Skip system requirements check for CPU

    # Resume training
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print(f"Training for additional {additional_epochs} epochs")

    results = trainer.train_with_memory_monitoring(
        dataset_config=dataset_config,
        epochs=15,  # Total epochs (5 + 10)
        resume_checkpoint=checkpoint_path
    )

    if results:
        print("\n✓ Training resumed and completed successfully!")

        # Validate the model
        model_path = f"{results.save_dir}/weights/best.pt"
        if os.path.exists(model_path):
            val_results = trainer.validate_model(model_path)

            if val_results:
                print("\n✓ Validation completed successfully!")
                print(f"Final mAP50: {val_results.box.map50:.4f}")
                print(f"Final mAP50-95: {val_results.box.map:.4f}")
            else:
                print("\n✗ Validation failed.")
        else:
            print(f"\n✗ Model not found at: {model_path}")
    else:
        print("\n✗ Training failed.")

if __name__ == "__main__":
    main()
