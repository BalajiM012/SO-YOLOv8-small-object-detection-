from ultralytics import YOLO
import os
import time
import torch
import argparse
from model_utils import save_model, load_model, save_checkpoint, load_checkpoint, resume_training

def check_gpu_availability():
    """Check GPU availability and memory"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Available: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"GPU Count: {gpu_count}")
        return True
    else:
        print("No GPU available. Training will be slow on CPU.")
        return False

def train_yolo_with_resume(resume_checkpoint=None, epochs=100):
    """Train YOLOv8X with resume capability and checkpoint saving every 10 epochs"""
    print("="*60)
    print("YOLOv8X TRAINING WITH RESUME & CHECKPOINT SAVING")
    print("="*60)

    # Check GPU
    gpu_available = check_gpu_availability()

    # Resume from checkpoint if provided
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        model, start_epoch, _, _ = resume_training(resume_checkpoint, 'yolov8x.pt')
        if model is None:
            print("Failed to resume from checkpoint. Starting fresh training.")
            model = YOLO('yolov8x.pt')
            start_epoch = 0
        else:
            print(f"Resumed from epoch {start_epoch}")
    else:
        model = YOLO('yolov8x.pt')
        start_epoch = 0

    # Training configuration for PASCAL VOC 2012 dataset (20 classes)
    config = {
        'data': 'voc2012.yaml',           # PASCAL VOC 2012 (20 classes)
        'epochs': epochs,                 # Number of epochs
        'imgsz': 640,                     # Image size for YOLOv8
        'batch': 8 if gpu_available else 4,  # Adjust based on GPU availability
        'device': 0 if gpu_available else 'cpu',  # Use GPU if available
        'workers': 4,                     # DataLoader workers
        'patience': 50,                   # Early stopping patience
        'save': True,                     # Save checkpoints
        'save_period': 10,                # Save every N epochs
        'cache': False,                   # Disable caching to avoid memory issues
        'lr0': 0.01,                      # Initial learning rate
        'lrf': 0.01,                      # Final learning rate
        'momentum': 0.937,                # SGD momentum/Adam beta1
        'weight_decay': 0.0005,           # Optimizer weight decay
        'warmup_epochs': 3.0,             # Warmup epochs
        'warmup_momentum': 0.8,           # Warmup initial momentum
        'warmup_bias_lr': 0.1,            # Warmup initial bias lr
        'box': 7.5,                       # Box loss gain
        'cls': 0.5,                       # Class loss gain
        'dfl': 1.5,                       # DFL loss gain
        'label_smoothing': 0.0,           # Label smoothing epsilon
        'val': True,                      # Validate during training
        'plots': True,                    # Save plots
        'amp': gpu_available,             # Automatic Mixed Precision (AMP) training
        'fraction': 1.0,                  # Use full dataset
        'profile': False,                 # Profile ONNX and TensorRT speeds during training
        'project': 'runs/detect',         # Project directory
        'name': 'voc_training_resume',    # Experiment name
        'exist_ok': True,                 # Allow existing project/name
    }

    print(f"Training configuration:")
    print(f"  Dataset: {config['data']}")
    print(f"  Epochs: {config['epochs']} (starting from epoch {start_epoch})")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Device: {config['device']}")
    print(f"  Workers: {config['workers']}")
    print(f"  AMP: {config['amp']}")
    print(f"  Resume from checkpoint: {resume_checkpoint or 'None'}")

    print("\nNote: Using PASCAL VOC 2012 dataset for training")
    print("Dataset path: D:/MINI Project Phase 1/Object Detection DL/Object DL DATASETS/SO - YOLO/PASCAL VOC 2012 DATASET")
    print("\nThe dataset contains 20 classes as defined in voc2012.yaml")
    if gpu_available:
        print("GPU: NVIDIA GeForce RTX 4080 SUPER (16GB VRAM)")
        print("Optimized for high-performance training!")
    else:
        print("Using CPU for training (will be slower)")

    # Start training with checkpoint saving
    try:
        start_time = time.time()

        # Custom training loop with checkpoint saving every 10 epochs
        checkpoint_interval = 10

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train for one epoch
            if epoch == 0 or resume_checkpoint:
                results = model.train(**config)
            else:
                # For resumed training, adjust epochs
                temp_config = config.copy()
                temp_config['epochs'] = epoch + 1
                results = model.train(**temp_config)

            # Save checkpoint every checkpoint_interval epochs
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_name = f"voc_checkpoint_epoch_{epoch + 1}"
                metrics = {
                    'epoch': epoch + 1,
                    'training_time': time.time() - start_time,
                    'config': config
                }

                checkpoint_path = save_checkpoint(model, epoch + 1, metrics=metrics, checkpoint_name=checkpoint_name)
                if checkpoint_path:
                    print(f"Checkpoint saved: {checkpoint_path}")

            # Save model with metadata every 25 epochs
            if (epoch + 1) % 25 == 0:
                metadata = {
                    'epochs_completed': epoch + 1,
                    'training_config': config,
                    'training_time': time.time() - start_time,
                    'dataset': 'voc2012.yaml',
                    'voc_classes': 20
                }

                saved_path = save_model(model, f"voc_epoch_{epoch + 1}", metadata=metadata)
                if saved_path:
                    print(f"Model saved with metadata: {saved_path}")

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours!")
        print(f"Best model saved at: {results.save_dir}")

        # Save final model with enhanced save function
        final_metadata = {
            'total_epochs': epochs,
            'training_config': config,
            'training_time': training_time,
            'dataset': 'voc2012.yaml',
            'voc_classes': 20,
            'final_model': True,
            'checkpoint_saving': True
        }

        final_model_path = save_model(model, "voc_final", metadata=final_metadata)
        if final_model_path:
            print(f"Final model saved with enhanced save: {final_model_path}")

        # Validate the trained model
        print("\nRunning validation...")
        val_results = model.val()

        # Print validation metrics
        print(f"\nValidation Results:")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        print(f"Precision: {val_results.box.mp:.4f}")
        print(f"Recall: {val_results.box.mr:.4f}")

        return results.save_dir, val_results

    except Exception as e:
        print(f"Training failed with error: {e}")
        return None, None

def main():
    """Main function with command-line support"""
    print("YOLOv8X Training Script with Resume & Checkpoint Saving")
    print("="*60)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8X with resume capability and checkpoint saving every 10 epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs to train')
    parser.add_argument('--dataset', type=str, default='voc2012.yaml', help='Path to dataset configuration file')

    args = parser.parse_args()

    print(f"Command: Train YOLOv8X with resume and checkpoint saving")
    print(f"Resume from: {args.resume if args.resume else 'None (fresh training)'}")
    print(f"Total epochs: {args.epochs}")
    print(f"Dataset: {args.dataset}")
    print("-" * 60)

    # Start training
    model_path, val_results = train_yolo_with_resume(
        resume_checkpoint=args.resume,
        epochs=args.epochs
    )

    if model_path:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Trained model: {model_path}")
        print(f"Dataset used: {args.dataset}")
        print("Features implemented:")
        print("- Resume training from checkpoint")
        print("- Automatic checkpoint saving every 10 epochs")
        print("- Enhanced model saving with metadata")
        print("- GPU optimization when available")
        print("- Comprehensive validation")
        if args.resume:
            print(f"- Resumed from checkpoint: {args.resume}")

        return True
    else:
        print("\nTraining failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    main()
