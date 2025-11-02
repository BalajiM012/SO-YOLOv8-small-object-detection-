if __name__ == '__main__':
    import torch
    import os
    import time
    import sys
    from pathlib import Path
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

    # Check GPU
    gpu_available = check_gpu_availability()

    # Validate required files
    def validate_requirements():
        """Validate that all required files exist"""
        requirements_ok = True

        # Check for YOLO model file
        model_path = 'yolov8x.pt'
        if not os.path.exists(model_path):
            print(f"✗ YOLO model file not found: {model_path}")
            print("  Please ensure yolov8x.pt is in the current directory")
            requirements_ok = False
        else:
            print(f"✓ YOLO model file found: {model_path}")

        # Check for dataset configuration
        dataset_config = 'voc2012.yaml'
        if not os.path.exists(dataset_config):
            print(f"✗ Dataset configuration not found: {dataset_config}")
            requirements_ok = False
        else:
            print(f"✓ Dataset configuration found: {dataset_config}")

        # Check for dataset directories
        dataset_path = 'voc2012_yolo_dataset'
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset directory not found: {dataset_path}")
            requirements_ok = False
        else:
            print(f"✓ Dataset directory found: {dataset_path}")

            # Check for train/val subdirectories
            train_dir = os.path.join(dataset_path, 'images', 'train')
            val_dir = os.path.join(dataset_path, 'images', 'val')

            if not os.path.exists(train_dir):
                print(f"✗ Training images directory not found: {train_dir}")
                requirements_ok = False
            else:
                train_images = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png'))])
                print(f"✓ Training images: {train_images} images found")

            if not os.path.exists(val_dir):
                print(f"✗ Validation images directory not found: {val_dir}")
                requirements_ok = False
            else:
                val_images = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png'))])
                print(f"✓ Validation images: {val_images} images found")

        return requirements_ok

    print("\n" + "="*60)
    print("VALIDATING REQUIREMENTS")
    print("="*60)

    if not validate_requirements():
        print("\n✗ Requirements validation failed. Please fix the issues above and try again.")
        sys.exit(1)

    print("\n✓ All requirements validated successfully!")

    # YOLOv8X Training Script for 1 Epoch on GPU (Optimized)
    from ultralytics import YOLO

    # Load the YOLOv8X model
    model = YOLO('yolov8x.pt')

    # Training configuration optimized for GPU
    config = {
        'data': 'voc2012.yaml',              # Path to dataset configuration file
        'epochs': 1,                         # 1 epoch for testing
        'imgsz': 640,                        # Image size
        'batch': 8 if gpu_available else 4, # Adjust based on GPU availability
        'device': 0 if gpu_available else 'cpu', # Use GPU if available
        'workers': 8 if gpu_available else 4,    # DataLoader workers
        'save': True,                        # Save model checkpoints
        'val': True,                         # Validate during training
        'plots': True,                       # Generate plots
        'amp': gpu_available,                # Automatic Mixed Precision (AMP) training
        'project': 'models/yolov8x_training', # Separate folder for training
        'name': 'run1',                      # Run name
        'lr0': 0.01,                         # Initial learning rate
        'lrf': 0.01,                         # Final learning rate
        'momentum': 0.937,                   # SGD momentum/Adam beta1
        'weight_decay': 0.0005,              # Optimizer weight decay
        'warmup_epochs': 3.0,                # Warmup epochs
        'warmup_momentum': 0.8,              # Warmup initial momentum
        'warmup_bias_lr': 0.1,               # Warmup initial bias lr
        'box': 7.5,                          # Box loss gain
        'cls': 0.5,                          # Class loss gain
        'dfl': 1.5,                          # DFL loss gain
        'label_smoothing': 0.0,              # Label smoothing epsilon
        'fraction': 1.0,                     # Use full dataset
    }

    print(f"Training configuration:")
    print(f"  Dataset: {config['data']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Device: {config['device']}")
    print(f"  Workers: {config['workers']}")
    print(f"  AMP: {config['amp']}")

    # Start training
    try:
        print("\nStarting training...")
        start_time = time.time()

        results = model.train(**config)

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours!")
        print(f"Results saved to: {results.save_dir}")

        # Save model with metadata using model_utils
        metadata = {
            'epochs_completed': 1,
            'training_config': config,
            'training_time': training_time,
            'dataset': 'voc2012.yaml',
            'gpu_used': gpu_available,
        }

        saved_path = save_model(model, "voc2012_epoch_1", metadata=metadata)
        if saved_path:
            print(f"Model saved with metadata: {saved_path}")

            # Load the trained model for testing from the saved path
            trained_model = load_model(saved_path, validate=True)

            if trained_model:
                # Perform validation on the saved model
                val_results = trained_model.val(data='voc2012.yaml')
                print("Testing completed!")
                print(f"Validation results: mAP50={val_results.box.map50:.4f}, mAP50-95={val_results.box.map:.4f}")
            else:
                print("Failed to load trained model for testing.")
        else:
            print("Failed to save model.")

    except Exception as e:
        print(f"Training failed with error: {e}")
