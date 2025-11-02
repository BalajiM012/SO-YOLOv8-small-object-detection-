from ultralytics import YOLO
import os
import time
import torch
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

def train_yolo_on_coco(resume_checkpoint=None):
    """Train YOLOv8X on MS-COCO dataset"""
    print("="*60)
    print("TRAINING YOLOv8X ON MS-COCO DATASET (80 CLASSES)")
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

    # Training configuration optimized for GPU
    config = {
        'data': 'coco.yaml',              # MS COCO 2017 (80 classes)
        'epochs': 100,                    # Number of epochs
        'imgsz': 640,                     # Image size for YOLOv8
        'batch': 16 if gpu_available else 4,  # Adjust based on GPU availability
        'device': 0 if gpu_available else 'cpu',  # Use GPU if available
        'workers': 8 if gpu_available else 4,     # DataLoader workers
        'patience': 50,                   # Early stopping patience
        'save': True,                     # Save checkpoints
        'save_period': 10,                # Save every N epochs
        'cache': True,                    # Cache images for faster training
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
        'name': 'coco_training',          # Experiment name
    }

    print(f"Training configuration:")
    print(f"  Dataset: {config['data']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Device: {config['device']}")
    print(f"  Workers: {config['workers']}")
    print(f"  AMP: {config['amp']}")

    print("\nNote: Make sure you have the MS-COCO dataset downloaded and organized as:")
    print("coco_dataset/")
    print("├── images/")
    print("│   ├── train2017/")
    print("│   └── val2017/")
    print("└── labels/")
    print("    ├── train2017/")
    print("    └── val2017/")
    print("\nThe dataset contains 80 classes as defined in coco.yaml")

    # Start training
    try:
        print("\nStarting training...")
        start_time = time.time()

        # Custom training loop with checkpoint saving
        checkpoint_interval = 10  # Save checkpoint every 10 epochs

        for epoch in range(start_epoch, 100):
            print(f"\nEpoch {epoch + 1}/100")

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
                checkpoint_name = f"coco_checkpoint_epoch_{epoch + 1}"
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
                    'dataset': 'coco.yaml',
                    'coco_classes': 80
                }

                saved_path = save_model(model, f"coco_epoch_{epoch + 1}", metadata=metadata)
                if saved_path:
                    print(f"Model saved with metadata: {saved_path}")

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours!")
        print(f"Best model saved at: {results.save_dir}")

        # Save final model with enhanced save function
        final_metadata = {
            'total_epochs': 100,
            'training_config': config,
            'training_time': training_time,
            'dataset': 'coco.yaml',
            'coco_classes': 80,
            'final_model': True
        }

        final_model_path = save_model(model, "coco_final", metadata=final_metadata)
        if final_model_path:
            print(f"Final model saved with enhanced save: {final_model_path}")

        # Validate the trained model
        print("\nRunning validation on COCO dataset...")
        val_results = model.val()

        # Print validation metrics
        print(f"\nCOCO Validation Results:")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        print(f"Precision: {val_results.box.mp:.4f}")
        print(f"Recall: {val_results.box.mr:.4f}")

        return results.save_dir, val_results

    except Exception as e:
        print(f"Training failed with error: {e}")
        return None, None

def test_on_voc2012(model_path):
    """Test the trained model on PASCAL VOC 2012 dataset"""
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL ON PASCAL VOC 2012 DATASET")
    print("="*60)

    if not model_path or not os.path.exists(f"{model_path}/weights/best.pt"):
        print("No trained model found. Please train the model first.")
        return None

    # Load the trained model using enhanced loader
    print(f"Loading model using enhanced loader: {model_path}/weights/best.pt")
    model = load_model(f"{model_path}/weights/best.pt", validate=True)

    if model is None:
        print(f"Failed to load model: {model_path}/weights/best.pt")
        return None
    
    # VOC dataset path
    voc_dataset_path = "D:/MINI Project Phase 1/Object Detection DL/Object DL DATASETS/SO - YOLO/PASCAL VOC 2012 DATASET"
    
    # Test images directory
    test_images_dir = os.path.join(voc_dataset_path, "VOC2012_test", "VOC2012_test", "JPEGImages")
    
    if not os.path.exists(test_images_dir):
        print(f"VOC test images directory not found: {test_images_dir}")
        print("Please ensure the PASCAL VOC 2012 dataset is available at the specified path.")
        return None
    
    # Get test images
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} test images in VOC 2012 dataset")
    
    # Test on a subset of images (first 50 for demo)
    test_images = image_files[:50]
    print(f"Testing on {len(test_images)} images...")
    
    # Create output directory for test results
    output_dir = "voc2012_test_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    
    # Run inference
    print("Running inference on VOC 2012 test images...")
    results = []
    
    for i, img_file in enumerate(test_images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(test_images)}")
        
        img_path = os.path.join(test_images_dir, img_file)
        
        # Run inference
        pred_results = model(img_path, conf=0.25, iou=0.45, device=0, save=True, 
                           project=output_dir, name="predictions")
        
        # Store results
        results.append({
            'image': img_file,
            'predictions': pred_results[0].boxes.data.cpu().numpy() if pred_results[0].boxes is not None else []
        })
    
    print(f"\nInference completed!")
    print(f"Results saved to: {output_dir}/predictions")
    
    # Print summary statistics
    total_detections = sum(len(r['predictions']) for r in results)
    avg_detections = total_detections / len(results) if results else 0
    
    print(f"\nTest Summary:")
    print(f"Images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    
    return results

def main():
    """Main function to train on COCO and test on VOC"""
    print("YOLOv8X Training on MS-COCO and Testing on PASCAL VOC 2012")
    print("="*70)
    
    # Step 1: Train on MS-COCO dataset
    print("\nStep 1: Training on MS-COCO dataset...")
    model_path, val_results = train_yolo_on_coco()
    
    if model_path:
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Model saved at: {model_path}")
        
        # Step 2: Test on PASCAL VOC 2012 dataset
        print("\nStep 2: Testing on PASCAL VOC 2012 dataset...")
        test_results = test_on_voc2012(model_path)
        
        if test_results:
            print(f"\n✓ Testing completed successfully!")
            print(f"✓ Test results saved to: voc2012_test_results/")
        
        print("\n" + "="*70)
        print("TRAINING AND TESTING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Trained model: {model_path}")
        print(f"Test results: voc2012_test_results/")
        print("\nYou can now use the trained model for inference on new images.")
        
    else:
        print("\n✗ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
