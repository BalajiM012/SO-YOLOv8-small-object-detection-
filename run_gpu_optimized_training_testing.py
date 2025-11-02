"""
Complete GPU-Optimized Training and Testing Pipeline for PASCAL VOC 2012 Dataset
Features:
- Automatic GPU memory management and optimization
- Data chunking for large datasets
- Real-time performance monitoring
- Comprehensive evaluation and reporting
"""

import os
import time
import torch
import psutil
import gc
from ultralytics import YOLO
from pathlib import Path
import json
import argparse
from datetime import datetime
from model_utils import save_model, load_model, save_checkpoint, load_checkpoint, resume_training

def check_system_requirements():
    """Check system requirements and GPU availability"""
    print("="*60)
    print("SYSTEM REQUIREMENTS CHECK")
    print("="*60)
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        gpu_available = True
    else:
        print("✗ No GPU available. Training will be very slow on CPU.")
        gpu_available = False
    
    # RAM Check
    ram = psutil.virtual_memory()
    print(f"✓ System RAM: {ram.total / 1024**3:.1f} GB (Available: {ram.available / 1024**3:.1f} GB)")
    
    # Disk Space Check
    disk = psutil.disk_usage('.')
    print(f"✓ Disk Space: {disk.free / 1024**3:.1f} GB available")
    
    return gpu_available

def get_optimal_training_config(gpu_available, epochs=100):
    """Get optimized training configuration based on available resources"""
    
    # Calculate optimal batch size based on GPU memory
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:  # RTX 4080 Super or better
            batch_size = 16
            workers = 8
        elif gpu_memory >= 8:  # RTX 3070 or similar
            batch_size = 8
            workers = 6
        else:  # Lower-end GPU
            batch_size = 4
            workers = 4
    else:
        batch_size = 2
        workers = 2
    
    config = {
        'data': 'voc2012.yaml',
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': 0 if gpu_available else 'cpu',
        'workers': workers,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'cache': False,  # Disable caching to save memory
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'val': True,
        'plots': True,
        'amp': gpu_available,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'project': 'runs/detect',
        'name': f'voc2012_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'close_mosaic': 10,
    }
    
    return config

def train_model(model_path='yolov8x.pt', epochs=100, resume_checkpoint=None):
    """Train YOLO model with GPU optimization"""

    print("="*60)
    print("STARTING GPU-OPTIMIZED TRAINING")
    print("="*60)

    # Check system requirements
    gpu_available = check_system_requirements()

    # Resume from checkpoint if provided
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        model, start_epoch, _, _ = resume_training(resume_checkpoint, model_path)
        if model is None:
            print("Failed to resume from checkpoint. Starting fresh training.")
            model = YOLO(model_path)
            start_epoch = 0
        else:
            print(f"Resumed from epoch {start_epoch}")
    else:
        model = YOLO(model_path)
        start_epoch = 0

    # Get optimized configuration
    config = get_optimal_training_config(gpu_available, epochs)

    print(f"\nTraining Configuration:")
    print(f"  Dataset: {config['data']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch']} (optimized for GPU memory)")
    print(f"  Image Size: {config['imgsz']}")
    print(f"  Device: {config['device']}")
    print(f"  Workers: {config['workers']}")
    print(f"  AMP: {config['amp']}")

    # Monitor initial memory
    if gpu_available:
        initial_memory = torch.cuda.memory_allocated(0) / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nInitial GPU Memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")

    try:
        # Custom training loop with checkpoint saving
        checkpoint_interval = 10  # Save checkpoint every 10 epochs
        start_time = time.time()

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
                checkpoint_name = f"gpu_pipeline_checkpoint_epoch_{epoch + 1}"
                metrics = {
                    'epoch': epoch + 1,
                    'training_time': time.time() - start_time,
                    'memory_usage': torch.cuda.memory_allocated(0) / 1024**3 if gpu_available else 0,
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
                    'gpu_memory': torch.cuda.memory_allocated(0) / 1024**3 if gpu_available else 0,
                    'dataset': 'voc2012.yaml',
                    'gpu_optimized': True
                }

                saved_path = save_model(model, f"gpu_optimized_epoch_{epoch + 1}", metadata=metadata)
                if saved_path:
                    print(f"Model saved with metadata: {saved_path}")

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours!")
        print(f"Best model saved at: {results.save_dir}")

        # Monitor final memory
        if gpu_available:
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Final GPU Memory: {final_memory:.2f}GB / {total_memory:.2f}GB")

        # Save final model with enhanced save function
        final_metadata = {
            'total_epochs': epochs,
            'training_config': config,
            'training_time': training_time,
            'final_memory': torch.cuda.memory_allocated(0) / 1024**3 if gpu_available else 0,
            'dataset': 'voc2012.yaml',
            'gpu_optimized': True,
            'final_model': True
        }

        final_model_path = save_model(model, "gpu_optimized_final", metadata=final_metadata)
        if final_model_path:
            print(f"Final model saved with enhanced save: {final_model_path}")

        # Clear GPU cache
        if gpu_available:
            torch.cuda.empty_cache()
            gc.collect()

        return results

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nGPU Out of Memory Error: {e}")
        print("Attempting to recover with smaller batch size...")

        # Try with smaller batch size
        config['batch'] = max(1, config['batch'] // 2)
        if gpu_available:
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Retrying with batch size: {config['batch']}")
        return train_model(model_path, epochs)

    except Exception as e:
        print(f"Training failed with error: {e}")
        return None

def test_model(model_path, test_images_dir, test_annotations_dir, max_images=1000):
    """Test trained model on PASCAL VOC 2012 dataset"""
    
    print("\n" + "="*60)
    print("STARTING GPU-OPTIMIZED TESTING")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    # Check if test directories exist
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return None
    
    # Load model using enhanced loader
    print(f"Loading trained model using enhanced loader: {model_path}")
    model = load_model(model_path, validate=True)

    if model is None:
        print(f"Failed to load model: {model_path}")
        return None
    
    # Get test images
    test_images = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    if max_images:
        test_images = test_images[:max_images]
    
    print(f"Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = f"voc2012_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    # Determine optimal batch size for inference
    optimal_batch_size = 1
    if torch.cuda.is_available():
        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            try:
                test_subset = test_images[:batch_size]
                results = model(test_subset, device=0, verbose=False)
                memory_usage = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                if memory_usage > 0.9:
                    optimal_batch_size = max(1, batch_size // 2)
                    break
                optimal_batch_size = batch_size
            except torch.cuda.OutOfMemoryError:
                optimal_batch_size = max(1, batch_size // 2)
                break
    
    print(f"Optimal batch size for inference: {optimal_batch_size}")
    
    # Run inference
    print(f"\nRunning inference on {len(test_images)} images...")
    start_time = time.time()
    
    results = []
    for i in range(0, len(test_images), optimal_batch_size):
        batch_images = test_images[i:i + optimal_batch_size]
        
        for img_path in batch_images:
            try:
                # Run inference
                pred_results = model(str(img_path), device=0 if torch.cuda.is_available() else 'cpu', 
                                   conf=0.25, iou=0.45, verbose=False)
                
                # Store results
                if pred_results[0].boxes is not None:
                    boxes_data = pred_results[0].boxes.data.cpu().numpy()
                    results.append({
                        'image': img_path.name,
                        'predictions': boxes_data.tolist(),
                        'num_detections': len(boxes_data)
                    })
                else:
                    results.append({
                        'image': img_path.name,
                        'predictions': [],
                        'num_detections': 0
                    })
                
                # Save visualization for first 20 images
                if len(results) <= 20:
                    try:
                        import cv2
                        image = cv2.imread(str(img_path))
                        if image is not None and pred_results[0].boxes is not None:
                            # Draw bounding boxes
                            for box in pred_results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                conf = box.conf[0].cpu().numpy()
                                cls_id = int(box.cls[0].cpu().numpy())
                                class_name = model.names[cls_id]
                                
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(image, f"{class_name}: {conf:.2f}", 
                                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                            cv2.imwrite(f"{output_dir}/visualizations/{img_path.stem}_prediction.jpg", image)
                    except Exception as e:
                        print(f"Error saving visualization for {img_path}: {e}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Clear GPU cache periodically
        if i % (optimal_batch_size * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Progress update
        if (i // optimal_batch_size) % 10 == 0:
            print(f"Processed {i + len(batch_images)}/{len(test_images)} images")
    
    inference_time = time.time() - start_time
    
    # Calculate statistics
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results) if results else 0
    
    # Save results
    results_data = {
        'total_images': len(results),
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'inference_time': inference_time,
        'avg_time_per_image': inference_time / len(results) if results else 0,
        'results': results
    }
    
    with open(f"{output_dir}/test_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print(f"\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Average time per image: {inference_time/len(results):.3f} seconds")
    print(f"Results saved to: {output_dir}")
    
    return results_data

def main():
    """Main function to run training and testing pipeline"""
    
    parser = argparse.ArgumentParser(description='GPU-Optimized YOLO Training and Testing Pipeline')
    parser.add_argument('--model', default='yolov8x.pt', help='YOLO model to use (default: yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--test-images', default='archive/VOC2012_test/VOC2012_test/JPEGImages', 
                       help='Path to test images directory')
    parser.add_argument('--test-annotations', default='archive/VOC2012_test/VOC2012_test/Annotations',
                       help='Path to test annotations directory')
    parser.add_argument('--max-test-images', type=int, default=1000,
                       help='Maximum number of test images to process (default: 1000)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run testing')
    parser.add_argument('--skip-testing', action='store_true',
                       help='Skip testing and only run training')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    print("GPU-Optimized YOLO Training and Testing Pipeline")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Test Images: {args.test_images}")
    print(f"Max Test Images: {args.max_test_images}")
    print(f"Skip Training: {args.skip_training}")
    print(f"Skip Testing: {args.skip_testing}")
    
    training_results = None
    testing_results = None
    
    # Step 1: Training
    if not args.skip_training:
        print("\n" + "="*70)
        print("STEP 1: TRAINING")
        print("="*70)
        
        training_results = train_model(args.model, args.epochs, args.resume_checkpoint)
        
        if training_results:
            print("\n✓ Training completed successfully!")
            model_path = f"{training_results.save_dir}/weights/best.pt"
            print(f"✓ Model saved at: {model_path}")
        else:
            print("\n✗ Training failed!")
            return
    else:
        print("\nSkipping training...")
        # Look for existing trained model
        model_path = "runs/detect/voc2012_training_*/weights/best.pt"
        import glob
        existing_models = glob.glob(model_path)
        if existing_models:
            model_path = existing_models[0]
            print(f"Using existing model: {model_path}")
        else:
            print("No existing model found. Please train first or remove --skip-training flag.")
            return
    
    # Step 2: Testing
    if not args.skip_testing:
        print("\n" + "="*70)
        print("STEP 2: TESTING")
        print("="*70)
        
        if training_results:
            model_path = f"{training_results.save_dir}/weights/best.pt"
        else:
            model_path = args.model
        
        testing_results = test_model(model_path, args.test_images, args.test_annotations, args.max_test_images)
        
        if testing_results:
            print("\n✓ Testing completed successfully!")
        else:
            print("\n✗ Testing failed!")
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    
    if training_results:
        print(f"✓ Training completed: {training_results.save_dir}")
    
    if testing_results:
        print(f"✓ Testing completed: voc2012_test_results_*")
    
    print("\nYou can now use the trained model for inference on new images.")

if __name__ == "__main__":
    main()
