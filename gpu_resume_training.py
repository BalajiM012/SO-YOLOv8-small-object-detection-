#!/usr/bin/env python3
"""
GPU Resume Training Script for YOLO
Resume training and save checkpoints every 10 epochs on GPU
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path
from model_utils import save_model, load_model, save_checkpoint, load_checkpoint, resume_training

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
    print("GPU Resume Training Script")
    print("=" * 50)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ GPU not available. Cannot run GPU training.")
        return

    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Dataset and model configuration
    dataset_config = 'voc2012.yaml'
    model_path = 'yolov8x.pt'
    checkpoint_path = 'models/checkpoints/test_checkpoint_epoch_5'  # Assuming this is after 50 epochs; adjust if needed

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Starting fresh training...")
        checkpoint_path = None

    # Resume training
    if checkpoint_path:
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        model, start_epoch, _, _ = resume_training(checkpoint_path, model_path)
        if model is None:
            print("❌ Failed to resume from checkpoint. Starting fresh training.")
            model = YOLO(model_path)
            start_epoch = 0
        else:
            print(f"✅ Resumed from epoch {start_epoch}")
    else:
        model = YOLO(model_path)
        start_epoch = 0

    # Training configuration
    config = {
        'data': dataset_config,
        'epochs': 100,  # Total epochs
        'imgsz': 640,
        'batch': 8,  # GPU batch size
        'device': 0,  # GPU device
        'workers': 4,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'val': True,
        'plots': True,
        'amp': True,  # Automatic Mixed Precision
        'project': 'runs/gpu_resume',
        'name': 'gpu_resume_training',
        'exist_ok': True,
    }

    print("\n🚀 Starting GPU Training with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Custom training loop with checkpoint saving
    checkpoint_interval = 10
    start_time = time.time()

    for epoch in range(start_epoch, config['epochs']):
        print(f"\n📊 Epoch {epoch + 1}/{config['epochs']}")

        # Train for one epoch
        if epoch == 0 or checkpoint_path:
            results = model.train(**config)
        else:
            # For resumed training, adjust epochs
            temp_config = config.copy()
            temp_config['epochs'] = epoch + 1
            results = model.train(**temp_config)

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_name = f"gpu_checkpoint_epoch_{epoch + 1}"
            metrics = {
                'epoch': epoch + 1,
                'training_time': time.time() - start_time,
                'gpu_memory': torch.cuda.memory_allocated(0) / 1024**3,
                'config': config
            }

            checkpoint_path_saved = save_checkpoint(model, epoch + 1, metrics=metrics, checkpoint_name=checkpoint_name)
            if checkpoint_path_saved:
                print(f"💾 Checkpoint saved: {checkpoint_path_saved}")

        # Save model with metadata every 25 epochs
        if (epoch + 1) % 25 == 0:
            metadata = {
                'epochs_completed': epoch + 1,
                'training_config': config,
                'training_time': time.time() - start_time,
                'gpu_memory': torch.cuda.memory_allocated(0) / 1024**3,
                'dataset': dataset_config,
                'gpu_training': True
            }

            saved_path = save_model(model, f"gpu_epoch_{epoch + 1}", metadata=metadata)
            if saved_path:
                print(f"💾 Model saved with metadata: {saved_path}")

    training_time = time.time() - start_time

    print(f"\n✅ Training completed in {training_time/3600:.2f} hours!")
    print(f"📁 Best model saved at: {results.save_dir}")

    # Save final model
    final_metadata = {
        'total_epochs': config['epochs'],
        'training_config': config,
        'training_time': training_time,
        'final_gpu_memory': torch.cuda.memory_allocated(0) / 1024**3,
        'dataset': dataset_config,
        'gpu_training': True,
        'checkpoint_saving': True
    }

    final_model_path = save_model(model, "gpu_final_model", metadata=final_metadata)
    if final_model_path:
        print(f"💾 Final model saved: {final_model_path}")

        # Run testing on VOC2012
        print("\n🧪 Running testing on PASCAL VOC 2012 dataset...")
        test_results = test_on_voc2012(final_model_path)
        if test_results:
            print("✅ Testing completed successfully!")
        else:
            print("❌ Testing failed.")

    print("\n🎉 GPU training and testing completed successfully!")
    print("✅ Checkpoints saved every 10 epochs")
    print("✅ GPU acceleration utilized")

if __name__ == "__main__":
    import time
    main()
