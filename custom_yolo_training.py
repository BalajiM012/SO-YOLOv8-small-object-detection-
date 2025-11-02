"""
Custom YOLO Training Script for Small Object Detection with Fixed Hyperparameters
Features:
- YOLOv8X model for superior small object detection
- GPU memory management and chunking for large datasets
- Fixed hyperparameters: batch_size=8, LR=0.005, dropout=0.3
- Squeeze-and-Excitation (SE) blocks integrated into YOLOv8X
- Multi-scale training (320, 416, 512, 640)
- Advanced data augmentation for small objects (mosaic, mixup, copy-paste, random erasing)
- Training on PASCAL VOC 2012 dataset
- Comprehensive testing and evaluation
"""

import os
import torch
import argparse
from ultralytics import YOLO
import yaml
from pathlib import Path
import numpy as np
import psutil
import gc
from collections import defaultdict
import json
import time
import cv2
from tqdm import tqdm
from model_utils import save_model, load_model, save_checkpoint, load_checkpoint, resume_training

# Custom SE Block
class SEBlock(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Function to add SE blocks to YOLO model
def add_se_to_yolo(model):
    """Add SE blocks to YOLO backbone for enhanced feature extraction"""
    print("Adding Squeeze-and-Excitation blocks to YOLO model...")

    # Add SE blocks to Conv2d layers in backbone
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and 'backbone' in name:
            # Get the output channels
            out_channels = module.out_channels

            # Create SE block
            se_block = SEBlock(out_channels)

            # Replace the conv layer with conv + SE
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent_module = model.model
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)

            # Wrap the conv layer
            original_conv = getattr(parent_module, child_name)
            setattr(parent_module, child_name, torch.nn.Sequential(original_conv, se_block))

    print("SE blocks added successfully")
    return model

class GPUMemoryManager:
    """GPU Memory Management and Optimization"""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.initial_memory = self.get_gpu_memory()
        self.max_memory_usage = 0.85
        self.min_batch_size = 1
        self.max_batch_size = 32

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device_id) / 1024**3
        return 0

    def get_gpu_memory_total(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
        return 0

    def get_gpu_memory_free(self):
        if torch.cuda.is_available():
            total = self.get_gpu_memory_total()
            used = self.get_gpu_memory()
            return total - used
        return 0

    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_optimal_batch_size(self, base_batch_size=8):
        free_memory = self.get_gpu_memory_free()
        memory_per_item = 0.5  # GB per image at 640x640
        max_possible = int(free_memory * self.max_memory_usage / memory_per_item)
        optimal_batch = min(base_batch_size, max_possible, self.max_batch_size)
        optimal_batch = max(optimal_batch, self.min_batch_size)
        return optimal_batch

def train_enhanced_model(dataset_config, epochs=100, resume_checkpoint=None):
    """Train the enhanced YOLOv8X model with fixed hyperparameters"""
    print("="*60)
    print("TRAINING ENHANCED YOLOv8X MODEL")
    print("="*60)

    memory_manager = GPUMemoryManager()
    optimal_batch = memory_manager.get_optimal_batch_size(8)  # Fixed batch size 8

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

    model = add_se_to_yolo(model)

    # Set dropout in the model
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.3  # Fixed dropout 0.3

    config = {
        'data': dataset_config,
        'epochs': epochs,
        'imgsz': [320, 416, 512, 640],  # Multi-scale training
        'batch': optimal_batch,
        'lr0': 0.005,  # Fixed LR
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'augment': True,
        'mosaic': 1.0,  # Mosaic for small objects
        'mixup': 0.1,  # MixUp
        'copy_paste': 0.1,  # Copy-Paste
        'erasing': 0.4,  # Random Erasing
        'device': 0,
        'workers': 4,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'amp': True,
        'project': 'runs/enhanced_small_object',
        'name': 'yolov8x_se_multiscale_fixed_params',
        'exist_ok': True,
    }

    print("Training configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    start_time = time.time()

    # Custom training loop with checkpoint saving
    checkpoint_interval = 10  # Save checkpoint every 10 epochs

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train for one epoch
        if epoch == 0 or not resume_checkpoint:
            results = model.train(**config)
        else:
            # For resumed training, adjust epochs
            temp_config = config.copy()
            temp_config['epochs'] = epoch + 1
            results = model.train(**temp_config)

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1}"
            metrics = {
                'epoch': epoch + 1,
                'training_time': time.time() - start_time,
                'memory_usage': memory_manager.get_gpu_memory()
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
                'gpu_memory': memory_manager.get_gpu_memory(),
                'dataset': dataset_config
            }

            saved_path = save_model(model, f"yolov8x_se_epoch_{epoch + 1}", metadata=metadata)
            if saved_path:
                print(f"Model saved with metadata: {saved_path}")

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best model saved at: {results.save_dir}")

    # Save final model with enhanced save function
    final_metadata = {
        'total_epochs': epochs,
        'training_config': config,
        'training_time': training_time,
        'final_memory': memory_manager.get_gpu_memory(),
        'dataset': dataset_config,
        'enhanced_features': ['SE_blocks', 'multiscale', 'augmentation']
    }

    final_model_path = save_model(model, "yolov8x_se_final", metadata=final_metadata)
    if final_model_path:
        print(f"Final model saved with enhanced save: {final_model_path}")

    memory_manager.clear_gpu_cache()

    return results, f"{results.save_dir}/weights/best.pt"

def test_enhanced_model(model_path, test_images_dir, test_annotations_dir, max_images=1000):
    """Test the enhanced model on PASCAL VOC 2012 test set"""
    print("="*60)
    print("TESTING ENHANCED MODEL")
    print("="*60)

    # Use enhanced model loading with validation
    print(f"Loading model using enhanced loader: {model_path}")
    model = load_model(model_path, validate=True)

    if model is None:
        print(f"Failed to load model: {model_path}")
        return None

    # Get test images (using val as test since no separate test set)
    test_images = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    if max_images:
        test_images = test_images[:max_images]

    print(f"Testing on {len(test_images)} images")

    # Create output directory
    output_dir = f"enhanced_test_results_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

    results = []
    start_time = time.time()

    for img_path in tqdm(test_images, desc="Testing"):
        try:
            # Run inference
            pred_results = model(str(img_path), device=0, conf=0.25, iou=0.45, verbose=False)

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
                image = cv2.imread(str(img_path))
                if image is not None and pred_results[0].boxes is not None:
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
            print(f"Error processing {img_path}: {e}")
            continue

    inference_time = time.time() - start_time

    # Calculate statistics
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results) if results else 0

    # Run validation on training set for metrics
    val_results = model.val(data='voc2012.yaml')

    # Save results
    results_data = {
        'model': model_path,
        'total_images': len(results),
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'inference_time': inference_time,
        'avg_time_per_image': inference_time / len(results) if results else 0,
        'validation_metrics': {
            'mAP50': val_results.box.map50,
            'mAP50-95': val_results.box.map,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr
        },
        'results': results
    }

    with open(f"{output_dir}/test_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)

    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Total images: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Average time per image: {inference_time/len(results):.3f}s")
    print(f"mAP@0.5: {val_results.box.map50:.4f}")
    print(f"mAP@0.5-0.95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")
    print(f"Results saved to: {output_dir}")

    return results_data

def main():
    """Main function with command-line support"""
    print("Custom Small Object Detection Pipeline with Fixed Hyperparameters")
    print("="*60)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Resume YOLO training and save checkpoints every 10 epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs to train')
    parser.add_argument('--dataset', type=str, default='voc2012.yaml', help='Path to dataset configuration file')
    parser.add_argument('--model', type=str, default='yolov8x.pt', help='Base model path')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--test', action='store_true', help='Run testing after training')
    parser.add_argument('--test-images', type=str, default='./voc2012_yolo_dataset/images/val', help='Test images directory')
    parser.add_argument('--max-test-images', type=int, default=500, help='Maximum number of test images')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device_id = 0 if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        device_id = 'cpu'
    else:
        try:
            device_id = int(args.device)
        except ValueError:
            device_id = args.device

    # Dataset configuration
    dataset_config = args.dataset

    # Check GPU availability (skip for CPU)
    if device_id != 'cpu' and not torch.cuda.is_available():
        print("GPU not available. This pipeline requires GPU for optimal performance.")
        return

    if device_id != 'cpu':
        print(f"GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU for training")

    print(f"Command: Resume training and save checkpoints every 10 epochs")
    print(f"Resume from: {args.resume if args.resume else 'None (fresh training)'}")
    print(f"Total epochs: {args.epochs}")
    print(f"Dataset: {dataset_config}")
    print(f"Model: {args.model}")
    print(f"Device: {device_id}")
    print("-" * 60)

    # Train enhanced model with fixed params
    print("\nTraining Enhanced Model with Fixed Hyperparameters")
    training_results, model_path = train_enhanced_model(
        dataset_config=dataset_config,
        epochs=args.epochs,
        resume_checkpoint=args.resume
    )

    # Test the model if requested
    if args.test:
        print("\nTesting Enhanced Model")
        test_results = test_enhanced_model(
            model_path=model_path,
            test_images_dir=args.test_images,
            test_annotations_dir=args.test_images.replace('images', 'labels'),
            max_images=args.max_test_images
        )

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Trained model: {model_path}")
    print("Enhanced features implemented:")
    print("- YOLOv8X architecture")
    print("- Squeeze-and-Excitation blocks")
    print("- Multi-scale training")
    print("- Advanced data augmentation (Mosaic, MixUp, Copy-Paste, Random Erasing)")
    print("- Fixed hyperparameters: batch=8, lr=0.005, dropout=0.3")
    print("- GPU memory optimization")
    print(f"- {args.epochs} epochs training")
    if args.resume:
        print(f"- Resumed from checkpoint: {args.resume}")
    print("- Checkpoints saved every 10 epochs")

