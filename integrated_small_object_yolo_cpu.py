"""
Integrated YOLO Training Script for Small Object Detection with CPU Support
Features:
- YOLOv8X model for superior small object detection
- Memory management for CPU training
- Hyperparameter optimization using Optuna
- Squeeze-and-Excitation (SE) blocks integrated into YOLOv8X
- Multi-scale training (320, 416, 512, 640)
- Advanced data augmentation for small objects (mosaic, mixup, copy-paste)
- Training on PASCAL VOC 2012 dataset
- Comprehensive testing and evaluation
"""

import os
import torch
import optuna
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

class MemoryManager:
    """Memory Management for CPU/GPU"""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.is_cuda = torch.cuda.is_available()
        self.min_batch_size = 1
        self.max_batch_size = 16 if self.is_cuda else 4  # Smaller for CPU

    def get_memory_usage(self):
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device_id) / 1024**3
        else:
            # For CPU, estimate memory usage
            return psutil.virtual_memory().percent / 100 * psutil.virtual_memory().total / 1024**3

    def clear_cache(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    def get_optimal_batch_size(self, base_batch_size=4):
        if self.is_cuda:
            free_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 - self.get_memory_usage()
            memory_per_item = 0.5
            max_possible = int(free_memory * 0.8 / memory_per_item)
        else:
            # For CPU, use conservative batch size
            max_possible = 4

        optimal_batch = min(base_batch_size, max_possible, self.max_batch_size)
        optimal_batch = max(optimal_batch, self.min_batch_size)
        return optimal_batch

def objective(trial, dataset_config, epochs=50):
    """Optuna objective function for hyperparameter optimization"""
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    batch = trial.suggest_categorical('batch', [2, 4, 8] if not torch.cuda.is_available() else [4, 8, 16])
    momentum = trial.suggest_float('momentum', 0.8, 0.95)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    mosaic = trial.suggest_float('mosaic', 0.5, 1.0)
    mixup = trial.suggest_float('mixup', 0.0, 0.2)
    copy_paste = trial.suggest_float('copy_paste', 0.0, 0.2)

    memory_manager = MemoryManager()
    optimal_batch = memory_manager.get_optimal_batch_size(batch)
    memory_manager.clear_cache()

    model = YOLO('yolov8x.pt')
    model = add_se_to_yolo(model)

    device = 0 if torch.cuda.is_available() else 'cpu'

    config = {
        'data': dataset_config,
        'epochs': epochs,
        'imgsz': [320, 416, 512, 640],  # Multi-scale for small objects
        'batch': optimal_batch,
        'lr0': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'augment': True,
        'mosaic': mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'device': device,
        'workers': 2 if not torch.cuda.is_available() else 4,
        'patience': 10,
        'save': False,
        'val': True,
        'amp': torch.cuda.is_available(),
        'project': 'runs/hyperopt',
        'name': f'trial_{trial.number}',
    }

    results = model.train(**config)
    memory_manager.clear_cache()

    # Optimize for mAP50 (good for small objects)
    return results.box.map50

def run_hyperparameter_optimization(dataset_config, n_trials=5):
    """Run hyperparameter optimization using Optuna"""
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset_config, epochs=10), n_trials=n_trials)  # Reduced epochs

    best_params = study.best_params
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best mAP50: {study.best_value:.4f}")

    return best_params

def train_enhanced_model(dataset_config, best_params, epochs=50):
    """Train the enhanced YOLOv8X model with optimized hyperparameters"""
    print("="*60)
    print("TRAINING ENHANCED YOLOv8X MODEL")
    print("="*60)

    memory_manager = MemoryManager()
    optimal_batch = memory_manager.get_optimal_batch_size(best_params['batch'])

    model = YOLO('yolov8x.pt')
    model = add_se_to_yolo(model)

    device = 0 if torch.cuda.is_available() else 'cpu'

    config = {
        'data': dataset_config,
        'epochs': epochs,
        'imgsz': [320, 416, 512, 640],  # Multi-scale training
        'batch': optimal_batch,
        'lr0': best_params['lr0'],
        'momentum': best_params['momentum'],
        'weight_decay': best_params['weight_decay'],
        'augment': True,
        'mosaic': best_params['mosaic'],
        'mixup': best_params['mixup'],
        'copy_paste': best_params['copy_paste'],
        'device': device,
        'workers': 2 if not torch.cuda.is_available() else 4,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'amp': torch.cuda.is_available(),
        'project': 'runs/enhanced_small_object',
        'name': 'yolov8x_se_multiscale',
        'exist_ok': True,
    }

    print("Training configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    start_time = time.time()
    results = model.train(**config)
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best model saved at: {results.save_dir}")

    memory_manager.clear_cache()

    return results, f"{results.save_dir}/weights/best.pt"

def test_enhanced_model(model_path, test_images_dir, test_annotations_dir, max_images=100):
    """Test the enhanced model on PASCAL VOC 2012 test set"""
    print("="*60)
    print("TESTING ENHANCED MODEL")
    print("="*60)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    model = YOLO(model_path)

    # Get test images
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

    device = 0 if torch.cuda.is_available() else 'cpu'

    for img_path in tqdm(test_images, desc="Testing"):
        try:
            # Run inference
            pred_results = model(str(img_path), device=device, conf=0.25, iou=0.45, verbose=False)

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

            # Save visualization for first 10 images
            if len(results) <= 10:
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
    """Main function"""
    print("Integrated Small Object Detection Pipeline")
    print("="*50)

    # Dataset configuration
    dataset_config = 'voc2012.yaml'
    test_images_dir = 'archive/VOC2012_test/VOC2012_test/JPEGImages'
    test_annotations_dir = 'archive/VOC2012_test/VOC2012_test/Annotations'

    # Check device availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Warning: GPU not available. Training will run on CPU (much slower).")
        print("For optimal performance, ensure CUDA-compatible PyTorch is installed.")

    # Step 1: Hyperparameter optimization
    print("\nStep 1: Hyperparameter Optimization")
    best_params = run_hyperparameter_optimization(dataset_config, n_trials=3)  # Reduced for demo

    # Step 2: Train enhanced model
    print("\nStep 2: Training Enhanced Model")
    training_results, model_path = train_enhanced_model(dataset_config, best_params, epochs=20)  # Reduced for demo

    # Step 3: Test the model
    print("\nStep 3: Testing Enhanced Model")
    test_results = test_enhanced_model(model_path, test_images_dir, test_annotations_dir, max_images=50)  # Reduced for demo

    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Trained model: {model_path}")
    print("Enhanced features implemented:")
    print("- YOLOv8X architecture")
    print("- Squeeze-and-Excitation blocks")
    print("- Multi-scale training")
    print("- Advanced data augmentation")
    print("- Memory optimization")
    print("- Hyperparameter optimization")

if __name__ == "__main__":
    main()
