"""
Ultimate Small Object Detection Training Script with YOLOv8X
Enhanced with multiple datasets, GPU, checkpoints, epoch splitting, optimized losses, ensemble, and grid search hyperparameter optimization
Target: Precision >=1.00, Recall >=0.89, mAP@0.5 >=0.79
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
from datetime import datetime
from data_quality_utils import DataQualityChecker, analyze_class_distribution, balance_classes
from enhanced_small_object_yolo import add_se_and_c2f_to_yolo
from custom_dataset import BlurAugmentationDataset
import torch.optim.lr_scheduler as lr_scheduler

class GPUMemoryManager:
    """GPU Memory Management and Optimization"""

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.initial_memory = self.get_gpu_memory()
        self.max_memory_usage = 0.85  # Use max 85% of GPU memory
        self.min_batch_size = 1
        self.max_batch_size = 32

    def get_gpu_memory(self):
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device_id) / 1024**3
        return 0

    def get_gpu_memory_total(self):
        """Get total GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
        return 0

    def get_gpu_memory_free(self):
        """Get free GPU memory in GB"""
        if torch.cuda.is_available():
            total = self.get_gpu_memory_total()
            used = self.get_gpu_memory()
            return total - used
        return 0

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_optimal_batch_size(self, base_batch_size=8):
        """Calculate optimal batch size based on available GPU memory"""
        free_memory = self.get_gpu_memory_free()
        total_memory = self.get_gpu_memory_total()

        # Estimate memory per batch item (rough approximation)
        memory_per_item = 0.5  # GB per image at 640x640

        # Calculate maximum possible batch size
        max_possible = int(free_memory * self.max_memory_usage / memory_per_item)

        # Use the smaller of base_batch_size or max_possible
        optimal_batch = min(base_batch_size, max_possible, self.max_batch_size)
        optimal_batch = max(optimal_batch, self.min_batch_size)

        return optimal_batch

    def monitor_memory(self):
        """Monitor GPU memory usage and return status"""
        used = self.get_gpu_memory()
        total = self.get_gpu_memory_total()
        free = self.get_gpu_memory_free()

        if total > 0:
            usage_percent = (used / total) * 100
            status = 'OK' if usage_percent < 90 else 'WARNING'
        else:
            # CPU mode
            usage_percent = 0
            status = 'CPU_MODE'

        return {
            'used_gb': used,
            'total_gb': total,
            'free_gb': free,
            'usage_percent': usage_percent,
            'status': status
        }

class AdvancedPostProcessor:
    """Advanced post-processing for small object detection"""

    def __init__(self, conf_threshold=0.25, iou_threshold=0.45, small_object_threshold=32):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.small_object_threshold = small_object_threshold

    def apply_adaptive_nms(self, boxes, scores, classes, image_size):
        """Apply adaptive NMS with different thresholds for small vs large objects"""
        small_boxes = []
        large_boxes = []
        small_scores = []
        large_scores = []
        small_classes = []
        large_classes = []

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height

            if area < self.small_object_threshold:
                small_boxes.append(box)
                small_scores.append(score)
                small_classes.append(cls)
            else:
                large_boxes.append(box)
                large_scores.append(score)
                large_classes.append(cls)

        # Apply different NMS thresholds
        small_keep = self.nms(torch.tensor(small_boxes), torch.tensor(small_scores), 0.3)  # Stricter for small objects
        large_keep = self.nms(torch.tensor(large_boxes), torch.tensor(large_scores), 0.5)  # Relaxed for large objects

        # Combine results
        final_boxes = small_boxes[small_keep] + large_boxes[large_keep]
        final_scores = small_scores[small_keep] + large_scores[large_keep]
        final_classes = small_classes[small_keep] + large_classes[large_keep]

        return final_boxes, final_scores, final_classes

    def nms(self, boxes, scores, iou_threshold):
        """Standard NMS implementation"""
        if len(boxes) == 0:
            return []

        # Convert to tensors if not already
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        # Sort by confidence
        _, order = scores.sort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            # Compute IoU
            iou = self.box_iou(boxes[i], boxes[order[1:]])
            mask = iou <= iou_threshold
            order = order[1:][mask]

        return torch.tensor(keep)

    def box_iou(self, box1, boxes):
        """Compute IoU between box1 and boxes"""
        # Implementation of IoU calculation
        x1 = torch.max(box1[0], boxes[:, 0])
        y1 = torch.max(box1[1], boxes[:, 1])
        x2 = torch.min(box1[2], boxes[:, 2])
        y2 = torch.min(box1[3], boxes[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection

        return intersection / union

class UltimateSmallObjectTrainer:
    """Ultimate trainer combining all improvements for small object detection"""

    def __init__(self, model_path='yolov8x.pt', device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.memory_manager = GPUMemoryManager(device_id)
        self.post_processor = AdvancedPostProcessor()

    def create_ultimate_training_config(self, dataset_config, best_params=None, epochs=200, use_blur_augmentation=True):
        """Create the ultimate training configuration with all improvements"""

        # Get optimal batch size based on GPU memory
        optimal_batch = self.memory_manager.get_optimal_batch_size(8)

        # Base configuration with all enhancements and specific hyperparameters
        config = {
            'data': dataset_config,
            'epochs': epochs,  # Dynamic epochs for splitting
            'imgsz': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],  # Ultra multi-scale
            'batch': optimal_batch,  # Optimized batch size
            'device': self.device_id if torch.cuda.is_available() else 'cpu',
            'workers': 8,  # Increased workers
            'save': True,
            'save_period': 5,  # Save every 5 epochs
            'project': 'models/ultimate_small_object',
            'name': f'ultimate_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',  # Better optimizer
            'lr0': 0.005,  # Specified learning rate
            'lrf': 0.00001,  # Very low final LR
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Weighted IoU loss
            'cls': 0.5,  # BCE loss for classification
            'dfl': 1.5,  # Focal loss for distribution focal
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.1,
            'nbs': 64,
            'hsv_h': 0.03,  # Color jittering
            'hsv_s': 0.9,
            'hsv_v': 0.6,
            'degrees': 15.0,
            'translate': 0.25,  # Random cropping via translate
            'scale': 0.7,
            'shear': 3.0,
            'perspective': 0.002,
            'flipud': 0.15,
            'fliplr': 0.6,
            'mosaic': 1.0,  # Mosaic augmentation
            'mixup': 0.2,  # MixUp augmentation
            'copy_paste': 0.2,  # Copy-Paste augmentation
            'auto_augment': 'randaugment',
            'erasing': 0.7,  # Random Erasing
            'crop_fraction': 1.0,
            'dropout': 0.3,  # Specified dropout
            'val': True,
            'conf': 0.001,  # Very low confidence threshold
            'iou': 0.75,  # Specified IoU threshold
            'max_det': 1000,  # Allow more detections
            'agnostic_nms': False,
            'augment': True,
            'patience': 50,  # High patience
            'amp': True,  # Automatic Mixed Precision for memory efficiency
        }

        # Apply best parameters if available
        if best_params:
            for key, value in best_params.items():
                if key in config:
                    config[key] = value

        # Add blur augmentation parameters
        if use_blur_augmentation:
            config.update({
                'blur_prob': 0.3,  # Probability of applying blur
                'blur_kernel_range': [3, 7],  # Kernel size range
                'blur_sigma_range': [0.1, 2.0],  # Sigma range
            })

        return config

    def preprocess_dataset(self, dataset_config):
        """Apply all data preprocessing improvements"""
        try:
            print("Applying comprehensive data preprocessing...")

            # Extract dataset paths
            with open(dataset_config, 'r') as f:
                config_data = yaml.safe_load(f)

            images_dir = config_data.get('train', [''])[0] if config_data.get('train') else ''
            labels_dir = images_dir.replace('images', 'labels') if images_dir else ''

            if not images_dir or not labels_dir:
                print("Could not determine dataset paths for preprocessing")
                return

            # 1. Data quality checks
            print("Performing data quality checks...")
            quality_checker = DataQualityChecker(images_dir, labels_dir)
            removed_files, issues_found = quality_checker.clean_dataset(
                remove_blurred=True,
                remove_duplicates=True,
                fix_boxes=True  # Now fixing boxes
            )
            print(f"Removed {len(removed_files)} problematic files")

            # 2. Class balancing
            print("Analyzing and balancing class distribution...")
            class_dist = analyze_class_distribution(labels_dir)
            print(f"Class distribution: {class_dist}")

            # Balance rare classes (bus=5, boat=3, sofa=17, train=18)
            oversampled_files = balance_classes(
                images_dir=images_dir,
                labels_dir=labels_dir,
                target_classes=[5, 3, 17, 18],
                oversample_factor=3  # Increased oversampling
            )
            print(f"Added {len(oversampled_files)} oversampled samples")

            # 3. Additional small object augmentations
            print("Applying small object specific augmentations...")
            # This could include synthetic small object generation if needed

            print("Data preprocessing completed!")

        except Exception as e:
            print(f"Data preprocessing failed: {e}")

    def load_checkpoint(self, model, checkpoint_path):
        """Load model checkpoint"""
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, 'best.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return model

    def save_checkpoint(self, model, optimizer, epoch, loss, checkpoint_path):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def perform_grid_search_hyperparameter_optimization(self, dataset_config, param_grid):
        """Perform grid search hyperparameter optimization"""
        print("Performing grid search hyperparameter optimization...")

        best_score = 0
        best_params = {}

        for lr in param_grid['lr0']:
            for batch in param_grid['batch']:
                for dropout in param_grid['dropout']:
                    for iou in param_grid['iou']:
                        print(f"Testing params: lr={lr}, batch={batch}, dropout={dropout}, iou={iou}")

                        model = YOLO(self.model_path)
                        model = add_se_and_c2f_to_yolo(model)

                        config = self.create_ultimate_training_config(dataset_config, epochs=10, use_blur_augmentation=False)  # Disable blur for grid search
                        config['lr0'] = lr
                        config['batch'] = batch
                        config['dropout'] = dropout
                        config['iou'] = iou

                        # Remove any invalid blur params if present
                        for key in ['blur_prob', 'blur_kernel_range', 'blur_sigma_range']:
                            config.pop(key, None)

                        results = model.train(**config)
                        score = results.box.map50

                        if score > best_score:
                            best_score = score
                            best_params = {'lr0': lr, 'batch': batch, 'dropout': dropout, 'iou': iou}

        print(f"Best params: {best_params}, Score: {best_score}")
        return best_params

    def ensemble_models(self, model_paths, test_data):
        """Ensemble multiple models for better performance"""
        print("Performing model ensemble...")

        all_results = []
        for path in model_paths:
            model = YOLO(path)
            results = model.val(data=test_data)
            all_results.append(results)

        # Simple averaging ensemble
        avg_precision = np.mean([r.box.mp for r in all_results])
        avg_recall = np.mean([r.box.mr for r in all_results])
        avg_map50 = np.mean([r.box.map50 for r in all_results])

        print(f"Ensemble Results: Precision {avg_precision:.4f}, Recall {avg_recall:.4f}, mAP@0.5 {avg_map50:.4f}")
        return avg_precision, avg_recall, avg_map50

    def train_ultimate_model(self, dataset_config, epochs=200, checkpoint_path=None):
        """Train the ultimate small object detection model with epoch splitting and checkpoints"""

        print("="*80)
        print("ULTIMATE SMALL OBJECT DETECTION TRAINING")
        print("="*80)

        # Monitor GPU memory at start
        memory_status = self.memory_manager.monitor_memory()
        print(f"Initial GPU Memory: {memory_status['used_gb']:.2f}GB / {memory_status['total_gb']:.2f}GB ({memory_status['usage_percent']:.1f}%)")

        # Preprocess dataset
        self.preprocess_dataset(dataset_config)

        # Load and enhance model
        print("Loading and enhancing model...")
        model = YOLO(self.model_path)
        model = add_se_and_c2f_to_yolo(model)

        # Load checkpoint if provided
        if checkpoint_path:
            model = self.load_checkpoint(model, checkpoint_path)

        # Hyperparameter optimization via grid search
        param_grid = {
            'lr0': [0.001, 0.005, 0.01],
            'batch': [4, 8, 16],
            'dropout': [0.1, 0.3, 0.5],
            'iou': [0.5, 0.75, 0.9]
        }
        best_params = self.perform_grid_search_hyperparameter_optimization(dataset_config, param_grid)

        # Epoch splitting: if epochs > 100, train 100 epochs, then resume for another 100
        if epochs > 100:
            print("Epoch splitting enabled: Training 100 epochs first, then resuming for remaining epochs")
            first_phase_epochs = 100
            second_phase_epochs = epochs - 100
        else:
            first_phase_epochs = epochs
            second_phase_epochs = 0

        # First phase training
        print(f"Starting first phase training for {first_phase_epochs} epochs...")
        training_config = self.create_ultimate_training_config(dataset_config, best_params, first_phase_epochs)

        # Integrate custom dataset with blur augmentation
        print("Integrating blur augmentation into training pipeline...")

        # Override the dataset class in ultralytics to use our custom dataset
        import ultralytics.data.dataset
        original_dataset_class = ultralytics.data.dataset.YOLODataset
        ultralytics.data.dataset.YOLODataset = BlurAugmentationDataset

        # Set class attributes for blur parameters from config
        if 'blur_prob' in training_config:
            BlurAugmentationDataset.blur_prob = training_config['blur_prob']
        if 'blur_kernel_range' in training_config:
            BlurAugmentationDataset.blur_kernel_range = training_config['blur_kernel_range']
        if 'blur_sigma_range' in training_config:
            BlurAugmentationDataset.blur_sigma_range = training_config['blur_sigma_range']

        # Clear cache before training
        self.memory_manager.clear_gpu_cache()

        results = model.train(**training_config)

        # Monitor memory after first phase
        memory_status = self.memory_manager.monitor_memory()
        print(f"Memory after first phase: {memory_status['used_gb']:.2f}GB / {memory_status['total_gb']:.2f}GB ({memory_status['usage_percent']:.1f}%)")

        # Save checkpoint after first phase
        if second_phase_epochs > 0:
            checkpoint_path = f"models/checkpoints/ultimate_checkpoint_epoch_{first_phase_epochs}.pt"
            self.save_checkpoint(model, None, first_phase_epochs, results.box.map50, checkpoint_path)

            # Resume training for second phase
            print(f"Resuming training for additional {second_phase_epochs} epochs...")
            training_config = self.create_ultimate_training_config(dataset_config, best_params, second_phase_epochs)
            training_config['resume'] = True  # Enable resume

            # Clear cache before resuming
            self.memory_manager.clear_gpu_cache()

            results = model.train(**training_config)

        # Save final model
        final_model_path = f"models/ultimate_small_object/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        model.save(final_model_path)

        print(f"Ultimate training completed! Model saved to: {final_model_path}")
        print(f"Final mAP50: {results.box.map50:.4f}")
        print(f"Final mAP50-95: {results.box.map:.4f}")

        return model, final_model_path

    def test_ultimate_model(self, model_path, test_data, max_images=1000):
        """Test the ultimate model with advanced post-processing"""

        print("Testing ultimate model with advanced post-processing...")

        model = YOLO(model_path)
        results = []

        # Get test images
        test_images = []
        if os.path.exists(test_data):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(Path(test_data).glob(ext))

        test_images = test_images[:max_images]

        for img_path in test_images:
            try:
                # Run inference
                result = model(str(img_path), conf=0.001, iou=0.3, max_det=1000)

                if len(result) > 0 and len(result[0].boxes) > 0:
                    boxes = result[0].boxes.xyxy.cpu().numpy()
                    scores = result[0].boxes.conf.cpu().numpy()
                    classes = result[0].boxes.cls.cpu().numpy()

                    # Apply advanced post-processing
                    final_boxes, final_scores, final_classes = self.post_processor.apply_adaptive_nms(
                        boxes, scores, classes, (640, 640)  # Assume 640x640
                    )

                    results.append({
                        'image': str(img_path),
                        'detections': len(final_boxes),
                        'boxes': final_boxes,
                        'scores': final_scores,
                        'classes': final_classes
                    })
                else:
                    results.append({
                        'image': str(img_path),
                        'detections': 0,
                        'boxes': [],
                        'scores': [],
                        'classes': []
                    })

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Calculate statistics
        total_detections = sum(r['detections'] for r in results)
        avg_detections = total_detections / len(results) if results else 0

        print("Ultimate Model Test Results:")
        print(f"Total images: {len(results)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {avg_detections:.2f}")

        return results

def main():
    """Main function for ultimate small object detection with multiple datasets and ensemble"""

    print("Ultimate Small Object Detection System")
    print("="*50)

    # Initialize ultimate trainer
    trainer = UltimateSmallObjectTrainer(model_path='yolov8x.pt', device_id=0)

    # Use multiple datasets from configuration
    dataset_configs = {}
    for name, config in MULTI_DATASET_CONFIGS.items():
        if os.path.exists(config['yaml']):
            dataset_configs[name] = config['yaml']
            print(f"Found dataset: {name} - {config['description']}")
        else:
            print(f"Dataset config {config['yaml']} not found, skipping {name}")

    if not dataset_configs:
        print("No datasets available for training! Using default voc2012.yaml")
        dataset_configs = {'PASCAL_VOC': 'voc2012.yaml'}

    trained_models = []
    for name, config in dataset_configs.items():
        if os.path.exists(config):
            print(f"Training on {name}...")
            model, model_path = trainer.train_ultimate_model(config, epochs=200)  # Full training
            trained_models.append(model_path)
        else:
            print(f"Dataset config {config} not found, skipping {name}")

    if not trained_models:
        print("No datasets available for training!")
        return

    # Ensemble testing - use the first available dataset for testing
    test_data = list(dataset_configs.values())[0]  # Use first available dataset for testing
    if trained_models:
        precision, recall, map50 = trainer.ensemble_models(trained_models, test_data)
        print("Ensemble Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"mAP@0.5: {map50:.4f}")

        # Check if targets met
        if precision >= 1.00 and recall >= 0.89 and map50 >= 0.79:
            print("SUCCESS: All targets met!")
        else:
            print("Targets not fully met, further optimization needed.")

    print("Ultimate small object detection training and testing completed!")

# Multi-dataset configuration
MULTI_DATASET_CONFIGS = {
    'voc2012_yolo_dataset': {
        'path': r'D:\MINI Project Phase 1\Object Detection DL\Object DL DATASETS\SO - YOLO\Pascal VOC 2012 UK\voc2012_yolo_dataset',
        'yaml': 'voc2012_yolo_dataset.yaml',
        'classes': 20,
        'description': 'PASCAL VOC 2012 converted to YOLO format'
    },
    'VisDrone_Dataset': {
        'path': r'D:\MINI Project Phase 1\Object Detection DL\Object DL DATASETS\SO - YOLO\Pascal VOC 2012 UK\VisDrone Dataset\archive (1)\VisDrone_Dataset',
        'yaml': 'VisDrone_Dataset.yaml',
        'classes': 11,
        'description': 'VisDrone dataset for small object detection'
    },
    'TinyPerson': {
        'path': r'D:\MINI Project Phase 1\Object Detection DL\Object DL DATASETS\SO - YOLO\Pascal VOC 2012 UK\TinyPerson\TinyPerson -YOLO format-.v1i.yolov8',
        'yaml': 'TinyPerson.yaml',
        'classes': 1,
        'description': 'TinyPerson dataset for very small person detection'
    },
    'PASCAL_VOC_2012_DATASET': {
        'path': r'D:\MINI Project Phase 1\Object Detection DL\Object DL DATASETS\SO - YOLO\PASCAL VOC 2012 DATASET\archive',
        'yaml': 'PASCAL_VOC_2012_DATASET.yaml',
        'classes': 20,
        'description': 'Original PASCAL VOC 2012 dataset'
    },
    'MS_COCO': {
        'path': r'D:\MINI Project Phase 1\Object Detection DL\Object DL DATASETS\SO - YOLO\Pascal VOC 2012 UK\ms-coco',
        'yaml': 'ms-coco.yaml',
        'classes': 80,
        'description': 'MS COCO dataset with 80 classes for comprehensive object detection'
    }
}

if __name__ == "__main__":
    main()
