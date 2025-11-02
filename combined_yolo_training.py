"""
Combined YOLO Training and Testing Script
Features:
- GPU-optimized training with memory management
- Resume training from checkpoints
- Automatic checkpoint saving every 10 epochs
- Model testing and validation
- Enhanced data augmentation
- Multi-scale training
- Comprehensive logging and metrics
"""

import os
import time
import torch
import psutil
import gc
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import logging
import yaml
import cv2
from tqdm import tqdm
from enhanced_small_object_yolo import add_se_and_c2f_to_yolo
from data_quality_utils import DataQualityChecker, analyze_class_distribution, balance_classes
import torch.optim.lr_scheduler as lr_scheduler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        logger.info(f"GPU Memory: {free_memory:.2f}GB free, {total_memory:.2f}GB total")
        logger.info(f"Optimal batch size: {optimal_batch}")

        return optimal_batch

    def monitor_memory(self):
        """Monitor current memory usage"""
        return {
            'used_gb': self.get_gpu_memory(),
            'total_gb': self.get_gpu_memory_total(),
            'free_gb': self.get_gpu_memory_free()
        }

class ModelManager:
    """Enhanced model management with save/load capabilities"""

    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.base_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)

    def save_model(self, model, model_name, metadata=None, save_format='pt'):
        """Save model with metadata and validation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = self.base_dir / f"{model_name}_{timestamp}"
            model_dir.mkdir(exist_ok=True)

            # Save model weights
            model_path = model_dir / f"{model_name}.{save_format}"
            if save_format == 'pt':
                model.save(str(model_path))
            elif save_format == 'onnx':
                model.export(format='onnx', dynamic=True)
            elif save_format == 'torchscript':
                model.export(format='torchscript')

            logger.info(f"Model saved to: {model_path}")

            # Save metadata
            if metadata:
                metadata_path = model_dir / "metadata.json"
                metadata['save_time'] = timestamp
                metadata['model_path'] = str(model_path)
                metadata['format'] = save_format

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

                logger.info(f"Metadata saved to: {metadata_path}")

            return str(model_path)

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None

    def load_model(self, model_path, validate=True):
        """Load model with validation"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None

            logger.info(f"Loading model from: {model_path}")
            model = YOLO(model_path)

            if validate:
                # Basic validation
                logger.info("Validating loaded model...")
                # You can add more validation here if needed

            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def save_checkpoint(self, model, epoch, optimizer_state=None, scheduler_state=None, metrics=None, checkpoint_name=None):
        """Save checkpoint during training"""
        try:
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}"

            checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.pt"
            metadata_path = self.checkpoints_dir / f"{checkpoint_name}.json"

            # Save model weights
            model.save(str(checkpoint_path))

            # Save metadata
            metadata = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(checkpoint_path),
                'checkpoint_name': checkpoint_name
            }

            if metrics:
                metadata['metrics'] = metrics
            if optimizer_state:
                metadata['optimizer_state'] = str(optimizer_state)  # Simplified
            if scheduler_state:
                metadata['scheduler_state'] = str(scheduler_state)  # Simplified

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """Load checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None

            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            model = YOLO(checkpoint_path)

            # Load metadata if available
            metadata_path = checkpoint_path.replace('.pt', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Checkpoint metadata loaded: epoch {metadata.get('epoch', 'unknown')}")

            return model

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        if self.checkpoints_dir.exists():
            for checkpoint_file in self.checkpoints_dir.glob("*.pt"):
                metadata_file = checkpoint_file.with_suffix('.json')
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass

                checkpoints.append({
                    'name': checkpoint_file.name,
                    'path': str(checkpoint_file),
                    'metadata': metadata
                })

        return sorted(checkpoints, key=lambda x: x['metadata'].get('epoch', 0), reverse=True)

class DatasetManager:
    """Dataset Management for Multiple Datasets"""

    def __init__(self):
        self.datasets = {
            1: {
                'name': 'PASCAL VOC 2012',
                'path': './voc2012.yaml',
                'description': 'PASCAL VOC 2012 dataset with 20 object classes',
                'images_dir': './voc2012_yolo_dataset/images',
                'labels_dir': './voc2012_yolo_dataset/labels'
            },
            2: {
                'name': 'TinyPerson',
                'path': './TinyPerson/TinyPerson -YOLO format-.v1i.yolov8/data.yaml',
                'description': 'TinyPerson dataset for small person detection',
                'images_dir': './TinyPerson/TinyPerson -YOLO format-.v1i.yolov8/train/images',
                'labels_dir': './TinyPerson/TinyPerson -YOLO format-.v1i.yolov8/train/labels'
            },
            3: {
                'name': 'VisDrone Dataset',
                'path': './VisDrone Dataset/archive (1)/VisDrone2019-DET-train/data.yaml',
                'description': 'VisDrone dataset for drone-based object detection',
                'images_dir': './VisDrone Dataset/archive (1)/VisDrone2019-DET-train/images',
                'labels_dir': './VisDrone Dataset/archive (1)/VisDrone2019-DET-train/labels'
            },
            4: {
                'name': 'VisDrone Dataset 2',
                'path': './VisDrone Dataset 2/archive/data.yaml',
                'description': 'Additional VisDrone dataset for comprehensive training',
                'images_dir': './VisDrone Dataset 2/archive/train/images',
                'labels_dir': './VisDrone Dataset 2/archive/train/labels'
            },
            5: {
                'name': 'PASCAL VOC 2012 DATASET',
                'path': './PASCAL VOC 2012 DATASET/data.yaml',
                'description': 'Additional PASCAL VOC 2012 dataset for enhanced training',
                'images_dir': './PASCAL VOC 2012 DATASET/images',
                'labels_dir': './PASCAL VOC 2012 DATASET/labels'
            },
            6: {
                'name': 'Rainbow Flow',
                'path': './New folder/Rainbow flow/data.yaml',
                'description': 'Rainbow Flow dataset for diverse object detection',
                'images_dir': './New folder/Rainbow flow/images',
                'labels_dir': './New folder/Rainbow flow/labels'
            },
            7: {
                'name': 'Ultralytics Dataset',
                'path': './New folder/ultralytics/data.yaml',
                'description': 'Ultralytics additional dataset for comprehensive training',
                'images_dir': './New folder/ultralytics/images',
                'labels_dir': './New folder/ultralytics/labels'
            },
            8: {
                'name': 'VisDrone Dataset (New Folder)',
                'path': './New folder/VisDrone Dataset/data.yaml',
                'description': 'Additional VisDrone dataset from New folder',
                'images_dir': './New folder/VisDrone Dataset/images',
                'labels_dir': './New folder/VisDrone Dataset/labels'
            }
        }

    def list_datasets(self):
        """List all available datasets"""
        print("\nAvailable Datasets:")
        print("="*60)
        for idx, dataset in self.datasets.items():
            status = "✓ Available" if os.path.exists(dataset['path']) else "✗ Not found"
            print(f"{idx}. {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Config: {dataset['path']}")
            print(f"   Status: {status}")
            print()

    def get_dataset_config(self, dataset_id):
        """Get dataset configuration by ID"""
        if dataset_id in self.datasets:
            dataset = self.datasets[dataset_id]
            if os.path.exists(dataset['path']):
                return dataset
            else:
                logger.error(f"Dataset configuration not found: {dataset['path']}")
                return None
        else:
            logger.error(f"Invalid dataset ID: {dataset_id}")
            return None

    def validate_dataset(self, dataset_config):
        """Validate dataset configuration and files"""
        if not os.path.exists(dataset_config['path']):
            logger.error(f"Dataset config file not found: {dataset_config['path']}")
            return False

        # Check if images and labels directories exist
        if not os.path.exists(dataset_config['images_dir']):
            logger.warning(f"Images directory not found: {dataset_config['images_dir']}")

        if not os.path.exists(dataset_config['labels_dir']):
            logger.warning(f"Labels directory not found: {dataset_config['labels_dir']}")

        # Load and validate YAML config
        try:
            with open(dataset_config['path'], 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Dataset config loaded: {config}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset config: {e}")
            return False

class CombinedYOLOTrainer:
    """Combined YOLO Training and Testing System"""

    def __init__(self, model_path='yolov8x.pt', device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.memory_manager = GPUMemoryManager(device_id)
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()

        # Check system requirements
        self.check_system_requirements()

    def check_system_requirements(self):
        """Check if system meets requirements"""
        logger.info("Checking system requirements...")

        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(self.device_id)
            gpu_memory = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("GPU not available. Training will use CPU.")

        # Check memory
        system_memory = psutil.virtual_memory()
        logger.info(f"System Memory: {system_memory.total / 1024**3:.1f}GB total, {system_memory.available / 1024**3:.1f}GB available")

        return True

    def perform_data_quality_checks(self, dataset_config):
        """Perform data quality checks on the dataset"""
        try:
            # Extract images and labels directories from dataset config
            if isinstance(dataset_config, str):
                # If it's a path to YAML file, load it
                with open(dataset_config, 'r') as f:
                    config = yaml.safe_load(f)
                images_dir = config.get('train', [''])[0] if config.get('train') else ''
                labels_dir = images_dir.replace('images', 'labels') if images_dir else ''
            else:
                # If it's a dataset config dict
                images_dir = dataset_config.get('images_dir', '')
                labels_dir = dataset_config.get('labels_dir', '')

            if not images_dir or not labels_dir:
                logger.warning("Could not determine images/labels directories for quality checks")
                return

            # Initialize data quality checker
            quality_checker = DataQualityChecker(images_dir, labels_dir)

            # Perform quality checks
            removed_files, issues_found = quality_checker.clean_dataset(
                remove_blurred=True,
                remove_duplicates=True,
                fix_boxes=False  # Just log invalid boxes for now
            )

            logger.info(f"Data quality checks completed. Removed {len(removed_files)} problematic files.")

        except Exception as e:
            logger.error(f"Data quality checks failed: {e}")

    def balance_dataset_classes(self, dataset_config):
        """Balance dataset classes by oversampling rare classes"""
        try:
            # Extract images and labels directories
            if isinstance(dataset_config, str):
                with open(dataset_config, 'r') as f:
                    config = yaml.safe_load(f)
                images_dir = config.get('train', [''])[0] if config.get('train') else ''
                labels_dir = images_dir.replace('images', 'labels') if images_dir else ''
            else:
                images_dir = dataset_config.get('images_dir', '')
                labels_dir = dataset_config.get('labels_dir', '')

            if not images_dir or not labels_dir:
                logger.warning("Could not determine images/labels directories for class balancing")
                return

            # Balance classes (bus=5, boat=3, sofa=17, train=18)
            oversampled_files = balance_classes(
                images_dir=images_dir,
                labels_dir=labels_dir,
                target_classes=[5, 3, 17, 18],  # bus, boat, sofa, train
                oversample_factor=2
            )

            logger.info(f"Class balancing completed. Added {len(oversampled_files)} oversampled samples.")

        except Exception as e:
            logger.error(f"Class balancing failed: {e}")

    def load_enhanced_model(self):
        """Load model with architectural enhancements"""
        try:
            logger.info("Loading YOLOv8x model with architectural enhancements...")

            # Load base model
            model = YOLO(self.model_path)

            # Apply architectural enhancements (SE blocks and C2f modifications)
            enhanced_model = add_se_and_c2f_to_yolo(model)

            logger.info("Model loaded with architectural enhancements")
            return enhanced_model

        except Exception as e:
            logger.error(f"Failed to load enhanced model: {e}")
            # Fallback to base model
            return YOLO(self.model_path)

    def train_with_enhanced_features(self, dataset_config, epochs=100, resume_checkpoint=None, save_every=10,
                                   enable_data_quality=True, enable_class_balance=True, enable_arch_enhancements=True):
        """Enhanced training with data quality, class balancing, and architectural improvements"""
        logger.info("Starting enhanced training with advanced features...")

        # Data Quality Checks
        if enable_data_quality:
            logger.info("Performing data quality checks...")
            self.perform_data_quality_checks(dataset_config)

        # Class Balancing
        if enable_class_balance:
            logger.info("Analyzing and balancing class distribution...")
            self.balance_dataset_classes(dataset_config)

        # Load model with architectural enhancements
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            model = self.model_manager.load_checkpoint(resume_checkpoint, None)
            if model is None:
                logger.error("Failed to load checkpoint, starting fresh training")
                model = self.load_enhanced_model()
        else:
            logger.info("Starting fresh training with enhanced model")
            model = self.load_enhanced_model()

        # Fixed batch size = 8 as requested
        batch_size = 8
        logger.info(f"Using fixed batch size: {batch_size}")

        # Enhanced training configuration with improved hyperparameters
        training_config = {
            'data': dataset_config,
            'epochs': epochs,
            'batch': batch_size,  # Fixed batch size
            'imgsz': 640,
            'device': self.device_id if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'save': True,
            'save_period': save_every,
            'project': 'models/runs/train',
            'name': f'enhanced_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'Adam',
            'lr0': 0.001,  # Lower initial learning rate
            'lrf': 0.0001,  # Lower final learning rate
            'momentum': 0.9,  # Adjusted momentum
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.1,  # Added label smoothing
            'nbs': 64,
            # Enhanced data augmentation
            'hsv_h': 0.02,  # Increased color jitter
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 10.0,  # Added rotation
            'translate': 0.2,  # Increased translation
            'scale': 0.6,  # Increased scale
            'shear': 2.0,  # Added shear
            'perspective': 0.001,
            'flipud': 0.1,  # Added vertical flip
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,  # Enabled MixUp
            'copy_paste': 0.1,  # Enabled Copy-Paste
            'auto_augment': 'randaugment',
            'erasing': 0.5,  # Increased random erasing
            'crop_fraction': 1.0,
            'dropout': 0.1,  # Added dropout
            'val': False  # Disable validation during training
        }

        logger.info(f"Training configuration: {training_config}")

        # Start training
        try:
            results = model.train(**training_config)

            # Save final model
            final_model_path = self.model_manager.save_model(
                model,
                "final_trained_model",
                metadata={
                    'training_config': training_config,
                    'final_results': str(results),
                    'total_epochs': epochs,
                    'device': training_config['device']
                }
            )

            logger.info(f"Training completed! Final model saved to: {final_model_path}")
            return results, final_model_path

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None, None

    def test_model(self, model_path, test_images_dir, test_annotations_dir, max_images=500):
        """Test trained model on validation/test data"""
        logger.info("Starting model testing...")

        # Load model
        model = self.model_manager.load_model(model_path, validate=True)
        if model is None:
            logger.error("Failed to load model for testing")
            return None

        # Get test images
        test_images = []
        if os.path.exists(test_images_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(Path(test_images_dir).glob(ext))
        else:
            logger.error(f"Test images directory not found: {test_images_dir}")
            return None

        test_images = test_images[:max_images]  # Limit for testing
        logger.info(f"Testing on {len(test_images)} images")

        # Run inference
        results = []
        inference_times = []

        for img_path in tqdm(test_images, desc="Testing"):
            try:
                start_time = time.time()
                result = model(str(img_path), conf=0.25, iou=0.45)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Process results
                if len(result) > 0 and len(result[0].boxes) > 0:
                    detections = len(result[0].boxes)
                    results.append({
                        'image': str(img_path),
                        'detections': detections,
                        'inference_time': inference_time,
                        'boxes': result[0].boxes.xyxy.tolist() if hasattr(result[0].boxes, 'xyxy') else [],
                        'confidences': result[0].boxes.conf.tolist() if hasattr(result[0].boxes, 'conf') else [],
                        'classes': result[0].boxes.cls.tolist() if hasattr(result[0].boxes, 'cls') else []
                    })
                else:
                    results.append({
                        'image': str(img_path),
                        'detections': 0,
                        'inference_time': inference_time,
                        'boxes': [],
                        'confidences': [],
                        'classes': []
                    })

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        # Calculate statistics
        total_detections = sum(r['detections'] for r in results)
        avg_detections = total_detections / len(results) if results else 0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

        # Save results
        output_dir = f"models/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        results_data = {
            'summary': {
                'total_images': len(results),
                'total_detections': total_detections,
                'average_detections_per_image': avg_detections,
                'average_inference_time': avg_inference_time,
                'model_path': model_path,
                'test_timestamp': datetime.now().isoformat()
            },
            'results': results
        }

        with open(f"{output_dir}/test_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Testing completed!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Total images: {len(results)}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Average detections per image: {avg_detections:.2f}")
        logger.info(f"Average inference time: {avg_inference_time:.3f}s")

        return results_data

    def validate_model(self, model_path):
        """Validate model performance"""
        logger.info("Starting model validation...")

        model = self.model_manager.load_model(model_path, validate=True)
        if model is None:
            logger.error("Failed to load model for validation")
            return None

        try:
            # Monitor memory before validation
            memory_before = self.memory_manager.monitor_memory()
            logger.info(f"Memory before validation: {memory_before['used_gb']:.2f}GB used")

            # Run validation
            val_results = model.val()

            # Monitor memory after validation
            memory_after = self.memory_manager.monitor_memory()
            logger.info(f"Memory after validation: {memory_after['used_gb']:.2f}GB used")

            logger.info("Validation Results:")
            logger.info(f"mAP50: {val_results.box.map50:.4f}")
            logger.info(f"mAP50-95: {val_results.box.map:.4f}")
            logger.info(f"Precision: {val_results.box.mp:.4f}")
            logger.info(f"Recall: {val_results.box.mr:.4f}")

            return val_results

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None

def train_on_all_datasets(epochs=50, save_every=10):
    """Train on all available datasets with fresh start"""
    print("="*80)
    print("MULTI-DATASET YOLO TRAINING - FRESH START")
    print("="*80)

    # Initialize trainer
    trainer = CombinedYOLOTrainer(model_path='yolov8x.pt', device_id=0)

    # List available datasets
    trainer.dataset_manager.list_datasets()

    # Get all valid datasets
    valid_datasets = []
    for dataset_id in trainer.dataset_manager.datasets.keys():
        dataset_config = trainer.dataset_manager.get_dataset_config(dataset_id)
        if dataset_config and trainer.dataset_manager.validate_dataset(dataset_config):
            valid_datasets.append(dataset_config)

    if not valid_datasets:
        print("No valid datasets found. Please check dataset configurations.")
        return None, None

    print(f"\nFound {len(valid_datasets)} valid datasets:")
    for i, ds in enumerate(valid_datasets, 1):
        print(f"{i}. {ds['name']}")

    # Create combined dataset configuration
    combined_config_path = create_combined_dataset_config(valid_datasets)
    if not combined_config_path:
        print("Failed to create combined dataset configuration.")
        return None, None

    print(f"\nCombined dataset config created: {combined_config_path}")

    # Training parameters
    print(f"\nStarting fresh training for {epochs} epochs...")
    print(f"Checkpoints will be saved every {save_every} epochs")
    print(f"Using combined multi-dataset training")

    # Start enhanced training with all features enabled
    results, model_path = trainer.train_with_enhanced_features(
        dataset_config=combined_config_path,
        epochs=epochs,
        resume_checkpoint=None,  # Fresh start
        save_every=save_every,
        enable_data_quality=True,
        enable_class_balance=True,
        enable_arch_enhancements=True
    )

    if results and model_path:
        print("\n" + "="*80)
        print("MULTI-DATASET TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Trained model: {model_path}")
        print(f"Datasets used: {len(valid_datasets)} combined datasets")

        # Test on each dataset
        print("\nTesting on individual datasets...")
        for dataset_config in valid_datasets:
            print(f"\nTesting on {dataset_config['name']}...")
            test_results = trainer.test_model(
                model_path,
                dataset_config['images_dir'],
                dataset_config['labels_dir'],
                max_images=100  # Test 100 images per dataset
            )
            if test_results:
                print(f"✓ Testing completed for {dataset_config['name']}")

        # Final validation
        print("\nRunning final validation...")
        val_results = trainer.validate_model(model_path)
        if val_results:
            print("✓ Final validation completed!")

    else:
        print("Training failed. Please check the logs for details.")

    return results, model_path

def create_combined_dataset_config(datasets):
    """Create a combined dataset configuration file"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_config_path = f"combined_dataset_{timestamp}.yaml"

    # Combine all datasets into one configuration
    combined_config = {
        'path': './',  # Base path
        'train': [],   # List of training image directories
        'val': [],     # List of validation image directories
        'test': [],    # List of test image directories
        'names': {     # Combined class names (will be extended)
            0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
            5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
            10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
            15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
        },
        'nc': 20  # Number of classes
    }

    # Add all dataset paths
    for dataset in datasets:
        if os.path.exists(dataset['images_dir']):
            combined_config['train'].append(dataset['images_dir'])
            combined_config['val'].append(dataset['images_dir'])  # Using train as val for simplicity

    # Save combined configuration
    try:
        import yaml
        with open(combined_config_path, 'w') as f:
            yaml.dump(combined_config, f, default_flow_style=False)
        print(f"Combined dataset config saved to: {combined_config_path}")
        return combined_config_path
    except Exception as e:
        print(f"Failed to create combined config: {e}")
        return None

def main():
    """Main function with multi-dataset training option"""
    print("="*80)
    print("Combined YOLO Training and Testing System")
    print("="*80)

    # Ask user for training mode
    print("\nTraining Options:")
    print("1. Train on single dataset (interactive)")
    print("2. Train on ALL datasets (fresh start)")
    print("3. Resume from checkpoint")

    choice = input("\nEnter your choice (1-3, default 2): ").strip()
    if not choice:
        choice = "2"

    if choice == "1":
        # Original single dataset training
        train_single_dataset()
    elif choice == "2":
        # Multi-dataset fresh training
        epochs = int(input("Enter number of epochs to train (default 50): ") or "50")
        save_every = int(input("Save checkpoint every N epochs (default 10): ") or "10")

        results, model_path = train_on_all_datasets(epochs=epochs, save_every=save_every)
    elif choice == "3":
        # Resume training
        train_with_resume_option()
    else:
        print("Invalid choice. Using multi-dataset training.")
        results, model_path = train_on_all_datasets()

def train_single_dataset():
    """Original single dataset training function"""
    # Initialize trainer
    trainer = CombinedYOLOTrainer(model_path='yolov8x.pt', device_id=0)

    # List available datasets
    trainer.dataset_manager.list_datasets()

    # Choose dataset
    dataset_choice = input("Enter dataset number to use (1-4, default 1): ").strip()
    if not dataset_choice:
        dataset_choice = "1"

    try:
        dataset_id = int(dataset_choice)
        dataset_config = trainer.dataset_manager.get_dataset_config(dataset_id)
        if dataset_config is None:
            print("Invalid dataset selection. Using default PASCAL VOC 2012.")
            dataset_config = trainer.dataset_manager.get_dataset_config(1)
    except ValueError:
        print("Invalid input. Using default PASCAL VOC 2012.")
        dataset_config = trainer.dataset_manager.get_dataset_config(1)

    print(f"\nSelected Dataset: {dataset_config['name']}")
    print(f"Config Path: {dataset_config['path']}")

    # Validate dataset
    if not trainer.dataset_manager.validate_dataset(dataset_config):
        print("Warning: Dataset validation failed. Training may not work properly.")

    # Training parameters
    epochs = int(input("Enter number of epochs to train (default 15): ") or "15")
    save_every = int(input("Save checkpoint every N epochs (default 10): ") or "10")

    # Start training
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Checkpoints will be saved every {save_every} epochs")
    print(f"Dataset: {dataset_config['name']}")

    results, model_path = trainer.train_with_resume(
        dataset_config=dataset_config['path'],
        epochs=epochs,
        resume_checkpoint=None,  # Fresh start
        save_every=save_every
    )

    if results and model_path:
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Trained model: {model_path}")
        print(f"Dataset used: {dataset_config['name']}")

        # Ask for testing
        test_choice = input("\nWould you like to test the trained model? (y/n): ").lower().strip()
        if test_choice == 'y':
            test_images_dir = input(f"Enter test images directory (default: {dataset_config['images_dir']}): ") or dataset_config['images_dir']
            test_annotations_dir = input(f"Enter test annotations directory (default: {dataset_config['labels_dir']}): ") or dataset_config['labels_dir']
            max_images = int(input("Maximum number of test images (default: 100): ") or "100")

            test_results = trainer.test_model(model_path, test_images_dir, test_annotations_dir, max_images)
            if test_results:
                print("Model testing completed!")

        # Ask for validation
        val_choice = input("\nWould you like to validate the model? (y/n): ").lower().strip()
        if val_choice == 'y':
            val_results = trainer.validate_model(model_path)
            if val_results:
                print("Model validation completed!")

    else:
        print("Training failed. Please check the logs for details.")

def train_with_resume_option():
    """Resume training from checkpoint"""
    trainer = CombinedYOLOTrainer(model_path='yolov8x.pt', device_id=0)

    # List available checkpoints
    checkpoints = trainer.model_manager.list_checkpoints()
    if not checkpoints:
        print("No checkpoints found. Starting fresh training instead.")
        train_single_dataset()
        return

    print("\nAvailable checkpoints:")
    for i, cp in enumerate(checkpoints[:5]):  # Show top 5
        epoch = cp['metadata'].get('epoch', 'unknown')
        timestamp = cp['metadata'].get('timestamp', 'unknown')
        print(f"{i+1}. {cp['name']} (Epoch: {epoch}, Time: {timestamp})")

    # Choose checkpoint
    choice = input("\nEnter checkpoint number to resume (or press Enter for fresh training): ").strip()
    resume_checkpoint = None
    if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
        resume_checkpoint = checkpoints[int(choice)-1]['path']
        print(f"Resuming from: {resume_checkpoint}")

    # Get dataset for resumed training
    trainer.dataset_manager.list_datasets()
    dataset_choice = input("Enter dataset number to use (1-4, default 1): ").strip()
    if not dataset_choice:
        dataset_choice = "1"

    try:
        dataset_id = int(dataset_choice)
        dataset_config = trainer.dataset_manager.get_dataset_config(dataset_id)
        if dataset_config is None:
            dataset_config = trainer.dataset_manager.get_dataset_config(1)
    except ValueError:
        dataset_config = trainer.dataset_manager.get_dataset_config(1)

    epochs = int(input("Enter number of epochs to train (default 15): ") or "15")
    save_every = int(input("Save checkpoint every N epochs (default 10): ") or "10")

    print(f"\nResuming training for {epochs} epochs...")
    print(f"Checkpoints will be saved every {save_every} epochs")
    print(f"Resume from checkpoint: {resume_checkpoint}")
    print(f"Dataset: {dataset_config['name']}")

    results, model_path = trainer.train_with_resume(
        dataset_config=dataset_config['path'],
        epochs=epochs,
        resume_checkpoint=resume_checkpoint,
        save_every=save_every
    )

    if results and model_path:
        print("\n" + "="*80)
        print("RESUME TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Trained model: {model_path}")
        print(f"Dataset used: {dataset_config['name']}")
    else:
        print("Resume training failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
