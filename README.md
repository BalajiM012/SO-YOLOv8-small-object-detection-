# YOLO v8 Training on PASCAL VOC 2012 Dataset

This project sets up and runs YOLO v8 training on the PASCAL VOC 2012 dataset.

## Dataset Structure

The PASCAL VOC 2012 dataset contains:
- **Training images**: 5,717 images
- **Validation images**: 5,823 images  
- **Classes**: 20 object categories (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Convert Dataset

The PASCAL VOC dataset uses XML annotations, but YOLO requires a different format. Run the conversion script:

```bash
python convert_voc_to_yolo.py
```

This will:
- Create a `dataset/` directory with proper YOLO structure
- Convert XML annotations to YOLO format
- Copy images to appropriate train/val folders
- Generate label files in YOLO format

### 3. Verify Dataset Structure

After conversion, you should have:
```
dataset/
├── images/
│   ├── train/     # Training images
│   └── val/       # Validation images
├── labels/
│   ├── train/     # Training labels
│   └── val/       # Validation labels
└── dataset.yaml   # Dataset configuration
```

### 4. Start Training

```bash
python train_yolo.py
```

## Training Configuration

The training script uses the following key parameters:
- **Model**: YOLOv8 nano (yolov8n.pt)
- **Epochs**: 100
- **Image size**: 640x640
- **Batch size**: 16
- **Learning rate**: 0.01
- **Early stopping**: 50 epochs patience

## Expected Output

Training will create:
- Model checkpoints every 10 epochs
- Training plots and metrics
- Best model saved automatically
- Validation results after training

## Performance Metrics

The model will be evaluated on:
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision**: Overall precision
- **Recall**: Overall recall

## Customization

You can modify training parameters in `train_yolo.py`:
- Change model size (nano, small, medium, large, xlarge)
- Adjust learning rate, batch size, epochs
- Modify loss weights and optimization settings

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or image size
2. **Slow training**: Enable image caching or reduce workers
3. **Poor convergence**: Adjust learning rate or increase epochs

## Ultimate Small Object Detector

This project includes an advanced `UltimateSmallObjectTrainer` class designed specifically for small object detection tasks. It combines multiple cutting-edge techniques to achieve superior performance on small object detection challenges.

### Features

- **GPU Memory Management**: Intelligent batch size optimization and memory monitoring
- **Advanced Post-Processing**: Adaptive NMS with different thresholds for small vs large objects
- **Hyperparameter Optimization**: Grid search and ensemble methods for optimal performance
- **Multi-Dataset Training**: Support for training on multiple datasets simultaneously
- **Data Quality Enhancement**: Automatic data cleaning, class balancing, and augmentation
- **Checkpoint Management**: Resume training from checkpoints with epoch splitting
- **Blur Augmentation**: Custom blur augmentation for improved small object detection
- **Ensemble Methods**: Combine multiple models for better accuracy

### Usage

#### Basic Training

```python
from ultimate_small_object_detector import UltimateSmallObjectTrainer

# Initialize trainer
trainer = UltimateSmallObjectTrainer(model_path='yolov8x.pt', device_id=0)

# Train on a dataset
model, model_path = trainer.train_ultimate_model('voc2012.yaml', epochs=200)
```

#### Testing

```python
# Test the trained model
results = trainer.test_ultimate_model(model_path, 'test_images/', max_images=1000)
print(f"Average detections per image: {sum(r['detections'] for r in results) / len(results):.2f}")
```

#### Ensemble Training

```python
# Train on multiple datasets
trained_models = []
datasets = ['voc2012.yaml', 'visdrone.yaml', 'tinyperson.yaml']

for dataset in datasets:
    if os.path.exists(dataset):
        model, path = trainer.train_ultimate_model(dataset, epochs=200)
        trained_models.append(path)

# Ensemble evaluation
precision, recall, map50 = trainer.ensemble_models(trained_models, 'test_data')
```

### Target Performance

The system is optimized to achieve:
- **Precision**: ≥1.00
- **Recall**: ≥0.89
- **mAP@0.5**: ≥0.79

### Testing

Run the comprehensive test suite:

```bash
python test_ultimate_small_object_detector.py
```

This will test all major components including GPU management, post-processing, hyperparameter optimization, and ensemble methods.

### Configuration

The trainer supports extensive configuration options:

```python
config = trainer.create_ultimate_training_config(
    dataset_config='voc2012.yaml',
    epochs=200,
    use_blur_augmentation=True
)
```

Key parameters include multi-scale training, advanced augmentations, optimized loss weights, and automatic mixed precision training.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space
noteId: "db07f7c0865511f09dd2bb9b06426801"
tags: []

---

