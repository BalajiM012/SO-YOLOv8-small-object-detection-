---
noteId: "a57cc440b8c911f08966a38f1d154de1"
tags: []

---

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

The ultimate trainer uses the following key parameters (from `ultimate_small_object_detector.py`):

- **Model**: YOLOv8x (enhanced with SE + C2f)
- **Epochs**: 200 (with epoch splitting 100 + remaining)
- **Image size**: Ultra multi-scale `[320 … 768]` step 32
- **Batch size**: Adaptive to GPU memory (base 8, max 32)
- **Optimizer**: AdamW
- **Learning rate**: `lr0=0.005`, `lrf=1e-5`
- **Early stopping**: 50 epochs patience
- **Confidence/IoU**: `conf=0.001`, `iou=0.75`, `max_det=1000`
- **AMP**: Enabled
- **Save period**: every 5 epochs
- **Workers**: 8
- **Key augmentations**: mosaic=1.0, mixup=0.2, copy_paste=0.2, auto_augment=randaugment, erasing=0.7, hsv jitter, degrees=15, translate=0.25, scale=0.7, shear=3, perspective=0.002, flipud=0.15, fliplr=0.6

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
