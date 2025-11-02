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

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space
noteId: "db07f7c0865511f09dd2bb9b06426801"
tags: []

---

