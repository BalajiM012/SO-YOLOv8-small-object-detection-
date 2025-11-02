"""
Comprehensive Small Object Detection Testing Script
Tests all improvements and provides detailed metrics for small object detection
"""

import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

class SmallObjectMetrics:
    """Calculate comprehensive metrics for small object detection"""

    def __init__(self, small_threshold=32*32):  # 32x32 pixels area threshold
        self.small_threshold = small_threshold
        self.metrics = defaultdict(list)

    def calculate_object_size(self, bbox, img_width, img_height):
        """Calculate object area in pixels"""
        x1, y1, x2, y2 = bbox
        width = (x2 - x1) * img_width
        height = (y2 - y1) * img_height
        return width * height

    def is_small_object(self, bbox, img_width, img_height):
        """Determine if object is small based on area"""
        area = self.calculate_object_size(bbox, img_width, img_height)
        return area < self.small_threshold

    def update_metrics(self, predictions, ground_truth, img_width, img_height):
        """Update metrics with predictions and ground truth"""

        pred_boxes = predictions['boxes']
        pred_scores = predictions['scores']
        pred_classes = predictions['classes']

        gt_boxes = ground_truth['boxes']
        gt_classes = ground_truth['classes']

        # Separate small and large objects
        small_pred_indices = []
        large_pred_indices = []
        small_gt_indices = []
        large_gt_indices = []

        for i, box in enumerate(pred_boxes):
            if self.is_small_object(box, img_width, img_height):
                small_pred_indices.append(i)
            else:
                large_pred_indices.append(i)

        for i, box in enumerate(gt_boxes):
            if self.is_small_object(box, img_width, img_height):
                small_gt_indices.append(i)
            else:
                large_gt_indices.append(i)

        # Calculate metrics for small objects
        small_tp = 0
        small_fp = 0
        small_fn = len(small_gt_indices)

        # Simple matching for small objects (can be improved with IoU)
        for pred_idx in small_pred_indices:
            pred_class = pred_classes[pred_idx]
            pred_box = pred_boxes[pred_idx]

            best_iou = 0
            best_gt_idx = -1

            for gt_idx in small_gt_indices:
                if gt_classes[gt_idx] == pred_class:
                    iou = self.calculate_iou(pred_box, gt_boxes[gt_idx])
                    if iou > best_iou and iou > 0.5:  # IoU threshold
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_gt_idx != -1:
                small_tp += 1
                small_gt_indices.remove(best_gt_idx)
            else:
                small_fp += 1

        small_fn = len(small_gt_indices)

        # Store metrics
        self.metrics['small_tp'].append(small_tp)
        self.metrics['small_fp'].append(small_fp)
        self.metrics['small_fn'].append(small_fn)
        self.metrics['total_small_pred'].append(len(small_pred_indices))
        self.metrics['total_small_gt'].append(len(gt_boxes) - len(large_gt_indices))

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def get_final_metrics(self):
        """Calculate final metrics"""
        total_small_tp = sum(self.metrics['small_tp'])
        total_small_fp = sum(self.metrics['small_fp'])
        total_small_fn = sum(self.metrics['small_fn'])

        # Precision, Recall, F1 for small objects
        small_precision = total_small_tp / (total_small_tp + total_small_fp) if (total_small_tp + total_small_fp) > 0 else 0
        small_recall = total_small_tp / (total_small_tp + total_small_fn) if (total_small_tp + total_small_fn) > 0 else 0
        small_f1 = 2 * small_precision * small_recall / (small_precision + small_recall) if (small_precision + small_recall) > 0 else 0

        return {
            'small_precision': small_precision,
            'small_recall': small_recall,
            'small_f1': small_f1,
            'total_small_objects': sum(self.metrics['total_small_gt']),
            'detected_small_objects': total_small_tp,
            'avg_small_objects_per_image': np.mean(self.metrics['total_small_gt'])
        }

class ComprehensiveTester:
    """Comprehensive testing for small object detection"""

    def __init__(self, model_path, test_images_dir, test_labels_dir=None):
        self.model_path = model_path
        self.test_images_dir = Path(test_images_dir)
        self.test_labels_dir = Path(test_labels_dir) if test_labels_dir else None
        self.metrics = SmallObjectMetrics()
        self.results = []

        # Load model
        self.model = YOLO(model_path)

    def load_ground_truth(self, image_path):
        """Load ground truth annotations for an image"""
        if not self.test_labels_dir:
            return {'boxes': [], 'classes': []}

        label_path = self.test_labels_dir / (image_path.stem + '.txt')

        if not label_path.exists():
            return {'boxes': [], 'classes': []}

        boxes = []
        classes = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert to xyxy format
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    x2 = x_center + width/2
                    y2 = y_center + height/2

                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls)

        return {'boxes': boxes, 'classes': classes}

    def test_image(self, image_path, conf_threshold=0.001, iou_threshold=0.3):
        """Test single image and collect metrics"""

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img_height, img_width = img.shape[:2]

        # Load ground truth
        ground_truth = self.load_ground_truth(image_path)

        # Run inference
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold, max_det=1000)

        predictions = {'boxes': [], 'scores': [], 'classes': []}

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # Normalize boxes to 0-1
            boxes_normalized = []
            for box in boxes:
                x1, y1, x2, y2 = box
                boxes_normalized.append([
                    x1 / img_width, y1 / img_height,
                    x2 / img_width, y2 / img_height
                ])

            predictions = {
                'boxes': boxes_normalized,
                'scores': scores,
                'classes': classes
            }

        # Update metrics
        self.metrics.update_metrics(predictions, ground_truth, 1.0, 1.0)  # Normalized coordinates

        # Store results
        result = {
            'image_path': str(image_path),
            'image_size': (img_width, img_height),
            'predictions': predictions,
            'ground_truth': ground_truth,
            'num_predictions': len(predictions['boxes']),
            'num_ground_truth': len(ground_truth['boxes'])
        }

        self.results.append(result)
        return result

    def run_comprehensive_test(self, max_images=500, conf_threshold=0.001):
        """Run comprehensive testing on dataset"""

        print(f"Starting comprehensive small object detection test...")
        print(f"Model: {self.model_path}")
        print(f"Test images: {self.test_images_dir}")
        print(f"Max images: {max_images}")
        print(f"Confidence threshold: {conf_threshold}")

        # Get all test images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_images = []

        for ext in image_extensions:
            test_images.extend(self.test_images_dir.glob(ext))

        test_images = test_images[:max_images]
        print(f"Found {len(test_images)} test images")

        # Test each image
        for i, img_path in enumerate(test_images):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_images)} images...")

            try:
                self.test_image(img_path, conf_threshold=conf_threshold)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Calculate final metrics
        final_metrics = self.metrics.get_final_metrics()

        print("\n" + "="*60)
        print("COMPREHENSIVE SMALL OBJECT DETECTION RESULTS")
        print("="*60)
        print(".4f"        print(".4f"        print(".4f"        print(f"Total small objects in dataset: {final_metrics['total_small_objects']}")
        print(f"Detected small objects: {final_metrics['detected_small_objects']}")
        print(".2f"
        # Save detailed results
        self.save_results(final_metrics)

        return final_metrics

    def save_results(self, final_metrics):
        """Save detailed test results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"test_results_{timestamp}")
        results_dir.mkdir(exist_ok=True)

        # Save metrics
        with open(results_dir / "metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)

        # Save detailed results
        with open(results_dir / "detailed_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        # Create summary report
        summary = f"""
Small Object Detection Test Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Model: {self.model_path}
Test Images: {len(self.results)}

SMALL OBJECT METRICS:
- Precision: {final_metrics['small_precision']:.4f}
- Recall: {final_metrics['small_recall']:.4f}
- F1-Score: {final_metrics['small_f1']:.4f}
- Total Small Objects: {final_metrics['total_small_objects']}
- Detected Small Objects: {final_metrics['detected_small_objects']}
- Avg Small Objects per Image: {final_metrics['avg_small_objects_per_image']:.2f}

CONFIGURATION:
- Confidence Threshold: 0.001
- IoU Threshold: 0.3
- Small Object Threshold: 1024 pixels²
"""

        with open(results_dir / "summary.txt", 'w') as f:
            f.write(summary)

        print(f"Results saved to: {results_dir}")

        return results_dir

def main():
    """Main testing function"""

    # Model path (update with your trained model)
    model_path = "models/ultimate_small_object/final_model.pt"  # Update this path

    # Test data paths
    test_images_dir = "./voc2012_yolo_dataset/images"  # Update as needed
    test_labels_dir = "./voc2012_yolo_dataset/labels"  # Update as needed

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please update the model path in the script")
        return

    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return

    # Initialize tester
    tester = ComprehensiveTester(model_path, test_images_dir, test_labels_dir)

    # Run comprehensive test
    metrics = tester.run_comprehensive_test(max_images=200)  # Test on 200 images

    print("Comprehensive testing completed!")
    print(f"Small object F1-Score: {metrics['small_f1']:.4f}")

if __name__ == "__main__":
    main()
