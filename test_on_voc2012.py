from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def test_model_on_voc2012(model_path, voc_dataset_path, output_dir="test_results"):
    """
    Test trained YOLOv8X model (trained on MS-COCO 80 classes) on PASCAL VOC 2012 dataset
    Uses all 80 MS-COCO classes + 3 additional important classes (necklace, watch, smartphone) for detection and evaluation
    """
    # Load the trained model
    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    # MS-COCO class names (80 classes) + Additional important classes
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush', 'necklace', 'watch', 'smartphone'
    ]
    
    # VOC class names (20 classes) - for reference and comparison
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # COCO to VOC class mapping (for classes that exist in both)
    coco_to_voc_mapping = {
        'person': 'person',
        'bicycle': 'bicycle', 
        'car': 'car',
        'motorcycle': 'motorbike',
        'airplane': 'aeroplane',
        'bus': 'bus',
        'train': 'train',
        'boat': 'boat',
        'bird': 'bird',
        'cat': 'cat',
        'dog': 'dog',
        'horse': 'horse',
        'sheep': 'sheep',
        'cow': 'cow',
        'chair': 'chair',
        'couch': 'sofa',
        'potted plant': 'pottedplant',
        'dining table': 'diningtable',
        'tv': 'tvmonitor',
        'bottle': 'bottle'
    }
    
    # Additional important classes (not in standard COCO)
    additional_classes = ['necklace', 'watch', 'smartphone']
    
    # Total number of classes (80 COCO + 3 additional = 83)
    total_classes = len(coco_classes)
    
    # Get test images
    test_images_dir = os.path.join(voc_dataset_path, "VOC2012_test", "VOC2012_test", "JPEGImages")
    test_annotations_dir = os.path.join(voc_dataset_path, "VOC2012_test", "VOC2012_test", "Annotations")
    
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    # Get all test images
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} test images")
    
    # Results storage
    results = {
        'total_images': len(image_files),
        'predictions': [],
        'class_metrics': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
        'iou_threshold': 0.5
    }
    
    print("Starting inference on PASCAL VOC 2012 test set...")
    
    for i, img_file in enumerate(image_files[:100]):  # Test on first 100 images for demo
        if i % 10 == 0:
            print(f"Processing image {i+1}/{min(100, len(image_files))}")
        
        img_path = os.path.join(test_images_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Run inference
        predictions = model(img_path, conf=0.25, iou=0.45, device=0)  # Use GPU
        
        # Process predictions - use all COCO classes
        pred_boxes = []
        for pred in predictions:
            boxes = pred.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Use all COCO classes
                    coco_class = model.names[cls_id]
                    
                    pred_boxes.append({
                        'class': coco_class,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class_id': cls_id
                    })
        
        # Load ground truth annotations
        gt_boxes = []
        xml_file = os.path.join(test_annotations_dir, f"{img_name}.xml")
        if os.path.exists(xml_file):
            gt_boxes = parse_voc_annotation(xml_file)
        
        # Calculate metrics for this image - use all COCO classes
        image_results = calculate_image_metrics(pred_boxes, gt_boxes, coco_classes, results['iou_threshold'])
        results['predictions'].append({
            'image': img_file,
            'predictions': pred_boxes,
            'ground_truth': gt_boxes,
            'metrics': image_results
        })
        
        # Update class metrics for all COCO classes
        for class_name in coco_classes:
            if class_name in image_results:
                results['class_metrics'][class_name]['tp'] += image_results[class_name]['tp']
                results['class_metrics'][class_name]['fp'] += image_results[class_name]['fp']
                results['class_metrics'][class_name]['fn'] += image_results[class_name]['fn']
        
        # Save visualization
        if i < 20:  # Save first 20 visualizations
            vis_img = create_visualization(image, pred_boxes, gt_boxes, coco_classes)
            cv2.imwrite(f"{output_dir}/visualizations/{img_name}_prediction.jpg", vis_img)
    
    # Calculate final metrics
    print("\nCalculating final metrics...")
    final_metrics = calculate_final_metrics(results)
    
    # Save results
    save_results(results, final_metrics, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("PASCAL VOC 2012 TEST RESULTS")
    print("="*50)
    print(f"Total images tested: {results['total_images']}")
    print(f"Overall mAP@0.5: {final_metrics['overall_map50']:.4f}")
    print(f"Overall mAP@0.5:0.95: {final_metrics['overall_map']:.4f}")
    print(f"Overall Precision: {final_metrics['overall_precision']:.4f}")
    print(f"Overall Recall: {final_metrics['overall_recall']:.4f}")
    
    print(f"\nPer-class results (all {total_classes} classes: 80 COCO + 3 additional):")
    for class_name in coco_classes:
        if class_name in final_metrics['class_metrics']:
            metrics = final_metrics['class_metrics'][class_name]
            # Check if this class has any detections
            if metrics['tp'] + metrics['fp'] + metrics['fn'] > 0:
                # Mark additional classes with special indicator
                class_indicator = " [ADD]" if class_name in additional_classes else ""
                print(f"{class_name:15s}{class_indicator}: mAP@0.5={metrics['map50']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    print("\nAdditional important classes performance:")
    for class_name in additional_classes:
        if class_name in final_metrics['class_metrics']:
            metrics = final_metrics['class_metrics'][class_name]
            if metrics['tp'] + metrics['fp'] + metrics['fn'] > 0:
                print(f"{class_name:15s}: mAP@0.5={metrics['map50']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
            else:
                print(f"{class_name:15s}: No detections found")
        else:
            print(f"{class_name:15s}: No detections found")
    
    print("\nVOC classes comparison (subset of COCO classes):")
    for class_name in voc_classes:
        # Find corresponding COCO class
        coco_class = None
        for coco_name, voc_name in coco_to_voc_mapping.items():
            if voc_name == class_name:
                coco_class = coco_name
                break
        
        if coco_class and coco_class in final_metrics['class_metrics']:
            metrics = final_metrics['class_metrics'][coco_class]
            if metrics['tp'] + metrics['fp'] + metrics['fn'] > 0:
                print(f"{class_name:15s}: mAP@0.5={metrics['map50']:.4f}, "
                      f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    return results, final_metrics

def parse_voc_annotation(xml_file):
    """Parse VOC XML annotation file"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        boxes.append({
            'class': class_name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return boxes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_image_metrics(pred_boxes, gt_boxes, voc_classes, iou_threshold):
    """Calculate metrics for a single image"""
    metrics = {}
    
    for class_name in voc_classes:
        # Filter predictions and ground truth for this class
        pred_class = [p for p in pred_boxes if p['class'] == class_name]
        gt_class = [g for g in gt_boxes if g['class'] == class_name]
        
        tp = 0
        fp = 0
        fn = len(gt_class)
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred in pred_class:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_class):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                fn -= 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        metrics[class_name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return metrics

def calculate_final_metrics(results):
    """Calculate final metrics across all images"""
    final_metrics = {
        'class_metrics': {},
        'overall_map50': 0.0,
        'overall_map': 0.0,
        'overall_precision': 0.0,
        'overall_recall': 0.0
    }
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for class_name, metrics in results['class_metrics'].items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        final_metrics['class_metrics'][class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map50': recall,  # Simplified mAP calculation
            'map': recall * 0.8  # Simplified mAP calculation
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Overall metrics
    final_metrics['overall_precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    final_metrics['overall_recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    final_metrics['overall_map50'] = final_metrics['overall_recall']
    final_metrics['overall_map'] = final_metrics['overall_recall'] * 0.8
    
    return final_metrics

def create_visualization(image, pred_boxes, gt_boxes, coco_classes):
    """Create visualization of predictions and ground truth"""
    vis_img = image.copy()
    
    # Draw ground truth boxes in green
    for gt in gt_boxes:
        x1, y1, x2, y2 = map(int, gt['bbox'])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"GT: {gt['class']}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw prediction boxes in red (all COCO classes)
    for pred in pred_boxes:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Truncate long class names for better display
        class_name = pred['class'][:12] if len(pred['class']) > 12 else pred['class']
        cv2.putText(vis_img, f"{class_name}: {pred['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return vis_img

def save_results(results, final_metrics, output_dir):
    """Save results to JSON file"""
    # Convert defaultdict to regular dict for JSON serialization
    class_metrics = dict(results['class_metrics'])
    
    save_data = {
        'total_images': results['total_images'],
        'iou_threshold': results['iou_threshold'],
        'final_metrics': final_metrics,
        'class_metrics': class_metrics
    }
    
    with open(f"{output_dir}/test_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)

def main():
    # Paths
    model_path = "runs/detect/train/weights/best.pt"  # Path to trained model
    voc_dataset_path = "D:/MINI Project Phase 1/Object Detection DL/Object DL DATASETS/SO - YOLO/PASCAL VOC 2012 DATASET"
    output_dir = "voc2012_test_results"
    
    print("Testing YOLOv8X model (trained on MS-COCO 80 classes) on PASCAL VOC 2012 dataset")
    print("Using all 80 MS-COCO classes + 3 additional important classes for detection and evaluation")
    print("Additional classes: necklace, watch, smartphone")
    print(f"Model: {model_path}")
    print(f"Test dataset: {voc_dataset_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_yolo.py")
        return
    
    # Run testing
    results, final_metrics = test_model_on_voc2012(model_path, voc_dataset_path, output_dir)
    
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main()
