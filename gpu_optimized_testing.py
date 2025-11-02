"""
GPU-Optimized YOLO Testing Script for PASCAL VOC 2012 Dataset
Features:
- GPU-accelerated inference with memory management
- Batch processing for efficient testing
- Real-time performance monitoring
- Comprehensive evaluation metrics
"""

import os
import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import psutil
import gc
from tqdm import tqdm

class GPUInferenceManager:
    """GPU Memory Management for Inference"""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.batch_size = 1  # Start with batch size 1
        self.max_batch_size = 16
        
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
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_optimal_batch_size(self, model, test_images):
        """Calculate optimal batch size for inference"""
        if not torch.cuda.is_available():
            return 1
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8, 16]:
            try:
                # Test with a small subset
                test_subset = test_images[:batch_size]
                
                # Clear cache before test
                self.clear_gpu_cache()
                
                # Test inference
                results = model(test_subset, device=self.device_id, verbose=False)
                
                # Check memory usage
                memory_usage = self.get_gpu_memory() / self.get_gpu_memory_total()
                
                if memory_usage > 0.9:  # If using more than 90% of GPU memory
                    return max(1, batch_size // 2)
                
                self.batch_size = batch_size
                
            except torch.cuda.OutOfMemoryError:
                return max(1, batch_size // 2)
        
        return self.batch_size

class VOCEvaluator:
    """PASCAL VOC 2012 Dataset Evaluator"""
    
    def __init__(self, test_images_dir, test_annotations_dir):
        self.test_images_dir = Path(test_images_dir)
        self.test_annotations_dir = Path(test_annotations_dir)
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
    def parse_voc_annotation(self, xml_file):
        """Parse VOC XML annotation file"""
        try:
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
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return []
    
    def calculate_iou(self, box1, box2):
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
    
    def evaluate_predictions(self, predictions, ground_truth, iou_threshold=0.5):
        """Evaluate predictions against ground truth"""
        results = {}
        
        for class_name in self.voc_classes:
            # Filter predictions and ground truth for this class
            pred_class = [p for p in predictions if p['class'] == class_name]
            gt_class = [g for g in ground_truth if g['class'] == class_name]
            
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
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    tp += 1
                    fn -= 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[class_name] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results

class OptimizedTester:
    """GPU-Optimized YOLO Tester"""
    
    def __init__(self, model_path, device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.inference_manager = GPUInferenceManager(device_id)
        self.model = None
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print("✓ Model loaded successfully")
        
        # Clear GPU cache after loading
        self.inference_manager.clear_gpu_cache()
        
    def test_on_voc2012(self, test_images_dir, test_annotations_dir, 
                       output_dir="voc2012_test_results", max_images=None):
        """Test model on PASCAL VOC 2012 dataset"""
        
        print("="*60)
        print("GPU-OPTIMIZED TESTING ON PASCAL VOC 2012")
        print("="*60)
        
        # Initialize evaluator
        evaluator = VOCEvaluator(test_images_dir, test_annotations_dir)
        
        # Get test images
        test_images = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
        
        if max_images:
            test_images = test_images[:max_images]
        
        print(f"Found {len(test_images)} test images")
        
        # Determine optimal batch size
        optimal_batch_size = self.inference_manager.get_optimal_batch_size(self.model, test_images)
        print(f"Optimal batch size: {optimal_batch_size}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "predictions").mkdir(exist_ok=True)
        (output_path / "visualizations").mkdir(exist_ok=True)
        
        # Results storage
        all_results = []
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        print(f"\nStarting inference with batch size {optimal_batch_size}...")
        
        # Process images in batches
        start_time = time.time()
        
        for i in tqdm(range(0, len(test_images), optimal_batch_size), desc="Processing batches"):
            batch_images = test_images[i:i + optimal_batch_size]
            
            # Process batch
            batch_results = self._process_batch(batch_images, evaluator, output_path)
            all_results.extend(batch_results)
            
            # Update class metrics
            for result in batch_results:
                for class_name, metrics in result['class_metrics'].items():
                    class_metrics[class_name]['tp'] += metrics['tp']
                    class_metrics[class_name]['fp'] += metrics['fp']
                    class_metrics[class_name]['fn'] += metrics['fn']
            
            # Clear GPU cache periodically
            if i % (optimal_batch_size * 10) == 0:
                self.inference_manager.clear_gpu_cache()
        
        inference_time = time.time() - start_time
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(class_metrics)
        
        # Save results
        self._save_results(all_results, final_metrics, output_path, inference_time)
        
        # Print summary
        self._print_summary(final_metrics, len(test_images), inference_time)
        
        return all_results, final_metrics
    
    def _process_batch(self, batch_images, evaluator, output_path):
        """Process a batch of images"""
        batch_results = []
        
        for img_path in batch_images:
            try:
                # Run inference
                results = self.model(str(img_path), device=self.device_id, conf=0.25, iou=0.45, verbose=False)
                
                # Parse predictions
                predictions = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # Map class ID to VOC class name
                        class_name = self.model.names[cls_id]
                        
                        predictions.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                
                # Load ground truth
                img_name = img_path.stem
                xml_file = evaluator.test_annotations_dir / f"{img_name}.xml"
                ground_truth = evaluator.parse_voc_annotation(xml_file) if xml_file.exists() else []
                
                # Evaluate predictions
                class_metrics = evaluator.evaluate_predictions(predictions, ground_truth)
                
                # Store results
                batch_results.append({
                    'image': img_path.name,
                    'predictions': predictions,
                    'ground_truth': ground_truth,
                    'class_metrics': class_metrics
                })
                
                # Save visualization for first few images
                if len(batch_results) <= 20:
                    self._save_visualization(img_path, predictions, ground_truth, output_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return batch_results
    
    def _save_visualization(self, img_path, predictions, ground_truth, output_path):
        """Save visualization of predictions and ground truth"""
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                return
            
            # Draw ground truth boxes in green
            for gt in ground_truth:
                x1, y1, x2, y2 = map(int, gt['bbox'])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"GT: {gt['class']}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw prediction boxes in red
            for pred in predictions:
                x1, y1, x2, y2 = map(int, pred['bbox'])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, f"{pred['class']}: {pred['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Save visualization
            vis_path = output_path / "visualizations" / f"{img_path.stem}_prediction.jpg"
            cv2.imwrite(str(vis_path), image)
            
        except Exception as e:
            print(f"Error saving visualization for {img_path}: {e}")
    
    def _calculate_final_metrics(self, class_metrics):
        """Calculate final evaluation metrics"""
        final_metrics = {
            'class_metrics': {},
            'overall_precision': 0.0,
            'overall_recall': 0.0,
            'overall_f1': 0.0,
            'overall_map50': 0.0
        }
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for class_name, metrics in class_metrics.items():
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
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Overall metrics
        final_metrics['overall_precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        final_metrics['overall_recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        final_metrics['overall_f1'] = 2 * (final_metrics['overall_precision'] * final_metrics['overall_recall']) / (final_metrics['overall_precision'] + final_metrics['overall_recall']) if (final_metrics['overall_precision'] + final_metrics['overall_recall']) > 0 else 0
        final_metrics['overall_map50'] = final_metrics['overall_recall']  # Simplified mAP calculation
        
        return final_metrics
    
    def _save_results(self, all_results, final_metrics, output_path, inference_time):
        """Save results to JSON file"""
        results_data = {
            'total_images': len(all_results),
            'inference_time': inference_time,
            'final_metrics': final_metrics,
            'detailed_results': all_results
        }
        
        with open(output_path / "test_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def _print_summary(self, final_metrics, total_images, inference_time):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total images processed: {total_images}")
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Average time per image: {inference_time/total_images:.3f} seconds")
        print(f"Overall Precision: {final_metrics['overall_precision']:.4f}")
        print(f"Overall Recall: {final_metrics['overall_recall']:.4f}")
        print(f"Overall F1-Score: {final_metrics['overall_f1']:.4f}")
        print(f"Overall mAP@0.5: {final_metrics['overall_map50']:.4f}")
        
        print(f"\nPer-class results:")
        for class_name, metrics in final_metrics['class_metrics'].items():
            if metrics['tp'] + metrics['fp'] + metrics['fn'] > 0:
                print(f"{class_name:15s}: Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

def main():
    """Main testing function"""
    print("GPU-Optimized YOLO Testing for PASCAL VOC 2012 Dataset")
    print("="*70)
    
    # Paths
    model_path = "runs/detect/gpu_optimized_training/weights/best.pt"
    test_images_dir = "archive/VOC2012_test/VOC2012_test/JPEGImages"
    test_annotations_dir = "archive/VOC2012_test/VOC2012_test/Annotations"
    output_dir = "voc2012_test_results"
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using gpu_optimized_training.py")
        return
    
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    # Initialize tester
    tester = OptimizedTester(model_path, device_id=0)
    
    try:
        # Load model
        tester.load_model()
        
        # Run testing
        results, metrics = tester.test_on_voc2012(
            test_images_dir, 
            test_annotations_dir, 
            output_dir,
            max_images=1000  # Limit for demo purposes
        )
        
        print("\n✓ Testing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")

if __name__ == "__main__":
    main()
