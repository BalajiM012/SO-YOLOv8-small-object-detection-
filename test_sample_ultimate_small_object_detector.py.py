"""
Ultimate Small Object Detection Test Script
Uses the advanced UltimateSmallObjectTrainer to test models with enhanced post-processing
Tests on datasets with adaptive NMS and comprehensive statistics
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultimate_small_object_detector import UltimateSmallObjectTrainer

def load_dataset_config(config_path='voc2012_yolo_dataset.yaml'):
    """Load dataset configuration"""
    if not os.path.exists(config_path):
        print(f"Dataset config not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded dataset config: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def get_test_images_path(dataset_config):
    """Extract test images path from dataset config"""
    if not dataset_config:
        return None

    # Try 'test' first, then 'val' as fallback
    test_path = None
    if 'test' in dataset_config:
        test_path = dataset_config['test']
    elif 'val' in dataset_config:
        test_path = dataset_config['val']
        print("Using 'val' as test data since 'test' is not available")

    if not test_path:
        return None

    if isinstance(test_path, list):
        test_path = test_path[0]

    # If it's a relative path, combine with dataset path
    if 'path' in dataset_config and not os.path.isabs(test_path):
        base_path = dataset_config['path']
        test_path = os.path.join(base_path, test_path)

    if os.path.exists(test_path):
        print(f"Test images path: {test_path}")
        return test_path
    else:
        print(f"Test path does not exist: {test_path}")
        return None

def run_ultimate_test(model_path, test_data_path, max_images=100):
    """Run advanced testing using UltimateSmallObjectTrainer"""
    print(f"Running ultimate test on model: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Max images: {max_images}")

    # Initialize ultimate trainer
    trainer = UltimateSmallObjectTrainer(model_path=model_path, device_id=0)

    # Run advanced testing
    results = trainer.test_ultimate_model(model_path, test_data_path, max_images=max_images)

    return results

def display_advanced_results(results):
    """Display comprehensive test results"""
    if not results:
        print("No test results available")
        return

    print("\n" + "="*60)
    print("ULTIMATE SMALL OBJECT DETECTION TEST RESULTS")
    print("="*60)

    total_images = len(results)
    total_detections = sum(r['detections'] for r in results)
    avg_detections = total_detections / total_images if total_images > 0 else 0

    # Calculate detection statistics
    detection_counts = [r['detections'] for r in results]
    max_detections = max(detection_counts) if detection_counts else 0
    min_detections = min(detection_counts) if detection_counts else 0

    print(f"Total Images Processed: {total_images}")
    print(f"Total Detections: {total_detections}")
    print(f"Average Detections per Image: {avg_detections:.2f}")
    print(f"Max Detections in Single Image: {max_detections}")
    print(f"Min Detections in Single Image: {min_detections}")

    # Class-wise statistics
    class_stats = {}
    for result in results:
        for cls in result['classes']:
            cls_name = f"class_{int(cls)}"
            class_stats[cls_name] = class_stats.get(cls_name, 0) + 1

    print(f"\nClass Distribution ({len(class_stats)} classes found):")
    for cls, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} detections")

    # Sample results
    print("\nSample Results (first 5 images):")
    for i, result in enumerate(results[:5]):
        print(f"  Image {i+1}: {Path(result['image']).name} - {result['detections']} detections")

    print("\nAdvanced post-processing applied:")
    print("- Adaptive NMS with different thresholds for small vs large objects")
    print("- Stricter NMS for small objects (IoU=0.3) and relaxed for large objects (IoU=0.5)")
    print("- Enhanced detection confidence and IoU thresholds")

def run_single_image_test(model_path, image_path):
    """Run advanced testing on a single image"""
    print(f"Running single image test on: {image_path}")

    # Initialize ultimate trainer
    trainer = UltimateSmallObjectTrainer(model_path=model_path, device_id=0)

    # Load model
    model = YOLO(model_path)

    # Run inference with saving
    results = model.predict(source=image_path, save=True, conf=0.001, iou=0.3, max_det=1000, project='runs', name='single_image_test')

    # Apply advanced post-processing to the results
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Apply adaptive NMS
        final_boxes, final_scores, final_classes = trainer.post_processor.apply_adaptive_nms(
            boxes, scores, classes, (640, 640)  # Assume 640x640
        )

        print("Single Image Detection Results:")
        print(f"  Image: {image_path}")
        print(f"  Total detections: {len(final_boxes)}")
        for i, (box, score, cls) in enumerate(zip(final_boxes, final_scores, final_classes)):
            print(f"    Detection {i+1}: Class {int(cls)}, Confidence {score:.2f}, Box {box}")

        # Save results to file
        result_file = 'test_sample_ultimate_small_object_detector_results.txt'
        with open(result_file, 'w') as f:
            f.write(f"Results for image: {image_path}\n")
            f.write(f"Total detections: {len(final_boxes)}\n")
            for i, (box, score, cls) in enumerate(zip(final_boxes, final_scores, final_classes)):
                f.write(f"Detection {i+1}: Class {int(cls)}, Confidence {score:.2f}, Box {box}\n")
        print(f"Results saved to: {result_file}")

    else:
        print("No detections found in the image")

    return results

def main():
    # Configuration
    model_path = 'yolov8x.pt'  # Can be changed to a trained ultimate model path
    dataset_config_path = 'voc2012_yolo_dataset.yaml'  # Dataset configuration
    max_test_images = 100  # Limit for testing

    # Single image path for testing
    single_image_path = r"C:\Users\admin\Desktop\0000025_03777_d_0000005.jpg"

    print("Ultimate Small Object Detection Test")
    print("="*40)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please ensure the model file exists or update the model_path variable")
        return

    # Check if single image path is provided and exists
    if os.path.exists(single_image_path):
        print(f"Single image mode: Testing image {single_image_path}")
        try:
            run_single_image_test(model_path, single_image_path)
            print("\nSingle image testing completed successfully!")
        except Exception as e:
            print(f"Error during single image testing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Single image not found: {single_image_path}")
        print("Falling back to dataset testing mode")

        # Load dataset configuration
        dataset_config = load_dataset_config(dataset_config_path)
        if dataset_config is None:
            print("Failed to load dataset configuration")
            return

        # Get test images path
        test_images_path = get_test_images_path(dataset_config)
        if test_images_path is None:
            print("Could not determine test images path from config")
            return

        # Run ultimate testing
        try:
            results = run_ultimate_test(model_path, test_images_path, max_test_images)
            display_advanced_results(results)
            print("\nUltimate testing completed successfully!")
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
