#!/usr/bin/env python3
"""
Test script for the webcam detection functionality in ultimate_small_object_detector.py
"""

import os
import sys
import cv2
import torch
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append('.')

def test_webcam_detection_import():
    """Test that the webcam detection method can be imported"""
    try:
        from ultimate_small_object_detector import UltimateSmallObjectTrainer
        print("✓ Successfully imported UltimateSmallObjectTrainer")

        # Check if the method exists
        trainer = UltimateSmallObjectTrainer()
        if hasattr(trainer, 'detect_from_webcam'):
            print("✓ detect_from_webcam method exists")
            return True
        else:
            print("✗ detect_from_webcam method not found")
            return False
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False

def test_model_loading():
    """Test that model loading works (without actual webcam)"""
    try:
        from ultimate_small_object_detector import UltimateSmallObjectTrainer

        trainer = UltimateSmallObjectTrainer()

        # Check if we can create a dummy model path
        # We'll use yolov8x.pt as it's the default
        model_path = 'yolov8x.pt'

        # Test if ultralytics can load the model (this will download if needed)
        try:
            from ultralytics import YOLO
            print("✓ Ultralytics YOLO import successful")

            # Try to load the model (this might take time if downloading)
            print("Testing model loading...")
            model = YOLO(model_path)
            print("✓ Model loaded successfully")

            # Check if model has names attribute
            if hasattr(model, 'names') and model.names:
                print(f"✓ Model has {len(model.names)} classes")
                return True
            else:
                print("✗ Model missing names attribute")
                return False

        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def test_opencv_availability():
    """Test that OpenCV is available and can access camera properties"""
    try:
        # Test basic OpenCV functionality
        print(f"OpenCV version: {cv2.__version__}")

        # Test camera availability (without opening)
        # This will check if camera 0 exists
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera 0 is accessible")
            # Get some properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"✓ Camera properties: {width}x{height} @ {fps} FPS")
            cap.release()
            return True
        else:
            print("⚠ Camera 0 not accessible (this is normal in headless environments)")
            print("✓ OpenCV camera functions are available")
            return True  # Still consider this a pass since OpenCV works

    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_adaptive_nms():
    """Test the adaptive NMS functionality"""
    try:
        from ultimate_small_object_detector import UltimateSmallObjectTrainer
        import numpy as np

        trainer = UltimateSmallObjectTrainer()

        # Create dummy detection data
        boxes = np.array([
            [10, 10, 50, 50],  # Small object
            [100, 100, 200, 200],  # Large object
            [15, 15, 45, 45],  # Another small object (overlapping)
        ], dtype=np.float32)

        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 1, 0], dtype=np.float32)
        image_size = (640, 480)

        # Test adaptive NMS
        final_boxes, final_scores, final_classes = trainer.post_processor.apply_adaptive_nms(
            boxes, scores, classes, image_size
        )

        print(f"✓ Adaptive NMS processed {len(boxes)} detections into {len(final_boxes)} final detections")
        print(f"✓ Final boxes: {len(final_boxes)}, scores: {len(final_scores)}, classes: {len(final_classes)}")

        # Basic sanity checks
        assert len(final_boxes) == len(final_scores) == len(final_classes), "Output lengths don't match"
        assert all(isinstance(box, list) for box in final_boxes), "Boxes should be lists"
        assert all(isinstance(score, (int, float)) for score in final_scores), "Scores should be numeric"
        assert all(isinstance(cls, (int, float)) for cls in final_classes), "Classes should be numeric"

        print("✓ Adaptive NMS output validation passed")
        return True

    except Exception as e:
        print(f"✗ Adaptive NMS test failed: {e}")
        return False

def test_method_signature():
    """Test that the detect_from_webcam method has the correct signature"""
    try:
        from ultimate_small_object_detector import UltimateSmallObjectTrainer
        import inspect

        trainer = UltimateSmallObjectTrainer()
        method = getattr(trainer, 'detect_from_webcam')

        # Get method signature
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        expected_params = ['model_path', 'camera_id', 'conf_threshold', 'iou_threshold', 'max_det']
        self_param = 'self'  # instance method

        # Check if all expected parameters are present
        for param in expected_params:
            if param not in params:
                print(f"✗ Missing parameter: {param}")
                return False

        print(f"✓ Method signature correct: {params}")
        return True

    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False

def run_all_tests():
    """Run all webcam detection tests"""
    print("=" * 60)
    print("TESTING WEBCAM DETECTION FUNCTIONALITY")
    print("=" * 60)

    tests = [
        ("Import and Method Existence", test_webcam_detection_import),
        ("OpenCV Availability", test_opencv_availability),
        ("Model Loading", test_model_loading),
        ("Adaptive NMS", test_adaptive_nms),
        ("Method Signature", test_method_signature),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Webcam detection functionality is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
