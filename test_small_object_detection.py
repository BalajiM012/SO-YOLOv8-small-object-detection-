u"""
Test Small Object Detection Script
Loads a YOLO model (yolov8x.pt) and tests it on a sample image from the VOC2012 dataset.
Displays detection results and saves annotated image.
"""

import torch
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def load_model(model_path='yolov8x.pt'):
    """Load the YOLO model"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

def get_sample_image():
    """Use the provided sample image path"""
    image_path = r"C:\Users\admin\Desktop\9999992_00000_d_0000040.jpg"
    if not os.path.exists(image_path):
        print(f"Sample image not found: {image_path}")
        return None
    print(f"Using sample image: {image_path}")
    return image_path

def run_inference(model, image_path):
    """Run inference on the sample image"""
    print(f"Running inference on: {image_path}")
    results = model.predict(source=image_path, save=True, conf=0.25, iou=0.45, project='runs', name='Detected images')
    return results

def display_results(results):
    """Display detection results"""
    for result in results:
        print("Detection Results:")
        print(f"  Image: {result.path}")
        print(f"  Detected objects: {len(result.boxes)}")
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                print(f"    - {class_name}: {conf:.2f}")
        else:
            print("    No objects detected")

def main():
    # Load the model
    model = load_model('yolov8x.pt')
    if model is None:
        return

    # Get sample image
    sample_image = get_sample_image()
    if sample_image is None:
        return

    # Run inference
    results = run_inference(model, sample_image)

    # Display results
    display_results(results)

    print("Inference completed. Check the 'runs/detect' folder for saved results.")

if __name__ == "__main__":
    main()
