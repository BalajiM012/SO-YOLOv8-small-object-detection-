from ultralytics import YOLO
import torch

model = YOLO('yolov8x.pt')

# Find a C2f module
for name, module in model.model.named_modules():
    if module.__class__.__name__ == 'C2f':
        print(f"Module: {name}")
        print(f"Type: {type(module)}")
        print(f"Attributes: {dir(module)}")
        print(f"Dict: {module.__dict__}")
        break
