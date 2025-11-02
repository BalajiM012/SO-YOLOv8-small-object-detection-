import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'matplotlib',
        'numpy',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed successfully!")
    else:
        print("\nAll required packages are already installed!")

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n✓ GPU Available: {gpu_name}")
            print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
            print(f"✓ GPU Count: {gpu_count}")
            return True
        else:
            print("\n✗ No GPU available. Training will be slow on CPU.")
            return False
    except ImportError:
        print("\n✗ PyTorch not installed. Cannot check GPU.")
        return False

def check_datasets():
    """Check if datasets are available"""
    print("\nChecking dataset availability...")
    
    # Check COCO dataset
    coco_path = "./coco_dataset"
    if os.path.exists(coco_path):
        print(f"✓ COCO dataset found at: {coco_path}")
    else:
        print(f"✗ COCO dataset not found at: {coco_path}")
        print("  Please download and organize the MS-COCO dataset as:")
        print("  coco_dataset/")
        print("  ├── images/")
        print("  │   ├── train2017/")
        print("  │   └── val2017/")
        print("  └── labels/")
        print("      ├── train2017/")
        print("      └── val2017/")
    
    # Check VOC dataset
    voc_path = "D:/MINI Project Phase 1/Object Detection DL/Object DL DATASETS/SO - YOLO/PASCAL VOC 2012 DATASET"
    if os.path.exists(voc_path):
        print(f"✓ PASCAL VOC 2012 dataset found at: {voc_path}")
    else:
        print(f"✗ PASCAL VOC 2012 dataset not found at: {voc_path}")
        print("  Please ensure the dataset is available at the specified path.")

def create_directories():
    """Create necessary directories"""
    directories = [
        "runs/detect",
        "voc2012_test_results",
        "voc2012_test_results/predictions",
        "voc2012_test_results/visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function"""
    print("YOLOv8X Training Setup")
    print("="*50)
    
    # Check requirements
    print("1. Checking requirements...")
    check_requirements()
    
    # Check GPU
    print("\n2. Checking GPU availability...")
    gpu_available = check_gpu()
    
    # Check datasets
    print("\n3. Checking datasets...")
    check_datasets()
    
    # Create directories
    print("\n4. Creating directories...")
    create_directories()
    
    print("\n" + "="*50)
    print("SETUP COMPLETED!")
    print("="*50)
    
    if gpu_available:
        print("\nYou're ready to start training!")
        print("Run: python train_and_test_yolo.py")
    else:
        print("\nWarning: No GPU detected. Training will be slow.")
        print("Consider using a GPU-enabled environment for better performance.")
    
    print("\nAvailable scripts:")
    print("- train_and_test_yolo.py: Complete training and testing pipeline")
    print("- train_yolo.py: Train only on MS-COCO dataset")
    print("- test_on_voc2012.py: Test trained model on PASCAL VOC 2012")

if __name__ == "__main__":
    main()
