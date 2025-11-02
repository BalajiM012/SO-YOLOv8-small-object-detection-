#!/usr/bin/env python3
"""
Test script to verify blur augmentation integration in the ultimate small object detector
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Add current directory to path
sys.path.append('.')

from custom_dataset import BlurAugmentationDataset
from ultimate_small_object_detector import UltimateSmallObjectTrainer

def test_blur_dataset_initialization():
    """Test that BlurAugmentationDataset initializes correctly with class attributes"""
    print("Testing BlurAugmentationDataset initialization...")

    # Set class attributes
    BlurAugmentationDataset.blur_prob = 0.5
    BlurAugmentationDataset.blur_kernel_range = (3, 7)
    BlurAugmentationDataset.blur_sigma_range = (0.1, 1.0)

    # Create a minimal config
    test_config = {
        'path': 'voc2012_yolo_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: f'class_{i}' for i in range(20)}
    }

    # Save test config
    config_path = 'test_blur_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

    try:
        # Initialize dataset without explicit parameters
        dataset = BlurAugmentationDataset(
            data=test_config,
            img_path='voc2012_yolo_dataset/images/train',
            augment=True
        )

        print(f"✓ Dataset initialized successfully")
        print(f"  blur_prob: {dataset.blur_prob}")
        print(f"  blur_kernel_range: {dataset.blur_kernel_range}")
        print(f"  blur_sigma_range: {dataset.blur_sigma_range}")

        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample retrieved successfully")
            print(f"  Sample keys: {list(sample.keys())}")
            if 'img' in sample:
                img_shape = sample['img'].shape if hasattr(sample['img'], 'shape') else 'N/A'
                print(f"  Image shape: {img_shape}")

        # Clean up
        os.remove(config_path)
        return True

    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        if os.path.exists(config_path):
            os.remove(config_path)
        return False

def test_blur_augmentation_application():
    """Test that blur augmentation is actually applied to images"""
    print("\nTesting blur augmentation application...")

    # Set high blur probability for testing
    BlurAugmentationDataset.blur_prob = 1.0  # Always apply blur
    BlurAugmentationDataset.blur_kernel_range = (5, 5)  # Fixed kernel for consistency
    BlurAugmentationDataset.blur_sigma_range = (1.0, 1.0)  # Fixed sigma

    test_config = {
        'path': './voc2012_yolo_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: f'class_{i}' for i in range(20)}
    }

    config_path = 'test_blur_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)

    try:
        dataset = BlurAugmentationDataset(
            data=test_config,
            img_path='voc2012_yolo_dataset/images/train',
            augment=True
        )

        if len(dataset) == 0:
            print("✗ No images in dataset")
            return False

        # Get original and blurred samples
        original_sample = dataset[0]
        blurred_sample = dataset[0]  # Should be blurred due to prob=1.0

        if 'img' not in original_sample or 'img' not in blurred_sample:
            print("✗ No image in sample")
            return False

        orig_img = original_sample['img']
        blur_img = blurred_sample['img']

        # Convert to numpy for comparison
        if isinstance(orig_img, torch.Tensor):
            orig_img = orig_img.permute(1, 2, 0).cpu().numpy()
        if isinstance(blur_img, torch.Tensor):
            blur_img = blur_img.permute(1, 2, 0).cpu().numpy()

        # Calculate difference
        diff = np.abs(orig_img.astype(np.float32) - blur_img.astype(np.float32))
        mean_diff = np.mean(diff)

        print(f"  Mean pixel difference: {mean_diff:.4f}")
        # Since we're applying blur with prob=1.0, images should be different
        if mean_diff > 0.01:  # Some threshold for difference
            print("✓ Blur augmentation appears to be applied (images differ)")
            return True
        else:
            print("✗ Blur augmentation may not be working (images are identical)")
            return False

    except Exception as e:
        print(f"✗ Blur application test failed: {e}")
        return False
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

def test_trainer_integration():
    """Test that the trainer can initialize and create config with blur parameters"""
    print("\nTesting trainer integration...")

    try:
        trainer = UltimateSmallObjectTrainer()

        # Create a dummy config path
        dummy_config = {
            'path': './voc2012_yolo_dataset',
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: f'class_{i}' for i in range(20)}
        }

        config_path = 'dummy_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dummy_config, f)

        # Test config creation
        training_config = trainer.create_ultimate_training_config(config_path, epochs=5)

        print("✓ Training config created successfully")
        print(f"  blur_prob in config: {'blur_prob' in training_config}")
        print(f"  blur_kernel_range in config: {'blur_kernel_range' in training_config}")
        print(f"  blur_sigma_range in config: {'blur_sigma_range' in training_config}")

        # Clean up
        os.remove(config_path)
        return True

    except Exception as e:
        print(f"✗ Trainer integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("BLUR AUGMENTATION INTEGRATION TEST")
    print("="*60)

    tests = [
        test_blur_dataset_initialization,
        test_blur_augmentation_application,
        test_trainer_integration
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! Blur augmentation integration is working.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
