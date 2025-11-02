"""
Custom Dataset Class for YOLO Training with Blur Augmentation
Implements Gaussian blur augmentation through the dataset pipeline
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, RandomPerspective, MixUp, Mosaic, CopyPaste, Albumentations


class BlurAugmentationDataset(YOLODataset):
    """
    Custom YOLO dataset with integrated blur augmentation
    Extends Ultralytics YOLODataset to add Gaussian blur during training
    """

    def __init__(self, *args, blur_prob=None, blur_kernel_range=None, blur_sigma_range=None, **kwargs):
        """
        Initialize the blur augmentation dataset

        Args:
            blur_prob (float): Probability of applying blur augmentation (0.0 to 1.0)
            blur_kernel_range (tuple): Range of kernel sizes for Gaussian blur (min, max)
            blur_sigma_range (tuple): Range of sigma values for Gaussian blur (min, max)
        """
        # Use class attributes if parameters not provided
        if blur_prob is None:
            blur_prob = getattr(self.__class__, 'blur_prob', 0.3)
        if blur_kernel_range is None:
            blur_kernel_range = getattr(self.__class__, 'blur_kernel_range', (3, 7))
        if blur_sigma_range is None:
            blur_sigma_range = getattr(self.__class__, 'blur_sigma_range', (0.1, 2.0))

        super().__init__(*args, **kwargs)

        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range

        print(f"BlurAugmentationDataset initialized with blur_prob={blur_prob}")

    def __getitem__(self, index):
        """
        Get a single data sample with optional blur augmentation
        """
        # Get the original sample from parent class
        sample = super().__getitem__(index)

        # Apply blur augmentation during training
        if self.augment and random.random() < self.blur_prob:
            sample = self.apply_blur_augmentation(sample)

        return sample

    def apply_blur_augmentation(self, sample):
        """
        Apply Gaussian blur augmentation to the image

        Args:
            sample (dict): Sample containing 'img' and other data

        Returns:
            dict: Sample with blurred image
        """
        img = sample['img']

        # Convert to numpy if tensor
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img.copy()

        # Ensure image is in uint8 format for OpenCV
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)

        # Random kernel size (must be odd)
        kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Random sigma
        sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)

        # Convert back to original format
        if isinstance(img, torch.Tensor):
            blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).float() / 255.0
            sample['img'] = blurred_tensor
        else:
            sample['img'] = blurred

        return sample

class AdvancedBlurAugmentationDataset(BlurAugmentationDataset):
    """
    Advanced blur augmentation with multiple blur types and adaptive parameters
    """

    def __init__(self, *args, blur_types=['gaussian', 'median', 'bilateral'], **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_types = blur_types

    def apply_blur_augmentation(self, sample):
        """
        Apply advanced blur augmentation with multiple blur types
        """
        img = sample['img']

        # Convert to numpy if tensor
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img.copy()

        # Ensure image is in uint8 format for OpenCV
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)

        # Randomly select blur type
        blur_type = random.choice(self.blur_types)

        if blur_type == 'gaussian':
            # Gaussian blur
            kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)

        elif blur_type == 'median':
            # Median blur
            kernel_size = random.choice(range(3, 8, 2))  # Smaller kernel for median
            blurred = cv2.medianBlur(img_np, kernel_size)

        elif blur_type == 'bilateral':
            # Bilateral filter (edge-preserving blur)
            d = random.randint(5, 15)
            sigma_color = random.uniform(10, 100)
            sigma_space = random.uniform(10, 100)
            blurred = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)

        else:
            # Fallback to Gaussian
            blurred = cv2.GaussianBlur(img_np, (5, 5), 1.0)

        # Convert back to original format
        if isinstance(img, torch.Tensor):
            blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).float() / 255.0
            sample['img'] = blurred_tensor
        else:
            sample['img'] = blurred

        return sample

def create_blur_augmented_dataset(data_config, blur_prob=0.3, advanced=False, **kwargs):
    """
    Factory function to create blur-augmented dataset

    Args:
        data_config (str): Path to data YAML config
        blur_prob (float): Probability of applying blur
        advanced (bool): Use advanced blur augmentation
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset: Configured dataset with blur augmentation
    """
    if advanced:
        dataset_class = AdvancedBlurAugmentationDataset
    else:
        dataset_class = BlurAugmentationDataset

    # Load data config
    with open(data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)

    # Create dataset with blur augmentation
    dataset = dataset_class(
        data_config,
        blur_prob=blur_prob,
        **kwargs
    )

    return dataset

# Example usage and testing functions
def test_blur_augmentation():
    """Test function for blur augmentation"""
    import matplotlib.pyplot as plt

    # Create a simple test dataset
    test_config = {
        'path': './voc2012_yolo_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: f'class_{i}' for i in range(20)}
    }

    # Save test config
    with open('test_config.yaml', 'w') as f:
        yaml.dump(test_config, f)

    # Create dataset with blur augmentation
    dataset = BlurAugmentationDataset(
        'test_config.yaml',
        img_path=test_config['train'],
        augment=True,
        blur_prob=1.0  # Always apply for testing
    )

    # Test a few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        img = sample['img']

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Blur Augmented Image {i}')
        plt.axis('off')
        plt.show()

    print("Blur augmentation test completed!")

if __name__ == "__main__":
    test_blur_augmentation()
