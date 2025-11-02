"""
Data Quality Utilities for YOLO Training
Provides functions for data validation, duplicate detection, and quality checks
"""

import os
import cv2
import numpy as np
import imagehash
from PIL import Image
import logging
from pathlib import Path
import yaml
from collections import defaultdict
import shutil

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Data quality validation and cleaning utilities"""

    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.blur_threshold = 100  # Laplacian variance threshold for blur detection
        self.duplicate_threshold = 5  # Hamming distance threshold for duplicates

    def get_image_label_pairs(self):
        """Get all image-label pairs"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        pairs = []

        for img_path in self.images_dir.glob('*'):
            if img_path.suffix.lower() in image_extensions:
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))

        return pairs

    def detect_blurred_images(self, threshold=None):
        """Detect blurred images using Laplacian variance"""
        if threshold:
            self.blur_threshold = threshold

        blurred_images = []
        pairs = self.get_image_label_pairs()

        logger.info(f"Checking {len(pairs)} images for blur...")

        for img_path, _ in pairs:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Calculate Laplacian variance
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()

                if variance < self.blur_threshold:
                    blurred_images.append((str(img_path), variance))

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        logger.info(f"Found {len(blurred_images)} blurred images")
        return blurred_images

    def detect_duplicate_images(self, threshold=None):
        """Detect duplicate images using perceptual hashing"""
        if threshold:
            self.duplicate_threshold = threshold

        pairs = self.get_image_label_pairs()
        hashes = {}
        duplicates = []

        logger.info(f"Checking {len(pairs)} images for duplicates...")

        for img_path, _ in pairs:
            try:
                image = Image.open(img_path)
                # Calculate perceptual hash
                phash = imagehash.phash(image)

                # Check for duplicates
                is_duplicate = False
                duplicate_of = None

                for existing_path, existing_hash in hashes.items():
                    if phash - existing_hash <= self.duplicate_threshold:
                        is_duplicate = True
                        duplicate_of = existing_path
                        break

                if is_duplicate:
                    duplicates.append((str(img_path), str(duplicate_of)))
                else:
                    hashes[str(img_path)] = phash

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

        logger.info(f"Found {len(duplicates)} duplicate image pairs")
        return duplicates

    def validate_bounding_boxes(self):
        """Validate bounding box coordinates and formats"""
        invalid_boxes = []
        pairs = self.get_image_label_pairs()

        logger.info(f"Validating bounding boxes for {len(pairs)} images...")

        for img_path, label_path in pairs:
            try:
                # Read image to get dimensions
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                img_height, img_width = image.shape[:2]

                # Read label file
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_boxes.append((str(label_path), line_num, "Invalid format"))
                        continue

                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Check if coordinates are normalized (0-1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                0 <= width <= 1 and 0 <= height <= 1):
                            invalid_boxes.append((str(label_path), line_num, "Coordinates not normalized"))

                        # Check if bounding box is valid
                        if width <= 0 or height <= 0:
                            invalid_boxes.append((str(label_path), line_num, "Invalid box dimensions"))

                    except ValueError:
                        invalid_boxes.append((str(label_path), line_num, "Non-numeric values"))

            except Exception as e:
                logger.error(f"Error processing {label_path}: {e}")
                continue

        logger.info(f"Found {len(invalid_boxes)} invalid bounding boxes")
        return invalid_boxes

    def clean_dataset(self, remove_blurred=True, remove_duplicates=True, fix_boxes=True):
        """Clean dataset by removing problematic samples"""
        issues_found = {
            'blurred': [],
            'duplicates': [],
            'invalid_boxes': []
        }

        # Detect issues
        if remove_blurred:
            issues_found['blurred'] = self.detect_blurred_images()

        if remove_duplicates:
            issues_found['duplicates'] = self.detect_duplicate_images()

        if fix_boxes:
            issues_found['invalid_boxes'] = self.validate_bounding_boxes()

        # Remove problematic files
        removed_files = []

        # Remove blurred images
        for img_path, _ in issues_found['blurred']:
            try:
                os.remove(img_path)
                label_path = self.labels_dir / f"{Path(img_path).stem}.txt"
                if label_path.exists():
                    os.remove(str(label_path))
                removed_files.append(img_path)
                logger.info(f"Removed blurred image: {img_path}")
            except Exception as e:
                logger.error(f"Failed to remove {img_path}: {e}")

        # Remove duplicates (keep first occurrence)
        duplicate_files = set()
        for dup_path, orig_path in issues_found['duplicates']:
            duplicate_files.add(dup_path)

        for dup_path in duplicate_files:
            try:
                os.remove(dup_path)
                label_path = self.labels_dir / f"{Path(dup_path).stem}.txt"
                if label_path.exists():
                    os.remove(str(label_path))
                removed_files.append(dup_path)
                logger.info(f"Removed duplicate image: {dup_path}")
            except Exception as e:
                logger.error(f"Failed to remove {dup_path}: {e}")

        # For invalid boxes, we could either remove or attempt to fix
        # For now, just log them
        if issues_found['invalid_boxes']:
            logger.warning(f"Found {len(issues_found['invalid_boxes'])} invalid bounding boxes. Manual review recommended.")

        logger.info(f"Dataset cleaning completed. Removed {len(removed_files)} files.")
        return removed_files, issues_found

def analyze_class_distribution(labels_dir):
    """Analyze class distribution in dataset"""
    class_counts = defaultdict(int)
    total_labels = 0

    for label_file in Path(labels_dir).glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_labels += 1
        except Exception as e:
            logger.error(f"Error reading {label_file}: {e}")

    return dict(class_counts), total_labels

def balance_classes(images_dir, labels_dir, target_classes=None, oversample_factor=2):
    """Balance classes by oversampling rare classes"""
    if target_classes is None:
        target_classes = [5, 3, 17, 18]  # bus, boat, sofa, train

    # Analyze current distribution
    class_counts, _ = analyze_class_distribution(labels_dir)
    max_count = max(class_counts.values()) if class_counts else 0

    logger.info(f"Current class distribution: {class_counts}")
    logger.info(f"Max class count: {max_count}")

    # Oversample rare classes
    oversampled_files = []

    for class_id in target_classes:
        if class_id in class_counts:
            current_count = class_counts[class_id]
            if current_count < max_count:
                # Find images with this class
                class_images = []

                for label_file in Path(labels_dir).glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            content = f.read()
                            if f"{class_id} " in content:
                                img_name = label_file.stem
                                img_path = Path(images_dir) / f"{img_name}.jpg"
                                if not img_path.exists():
                                    for ext in ['.jpeg', '.png', '.bmp']:
                                        img_path = Path(images_dir) / f"{img_name}{ext}"
                                        if img_path.exists():
                                            break

                                if img_path.exists():
                                    class_images.append((img_path, label_file))
                    except Exception as e:
                        continue

                # Oversample
                needed = min(max_count * oversample_factor - current_count, len(class_images) * 2)
                oversample_count = max(0, needed)

                logger.info(f"Oversampling class {class_id}: {current_count} -> {current_count + oversample_count}")

                for i in range(oversample_count):
                    if i >= len(class_images):
                        break

                    src_img, src_label = class_images[i % len(class_images)]

                    # Create copies with suffix
                    new_img_name = f"{src_img.stem}_oversample_{i}{src_img.suffix}"
                    new_label_name = f"{src_label.stem}_oversample_{i}.txt"

                    new_img_path = src_img.parent / new_img_name
                    new_label_path = src_label.parent / new_label_name

                    try:
                        shutil.copy2(str(src_img), str(new_img_path))
                        shutil.copy2(str(src_label), str(new_label_path))
                        oversampled_files.append(str(new_img_path))
                    except Exception as e:
                        logger.error(f"Failed to oversample {src_img}: {e}")

    logger.info(f"Oversampling completed. Added {len(oversampled_files)} new samples.")
    return oversampled_files
