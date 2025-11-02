import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def convert_voc_to_yolo(voc_dataset_path, output_path):
    """
    Convert PASCAL VOC 2012 dataset to YOLO format
    """
    # Create output directories
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/labels/train", exist_ok=True)
    os.makedirs(f"{output_path}/labels/val", exist_ok=True)

    # VOC class names (20 classes)
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Create class mapping
    class_to_id = {cls: idx for idx, cls in enumerate(voc_classes)}

    def convert_annotation(xml_file, img_width, img_height):
        """Convert VOC XML annotation to YOLO format"""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        yolo_annotations = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_to_id:
                continue

            class_id = class_to_id[class_name]

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_annotations

    def process_dataset(split_name, image_list, images_dir, annotations_dir, output_images_dir, output_labels_dir):
        """Process train or val dataset split"""
        print(f"Processing {split_name} dataset...")

        processed_count = 0
        for img_name in image_list:
            img_file = f"{img_name}.jpg"
            img_path = os.path.join(images_dir, img_file)

            if not os.path.exists(img_path):
                continue

            # Find corresponding XML file
            xml_file = os.path.join(annotations_dir, f"{img_name}.xml")
            if not os.path.exists(xml_file):
                print(f"Warning: No annotation found for {img_file}")
                continue

            # Get image dimensions
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except ImportError:
                print("PIL not available, using default dimensions")
                img_width, img_height = 640, 640

            # Convert annotation
            yolo_annotations = convert_annotation(xml_file, img_width, img_height)

            if yolo_annotations:
                # Copy image
                shutil.copy2(img_path, output_images_dir)

                # Write YOLO annotation
                label_file = os.path.join(output_labels_dir, f"{img_name}.txt")
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))

                processed_count += 1

        print(f"Processed {processed_count} {split_name} images")
        return processed_count

    # Load train and val lists
    imagesets_dir = os.path.join(voc_dataset_path, "ImageSets", "Main")
    train_txt = os.path.join(imagesets_dir, "train.txt")
    val_txt = os.path.join(imagesets_dir, "val.txt")

    with open(train_txt, 'r') as f:
        train_images = [line.strip() for line in f.readlines()]

    with open(val_txt, 'r') as f:
        val_images = [line.strip() for line in f.readlines()]

    images_dir = os.path.join(voc_dataset_path, "JPEGImages")
    annotations_dir = os.path.join(voc_dataset_path, "Annotations")

    # Process training dataset
    train_count = process_dataset(
        "train",
        train_images,
        images_dir,
        annotations_dir,
        f"{output_path}/images/train",
        f"{output_path}/labels/train"
    )

    # Process validation dataset
    val_count = process_dataset(
        "val",
        val_images,
        images_dir,
        annotations_dir,
        f"{output_path}/images/val",
        f"{output_path}/labels/val"
    )

    print(f"\nConversion completed!")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Output directory: {output_path}")

if __name__ == "__main__":
    # Dataset paths
    voc_dataset_path = "./VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
    output_path = "./voc2012_yolo_dataset"

    print("Converting PASCAL VOC 2012 to YOLO format...")
    print(f"Source: {voc_dataset_path}")
    print(f"Output: {output_path}")

    convert_voc_to_yolo(voc_dataset_path, output_path)
