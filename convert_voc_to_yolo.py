import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# PASCAL VOC class names to YOLO class indices
VOC_CLASSES = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

def convert_bbox_to_yolo(size, box):
    """Convert PASCAL VOC bbox to YOLO format"""
    dw = 1.0 / size[0]  # 1/width
    dh = 1.0 / size[1]  # 1/height
    
    # Center coordinates
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    
    # Width and height
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return x, y, w, h

def convert_voc_to_yolo(xml_path, output_dir):
    """Convert a single VOC XML file to YOLO format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Create output file path
    filename = root.find('filename').text
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            # Get class name and convert to index
            class_name_elem = obj.find('n')
            if class_name_elem is None:
                continue  # Skip objects without class name
            class_name = class_name_elem.text
            if class_name in VOC_CLASSES:
                class_id = VOC_CLASSES[class_name]
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                xmax = float(bbox.find('xmax').text)
                ymin = float(bbox.find('ymin').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x, y, w, h = convert_bbox_to_yolo((width, height), (xmin, xmax, ymin, ymax))
                
                # Write to file
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def main():
    # Paths
    voc_root = "archive/VOC2012_train_val/VOC2012_train_val"
    output_root = "dataset"
    
    # Create output directories
    os.makedirs(f"{output_root}/images/train", exist_ok=True)
    os.makedirs(f"{output_root}/images/val", exist_ok=True)
    os.makedirs(f"{output_root}/labels/train", exist_ok=True)
    os.makedirs(f"{output_root}/labels/val", exist_ok=True)
    
    # Read train and validation splits
    with open(f"{voc_root}/ImageSets/Main/train.txt", 'r') as f:
        train_images = [line.strip() for line in f.readlines()]
    
    with open(f"{voc_root}/ImageSets/Main/val.txt", 'r') as f:
        val_images = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Process training data
    print("Processing training data...")
    for img_name in train_images:
        # Copy image
        src_img = f"{voc_root}/JPEGImages/{img_name}.jpg"
        dst_img = f"{output_root}/images/train/{img_name}.jpg"
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Convert annotations
        src_xml = f"{voc_root}/Annotations/{img_name}.xml"
        if os.path.exists(src_xml):
            convert_voc_to_yolo(src_xml, f"{output_root}/labels/train")
    
    # Process validation data
    print("Processing validation data...")
    for img_name in val_images:
        # Copy image
        src_img = f"{voc_root}/JPEGImages/{img_name}.jpg"
        dst_img = f"{output_root}/images/val/{img_name}.jpg"
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Convert annotations
        src_xml = f"{voc_root}/Annotations/{img_name}.xml"
        if os.path.exists(src_xml):
            convert_voc_to_yolo(src_xml, f"{output_root}/labels/val")
    
    print("Dataset conversion completed!")
    print(f"Training images: {len([f for f in os.listdir(f'{output_root}/images/train') if f.endswith('.jpg')])}")
    print(f"Training labels: {len([f for f in os.listdir(f'{output_root}/labels/train') if f.endswith('.txt')])}")
    print(f"Validation images: {len([f for f in os.listdir(f'{output_root}/images/val') if f.endswith('.jpg')])}")
    print(f"Validation labels: {len([f for f in os.listdir(f'{output_root}/labels/val') if f.endswith('.txt')])}")

if __name__ == "__main__":
    main()
