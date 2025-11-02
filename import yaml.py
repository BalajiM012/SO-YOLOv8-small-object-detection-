import yaml
from pathlib import Path
import os

def count_images(img_dir):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return 0
    # Count jpg, png, jpeg files
    return sum(1 for ext in ['*.jpg', '*.png', '*.jpeg'] for _ in img_dir.glob(ext))

def get_counts_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    base_path = Path(yaml_path).parent / data.get('path', '')
    train_dir = base_path / data['train']
    val_dir = base_path / data['val']
    test_dir = base_path / data.get('test', '') if 'test' in data else None

    train_count = count_images(train_dir)
    val_count = count_images(val_dir)
    test_count = count_images(test_dir) if test_dir else 0

    return train_count, val_count, test_count

configs = {
    'PASCAL_VOC': 'voc2012.yaml',
    'TinyPerson': '../TinyPerson/TinyPerson -YOLO format-.v1i.yolov8/data.yaml',
    'VisDrone': '../VisDrone Dataset/archive (1)/VisDrone_Dataset/visdrone.yaml',
}

total_train, total_val, total_test = 0, 0, 0

for name, yaml_path in configs.items():
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"{name} config not found: {yaml_path}")
        continue
    train, val, test = get_counts_from_yaml(yaml_path)
    print(f"{name}: Train={train}, Val={val}, Test={test}")
    total_train += train
    total_val += val
    total_test += test

print(f"\nCombined totals:")
print(f"Train: {total_train}")
print(f"Val: {total_val}")
print(f"Test: {total_test}")