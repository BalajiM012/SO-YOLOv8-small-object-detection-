#!/usr/bin/env python3
"""
Multi-Dataset Training Script for Small Object Detection
Combines multiple datasets for enhanced training
"""

import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
from enhanced_small_object_yolo import MULTI_DATASET_CONFIGS, add_se_and_c2f_to_yolo
from ultimate_small_object_detector import UltimateSmallObjectTrainer

def create_combined_yaml(datasets_to_combine, output_path='combined_datasets.yaml'):
    """Create a combined YAML configuration for multiple datasets"""

    combined = {
        'path': os.getcwd(),
        'train': [],
        'val': [],
        'names': [],
        'nc': 0
    }

    all_classes = set()

    for dataset_name in datasets_to_combine:
        if dataset_name in MULTI_DATASET_CONFIGS:
            config = MULTI_DATASET_CONFIGS[dataset_name]
            yaml_file = config['yaml']

            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)

                # Add train paths
                if 'train' in data:
                    train_path = data['train']
                    if isinstance(train_path, str):
                        train_path = [train_path]
                    combined['train'].extend(train_path)

                # Add val paths
                if 'val' in data:
                    val_path = data['val']
                    if isinstance(val_path, str):
                        val_path = [val_path]
                    combined['val'].extend(val_path)

                # Add class names
                if 'names' in data:
                    for i, name in data['names'].items():
                        if name not in all_classes:
                            all_classes.add(name)
                            combined['names'].append(name)

    combined['nc'] = len(combined['names'])

    with open(output_path, 'w') as f:
        yaml.dump(combined, f, default_flow_style=False)

    print(f"Combined YAML created with {len(combined['train'])} train paths and {len(combined['val'])} val paths")
    print(f"Total classes: {combined['nc']}")

    return output_path

def train_on_multiple_datasets(datasets_to_use, epochs=100, output_dir='models/multi_dataset'):
    """Train on multiple datasets combined"""

    print(f"Training on datasets: {datasets_to_use}")

    # Create combined dataset
    combined_yaml = create_combined_yaml(datasets_to_use)

    # Initialize trainer
    trainer = UltimateSmallObjectTrainer()

    # Train the model
    model, model_path = trainer.train_ultimate_model(combined_yaml, epochs=epochs)

    # Move to output directory
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f'multi_dataset_model_{len(datasets_to_use)}_datasets.pt')
    os.rename(model_path, final_path)

    print(f"Multi-dataset training completed. Model saved to: {final_path}")
    return final_path

def main():
    """Main function for multi-dataset training"""

    # Available datasets
    available_datasets = list(MULTI_DATASET_CONFIGS.keys())
    print(f"Available datasets: {available_datasets}")

    # Select datasets to combine (you can modify this list)
    datasets_to_use = ['voc2012_yolo_dataset', 'VisDrone_Dataset', 'TinyPerson']

    # Check if datasets exist
    valid_datasets = []
    for dataset in datasets_to_use:
        if dataset in MULTI_DATASET_CONFIGS:
            yaml_file = MULTI_DATASET_CONFIGS[dataset]['yaml']
            if os.path.exists(yaml_file):
                valid_datasets.append(dataset)
            else:
                print(f"Warning: YAML file for {dataset} not found: {yaml_file}")
        else:
            print(f"Warning: Dataset {dataset} not configured")

    if not valid_datasets:
        print("No valid datasets found. Please check your dataset configurations.")
        return

    print(f"Using datasets: {valid_datasets}")

    # Train on combined datasets
    model_path = train_on_multiple_datasets(valid_datasets, epochs=150)

    print(f"Multi-dataset training completed successfully!")
    print(f"Final model: {model_path}")

if __name__ == "__main__":
    main()
