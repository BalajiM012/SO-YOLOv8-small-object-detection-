#!/usr/bin/env python3
"""
Test script for combined_yolo_training.py
"""

from combined_yolo_training import DatasetManager, CombinedYOLOTrainer

def test_dataset_manager():
    """Test DatasetManager functionality"""
    print("Testing DatasetManager...")

    dm = DatasetManager()
    print(f"Number of datasets: {len(dm.datasets)}")

    # Test listing datasets
    dm.list_datasets()

    # Test getting valid datasets
    valid_datasets = []
    for dataset_id in dm.datasets.keys():
        dataset_config = dm.get_dataset_config(dataset_id)
        if dataset_config and dm.validate_dataset(dataset_config):
            valid_datasets.append(dataset_config)

    print(f"\nValid datasets found: {len(valid_datasets)}")
    for ds in valid_datasets:
        print(f"- {ds['name']}")

    return valid_datasets

def test_trainer_initialization():
    """Test CombinedYOLOTrainer initialization"""
    print("\nTesting CombinedYOLOTrainer initialization...")

    try:
        trainer = CombinedYOLOTrainer()
        print("✓ Trainer initialized successfully")
        return trainer
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        return None

def test_combined_dataset_creation(valid_datasets):
    """Test combined dataset configuration creation"""
    print("\nTesting combined dataset creation...")

    if not valid_datasets:
        print("✗ No valid datasets to combine")
        return None

    try:
        from combined_yolo_training import create_combined_dataset_config
        config_path = create_combined_dataset_config(valid_datasets)
        if config_path:
            print(f"✓ Combined dataset config created: {config_path}")
            return config_path
        else:
            print("✗ Failed to create combined dataset config")
            return None
    except Exception as e:
        print(f"✗ Combined dataset creation failed: {e}")
        return None

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING COMBINED YOLO TRAINING SCRIPT")
    print("="*60)

    # Test DatasetManager
    valid_datasets = test_dataset_manager()

    # Test Trainer
    trainer = test_trainer_initialization()

    # Test combined dataset creation
    combined_config = test_combined_dataset_creation(valid_datasets)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"DatasetManager: {'✓' if len(valid_datasets) > 0 else '✗'}")
    print(f"Trainer Init: {'✓' if trainer else '✗'}")
    print(f"Combined Config: {'✓' if combined_config else '✗'}")

    if len(valid_datasets) > 0 and trainer and combined_config:
        print("\n✓ All tests passed! Ready for training.")
    else:
        print("\n✗ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()
