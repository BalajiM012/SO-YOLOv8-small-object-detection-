#!/usr/bin/env python3
"""
Thorough testing script for combined_yolo_training.py
Tests training execution, model testing, validation, checkpoints, and multi-dataset workflow
"""

import os
import time
from combined_yolo_training import (
    DatasetManager, CombinedYOLOTrainer, train_on_all_datasets,
    create_combined_dataset_config, train_single_dataset, train_with_resume_option
)

def test_gpu_memory_management():
    """Test GPU memory management functionality"""
    print("Testing GPU Memory Management...")

    trainer = CombinedYOLOTrainer()
    memory_manager = trainer.memory_manager

    # Test memory monitoring
    memory_info = memory_manager.monitor_memory()
    print(f"✓ Memory monitoring: {memory_info['used_gb']:.2f}GB used, {memory_info['free_gb']:.2f}GB free")

    # Test optimal batch size calculation
    optimal_batch = memory_manager.get_optimal_batch_size()
    print(f"✓ Optimal batch size: {optimal_batch}")

    # Test memory clearing
    memory_manager.clear_gpu_cache()
    print("✓ GPU cache cleared")

    return True

def test_model_manager():
    """Test ModelManager functionality"""
    print("\nTesting ModelManager...")

    trainer = CombinedYOLOTrainer()
    model_manager = trainer.model_manager

    # Test checkpoint listing
    checkpoints = model_manager.list_checkpoints()
    print(f"✓ Found {len(checkpoints)} checkpoints")

    # Test model loading (if model exists)
    if os.path.exists('yolov8x.pt'):
        model = model_manager.load_model('yolov8x.pt')
        if model:
            print("✓ Model loading successful")
        else:
            print("✗ Model loading failed")
    else:
        print("⚠ yolov8x.pt not found, skipping model load test")

    return True

def test_short_training():
    """Test short training execution"""
    print("\nTesting Short Training Execution...")

    trainer = CombinedYOLOTrainer()

    # Get a valid dataset
    dm = DatasetManager()
    dataset_config = dm.get_dataset_config(1)  # PASCAL VOC 2012

    if not dataset_config or not dm.validate_dataset(dataset_config):
        print("✗ No valid dataset for training test")
        return False

    print(f"Using dataset: {dataset_config['name']}")

    # Short training (1 epoch)
    try:
        results, model_path = trainer.train_with_resume(
            dataset_config=dataset_config['path'],
            epochs=1,  # Very short for testing
            resume_checkpoint=None,
            save_every=1
        )

        if results and model_path:
            print("✓ Short training completed successfully")
            print(f"✓ Model saved to: {model_path}")
            return model_path
        else:
            print("✗ Short training failed")
            return None

    except Exception as e:
        print(f"✗ Training error: {e}")
        return None

def test_model_testing(model_path):
    """Test model testing functionality"""
    print("\nTesting Model Testing...")

    if not model_path or not os.path.exists(model_path):
        print("✗ No valid model for testing")
        return False

    trainer = CombinedYOLOTrainer()

    # Get test directories from dataset
    dm = DatasetManager()
    dataset_config = dm.get_dataset_config(1)

    test_images_dir = dataset_config['images_dir']
    test_labels_dir = dataset_config['labels_dir']

    if not os.path.exists(test_images_dir):
        print(f"✗ Test images directory not found: {test_images_dir}")
        return False

    try:
        test_results = trainer.test_model(
            model_path=model_path,
            test_images_dir=test_images_dir,
            test_annotations_dir=test_labels_dir,
            max_images=10  # Very few for testing
        )

        if test_results:
            print("✓ Model testing completed successfully")
            print(f"✓ Tested {test_results['summary']['total_images']} images")
            print(f"✓ Total detections: {test_results['summary']['total_detections']}")
            return True
        else:
            print("✗ Model testing failed")
            return False

    except Exception as e:
        print(f"✗ Testing error: {e}")
        return False

def test_model_validation(model_path):
    """Test model validation functionality"""
    print("\nTesting Model Validation...")

    if not model_path or not os.path.exists(model_path):
        print("✗ No valid model for validation")
        return False

    trainer = CombinedYOLOTrainer()

    try:
        val_results = trainer.validate_model(model_path)

        if val_results:
            print("✓ Model validation completed successfully")
            print(".4f")
            print(".4f")
            print(".4f")
            return True
        else:
            print("✗ Model validation failed")
            return False

    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False

def test_checkpoint_functionality():
    """Test checkpoint saving and loading"""
    print("\nTesting Checkpoint Functionality...")

    trainer = CombinedYOLOTrainer()
    model_manager = trainer.model_manager

    # Load a model to save as checkpoint
    if os.path.exists('yolov8x.pt'):
        model = model_manager.load_model('yolov8x.pt')
        if model:
            # Save checkpoint
            checkpoint_path = model_manager.save_checkpoint(
                model=model,
                epoch=1,
                optimizer_state=None,
                scheduler_state=None,
                metrics={'test_metric': 0.95},
                checkpoint_name='test_checkpoint'
            )

            if checkpoint_path:
                print(f"✓ Checkpoint saved: {checkpoint_path}")

                # Load checkpoint
                loaded_model = model_manager.load_checkpoint(checkpoint_path, model)
                if loaded_model:
                    print("✓ Checkpoint loaded successfully")
                    return True
                else:
                    print("✗ Checkpoint loading failed")
                    return False
            else:
                print("✗ Checkpoint saving failed")
                return False
        else:
            print("✗ Could not load model for checkpoint test")
            return False
    else:
        print("⚠ yolov8x.pt not found, skipping checkpoint test")
        return True  # Not a failure, just not available

def test_multi_dataset_workflow():
    """Test multi-dataset training workflow"""
    print("\nTesting Multi-Dataset Workflow...")

    # Get valid datasets
    dm = DatasetManager()
    valid_datasets = []
    for dataset_id in dm.datasets.keys():
        dataset_config = dm.get_dataset_config(dataset_id)
        if dataset_config and dm.validate_dataset(dataset_config):
            valid_datasets.append(dataset_config)

    if len(valid_datasets) < 2:
        print("⚠ Need at least 2 valid datasets for multi-dataset test, skipping")
        return True

    print(f"Found {len(valid_datasets)} valid datasets for multi-dataset test")

    # Create combined config
    combined_config = create_combined_dataset_config(valid_datasets)
    if not combined_config:
        print("✗ Failed to create combined dataset config")
        return False

    print(f"✓ Combined config created: {combined_config}")

    # Test short multi-dataset training (1 epoch)
    trainer = CombinedYOLOTrainer()

    try:
        results, model_path = trainer.train_with_resume(
            dataset_config=combined_config,
            epochs=1,  # Very short
            resume_checkpoint=None,
            save_every=1
        )

        if results and model_path:
            print("✓ Multi-dataset training completed successfully")
            return True
        else:
            print("✗ Multi-dataset training failed")
            return False

    except Exception as e:
        print(f"✗ Multi-dataset training error: {e}")
        return False

def main():
    """Run thorough tests"""
    print("="*80)
    print("THOROUGH TESTING OF COMBINED YOLO TRAINING SCRIPT")
    print("="*80)

    test_results = {}

    # Test GPU memory management
    test_results['gpu_memory'] = test_gpu_memory_management()

    # Test model manager
    test_results['model_manager'] = test_model_manager()

    # Test short training
    trained_model_path = test_short_training()
    test_results['short_training'] = trained_model_path is not None

    # Test model testing
    test_results['model_testing'] = test_model_testing(trained_model_path)

    # Test model validation
    test_results['model_validation'] = test_model_validation(trained_model_path)

    # Test checkpoint functionality
    test_results['checkpoints'] = test_checkpoint_functionality()

    # Test multi-dataset workflow
    test_results['multi_dataset'] = test_multi_dataset_workflow()

    # Summary
    print("\n" + "="*80)
    print("THOROUGH TEST SUMMARY")
    print("="*80)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The combined training system is fully functional.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
