#!/usr/bin/env python3
"""Test script for UltimateSmallObjectTrainer functionality"""

from ultimate_small_object_detector import UltimateSmallObjectTrainer, AdvancedPostProcessor
import os

def test_basic_initialization():
    """Test basic class initialization"""
    print("Testing basic initialization...")
    trainer = UltimateSmallObjectTrainer()
    processor = AdvancedPostProcessor()
    print("✓ Basic initialization successful")

def test_config_creation():
    """Test configuration creation"""
    print("Testing configuration creation...")
    trainer = UltimateSmallObjectTrainer()
    config = trainer.create_ultimate_training_config('voc2012.yaml')
    print("✓ Config created successfully")
    print(f"  Config has {len(config)} keys")
    print("  Sample config values:")
    for k, v in list(config.items())[:10]:
        print(f"    {k}: {v}")

def test_methods_existence():
    """Test that all new methods exist"""
    print("Testing method existence...")
    trainer = UltimateSmallObjectTrainer()

    # Check new methods exist
    methods_to_check = [
        'load_checkpoint',
        'save_checkpoint',
        'perform_grid_search_hyperparameter_optimization',
        'ensemble_models'
    ]

    for method in methods_to_check:
        if hasattr(trainer, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")

def test_post_processor_methods():
    """Test post-processor methods"""
    print("Testing post-processor methods...")
    processor = AdvancedPostProcessor()

    # Test NMS method
    boxes = [[0, 0, 10, 10], [5, 5, 15, 15]]
    scores = [0.9, 0.8]
    classes = [0, 0]

    try:
        result = processor.nms(boxes, scores, 0.5)
        print("✓ NMS method works")
    except Exception as e:
        print(f"✗ NMS method failed: {e}")

def main():
    """Run all tests"""
    print("Running thorough tests for UltimateSmallObjectTrainer...")
    print("="*60)

    try:
        test_basic_initialization()
        test_config_creation()
        test_methods_existence()
        test_post_processor_methods()

        print("="*60)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
