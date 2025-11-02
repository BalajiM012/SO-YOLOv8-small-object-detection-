"""
Test Script for Enhanced Model Save/Load Functionality
Tests the new save/load functions with checkpoint management and resume training
"""

import os
import torch
import json
from pathlib import Path
from ultralytics import YOLO
from model_utils import ModelManager, save_model, load_model, save_checkpoint, load_checkpoint, resume_training, list_saved_models, list_checkpoints

def test_basic_save_load():
    """Test basic save and load functionality"""
    print("="*60)
    print("TEST 1: Basic Save/Load Functionality")
    print("="*60)

    try:
        # Load a pretrained model
        model = YOLO('yolov8n.pt')  # Use nano model for faster testing

        # Test metadata
        metadata = {
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 0.01,
            'dataset': 'voc2012',
            'architecture': 'YOLOv8n'
        }

        # Save model
        print("Saving model...")
        saved_path = save_model(model, 'test_basic_model', metadata, 'pt')

        if saved_path:
            print(f"✓ Model saved successfully: {saved_path}")

            # Load model
            print("Loading model...")
            loaded_model = load_model(saved_path, validate=True)

            if loaded_model:
                print("✓ Model loaded and validated successfully")

                # Verify model integrity
                if hasattr(loaded_model, 'model') and hasattr(loaded_model, 'names'):
                    print("✓ Model structure verified")
                    return True
                else:
                    print("✗ Model structure invalid")
                    return False
            else:
                print("✗ Model loading failed")
                return False
        else:
            print("✗ Model saving failed")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_checkpoint_save_load():
    """Test checkpoint save and load functionality"""
    print("\n" + "="*60)
    print("TEST 2: Checkpoint Save/Load Functionality")
    print("="*60)

    try:
        # Load a model
        model = YOLO('yolov8n.pt')

        # Simulate training state
        epoch = 5
        optimizer_state = {'lr': 0.01, 'momentum': 0.9}  # Mock optimizer state
        scheduler_state = {'last_epoch': epoch}  # Mock scheduler state
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.6,
            'mAP': 0.75
        }

        # Save checkpoint
        print("Saving checkpoint...")
        checkpoint_path = save_checkpoint(
            model, epoch, optimizer_state, scheduler_state, metrics,
            'test_checkpoint_epoch_5'
        )

        if checkpoint_path:
            print(f"✓ Checkpoint saved: {checkpoint_path}")

            # Load checkpoint
            print("Loading checkpoint...")
            loaded_epoch, loaded_metrics = load_checkpoint(checkpoint_path, model)

            if loaded_epoch is not None:
                print(f"✓ Checkpoint loaded successfully")
                print(f"  - Epoch: {loaded_epoch}")
                print(f"  - Metrics: {loaded_metrics}")

                if loaded_epoch == epoch and loaded_metrics == metrics:
                    print("✓ Checkpoint data verified")
                    return True
                else:
                    print("✗ Checkpoint data mismatch")
                    return False
            else:
                print("✗ Checkpoint loading failed")
                return False
        else:
            print("✗ Checkpoint saving failed")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_resume_training():
    """Test resume training functionality"""
    print("\n" + "="*60)
    print("TEST 3: Resume Training Functionality")
    print("="*60)

    try:
        # First save a checkpoint
        model = YOLO('yolov8n.pt')
        epoch = 3
        metrics = {'loss': 0.4}

        checkpoint_path = save_checkpoint(model, epoch, metrics=metrics, checkpoint_name='resume_test_checkpoint')

        if checkpoint_path:
            print(f"✓ Checkpoint saved for resume test: {checkpoint_path}")

            # Now test resume using the same model path
            print("Testing resume training...")
            resumed_model, resumed_epoch, _, _ = resume_training(checkpoint_path, model_path='yolov8n.pt')

            if resumed_model and resumed_epoch is not None:
                print(f"✓ Resume training successful")
                print(f"  - Resumed from epoch: {resumed_epoch}")
                return True
            else:
                print("✗ Resume training failed")
                return False
        else:
            print("✗ Could not create checkpoint for resume test")
            return False

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_model_listing():
    """Test model and checkpoint listing functionality"""
    print("\n" + "="*60)
    print("TEST 4: Model and Checkpoint Listing")
    print("="*60)

    try:
        # List saved models
        print("Listing saved models...")
        saved_models = list_saved_models()

        if saved_models:
            print(f"✓ Found {len(saved_models)} saved models:")
            for model_info in saved_models:
                print(f"  - {model_info['name']}: {model_info['path']}")
        else:
            print("ℹ No saved models found")

        # List checkpoints
        print("\nListing checkpoints...")
        checkpoints = list_checkpoints()

        if checkpoints:
            print(f"✓ Found {len(checkpoints)} checkpoints:")
            for ckpt_info in checkpoints:
                print(f"  - {ckpt_info['name']}: Epoch {ckpt_info['metadata'].get('epoch', 'N/A')}")
        else:
            print("ℹ No checkpoints found")

        return True

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def test_error_handling():
    """Test error handling in save/load operations"""
    print("\n" + "="*60)
    print("TEST 5: Error Handling")
    print("="*60)

    try:
        # Test loading non-existent model
        print("Testing load of non-existent model...")
        result = load_model('non_existent_model.pt')
        if result is None:
            print("✓ Correctly handled non-existent model")
        else:
            print("✗ Should have failed for non-existent model")
            return False

        # Test saving with invalid model
        print("Testing save with invalid model...")
        result = save_model(None, 'invalid_test')
        if result is None:
            print("✓ Correctly handled invalid model")
        else:
            print("✗ Should have failed for invalid model")
            return False

        return True

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Enhanced Model Save/Load Testing Suite")
    print("="*60)

    tests = [
        test_basic_save_load,
        test_checkpoint_save_load,
        test_resume_training,
        test_model_listing,
        test_error_handling
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED with exception: {e}")

    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()

    if success:
        print("\nEnhanced save/load functionality is working correctly!")
        print("You can now use the improved model management features in your training scripts.")
    else:
        print("\nSome issues were found. Please review the test output and fix any problems.")
