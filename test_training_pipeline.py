from combined_yolo_training import CombinedYOLOTrainer

if __name__ == '__main__':
    trainer = CombinedYOLOTrainer()

    # Test training pipeline with minimal epochs
    print("Testing enhanced training pipeline with minimal epochs...")
    try:
        results, model_path = trainer.train_with_enhanced_features(
            dataset_config='combined_dataset_20251023_061355.yaml',
            epochs=1,  # Just 1 epoch for testing
            resume_checkpoint=None,
            save_every=1,
            enable_data_quality=True,
            enable_class_balance=True,
            enable_arch_enhancements=True
        )
        if results and model_path:
            print("✓ Training pipeline completed successfully")
            print(f"Model saved to: {model_path}")
        else:
            print("✗ Training pipeline failed")
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
