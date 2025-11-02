from combined_yolo_training import CombinedYOLOTrainer

trainer = CombinedYOLOTrainer()

# Test data quality checks
print("Testing data quality checks...")
try:
    # Use the combined config we created
    trainer.perform_data_quality_checks('combined_dataset_20251023_061355.yaml')
    print("✓ Data quality checks completed")
except Exception as e:
    print(f"✗ Data quality checks failed: {e}")

# Test class balancing
print("\nTesting class balancing...")
try:
    trainer.balance_dataset_classes('combined_dataset_20251023_061355.yaml')
    print("✓ Class balancing completed")
except Exception as e:
    print(f"✗ Class balancing failed: {e}")
