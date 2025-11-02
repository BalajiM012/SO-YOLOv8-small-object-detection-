from combined_yolo_training import CombinedYOLOTrainer

trainer = CombinedYOLOTrainer()

# Test loading enhanced model
print("Testing enhanced model loading...")
try:
    model = trainer.load_enhanced_model()
    print("✓ Enhanced model loaded successfully")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"✗ Failed to load enhanced model: {e}")
