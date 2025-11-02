#!/usr/bin/env python3
"""
Load Checkpoints and Train to Target Metrics Script
Loads all 100 checkpoints and trains until achieving target metrics:
- Precision: 1.00
- Recall: 0.89
- mAP@0.5: 0.79
- mAP@0.5–0.95: 0.62–0.64
- Best F1 Score: ≈0.76
- Small object handling: Optimized with SE block
"""

import os
import torch
import json
from pathlib import Path
from ultralytics import YOLO
from gpu_optimized_training import OptimizedTrainer
from enhanced_small_object_yolo import add_se_to_yolo
from model_utils import load_checkpoint, save_checkpoint, resume_training

class MetricsTrainer:
    """Trainer that loads checkpoints and trains to target metrics"""

    def __init__(self, checkpoint_dir, dataset_config='voc2012.yaml'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_config = dataset_config
        self.target_metrics = {
            'precision': 1.00,
            'recall': 0.89,
            'map50': 0.79,
            'map': 0.62,  # Lower bound of 0.62–0.64
            'f1': 0.76
        }

        # Initialize GPU trainer
        device_id = 0 if torch.cuda.is_available() else 'cpu'
        self.trainer = OptimizedTrainer(model_path='yolov8x.pt', device_id=device_id)

    def load_all_checkpoints(self):
        """Load all checkpoint files"""
        checkpoints = []
        if self.checkpoint_dir.exists():
            for ckpt_file in self.checkpoint_dir.glob('*.pt'):
                checkpoints.append(str(ckpt_file))
        checkpoints.sort()
        return checkpoints

    def validate_checkpoint_metrics(self, model):
        """Validate model and check if metrics meet targets"""
        print("\n" + "="*60)
        print("VALIDATING CHECKPOINT METRICS")
        print("="*60)

        val_results = self.trainer.validate_model(model)

        if val_results is None:
            return False, {}

        metrics = {
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
            'map50': val_results.box.map50,
            'map': val_results.box.map,
            'f1': 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr) if (val_results.box.mp + val_results.box.mr) > 0 else 0
        }

        print(f"Current Metrics:")
        print(f"  Precision: {metrics['precision']:.4f} (Target: {self.target_metrics['precision']})")
        print(f"  Recall: {metrics['recall']:.4f} (Target: {self.target_metrics['recall']})")
        print(f"  mAP@0.5: {metrics['map50']:.4f} (Target: {self.target_metrics['map50']})")
        print(f"  mAP@0.5–0.95: {metrics['map']:.4f} (Target: {self.target_metrics['map']})")
        print(f"  F1 Score: {metrics['f1']:.4f} (Target: {self.target_metrics['f1']})")

        # Check if all metrics meet or exceed targets
        metrics_achieved = (
            metrics['precision'] >= self.target_metrics['precision'] and
            metrics['recall'] >= self.target_metrics['recall'] and
            metrics['map50'] >= self.target_metrics['map50'] and
            metrics['map'] >= self.target_metrics['map'] and
            metrics['f1'] >= self.target_metrics['f1']
        )

        if metrics_achieved:
            print("✓ All target metrics achieved!")
        else:
            print("✗ Metrics not yet achieved, continuing training...")

        return metrics_achieved, metrics

    def train_to_target_metrics(self, max_additional_epochs=50):
        """Load checkpoints and train until target metrics are achieved"""

        print("Loading Checkpoints and Training to Target Metrics")
        print("="*70)

        # Load all checkpoints
        checkpoints = self.load_all_checkpoints()
        print(f"Found {len(checkpoints)} checkpoints")

        if not checkpoints:
            print("No checkpoints found. Starting fresh training.")
            model = YOLO('yolov8x.pt')
            model = add_se_to_yolo(model)  # Add SE blocks for small object handling
            start_epoch = 0
        else:
            # Load the latest checkpoint
            latest_checkpoint = checkpoints[-1]
            print(f"Loading latest checkpoint: {latest_checkpoint}")

            model, start_epoch, _, _ = resume_training(latest_checkpoint, 'yolov8x.pt')
            if model is None:
                print("Failed to load checkpoint. Starting fresh.")
                model = YOLO('yolov8x.pt')
                start_epoch = 0

            model = add_se_to_yolo(model)  # Ensure SE blocks are added

        # Validate initial metrics
        metrics_achieved, initial_metrics = self.validate_checkpoint_metrics(model)
        if metrics_achieved:
            print("Target metrics already achieved!")
            return model, initial_metrics

        # Continue training until metrics are achieved or max epochs reached
        current_epoch = start_epoch
        additional_epochs = 0

        while not metrics_achieved and additional_epochs < max_additional_epochs:
            print(f"\n--- Training additional epochs (current: {current_epoch}, additional: {additional_epochs}) ---")

            # Train for 10 more epochs
            results = self.trainer.train_with_memory_monitoring(
                dataset_config=self.dataset_config,
                epochs=current_epoch + 10,
                resume_checkpoint=latest_checkpoint if checkpoints else None
            )

            if results is None:
                print("Training failed. Stopping.")
                break

            current_epoch += 10
            additional_epochs += 10

            # Update latest checkpoint
            if checkpoints:
                latest_checkpoint = checkpoints[-1]  # Assuming new checkpoints are saved

            # Validate metrics
            metrics_achieved, current_metrics = self.validate_checkpoint_metrics(results.save_dir + '/weights/best.pt')

            if metrics_achieved:
                print(f"✓ Target metrics achieved after {current_epoch} total epochs!")
                break

        if not metrics_achieved:
            print(f"✗ Target metrics not achieved after {current_epoch} epochs. Best metrics:")
            print(json.dumps(current_metrics, indent=2))

        return model, current_metrics

def main():
    """Main function"""
    checkpoint_dir = r"D:\MINI Project Phase 1\Object Detection DL\Object DL DATASETS\SO - YOLO\Pascal VOC 2012 UK\runs\gpu_resume\gpu_resume_training\weights"

    trainer = MetricsTrainer(checkpoint_dir)
    final_model, final_metrics = trainer.train_to_target_metrics()

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Final Metrics: {json.dumps(final_metrics, indent=2)}")

    if final_model:
        print("✓ Training completed!")
    else:
        print("✗ Training failed.")

if __name__ == "__main__":
    main()
