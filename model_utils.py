"""
Enhanced Model Save/Load Utilities for YOLO Training and Testing
Features:
- Explicit save_model() and load_model() functions with error handling
- Checkpoint management during training
- Resume training functionality
- Model validation after loading
- Metadata saving and loading
"""

import os
import torch
import json
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Enhanced model management with save/load capabilities"""

    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.base_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.pretrained_dir = self.base_dir / 'pretrained'
        self.pretrained_dir.mkdir(exist_ok=True)

    def save_model(self, model, model_name, metadata=None, save_format='pt'):
        """
        Save model with metadata and validation

        Args:
            model: YOLO model instance
            model_name: Name for the saved model
            metadata: Dictionary with training metadata
            save_format: Format to save ('pt', 'onnx', 'torchscript')
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = self.base_dir / f"{model_name}_{timestamp}"
            model_dir.mkdir(exist_ok=True)

            # Save model weights
            model_path = model_dir / f"{model_name}.{save_format}"
            if save_format == 'pt':
                model.save(str(model_path))
            elif save_format == 'onnx':
                model.export(format='onnx', dynamic=True)
            elif save_format == 'torchscript':
                model.export(format='torchscript')

            logger.info(f"Model saved to: {model_path}")

            # Save metadata
            if metadata:
                metadata_path = model_dir / "metadata.json"
                metadata['save_time'] = timestamp
                metadata['model_path'] = str(model_path)
                metadata['format'] = save_format

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

                logger.info(f"Metadata saved to: {metadata_path}")

            # Validate saved model
            if self._validate_saved_model(model_path, save_format):
                logger.info("Model validation successful")
                return str(model_path)
            else:
                logger.error("Model validation failed")
                return None

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, model_path, validate=True):
        """
        Load model with validation

        Args:
            model_path: Path to the saved model
            validate: Whether to validate the loaded model
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None

            logger.info(f"Loading model from: {model_path}")
            model = YOLO(model_path)

            if validate:
                if self._validate_loaded_model(model):
                    logger.info("Model loaded and validated successfully")
                    return model
                else:
                    logger.error("Model validation failed after loading")
                    return None
            else:
                logger.info("Model loaded successfully (validation skipped)")
                return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def save_checkpoint(self, model, epoch, optimizer_state=None, scheduler_state=None,
                       metrics=None, checkpoint_name=None):
        """
        Save training checkpoint

        Args:
            model: YOLO model instance
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metrics: Training metrics
            checkpoint_name: Name for checkpoint file
        """
        try:
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"

            checkpoint_path = self.checkpoints_dir / checkpoint_name

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.model.state_dict() if hasattr(model, 'model') else model.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'scheduler_state_dict': scheduler_state,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Save checkpoint metadata
            metadata = {
                'epoch': epoch,
                'checkpoint_path': str(checkpoint_path),
                'timestamp': checkpoint['timestamp'],
                'metrics': metrics
            }

            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """
        Load training checkpoint for resume training

        Args:
            checkpoint_path: Path to checkpoint file
            model: YOLO model instance to load state into
            optimizer: Optimizer instance to load state into
            scheduler: Scheduler instance to load state into

        Returns:
            epoch: Last completed epoch
            metrics: Training metrics from checkpoint
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None, None

            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load model state with strict=False to handle potential mismatches
            if hasattr(model, 'model'):
                # For YOLO models, try strict loading first, then non-strict if it fails
                try:
                    model.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    logger.info("Model state loaded with strict=True")
                except RuntimeError as e:
                    logger.warning(f"Strict loading failed: {e}")
                    logger.info("Attempting non-strict loading...")
                    model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    logger.info("Model state loaded with strict=False (some layers may not be loaded)")
            else:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    logger.info("Model state loaded with strict=True")
                except RuntimeError as e:
                    logger.warning(f"Strict loading failed: {e}")
                    logger.info("Attempting non-strict loading...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    logger.info("Model state loaded with strict=False (some layers may not be loaded)")

            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")

            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Scheduler state loaded")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")

            epoch = checkpoint.get('epoch', 0)
            metrics = checkpoint.get('metrics', {})

            logger.info(f"Checkpoint loaded successfully. Resuming from epoch {epoch}")
            return epoch, metrics

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None, None

    def resume_training(self, checkpoint_path, model_path='yolov8x.pt'):
        """
        Resume training from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model_path: Base model path for YOLO initialization

        Returns:
            model: Loaded YOLO model
            epoch: Epoch to resume from
            optimizer: Optimizer with loaded state
            scheduler: Scheduler with loaded state
        """
        try:
            # Load base model
            model = YOLO(model_path)

            # Load checkpoint
            epoch, metrics = self.load_checkpoint(checkpoint_path, model)

            if epoch is None:
                return None, None, None, None

            # Note: Optimizer and scheduler need to be recreated with same parameters
            # This would typically be done in the training script

            logger.info(f"Training resumed from epoch {epoch}")
            return model, epoch, None, None  # Return None for optimizer/scheduler as they need recreation

        except Exception as e:
            logger.error(f"Error resuming training: {e}")
            return None, None, None, None

    def _validate_saved_model(self, model_path, save_format):
        """Validate saved model file"""
        try:
            if save_format == 'pt':
                # Try to load the model
                test_model = YOLO(model_path)
                return test_model is not None
            elif save_format == 'onnx':
                import onnxruntime as ort
                session = ort.InferenceSession(model_path)
                return session is not None
            elif save_format == 'torchscript':
                model = torch.jit.load(model_path)
                return model is not None
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def _validate_loaded_model(self, model):
        """Validate loaded YOLO model"""
        try:
            # Check if model has required attributes
            required_attrs = ['model', 'names', 'predict']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    logger.error(f"Model missing required attribute: {attr}")
                    return False

            # Try a dummy prediction to ensure model works
            # This is optional and can be skipped for performance
            logger.info("Model validation passed")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def list_saved_models(self):
        """List all saved models"""
        models = []
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.onnx")) + list(model_dir.glob("*.torchscript"))
                if model_files:
                    metadata_file = model_dir / "metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass

                    models.append({
                        'name': model_dir.name,
                        'path': str(model_files[0]),
                        'metadata': metadata
                    })

        return models

    def list_checkpoints(self):
        """List all saved checkpoints"""
        checkpoints = []
        for checkpoint_file in self.checkpoints_dir.glob("*.pt"):
            metadata_file = checkpoint_file.with_suffix('.json')
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass

            checkpoints.append({
                'name': checkpoint_file.name,
                'path': str(checkpoint_file),
                'metadata': metadata
            })

        return sorted(checkpoints, key=lambda x: x['metadata'].get('epoch', 0), reverse=True)

    def list_pretrained_models(self):
        """List all available pretrained models"""
        pretrained_models = []
        if self.pretrained_dir.exists():
            for model_file in self.pretrained_dir.glob("*.pt"):
                pretrained_models.append({
                    'name': model_file.stem,
                    'path': str(model_file),
                    'size': model_file.stat().st_size
                })

        return sorted(pretrained_models, key=lambda x: x['name'])

    def load_pretrained_model(self, model_name, validate=True):
        """
        Load a pretrained model by name

        Args:
            model_name: Name of the pretrained model (e.g., 'yolov8n', 'yolov8x')
            validate: Whether to validate the loaded model

        Returns:
            YOLO model instance or None if failed
        """
        model_path = self.pretrained_dir / f"{model_name}.pt"
        if not model_path.exists():
            logger.error(f"Pretrained model '{model_name}' not found in {self.pretrained_dir}")
            return None

        return self.load_model(str(model_path), validate)

    def get_available_pretrained_models(self):
        """Get list of available pretrained model names"""
        models = self.list_pretrained_models()
        return [model['name'] for model in models]

# Global model manager instance
model_manager = ModelManager()

def save_model(model, model_name, metadata=None, save_format='pt'):
    """Convenience function to save model"""
    return model_manager.save_model(model, model_name, metadata, save_format)

def load_model(model_path, validate=True):
    """Convenience function to load model"""
    return model_manager.load_model(model_path, validate)

def save_checkpoint(model, epoch, optimizer_state=None, scheduler_state=None, metrics=None, checkpoint_name=None):
    """Convenience function to save checkpoint"""
    return model_manager.save_checkpoint(model, epoch, optimizer_state, scheduler_state, metrics, checkpoint_name)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Convenience function to load checkpoint"""
    return model_manager.load_checkpoint(checkpoint_path, model, optimizer, scheduler)

def resume_training(checkpoint_path, model_path='yolov8x.pt'):
    """Convenience function to resume training"""
    return model_manager.resume_training(checkpoint_path, model_path)

def list_saved_models():
    """Convenience function to list saved models"""
    return model_manager.list_saved_models()

def list_checkpoints():
    """Convenience function to list checkpoints"""
    return model_manager.list_checkpoints()

def list_pretrained_models():
    """Convenience function to list pretrained models"""
    return model_manager.list_pretrained_models()

def load_pretrained_model(model_name, validate=True):
    """Convenience function to load pretrained model by name"""
    return model_manager.load_pretrained_model(model_name, validate)

def get_available_pretrained_models():
    """Convenience function to get available pretrained model names"""
    return model_manager.get_available_pretrained_models()

if __name__ == "__main__":
    # Example usage
    print("Model Utilities Module")
    print("Available functions:")
    print("- save_model(model, model_name, metadata, save_format)")
    print("- load_model(model_path, validate)")
    print("- save_checkpoint(model, epoch, optimizer_state, scheduler_state, metrics)")
    print("- load_checkpoint(checkpoint_path, model, optimizer, scheduler)")
    print("- resume_training(checkpoint_path, model_path)")
    print("- list_saved_models()")
    print("- list_checkpoints()")
    print("- list_pretrained_models()")
    print("- load_pretrained_model(model_name, validate)")
    print("- get_available_pretrained_models()")
