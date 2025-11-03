#!/usr/bin/env python3
"""
Comprehensive test suite for UltimateSmallObjectTrainer
Tests all major functionality including GPU management, post-processing,
hyperparameter optimization, and ensemble methods.
"""

import unittest
import tempfile
import os
import yaml
import torch
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import the classes to test
from ultimate_small_object_detector import (
    UltimateSmallObjectTrainer,
    AdvancedPostProcessor,
    GPUMemoryManager
)

class TestGPUMemoryManager(unittest.TestCase):
    """Test GPU Memory Management functionality"""

    def setUp(self):
        self.manager = GPUMemoryManager(device_id=0)

    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsInstance(self.manager, GPUMemoryManager)
        self.assertEqual(self.manager.device_id, 0)
        self.assertEqual(self.manager.max_memory_usage, 0.85)
        self.assertEqual(self.manager.min_batch_size, 1)
        self.assertEqual(self.manager.max_batch_size, 32)

    def test_get_gpu_memory(self):
        """Test GPU memory retrieval"""
        memory = self.manager.get_gpu_memory()
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(memory, 0)

    def test_get_gpu_memory_total(self):
        """Test total GPU memory retrieval"""
        total_memory = self.manager.get_gpu_memory_total()
        self.assertIsInstance(total_memory, float)
        if torch.cuda.is_available():
            self.assertGreater(total_memory, 0)
        else:
            self.assertEqual(total_memory, 0)

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        batch_size = self.manager.get_optimal_batch_size(8)
        self.assertIsInstance(batch_size, int)
        self.assertGreaterEqual(batch_size, self.manager.min_batch_size)
        self.assertLessEqual(batch_size, self.manager.max_batch_size)

    def test_monitor_memory(self):
        """Test memory monitoring"""
        status = self.manager.monitor_memory()
        self.assertIn('used_gb', status)
        self.assertIn('total_gb', status)
        self.assertIn('free_gb', status)
        self.assertIn('usage_percent', status)
        self.assertIn('status', status)

class TestAdvancedPostProcessor(unittest.TestCase):
    """Test Advanced Post-Processing functionality"""

    def setUp(self):
        self.processor = AdvancedPostProcessor()

    def test_initialization(self):
        """Test processor initialization"""
        self.assertIsInstance(self.processor, AdvancedPostProcessor)
        self.assertEqual(self.processor.conf_threshold, 0.25)
        self.assertEqual(self.processor.iou_threshold, 0.45)
        self.assertEqual(self.processor.small_object_threshold, 32)

    def test_nms(self):
        """Test Non-Maximum Suppression"""
        boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)

        keep = self.processor.nms(boxes, scores, 0.5)
        self.assertIsInstance(keep, torch.Tensor)
        self.assertGreaterEqual(len(keep), 0)

    def test_box_iou(self):
        """Test IoU calculation"""
        box1 = torch.tensor([0, 0, 10, 10], dtype=torch.float32)
        boxes = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)

        iou = self.processor.box_iou(box1, boxes)
        self.assertIsInstance(iou, torch.Tensor)
        self.assertEqual(len(iou), 1)
        self.assertGreaterEqual(iou.item(), 0)
        self.assertLessEqual(iou.item(), 1)

    def test_apply_adaptive_nms(self):
        """Test adaptive NMS application"""
        boxes = [[0, 0, 10, 10], [50, 50, 100, 100]]
        scores = [0.9, 0.8]
        classes = [0, 0]
        image_size = (640, 640)

        final_boxes, final_scores, final_classes = self.processor.apply_adaptive_nms(
            boxes, scores, classes, image_size
        )

        self.assertIsInstance(final_boxes, list)
        self.assertIsInstance(final_scores, list)
        self.assertIsInstance(final_classes, list)
        self.assertEqual(len(final_boxes), len(final_scores))
        self.assertEqual(len(final_scores), len(final_classes))

class TestUltimateSmallObjectTrainer(unittest.TestCase):
    """Test Ultimate Small Object Trainer functionality"""

    def setUp(self):
        self.trainer = UltimateSmallObjectTrainer(model_path='yolov8n.pt', device_id=0)

    def test_initialization(self):
        """Test trainer initialization"""
        self.assertIsInstance(self.trainer, UltimateSmallObjectTrainer)
        self.assertEqual(self.trainer.model_path, 'yolov8n.pt')
        self.assertEqual(self.trainer.device_id, 0)
        self.assertIsInstance(self.trainer.memory_manager, GPUMemoryManager)
        self.assertIsInstance(self.trainer.post_processor, AdvancedPostProcessor)

    @patch('builtins.open', new_callable=mock_open, read_data='train: []\nval: []\nnc: 20\nnames: []')
    def test_create_ultimate_training_config(self, mock_file):
        """Test configuration creation"""
        config = self.trainer.create_ultimate_training_config('test.yaml')

        self.assertIsInstance(config, dict)
        self.assertIn('data', config)
        self.assertIn('epochs', config)
        self.assertIn('imgsz', config)
        self.assertIn('batch', config)
        self.assertIn('device', config)

        # Check specific values
        self.assertEqual(config['data'], 'test.yaml')
        self.assertIsInstance(config['imgsz'], list)
        self.assertGreater(len(config['imgsz']), 1)

        # Check blur augmentation parameters (included by default)
        self.assertIn('blur_prob', config)
        self.assertIn('blur_kernel_range', config)
        self.assertIn('blur_sigma_range', config)
        self.assertEqual(config['blur_prob'], 0.5)
        self.assertEqual(config['blur_kernel_range'], [3, 7])
        self.assertEqual(config['blur_sigma_range'], [0.1, 2.0])

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_preprocess_dataset(self, mock_file, mock_exists):
        """Test dataset preprocessing"""
        mock_exists.return_value = True

        # Mock yaml content
        mock_file.return_value.read.return_value = """
        train: ['images/train']
        val: ['images/val']
        nc: 20
        names: ['class1', 'class2']
        """

        with patch('ultimate_small_object_detector.DataQualityChecker') as mock_checker:
            mock_instance = MagicMock()
            mock_checker.return_value = mock_instance
            mock_instance.clean_dataset.return_value = ([], [])

            with patch('ultimate_small_object_detector.analyze_class_distribution') as mock_analyze:
                mock_analyze.return_value = {0: 100, 1: 50}

                with patch('ultimate_small_object_detector.balance_classes') as mock_balance:
                    mock_balance.return_value = []

                    # This should not raise an exception
                    self.trainer.preprocess_dataset('test.yaml')

                    # Verify calls were made
                    mock_checker.assert_called_once()
                    mock_analyze.assert_called_once()
                    mock_balance.assert_called_once()

    @patch('torch.load')
    @patch('os.path.exists')
    def test_load_checkpoint(self, mock_exists, mock_load):
        """Test checkpoint loading"""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_load.return_value = {'model_state_dict': {'layer.weight': torch.randn(10, 10)}}

        # Mock YOLO model
        with patch('ultimate_small_object_detector.YOLO') as mock_yolo:
            mock_yolo_instance = MagicMock()
            mock_yolo.return_value = mock_yolo_instance

            result = self.trainer.load_checkpoint(mock_yolo_instance, 'test.pt')

            self.assertEqual(result, mock_yolo_instance)
            mock_load.assert_called_once_with('test.pt')

    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        with patch('torch.save') as mock_save:
            with patch('ultimate_small_object_detector.datetime') as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = '2023-01-01T00:00:00'

                self.trainer.save_checkpoint(mock_model, mock_optimizer, 10, 0.85, 'test.pt')

                # Verify save was called
                mock_save.assert_called_once()
                args, kwargs = mock_save.call_args
                checkpoint = args[0]

                self.assertIn('epoch', checkpoint)
                self.assertIn('model_state_dict', checkpoint)
                self.assertIn('optimizer_state_dict', checkpoint)
                self.assertIn('loss', checkpoint)
                self.assertIn('timestamp', checkpoint)

    @patch('ultimate_small_object_detector.YOLO')
    def test_perform_grid_search_hyperparameter_optimization(self, mock_yolo):
        """Test grid search optimization"""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock training results
        mock_results = MagicMock()
        mock_results.box.map50 = 0.85
        mock_results.box.map = 0.82
        mock_model.train.return_value = mock_results

        param_grid = {
            'lr0': [0.001, 0.005],
            'batch': [4, 8],
            'dropout': [0.1, 0.3],
            'iou': [0.5, 0.75]
        }

        with patch('builtins.print'):  # Suppress print statements
            best_params = self.trainer.perform_grid_search_hyperparameter_optimization('test.yaml', param_grid)

        self.assertIsInstance(best_params, dict)
        self.assertIn('lr0', best_params)
        self.assertIn('batch', best_params)
        self.assertIn('dropout', best_params)
        self.assertIn('iou', best_params)

    @patch('ultimate_small_object_detector.YOLO')
    def test_ensemble_models(self, mock_yolo):
        """Test model ensemble"""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_yolo.side_effect = [mock_model1, mock_model2]

        # Mock validation results
        mock_results1 = MagicMock()
        mock_results1.box.mp = 0.85
        mock_results1.box.mr = 0.90
        mock_results1.box.map50 = 0.82

        mock_results2 = MagicMock()
        mock_results2.box.mp = 0.88
        mock_results2.box.mr = 0.87
        mock_results2.box.map50 = 0.85

        mock_model1.val.return_value = mock_results1
        mock_model2.val.return_value = mock_results2

        model_paths = ['model1.pt', 'model2.pt']

        with patch('builtins.print'):  # Suppress print statements
            precision, recall, map50 = self.trainer.ensemble_models(model_paths, 'test_data')

        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(map50, float)

        # Check ensemble averaging
        expected_precision = (0.85 + 0.88) / 2
        expected_recall = (0.90 + 0.87) / 2
        expected_map50 = (0.82 + 0.85) / 2

        self.assertAlmostEqual(precision, expected_precision)
        self.assertAlmostEqual(recall, expected_recall)
        self.assertAlmostEqual(map50, expected_map50)

    @patch('ultimate_small_object_detector.YOLO')
    @patch('os.path.exists')
    def test_test_ultimate_model(self, mock_exists, mock_yolo):
        """Test ultimate model testing"""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock inference results
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_result[0].boxes = mock_boxes
        mock_model.return_value = [mock_result]

        with patch('pathlib.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.glob.return_value = [Path('test_image.jpg')]
            mock_path.return_value = mock_path_instance

            with patch('builtins.print'):  # Suppress print statements
                results = self.trainer.test_ultimate_model('model.pt', 'test_images', max_images=1)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIn('image', results[0])
        self.assertIn('detections', results[0])
        self.assertIn('boxes', results[0])
        self.assertIn('scores', results[0])
        self.assertIn('classes', results[0])

    @patch('ultimate_small_object_detector.YOLO')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_train_ultimate_model(self, mock_file, mock_exists, mock_yolo):
        """Test ultimate model training"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = 'train: []\nval: []\nnc: 20\nnames: []'

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock training results
        mock_results = MagicMock()
        mock_results.box.map50 = 0.85
        mock_results.box.map = 0.82
        mock_model.train.return_value = mock_results

        # Mock other dependencies
        with patch('ultimate_small_object_detector.add_se_and_c2f_to_yolo') as mock_add:
            mock_add.return_value = mock_model

            with patch.object(self.trainer, 'preprocess_dataset') as mock_preprocess:
                with patch.object(self.trainer, 'perform_grid_search_hyperparameter_optimization') as mock_grid:
                    mock_grid.return_value = {'lr0': 0.005, 'batch': 8, 'dropout': 0.3, 'iou': 0.75}

                    with patch.object(self.trainer.memory_manager, 'clear_gpu_cache') as mock_clear:
                        with patch.object(self.trainer, 'save_checkpoint') as mock_save:
                            with patch('ultimate_small_object_detector.datetime') as mock_datetime:
                                mock_datetime.now.return_value.strftime.return_value = '20231023_123456'

                                with patch('builtins.print'):  # Suppress print statements
                                    model, model_path = self.trainer.train_ultimate_model('test.yaml', epochs=1)

        self.assertIsNotNone(model)
        self.assertIsInstance(model_path, str)
        self.assertIn('ultimate_', model_path)
        self.assertIn('.pt', model_path)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_full_workflow_simulation(self):
        """Test a simulated full workflow"""
        trainer = UltimateSmallObjectTrainer()

        # Test config creation
        config = trainer.create_ultimate_training_config('voc2012.yaml', epochs=1)
        self.assertIsInstance(config, dict)

        # Test memory management
        memory_status = trainer.memory_manager.monitor_memory()
        self.assertIsInstance(memory_status, dict)

        # Test post-processing
        processor = trainer.post_processor
        boxes = [[0, 0, 10, 10]]
        scores = [0.9]
        classes = [0]

        final_boxes, final_scores, final_classes = processor.apply_adaptive_nms(
            boxes, scores, classes, (640, 640)
        )

        self.assertEqual(len(final_boxes), len(final_scores))
        self.assertEqual(len(final_scores), len(final_classes))

def run_tests():
    """Run all tests with verbose output"""
    unittest.main(verbosity=2, exit=False)

if __name__ == '__main__':
    print("Running comprehensive tests for Ultimate Small Object Detector...")
    print("="*70)
    run_tests()
    print("="*70)
    print("All tests completed!")
