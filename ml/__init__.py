"""
ML Package for YouTube Viral Prediction
"""
from ml.data_processor import DataProcessor
from ml.model_trainer import ModelTrainer
from ml.evaluator import ModelEvaluator

__all__ = [
    'DataProcessor',
    'ModelTrainer',
    'ModelEvaluator'
]

__version__ = '1.0.0'
