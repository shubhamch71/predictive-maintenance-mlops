"""
Training and evaluation modules.

This package provides training orchestration and model evaluation
for the Predictive Maintenance MLOps Platform.

Modules:
    train: Training orchestrator for RF, XGBoost, and LSTM models
    evaluate: Model evaluation and comparison utilities

Example:
    >>> from src.training import TrainingOrchestrator, ModelEvaluator
    >>> orchestrator = TrainingOrchestrator(experiment_name="my-exp")
    >>> results = orchestrator.train_all_models()
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.compute_metrics(y_true, y_pred)
"""

from src.training.evaluate import (
    ModelEvaluator,
    EvaluationError,
)

# Conditional imports for train module (may fail if model dependencies missing)
try:
    from src.training.train import (
        TrainingOrchestrator,
        TrainingError,
        DataPreparationError,
        ModelTrainingError,
        MLflowError,
        parse_args,
        main,
        RF_AVAILABLE,
        XGB_AVAILABLE,
        LSTM_AVAILABLE,
    )
    TRAIN_MODULE_AVAILABLE = True
except Exception as e:
    # If model imports fail (ImportError, XGBoostError, etc.), still allow evaluate module to work
    TrainingOrchestrator = None
    TrainingError = Exception
    DataPreparationError = Exception
    ModelTrainingError = Exception
    MLflowError = Exception
    parse_args = None
    main = None
    RF_AVAILABLE = False
    XGB_AVAILABLE = False
    LSTM_AVAILABLE = False
    TRAIN_MODULE_AVAILABLE = False
    import logging
    logging.warning(f"Training module not fully available: {e}")

__all__ = [
    # Evaluate module
    "ModelEvaluator",
    "EvaluationError",
    # Train module
    "TrainingOrchestrator",
    "TrainingError",
    "DataPreparationError",
    "ModelTrainingError",
    "MLflowError",
    "parse_args",
    "main",
    # Availability flags
    "RF_AVAILABLE",
    "XGB_AVAILABLE",
    "LSTM_AVAILABLE",
    "TRAIN_MODULE_AVAILABLE",
]
