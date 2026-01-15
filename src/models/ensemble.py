"""
Ensemble model combining Random Forest, XGBoost, and LSTM predictions.

This module provides an EnsembleModel class that combines predictions
from multiple models using configurable weighted averaging, with
fallback logic if individual models fail.

Example:
    >>> from src.models.ensemble import EnsembleModel
    >>> ensemble = EnsembleModel()
    >>> ensemble.load_models(
    ...     rf_path="models/random_forest.joblib",
    ...     xgb_path="models/xgboost.json",
    ...     lstm_path="models/lstm/"
    ... )
    >>> predictions = ensemble.predict(X_flat, X_sequences)

Ensemble Methods:
    - weighted_average: Combine predictions using configurable weights
    - simple_average: Equal weight for all models
    - median: Use median of predictions (robust to outliers)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import settings
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

# Conditional imports
try:
    from src.models.lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnsembleError(Exception):
    """Exception raised for ensemble-related errors."""
    pass


class EnsembleModel:
    """
    Ensemble model combining RF, XGBoost, and LSTM predictions.

    This class provides a unified interface for making predictions using
    multiple models, with support for weighted averaging, fallback logic
    when individual models fail, and confidence estimation.

    Attributes:
        models: Dictionary of loaded models {name: model_instance}
        weights: Dictionary of model weights {name: weight}
        ensemble_method: Method for combining predictions
        is_loaded: Whether models have been loaded
        metrics: Dictionary of evaluation metrics

    Example:
        >>> ensemble = EnsembleModel(
        ...     weights={"rf": 0.3, "xgb": 0.3, "lstm": 0.4}
        ... )
        >>> ensemble.load_models(rf_path="...", xgb_path="...", lstm_path="...")
        >>> predictions = ensemble.predict(X, X_seq)
        >>> confidence = ensemble.get_prediction_confidence(X, X_seq)
    """

    # Default weights for each model
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "rf": 0.25,
        "xgb": 0.35,
        "lstm": 0.40,
    }

    # Supported ensemble methods
    SUPPORTED_METHODS = ["weighted_average", "simple_average", "median"]

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ensemble_method: str = "weighted_average",
    ) -> None:
        """
        Initialize the ensemble model.

        Args:
            weights: Dictionary of model weights. Must sum to 1.0.
                    Keys: 'rf', 'xgb', 'lstm'
            ensemble_method: Method for combining predictions.
                           Options: 'weighted_average', 'simple_average', 'median'

        Raises:
            ValueError: If weights don't sum to 1.0 or method is unsupported
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

        if ensemble_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported ensemble method: {ensemble_method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        self.ensemble_method = ensemble_method
        self.models: Dict[str, Any] = {}
        self.is_loaded = False
        self.metrics: Dict[str, float] = {}
        self._model_availability: Dict[str, bool] = {}

        logger.info(
            f"Initialized EnsembleModel with method={ensemble_method}, "
            f"weights={self.weights}"
        )

    def load_models(
        self,
        rf_path: Optional[Union[str, Path]] = None,
        xgb_path: Optional[Union[str, Path]] = None,
        lstm_path: Optional[Union[str, Path]] = None,
        require_all: bool = False,
    ) -> "EnsembleModel":
        """
        Load all models from disk.

        Args:
            rf_path: Path to Random Forest model
            xgb_path: Path to XGBoost model
            lstm_path: Path to LSTM model directory
            require_all: If True, raise error if any model fails to load

        Returns:
            Self for method chaining

        Raises:
            EnsembleError: If require_all=True and any model fails to load
        """
        self.models = {}
        self._model_availability = {}

        # Load Random Forest
        if rf_path:
            try:
                rf_model = RandomForestModel()
                rf_model.load(rf_path)
                self.models["rf"] = rf_model
                self._model_availability["rf"] = True
                logger.info(f"Loaded Random Forest model from {rf_path}")
            except Exception as e:
                self._model_availability["rf"] = False
                logger.warning(f"Failed to load Random Forest model: {e}")
                if require_all:
                    raise EnsembleError(f"Failed to load Random Forest: {e}")

        # Load XGBoost
        if xgb_path:
            try:
                xgb_model = XGBoostModel()
                xgb_model.load(xgb_path)
                self.models["xgb"] = xgb_model
                self._model_availability["xgb"] = True
                logger.info(f"Loaded XGBoost model from {xgb_path}")
            except Exception as e:
                self._model_availability["xgb"] = False
                logger.warning(f"Failed to load XGBoost model: {e}")
                if require_all:
                    raise EnsembleError(f"Failed to load XGBoost: {e}")

        # Load LSTM
        if lstm_path and LSTM_AVAILABLE:
            try:
                lstm_model = LSTMModel()
                lstm_model.load(lstm_path)
                self.models["lstm"] = lstm_model
                self._model_availability["lstm"] = True
                logger.info(f"Loaded LSTM model from {lstm_path}")
            except Exception as e:
                self._model_availability["lstm"] = False
                logger.warning(f"Failed to load LSTM model: {e}")
                if require_all:
                    raise EnsembleError(f"Failed to load LSTM: {e}")
        elif lstm_path and not LSTM_AVAILABLE:
            logger.warning("LSTM model requested but TensorFlow not available")
            self._model_availability["lstm"] = False

        if not self.models:
            raise EnsembleError("No models were loaded successfully")

        self.is_loaded = True
        self._rebalance_weights()

        logger.info(f"Ensemble loaded with models: {list(self.models.keys())}")

        return self

    def add_model(
        self,
        name: str,
        model: Any,
        weight: Optional[float] = None,
    ) -> "EnsembleModel":
        """
        Add a pre-trained model to the ensemble.

        Args:
            name: Name for the model ('rf', 'xgb', 'lstm', or custom)
            model: Trained model instance with predict() method
            weight: Weight for this model (will trigger rebalancing)

        Returns:
            Self for method chaining
        """
        self.models[name] = model
        self._model_availability[name] = True

        if weight is not None:
            self.weights[name] = weight
            self._validate_weights()
        else:
            self._rebalance_weights()

        self.is_loaded = True
        logger.info(f"Added model '{name}' to ensemble")

        return self

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        X_sequences: Optional[np.ndarray] = None,
        return_individual: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Make ensemble predictions.

        Args:
            X: Features for RF and XGBoost (n_samples, n_features)
            X_sequences: Sequences for LSTM (n_samples, seq_len, n_features)
            return_individual: If True, also return individual model predictions

        Returns:
            If return_individual=False: Ensemble predictions (n_samples,)
            If return_individual=True: (ensemble_predictions, {model: predictions})

        Raises:
            EnsembleError: If no models are loaded or no valid predictions
        """
        if not self.is_loaded:
            raise EnsembleError("Models must be loaded before prediction")

        if X is None and X_sequences is None:
            raise ValueError("Must provide X and/or X_sequences")

        individual_predictions = {}

        # Get predictions from each model with fallback
        for name, model in self.models.items():
            try:
                if name == "lstm":
                    if X_sequences is not None:
                        pred = model.predict(X_sequences)
                        individual_predictions[name] = pred
                    else:
                        logger.warning(f"Skipping {name}: no sequence data provided")
                else:
                    if X is not None:
                        pred = model.predict(X)
                        individual_predictions[name] = pred
                    else:
                        logger.warning(f"Skipping {name}: no flat data provided")

            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                self._model_availability[name] = False

        if not individual_predictions:
            raise EnsembleError("All models failed to produce predictions")

        # Combine predictions
        ensemble_pred = self._combine_predictions(individual_predictions)

        if return_individual:
            return ensemble_pred, individual_predictions
        return ensemble_pred

    def get_prediction_confidence(
        self,
        X: Optional[np.ndarray] = None,
        X_sequences: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get confidence scores for predictions.

        Confidence is calculated as the inverse of the standard deviation
        of predictions across models, normalized to [0, 1].

        Higher confidence = models agree more closely.

        Args:
            X: Features for RF and XGBoost
            X_sequences: Sequences for LSTM

        Returns:
            Confidence scores (n_samples,) in range [0, 1]
        """
        _, individual_predictions = self.predict(
            X, X_sequences, return_individual=True
        )

        if len(individual_predictions) < 2:
            # Can't compute variance with only one model
            logger.warning("Cannot compute confidence with fewer than 2 models")
            return np.ones(len(list(individual_predictions.values())[0]))

        # Stack predictions
        pred_matrix = np.stack(list(individual_predictions.values()), axis=0)

        # Calculate standard deviation across models
        std = np.std(pred_matrix, axis=0)

        # Convert to confidence (inverse relationship)
        # Use sigmoid-like transformation: confidence = 1 / (1 + std/scale)
        scale = 10.0  # Scaling factor for std
        confidence = 1.0 / (1.0 + std / scale)

        return confidence

    def get_prediction_uncertainty(
        self,
        X: Optional[np.ndarray] = None,
        X_sequences: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates for predictions.

        Args:
            X: Features for RF and XGBoost
            X_sequences: Sequences for LSTM

        Returns:
            Dictionary with:
            - 'mean': Mean prediction
            - 'std': Standard deviation across models
            - 'min': Minimum prediction
            - 'max': Maximum prediction
            - 'range': Max - min (prediction spread)
        """
        _, individual_predictions = self.predict(
            X, X_sequences, return_individual=True
        )

        pred_matrix = np.stack(list(individual_predictions.values()), axis=0)

        return {
            "mean": np.mean(pred_matrix, axis=0),
            "std": np.std(pred_matrix, axis=0),
            "min": np.min(pred_matrix, axis=0),
            "max": np.max(pred_matrix, axis=0),
            "range": np.max(pred_matrix, axis=0) - np.min(pred_matrix, axis=0),
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_sequences: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X: Features
            y: True RUL values
            X_sequences: Sequences for LSTM

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X, X_sequences)

        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions)),
        }

        logger.info(f"Ensemble evaluation metrics: {self.metrics}")

        return self.metrics

    def set_weights(self, weights: Dict[str, float]) -> "EnsembleModel":
        """
        Set new model weights.

        Args:
            weights: Dictionary of weights

        Returns:
            Self for method chaining

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        self.weights = weights
        self._validate_weights()
        logger.info(f"Updated weights: {self.weights}")
        return self

    def set_ensemble_method(self, method: str) -> "EnsembleModel":
        """
        Set the ensemble method.

        Args:
            method: Ensemble method ('weighted_average', 'simple_average', 'median')

        Returns:
            Self for method chaining
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}")

        self.ensemble_method = method
        logger.info(f"Updated ensemble method: {method}")
        return self

    def save(self, path: Union[str, Path]) -> None:
        """
        Save ensemble configuration and metadata.

        Note: Individual models should be saved separately.
        This saves the ensemble configuration for reconstruction.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "weights": self.weights,
            "ensemble_method": self.ensemble_method,
            "model_availability": self._model_availability,
            "metrics": self.metrics,
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved ensemble configuration to {path}")

    def load_config(self, path: Union[str, Path]) -> "EnsembleModel":
        """
        Load ensemble configuration.

        Note: Models must be loaded separately after loading config.

        Args:
            path: Path to configuration file

        Returns:
            Self for method chaining
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "r") as f:
            config = json.load(f)

        self.weights = config.get("weights", self.DEFAULT_WEIGHTS)
        self.ensemble_method = config.get("ensemble_method", "weighted_average")
        self.metrics = config.get("metrics", {})

        logger.info(f"Loaded ensemble configuration from {path}")

        return self

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        return {
            "loaded_models": list(self.models.keys()),
            "model_availability": self._model_availability.copy(),
            "weights": self.weights.copy(),
            "ensemble_method": self.ensemble_method,
            "effective_weights": self._get_effective_weights(),
        }

    def _combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine predictions using the configured method."""
        if self.ensemble_method == "weighted_average":
            return self._weighted_average(predictions)
        elif self.ensemble_method == "simple_average":
            return self._simple_average(predictions)
        elif self.ensemble_method == "median":
            return self._median(predictions)
        else:
            raise ValueError(f"Unknown method: {self.ensemble_method}")

    def _weighted_average(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute weighted average of predictions."""
        effective_weights = self._get_effective_weights(list(predictions.keys()))

        result = np.zeros_like(list(predictions.values())[0], dtype=np.float64)
        total_weight = 0.0

        for name, pred in predictions.items():
            weight = effective_weights.get(name, 0)
            result += weight * pred
            total_weight += weight

        if total_weight > 0:
            result /= total_weight

        return np.clip(result, 0, None)

    def _simple_average(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute simple average of predictions."""
        pred_list = list(predictions.values())
        result = np.mean(pred_list, axis=0)
        return np.clip(result, 0, None)

    def _median(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute median of predictions."""
        pred_matrix = np.stack(list(predictions.values()), axis=0)
        result = np.median(pred_matrix, axis=0)
        return np.clip(result, 0, None)

    def _get_effective_weights(
        self,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get weights normalized for available models.

        Args:
            available_models: List of available model names

        Returns:
            Normalized weights for available models
        """
        if available_models is None:
            available_models = list(self.models.keys())

        # Get weights for available models
        available_weights = {
            name: self.weights.get(name, 0)
            for name in available_models
            if name in self.weights
        }

        # Normalize to sum to 1
        total = sum(available_weights.values())
        if total > 0:
            return {k: v / total for k, v in available_weights.items()}

        # Equal weights if no weights specified
        n_models = len(available_models)
        if n_models > 0:
            return {name: 1.0 / n_models for name in available_models}

        return {}

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def _rebalance_weights(self) -> None:
        """Rebalance weights for available models."""
        available = [
            name for name in self.models.keys()
            if self._model_availability.get(name, True)
        ]

        if not available:
            return

        # Get weights for available models
        available_weights = {
            name: self.weights.get(name, 1.0 / len(available))
            for name in available
        }

        # Normalize
        total = sum(available_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in available_weights.items()}

        logger.info(f"Rebalanced weights: {self.weights}")
