"""
Predictor Module for Predictive Maintenance Models

This module provides a thread-safe predictor class that:
- Loads models from MLflow Model Registry or local paths
- Preprocesses input using the same scaler from training
- Runs ensemble predictions combining RF, XGBoost, and LSTM
- Caches models in memory for fast inference

Usage:
    predictor = Predictor()
    predictor.load_models(mlflow_uri="http://mlflow:5000")
    result = predictor.predict(sensor_data)
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("predictor")


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    version: str
    model_type: str
    loaded_from: str
    load_time: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a prediction."""
    rul: float
    confidence: float
    model_version: str
    model_type: str
    individual_predictions: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None


class Predictor:
    """
    Thread-safe predictor for RUL (Remaining Useful Life) predictions.

    Loads and manages ensemble models (Random Forest, XGBoost, LSTM)
    and provides preprocessing and prediction capabilities.
    """

    # Feature configuration (must match training)
    SENSOR_COLUMNS = [
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
        "sensor_21",
    ]

    OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

    SENSORS_FOR_FEATURES = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ]

    DEFAULT_ROLLING_WINDOWS = [5, 10, 20]
    DEFAULT_LAG_PERIODS = [1, 5, 10]
    DEFAULT_SEQUENCE_LENGTH = 50

    # Ensemble weights (can be tuned based on validation performance)
    DEFAULT_WEIGHTS = {
        "random_forest": 0.35,
        "xgboost": 0.40,
        "lstm": 0.25,
    }

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        sequence_length: int = 50,
    ):
        """
        Initialize the predictor.

        Args:
            model_weights: Weights for ensemble voting (default: equal weights)
            sequence_length: Sequence length for LSTM model
        """
        self._lock = threading.RLock()
        self._models: Dict[str, Any] = {}
        self._scaler = None
        self._feature_columns: List[str] = []
        self._model_info: Dict[str, ModelInfo] = {}
        self._is_loaded = False

        self.model_weights = model_weights or self.DEFAULT_WEIGHTS
        self.sequence_length = sequence_length

        # Rolling buffer for LSTM sequences
        self._sequence_buffer: List[np.ndarray] = []

        # Last prediction for explainability
        self._last_input: Optional[np.ndarray] = None
        self._last_prediction: Optional[PredictionResult] = None

        logger.info("Predictor initialized")

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        with self._lock:
            return self._is_loaded

    @property
    def model_info(self) -> Dict[str, ModelInfo]:
        """Get information about loaded models."""
        with self._lock:
            return self._model_info.copy()

    def load_models(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        model_registry_name: Optional[str] = None,
        model_stage: str = "Production",
        local_model_dir: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ) -> None:
        """
        Load models from MLflow Model Registry or local directory.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            model_registry_name: Base name for registered models
            model_stage: Model stage to load (Production, Staging, etc.)
            local_model_dir: Local directory containing saved models
            scaler_path: Path to saved scaler pickle file
        """
        from datetime import datetime

        with self._lock:
            logger.info("Loading models...")

            # Try MLflow first, then fall back to local
            if mlflow_tracking_uri:
                self._load_from_mlflow(
                    mlflow_tracking_uri,
                    model_registry_name or "predictive-maintenance",
                    model_stage,
                )
            elif local_model_dir:
                self._load_from_local(local_model_dir)
            else:
                raise ValueError(
                    "Must provide either mlflow_tracking_uri or local_model_dir"
                )

            # Load scaler
            if scaler_path:
                self._load_scaler(scaler_path)

            self._is_loaded = len(self._models) > 0

            if self._is_loaded:
                logger.info(f"Successfully loaded {len(self._models)} models")
                for name, info in self._model_info.items():
                    logger.info(f"  - {name}: {info.model_type} v{info.version}")
            else:
                logger.warning("No models were loaded!")

    def _load_from_mlflow(
        self,
        tracking_uri: str,
        model_registry_name: str,
        stage: str,
    ) -> None:
        """Load models from MLflow Model Registry."""
        from datetime import datetime

        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Loading models from MLflow: {tracking_uri}")

            model_configs = [
                ("random_forest", f"{model_registry_name}-rf", "sklearn"),
                ("xgboost", f"{model_registry_name}-xgb", "xgboost"),
                ("lstm", f"{model_registry_name}-lstm", "keras"),
            ]

            for model_type, registry_name, flavor in model_configs:
                try:
                    model_uri = f"models:/{registry_name}/{stage}"

                    if flavor == "sklearn":
                        model = mlflow.sklearn.load_model(model_uri)
                    elif flavor == "xgboost":
                        model = mlflow.xgboost.load_model(model_uri)
                    elif flavor == "keras":
                        model = mlflow.keras.load_model(model_uri)
                    else:
                        model = mlflow.pyfunc.load_model(model_uri)

                    self._models[model_type] = model

                    # Get model version info
                    client = mlflow.tracking.MlflowClient()
                    versions = client.get_latest_versions(registry_name, stages=[stage])
                    version = versions[0].version if versions else "unknown"

                    self._model_info[model_type] = ModelInfo(
                        name=registry_name,
                        version=version,
                        model_type=model_type,
                        loaded_from=f"mlflow://{registry_name}/{stage}",
                        load_time=datetime.now().isoformat(),
                    )

                    logger.info(f"Loaded {model_type} from MLflow (v{version})")

                except Exception as e:
                    logger.warning(f"Failed to load {model_type} from MLflow: {e}")

        except ImportError:
            logger.error("MLflow not installed. Install with: pip install mlflow")
            raise

    def _load_from_local(self, model_dir: str) -> None:
        """Load models from local directory."""
        from datetime import datetime

        import joblib

        model_dir = Path(model_dir)
        logger.info(f"Loading models from local directory: {model_dir}")

        # Load Random Forest
        rf_path = model_dir / "rf_model.joblib"
        if rf_path.exists():
            self._models["random_forest"] = joblib.load(rf_path)
            self._model_info["random_forest"] = ModelInfo(
                name="random_forest",
                version="local",
                model_type="random_forest",
                loaded_from=str(rf_path),
                load_time=datetime.now().isoformat(),
            )
            logger.info(f"Loaded Random Forest from {rf_path}")

        # Load XGBoost
        xgb_path = model_dir / "xgb_model.json"
        if xgb_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(str(xgb_path))
                self._models["xgboost"] = model
                self._model_info["xgboost"] = ModelInfo(
                    name="xgboost",
                    version="local",
                    model_type="xgboost",
                    loaded_from=str(xgb_path),
                    load_time=datetime.now().isoformat(),
                )
                logger.info(f"Loaded XGBoost from {xgb_path}")
            except ImportError:
                logger.warning("XGBoost not installed")

        # Load LSTM
        lstm_path = model_dir / "lstm_model"
        if lstm_path.exists():
            try:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                from tensorflow import keras
                self._models["lstm"] = keras.models.load_model(str(lstm_path))
                self._model_info["lstm"] = ModelInfo(
                    name="lstm",
                    version="local",
                    model_type="lstm",
                    loaded_from=str(lstm_path),
                    load_time=datetime.now().isoformat(),
                )
                logger.info(f"Loaded LSTM from {lstm_path}")
            except ImportError:
                logger.warning("TensorFlow not installed")

        # Load scaler if present
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            self._load_scaler(str(scaler_path))

    def _load_scaler(self, scaler_path: str) -> None:
        """Load the feature scaler."""
        import joblib

        scaler_data = joblib.load(scaler_path)

        if isinstance(scaler_data, dict):
            self._scaler = scaler_data.get("scaler")
            self._feature_columns = scaler_data.get("feature_columns", [])
        else:
            self._scaler = scaler_data

        logger.info(f"Loaded scaler from {scaler_path}")
        logger.info(f"Feature columns: {len(self._feature_columns)}")

    def preprocess(
        self,
        sensor_data: Dict[str, float],
        history: Optional[List[Dict[str, float]]] = None,
    ) -> np.ndarray:
        """
        Preprocess input data for prediction.

        Args:
            sensor_data: Dictionary with sensor readings and op settings
            history: Optional historical readings for feature engineering

        Returns:
            Preprocessed feature array
        """
        import pandas as pd

        # Create DataFrame from input
        df = pd.DataFrame([sensor_data])

        # If history provided, use it for feature engineering
        if history:
            history_df = pd.DataFrame(history + [sensor_data])
            df = history_df

        # Add rolling features if we have enough history
        if len(df) >= max(self.DEFAULT_ROLLING_WINDOWS):
            for col in self.SENSORS_FOR_FEATURES:
                if col in df.columns:
                    for window in self.DEFAULT_ROLLING_WINDOWS:
                        df[f"{col}_rolling_mean_{window}"] = (
                            df[col].rolling(window=window, min_periods=1).mean()
                        )
                        df[f"{col}_rolling_std_{window}"] = (
                            df[col].rolling(window=window, min_periods=1).std()
                        )

        # Add lag features if we have enough history
        if len(df) >= max(self.DEFAULT_LAG_PERIODS):
            for col in self.SENSORS_FOR_FEATURES:
                if col in df.columns:
                    for lag in self.DEFAULT_LAG_PERIODS:
                        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Add delta features
        for col in self.SENSORS_FOR_FEATURES:
            if col in df.columns:
                df[f"{col}_delta"] = df[col].diff()

        # Get the last row (current reading)
        features = df.iloc[-1:].copy()

        # Fill NaN with 0 for missing features
        features = features.fillna(0)

        # Select feature columns or use all numeric
        if self._feature_columns:
            # Ensure all required columns exist
            for col in self._feature_columns:
                if col not in features.columns:
                    features[col] = 0
            features = features[self._feature_columns]
        else:
            # Use all columns except non-features
            exclude = ["unit_id", "time_cycle", "RUL"]
            features = features[[c for c in features.columns if c not in exclude]]

        X = features.values.astype(np.float32)

        # Apply scaler if available
        if self._scaler is not None:
            try:
                X = self._scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        return X

    def predict(
        self,
        sensor_data: Dict[str, float],
        history: Optional[List[Dict[str, float]]] = None,
        return_individual: bool = False,
    ) -> PredictionResult:
        """
        Make a RUL prediction.

        Args:
            sensor_data: Dictionary with sensor readings
            history: Optional historical readings for LSTM
            return_individual: Whether to return individual model predictions

        Returns:
            PredictionResult with RUL and confidence
        """
        with self._lock:
            if not self._is_loaded:
                raise RuntimeError("Models not loaded. Call load_models() first.")

            # Preprocess input
            X = self.preprocess(sensor_data, history)
            self._last_input = X.copy()

            predictions = {}
            weights_used = {}

            # Random Forest prediction
            if "random_forest" in self._models:
                try:
                    rf_pred = float(self._models["random_forest"].predict(X)[0])
                    predictions["random_forest"] = rf_pred
                    weights_used["random_forest"] = self.model_weights.get("random_forest", 0.33)
                except Exception as e:
                    logger.warning(f"RF prediction failed: {e}")

            # XGBoost prediction
            if "xgboost" in self._models:
                try:
                    xgb_pred = float(self._models["xgboost"].predict(X)[0])
                    predictions["xgboost"] = xgb_pred
                    weights_used["xgboost"] = self.model_weights.get("xgboost", 0.33)
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed: {e}")

            # LSTM prediction (requires sequence)
            if "lstm" in self._models:
                try:
                    lstm_pred = self._predict_lstm(X, history)
                    if lstm_pred is not None:
                        predictions["lstm"] = lstm_pred
                        weights_used["lstm"] = self.model_weights.get("lstm", 0.34)
                except Exception as e:
                    logger.warning(f"LSTM prediction failed: {e}")

            if not predictions:
                raise RuntimeError("All model predictions failed")

            # Compute weighted ensemble prediction
            total_weight = sum(weights_used.values())
            ensemble_pred = sum(
                pred * weights_used[name] / total_weight
                for name, pred in predictions.items()
            )

            # Compute confidence based on prediction agreement
            if len(predictions) > 1:
                pred_values = list(predictions.values())
                std = np.std(pred_values)
                mean = np.mean(pred_values)
                # Confidence decreases with higher disagreement
                cv = std / (mean + 1e-6)  # Coefficient of variation
                confidence = max(0.0, min(1.0, 1.0 - cv))
            else:
                confidence = 0.8  # Single model confidence

            # Clip RUL to valid range
            ensemble_pred = max(0, min(125, ensemble_pred))

            # Get model version string
            versions = [f"{k}:v{v.version}" for k, v in self._model_info.items()]
            model_version = ", ".join(versions)

            result = PredictionResult(
                rul=round(ensemble_pred, 2),
                confidence=round(confidence, 3),
                model_version=model_version,
                model_type="ensemble",
                individual_predictions=predictions if return_individual else {},
            )

            self._last_prediction = result
            return result

    def _predict_lstm(
        self,
        X: np.ndarray,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Optional[float]:
        """Make LSTM prediction using sequence buffer."""
        # Update sequence buffer
        self._sequence_buffer.append(X.flatten())

        # Keep only last sequence_length items
        if len(self._sequence_buffer) > self.sequence_length:
            self._sequence_buffer = self._sequence_buffer[-self.sequence_length:]

        # Need full sequence for prediction
        if len(self._sequence_buffer) < self.sequence_length:
            logger.debug(
                f"LSTM buffer: {len(self._sequence_buffer)}/{self.sequence_length}"
            )
            return None

        # Create sequence array
        sequence = np.array(self._sequence_buffer)
        sequence = sequence.reshape(1, self.sequence_length, -1)

        # Predict
        pred = self._models["lstm"].predict(sequence, verbose=0)
        return float(pred[0][0])

    def predict_batch(
        self,
        samples: List[Dict[str, float]],
    ) -> List[PredictionResult]:
        """
        Make predictions for multiple samples.

        Args:
            samples: List of sensor data dictionaries

        Returns:
            List of PredictionResult objects
        """
        results = []

        # Reset sequence buffer for batch
        self._sequence_buffer = []

        for i, sample in enumerate(samples):
            # Use previous samples as history for current
            history = samples[max(0, i - self.sequence_length):i] if i > 0 else None

            try:
                result = self.predict(sample, history=history, return_individual=True)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for sample {i}: {e}")
                # Return a failure result
                results.append(PredictionResult(
                    rul=-1,
                    confidence=0,
                    model_version="error",
                    model_type="error",
                ))

        return results

    def explain_prediction(
        self,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain the last prediction using feature importance.

        Args:
            top_k: Number of top features to return

        Returns:
            Dictionary with explanation data
        """
        with self._lock:
            if self._last_input is None or self._last_prediction is None:
                raise RuntimeError("No prediction to explain")

            explanation = {
                "prediction": self._last_prediction.rul,
                "confidence": self._last_prediction.confidence,
                "top_features": [],
            }

            # Try SHAP if available
            try:
                import shap

                # Use RF for SHAP (most interpretable)
                if "random_forest" in self._models:
                    model = self._models["random_forest"]
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self._last_input)

                    # Get feature importances
                    feature_names = self._feature_columns or [
                        f"feature_{i}" for i in range(self._last_input.shape[1])
                    ]

                    importances = np.abs(shap_values[0])
                    sorted_idx = np.argsort(importances)[::-1][:top_k]

                    explanation["top_features"] = [
                        {
                            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                            "importance": float(importances[i]),
                            "value": float(self._last_input[0, i]),
                            "shap_value": float(shap_values[0, i]),
                        }
                        for i in sorted_idx
                    ]
                    explanation["method"] = "shap"

            except ImportError:
                logger.warning("SHAP not available, using model feature importance")

                # Fall back to model's feature importance
                if "random_forest" in self._models:
                    model = self._models["random_forest"]
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        feature_names = self._feature_columns or [
                            f"feature_{i}" for i in range(len(importances))
                        ]

                        sorted_idx = np.argsort(importances)[::-1][:top_k]

                        explanation["top_features"] = [
                            {
                                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                                "importance": float(importances[i]),
                                "value": float(self._last_input[0, i]) if i < self._last_input.shape[1] else 0,
                            }
                            for i in sorted_idx
                        ]
                        explanation["method"] = "feature_importance"

            return explanation

    def get_health_info(self) -> Dict[str, Any]:
        """Get health and status information."""
        with self._lock:
            return {
                "status": "healthy" if self._is_loaded else "not_ready",
                "models_loaded": list(self._models.keys()),
                "model_count": len(self._models),
                "scaler_loaded": self._scaler is not None,
                "feature_count": len(self._feature_columns),
                "sequence_buffer_size": len(self._sequence_buffer),
                "model_info": {
                    name: {
                        "version": info.version,
                        "loaded_from": info.loaded_from,
                        "load_time": info.load_time,
                    }
                    for name, info in self._model_info.items()
                },
            }

    def clear_sequence_buffer(self) -> None:
        """Clear the LSTM sequence buffer."""
        with self._lock:
            self._sequence_buffer = []
            logger.info("Sequence buffer cleared")


# Singleton predictor instance
_predictor_instance: Optional[Predictor] = None
_predictor_lock = threading.Lock()


def get_predictor() -> Predictor:
    """Get or create the singleton predictor instance."""
    global _predictor_instance

    with _predictor_lock:
        if _predictor_instance is None:
            _predictor_instance = Predictor()
        return _predictor_instance


def initialize_predictor(
    mlflow_tracking_uri: Optional[str] = None,
    local_model_dir: Optional[str] = None,
    scaler_path: Optional[str] = None,
    **kwargs,
) -> Predictor:
    """
    Initialize the singleton predictor with models.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        local_model_dir: Local directory with saved models
        scaler_path: Path to scaler pickle file
        **kwargs: Additional arguments for load_models()

    Returns:
        Initialized Predictor instance
    """
    predictor = get_predictor()

    if not predictor.is_loaded:
        predictor.load_models(
            mlflow_tracking_uri=mlflow_tracking_uri,
            local_model_dir=local_model_dir,
            scaler_path=scaler_path,
            **kwargs,
        )

    return predictor
