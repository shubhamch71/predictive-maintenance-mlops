"""
Random Forest model for Remaining Useful Life (RUL) prediction.

This module provides a production-ready Random Forest regressor with
MLflow integration for experiment tracking, hyperparameter logging,
and model artifact storage.

Example:
    >>> from src.models.random_forest import RandomForestModel
    >>> from src.data.loader import DataLoader
    >>> from src.data.preprocessor import FeatureEngineer
    >>>
    >>> # Prepare data
    >>> loader = DataLoader()
    >>> train_df = loader.load_train("FD001")
    >>> fe = FeatureEngineer()
    >>> X_train, y_train = fe.full_pipeline(train_df, fit=True)
    >>>
    >>> # Train model
    >>> model = RandomForestModel()
    >>> model.train(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)

Hyperparameter Examples:
    Default (balanced):
        n_estimators=100, max_depth=None, min_samples_split=2

    High accuracy (slower):
        n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2

    Fast training:
        n_estimators=50, max_depth=10, min_samples_split=10
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import settings

# Conditional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Exception raised for model-related errors."""
    pass


class RandomForestModel:
    """
    Random Forest regressor for RUL prediction with MLflow integration.

    This class wraps scikit-learn's RandomForestRegressor with additional
    functionality for MLflow experiment tracking, feature importance analysis,
    and model persistence.

    Attributes:
        model: The underlying RandomForestRegressor instance
        hyperparameters: Dictionary of model hyperparameters
        feature_names: List of feature names (set during training)
        is_trained: Whether the model has been trained
        metrics: Dictionary of evaluation metrics from last training

    Example:
        >>> model = RandomForestModel(hyperparameters={
        ...     'n_estimators': 200,
        ...     'max_depth': 15,
        ...     'min_samples_split': 5
        ... })
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> print(model.metrics)
    """

    # Default hyperparameters
    DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
    }

    def __init__(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the Random Forest model.

        Args:
            hyperparameters: Model hyperparameters. Missing keys use defaults.
            feature_names: Optional list of feature names for importance tracking.
        """
        # Merge with defaults
        self.hyperparameters = {**self.DEFAULT_HYPERPARAMETERS}
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        self.model = RandomForestRegressor(**self.hyperparameters)
        self.feature_names = feature_names
        self.is_trained = False
        self.metrics: Dict[str, float] = {}

        logger.info(f"Initialized RandomForestModel with {self.hyperparameters}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        log_to_mlflow: bool = True,
        experiment_name: Optional[str] = None,
        run_name: str = "random_forest",
    ) -> Dict[str, float]:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training targets of shape (n_samples,)
            X_val: Validation features (optional, for evaluation)
            y_val: Validation targets (optional, for evaluation)
            feature_names: Names of features for importance tracking
            log_to_mlflow: Whether to log to MLflow
            experiment_name: MLflow experiment name (uses config default if None)
            run_name: Name for the MLflow run

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ModelError: If training fails
        """
        if feature_names:
            self.feature_names = feature_names

        logger.info(f"Training Random Forest on {X_train.shape[0]} samples")

        try:
            # Train the model
            self.model.fit(X_train, y_train)
            self.is_trained = True

            # Calculate training metrics
            train_pred = self.model.predict(X_train)
            self.metrics = self._calculate_metrics(y_train, train_pred, prefix="train_")

            # Calculate validation metrics if provided
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred, prefix="val_")
                self.metrics.update(val_metrics)

            logger.info(f"Training complete. Metrics: {self.metrics}")

            # Log to MLflow
            if log_to_mlflow and MLFLOW_AVAILABLE:
                self._log_to_mlflow(
                    X_val, y_val,
                    experiment_name=experiment_name,
                    run_name=run_name,
                )

            return self.metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ModelError(f"Failed to train Random Forest model: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted RUL values of shape (n_samples,)

        Raises:
            ModelError: If model is not trained
        """
        if not self.is_trained:
            raise ModelError("Model must be trained before making predictions")

        try:
            predictions = self.model.predict(X)
            # Clip predictions to non-negative values (RUL can't be negative)
            predictions = np.clip(predictions, 0, None)
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get feature importance scores.

        Args:
            feature_names: Names of features (uses stored names if None)
            top_n: Return only top N features (all if None)

        Returns:
            Tuple of (feature_names, importance_scores) sorted by importance

        Raises:
            ModelError: If model is not trained
        """
        if not self.is_trained:
            raise ModelError("Model must be trained to get feature importance")

        importances = self.model.feature_importances_

        # Use provided or stored feature names, or generate defaults
        names = feature_names or self.feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(importances))]

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_names = [names[i] for i in indices]
        sorted_importances = importances[indices]

        # Limit to top_n if specified
        if top_n:
            sorted_names = sorted_names[:top_n]
            sorted_importances = sorted_importances[:top_n]

        return sorted_names, sorted_importances

    def plot_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a feature importance plot.

        Args:
            feature_names: Names of features
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure object
        """
        names, importances = self.get_feature_importance(feature_names, top_n)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))

        ax.barh(y_pos, importances, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances - Random Forest")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {save_path}")

        return fig

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model

        Raises:
            ModelError: If model is not trained
        """
        if not self.is_trained:
            raise ModelError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self.model,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }

        joblib.dump(save_data, path)
        logger.info(f"Saved model to {path}")

    def load(self, path: Union[str, Path]) -> "RandomForestModel":
        """
        Load the model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_data = joblib.load(path)

        self.model = save_data["model"]
        self.hyperparameters = save_data["hyperparameters"]
        self.feature_names = save_data["feature_names"]
        self.metrics = save_data["metrics"]
        self.is_trained = save_data["is_trained"]

        logger.info(f"Loaded model from {path}")
        return self

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        return {
            f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}r2": float(r2_score(y_true, y_pred)),
        }

    def _log_to_mlflow(
        self,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        experiment_name: Optional[str] = None,
        run_name: str = "random_forest",
    ) -> None:
        """
        Log training results to MLflow.

        Args:
            X_val: Validation features
            y_val: Validation targets
            experiment_name: MLflow experiment name
            run_name: Name for the run
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping logging")
            return

        try:
            # Set experiment
            exp_name = experiment_name or settings.mlflow.experiment_name
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=run_name):
                # Log hyperparameters
                mlflow.log_params(self.hyperparameters)

                # Log metrics
                mlflow.log_metrics(self.metrics)

                # Log model
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    registered_model_name=f"{exp_name}_random_forest",
                )

                # Log feature importance plot
                if self.feature_names:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        plot_path = Path(tmpdir) / "feature_importance.png"
                        self.plot_feature_importance(save_path=plot_path)
                        mlflow.log_artifact(str(plot_path))
                        plt.close()

                # Log feature importances as artifact
                if self.feature_names:
                    names, importances = self.get_feature_importance()
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False
                    ) as f:
                        for name, imp in zip(names, importances):
                            f.write(f"{name}: {imp:.6f}\n")
                        mlflow.log_artifact(f.name, "feature_importance")

                logger.info(f"Logged to MLflow experiment: {exp_name}")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.hyperparameters.copy()

    def set_params(self, **params: Any) -> "RandomForestModel":
        """Set model hyperparameters."""
        self.hyperparameters.update(params)
        self.model.set_params(**params)
        return self
