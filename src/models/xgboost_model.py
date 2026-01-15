"""
XGBoost model for Remaining Useful Life (RUL) prediction.

This module provides a production-ready XGBoost regressor with MLflow
integration, early stopping, and learning curve logging.

Example:
    >>> from src.models.xgboost_model import XGBoostModel
    >>> model = XGBoostModel()
    >>> model.train(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)

Hyperparameter Examples:
    Default (balanced):
        n_estimators=100, max_depth=6, learning_rate=0.1

    High accuracy (slower, risk of overfitting):
        n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8

    Fast training:
        n_estimators=50, max_depth=4, learning_rate=0.2

    Regularized (prevent overfitting):
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import settings

# Conditional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Exception raised for model-related errors."""
    pass


class XGBoostModel:
    """
    XGBoost regressor for RUL prediction with MLflow integration.

    This class wraps XGBoost's XGBRegressor with additional functionality
    for early stopping, learning curve tracking, MLflow logging, and
    model persistence.

    Attributes:
        model: The underlying XGBRegressor instance
        hyperparameters: Dictionary of model hyperparameters
        feature_names: List of feature names
        is_trained: Whether the model has been trained
        metrics: Dictionary of evaluation metrics
        learning_curve: Training history (loss per iteration)

    Example:
        >>> model = XGBoostModel(hyperparameters={
        ...     'n_estimators': 200,
        ...     'max_depth': 8,
        ...     'learning_rate': 0.05,
        ...     'early_stopping_rounds': 20
        ... })
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> print(model.metrics)
    """

    # Default hyperparameters
    DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "gamma": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }

    # Early stopping defaults
    DEFAULT_EARLY_STOPPING_ROUNDS = 20

    def __init__(
        self,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> None:
        """
        Initialize the XGBoost model.

        Args:
            hyperparameters: Model hyperparameters. Missing keys use defaults.
            feature_names: Optional list of feature names.
            early_stopping_rounds: Number of rounds for early stopping.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        # Merge with defaults
        self.hyperparameters = {**self.DEFAULT_HYPERPARAMETERS}
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        self.early_stopping_rounds = early_stopping_rounds
        self.model = xgb.XGBRegressor(**self.hyperparameters)
        self.feature_names = feature_names
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
        self.learning_curve: Dict[str, List[float]] = {}
        self.best_iteration: Optional[int] = None

        logger.info(f"Initialized XGBoostModel with {self.hyperparameters}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        log_to_mlflow: bool = True,
        experiment_name: Optional[str] = None,
        run_name: str = "xgboost",
    ) -> Dict[str, float]:
        """
        Train the XGBoost model with early stopping.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training targets of shape (n_samples,)
            X_val: Validation features (required for early stopping)
            y_val: Validation targets (required for early stopping)
            feature_names: Names of features
            log_to_mlflow: Whether to log to MLflow
            experiment_name: MLflow experiment name
            run_name: Name for the MLflow run

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ModelError: If training fails
        """
        if feature_names:
            self.feature_names = feature_names

        logger.info(f"Training XGBoost on {X_train.shape[0]} samples")

        try:
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            # Train with early stopping
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )

            self.is_trained = True

            # Get best iteration (only available when early stopping is used)
            try:
                self.best_iteration = self.model.best_iteration
            except AttributeError:
                # Early stopping was not triggered
                self.best_iteration = self.hyperparameters.get("n_estimators", 100)

            # Extract learning curve
            self._extract_learning_curve()

            # Calculate training metrics
            train_pred = self.model.predict(X_train)
            self.metrics = self._calculate_metrics(y_train, train_pred, prefix="train_")

            # Calculate validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred, prefix="val_")
                self.metrics.update(val_metrics)

            # Add best iteration to metrics
            if self.best_iteration:
                self.metrics["best_iteration"] = self.best_iteration

            logger.info(f"Training complete. Best iteration: {self.best_iteration}")
            logger.info(f"Metrics: {self.metrics}")

            # Log to MLflow
            if log_to_mlflow and MLFLOW_AVAILABLE:
                self._log_to_mlflow(
                    experiment_name=experiment_name,
                    run_name=run_name,
                )

            return self.metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ModelError(f"Failed to train XGBoost model: {e}")

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
            # Use best iteration if available
            if self.best_iteration:
                predictions = self.model.predict(
                    X, iteration_range=(0, self.best_iteration + 1)
                )
            else:
                predictions = self.model.predict(X)

            # Clip to non-negative
            predictions = np.clip(predictions, 0, None)
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        feature_names: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            feature_names: Names of features
            top_n: Return only top N features

        Returns:
            Tuple of (feature_names, importance_scores) sorted by importance

        Raises:
            ModelError: If model is not trained
        """
        if not self.is_trained:
            raise ModelError("Model must be trained to get feature importance")

        # Get importance scores
        booster = self.model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)

        # Use provided or stored feature names
        names = feature_names or self.feature_names
        n_features = self.model.n_features_in_

        if names is None:
            names = [f"f{i}" for i in range(n_features)]

        # Build importance array
        importances = []
        feature_list = []

        for i, name in enumerate(names):
            # XGBoost uses f0, f1, etc. as default feature names
            key = f"f{i}"
            imp = importance_dict.get(key, 0)
            importances.append(imp)
            feature_list.append(name)

        importances = np.array(importances)

        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_list[i] for i in indices]
        sorted_importances = importances[indices]

        if top_n:
            sorted_names = sorted_names[:top_n]
            sorted_importances = sorted_importances[:top_n]

        return sorted_names, sorted_importances

    def plot_learning_curve(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot the learning curves (training and validation loss).

        Args:
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure object
        """
        if not self.learning_curve:
            raise ModelError("No learning curve available. Train the model first.")

        fig, ax = plt.subplots(figsize=figsize)

        for label, values in self.learning_curve.items():
            ax.plot(values, label=label)

        if self.best_iteration:
            ax.axvline(
                x=self.best_iteration,
                color="r",
                linestyle="--",
                label=f"Best iteration ({self.best_iteration})",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.set_title("XGBoost Learning Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved learning curve plot to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a feature importance plot.

        Args:
            importance_type: Type of importance
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure object
        """
        names, importances = self.get_feature_importance(
            importance_type=importance_type, top_n=top_n
        )

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))

        ax.barh(y_pos, importances, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(f"Feature Importance ({importance_type})")
        ax.set_title(f"Top {top_n} Feature Importances - XGBoost")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

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

        # Save XGBoost model in native format
        model_path = path.with_suffix(".json")
        self.model.save_model(model_path)

        # Save metadata
        metadata = {
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "learning_curve": self.learning_curve,
            "best_iteration": self.best_iteration,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {model_path}")

    def load(self, path: Union[str, Path]) -> "XGBoostModel":
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
        model_path = path.with_suffix(".json")
        metadata_path = path.with_suffix(".meta.json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.hyperparameters = metadata.get("hyperparameters", {})
            self.feature_names = metadata.get("feature_names")
            self.metrics = metadata.get("metrics", {})
            self.learning_curve = metadata.get("learning_curve", {})
            self.best_iteration = metadata.get("best_iteration")
            self.early_stopping_rounds = metadata.get(
                "early_stopping_rounds", self.DEFAULT_EARLY_STOPPING_ROUNDS
            )

        self.is_trained = True
        logger.info(f"Loaded model from {model_path}")

        return self

    def _extract_learning_curve(self) -> None:
        """Extract learning curve from training results."""
        try:
            evals_result = self.model.evals_result()

            if evals_result:
                for i, (key, metrics) in enumerate(evals_result.items()):
                    label = "train" if i == 0 else "validation"
                    for metric_name, values in metrics.items():
                        self.learning_curve[f"{label}_{metric_name}"] = values

        except Exception as e:
            logger.warning(f"Failed to extract learning curve: {e}")

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}r2": float(r2_score(y_true, y_pred)),
        }

    def _log_to_mlflow(
        self,
        experiment_name: Optional[str] = None,
        run_name: str = "xgboost",
    ) -> None:
        """Log training results to MLflow."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping logging")
            return

        try:
            exp_name = experiment_name or settings.mlflow.experiment_name
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=run_name):
                # Log hyperparameters
                mlflow.log_params(self.hyperparameters)
                mlflow.log_param("early_stopping_rounds", self.early_stopping_rounds)

                # Log metrics
                mlflow.log_metrics(self.metrics)

                # Log learning curves as metrics over steps
                for curve_name, values in self.learning_curve.items():
                    for step, value in enumerate(values):
                        mlflow.log_metric(curve_name, value, step=step)

                # Log model
                mlflow.xgboost.log_model(
                    self.model,
                    "model",
                    registered_model_name=f"{exp_name}_xgboost",
                )

                # Log learning curve plot
                with tempfile.TemporaryDirectory() as tmpdir:
                    if self.learning_curve:
                        plot_path = Path(tmpdir) / "learning_curve.png"
                        self.plot_learning_curve(save_path=plot_path)
                        mlflow.log_artifact(str(plot_path))
                        plt.close()

                    # Log feature importance plot
                    if self.feature_names:
                        imp_path = Path(tmpdir) / "feature_importance.png"
                        self.plot_feature_importance(save_path=imp_path)
                        mlflow.log_artifact(str(imp_path))
                        plt.close()

                logger.info(f"Logged to MLflow experiment: {exp_name}")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.hyperparameters.copy()

    def set_params(self, **params: Any) -> "XGBoostModel":
        """Set model hyperparameters."""
        self.hyperparameters.update(params)
        self.model.set_params(**params)
        return self
