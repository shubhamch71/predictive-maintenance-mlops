"""
Model evaluation and comparison utilities.

This module provides the ModelEvaluator class for evaluating trained models,
generating visualization plots, and creating comparison reports.

Example:
    >>> from src.training.evaluate import ModelEvaluator
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.evaluate_model(model, X_test, y_test)
    >>> evaluator.plot_predictions(y_test, y_pred, save_path="predictions.png")
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import settings

# Conditional MLflow import for graceful degradation
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Exception raised for evaluation errors."""
    pass


class ModelEvaluator:
    """
    Evaluate and compare predictive maintenance models.

    This class provides methods for computing regression metrics, generating
    visualization plots, comparing multiple models, and creating evaluation reports.

    Attributes:
        results: Dict storing evaluation results per model
        figures: List of generated figure paths

    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.compute_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.2f}")
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.results: Dict[str, Dict[str, float]] = {}
        self.figures: List[Path] = []

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute regression metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dict with keys: rmse, mae, r2, mape

        Raises:
            EvaluationError: If arrays have incompatible shapes

        Example:
            >>> metrics = evaluator.compute_metrics(y_true, y_pred)
            >>> print(f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise EvaluationError(
                f"Shape mismatch: y_true has {len(y_true)} samples, "
                f"y_pred has {len(y_pred)} samples"
            )

        if len(y_true) == 0:
            raise EvaluationError("Cannot compute metrics on empty arrays")

        # Core metrics
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE: avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            )
        else:
            mape = float('inf')
            logger.warning("MAPE is undefined: all true values are zero")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model with predict() method
            X_test: Test features
            y_test: Test targets
            model_name: Optional name for storing results

        Returns:
            Dict of metrics (rmse, mae, r2, mape)

        Raises:
            EvaluationError: If prediction fails

        Example:
            >>> metrics = evaluator.evaluate_model(model, X_test, y_test, "random_forest")
        """
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            raise EvaluationError(f"Model prediction failed: {e}") from e

        metrics = self.compute_metrics(y_test, y_pred)

        if model_name:
            self.results[model_name] = metrics
            logger.info(f"Evaluated {model_name}: RMSE={metrics['rmse']:.4f}, "
                       f"MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

        return metrics

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Predicted vs Actual RUL",
        log_to_mlflow: bool = False,
    ) -> plt.Figure:
        """
        Create scatter plot of predictions vs actual values.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            save_path: Path to save figure (optional)
            title: Plot title
            log_to_mlflow: Whether to log figure to MLflow

        Returns:
            Matplotlib figure

        Example:
            >>> fig = evaluator.plot_predictions(y_true, y_pred, "predictions.png")
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Compute metrics for annotation
        metrics = self.compute_metrics(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none', s=30)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Labels and title
        ax.set_xlabel('Actual RUL (cycles)', fontsize=12)
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=12)
        ax.set_title(title, fontsize=14)

        # Metrics annotation
        textstr = f'RMSE: {metrics["rmse"]:.2f}\nMAE: {metrics["mae"]:.2f}\nR²: {metrics["r2"]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.figures.append(save_path)
            logger.info(f"Saved predictions plot to {save_path}")

        # Log to MLflow
        if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "plots")
                logger.info("Logged predictions plot to MLflow")

        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Residual Analysis",
        log_to_mlflow: bool = False,
    ) -> plt.Figure:
        """
        Create residual distribution plot.

        Creates a 2-panel figure with:
        - Top: Histogram of residuals with KDE
        - Bottom: Residuals vs Predicted values

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            save_path: Path to save figure (optional)
            title: Plot title
            log_to_mlflow: Whether to log figure to MLflow

        Returns:
            Matplotlib figure

        Example:
            >>> fig = evaluator.plot_residuals(y_true, y_pred, "residuals.png")
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Top: Histogram of residuals
        ax1 = axes[0]
        ax1.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Residual')
        ax1.axvline(x=residuals.mean(), color='orange', linestyle='-', lw=2,
                   label=f'Mean: {residuals.mean():.2f}')
        ax1.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Distribution of Residuals', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Residuals vs Predicted
        ax2 = axes[1]
        ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='none', s=30)
        ax2.axhline(y=0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted RUL (cycles)', fontsize=11)
        ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
        ax2.set_title('Residuals vs Predicted Values', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add std bands
        std_residual = residuals.std()
        ax2.axhline(y=std_residual, color='gray', linestyle=':', alpha=0.7, label=f'+1 std ({std_residual:.2f})')
        ax2.axhline(y=-std_residual, color='gray', linestyle=':', alpha=0.7, label=f'-1 std ({-std_residual:.2f})')
        ax2.legend()

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        # Save figure
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.figures.append(save_path)
            logger.info(f"Saved residuals plot to {save_path}")

        # Log to MLflow
        if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "plots")
                logger.info("Logged residuals plot to MLflow")

        return fig

    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_test_sequences: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models side-by-side.

        Args:
            models_dict: Dict mapping model_name to model instance
            X_test: Test features (2D for RF/XGB)
            y_test: Test targets
            X_test_sequences: Optional 3D sequences for LSTM
            save_path: Path to save comparison plot

        Returns:
            Dict mapping model_name to metrics

        Note:
            For LSTM models, provide X_test_sequences with shape
            (n_samples, sequence_length, n_features).

        Example:
            >>> results = evaluator.compare_models(
            ...     {"rf": rf_model, "xgb": xgb_model},
            ...     X_test, y_test
            ... )
        """
        comparison_results: Dict[str, Dict[str, float]] = {}

        for model_name, model in models_dict.items():
            try:
                # Use sequences for LSTM if provided
                if "lstm" in model_name.lower() and X_test_sequences is not None:
                    X_input = X_test_sequences
                else:
                    X_input = X_test

                metrics = self.evaluate_model(model, X_input, y_test, model_name)
                comparison_results[model_name] = metrics

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                comparison_results[model_name] = {
                    "rmse": float('inf'),
                    "mae": float('inf'),
                    "r2": float('-inf'),
                    "mape": float('inf'),
                    "error": str(e),
                }

        # Generate comparison bar chart
        if save_path and len(comparison_results) > 0:
            self.plot_comparison_bar_chart(comparison_results, save_path=save_path)

        return comparison_results

    def plot_comparison_bar_chart(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        log_to_mlflow: bool = False,
    ) -> plt.Figure:
        """
        Create bar chart comparing models across metrics.

        Args:
            metrics_dict: Dict of model metrics
            metrics_to_plot: Which metrics to include (default: ["rmse", "mae"])
            save_path: Path to save figure
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Matplotlib figure

        Example:
            >>> fig = evaluator.plot_comparison_bar_chart(
            ...     {"rf": {...}, "xgb": {...}},
            ...     metrics_to_plot=["rmse", "mae"]
            ... )
        """
        if metrics_to_plot is None:
            metrics_to_plot = ["rmse", "mae"]

        # Filter out models with errors
        valid_models = {k: v for k, v in metrics_dict.items() if "error" not in v}

        if not valid_models:
            logger.warning("No valid models to compare")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No valid models to compare", ha='center', va='center')
            return fig

        model_names = list(valid_models.keys())
        n_models = len(model_names)
        n_metrics = len(metrics_to_plot)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_models)
        width = 0.8 / n_metrics

        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

        for i, metric in enumerate(metrics_to_plot):
            values = [valid_models[model].get(metric, 0) for model in model_names]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.upper(), color=colors[i])

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([name.upper() for name in model_names])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.figures.append(save_path)
            logger.info(f"Saved comparison plot to {save_path}")

        # Log to MLflow
        if log_to_mlflow and MLFLOW_AVAILABLE and mlflow.active_run():
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "plots")
                logger.info("Logged comparison plot to MLflow")

        return fig

    def generate_evaluation_report(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Union[str, Path],
        dataset: str = "FD001",
        include_timestamp: bool = True,
    ) -> Path:
        """
        Generate JSON evaluation report.

        Args:
            metrics_dict: Dict of model metrics
            save_path: Path to save JSON report
            dataset: Dataset name for report
            include_timestamp: Whether to include timestamp in report

        Returns:
            Path to saved report

        Report Structure:
            {
                "timestamp": "2024-01-15T10:30:00",
                "dataset": "FD001",
                "models": {
                    "random_forest": {"rmse": 12.5, "mae": 9.2, ...},
                    "xgboost": {"rmse": 11.8, "mae": 8.7, ...}
                },
                "best_model": {
                    "name": "xgboost",
                    "metric": "rmse",
                    "value": 11.8
                }
            }

        Example:
            >>> report_path = evaluator.generate_evaluation_report(
            ...     metrics_dict, "reports/evaluation.json"
            ... )
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Filter valid models (no errors)
        valid_models = {
            k: {m: v for m, v in metrics.items() if m != "error"}
            for k, metrics in metrics_dict.items()
            if "error" not in metrics or metrics.get("rmse") != float('inf')
        }

        # Find best model by RMSE
        best_model_name = None
        best_rmse = float('inf')

        for model_name, metrics in valid_models.items():
            rmse = metrics.get("rmse", float('inf'))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name

        # Build report
        report = {
            "dataset": dataset,
            "models": valid_models,
            "best_model": {
                "name": best_model_name,
                "metric": "rmse",
                "value": best_rmse,
            } if best_model_name else None,
            "summary": {
                "total_models_evaluated": len(metrics_dict),
                "successful_evaluations": len(valid_models),
            }
        }

        if include_timestamp:
            report["timestamp"] = datetime.now().isoformat()

        # Write JSON
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved evaluation report to {save_path}")

        return save_path

    def clear_results(self) -> None:
        """Clear stored results and figures."""
        self.results.clear()
        self.figures.clear()
        logger.info("Cleared evaluation results")
