"""
Training orchestrator for predictive maintenance models.

This module provides a unified training pipeline that orchestrates
training of Random Forest, XGBoost, and LSTM models with MLflow
experiment tracking.

Example:
    # Train all models
    python -m src.training.train --experiment-name "pm-experiment" --dataset FD001

    # Train specific model
    python -m src.training.train --model-type rf --dataset FD001

    # Train without MLflow
    python -m src.training.train --model-type xgb --no-mlflow
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings, AVAILABLE_DATASETS
from src.data.loader import DataLoader
from src.data.preprocessor import FeatureEngineer
from src.training.evaluate import ModelEvaluator

# Conditional model imports for graceful degradation
try:
    from src.models.random_forest import RandomForestModel
    RF_AVAILABLE = True
except ImportError:
    RandomForestModel = None
    RF_AVAILABLE = False

try:
    from src.models.xgboost_model import XGBoostModel
    XGB_AVAILABLE = True
except ImportError:
    XGBoostModel = None
    XGB_AVAILABLE = False

try:
    from src.models.lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTMModel = None
    LSTM_AVAILABLE = False

# Conditional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Base exception for training errors."""
    pass


class DataPreparationError(TrainingError):
    """Raised when data preparation fails."""
    pass


class ModelTrainingError(TrainingError):
    """Raised when model training fails."""
    pass


class MLflowError(TrainingError):
    """Raised when MLflow operations fail."""
    pass


class TrainingOrchestrator:
    """
    Orchestrates training of multiple ML models with MLflow tracking.

    This class provides a unified interface for loading data, preprocessing,
    training models (Random Forest, XGBoost, LSTM), and tracking experiments
    with MLflow.

    Attributes:
        experiment_name: MLflow experiment name
        dataset: Dataset to train on (FD001-FD004)
        data_path: Optional custom data path
        feature_engineer: FeatureEngineer instance
        evaluator: ModelEvaluator instance
        trained_models: Dict of trained model instances
        results: Dict of training results per model

    Example:
        >>> orchestrator = TrainingOrchestrator(experiment_name="my-experiment")
        >>> results = orchestrator.train_all_models()
        >>> best_type, best_model, metrics = orchestrator.select_best_model()
    """

    MODEL_TYPES = ["rf", "xgb", "lstm"]

    def __init__(
        self,
        experiment_name: str = "predictive-maintenance",
        dataset: str = "FD001",
        data_path: Optional[Path] = None,
        use_mlflow: bool = True,
    ) -> None:
        """
        Initialize the training orchestrator.

        Args:
            experiment_name: MLflow experiment name
            dataset: Dataset to train on (FD001, FD002, FD003, FD004)
            data_path: Optional custom path to data directory
            use_mlflow: Whether to use MLflow tracking
        """
        self.experiment_name = experiment_name
        self.dataset = dataset
        self.data_path = Path(data_path) if data_path else settings.paths.raw_data_dir
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()

        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

        # Data storage
        self._train_df: Optional[pd.DataFrame] = None
        self._X_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_train_seq: Optional[np.ndarray] = None
        self._X_val_seq: Optional[np.ndarray] = None
        self._y_train_seq: Optional[np.ndarray] = None
        self._y_val_seq: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

        if dataset not in AVAILABLE_DATASETS:
            raise ValueError(f"Dataset must be one of {AVAILABLE_DATASETS}, got {dataset}")

        logger.info(f"Initialized TrainingOrchestrator: experiment={experiment_name}, "
                   f"dataset={dataset}, mlflow={self.use_mlflow}")

    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and run feature engineering pipeline.

        Returns:
            Tuple of (X_train, X_val, y_train, y_val) as 2D arrays

        Raises:
            DataPreparationError: If data loading or preprocessing fails
        """
        logger.info(f"Loading data from {self.data_path} for dataset {self.dataset}")

        try:
            # Load raw data
            loader = DataLoader(data_dir=self.data_path)
            train_df, _, _ = loader.load_all(self.dataset)
            self._train_df = train_df

            logger.info(f"Loaded training data: {len(train_df)} samples, "
                       f"{train_df['unit_id'].nunique()} units")

            # Run feature engineering pipeline
            X, y = self.feature_engineer.full_pipeline(train_df, fit=True)
            self._feature_names = self.feature_engineer.feature_columns

            logger.info(f"Feature engineering complete: {X.shape[1]} features")

            # Train/validation split
            self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
                X, y,
                test_size=settings.model.validation_size,
                random_state=settings.model.random_state,
            )

            logger.info(f"Data split: train={len(self._X_train)}, val={len(self._X_val)}")

            return self._X_train, self._X_val, self._y_train, self._y_val

        except Exception as e:
            raise DataPreparationError(f"Failed to load and preprocess data: {e}") from e

    def prepare_lstm_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare 3D sequences for LSTM from processed DataFrame.

        Uses create_sequences_by_unit to respect unit boundaries,
        preventing data leakage between different engine units.

        Returns:
            Tuple of (X_train_seq, X_val_seq, y_train_seq, y_val_seq)

        Raises:
            DataPreparationError: If data hasn't been loaded or sequence creation fails
        """
        if self._train_df is None:
            raise DataPreparationError("Data not loaded. Call load_and_preprocess_data() first.")

        logger.info("Preparing LSTM sequences...")

        try:
            # Get processed DataFrame with all features
            train_df = self._train_df.copy()

            # Create RUL target
            train_df = self.feature_engineer.create_rul_target(train_df)

            # Create all features
            train_df = self.feature_engineer.create_rolling_features(train_df)
            train_df = self.feature_engineer.create_lag_features(train_df)
            train_df = self.feature_engineer.create_rate_of_change(train_df)

            # Drop NaN rows
            train_df = train_df.dropna()

            # Get feature columns (exclude non-features)
            exclude_cols = ['unit_id', 'time_cycle', 'RUL', 'max_cycle']
            feature_columns = [c for c in train_df.columns if c not in exclude_cols]

            # Create sequences by unit
            sequence_length = settings.features.sequence_length
            X_seq, y_seq = self.feature_engineer.create_sequences_by_unit(
                df=train_df,
                feature_columns=feature_columns,
                target_column="RUL",
                sequence_length=sequence_length,
            )

            logger.info(f"Created sequences: shape={X_seq.shape}")

            # Split sequences
            self._X_train_seq, self._X_val_seq, self._y_train_seq, self._y_val_seq = train_test_split(
                X_seq, y_seq,
                test_size=settings.model.validation_size,
                random_state=settings.model.random_state,
            )

            logger.info(f"LSTM data split: train={len(self._X_train_seq)}, val={len(self._X_val_seq)}")

            return self._X_train_seq, self._X_val_seq, self._y_train_seq, self._y_val_seq

        except Exception as e:
            raise DataPreparationError(f"Failed to prepare LSTM data: {e}") from e

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_mlflow: bool = False,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            log_to_mlflow: Whether to log to MLflow (handled by model)

        Returns:
            Tuple of (trained model, metrics dict)

        Raises:
            ModelTrainingError: If training fails or model unavailable
        """
        if not RF_AVAILABLE:
            raise ModelTrainingError("RandomForestModel not available. Check scikit-learn installation.")

        logger.info("Training Random Forest model...")

        try:
            model = RandomForestModel()
            metrics = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=self._feature_names,
                log_to_mlflow=log_to_mlflow,
                experiment_name=self.experiment_name,
                run_name="random_forest",
            )

            self.trained_models["rf"] = model
            logger.info(f"Random Forest trained: val_rmse={metrics.get('val_rmse', 'N/A'):.4f}")

            return model, metrics

        except Exception as e:
            raise ModelTrainingError(f"Random Forest training failed: {e}") from e

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_mlflow: bool = False,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            log_to_mlflow: Whether to log to MLflow (handled by model)

        Returns:
            Tuple of (trained model, metrics dict)

        Raises:
            ModelTrainingError: If training fails or model unavailable
        """
        if not XGB_AVAILABLE:
            raise ModelTrainingError("XGBoostModel not available. Check xgboost installation.")

        logger.info("Training XGBoost model...")

        try:
            model = XGBoostModel()
            metrics = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=self._feature_names,
                log_to_mlflow=log_to_mlflow,
                experiment_name=self.experiment_name,
                run_name="xgboost",
            )

            self.trained_models["xgb"] = model
            logger.info(f"XGBoost trained: val_rmse={metrics.get('val_rmse', 'N/A'):.4f}, "
                       f"best_iteration={metrics.get('best_iteration', 'N/A')}")

            return model, metrics

        except Exception as e:
            raise ModelTrainingError(f"XGBoost training failed: {e}") from e

    def train_lstm(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
        log_to_mlflow: bool = False,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train LSTM model with 3D input.

        Args:
            X_train_seq: Training sequences (3D: samples, timesteps, features)
            y_train_seq: Training targets
            X_val_seq: Validation sequences
            y_val_seq: Validation targets
            log_to_mlflow: Whether to log to MLflow (handled by model)

        Returns:
            Tuple of (trained model, metrics dict)

        Raises:
            ModelTrainingError: If training fails or model unavailable
        """
        if not LSTM_AVAILABLE:
            raise ModelTrainingError("LSTMModel not available. Check tensorflow installation.")

        logger.info("Training LSTM model...")

        try:
            model = LSTMModel()
            metrics = model.train(
                X_train=X_train_seq,
                y_train=y_train_seq,
                X_val=X_val_seq,
                y_val=y_val_seq,
                epochs=settings.model.epochs,
                batch_size=settings.model.batch_size,
                log_to_mlflow=log_to_mlflow,
                experiment_name=self.experiment_name,
                run_name="lstm",
            )

            self.trained_models["lstm"] = model
            logger.info(f"LSTM trained: val_rmse={metrics.get('val_rmse', 'N/A'):.4f}, "
                       f"epochs_trained={metrics.get('epochs_trained', 'N/A')}")

            return model, metrics

        except Exception as e:
            raise ModelTrainingError(f"LSTM training failed: {e}") from e

    def train_all_models(
        self,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train specified models (or all if None).

        Creates MLflow parent run with nested child runs for each model.

        Args:
            model_types: List of model types to train (["rf", "xgb", "lstm"])
                        If None, trains all models.

        Returns:
            Dict mapping model_type to {model, metrics, run_id}

        Raises:
            DataPreparationError: If data loading fails
            ModelTrainingError: If all model trainings fail
        """
        if model_types is None:
            model_types = self.MODEL_TYPES.copy()

        # Validate model types
        for mt in model_types:
            if mt not in self.MODEL_TYPES:
                raise ValueError(f"Invalid model type '{mt}'. Must be one of {self.MODEL_TYPES}")

        # Load and preprocess data
        X_train, X_val, y_train, y_val = self.load_and_preprocess_data()

        # Prepare LSTM data if needed
        if "lstm" in model_types:
            self.prepare_lstm_data()

        results: Dict[str, Dict[str, Any]] = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Training with MLflow parent run
        if self.use_mlflow:
            try:
                mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
                mlflow.set_experiment(self.experiment_name)

                with mlflow.start_run(run_name=f"training_run_{timestamp}") as parent_run:
                    parent_run_id = parent_run.info.run_id
                    logger.info(f"Started MLflow parent run: {parent_run_id}")

                    # Log dataset info
                    mlflow.log_param("dataset", self.dataset)
                    mlflow.log_param("model_types", model_types)
                    mlflow.log_param("n_train_samples", len(X_train))
                    mlflow.log_param("n_val_samples", len(X_val))
                    mlflow.log_param("n_features", X_train.shape[1])

                    # Train each model as nested run
                    results = self._train_models_with_mlflow(
                        model_types, X_train, y_train, X_val, y_val, parent_run_id
                    )

                    # Log best model info to parent
                    if results:
                        best_type, _, best_metrics = self.select_best_model()
                        mlflow.set_tag("best_model", best_type)
                        mlflow.log_metric("best_val_rmse", best_metrics.get("val_rmse", float('inf')))

            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}. Training without MLflow.")
                results = self._train_models_without_mlflow(
                    model_types, X_train, y_train, X_val, y_val
                )
        else:
            results = self._train_models_without_mlflow(
                model_types, X_train, y_train, X_val, y_val
            )

        self.results = results

        if not results:
            raise ModelTrainingError("All model trainings failed")

        return results

    def _train_models_with_mlflow(
        self,
        model_types: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        parent_run_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Train models with MLflow nested runs."""
        results = {}

        for model_type in model_types:
            try:
                with mlflow.start_run(run_name=model_type, nested=True) as child_run:
                    run_id = child_run.info.run_id
                    logger.info(f"Training {model_type} in nested run {run_id}")

                    if model_type == "rf":
                        model, metrics = self.train_random_forest(
                            X_train, y_train, X_val, y_val, log_to_mlflow=False
                        )
                        # Log manually since we're in nested run
                        mlflow.log_params(model.get_params())
                        mlflow.log_metrics(metrics)

                    elif model_type == "xgb":
                        model, metrics = self.train_xgboost(
                            X_train, y_train, X_val, y_val, log_to_mlflow=False
                        )
                        mlflow.log_params(model.get_params())
                        mlflow.log_metrics(metrics)

                    elif model_type == "lstm":
                        model, metrics = self.train_lstm(
                            self._X_train_seq, self._y_train_seq,
                            self._X_val_seq, self._y_val_seq,
                            log_to_mlflow=False
                        )
                        # Convert tuple to list for MLflow
                        params = model.get_params()
                        if "lstm_units" in params and isinstance(params["lstm_units"], tuple):
                            params["lstm_units"] = list(params["lstm_units"])
                        mlflow.log_params(params)
                        mlflow.log_metrics(metrics)

                    results[model_type] = {
                        "model": model,
                        "metrics": metrics,
                        "run_id": run_id,
                    }

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                results[model_type] = {
                    "model": None,
                    "metrics": {"error": str(e)},
                    "run_id": None,
                }

        return results

    def _train_models_without_mlflow(
        self,
        model_types: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Train models without MLflow."""
        results = {}

        for model_type in model_types:
            try:
                if model_type == "rf":
                    model, metrics = self.train_random_forest(
                        X_train, y_train, X_val, y_val, log_to_mlflow=False
                    )
                elif model_type == "xgb":
                    model, metrics = self.train_xgboost(
                        X_train, y_train, X_val, y_val, log_to_mlflow=False
                    )
                elif model_type == "lstm":
                    model, metrics = self.train_lstm(
                        self._X_train_seq, self._y_train_seq,
                        self._X_val_seq, self._y_val_seq,
                        log_to_mlflow=False
                    )

                results[model_type] = {
                    "model": model,
                    "metrics": metrics,
                    "run_id": None,
                }

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                results[model_type] = {
                    "model": None,
                    "metrics": {"error": str(e)},
                    "run_id": None,
                }

        return results

    def select_best_model(
        self,
        metric: str = "val_rmse",
        lower_is_better: bool = True,
    ) -> Tuple[str, Any, Dict[str, float]]:
        """
        Select best model based on specified metric.

        Args:
            metric: Metric to use for selection (default: "val_rmse")
            lower_is_better: Whether lower metric values are better

        Returns:
            Tuple of (model_type, model_instance, metrics)

        Raises:
            ValueError: If no models have been trained
        """
        if not self.results:
            raise ValueError("No models have been trained. Call train_all_models() first.")

        best_type = None
        best_value = float('inf') if lower_is_better else float('-inf')
        best_model = None
        best_metrics = {}

        for model_type, result in self.results.items():
            # Skip failed models
            if result.get("model") is None:
                continue

            metrics = result.get("metrics", {})
            if metric not in metrics:
                continue

            value = metrics[metric]

            if lower_is_better:
                if value < best_value:
                    best_value = value
                    best_type = model_type
                    best_model = result["model"]
                    best_metrics = metrics
            else:
                if value > best_value:
                    best_value = value
                    best_type = model_type
                    best_model = result["model"]
                    best_metrics = metrics

        if best_type is None:
            raise ValueError(f"No valid models found with metric '{metric}'")

        logger.info(f"Best model: {best_type} with {metric}={best_value:.4f}")

        return best_type, best_model, best_metrics

    def save_best_model(
        self,
        model_type: Optional[str] = None,
        model: Optional[Any] = None,
        save_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save the best model to disk.

        Args:
            model_type: Type of model to save (auto-selected if None)
            model: Model instance to save (auto-selected if None)
            save_dir: Directory to save model (defaults to settings.paths.models_dir)

        Returns:
            Path to saved model

        Raises:
            ValueError: If no model to save
        """
        if model_type is None or model is None:
            model_type, model, _ = self.select_best_model()

        save_dir = save_dir or settings.paths.models_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Different save paths based on model type
        if model_type == "rf":
            save_path = save_dir / f"random_forest_{timestamp}.joblib"
        elif model_type == "xgb":
            save_path = save_dir / f"xgboost_{timestamp}"
        elif model_type == "lstm":
            save_path = save_dir / f"lstm_{timestamp}"
        else:
            save_path = save_dir / f"{model_type}_{timestamp}"

        model.save(save_path)
        logger.info(f"Saved {model_type} model to {save_path}")

        # Also save the scaler
        scaler_path = settings.paths.artifacts_dir / f"scaler_{timestamp}.joblib"
        self.feature_engineer.save_scaler(scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        return save_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train predictive maintenance models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="predictive-maintenance",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to data directory",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["all", "rf", "xgb", "lstm"],
        default="all",
        help="Model type to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=AVAILABLE_DATASETS,
        default="FD001",
        help="NASA Turbofan dataset to use",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the best model after training",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    return parser.parse_args()


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """
    Main entry point for training orchestrator.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("Predictive Maintenance Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"MLflow: {'disabled' if args.no_mlflow else 'enabled'}")

    try:
        # Initialize orchestrator
        orchestrator = TrainingOrchestrator(
            experiment_name=args.experiment_name,
            dataset=args.dataset,
            data_path=args.data_path,
            use_mlflow=not args.no_mlflow,
        )

        # Determine which models to train
        if args.model_type == "all":
            model_types = None  # Train all
        else:
            model_types = [args.model_type]

        # Train models
        results = orchestrator.train_all_models(model_types=model_types)

        # Report results
        logger.info("=" * 60)
        logger.info("Training Results:")
        logger.info("=" * 60)

        for model_type, result in results.items():
            metrics = result.get("metrics", {})
            if "error" in metrics:
                logger.error(f"  {model_type.upper()}: FAILED - {metrics['error']}")
            else:
                logger.info(f"  {model_type.upper()}:")
                logger.info(f"    val_rmse: {metrics.get('val_rmse', 'N/A'):.4f}")
                logger.info(f"    val_mae:  {metrics.get('val_mae', 'N/A'):.4f}")
                logger.info(f"    val_r2:   {metrics.get('val_r2', 'N/A'):.4f}")

        # Select and optionally save best model
        best_type, best_model, best_metrics = orchestrator.select_best_model()
        logger.info("=" * 60)
        logger.info(f"Best Model: {best_type.upper()}")
        logger.info(f"  val_rmse: {best_metrics.get('val_rmse', 'N/A'):.4f}")

        if args.save_model:
            save_path = orchestrator.save_best_model()
            logger.info(f"Model saved to: {save_path}")

        logger.info("=" * 60)
        logger.info("Training completed successfully!")

        return 0

    except DataPreparationError as e:
        logger.error(f"Data preparation failed: {e}")
        return 1
    except ModelTrainingError as e:
        logger.error(f"Model training failed: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
