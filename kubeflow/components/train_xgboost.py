#!/usr/bin/env python3
"""
Kubeflow Component: Train XGBoost Model

Trains an XGBoost model on preprocessed features with early stopping
and logs to MLflow.

Usage:
    python train_xgboost.py \
        --features-path /data/features/features.csv \
        --labels-path /data/features/labels.csv \
        --model-output /data/models/xgb_model_path.txt \
        --mlflow-tracking-uri http://mlflow:5000 \
        --experiment-name predictive-maintenance

Outputs:
    - model_path.txt: Contains path/URI to the MLflow model
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Conditional XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_xgboost")

# Conditional MLflow import
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Model will be saved locally only.")


# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
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
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 20


def load_data(features_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels from CSV files."""
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    X = features_df.values

    logger.info(f"Loading labels from {labels_path}")
    labels_df = pd.read_csv(labels_path)
    y = labels_df["RUL"].values

    logger.info(f"Loaded data: X={X.shape}, y={y.shape}")
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Calculate regression metrics."""
    metrics = {
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
    }
    return metrics


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparameters: Dict,
    early_stopping_rounds: int = 20,
) -> Tuple[xgb.XGBRegressor, Dict[str, float], int]:
    """Train XGBoost model with early stopping."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed")

    logger.info("Training XGBoost model...")
    logger.info(f"Hyperparameters: {hyperparameters}")
    logger.info(f"Early stopping rounds: {early_stopping_rounds}")

    model = xgb.XGBRegressor(**hyperparameters)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    # Get best iteration
    try:
        best_iteration = model.best_iteration
    except AttributeError:
        best_iteration = hyperparameters.get("n_estimators", 100)

    logger.info(f"Best iteration: {best_iteration}")

    # Calculate metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    metrics = {}
    metrics.update(calculate_metrics(y_train, train_pred, prefix="train_"))
    metrics.update(calculate_metrics(y_val, val_pred, prefix="val_"))
    metrics["best_iteration"] = best_iteration

    logger.info(f"Training metrics: {metrics}")
    return model, metrics, best_iteration


def log_to_mlflow(
    model: xgb.XGBRegressor,
    metrics: Dict[str, float],
    hyperparameters: Dict,
    tracking_uri: str,
    experiment_name: str,
    run_name: str = "xgboost",
) -> str:
    """Log model and metrics to MLflow."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping logging")
        return None

    logger.info(f"Logging to MLflow at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params(hyperparameters)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="xgboost_rul",
        )

        # Log feature importance
        if hasattr(model, "feature_importances_"):
            importance_dict = {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(model.feature_importances_[:20])
            }
            mlflow.log_params({f"importance_{k}": v for k, v in list(importance_dict.items())[:10]})

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Logged model to MLflow: {model_uri}")

        return model_uri


def save_model_locally(model: xgb.XGBRegressor, output_dir: Path) -> Path:
    """Save model locally in JSON format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgb_model.json"
    model.save_model(str(model_path))
    logger.info(f"Saved model locally to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kubeflow Component: Train XGBoost Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features-path",
        type=str,
        required=True,
        help="Path to features.csv",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        required=True,
        help="Path to labels.csv",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        required=True,
        help="Path to save model_path.txt (contains MLflow URI or local path)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="predictive-maintenance",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation set size (fraction)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth of trees",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (eta)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=20,
        help="Early stopping rounds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Kubeflow Component: Train XGBoost")
    logger.info("=" * 60)
    logger.info(f"Features path: {args.features_path}")
    logger.info(f"Labels path: {args.labels_path}")
    logger.info(f"Model output: {args.model_output}")
    logger.info(f"MLflow URI: {args.mlflow_tracking_uri}")
    logger.info(f"Experiment: {args.experiment_name}")

    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost is not installed. Cannot proceed.")
        return 1

    try:
        # Load data
        X, y = load_data(Path(args.features_path), Path(args.labels_path))

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=args.validation_split,
            random_state=args.random_state,
        )
        logger.info(f"Train/Val split: {len(X_train)}/{len(X_val)} samples")

        # Prepare hyperparameters
        hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
        hyperparameters["n_estimators"] = args.n_estimators
        hyperparameters["max_depth"] = args.max_depth
        hyperparameters["learning_rate"] = args.learning_rate
        hyperparameters["random_state"] = args.random_state

        # Train model
        model, metrics, best_iteration = train_model(
            X_train, y_train, X_val, y_val,
            hyperparameters,
            early_stopping_rounds=args.early_stopping_rounds,
        )

        # Log to MLflow
        model_uri = None
        if MLFLOW_AVAILABLE:
            try:
                model_uri = log_to_mlflow(
                    model=model,
                    metrics=metrics,
                    hyperparameters=hyperparameters,
                    tracking_uri=args.mlflow_tracking_uri,
                    experiment_name=args.experiment_name,
                )
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}. Saving locally instead.")

        # Save model locally as backup
        output_path = Path(args.model_output)
        model_dir = output_path.parent / "xgb_model"
        local_path = save_model_locally(model, model_dir)

        # Write model path output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_info = {
            "model_type": "xgboost",
            "mlflow_uri": model_uri,
            "local_path": str(local_path),
            "metrics": metrics,
            "best_iteration": best_iteration,
        }
        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved model info to {output_path}")

        logger.info("XGBoost training completed successfully!")
        logger.info(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"Validation MAE: {metrics['val_mae']:.4f}")
        logger.info(f"Validation R2: {metrics['val_r2']:.4f}")
        logger.info(f"Best iteration: {best_iteration}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
