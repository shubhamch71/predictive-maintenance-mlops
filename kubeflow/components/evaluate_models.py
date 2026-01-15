#!/usr/bin/env python3
"""
Kubeflow Component: Evaluate and Compare Models

Loads all trained models, evaluates them on validation set,
compares metrics, and outputs the best model information.

Usage:
    python evaluate_models.py \
        --rf-model-info /data/models/rf_model_path.txt \
        --xgb-model-info /data/models/xgb_model_path.txt \
        --lstm-model-info /data/models/lstm_model_path.txt \
        --features-path /data/features/features.csv \
        --labels-path /data/features/labels.csv \
        --output-path /data/evaluation/best_model_info.json \
        --mlflow-tracking-uri http://mlflow:5000

Outputs:
    - best_model_info.json: Contains best model type, URI, and comparison metrics
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_models")

# Conditional imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


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


def load_model_info(model_info_path: Path) -> Optional[Dict]:
    """Load model information from JSON file."""
    if not model_info_path.exists():
        logger.warning(f"Model info file not found: {model_info_path}")
        return None

    with open(model_info_path, "r") as f:
        return json.load(f)


def load_rf_model(model_info: Dict) -> Optional[Any]:
    """Load Random Forest model from local path or MLflow."""
    if not JOBLIB_AVAILABLE:
        logger.warning("joblib not available, cannot load RF model")
        return None

    local_path = model_info.get("local_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Loading RF model from {local_path}")
        return joblib.load(local_path)

    mlflow_uri = model_info.get("mlflow_uri")
    if mlflow_uri and MLFLOW_AVAILABLE:
        logger.info(f"Loading RF model from MLflow: {mlflow_uri}")
        return mlflow.sklearn.load_model(mlflow_uri)

    return None


def load_xgb_model(model_info: Dict) -> Optional[Any]:
    """Load XGBoost model from local path or MLflow."""
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, cannot load XGB model")
        return None

    local_path = model_info.get("local_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Loading XGB model from {local_path}")
        model = xgb.XGBRegressor()
        model.load_model(local_path)
        return model

    mlflow_uri = model_info.get("mlflow_uri")
    if mlflow_uri and MLFLOW_AVAILABLE:
        logger.info(f"Loading XGB model from MLflow: {mlflow_uri}")
        return mlflow.xgboost.load_model(mlflow_uri)

    return None


def load_lstm_model(model_info: Dict) -> Optional[Any]:
    """Load LSTM model from local path or MLflow."""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, cannot load LSTM model")
        return None

    local_path = model_info.get("local_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Loading LSTM model from {local_path}")
        return keras.models.load_model(local_path)

    mlflow_uri = model_info.get("mlflow_uri")
    if mlflow_uri and MLFLOW_AVAILABLE:
        logger.info(f"Loading LSTM model from MLflow: {mlflow_uri}")
        return mlflow.keras.load_model(mlflow_uri)

    return None


def create_sequences(X: np.ndarray, sequence_length: int) -> np.ndarray:
    """Create sequences for LSTM input."""
    if len(X) < sequence_length:
        raise ValueError(f"Not enough samples for sequence length {sequence_length}")

    X_seq = []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])

    return np.array(X_seq)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    # Align lengths if different (for LSTM sequences)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    sequence_length: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a model and return metrics."""
    logger.info(f"Evaluating {model_type} model...")

    try:
        if model_type == "lstm" and sequence_length:
            # Create sequences for LSTM
            X_seq = create_sequences(X, sequence_length)
            y_aligned = y[sequence_length - 1:]
            y_pred = model.predict(X_seq, verbose=0)
            metrics = calculate_metrics(y_aligned, y_pred)
        else:
            y_pred = model.predict(X)
            metrics = calculate_metrics(y, y_pred)

        logger.info(f"{model_type} metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to evaluate {model_type}: {e}")
        return {"rmse": float("inf"), "mae": float("inf"), "r2": float("-inf"), "error": str(e)}


def select_best_model(
    results: Dict[str, Dict[str, Any]],
    metric: str = "rmse",
    lower_is_better: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Select the best model based on a metric."""
    best_model = None
    best_value = float("inf") if lower_is_better else float("-inf")
    best_info = {}

    for model_type, info in results.items():
        if "error" in info.get("eval_metrics", {}):
            continue

        value = info.get("eval_metrics", {}).get(metric, float("inf") if lower_is_better else float("-inf"))

        if lower_is_better:
            if value < best_value:
                best_value = value
                best_model = model_type
                best_info = info
        else:
            if value > best_value:
                best_value = value
                best_model = model_type
                best_info = info

    logger.info(f"Best model: {best_model} with {metric}={best_value:.4f}")
    return best_model, best_info


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kubeflow Component: Evaluate and Compare Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rf-model-info",
        type=str,
        default=None,
        help="Path to RF model_path.txt",
    )
    parser.add_argument(
        "--xgb-model-info",
        type=str,
        default=None,
        help="Path to XGBoost model_path.txt",
    )
    parser.add_argument(
        "--lstm-model-info",
        type=str,
        default=None,
        help="Path to LSTM model_path.txt",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        required=True,
        help="Path to features.csv for evaluation",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        required=True,
        help="Path to labels.csv for evaluation",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save best_model_info.json",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Sequence length for LSTM evaluation",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="rmse",
        choices=["rmse", "mae", "r2"],
        help="Metric for best model selection",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Kubeflow Component: Evaluate Models")
    logger.info("=" * 60)
    logger.info(f"RF model info: {args.rf_model_info}")
    logger.info(f"XGB model info: {args.xgb_model_info}")
    logger.info(f"LSTM model info: {args.lstm_model_info}")
    logger.info(f"Features path: {args.features_path}")
    logger.info(f"Labels path: {args.labels_path}")
    logger.info(f"Output path: {args.output_path}")

    # Set MLflow tracking URI if available
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    try:
        # Load evaluation data
        X, y = load_data(Path(args.features_path), Path(args.labels_path))

        results = {}

        # Evaluate Random Forest
        if args.rf_model_info:
            rf_info = load_model_info(Path(args.rf_model_info))
            if rf_info:
                rf_model = load_rf_model(rf_info)
                if rf_model:
                    eval_metrics = evaluate_model(rf_model, X, y, "random_forest")
                    results["random_forest"] = {
                        "model_info": rf_info,
                        "eval_metrics": eval_metrics,
                    }

        # Evaluate XGBoost
        if args.xgb_model_info:
            xgb_info = load_model_info(Path(args.xgb_model_info))
            if xgb_info:
                xgb_model = load_xgb_model(xgb_info)
                if xgb_model:
                    eval_metrics = evaluate_model(xgb_model, X, y, "xgboost")
                    results["xgboost"] = {
                        "model_info": xgb_info,
                        "eval_metrics": eval_metrics,
                    }

        # Evaluate LSTM
        if args.lstm_model_info:
            lstm_info = load_model_info(Path(args.lstm_model_info))
            if lstm_info:
                lstm_model = load_lstm_model(lstm_info)
                if lstm_model:
                    seq_len = lstm_info.get("sequence_length", args.sequence_length)
                    eval_metrics = evaluate_model(
                        lstm_model, X, y, "lstm",
                        sequence_length=seq_len,
                    )
                    results["lstm"] = {
                        "model_info": lstm_info,
                        "eval_metrics": eval_metrics,
                    }

        # Check if any models were evaluated
        if not results:
            logger.error("No models were successfully loaded and evaluated")
            return 1

        # Select best model
        lower_is_better = args.selection_metric != "r2"
        best_model_type, best_model_info = select_best_model(
            results,
            metric=args.selection_metric,
            lower_is_better=lower_is_better,
        )

        # Prepare output
        output = {
            "best_model": {
                "type": best_model_type,
                "mlflow_uri": best_model_info.get("model_info", {}).get("mlflow_uri"),
                "local_path": best_model_info.get("model_info", {}).get("local_path"),
                "metrics": best_model_info.get("eval_metrics", {}),
            },
            "comparison": {
                model_type: {
                    "metrics": info.get("eval_metrics", {}),
                    "mlflow_uri": info.get("model_info", {}).get("mlflow_uri"),
                }
                for model_type, info in results.items()
            },
            "selection_metric": args.selection_metric,
            "models_evaluated": list(results.keys()),
        }

        # Save output
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")

        # Log summary
        logger.info("=" * 60)
        logger.info("Evaluation Summary:")
        logger.info("=" * 60)
        for model_type, info in results.items():
            metrics = info.get("eval_metrics", {})
            logger.info(f"  {model_type}:")
            logger.info(f"    RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"    MAE:  {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"    R2:   {metrics.get('r2', 'N/A'):.4f}")

        logger.info("-" * 60)
        logger.info(f"Best Model: {best_model_type}")
        logger.info(f"Selection Metric: {args.selection_metric}")
        logger.info("=" * 60)

        logger.info("Model evaluation completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
