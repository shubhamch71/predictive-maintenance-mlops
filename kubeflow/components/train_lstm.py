#!/usr/bin/env python3
"""
Kubeflow Component: Train LSTM Model

Trains an LSTM model on preprocessed features (converted to sequences)
and logs to MLflow.

Usage:
    python train_lstm.py \
        --features-path /data/features/features.csv \
        --labels-path /data/features/labels.csv \
        --model-output /data/models/lstm_model_path.txt \
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lstm")

# Conditional TensorFlow import
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM training will fail.")

# Conditional MLflow import
try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Model will be saved locally only.")


# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    "lstm_units": [64, 32],
    "dense_units": 32,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "sequence_length": 50,
}


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


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        sequence_length: Length of sequences

    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    logger.info(f"Creating sequences with length {sequence_length}")

    if len(X) < sequence_length:
        raise ValueError(
            f"Not enough samples ({len(X)}) to create sequences of length {sequence_length}"
        )

    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    logger.info(f"Created sequences: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq


def build_model(
    sequence_length: int,
    n_features: int,
    lstm_units: List[int],
    dense_units: int,
    dropout_rate: float,
    learning_rate: float,
) -> keras.Model:
    """Build LSTM model architecture."""
    logger.info(f"Building LSTM model: lstm_units={lstm_units}, dense_units={dense_units}")

    model = keras.Sequential()

    # First LSTM layer
    model.add(layers.LSTM(
        lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=(sequence_length, n_features),
    ))
    model.add(layers.Dropout(dropout_rate))

    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:], 1):
        return_seq = i < len(lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout_rate))

    # Dense layers
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(1))

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    logger.info(f"Model built with {model.count_params()} parameters")
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Calculate regression metrics."""
    y_pred = y_pred.flatten()
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
) -> Tuple[keras.Model, Dict[str, float], int]:
    """Train LSTM model with early stopping."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed")

    logger.info("Training LSTM model...")
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Build model
    model = build_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        lstm_units=hyperparameters["lstm_units"],
        dense_units=hyperparameters["dense_units"],
        dropout_rate=hyperparameters["dropout_rate"],
        learning_rate=hyperparameters["learning_rate"],
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=hyperparameters["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=hyperparameters["reduce_lr_factor"],
            patience=hyperparameters["reduce_lr_patience"],
            min_lr=hyperparameters["min_lr"],
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=hyperparameters["epochs"],
        batch_size=hyperparameters["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    epochs_trained = len(history.history["loss"])
    logger.info(f"Training completed after {epochs_trained} epochs")

    # Calculate metrics
    train_pred = model.predict(X_train, verbose=0)
    val_pred = model.predict(X_val, verbose=0)

    metrics = {}
    metrics.update(calculate_metrics(y_train, train_pred, prefix="train_"))
    metrics.update(calculate_metrics(y_val, val_pred, prefix="val_"))
    metrics["epochs_trained"] = epochs_trained

    logger.info(f"Training metrics: {metrics}")
    return model, metrics, epochs_trained


def log_to_mlflow(
    model: keras.Model,
    metrics: Dict[str, float],
    hyperparameters: Dict,
    tracking_uri: str,
    experiment_name: str,
    run_name: str = "lstm",
) -> str:
    """Log model and metrics to MLflow."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping logging")
        return None

    logger.info(f"Logging to MLflow at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters (convert lists to strings for MLflow)
        params_to_log = hyperparameters.copy()
        params_to_log["lstm_units"] = str(params_to_log["lstm_units"])
        mlflow.log_params(params_to_log)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name="lstm_rul",
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Logged model to MLflow: {model_uri}")

        return model_uri


def save_model_locally(model: keras.Model, output_dir: Path) -> Path:
    """Save model locally in SavedModel format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lstm_model"
    model.save(str(model_path))
    logger.info(f"Saved model locally to {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kubeflow Component: Train LSTM Model",
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
        "--sequence-length",
        type=int,
        default=50,
        help="Sequence length for LSTM",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience",
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
    logger.info("Kubeflow Component: Train LSTM")
    logger.info("=" * 60)
    logger.info(f"Features path: {args.features_path}")
    logger.info(f"Labels path: {args.labels_path}")
    logger.info(f"Model output: {args.model_output}")
    logger.info(f"MLflow URI: {args.mlflow_tracking_uri}")
    logger.info(f"Experiment: {args.experiment_name}")

    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed. Cannot proceed.")
        return 1

    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    try:
        # Load data
        X, y = load_data(Path(args.features_path), Path(args.labels_path))

        # Create sequences
        X_seq, y_seq = create_sequences(X, y, args.sequence_length)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq,
            test_size=args.validation_split,
            random_state=args.random_state,
        )
        logger.info(f"Train/Val split: {len(X_train)}/{len(X_val)} sequences")

        # Prepare hyperparameters
        hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
        hyperparameters["sequence_length"] = args.sequence_length
        hyperparameters["epochs"] = args.epochs
        hyperparameters["batch_size"] = args.batch_size
        hyperparameters["learning_rate"] = args.learning_rate
        hyperparameters["early_stopping_patience"] = args.early_stopping_patience

        # Train model
        model, metrics, epochs_trained = train_model(
            X_train, y_train, X_val, y_val,
            hyperparameters,
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
        model_dir = output_path.parent / "lstm_model_saved"
        local_path = save_model_locally(model, model_dir)

        # Write model path output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_info = {
            "model_type": "lstm",
            "mlflow_uri": model_uri,
            "local_path": str(local_path),
            "metrics": metrics,
            "epochs_trained": epochs_trained,
            "sequence_length": args.sequence_length,
        }
        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved model info to {output_path}")

        logger.info("LSTM training completed successfully!")
        logger.info(f"Validation RMSE: {metrics['val_rmse']:.4f}")
        logger.info(f"Validation MAE: {metrics['val_mae']:.4f}")
        logger.info(f"Validation R2: {metrics['val_r2']:.4f}")
        logger.info(f"Epochs trained: {epochs_trained}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
