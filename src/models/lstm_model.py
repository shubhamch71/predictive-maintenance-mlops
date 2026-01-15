"""
LSTM model for Remaining Useful Life (RUL) prediction.

This module provides a production-ready LSTM neural network with
TensorFlow/Keras, including sequence windowing, callbacks for early
stopping and checkpointing, and MLflow integration.

Example:
    >>> from src.models.lstm_model import LSTMModel
    >>> model = LSTMModel(sequence_length=50, n_features=24)
    >>> model.train(X_train_seq, y_train, X_val_seq, y_val)
    >>> predictions = model.predict(X_test_seq)

Architecture:
    Input (sequence_length, n_features)
    -> LSTM(64, return_sequences=True)
    -> Dropout(0.2)
    -> LSTM(32, return_sequences=False)
    -> Dropout(0.2)
    -> Dense(32, relu)
    -> Dropout(0.2)
    -> Dense(1, linear)

Hyperparameter Examples:
    Default:
        lstm_units=(64, 32), dropout_rate=0.2, learning_rate=0.001

    Deeper network:
        lstm_units=(128, 64, 32), dropout_rate=0.3

    Faster training:
        lstm_units=(32, 16), epochs=50, batch_size=64
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

# Conditional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True

    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TF_AVAILABLE = False

# Conditional MLflow import
try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Exception raised for model-related errors."""
    pass


class LSTMModel:
    """
    LSTM neural network for RUL prediction with MLflow integration.

    This class implements a multi-layer LSTM architecture for time-series
    regression, with built-in support for sequence windowing, early stopping,
    model checkpointing, and experiment tracking via MLflow.

    Attributes:
        model: The underlying Keras Sequential model
        sequence_length: Number of time steps in input sequences
        n_features: Number of features per time step
        hyperparameters: Dictionary of model hyperparameters
        is_trained: Whether the model has been trained
        metrics: Dictionary of evaluation metrics
        history: Training history from Keras

    Example:
        >>> model = LSTMModel(
        ...     sequence_length=50,
        ...     n_features=24,
        ...     lstm_units=(64, 32),
        ...     dropout_rate=0.2
        ... )
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> print(model.metrics)
    """

    # Default hyperparameters
    DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
        "lstm_units": (64, 32),
        "dense_units": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "reduce_lr_factor": 0.5,
        "min_lr": 1e-6,
    }

    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = 24,
        lstm_units: Tuple[int, ...] = (64, 32),
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of features per time step
            lstm_units: Tuple of units for each LSTM layer
            dense_units: Units in the dense layer before output
            dropout_rate: Dropout rate between layers
            learning_rate: Learning rate for Adam optimizer
            hyperparameters: Additional hyperparameters (overrides others)
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            )

        self.sequence_length = sequence_length
        self.n_features = n_features

        # Build hyperparameters
        self.hyperparameters = {**self.DEFAULT_HYPERPARAMETERS}
        self.hyperparameters.update({
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "n_features": n_features,
        })

        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        self.model: Optional[keras.Model] = None
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
        self.history: Optional[Dict[str, List[float]]] = None

        # Set random seed for reproducibility
        tf.random.set_seed(settings.model.random_state)
        np.random.seed(settings.model.random_state)

        logger.info(
            f"Initialized LSTMModel: sequence_length={sequence_length}, "
            f"n_features={n_features}, lstm_units={lstm_units}"
        )

    def build_model(self) -> keras.Model:
        """
        Build the LSTM architecture.

        Returns:
            Compiled Keras model

        Architecture:
            - Multiple LSTM layers with dropout
            - Dense layer with ReLU activation
            - Linear output for regression
        """
        lstm_units = self.hyperparameters["lstm_units"]
        dense_units = self.hyperparameters["dense_units"]
        dropout_rate = self.hyperparameters["dropout_rate"]
        learning_rate = self.hyperparameters["learning_rate"]

        # Input layer
        inputs = keras.Input(
            shape=(self.sequence_length, self.n_features),
            name="input"
        )
        x = inputs

        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f"lstm_{i+1}"
            )(x)
            x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

        # Dense layers
        x = layers.Dense(dense_units, activation="relu", name="dense")(x)
        x = layers.Dropout(dropout_rate, name="dropout_dense")(x)

        # Output layer (linear for regression)
        outputs = layers.Dense(1, activation="linear", name="output")(x)

        # Create model
        self.model = keras.Model(inputs, outputs, name="lstm_rul")

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        log_to_mlflow: bool = True,
        experiment_name: Optional[str] = None,
        run_name: str = "lstm",
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Train the LSTM model.

        Args:
            X_train: Training sequences of shape (n_samples, sequence_length, n_features)
            y_train: Training targets of shape (n_samples,)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs (uses hyperparameter if None)
            batch_size: Training batch size (uses hyperparameter if None)
            log_to_mlflow: Whether to log to MLflow
            experiment_name: MLflow experiment name
            run_name: Name for the MLflow run
            checkpoint_path: Path to save model checkpoints

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ModelError: If training fails
            ValueError: If input shapes are incorrect
        """
        # Validate input shapes
        self._validate_input_shape(X_train, "X_train")
        if X_val is not None:
            self._validate_input_shape(X_val, "X_val")

        # Update hyperparameters if provided
        epochs = epochs or self.hyperparameters["epochs"]
        batch_size = batch_size or self.hyperparameters["batch_size"]

        # Build model if not already built
        if self.model is None:
            self.build_model()

        logger.info(f"Training LSTM on {X_train.shape[0]} sequences for {epochs} epochs")

        try:
            # Prepare callbacks
            callback_list = self._create_callbacks(checkpoint_path)

            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)

            # Train the model
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callback_list,
                verbose=1,
            )

            self.is_trained = True
            self.history = history.history

            # Calculate metrics
            train_pred = self.predict(X_train)
            self.metrics = self._calculate_metrics(y_train, train_pred, prefix="train_")

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred, prefix="val_")
                self.metrics.update(val_metrics)

            # Add training info
            self.metrics["epochs_trained"] = len(history.history["loss"])

            logger.info(f"Training complete. Metrics: {self.metrics}")

            # Log to MLflow
            if log_to_mlflow and MLFLOW_AVAILABLE:
                self._log_to_mlflow(
                    experiment_name=experiment_name,
                    run_name=run_name,
                )

            return self.metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ModelError(f"Failed to train LSTM model: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Sequences of shape (n_samples, sequence_length, n_features)

        Returns:
            Predicted RUL values of shape (n_samples,)

        Raises:
            ModelError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model must be trained before making predictions")

        try:
            self._validate_input_shape(X, "X")
            predictions = self.model.predict(X, verbose=0).flatten()
            # Clip to non-negative
            predictions = np.clip(predictions, 0, None)
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")

    def create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Create sequences from flat feature array.

        Args:
            X: Features of shape (n_samples, n_features)
            y: Targets of shape (n_samples,)

        Returns:
            If y is None: X_sequences of shape (n_sequences, sequence_length, n_features)
            If y is provided: Tuple of (X_sequences, y_sequences)
        """
        if len(X) < self.sequence_length:
            raise ValueError(
                f"Not enough samples ({len(X)}) for sequence length {self.sequence_length}"
            )

        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i : i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length - 1])

        X_seq = np.array(X_seq)

        if y is not None:
            return X_seq, np.array(y_seq)
        return X_seq

    def plot_training_history(
        self,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot training history (loss and MAE curves).

        Args:
            figsize: Figure size
            save_path: Path to save the figure

        Returns:
            Matplotlib figure object
        """
        if not self.history:
            raise ModelError("No training history available")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Loss plot
        axes[0].plot(self.history["loss"], label="Training Loss")
        if "val_loss" in self.history:
            axes[0].plot(self.history["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE plot
        axes[1].plot(self.history["mae"], label="Training MAE")
        if "val_mae" in self.history:
            axes[1].plot(self.history["val_mae"], label="Validation MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Training and Validation MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved training history plot to {save_path}")

        return fig

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk in SavedModel format.

        Args:
            path: Directory path to save the model

        Raises:
            ModelError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Cannot save untrained model")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model in SavedModel format
        model_path = path / "saved_model"
        self.model.save(model_path)

        # Save metadata
        metadata = {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "hyperparameters": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in self.hyperparameters.items()
            },
            "metrics": self.metrics,
            "history": self.history,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {path}")

    def load(self, path: Union[str, Path]) -> "LSTMModel":
        """
        Load the model from disk.

        Args:
            path: Directory path to load the model from

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If model directory doesn't exist
        """
        path = Path(path)
        model_path = path / "saved_model"
        metadata_path = path / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load Keras model
        self.model = keras.models.load_model(model_path)

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.sequence_length = metadata.get("sequence_length", self.sequence_length)
            self.n_features = metadata.get("n_features", self.n_features)
            self.hyperparameters = metadata.get("hyperparameters", self.hyperparameters)

            # Convert lists back to tuples where needed
            if "lstm_units" in self.hyperparameters:
                self.hyperparameters["lstm_units"] = tuple(
                    self.hyperparameters["lstm_units"]
                )

            self.metrics = metadata.get("metrics", {})
            self.history = metadata.get("history")

        self.is_trained = True
        logger.info(f"Loaded model from {path}")

        return self

    def get_model_summary(self) -> str:
        """Get model architecture summary as string."""
        if self.model is None:
            self.build_model()

        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def _validate_input_shape(self, X: np.ndarray, name: str) -> None:
        """Validate input array shape."""
        if len(X.shape) != 3:
            raise ValueError(
                f"{name} must be 3D (n_samples, sequence_length, n_features), "
                f"got shape {X.shape}"
            )

        if X.shape[1] != self.sequence_length:
            raise ValueError(
                f"{name} sequence length ({X.shape[1]}) doesn't match "
                f"model sequence length ({self.sequence_length})"
            )

        if X.shape[2] != self.n_features:
            raise ValueError(
                f"{name} n_features ({X.shape[2]}) doesn't match "
                f"model n_features ({self.n_features})"
            )

    def _create_callbacks(
        self,
        checkpoint_path: Optional[Path] = None,
    ) -> List[callbacks.Callback]:
        """Create training callbacks."""
        callback_list = []

        # Early stopping
        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss" if True else "loss",
                patience=self.hyperparameters["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            )
        )

        # Learning rate reduction
        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if True else "loss",
                factor=self.hyperparameters["reduce_lr_factor"],
                patience=self.hyperparameters["reduce_lr_patience"],
                min_lr=self.hyperparameters["min_lr"],
                verbose=1,
            )
        )

        # Model checkpoint
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            callback_list.append(
                callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path / "best_model.keras"),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                )
            )

        return callback_list

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
        run_name: str = "lstm",
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
                # Log hyperparameters (convert tuples to lists for MLflow)
                params_to_log = {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in self.hyperparameters.items()
                }
                mlflow.log_params(params_to_log)

                # Log metrics
                mlflow.log_metrics(self.metrics)

                # Log training history
                if self.history:
                    for metric_name, values in self.history.items():
                        for step, value in enumerate(values):
                            mlflow.log_metric(metric_name, value, step=step)

                # Log model
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name=f"{exp_name}_lstm",
                )

                # Log training history plot
                with tempfile.TemporaryDirectory() as tmpdir:
                    if self.history:
                        plot_path = Path(tmpdir) / "training_history.png"
                        self.plot_training_history(save_path=plot_path)
                        mlflow.log_artifact(str(plot_path))
                        plt.close()

                    # Log model summary
                    summary_path = Path(tmpdir) / "model_summary.txt"
                    with open(summary_path, "w") as f:
                        f.write(self.get_model_summary())
                    mlflow.log_artifact(str(summary_path))

                logger.info(f"Logged to MLflow experiment: {exp_name}")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.hyperparameters.copy()
