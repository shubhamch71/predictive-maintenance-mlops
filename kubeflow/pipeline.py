#!/usr/bin/env python3
"""
Kubeflow Pipeline: Predictive Maintenance Training Pipeline

Orchestrates the full ML training workflow:
1. Data Ingestion - Load and preprocess NASA Turbofan dataset
2. Feature Engineering - Create rolling/lag features and normalize
3. Model Training (parallel) - Train RF, XGBoost, and LSTM models
4. Model Evaluation - Compare models and select the best

Usage:
    # Compile pipeline
    python compile_pipeline.py

    # Submit to Kubeflow
    kfp run create -f pipeline.yaml -r run-name

Environment Variables:
    CONTAINER_REGISTRY: Container registry for images (default: gcr.io/your-project)
    IMAGE_TAG: Image tag version (default: latest)
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Model,
    Output,
    component,
    pipeline,
)

# =============================================================================
# Configuration
# =============================================================================

CONTAINER_REGISTRY = "gcr.io/predictive-maintenance"
IMAGE_TAG = "latest"

# Base images for components
BASE_IMAGE = f"{CONTAINER_REGISTRY}/base:{IMAGE_TAG}"
TRAINING_IMAGE = f"{CONTAINER_REGISTRY}/training:{IMAGE_TAG}"
TENSORFLOW_IMAGE = f"{CONTAINER_REGISTRY}/tensorflow:{IMAGE_TAG}"

# Resource configurations
RESOURCE_CONFIGS = {
    "data_ingestion": {
        "cpu_request": "500m",
        "cpu_limit": "1",
        "memory_request": "1Gi",
        "memory_limit": "2Gi",
    },
    "feature_engineering": {
        "cpu_request": "1",
        "cpu_limit": "2",
        "memory_request": "2Gi",
        "memory_limit": "4Gi",
    },
    "train_rf": {
        "cpu_request": "2",
        "cpu_limit": "4",
        "memory_request": "4Gi",
        "memory_limit": "8Gi",
    },
    "train_xgboost": {
        "cpu_request": "2",
        "cpu_limit": "4",
        "memory_request": "4Gi",
        "memory_limit": "8Gi",
    },
    "train_lstm": {
        "cpu_request": "2",
        "cpu_limit": "4",
        "memory_request": "8Gi",
        "memory_limit": "16Gi",
        "gpu_limit": "1",  # Optional GPU
    },
    "evaluate": {
        "cpu_request": "1",
        "cpu_limit": "2",
        "memory_request": "4Gi",
        "memory_limit": "8Gi",
    },
}


# =============================================================================
# Component Definitions (using @component decorator for KFP v2)
# =============================================================================

@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy"],
)
def data_ingestion_op(
    dataset: str,
    data_path: str,
    max_rul: int,
    processed_data: Output[Dataset],
) -> NamedTuple("Outputs", [("num_samples", int), ("num_units", int)]):
    """
    Load NASA Turbofan dataset and compute RUL labels.

    Args:
        dataset: Dataset name (FD001-FD004)
        data_path: Path to raw data directory
        max_rul: Maximum RUL value for clipping
        processed_data: Output artifact for processed CSV

    Returns:
        num_samples: Total number of samples
        num_units: Number of unique engine units
    """
    import os
    from pathlib import Path
    from collections import namedtuple

    import numpy as np
    import pandas as pd

    COLUMN_NAMES = [
        "unit_id", "time_cycle",
        "op_setting_1", "op_setting_2", "op_setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
        "sensor_21",
    ]

    # Load training data
    train_file = Path(data_path) / f"train_{dataset}.txt"
    print(f"Loading data from {train_file}")

    df = pd.read_csv(
        train_file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    # Compute RUL
    max_cycles = df.groupby("unit_id")["time_cycle"].max()
    df["max_cycle"] = df["unit_id"].map(max_cycles)
    df["RUL"] = df["max_cycle"] - df["time_cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    df = df.drop(columns=["max_cycle"])

    # Save output
    os.makedirs(os.path.dirname(processed_data.path), exist_ok=True)
    df.to_csv(processed_data.path, index=False)

    print(f"Saved {len(df)} samples from {df['unit_id'].nunique()} units")

    outputs = namedtuple("Outputs", ["num_samples", "num_units"])
    return outputs(len(df), df["unit_id"].nunique())


@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"],
)
def feature_engineering_op(
    processed_data: Input[Dataset],
    rolling_windows: str,
    lag_periods: str,
    features_output: Output[Dataset],
    labels_output: Output[Dataset],
    scaler_output: Output[Artifact],
) -> NamedTuple("Outputs", [("num_features", int), ("num_samples", int)]):
    """
    Apply feature engineering: rolling stats, lag features, normalization.

    Args:
        processed_data: Input processed CSV from data ingestion
        rolling_windows: Comma-separated window sizes
        lag_periods: Comma-separated lag periods
        features_output: Output normalized features CSV
        labels_output: Output labels CSV
        scaler_output: Output fitted scaler pickle

    Returns:
        num_features: Number of engineered features
        num_samples: Number of samples after dropna
    """
    import os
    from pathlib import Path
    from collections import namedtuple

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler

    SENSORS_TO_USE = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ]
    OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

    # Parse parameters
    windows = [int(x) for x in rolling_windows.split(",")]
    lags = [int(x) for x in lag_periods.split(",")]

    # Load data
    print(f"Loading data from {processed_data.path}")
    df = pd.read_csv(processed_data.path)

    # Rolling features
    grouped = df.groupby("unit_id")
    for col in SENSORS_TO_USE:
        for window in windows:
            df[f"{col}_rolling_mean_{window}"] = grouped[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_rolling_std_{window}"] = grouped[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

    # Lag features
    grouped = df.groupby("unit_id")
    for col in SENSORS_TO_USE:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = grouped[col].transform(lambda x: x.shift(lag))

    # Rate of change
    grouped = df.groupby("unit_id")
    for col in SENSORS_TO_USE:
        df[f"{col}_delta"] = grouped[col].transform(lambda x: x.diff())

    # Drop NaN
    df = df.dropna()

    # Get feature columns
    exclude_cols = ["unit_id", "time_cycle", "RUL"]
    feature_columns = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_columns].values
    y = df["RUL"].values

    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Save outputs
    for path in [features_output.path, labels_output.path, scaler_output.path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    features_df = pd.DataFrame(X_normalized, columns=feature_columns)
    features_df.to_csv(features_output.path, index=False)

    labels_df = pd.DataFrame({"RUL": y})
    labels_df.to_csv(labels_output.path, index=False)

    scaler_data = {
        "scaler": scaler,
        "feature_columns": feature_columns,
        "rolling_windows": windows,
        "lag_periods": lags,
    }
    joblib.dump(scaler_data, scaler_output.path)

    print(f"Created {len(feature_columns)} features for {len(y)} samples")

    outputs = namedtuple("Outputs", ["num_features", "num_samples"])
    return outputs(len(feature_columns), len(y))


@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "mlflow", "joblib"],
)
def train_rf_op(
    features: Input[Dataset],
    labels: Input[Dataset],
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    model_output: Output[Artifact],
) -> NamedTuple("Outputs", [("val_rmse", float), ("val_mae", float), ("val_r2", float)]):
    """
    Train Random Forest model and log to MLflow.
    """
    import json
    import os
    from collections import namedtuple

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # Load data
    X = pd.read_csv(features.path).values
    y = pd.read_csv(labels.path)["RUL"].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Train
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate
    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))
    val_r2 = float(r2_score(y_val, val_pred))

    # Log to MLflow
    model_uri = None
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="random_forest") as run:
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth > 0 else "None",
                "random_state": random_state,
            })
            mlflow.log_metrics({
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
            })
            mlflow.sklearn.log_model(model, "model", registered_model_name="random_forest_rul")
            model_uri = f"runs:/{run.info.run_id}/model"
    except Exception as e:
        print(f"MLflow logging failed: {e}")

    # Save locally
    os.makedirs(os.path.dirname(model_output.path), exist_ok=True)
    local_model_path = model_output.path.replace(".json", "_model.joblib")
    joblib.dump(model, local_model_path)

    model_info = {
        "model_type": "random_forest",
        "mlflow_uri": model_uri,
        "local_path": local_model_path,
        "metrics": {"val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2},
    }
    with open(model_output.path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"RF Training complete: RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R2={val_r2:.4f}")

    outputs = namedtuple("Outputs", ["val_rmse", "val_mae", "val_r2"])
    return outputs(val_rmse, val_mae, val_r2)


@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "mlflow"],
)
def train_xgboost_op(
    features: Input[Dataset],
    labels: Input[Dataset],
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    model_output: Output[Artifact],
) -> NamedTuple("Outputs", [("val_rmse", float), ("val_mae", float), ("val_r2", float)]):
    """
    Train XGBoost model and log to MLflow.
    """
    import json
    import os
    from collections import namedtuple

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # Load data
    X = pd.read_csv(features.path).values
    y = pd.read_csv(labels.path)["RUL"].values

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Train
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))
    val_r2 = float(r2_score(y_val, val_pred))

    # Log to MLflow
    model_uri = None
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="xgboost") as run:
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "random_state": random_state,
            })
            mlflow.log_metrics({
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
            })
            mlflow.xgboost.log_model(model, "model", registered_model_name="xgboost_rul")
            model_uri = f"runs:/{run.info.run_id}/model"
    except Exception as e:
        print(f"MLflow logging failed: {e}")

    # Save locally
    os.makedirs(os.path.dirname(model_output.path), exist_ok=True)
    local_model_path = model_output.path.replace(".json", "_model.json")
    model.save_model(local_model_path)

    model_info = {
        "model_type": "xgboost",
        "mlflow_uri": model_uri,
        "local_path": local_model_path,
        "metrics": {"val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2},
    }
    with open(model_output.path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"XGB Training complete: RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R2={val_r2:.4f}")

    outputs = namedtuple("Outputs", ["val_rmse", "val_mae", "val_r2"])
    return outputs(val_rmse, val_mae, val_r2)


@component(
    base_image="tensorflow/tensorflow:2.13.0",
    packages_to_install=["pandas", "numpy", "scikit-learn", "mlflow"],
)
def train_lstm_op(
    features: Input[Dataset],
    labels: Input[Dataset],
    mlflow_tracking_uri: str,
    experiment_name: str,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
    model_output: Output[Artifact],
) -> NamedTuple("Outputs", [("val_rmse", float), ("val_mae", float), ("val_r2", float)]):
    """
    Train LSTM model and log to MLflow.
    """
    import json
    import os
    from collections import namedtuple

    import numpy as np
    import pandas as pd
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # Set seeds
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Load data
    X = pd.read_csv(features.path).values
    y = pd.read_csv(labels.path)["RUL"].values

    # Create sequences
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=random_state
    )

    # Build model
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    # Train
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_pred = model.predict(X_val, verbose=0).flatten()
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_mae = float(mean_absolute_error(y_val, val_pred))
    val_r2 = float(r2_score(y_val, val_pred))

    # Log to MLflow
    model_uri = None
    try:
        import mlflow
        import mlflow.keras

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="lstm") as run:
            mlflow.log_params({
                "sequence_length": sequence_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            })
            mlflow.log_metrics({
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
            })
            mlflow.keras.log_model(model, "model", registered_model_name="lstm_rul")
            model_uri = f"runs:/{run.info.run_id}/model"
    except Exception as e:
        print(f"MLflow logging failed: {e}")

    # Save locally
    os.makedirs(os.path.dirname(model_output.path), exist_ok=True)
    local_model_path = model_output.path.replace(".json", "_model")
    model.save(local_model_path)

    model_info = {
        "model_type": "lstm",
        "mlflow_uri": model_uri,
        "local_path": local_model_path,
        "metrics": {"val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2},
        "sequence_length": sequence_length,
    }
    with open(model_output.path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"LSTM Training complete: RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R2={val_r2:.4f}")

    outputs = namedtuple("Outputs", ["val_rmse", "val_mae", "val_r2"])
    return outputs(val_rmse, val_mae, val_r2)


@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def evaluate_models_op(
    rf_model_info: Input[Artifact],
    xgb_model_info: Input[Artifact],
    lstm_model_info: Input[Artifact],
    rf_rmse: float,
    rf_mae: float,
    rf_r2: float,
    xgb_rmse: float,
    xgb_mae: float,
    xgb_r2: float,
    lstm_rmse: float,
    lstm_mae: float,
    lstm_r2: float,
    selection_metric: str,
    best_model_output: Output[Artifact],
) -> NamedTuple("Outputs", [("best_model_type", str), ("best_rmse", float), ("best_mae", float)]):
    """
    Compare all trained models and select the best one.
    """
    import json
    import os
    from collections import namedtuple

    results = {
        "random_forest": {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2, "info_path": rf_model_info.path},
        "xgboost": {"rmse": xgb_rmse, "mae": xgb_mae, "r2": xgb_r2, "info_path": xgb_model_info.path},
        "lstm": {"rmse": lstm_rmse, "mae": lstm_mae, "r2": lstm_r2, "info_path": lstm_model_info.path},
    }

    # Select best model
    if selection_metric == "r2":
        best_model = max(results.items(), key=lambda x: x[1]["r2"])
    else:  # rmse or mae (lower is better)
        best_model = min(results.items(), key=lambda x: x[1][selection_metric])

    best_model_type = best_model[0]
    best_metrics = best_model[1]

    # Load best model info
    with open(best_metrics["info_path"], "r") as f:
        best_model_info = json.load(f)

    # Prepare output
    output = {
        "best_model": {
            "type": best_model_type,
            "mlflow_uri": best_model_info.get("mlflow_uri"),
            "local_path": best_model_info.get("local_path"),
            "metrics": {
                "rmse": best_metrics["rmse"],
                "mae": best_metrics["mae"],
                "r2": best_metrics["r2"],
            },
        },
        "comparison": {
            model_type: {
                "rmse": info["rmse"],
                "mae": info["mae"],
                "r2": info["r2"],
            }
            for model_type, info in results.items()
        },
        "selection_metric": selection_metric,
    }

    # Save output
    os.makedirs(os.path.dirname(best_model_output.path), exist_ok=True)
    with open(best_model_output.path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("Model Evaluation Summary:")
    print("=" * 60)
    for model_type, info in results.items():
        print(f"  {model_type}:")
        print(f"    RMSE: {info['rmse']:.4f}")
        print(f"    MAE:  {info['mae']:.4f}")
        print(f"    R2:   {info['r2']:.4f}")
    print("-" * 60)
    print(f"Best Model: {best_model_type}")
    print("=" * 60)

    outputs = namedtuple("Outputs", ["best_model_type", "best_rmse", "best_mae"])
    return outputs(best_model_type, best_metrics["rmse"], best_metrics["mae"])


# =============================================================================
# Pipeline Definition
# =============================================================================

@pipeline(
    name="predictive-maintenance-training",
    description="End-to-end ML pipeline for predictive maintenance using NASA Turbofan dataset",
)
def predictive_maintenance_pipeline(
    # Data parameters
    data_path: str = "/data/raw",
    dataset: str = "FD001",
    max_rul: int = 125,

    # Feature engineering parameters
    rolling_windows: str = "5,10,20",
    lag_periods: str = "1,5,10",

    # MLflow parameters
    mlflow_tracking_uri: str = "http://mlflow:5000",
    experiment_name: str = "predictive-maintenance",

    # Random Forest hyperparameters
    rf_n_estimators: int = 100,
    rf_max_depth: int = 0,  # 0 means None

    # XGBoost hyperparameters
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,

    # LSTM hyperparameters
    lstm_sequence_length: int = 50,
    lstm_epochs: int = 100,
    lstm_batch_size: int = 32,
    lstm_learning_rate: float = 0.001,

    # Evaluation
    selection_metric: str = "rmse",

    # General
    random_state: int = 42,
):
    """
    Predictive Maintenance Training Pipeline

    Pipeline DAG:

        data_ingestion
              |
              v
        feature_engineering
              |
        +-----------+-----------+
        |           |           |
        v           v           v
      train_rf  train_xgb  train_lstm  (parallel)
        |           |           |
        +-----------+-----------+
              |
              v
        evaluate_models
    """

    # Step 1: Data Ingestion
    data_ingestion_task = data_ingestion_op(
        dataset=dataset,
        data_path=data_path,
        max_rul=max_rul,
    )
    data_ingestion_task.set_display_name("Data Ingestion")
    data_ingestion_task.set_cpu_request("500m")
    data_ingestion_task.set_cpu_limit("1")
    data_ingestion_task.set_memory_request("1Gi")
    data_ingestion_task.set_memory_limit("2Gi")

    # Step 2: Feature Engineering (depends on data ingestion)
    feature_engineering_task = feature_engineering_op(
        processed_data=data_ingestion_task.outputs["processed_data"],
        rolling_windows=rolling_windows,
        lag_periods=lag_periods,
    )
    feature_engineering_task.set_display_name("Feature Engineering")
    feature_engineering_task.set_cpu_request("1")
    feature_engineering_task.set_cpu_limit("2")
    feature_engineering_task.set_memory_request("2Gi")
    feature_engineering_task.set_memory_limit("4Gi")
    feature_engineering_task.after(data_ingestion_task)

    # Step 3a: Train Random Forest (parallel with 3b, 3c)
    train_rf_task = train_rf_op(
        features=feature_engineering_task.outputs["features_output"],
        labels=feature_engineering_task.outputs["labels_output"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=random_state,
    )
    train_rf_task.set_display_name("Train Random Forest")
    train_rf_task.set_cpu_request("2")
    train_rf_task.set_cpu_limit("4")
    train_rf_task.set_memory_request("4Gi")
    train_rf_task.set_memory_limit("8Gi")
    train_rf_task.after(feature_engineering_task)

    # Step 3b: Train XGBoost (parallel with 3a, 3c)
    train_xgboost_task = train_xgboost_op(
        features=feature_engineering_task.outputs["features_output"],
        labels=feature_engineering_task.outputs["labels_output"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        random_state=random_state,
    )
    train_xgboost_task.set_display_name("Train XGBoost")
    train_xgboost_task.set_cpu_request("2")
    train_xgboost_task.set_cpu_limit("4")
    train_xgboost_task.set_memory_request("4Gi")
    train_xgboost_task.set_memory_limit("8Gi")
    train_xgboost_task.after(feature_engineering_task)

    # Step 3c: Train LSTM (parallel with 3a, 3b)
    train_lstm_task = train_lstm_op(
        features=feature_engineering_task.outputs["features_output"],
        labels=feature_engineering_task.outputs["labels_output"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        sequence_length=lstm_sequence_length,
        epochs=lstm_epochs,
        batch_size=lstm_batch_size,
        learning_rate=lstm_learning_rate,
        random_state=random_state,
    )
    train_lstm_task.set_display_name("Train LSTM")
    train_lstm_task.set_cpu_request("2")
    train_lstm_task.set_cpu_limit("4")
    train_lstm_task.set_memory_request("8Gi")
    train_lstm_task.set_memory_limit("16Gi")
    train_lstm_task.after(feature_engineering_task)

    # Step 4: Evaluate and Compare Models (depends on all training tasks)
    evaluate_task = evaluate_models_op(
        rf_model_info=train_rf_task.outputs["model_output"],
        xgb_model_info=train_xgboost_task.outputs["model_output"],
        lstm_model_info=train_lstm_task.outputs["model_output"],
        rf_rmse=train_rf_task.outputs["val_rmse"],
        rf_mae=train_rf_task.outputs["val_mae"],
        rf_r2=train_rf_task.outputs["val_r2"],
        xgb_rmse=train_xgboost_task.outputs["val_rmse"],
        xgb_mae=train_xgboost_task.outputs["val_mae"],
        xgb_r2=train_xgboost_task.outputs["val_r2"],
        lstm_rmse=train_lstm_task.outputs["val_rmse"],
        lstm_mae=train_lstm_task.outputs["val_mae"],
        lstm_r2=train_lstm_task.outputs["val_r2"],
        selection_metric=selection_metric,
    )
    evaluate_task.set_display_name("Evaluate & Select Best Model")
    evaluate_task.set_cpu_request("1")
    evaluate_task.set_cpu_limit("2")
    evaluate_task.set_memory_request("2Gi")
    evaluate_task.set_memory_limit("4Gi")
    evaluate_task.after(train_rf_task)
    evaluate_task.after(train_xgboost_task)
    evaluate_task.after(train_lstm_task)


# =============================================================================
# Alternative Pipeline using ContainerOp (for custom images)
# =============================================================================

def create_container_op_pipeline():
    """
    Alternative pipeline definition using ContainerOp for custom Docker images.
    Use this when you have pre-built component images.
    """
    from kfp import dsl
    from kfp.dsl import ContainerOp, PipelineVolume

    @dsl.pipeline(
        name="predictive-maintenance-training-containerop",
        description="Pipeline using ContainerOp with custom images",
    )
    def container_op_pipeline(
        data_path: str = "/data/raw",
        dataset: str = "FD001",
        mlflow_tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "predictive-maintenance",
    ):
        # Create shared volume for artifacts
        vop = dsl.VolumeOp(
            name="create-pvc",
            resource_name="pipeline-pvc",
            size="10Gi",
            modes=dsl.VOLUME_MODE_RWO,
        )

        # Data Ingestion
        data_ingestion = ContainerOp(
            name="data-ingestion",
            image=f"{CONTAINER_REGISTRY}/data-ingestion:{IMAGE_TAG}",
            command=["python", "/app/data_ingestion.py"],
            arguments=[
                "--dataset", dataset,
                "--data-path", data_path,
                "--output-path", "/data/processed/processed_data.csv",
            ],
            pvolumes={"/data": vop.volume},
        )
        data_ingestion.container.set_cpu_request("500m")
        data_ingestion.container.set_memory_request("1Gi")

        # Feature Engineering
        feature_engineering = ContainerOp(
            name="feature-engineering",
            image=f"{CONTAINER_REGISTRY}/feature-engineering:{IMAGE_TAG}",
            command=["python", "/app/feature_engineering.py"],
            arguments=[
                "--input-path", "/data/processed/processed_data.csv",
                "--features-output", "/data/features/features.csv",
                "--labels-output", "/data/features/labels.csv",
                "--scaler-output", "/data/features/scaler.pkl",
            ],
            pvolumes={"/data": data_ingestion.pvolume},
        )
        feature_engineering.container.set_cpu_request("1")
        feature_engineering.container.set_memory_request("2Gi")

        # Training tasks (parallel)
        train_rf = ContainerOp(
            name="train-random-forest",
            image=f"{CONTAINER_REGISTRY}/train-rf:{IMAGE_TAG}",
            command=["python", "/app/train_rf.py"],
            arguments=[
                "--features-path", "/data/features/features.csv",
                "--labels-path", "/data/features/labels.csv",
                "--model-output", "/data/models/rf_model_info.json",
                "--mlflow-tracking-uri", mlflow_tracking_uri,
                "--experiment-name", experiment_name,
            ],
            pvolumes={"/data": feature_engineering.pvolume},
        )

        train_xgboost = ContainerOp(
            name="train-xgboost",
            image=f"{CONTAINER_REGISTRY}/train-xgboost:{IMAGE_TAG}",
            command=["python", "/app/train_xgboost.py"],
            arguments=[
                "--features-path", "/data/features/features.csv",
                "--labels-path", "/data/features/labels.csv",
                "--model-output", "/data/models/xgb_model_info.json",
                "--mlflow-tracking-uri", mlflow_tracking_uri,
                "--experiment-name", experiment_name,
            ],
            pvolumes={"/data": feature_engineering.pvolume},
        )

        train_lstm = ContainerOp(
            name="train-lstm",
            image=f"{CONTAINER_REGISTRY}/train-lstm:{IMAGE_TAG}",
            command=["python", "/app/train_lstm.py"],
            arguments=[
                "--features-path", "/data/features/features.csv",
                "--labels-path", "/data/features/labels.csv",
                "--model-output", "/data/models/lstm_model_info.json",
                "--mlflow-tracking-uri", mlflow_tracking_uri,
                "--experiment-name", experiment_name,
            ],
            pvolumes={"/data": feature_engineering.pvolume},
        )
        train_lstm.container.set_memory_request("8Gi")

        # Evaluation
        evaluate = ContainerOp(
            name="evaluate-models",
            image=f"{CONTAINER_REGISTRY}/evaluate:{IMAGE_TAG}",
            command=["python", "/app/evaluate_models.py"],
            arguments=[
                "--rf-model-info", "/data/models/rf_model_info.json",
                "--xgb-model-info", "/data/models/xgb_model_info.json",
                "--lstm-model-info", "/data/models/lstm_model_info.json",
                "--features-path", "/data/features/features.csv",
                "--labels-path", "/data/features/labels.csv",
                "--output-path", "/data/evaluation/best_model_info.json",
            ],
            pvolumes={"/data": train_rf.pvolume},
        )
        evaluate.after(train_rf)
        evaluate.after(train_xgboost)
        evaluate.after(train_lstm)

    return container_op_pipeline


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # This allows testing the pipeline locally
    from kfp import compiler

    # Compile the main pipeline
    compiler.Compiler().compile(
        pipeline_func=predictive_maintenance_pipeline,
        package_path="pipeline.yaml",
    )
    print("Pipeline compiled to pipeline.yaml")
