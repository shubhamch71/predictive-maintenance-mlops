#!/usr/bin/env python3
"""
Kubeflow Component: Feature Engineering

Reads processed data from data ingestion component and applies
feature engineering transformations.

Usage:
    python feature_engineering.py \
        --input-path /data/processed/processed_data.csv \
        --features-output /data/features/features.csv \
        --labels-output /data/features/labels.csv \
        --scaler-output /data/features/scaler.pkl

Outputs:
    - features.csv: Normalized feature matrix
    - labels.csv: RUL labels
    - scaler.pkl: Fitted StandardScaler for inference
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("feature_engineering")

# Default configuration
DEFAULT_ROLLING_WINDOWS = [5, 10, 20]
DEFAULT_LAG_PERIODS = [1, 5, 10]
SENSORS_TO_USE = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """
    Create rolling window statistics features.

    Args:
        df: Input DataFrame
        columns: Columns to compute rolling features for
        windows: List of window sizes

    Returns:
        DataFrame with rolling features added
    """
    logger.info(f"Creating rolling features for {len(columns)} columns, windows={windows}")

    df = df.copy()
    grouped = df.groupby("unit_id")

    for col in columns:
        for window in windows:
            # Rolling mean
            mean_col = f"{col}_rolling_mean_{window}"
            df[mean_col] = grouped[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # Rolling std
            std_col = f"{col}_rolling_std_{window}"
            df[std_col] = grouped[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

    logger.info(f"Added {len(columns) * len(windows) * 2} rolling features")
    return df


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
    """
    Create lag features for time series.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with lag features added
    """
    logger.info(f"Creating lag features for {len(columns)} columns, lags={lags}")

    df = df.copy()
    grouped = df.groupby("unit_id")

    for col in columns:
        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = grouped[col].transform(lambda x: x.shift(lag))

    logger.info(f"Added {len(columns) * len(lags)} lag features")
    return df


def create_rate_of_change(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Create rate of change (delta) features.

    Args:
        df: Input DataFrame
        columns: Columns to compute rate of change for

    Returns:
        DataFrame with rate of change features added
    """
    logger.info(f"Creating rate of change features for {len(columns)} columns")

    df = df.copy()
    grouped = df.groupby("unit_id")

    for col in columns:
        delta_col = f"{col}_delta"
        df[delta_col] = grouped[col].transform(lambda x: x.diff())

    logger.info(f"Added {len(columns)} rate of change features")
    return df


def normalize_features(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.

    Args:
        X: Feature matrix
        scaler: Existing scaler (if not fitting)
        fit: Whether to fit the scaler

    Returns:
        Tuple of (normalized features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        logger.info("Fitting and transforming features with StandardScaler")
        X_normalized = scaler.fit_transform(X)
    else:
        logger.info("Transforming features with existing scaler")
        X_normalized = scaler.transform(X)

    return X_normalized, scaler


def log_feature_statistics(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """Log statistics about engineered features."""
    logger.info("=" * 50)
    logger.info("Feature Statistics:")
    logger.info(f"  Total features: {len(feature_columns)}")
    logger.info(f"  Total samples: {len(df)}")

    # Sample statistics
    feature_df = df[feature_columns]
    logger.info(f"  Feature matrix shape: {feature_df.shape}")
    logger.info(f"  Memory usage: {feature_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Missing values after feature engineering
    missing = feature_df.isnull().sum().sum()
    logger.info(f"  Missing values (before dropna): {missing}")

    # Feature value ranges
    logger.info("  Feature value ranges (sample of 5):")
    for col in feature_columns[:5]:
        logger.info(f"    {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")

    logger.info("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kubeflow Component: Feature Engineering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to processed_data.csv from data ingestion",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        required=True,
        help="Path to save features.csv",
    )
    parser.add_argument(
        "--labels-output",
        type=str,
        required=True,
        help="Path to save labels.csv",
    )
    parser.add_argument(
        "--scaler-output",
        type=str,
        required=True,
        help="Path to save scaler.pkl",
    )
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default="5,10,20",
        help="Comma-separated rolling window sizes",
    )
    parser.add_argument(
        "--lag-periods",
        type=str,
        default="1,5,10",
        help="Comma-separated lag periods",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Kubeflow Component: Feature Engineering")
    logger.info("=" * 60)
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Features output: {args.features_output}")
    logger.info(f"Labels output: {args.labels_output}")
    logger.info(f"Scaler output: {args.scaler_output}")

    try:
        # Parse configuration
        rolling_windows = [int(x) for x in args.rolling_windows.split(",")]
        lag_periods = [int(x) for x in args.lag_periods.split(",")]

        logger.info(f"Rolling windows: {rolling_windows}")
        logger.info(f"Lag periods: {lag_periods}")

        # Load input data
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Determine base feature columns
        base_columns = SENSORS_TO_USE + OP_SETTINGS

        # Apply feature engineering
        logger.info("Applying feature engineering transformations...")

        # Rolling features
        df = create_rolling_features(df, SENSORS_TO_USE, rolling_windows)

        # Lag features
        df = create_lag_features(df, SENSORS_TO_USE, lag_periods)

        # Rate of change features
        df = create_rate_of_change(df, SENSORS_TO_USE)

        # Drop rows with NaN (from rolling/lag operations)
        original_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {original_len - len(df)} rows with NaN values")

        # Determine all feature columns (exclude non-features)
        exclude_cols = ["unit_id", "time_cycle", "RUL"]
        feature_columns = [c for c in df.columns if c not in exclude_cols]

        # Log feature statistics
        log_feature_statistics(df, feature_columns)

        # Extract features and labels
        X = df[feature_columns].values
        y = df["RUL"].values

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")

        # Normalize features
        X_normalized, scaler = normalize_features(X, fit=True)

        # Create output directories
        features_path = Path(args.features_output)
        labels_path = Path(args.labels_output)
        scaler_path = Path(args.scaler_output)

        features_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        # Save features with column names
        features_df = pd.DataFrame(X_normalized, columns=feature_columns)
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved features to {features_path}")

        # Save labels
        labels_df = pd.DataFrame({"RUL": y})
        labels_df.to_csv(labels_path, index=False)
        logger.info(f"Saved labels to {labels_path}")

        # Save scaler with metadata
        scaler_data = {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "rolling_windows": rolling_windows,
            "lag_periods": lag_periods,
        }
        joblib.dump(scaler_data, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        logger.info("Feature engineering completed successfully!")
        logger.info(f"Output shapes: features={X_normalized.shape}, labels={y.shape}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Feature engineering failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
