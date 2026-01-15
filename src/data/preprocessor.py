"""
Feature engineering and preprocessing for predictive maintenance models.

This module provides the FeatureEngineer class for transforming raw sensor data
into features suitable for machine learning models. It includes methods for
creating rolling statistics, lag features, rate of change features, and RUL targets.

Example:
    >>> from src.data.preprocessor import FeatureEngineer
    >>> from src.data.loader import DataLoader
    >>>
    >>> loader = DataLoader()
    >>> train_df = loader.load_train("FD001")
    >>>
    >>> fe = FeatureEngineer()
    >>> train_df = fe.create_rul_target(train_df)
    >>> train_df = fe.create_rolling_features(train_df)
    >>> train_df = fe.create_lag_features(train_df)
    >>> X_scaled = fe.normalize_features(train_df, fit=True)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import SENSOR_COLUMNS, settings

logger = logging.getLogger(__name__)


class FeatureEngineerError(Exception):
    """Exception raised for feature engineering errors."""

    pass


class FeatureEngineer:
    """
    Feature engineering pipeline for turbofan engine data.

    This class provides methods to transform raw sensor readings into
    engineered features for predictive maintenance models. Features include:
    - Rolling window statistics (mean, std, min, max)
    - Lag features for time-series patterns
    - Rate of change features
    - Normalized features using StandardScaler

    Attributes:
        rolling_windows: List of window sizes for rolling features
        lag_periods: List of lag periods for lag features
        scaler: StandardScaler instance for normalization
        feature_columns: List of feature columns after transformation
        is_fitted: Whether the scaler has been fitted

    Example:
        >>> fe = FeatureEngineer()
        >>> df = fe.create_rul_target(df)
        >>> df = fe.create_rolling_features(df)
        >>> X = fe.normalize_features(df, fit=True)
        >>> fe.save_scaler("artifacts/scaler.joblib")
    """

    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        lag_periods: Optional[List[int]] = None,
        max_rul: int = 125,
    ) -> None:
        """
        Initialize the FeatureEngineer.

        Args:
            rolling_windows: Window sizes for rolling statistics.
                           Defaults to [5, 10, 20]
            lag_periods: Lag periods for lag features.
                        Defaults to [1, 5, 10]
            max_rul: Maximum RUL value (values above this are clipped).
                    Defaults to 125
        """
        self.rolling_windows = rolling_windows or settings.features.rolling_windows
        self.lag_periods = lag_periods or settings.features.lag_periods
        self.max_rul = max_rul

        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

        # Track which features were created
        self._created_features: Dict[str, List[str]] = {
            "rolling": [],
            "lag": [],
            "rate_of_change": [],
            "original": [],
        }

    def create_rul_target(
        self,
        df: pd.DataFrame,
        clip: bool = True,
    ) -> pd.DataFrame:
        """
        Create Remaining Useful Life (RUL) target variable.

        RUL is calculated as the difference between the maximum cycle
        for each unit and the current cycle. Optionally clips RUL values
        to a maximum threshold (piecewise linear degradation assumption).

        Args:
            df: DataFrame with 'unit_id' and 'time_cycle' columns
            clip: Whether to clip RUL values to max_rul. Defaults to True

        Returns:
            DataFrame with added 'RUL' column

        Raises:
            FeatureEngineerError: If required columns are missing

        Example:
            >>> fe = FeatureEngineer()
            >>> df = fe.create_rul_target(df, clip=True)
            >>> print(df['RUL'].max())
            125
        """
        self._validate_required_columns(df, ["unit_id", "time_cycle"])

        df = df.copy()

        # Calculate max cycle for each unit
        max_cycles = df.groupby("unit_id")["time_cycle"].transform("max")

        # RUL = max_cycle - current_cycle
        df["RUL"] = max_cycles - df["time_cycle"]

        # Clip RUL values if requested
        if clip:
            df["RUL"] = df["RUL"].clip(upper=self.max_rul)
            logger.info(f"Created RUL target with clipping at {self.max_rul}")
        else:
            logger.info("Created RUL target without clipping")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create rolling window statistics for sensor columns.

        For each specified column and window size, creates:
        - Rolling mean
        - Rolling standard deviation
        - Rolling minimum
        - Rolling maximum

        Args:
            df: DataFrame with sensor columns
            columns: Columns to create rolling features for.
                    Defaults to all sensor columns
            windows: Window sizes to use. Defaults to self.rolling_windows

        Returns:
            DataFrame with added rolling feature columns

        Example:
            >>> fe = FeatureEngineer()
            >>> df = fe.create_rolling_features(df, windows=[5, 10, 20])
            >>> # Creates columns like: sensor_2_rolling_mean_5, sensor_2_rolling_std_5, etc.
        """
        df = df.copy()
        columns = columns or self._get_sensor_columns(df)
        windows = windows or self.rolling_windows

        self._created_features["rolling"] = []

        for window in windows:
            logger.debug(f"Creating rolling features with window={window}")

            for col in columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found, skipping")
                    continue

                # Group by unit_id to prevent data leakage between units
                grouped = df.groupby("unit_id")[col]

                # Rolling mean
                mean_col = f"{col}_rolling_mean_{window}"
                df[mean_col] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                self._created_features["rolling"].append(mean_col)

                # Rolling std
                std_col = f"{col}_rolling_std_{window}"
                df[std_col] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                # Fill NaN in std (first few values)
                df[std_col] = df[std_col].fillna(0)
                self._created_features["rolling"].append(std_col)

                # Rolling min
                min_col = f"{col}_rolling_min_{window}"
                df[min_col] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                self._created_features["rolling"].append(min_col)

                # Rolling max
                max_col = f"{col}_rolling_max_{window}"
                df[max_col] = grouped.transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                self._created_features["rolling"].append(max_col)

        logger.info(
            f"Created {len(self._created_features['rolling'])} rolling features "
            f"for {len(columns)} columns with windows {windows}"
        )

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create lag features for time-series patterns.

        For each specified column and lag period, creates a new column
        with the value from N cycles ago.

        Args:
            df: DataFrame with sensor columns
            columns: Columns to create lag features for.
                    Defaults to all sensor columns
            lags: Lag periods to use. Defaults to self.lag_periods

        Returns:
            DataFrame with added lag feature columns

        Example:
            >>> fe = FeatureEngineer()
            >>> df = fe.create_lag_features(df, lags=[1, 5, 10])
            >>> # Creates columns like: sensor_2_lag_1, sensor_2_lag_5, etc.
        """
        df = df.copy()
        columns = columns or self._get_sensor_columns(df)
        lags = lags or self.lag_periods

        self._created_features["lag"] = []

        for lag in lags:
            logger.debug(f"Creating lag features with lag={lag}")

            for col in columns:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found, skipping")
                    continue

                # Group by unit_id to prevent data leakage between units
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df.groupby("unit_id")[col].shift(lag)

                # Fill NaN values with original value (for first few rows)
                df[lag_col] = df[lag_col].fillna(df[col])

                self._created_features["lag"].append(lag_col)

        logger.info(
            f"Created {len(self._created_features['lag'])} lag features "
            f"for {len(columns)} columns with lags {lags}"
        )

        return df

    def create_rate_of_change(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create rate of change (delta) features.

        For each specified column, creates a new column with the
        difference between the current value and the previous value.

        Args:
            df: DataFrame with sensor columns
            columns: Columns to create rate of change for.
                    Defaults to all sensor columns

        Returns:
            DataFrame with added rate of change columns

        Example:
            >>> fe = FeatureEngineer()
            >>> df = fe.create_rate_of_change(df)
            >>> # Creates columns like: sensor_2_delta, sensor_3_delta, etc.
        """
        df = df.copy()
        columns = columns or self._get_sensor_columns(df)

        self._created_features["rate_of_change"] = []

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            # Group by unit_id to prevent data leakage between units
            delta_col = f"{col}_delta"
            df[delta_col] = df.groupby("unit_id")[col].diff()

            # Fill NaN values with 0 (first row of each unit)
            df[delta_col] = df[delta_col].fillna(0)

            self._created_features["rate_of_change"].append(delta_col)

        logger.info(
            f"Created {len(self._created_features['rate_of_change'])} "
            f"rate of change features"
        )

        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Normalize features using StandardScaler.

        Args:
            df: DataFrame with feature columns
            columns: Columns to normalize. Defaults to all numeric columns
                    except 'unit_id', 'time_cycle', and 'RUL'
            fit: Whether to fit the scaler on this data. Set True for
                training data, False for test data.

        Returns:
            NumPy array of normalized features

        Raises:
            FeatureEngineerError: If fit=False and scaler is not fitted

        Example:
            >>> fe = FeatureEngineer()
            >>> X_train = fe.normalize_features(train_df, fit=True)
            >>> X_test = fe.normalize_features(test_df, fit=False)
        """
        # Determine columns to normalize
        if columns is None:
            exclude_cols = {"unit_id", "time_cycle", "RUL", "max_cycle"}
            columns = [
                col for col in df.columns
                if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            ]

        self.feature_columns = columns

        if fit:
            logger.info(f"Fitting scaler on {len(columns)} features")
            X = self.scaler.fit_transform(df[columns])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise FeatureEngineerError(
                    "Scaler is not fitted. Call normalize_features with fit=True first."
                )
            logger.info(f"Transforming {len(columns)} features")
            X = self.scaler.transform(df[columns])

        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Fit scaler and transform features (convenience method).

        Args:
            df: DataFrame with feature columns
            columns: Columns to normalize

        Returns:
            NumPy array of normalized features
        """
        return self.normalize_features(df, columns=columns, fit=True)

    def transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Transform features using fitted scaler (convenience method).

        Args:
            df: DataFrame with feature columns
            columns: Columns to normalize

        Returns:
            NumPy array of normalized features
        """
        columns = columns or self.feature_columns
        return self.normalize_features(df, columns=columns, fit=False)

    def save_scaler(self, path: Union[str, Path]) -> None:
        """
        Save fitted scaler to disk.

        Args:
            path: Path to save the scaler

        Raises:
            FeatureEngineerError: If scaler is not fitted

        Example:
            >>> fe.save_scaler("artifacts/scaler.joblib")
        """
        if not self.is_fitted:
            raise FeatureEngineerError(
                "Cannot save unfitted scaler. Call normalize_features with fit=True first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save scaler and metadata
        save_data = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "rolling_windows": self.rolling_windows,
            "lag_periods": self.lag_periods,
            "max_rul": self.max_rul,
            "created_features": self._created_features,
        }

        joblib.dump(save_data, path)
        logger.info(f"Saved scaler and metadata to {path}")

    def load_scaler(self, path: Union[str, Path]) -> "FeatureEngineer":
        """
        Load fitted scaler from disk.

        Args:
            path: Path to load the scaler from

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If scaler file doesn't exist

        Example:
            >>> fe = FeatureEngineer()
            >>> fe.load_scaler("artifacts/scaler.joblib")
            >>> X_test = fe.transform(test_df)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        save_data = joblib.load(path)

        self.scaler = save_data["scaler"]
        self.feature_columns = save_data["feature_columns"]
        self.rolling_windows = save_data["rolling_windows"]
        self.lag_periods = save_data["lag_periods"]
        self.max_rul = save_data["max_rul"]
        self._created_features = save_data.get("created_features", {})
        self.is_fitted = True

        logger.info(f"Loaded scaler and metadata from {path}")

        return self

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/sequence models.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            sequence_length: Length of each sequence

        Returns:
            Tuple of (X_sequences, y_sequences) where:
            - X_sequences has shape (n_sequences, sequence_length, n_features)
            - y_sequences has shape (n_sequences,)

        Example:
            >>> X_seq, y_seq = fe.create_sequences(X, y, sequence_length=50)
            >>> print(X_seq.shape)  # (n_sequences, 50, n_features)
        """
        if len(X) < sequence_length:
            raise FeatureEngineerError(
                f"Not enough samples ({len(X)}) for sequence length {sequence_length}"
            )

        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X[i : i + sequence_length])
            y_seq.append(y[i + sequence_length - 1])

        return np.array(X_seq), np.array(y_seq)

    def create_sequences_by_unit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "RUL",
        sequence_length: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences respecting unit boundaries.

        Unlike create_sequences, this method ensures sequences don't
        cross unit boundaries.

        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Name of target column
            sequence_length: Length of each sequence

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_all, y_all = [], []

        for unit_id in df["unit_id"].unique():
            unit_data = df[df["unit_id"] == unit_id]

            if len(unit_data) < sequence_length:
                logger.debug(f"Unit {unit_id} has fewer than {sequence_length} samples, skipping")
                continue

            X = unit_data[feature_columns].values
            y = unit_data[target_column].values

            X_seq, y_seq = self.create_sequences(X, y, sequence_length)
            X_all.append(X_seq)
            y_all.append(y_seq)

        return np.vstack(X_all), np.concatenate(y_all)

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature column names.

        Returns:
            List of feature column names
        """
        return self.feature_columns.copy()

    def get_created_features(self) -> Dict[str, List[str]]:
        """
        Get dictionary of created features by type.

        Returns:
            Dictionary mapping feature type to list of column names
        """
        return self._created_features.copy()

    def _get_sensor_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get sensor columns present in DataFrame.

        Args:
            df: DataFrame to check

        Returns:
            List of sensor column names present in DataFrame
        """
        return [col for col in SENSOR_COLUMNS if col in df.columns]

    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        required: List[str],
    ) -> None:
        """
        Validate that required columns exist in DataFrame.

        Args:
            df: DataFrame to validate
            required: List of required column names

        Raises:
            FeatureEngineerError: If any required columns are missing
        """
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise FeatureEngineerError(f"Missing required columns: {missing}")

    def drop_constant_columns(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Drop columns with near-constant values.

        Args:
            df: DataFrame to process
            threshold: Columns with std < threshold * mean are dropped

        Returns:
            DataFrame with constant columns removed
        """
        df = df.copy()
        sensor_cols = self._get_sensor_columns(df)
        cols_to_drop = []

        for col in sensor_cols:
            std = df[col].std()
            mean = abs(df[col].mean())

            if mean > 0 and std / mean < threshold:
                cols_to_drop.append(col)
            elif std == 0:
                cols_to_drop.append(col)

        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} near-constant columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        return df

    def full_pipeline(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        create_rul: bool = True,
        drop_constant: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the full feature engineering pipeline.

        Args:
            df: Raw DataFrame from DataLoader
            fit: Whether to fit the scaler
            create_rul: Whether to create RUL target
            drop_constant: Whether to drop constant columns

        Returns:
            Tuple of (X, y) where y is None if create_rul is False

        Example:
            >>> fe = FeatureEngineer()
            >>> X_train, y_train = fe.full_pipeline(train_df, fit=True)
            >>> X_test, _ = fe.full_pipeline(test_df, fit=False, create_rul=False)
        """
        df = df.copy()

        # Drop constant columns
        if drop_constant:
            df = self.drop_constant_columns(df)

        # Create RUL target
        if create_rul:
            df = self.create_rul_target(df)

        # Create features
        df = self.create_rolling_features(df)
        df = self.create_lag_features(df)
        df = self.create_rate_of_change(df)

        # Drop rows with NaN (from feature creation)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")

        # Normalize
        X = self.normalize_features(df, fit=fit)

        # Get target if created
        y = df["RUL"].values if create_rul else None

        return X, y
