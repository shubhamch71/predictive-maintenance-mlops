"""
Data loading utilities for NASA Turbofan Engine Degradation dataset.

This module provides the DataLoader class for loading and parsing the C-MAPSS
(Commercial Modular Aero-Propulsion System Simulation) dataset. The dataset
contains run-to-failure simulated data from turbofan engine degradation simulation.

Dataset Structure:
    - Each row represents a snapshot of data at a given cycle
    - Columns: unit_id, time_cycle, 3 operational settings, 21 sensors
    - Training data: Complete run-to-failure trajectories
    - Test data: Truncated trajectories (predict remaining cycles)
    - RUL files: True remaining cycles for test data

Example:
    >>> from src.data.loader import DataLoader
    >>> loader = DataLoader()
    >>> train_df = loader.load_train("FD001")
    >>> test_df = loader.load_test("FD001")
    >>> rul = loader.load_rul("FD001")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    AVAILABLE_DATASETS,
    COLUMN_NAMES,
    settings,
)

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Exception raised when data loading fails."""

    pass


class DataValidationError(Exception):
    """Exception raised when data validation fails."""

    pass


class DataLoader:
    """
    Load and parse NASA Turbofan Engine Degradation dataset.

    The DataLoader handles reading the space-delimited text files from the NASA
    C-MAPSS dataset and returns properly formatted pandas DataFrames with
    appropriate column names and data types.

    Attributes:
        data_dir: Path to the directory containing raw data files
        column_names: List of column names for the dataset

    Example:
        >>> loader = DataLoader()
        >>> train_df = loader.load_train("FD001")
        >>> print(train_df.shape)
        (20631, 26)

        >>> # Load all datasets at once
        >>> train, test, rul = loader.load_all("FD001")
    """

    # Column names for the NASA Turbofan dataset
    COLUMN_NAMES: List[str] = COLUMN_NAMES

    # Number of expected columns (before dropping NaN columns)
    EXPECTED_COLUMNS: int = 26

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize the DataLoader.

        Args:
            data_dir: Path to directory containing raw data files.
                     Defaults to settings.paths.raw_data_dir
        """
        self.data_dir = Path(data_dir) if data_dir else settings.paths.raw_data_dir
        self._validate_data_dir()

    def _validate_data_dir(self) -> None:
        """
        Validate that the data directory exists.

        Raises:
            DataLoadError: If data directory doesn't exist
        """
        if not self.data_dir.exists():
            logger.warning(
                f"Data directory does not exist: {self.data_dir}. "
                "Run download_data.py to fetch the dataset."
            )

    def _get_file_path(self, prefix: str, dataset: str) -> Path:
        """
        Get the path to a data file.

        Args:
            prefix: File prefix ('train', 'test', or 'RUL')
            dataset: Dataset name (e.g., 'FD001')

        Returns:
            Path to the data file

        Raises:
            DataLoadError: If file doesn't exist
        """
        file_path = self.data_dir / f"{prefix}_{dataset}.txt"

        if not file_path.exists():
            raise DataLoadError(
                f"Data file not found: {file_path}. "
                f"Please run 'python data/download_data.py' to download the dataset."
            )

        return file_path

    def _validate_dataset_name(self, dataset: str) -> None:
        """
        Validate the dataset name.

        Args:
            dataset: Dataset name to validate

        Raises:
            ValueError: If dataset name is invalid
        """
        if dataset not in AVAILABLE_DATASETS:
            raise ValueError(
                f"Invalid dataset: {dataset}. "
                f"Available datasets: {AVAILABLE_DATASETS}"
            )

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a space-delimited data file.

        The NASA dataset files are space-delimited with no header.
        Some files have trailing spaces creating extra NaN columns.

        Args:
            file_path: Path to the data file

        Returns:
            DataFrame with proper column names

        Raises:
            DataLoadError: If file cannot be parsed
        """
        try:
            logger.debug(f"Loading file: {file_path}")

            df = pd.read_csv(
                file_path,
                sep=r"\s+",  # Handle multiple spaces
                header=None,
                names=self.COLUMN_NAMES,
                index_col=False,
                dtype={
                    "unit_id": np.int32,
                    "time_cycle": np.int32,
                },
            )

            # Remove any trailing NaN columns (from trailing spaces in file)
            df = df.dropna(axis=1, how="all")

            # Validate column count
            if len(df.columns) != self.EXPECTED_COLUMNS:
                logger.warning(
                    f"Expected {self.EXPECTED_COLUMNS} columns, got {len(df.columns)}"
                )

            logger.info(f"Loaded {len(df)} records from {file_path.name}")

            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load {file_path}: {e}")

    def _validate_dataframe(
        self, df: pd.DataFrame, dataset: str, data_type: str
    ) -> None:
        """
        Validate loaded DataFrame.

        Args:
            df: DataFrame to validate
            dataset: Dataset name
            data_type: Type of data ('train' or 'test')

        Raises:
            DataValidationError: If validation fails
        """
        # Check for empty DataFrame
        if df.empty:
            raise DataValidationError(f"Empty DataFrame loaded for {dataset} {data_type}")

        # Check required columns exist
        required_cols = ["unit_id", "time_cycle"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Check for NaN values in critical columns
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            raise DataValidationError(f"NaN values found in critical columns: {nan_counts}")

        # Validate unit_id is positive
        if (df["unit_id"] <= 0).any():
            raise DataValidationError("Invalid unit_id values detected (must be positive)")

        # Validate time_cycle is positive
        if (df["time_cycle"] <= 0).any():
            raise DataValidationError("Invalid time_cycle values detected (must be positive)")

        logger.debug(f"Validation passed for {dataset} {data_type}")

    def load_train(self, dataset: str = "FD001") -> pd.DataFrame:
        """
        Load training data for specified dataset.

        Training data contains complete run-to-failure trajectories where
        each unit runs until failure. The last cycle of each unit represents
        the failure point (RUL = 0).

        Args:
            dataset: Dataset name ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
            DataFrame with columns: unit_id, time_cycle, op_setting_1-3, sensor_1-21

        Raises:
            ValueError: If dataset name is invalid
            DataLoadError: If file cannot be loaded

        Example:
            >>> loader = DataLoader()
            >>> train_df = loader.load_train("FD001")
            >>> print(f"Units: {train_df['unit_id'].nunique()}")
            Units: 100
        """
        self._validate_dataset_name(dataset)
        file_path = self._get_file_path("train", dataset)
        df = self._load_file(file_path)
        self._validate_dataframe(df, dataset, "train")
        return df

    def load_test(self, dataset: str = "FD001") -> pd.DataFrame:
        """
        Load test data for specified dataset.

        Test data contains truncated trajectories where each unit is cut off
        at some point before failure. The goal is to predict how many cycles
        remain before failure (RUL).

        Args:
            dataset: Dataset name ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
            DataFrame with columns: unit_id, time_cycle, op_setting_1-3, sensor_1-21

        Raises:
            ValueError: If dataset name is invalid
            DataLoadError: If file cannot be loaded

        Example:
            >>> loader = DataLoader()
            >>> test_df = loader.load_test("FD001")
            >>> print(f"Units: {test_df['unit_id'].nunique()}")
            Units: 100
        """
        self._validate_dataset_name(dataset)
        file_path = self._get_file_path("test", dataset)
        df = self._load_file(file_path)
        self._validate_dataframe(df, dataset, "test")
        return df

    def load_rul(self, dataset: str = "FD001") -> np.ndarray:
        """
        Load RUL (Remaining Useful Life) labels for test data.

        The RUL file contains one value per test unit, representing
        the true remaining cycles at the end of each test trajectory.

        Args:
            dataset: Dataset name ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
            NumPy array of RUL values (one per test unit)

        Raises:
            ValueError: If dataset name is invalid
            DataLoadError: If file cannot be loaded

        Example:
            >>> loader = DataLoader()
            >>> rul = loader.load_rul("FD001")
            >>> print(f"RUL values: {len(rul)}")
            RUL values: 100
        """
        self._validate_dataset_name(dataset)
        file_path = self._get_file_path("RUL", dataset)

        try:
            rul = np.loadtxt(file_path, dtype=np.int32)
            logger.info(f"Loaded {len(rul)} RUL values from {file_path.name}")
            return rul

        except Exception as e:
            raise DataLoadError(f"Failed to load RUL file {file_path}: {e}")

    def load_all(
        self, dataset: str = "FD001"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Load all data (train, test, RUL) for a dataset.

        Convenience method to load all three data files at once.

        Args:
            dataset: Dataset name ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
            Tuple of (train_df, test_df, rul_array)

        Example:
            >>> loader = DataLoader()
            >>> train, test, rul = loader.load_all("FD001")
            >>> print(f"Train: {train.shape}, Test: {test.shape}, RUL: {len(rul)}")
        """
        train = self.load_train(dataset)
        test = self.load_test(dataset)
        rul = self.load_rul(dataset)

        return train, test, rul

    def get_dataset_info(self, dataset: str = "FD001") -> Dict[str, Union[int, float]]:
        """
        Get summary information about a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary with dataset statistics

        Example:
            >>> loader = DataLoader()
            >>> info = loader.get_dataset_info("FD001")
            >>> print(info)
        """
        train = self.load_train(dataset)
        test = self.load_test(dataset)
        rul = self.load_rul(dataset)

        return {
            "train_samples": len(train),
            "test_samples": len(test),
            "train_units": train["unit_id"].nunique(),
            "test_units": test["unit_id"].nunique(),
            "train_max_cycle": int(train["time_cycle"].max()),
            "test_max_cycle": int(test["time_cycle"].max()),
            "rul_min": int(rul.min()),
            "rul_max": int(rul.max()),
            "rul_mean": float(rul.mean()),
        }

    def get_unit_data(
        self, df: pd.DataFrame, unit_id: int
    ) -> pd.DataFrame:
        """
        Extract data for a specific unit.

        Args:
            df: DataFrame containing multiple units
            unit_id: ID of the unit to extract

        Returns:
            DataFrame containing only the specified unit's data

        Raises:
            ValueError: If unit_id not found in DataFrame
        """
        if unit_id not in df["unit_id"].values:
            raise ValueError(f"Unit {unit_id} not found in DataFrame")

        return df[df["unit_id"] == unit_id].copy()

    @property
    def available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return AVAILABLE_DATASETS.copy()

    def check_data_availability(self) -> Dict[str, bool]:
        """
        Check which datasets are available locally.

        Returns:
            Dictionary mapping dataset names to availability status

        Example:
            >>> loader = DataLoader()
            >>> availability = loader.check_data_availability()
            >>> print(availability)
            {'FD001': True, 'FD002': True, 'FD003': True, 'FD004': True}
        """
        availability = {}

        for dataset in AVAILABLE_DATASETS:
            try:
                train_path = self.data_dir / f"train_{dataset}.txt"
                test_path = self.data_dir / f"test_{dataset}.txt"
                rul_path = self.data_dir / f"RUL_{dataset}.txt"

                availability[dataset] = (
                    train_path.exists()
                    and test_path.exists()
                    and rul_path.exists()
                )
            except Exception:
                availability[dataset] = False

        return availability
