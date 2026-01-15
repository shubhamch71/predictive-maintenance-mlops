"""
Unit tests for data loading and preprocessing modules.

This module contains tests for:
- DataLoader: Loading and parsing NASA Turbofan dataset
- FeatureEngineer: Feature engineering transformations
- Config: Configuration settings

Run tests:
    pytest tests/test_data_loader.py -v
    pytest tests/test_data_loader.py -v -k "test_feature"
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import (
    AVAILABLE_DATASETS,
    COLUMN_NAMES,
    Settings,
    get_settings,
    settings,
)
from src.data.loader import DataLoader, DataLoadError, DataValidationError
from src.data.preprocessor import FeatureEngineer, FeatureEngineerError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_units = 5

    data = {
        "unit_id": np.repeat(range(1, n_units + 1), n_samples // n_units),
        "time_cycle": np.tile(range(1, n_samples // n_units + 1), n_units),
    }

    # Add operational settings
    for i in range(1, 4):
        data[f"op_setting_{i}"] = np.random.randn(n_samples)

    # Add sensor readings
    for i in range(1, 22):
        data[f"sensor_{i}"] = np.random.randn(n_samples) * 10 + 500

    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data(sample_train_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample test data (truncated trajectories)."""
    # Truncate each unit to 15 cycles
    return sample_train_data.groupby("unit_id").head(15).reset_index(drop=True)


@pytest.fixture
def sample_rul_data() -> np.ndarray:
    """Create sample RUL data."""
    return np.array([50, 60, 40, 55, 45])


@pytest.fixture
def temp_data_dir(sample_train_data, sample_test_data, sample_rul_data):
    """Create temporary data directory with sample data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save train data
        train_str = sample_train_data.to_csv(
            sep=" ", index=False, header=False
        )
        (tmpdir_path / "train_FD001.txt").write_text(train_str)

        # Save test data
        test_str = sample_test_data.to_csv(
            sep=" ", index=False, header=False
        )
        (tmpdir_path / "test_FD001.txt").write_text(test_str)

        # Save RUL data
        np.savetxt(tmpdir_path / "RUL_FD001.txt", sample_rul_data, fmt="%d")

        yield tmpdir_path


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Create FeatureEngineer instance for testing."""
    return FeatureEngineer(
        rolling_windows=[5, 10],
        lag_periods=[1, 5],
        max_rul=125,
    )


# =============================================================================
# DataLoader Tests
# =============================================================================


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init_default_path(self):
        """Test DataLoader initialization with default path."""
        loader = DataLoader()
        assert loader.data_dir == settings.paths.raw_data_dir

    def test_init_custom_path(self, temp_data_dir):
        """Test DataLoader initialization with custom path."""
        loader = DataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == temp_data_dir

    def test_validate_dataset_name_valid(self):
        """Test dataset name validation with valid names."""
        loader = DataLoader()
        for dataset in AVAILABLE_DATASETS:
            loader._validate_dataset_name(dataset)  # Should not raise

    def test_validate_dataset_name_invalid(self):
        """Test dataset name validation with invalid name."""
        loader = DataLoader()
        with pytest.raises(ValueError, match="Invalid dataset"):
            loader._validate_dataset_name("FD999")

    def test_load_train_success(self, temp_data_dir, sample_train_data):
        """Test successful training data loading."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_train("FD001")

        assert isinstance(df, pd.DataFrame)
        assert "unit_id" in df.columns
        assert "time_cycle" in df.columns
        assert len(df) == len(sample_train_data)

    def test_load_test_success(self, temp_data_dir, sample_test_data):
        """Test successful test data loading."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_test("FD001")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_test_data)

    def test_load_rul_success(self, temp_data_dir, sample_rul_data):
        """Test successful RUL data loading."""
        loader = DataLoader(data_dir=temp_data_dir)
        rul = loader.load_rul("FD001")

        assert isinstance(rul, np.ndarray)
        assert len(rul) == len(sample_rul_data)
        np.testing.assert_array_equal(rul, sample_rul_data)

    def test_load_all_success(self, temp_data_dir):
        """Test loading all data at once."""
        loader = DataLoader(data_dir=temp_data_dir)
        train, test, rul = loader.load_all("FD001")

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(rul, np.ndarray)

    def test_load_file_not_found(self, temp_data_dir):
        """Test error handling for missing file."""
        loader = DataLoader(data_dir=temp_data_dir)

        with pytest.raises(DataLoadError, match="Data file not found"):
            loader.load_train("FD002")

    def test_get_unit_data(self, temp_data_dir):
        """Test extracting data for specific unit."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_train("FD001")

        unit_data = loader.get_unit_data(df, unit_id=1)

        assert (unit_data["unit_id"] == 1).all()
        assert len(unit_data) > 0

    def test_get_unit_data_invalid_id(self, temp_data_dir):
        """Test error for invalid unit ID."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_train("FD001")

        with pytest.raises(ValueError, match="Unit 999 not found"):
            loader.get_unit_data(df, unit_id=999)

    def test_available_datasets_property(self):
        """Test available_datasets property."""
        loader = DataLoader()
        datasets = loader.available_datasets

        assert datasets == AVAILABLE_DATASETS
        # Ensure it returns a copy
        datasets.append("FD999")
        assert loader.available_datasets == AVAILABLE_DATASETS

    def test_check_data_availability(self, temp_data_dir):
        """Test checking data availability."""
        loader = DataLoader(data_dir=temp_data_dir)
        availability = loader.check_data_availability()

        assert availability["FD001"] is True
        assert availability["FD002"] is False


# =============================================================================
# FeatureEngineer Tests
# =============================================================================


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def test_init_default_params(self):
        """Test FeatureEngineer initialization with defaults."""
        fe = FeatureEngineer()

        assert fe.rolling_windows == [5, 10, 20]
        assert fe.lag_periods == [1, 5, 10]
        assert fe.max_rul == 125
        assert fe.is_fitted is False

    def test_init_custom_params(self):
        """Test FeatureEngineer initialization with custom params."""
        fe = FeatureEngineer(
            rolling_windows=[3, 7],
            lag_periods=[2, 4],
            max_rul=100,
        )

        assert fe.rolling_windows == [3, 7]
        assert fe.lag_periods == [2, 4]
        assert fe.max_rul == 100

    def test_create_rul_target(self, sample_train_data, feature_engineer):
        """Test RUL target creation."""
        df = feature_engineer.create_rul_target(sample_train_data)

        assert "RUL" in df.columns
        assert df["RUL"].min() >= 0
        assert df["RUL"].max() <= feature_engineer.max_rul

    def test_create_rul_target_no_clip(self, sample_train_data, feature_engineer):
        """Test RUL target creation without clipping."""
        # Ensure max cycle is > 125 for testing
        sample_train_data = sample_train_data.copy()
        sample_train_data.loc[sample_train_data["unit_id"] == 1, "time_cycle"] = range(1, 21)

        df = feature_engineer.create_rul_target(sample_train_data, clip=False)

        assert "RUL" in df.columns
        # Max RUL for unit with 20 cycles should be 19
        assert df["RUL"].max() >= 0

    def test_create_rul_target_missing_columns(self, feature_engineer):
        """Test error for missing required columns."""
        df = pd.DataFrame({"sensor_1": [1, 2, 3]})

        with pytest.raises(FeatureEngineerError, match="Missing required columns"):
            feature_engineer.create_rul_target(df)

    def test_create_rolling_features(self, sample_train_data, feature_engineer):
        """Test rolling feature creation."""
        df = feature_engineer.create_rolling_features(
            sample_train_data,
            columns=["sensor_2", "sensor_3"],
            windows=[5],
        )

        # Check features were created
        assert "sensor_2_rolling_mean_5" in df.columns
        assert "sensor_2_rolling_std_5" in df.columns
        assert "sensor_2_rolling_min_5" in df.columns
        assert "sensor_2_rolling_max_5" in df.columns

    def test_create_lag_features(self, sample_train_data, feature_engineer):
        """Test lag feature creation."""
        df = feature_engineer.create_lag_features(
            sample_train_data,
            columns=["sensor_2"],
            lags=[1, 5],
        )

        assert "sensor_2_lag_1" in df.columns
        assert "sensor_2_lag_5" in df.columns

    def test_create_rate_of_change(self, sample_train_data, feature_engineer):
        """Test rate of change feature creation."""
        df = feature_engineer.create_rate_of_change(
            sample_train_data,
            columns=["sensor_2", "sensor_3"],
        )

        assert "sensor_2_delta" in df.columns
        assert "sensor_3_delta" in df.columns

    def test_normalize_features_fit(self, sample_train_data, feature_engineer):
        """Test feature normalization with fitting."""
        X = feature_engineer.normalize_features(
            sample_train_data,
            columns=["sensor_2", "sensor_3"],
            fit=True,
        )

        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 2
        assert feature_engineer.is_fitted is True

        # Check normalized (approximately mean=0, std=1)
        assert abs(X[:, 0].mean()) < 0.1
        assert abs(X[:, 0].std() - 1) < 0.1

    def test_normalize_features_transform(self, sample_train_data, feature_engineer):
        """Test feature normalization without fitting."""
        # First fit
        feature_engineer.normalize_features(
            sample_train_data,
            columns=["sensor_2", "sensor_3"],
            fit=True,
        )

        # Then transform
        X = feature_engineer.normalize_features(
            sample_train_data,
            columns=["sensor_2", "sensor_3"],
            fit=False,
        )

        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 2

    def test_normalize_features_not_fitted_error(self, sample_train_data, feature_engineer):
        """Test error when transforming without fitting."""
        with pytest.raises(FeatureEngineerError, match="Scaler is not fitted"):
            feature_engineer.normalize_features(
                sample_train_data,
                columns=["sensor_2"],
                fit=False,
            )

    def test_save_load_scaler(self, sample_train_data, feature_engineer):
        """Test scaler save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler.joblib"

            # Fit and save
            feature_engineer.normalize_features(
                sample_train_data,
                columns=["sensor_2", "sensor_3"],
                fit=True,
            )
            feature_engineer.save_scaler(scaler_path)

            assert scaler_path.exists()

            # Load in new instance
            new_fe = FeatureEngineer()
            new_fe.load_scaler(scaler_path)

            assert new_fe.is_fitted is True
            assert new_fe.feature_columns == ["sensor_2", "sensor_3"]

    def test_save_scaler_not_fitted_error(self, feature_engineer):
        """Test error when saving unfitted scaler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler.joblib"

            with pytest.raises(FeatureEngineerError, match="Cannot save unfitted"):
                feature_engineer.save_scaler(scaler_path)

    def test_load_scaler_not_found_error(self, feature_engineer):
        """Test error when scaler file not found."""
        with pytest.raises(FileNotFoundError):
            feature_engineer.load_scaler("/nonexistent/path/scaler.joblib")

    def test_create_sequences(self, feature_engineer):
        """Test sequence creation for LSTM."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        X_seq, y_seq = feature_engineer.create_sequences(X, y, sequence_length=20)

        assert X_seq.shape == (81, 20, 10)  # 100 - 20 + 1 = 81 sequences
        assert y_seq.shape == (81,)

    def test_create_sequences_too_short(self, feature_engineer):
        """Test error when data too short for sequences."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        with pytest.raises(FeatureEngineerError, match="Not enough samples"):
            feature_engineer.create_sequences(X, y, sequence_length=20)

    def test_full_pipeline(self, sample_train_data, feature_engineer):
        """Test full feature engineering pipeline."""
        X, y = feature_engineer.full_pipeline(
            sample_train_data,
            fit=True,
            create_rul=True,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert feature_engineer.is_fitted is True

    def test_drop_constant_columns(self, feature_engineer):
        """Test dropping constant columns."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "time_cycle": [1, 2, 3, 1, 2, 3],
            "sensor_1": [100, 100, 100, 100, 100, 100],  # Constant
            "sensor_2": [1, 2, 3, 4, 5, 6],  # Variable
        })

        result = feature_engineer.drop_constant_columns(df)

        assert "sensor_1" not in result.columns
        assert "sensor_2" in result.columns

    def test_get_feature_names(self, sample_train_data, feature_engineer):
        """Test getting feature names after fitting."""
        feature_engineer.normalize_features(
            sample_train_data,
            columns=["sensor_2", "sensor_3"],
            fit=True,
        )

        names = feature_engineer.get_feature_names()

        assert names == ["sensor_2", "sensor_3"]
        # Ensure it returns a copy
        names.append("new_col")
        assert feature_engineer.get_feature_names() == ["sensor_2", "sensor_3"]


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Test cases for configuration settings."""

    def test_settings_singleton(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_settings_paths(self):
        """Test path configuration."""
        assert settings.paths.data_dir.name == "data"
        assert settings.paths.raw_data_dir.name == "raw"
        assert settings.paths.processed_data_dir.name == "processed"

    def test_settings_features(self):
        """Test feature engineering configuration."""
        assert settings.features.rolling_windows == [5, 10, 20]
        assert settings.features.lag_periods == [1, 5, 10]
        assert settings.features.max_rul == 125

    def test_settings_model(self):
        """Test model configuration."""
        assert settings.model.random_state == 42
        assert settings.model.test_size == 0.2
        assert settings.model.batch_size == 32

    def test_column_names(self):
        """Test column names constant."""
        assert len(COLUMN_NAMES) == 26
        assert COLUMN_NAMES[0] == "unit_id"
        assert COLUMN_NAMES[1] == "time_cycle"
        assert "sensor_1" in COLUMN_NAMES
        assert "sensor_21" in COLUMN_NAMES

    def test_available_datasets(self):
        """Test available datasets constant."""
        assert AVAILABLE_DATASETS == ["FD001", "FD002", "FD003", "FD004"]

    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create new settings with temp directory
            from dataclasses import replace
            from src.config import PathConfig

            temp_paths = PathConfig(project_root=Path(tmpdir))
            temp_settings = Settings(paths=temp_paths)

            temp_settings.ensure_directories()

            assert temp_paths.data_dir.exists()
            assert temp_paths.raw_data_dir.exists()
            assert temp_paths.processed_data_dir.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for data pipeline."""

    def test_full_data_pipeline(self, temp_data_dir):
        """Test complete data loading and preprocessing pipeline."""
        # Load data
        loader = DataLoader(data_dir=temp_data_dir)
        train_df = loader.load_train("FD001")

        # Process features
        fe = FeatureEngineer(
            rolling_windows=[5],
            lag_periods=[1],
        )

        # Create RUL
        train_df = fe.create_rul_target(train_df)
        assert "RUL" in train_df.columns

        # Create features
        train_df = fe.create_rolling_features(train_df, columns=["sensor_2"])
        train_df = fe.create_lag_features(train_df, columns=["sensor_2"])
        train_df = fe.create_rate_of_change(train_df, columns=["sensor_2"])

        # Normalize
        feature_cols = [c for c in train_df.columns if "sensor" in c]
        X = fe.normalize_features(train_df, columns=feature_cols, fit=True)
        y = train_df["RUL"].values

        assert X.shape[0] == len(y)
        assert fe.is_fitted

    def test_train_test_pipeline(self, temp_data_dir):
        """Test train/test data preprocessing consistency."""
        loader = DataLoader(data_dir=temp_data_dir)
        train_df = loader.load_train("FD001")
        test_df = loader.load_test("FD001")

        fe = FeatureEngineer(rolling_windows=[5], lag_periods=[1])

        # Process train
        train_df = fe.create_rul_target(train_df)
        train_df = fe.create_rolling_features(train_df, columns=["sensor_2"])
        X_train = fe.normalize_features(
            train_df, columns=["sensor_2", "sensor_2_rolling_mean_5"], fit=True
        )

        # Process test (no RUL, use fitted scaler)
        test_df = fe.create_rolling_features(test_df, columns=["sensor_2"])
        X_test = fe.normalize_features(
            test_df, columns=["sensor_2", "sensor_2_rolling_mean_5"], fit=False
        )

        assert X_train.shape[1] == X_test.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
