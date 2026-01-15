"""
Configuration settings for the Predictive Maintenance MLOps Platform.

This module provides centralized configuration using Pydantic for validation
and environment variable support. All settings can be overridden via environment
variables with the prefix specified in the Config class.

Example:
    >>> from src.config import settings
    >>> print(settings.data_dir)
    >>> print(settings.mlflow_tracking_uri)
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class PathConfig:
    """Configuration for project paths."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Raw data directory for original NASA dataset files."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Processed data directory for transformed features."""
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        """Directory for saved model artifacts."""
        return self.project_root / "models"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for pipeline artifacts (scalers, encoders, etc.)."""
        return self.project_root / "artifacts"


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for feature engineering parameters."""

    # Rolling window sizes for statistical features
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Lag periods for time-series features
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10])

    # Sequence length for LSTM models
    sequence_length: int = 50

    # Maximum RUL value (clip higher values)
    max_rul: int = 125

    # Sensors to drop (constant or near-constant values in FD001)
    sensors_to_drop: List[str] = field(default_factory=lambda: [
        "sensor_1", "sensor_5", "sensor_6", "sensor_10",
        "sensor_16", "sensor_18", "sensor_19"
    ])

    # Operational settings columns
    op_settings: List[str] = field(default_factory=lambda: [
        "op_setting_1", "op_setting_2", "op_setting_3"
    ])


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model training parameters."""

    # Random seed for reproducibility
    random_state: int = 42

    # Train/test split ratio
    test_size: float = 0.2

    # Validation split ratio (from training set)
    validation_size: float = 0.1

    # Training batch size
    batch_size: int = 32

    # Maximum training epochs
    epochs: int = 100

    # Early stopping patience
    early_stopping_patience: int = 10

    # Learning rate
    learning_rate: float = 0.001


@dataclass(frozen=True)
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""

    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive-maintenance")
    )
    registry_uri: Optional[str] = field(
        default_factory=lambda: os.getenv("MLFLOW_REGISTRY_URI", None)
    )


@dataclass(frozen=True)
class ServingConfig:
    """Configuration for model serving API."""

    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "4")))
    reload: bool = field(default_factory=lambda: os.getenv("API_RELOAD", "false").lower() == "true")


@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration for model monitoring and drift detection."""

    # Drift detection threshold (p-value)
    drift_threshold: float = 0.05

    # Monitoring check interval in seconds
    monitoring_interval: int = 300

    # Prometheus metrics port
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "8001")))


@dataclass
class Settings:
    """
    Main settings class aggregating all configuration sections.

    Attributes:
        paths: File system path configuration
        features: Feature engineering parameters
        model: Model training parameters
        mlflow: MLflow tracking configuration
        serving: API serving configuration
        monitoring: Drift detection and monitoring configuration

    Example:
        >>> settings = Settings()
        >>> print(settings.paths.raw_data_dir)
        >>> print(settings.features.rolling_windows)
    """

    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.paths.data_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.models_dir,
            self.paths.artifacts_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: The application settings singleton.

    Example:
        >>> settings = get_settings()
        >>> print(settings.model.random_state)
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()


# Column names for NASA Turbofan dataset
COLUMN_NAMES: List[str] = [
    "unit_id",
    "time_cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

# Sensor columns only
SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in range(1, 22)]

# Available datasets
AVAILABLE_DATASETS: List[str] = ["FD001", "FD002", "FD003", "FD004"]
