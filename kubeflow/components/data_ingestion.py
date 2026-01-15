#!/usr/bin/env python3
"""
Kubeflow Component: Data Ingestion

Loads NASA Turbofan Engine Degradation dataset from mounted volume or path
and outputs processed data for downstream components.

Usage:
    python data_ingestion.py \
        --dataset FD001 \
        --data-path /data/raw \
        --output-path /data/processed/processed_data.csv

Kubeflow Pipeline Usage:
    This component reads raw NASA dataset files and outputs a single
    processed CSV file containing merged training data with RUL labels.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_ingestion")

# NASA C-MAPSS dataset column names
COLUMN_NAMES = [
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

AVAILABLE_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def load_train_data(data_path: Path, dataset: str) -> pd.DataFrame:
    """
    Load training data from NASA C-MAPSS format.

    Args:
        data_path: Directory containing dataset files
        dataset: Dataset name (FD001-FD004)

    Returns:
        DataFrame with training data
    """
    train_file = data_path / f"train_{dataset}.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    logger.info(f"Loading training data from {train_file}")

    df = pd.read_csv(
        train_file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    logger.info(f"Loaded {len(df)} rows, {df['unit_id'].nunique()} units")
    return df


def load_rul_data(data_path: Path, dataset: str) -> np.ndarray:
    """
    Load RUL (Remaining Useful Life) data for test set.

    Args:
        data_path: Directory containing dataset files
        dataset: Dataset name (FD001-FD004)

    Returns:
        NumPy array of RUL values
    """
    rul_file = data_path / f"RUL_{dataset}.txt"

    if not rul_file.exists():
        logger.warning(f"RUL file not found: {rul_file}")
        return None

    logger.info(f"Loading RUL data from {rul_file}")
    rul = np.loadtxt(rul_file, dtype=np.int32)

    logger.info(f"Loaded RUL for {len(rul)} units")
    return rul


def compute_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Compute RUL (Remaining Useful Life) for each record.

    RUL = max_cycle_for_unit - current_cycle, clipped at max_rul.

    Args:
        df: DataFrame with unit_id and time_cycle columns
        max_rul: Maximum RUL value (default: 125, standard for this dataset)

    Returns:
        DataFrame with RUL column added
    """
    logger.info(f"Computing RUL with max_rul={max_rul}")

    df = df.copy()

    # Get max cycle for each unit
    max_cycles = df.groupby("unit_id")["time_cycle"].max()
    df["max_cycle"] = df["unit_id"].map(max_cycles)

    # Compute RUL
    df["RUL"] = df["max_cycle"] - df["time_cycle"]

    # Clip RUL at max value (piece-wise linear degradation assumption)
    df["RUL"] = df["RUL"].clip(upper=max_rul)

    # Drop helper column
    df = df.drop(columns=["max_cycle"])

    logger.info(f"RUL range: {df['RUL'].min()} - {df['RUL'].max()}")
    return df


def log_data_statistics(df: pd.DataFrame) -> None:
    """Log statistics about the dataset."""
    logger.info("=" * 50)
    logger.info("Data Statistics:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Number of units: {df['unit_id'].nunique()}")
    logger.info(f"  Number of features: {len(df.columns)}")
    logger.info(f"  Columns: {list(df.columns)}")

    if "RUL" in df.columns:
        logger.info(f"  RUL statistics:")
        logger.info(f"    Min: {df['RUL'].min()}")
        logger.info(f"    Max: {df['RUL'].max()}")
        logger.info(f"    Mean: {df['RUL'].mean():.2f}")
        logger.info(f"    Std: {df['RUL'].std():.2f}")

    # Check for missing values
    missing = df.isnull().sum().sum()
    logger.info(f"  Missing values: {missing}")
    logger.info("=" * 50)


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024:.2f} KB")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kubeflow Component: Data Ingestion for NASA Turbofan Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=AVAILABLE_DATASETS,
        default="FD001",
        help="NASA Turbofan dataset to load",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to directory containing raw data files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save processed_data.csv",
    )
    parser.add_argument(
        "--max-rul",
        type=int,
        default=125,
        help="Maximum RUL value for clipping",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Kubeflow Component: Data Ingestion")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Max RUL: {args.max_rul}")

    try:
        data_path = Path(args.data_path)
        output_path = Path(args.output_path)

        # Validate input path
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        # Load training data
        df = load_train_data(data_path, args.dataset)

        # Compute RUL labels
        df = compute_rul(df, max_rul=args.max_rul)

        # Log statistics
        log_data_statistics(df)

        # Save output
        save_output(df, output_path)

        logger.info("Data ingestion completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
