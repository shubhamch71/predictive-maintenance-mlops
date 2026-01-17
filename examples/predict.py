#!/usr/bin/env python3
"""
Single Prediction Example for Predictive Maintenance API.

This script demonstrates how to:
1. Load sample data
2. Send prediction requests to the API
3. Display results in a formatted table

Usage:
    # Basic usage
    python examples/predict.py

    # With custom API URL
    python examples/predict.py --api-url http://localhost:8000

    # With custom data file
    python examples/predict.py --data-file data/processed/test_sample.csv

    # Save results to file
    python examples/predict.py --output results.csv

    # Limit number of samples
    python examples/predict.py --num-samples 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests


def load_sample_data(
    data_file: Optional[str] = None, num_samples: int = 10
) -> pd.DataFrame:
    """
    Load sample data for predictions.

    Args:
        data_file: Path to CSV file with features. If None, generates synthetic data.
        num_samples: Number of samples to load/generate.

    Returns:
        DataFrame with feature data.
    """
    if data_file and Path(data_file).exists():
        print(f"Loading data from: {data_file}")
        df = pd.read_csv(data_file)
        return df.head(num_samples)

    # Generate synthetic data matching the expected feature structure
    print("Generating synthetic sample data...")
    np.random.seed(42)

    # Feature columns based on NASA turbofan dataset
    feature_cols = [
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21",
    ]

    data = {}
    for col in feature_cols:
        if col.startswith("op_setting"):
            data[col] = np.random.uniform(-0.5, 0.5, num_samples)
        else:
            data[col] = np.random.uniform(0, 1, num_samples)

    return pd.DataFrame(data)


def send_prediction_request(
    api_url: str, features: Dict[str, float], timeout: int = 30
) -> Dict[str, Any]:
    """
    Send a prediction request to the API.

    Args:
        api_url: Base URL of the prediction API.
        features: Dictionary of feature name to value.
        timeout: Request timeout in seconds.

    Returns:
        API response as dictionary.

    Raises:
        requests.RequestException: If request fails.
    """
    endpoint = f"{api_url.rstrip('/')}/predict"

    payload = {"features": features}

    start_time = time.time()
    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    elapsed_time = time.time() - start_time

    response.raise_for_status()

    result = response.json()
    result["inference_time_ms"] = round(elapsed_time * 1000, 2)

    return result


def display_results(results: List[Dict[str, Any]]) -> None:
    """
    Display prediction results in a formatted table.

    Args:
        results: List of prediction results.
    """
    print("\n" + "=" * 80)
    print(" PREDICTION RESULTS ".center(80))
    print("=" * 80)

    # Create table header
    header = f"{'#':>3} | {'RUL':>8} | {'Confidence':>10} | {'Model':>12} | {'Latency (ms)':>12}"
    print(header)
    print("-" * 80)

    for i, result in enumerate(results, 1):
        rul = result.get("predicted_rul", "N/A")
        confidence = result.get("confidence", "N/A")
        model = result.get("model_type", "unknown")
        latency = result.get("inference_time_ms", "N/A")

        if isinstance(rul, (int, float)):
            rul_str = f"{rul:.1f}"
        else:
            rul_str = str(rul)

        if isinstance(confidence, (int, float)):
            conf_str = f"{confidence:.2%}"
        else:
            conf_str = str(confidence)

        print(f"{i:>3} | {rul_str:>8} | {conf_str:>10} | {model:>12} | {latency:>12}")

    print("=" * 80)

    # Summary statistics
    if results:
        rul_values = [r["predicted_rul"] for r in results if "predicted_rul" in r]
        latencies = [r["inference_time_ms"] for r in results if "inference_time_ms" in r]

        if rul_values:
            print(f"\nRUL Statistics:")
            print(f"  Mean:   {np.mean(rul_values):.2f}")
            print(f"  Min:    {np.min(rul_values):.2f}")
            print(f"  Max:    {np.max(rul_values):.2f}")
            print(f"  Std:    {np.std(rul_values):.2f}")

        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Mean:   {np.mean(latencies):.2f} ms")
            print(f"  P50:    {np.percentile(latencies, 50):.2f} ms")
            print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
            print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save prediction results to CSV file.

    Args:
        results: List of prediction results.
        output_file: Path to output CSV file.
    """
    df = pd.DataFrame(results)
    df["timestamp"] = pd.Timestamp.now()
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def check_api_health(api_url: str) -> bool:
    """
    Check if the API is healthy and accessible.

    Args:
        api_url: Base URL of the API.

    Returns:
        True if API is healthy, False otherwise.
    """
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    """Main entry point for the prediction example."""
    parser = argparse.ArgumentParser(
        description="Send predictions to the Predictive Maintenance API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/predict.py
  python examples/predict.py --api-url http://localhost:8000
  python examples/predict.py --data-file data/test.csv --output results.csv
        """,
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the prediction API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to CSV file with feature data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" PREDICTIVE MAINTENANCE - PREDICTION EXAMPLE ".center(80))
    print("=" * 80)
    print(f"\nAPI URL: {args.api_url}")

    # Check API health
    print("\nChecking API health...")
    if not check_api_health(args.api_url):
        print("ERROR: API is not accessible. Please ensure the API is running.")
        print(f"  Expected URL: {args.api_url}")
        print("\nTo start the API:")
        print("  docker-compose up -d api")
        print("  # or")
        print("  uvicorn src.serving.api:app --reload")
        sys.exit(1)

    print("API is healthy!")

    # Load data
    df = load_sample_data(args.data_file, args.num_samples)
    print(f"\nLoaded {len(df)} samples with {len(df.columns)} features")

    if args.verbose:
        print("\nFeature columns:")
        for col in df.columns:
            print(f"  - {col}")

    # Send predictions
    print("\nSending predictions...")
    results = []

    for idx, row in df.iterrows():
        try:
            features = row.to_dict()
            result = send_prediction_request(args.api_url, features)
            result["sample_index"] = idx
            results.append(result)

            if args.verbose:
                print(f"  Sample {idx}: RUL={result.get('predicted_rul', 'N/A'):.1f}")

        except requests.RequestException as e:
            print(f"  Sample {idx}: ERROR - {e}")
            results.append({"sample_index": idx, "error": str(e)})

    # Display results
    display_results(results)

    # Save results if requested
    if args.output:
        save_results(results, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
