#!/usr/bin/env python3
"""
Batch Prediction Example for Predictive Maintenance API.

This script demonstrates how to:
1. Load a large dataset
2. Send predictions in configurable batches
3. Handle rate limiting and retries
4. Display progress with a progress bar
5. Save results with statistics

Usage:
    # Basic usage
    python examples/batch_predict.py --data-file data/processed/test.csv

    # With custom batch size
    python examples/batch_predict.py --data-file data.csv --batch-size 50

    # With output file
    python examples/batch_predict.py --data-file data.csv --output predictions.csv
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


class BatchPredictor:
    """Handles batch predictions with retries and rate limiting."""

    def __init__(
        self,
        api_url: str,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        max_workers: int = 4,
    ):
        """
        Initialize the batch predictor.

        Args:
            api_url: Base URL of the prediction API.
            batch_size: Number of predictions per batch.
            max_retries: Maximum retry attempts for failed requests.
            retry_delay: Delay between retries in seconds.
            timeout: Request timeout in seconds.
            max_workers: Maximum parallel workers for predictions.
        """
        self.api_url = api_url.rstrip("/")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.max_workers = max_workers

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.latencies: List[float] = []

    def _send_single_prediction(
        self, features: Dict[str, float], sample_id: int
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Send a single prediction with retry logic.

        Args:
            features: Feature dictionary.
            sample_id: Sample identifier.

        Returns:
            Tuple of (sample_id, result_dict).
        """
        endpoint = f"{self.api_url}/predict"

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.session.post(
                    endpoint,
                    json={"features": features},
                    timeout=self.timeout,
                )
                elapsed = time.time() - start_time

                self.total_requests += 1
                self.latencies.append(elapsed * 1000)
                self.total_latency += elapsed

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = self.retry_delay * (2**attempt)
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                self.successful_requests += 1
                result = response.json()
                result["sample_id"] = sample_id
                result["inference_time_ms"] = round(elapsed * 1000, 2)
                return sample_id, result

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    self.failed_requests += 1
                    return sample_id, {"sample_id": sample_id, "error": str(e)}

                time.sleep(self.retry_delay * (2**attempt))

        self.failed_requests += 1
        return sample_id, {"sample_id": sample_id, "error": "Max retries exceeded"}

    def predict_batch(
        self, df: pd.DataFrame, show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Send predictions for a dataframe in batches.

        Args:
            df: DataFrame with feature columns.
            show_progress: Whether to show progress bar.

        Returns:
            List of prediction results.
        """
        results = [None] * len(df)
        samples = list(df.iterrows())

        # Create progress bar
        if show_progress and HAS_TQDM:
            progress = tqdm(total=len(samples), desc="Predicting", unit="sample")
        else:
            progress = None

        # Process in batches with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch_start in range(0, len(samples), self.batch_size):
                batch = samples[batch_start : batch_start + self.batch_size]

                # Submit batch
                futures = {}
                for idx, row in batch:
                    features = row.to_dict()
                    future = executor.submit(
                        self._send_single_prediction, features, idx
                    )
                    futures[future] = idx

                # Collect results
                for future in as_completed(futures):
                    sample_id, result = future.result()
                    results[sample_id] = result

                    if progress:
                        progress.update(1)

                # Small delay between batches to avoid overwhelming the API
                if batch_start + self.batch_size < len(samples):
                    time.sleep(0.1)

        if progress:
            progress.close()

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get prediction statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self.latencies:
            return {}

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "total_time_seconds": self.total_latency,
            "avg_latency_ms": np.mean(self.latencies),
            "p50_latency_ms": np.percentile(self.latencies, 50),
            "p95_latency_ms": np.percentile(self.latencies, 95),
            "p99_latency_ms": np.percentile(self.latencies, 99),
            "min_latency_ms": np.min(self.latencies),
            "max_latency_ms": np.max(self.latencies),
            "throughput_rps": self.total_requests / max(self.total_latency, 0.001),
        }


def load_data(data_file: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Drop non-feature columns if present
    drop_cols = ["unit_id", "time_cycle", "rul", "RUL", "label", "target"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def generate_sample_data(num_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic sample data."""
    np.random.seed(42)

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

    data = {col: np.random.uniform(0, 1, num_samples) for col in feature_cols}
    return pd.DataFrame(data)


def display_summary(results: List[Dict], stats: Dict[str, Any]) -> None:
    """Display prediction summary."""
    print("\n" + "=" * 80)
    print(" BATCH PREDICTION SUMMARY ".center(80))
    print("=" * 80)

    # Request statistics
    print("\n[Request Statistics]")
    print(f"  Total Requests:      {stats.get('total_requests', 0):,}")
    print(f"  Successful:          {stats.get('successful_requests', 0):,}")
    print(f"  Failed:              {stats.get('failed_requests', 0):,}")
    print(f"  Success Rate:        {stats.get('success_rate', 0):.2%}")

    # Performance statistics
    print("\n[Performance Statistics]")
    print(f"  Total Time:          {stats.get('total_time_seconds', 0):.2f}s")
    print(f"  Throughput:          {stats.get('throughput_rps', 0):.1f} req/s")
    print(f"  Avg Latency:         {stats.get('avg_latency_ms', 0):.2f}ms")
    print(f"  P50 Latency:         {stats.get('p50_latency_ms', 0):.2f}ms")
    print(f"  P95 Latency:         {stats.get('p95_latency_ms', 0):.2f}ms")
    print(f"  P99 Latency:         {stats.get('p99_latency_ms', 0):.2f}ms")

    # RUL statistics
    rul_values = [
        r.get("predicted_rul") for r in results if r and "predicted_rul" in r
    ]
    if rul_values:
        print("\n[RUL Prediction Statistics]")
        print(f"  Count:               {len(rul_values):,}")
        print(f"  Mean RUL:            {np.mean(rul_values):.2f}")
        print(f"  Std RUL:             {np.std(rul_values):.2f}")
        print(f"  Min RUL:             {np.min(rul_values):.2f}")
        print(f"  Max RUL:             {np.max(rul_values):.2f}")

        # Critical predictions (RUL < 25)
        critical = sum(1 for r in rul_values if r < 25)
        print(f"  Critical (RUL<25):   {critical:,} ({critical/len(rul_values):.1%})")

    print("=" * 80)


def save_results(
    results: List[Dict], stats: Dict[str, Any], output_file: str
) -> None:
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df["timestamp"] = pd.Timestamp.now()
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save statistics
    stats_file = Path(output_file).stem + "_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")


def check_api_health(api_url: str) -> bool:
    """Check if API is accessible."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Send batch predictions to the Predictive Maintenance API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the prediction API",
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
        default=1000,
        help="Number of synthetic samples if no data file (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Predictions per batch (default: 32)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV file",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" PREDICTIVE MAINTENANCE - BATCH PREDICTION ".center(80))
    print("=" * 80)
    print(f"\nAPI URL: {args.api_url}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Workers: {args.max_workers}")

    # Check API health
    print("\nChecking API health...")
    if not check_api_health(args.api_url):
        print("ERROR: API is not accessible. Please ensure the API is running.")
        sys.exit(1)
    print("API is healthy!")

    # Load or generate data
    if args.data_file:
        print(f"\nLoading data from: {args.data_file}")
        df = load_data(args.data_file)
    else:
        print(f"\nGenerating {args.num_samples} synthetic samples...")
        df = generate_sample_data(args.num_samples)

    print(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # Create predictor
    predictor = BatchPredictor(
        api_url=args.api_url,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
    )

    # Run predictions
    print("\nStarting batch predictions...")
    start_time = time.time()
    results = predictor.predict_batch(df)
    total_time = time.time() - start_time

    # Get statistics
    stats = predictor.get_statistics()
    stats["total_elapsed_time"] = total_time

    # Display summary
    display_summary(results, stats)

    # Save results if requested
    if args.output:
        save_results(results, stats, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
