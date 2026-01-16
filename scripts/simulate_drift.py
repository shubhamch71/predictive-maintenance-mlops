#!/usr/bin/env python3
"""
=============================================================================
Simulate Data Drift for Demo/Testing
=============================================================================
This script simulates data drift by:
1. Loading reference data (or generating synthetic data)
2. Creating drifted data with shifted distributions
3. Sending predictions to the API
4. Monitoring drift detection
5. Verifying Argo Workflow is triggered

Usage:
    python scripts/simulate_drift.py [--api-url URL] [--num-requests N]
    python scripts/simulate_drift.py --cleanup
=============================================================================
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_NUM_REQUESTS = 1000
DRIFT_WEBHOOK_URL = "http://localhost:12000/drift"

# Sensor names matching NASA Turbofan dataset
SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]
OPERATIONAL_SETTINGS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3"
]


# =============================================================================
# Color Output
# =============================================================================
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'
    BOLD = '\033[1m'


def print_header(message: str) -> None:
    """Print a formatted header."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {message}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.NC}")
    print()


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.NC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.NC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.NC}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.NC}")


# =============================================================================
# Data Generation
# =============================================================================
def generate_reference_data(n_samples: int = 5000) -> np.ndarray:
    """Generate reference (baseline) sensor data."""
    np.random.seed(42)

    # Reference distributions for each sensor
    # Based on typical NASA Turbofan dataset ranges
    reference_means = [
        518.67,   # sensor_1: T2 (Total temperature at fan inlet)
        642.15,   # sensor_2: T24 (Total temperature at LPC outlet)
        1589.70,  # sensor_3: T30 (Total temperature at HPC outlet)
        1400.60,  # sensor_4: T50 (Total temperature at LPT outlet)
        14.62,    # sensor_5: P2 (Pressure at fan inlet)
        21.61,    # sensor_6: P15 (Total pressure in bypass-duct)
        554.36,   # sensor_7: P30 (Total pressure at HPC outlet)
        2388.02,  # sensor_8: Nf (Physical fan speed)
        9046.19,  # sensor_9: Nc (Physical core speed)
        1.30,     # sensor_10: epr (Engine pressure ratio)
        47.47,    # sensor_11: Ps30 (Static pressure at HPC outlet)
        521.66,   # sensor_12: phi (Ratio of fuel flow to Ps30)
        2388.02,  # sensor_13: NRf (Corrected fan speed)
        8138.62,  # sensor_14: NRc (Corrected core speed)
        8.4195,   # sensor_15: BPR (Bypass Ratio)
        0.03,     # sensor_16: farB (Burner fuel-air ratio)
        392,      # sensor_17: htBleed (Bleed Enthalpy)
        2388,     # sensor_18: Nf_dmd (Demanded fan speed)
        100.0,    # sensor_19: PCNfR_dmd (Demanded corrected fan speed)
        39.06,    # sensor_20: W31 (HPT coolant bleed)
        23.4190   # sensor_21: W32 (LPT coolant bleed)
    ]

    reference_stds = [
        0.0,      # sensor_1 (constant)
        0.5,      # sensor_2
        2.5,      # sensor_3
        6.0,      # sensor_4
        0.0,      # sensor_5 (constant)
        0.4,      # sensor_6
        10.0,     # sensor_7
        7.5,      # sensor_8
        20.0,     # sensor_9
        0.02,     # sensor_10
        0.3,      # sensor_11
        3.0,      # sensor_12
        7.5,      # sensor_13
        15.0,     # sensor_14
        0.08,     # sensor_15
        0.001,    # sensor_16
        3.0,      # sensor_17
        7.5,      # sensor_18
        0.0,      # sensor_19 (constant)
        0.4,      # sensor_20
        0.15      # sensor_21
    ]

    data = np.zeros((n_samples, 21))
    for i in range(21):
        if reference_stds[i] > 0:
            data[:, i] = np.random.normal(reference_means[i], reference_stds[i], n_samples)
        else:
            data[:, i] = reference_means[i]

    return data


def generate_drifted_data(
    reference_data: np.ndarray,
    drift_magnitude: float = 0.1,
    drift_type: str = "shift"
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate drifted data from reference data.

    Args:
        reference_data: Original reference data
        drift_magnitude: Magnitude of drift (0.1 = 10% shift)
        drift_type: Type of drift ('shift', 'scale', 'mixed')

    Returns:
        Tuple of (drifted_data, indices_of_drifted_features)
    """
    n_samples = reference_data.shape[0]
    drifted_data = reference_data.copy()

    # Select features to drift (random subset)
    n_features_to_drift = random.randint(3, 8)
    drifted_features = random.sample(range(21), n_features_to_drift)

    for feature_idx in drifted_features:
        feature_mean = np.mean(reference_data[:, feature_idx])
        feature_std = np.std(reference_data[:, feature_idx])

        if feature_std == 0:
            continue  # Skip constant features

        if drift_type == "shift":
            # Mean shift drift
            shift = feature_mean * drift_magnitude
            drifted_data[:, feature_idx] += shift

        elif drift_type == "scale":
            # Variance change drift
            scale_factor = 1 + drift_magnitude
            drifted_data[:, feature_idx] = (
                feature_mean +
                (drifted_data[:, feature_idx] - feature_mean) * scale_factor
            )

        elif drift_type == "mixed":
            # Both shift and scale
            shift = feature_mean * drift_magnitude * 0.5
            scale_factor = 1 + (drift_magnitude * 0.5)
            drifted_data[:, feature_idx] = (
                feature_mean + shift +
                (drifted_data[:, feature_idx] - feature_mean) * scale_factor
            )

    # Add some noise
    noise = np.random.normal(0, 0.01, drifted_data.shape)
    drifted_data += noise * np.abs(drifted_data)

    return drifted_data, drifted_features


# =============================================================================
# API Interaction
# =============================================================================
def create_prediction_payload(
    sensor_values: np.ndarray,
    unit_id: int,
    cycle: int
) -> Dict[str, Any]:
    """Create a prediction request payload."""
    payload = {
        "unit_id": unit_id,
        "cycle": cycle,
        "operational_setting_1": 0.0023 + random.uniform(-0.001, 0.001),
        "operational_setting_2": 0.0003 + random.uniform(-0.0001, 0.0001),
        "operational_setting_3": 100.0,
    }

    for i, name in enumerate(SENSOR_NAMES):
        payload[name] = float(sensor_values[i])

    return payload


def send_prediction(api_url: str, payload: Dict[str, Any]) -> Optional[Dict]:
    """Send a prediction request to the API."""
    if not HAS_REQUESTS:
        print_error("requests library not installed. Run: pip install requests")
        return None

    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code, "message": response.text}
    except requests.exceptions.RequestException as e:
        return {"error": "connection", "message": str(e)}


def send_drift_webhook(
    drift_score: float,
    drift_type: str,
    drifted_features: List[int]
) -> bool:
    """Send drift detection webhook to trigger retraining."""
    if not HAS_REQUESTS:
        return False

    payload = {
        "drift_score": drift_score,
        "drift_type": drift_type,
        "detected_at": datetime.now().isoformat(),
        "features_affected": [f"sensor_{i+1}" for i in drifted_features],
        "drift_details": {
            "max_psi": drift_score,
            "mean_psi": drift_score * 0.8,
            "features_above_threshold": len(drifted_features)
        },
        "recommendation": "retrain",
        "model_name": "rul-ensemble"
    }

    try:
        response = requests.post(
            DRIFT_WEBHOOK_URL,
            json=payload,
            timeout=5
        )
        return response.status_code in [200, 202]
    except:
        return False


# =============================================================================
# Drift Analysis
# =============================================================================
def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI)."""
    # Create bins from reference data
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Calculate proportions
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_prop = ref_counts / len(reference)
    cur_prop = cur_counts / len(current)

    # Avoid division by zero
    ref_prop = np.clip(ref_prop, 0.0001, 1)
    cur_prop = np.clip(cur_prop, 0.0001, 1)

    # Calculate PSI
    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))

    return psi


def analyze_drift(
    reference_data: np.ndarray,
    drifted_data: np.ndarray
) -> Dict[str, Any]:
    """Analyze drift between reference and current data."""
    results = {
        "features": {},
        "summary": {
            "max_psi": 0,
            "mean_psi": 0,
            "drifted_features": [],
            "total_features": 21
        }
    }

    psi_values = []

    for i, sensor_name in enumerate(SENSOR_NAMES):
        psi = calculate_psi(reference_data[:, i], drifted_data[:, i])
        psi_values.append(psi)

        results["features"][sensor_name] = {
            "psi": round(psi, 4),
            "ref_mean": round(np.mean(reference_data[:, i]), 4),
            "cur_mean": round(np.mean(drifted_data[:, i]), 4),
            "ref_std": round(np.std(reference_data[:, i]), 4),
            "cur_std": round(np.std(drifted_data[:, i]), 4),
            "drifted": psi > 0.1
        }

        if psi > 0.1:
            results["summary"]["drifted_features"].append(sensor_name)

    results["summary"]["max_psi"] = round(max(psi_values), 4)
    results["summary"]["mean_psi"] = round(np.mean(psi_values), 4)
    results["summary"]["num_drifted"] = len(results["summary"]["drifted_features"])

    return results


# =============================================================================
# Kubernetes Integration
# =============================================================================
def check_argo_workflows() -> List[Dict]:
    """Check for recent Argo Workflows."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "workflows", "-n", "mlops", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            workflows = []
            for item in data.get("items", []):
                workflows.append({
                    "name": item["metadata"]["name"],
                    "status": item.get("status", {}).get("phase", "Unknown"),
                    "created": item["metadata"]["creationTimestamp"]
                })
            return workflows
        return []
    except Exception as e:
        print_warning(f"Could not check Argo workflows: {e}")
        return []


def check_drift_detector_metrics() -> Optional[Dict]:
    """Check drift metrics from Prometheus."""
    try:
        result = subprocess.run(
            [
                "kubectl", "exec", "-n", "monitoring",
                "deploy/prometheus", "--",
                "wget", "-q", "-O", "-",
                "http://localhost:9090/api/v1/query?query=predictive_maintenance_drift_score"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except:
        return None


# =============================================================================
# Visualization
# =============================================================================
def print_drift_report(
    drift_analysis: Dict[str, Any],
    drifted_feature_indices: List[int],
    workflow_triggered: bool
) -> None:
    """Print a formatted drift report."""
    print_header("Drift Analysis Report")

    summary = drift_analysis["summary"]

    # Summary box
    print(f"{Colors.CYAN}┌{'─' * 58}┐{Colors.NC}")
    print(f"{Colors.CYAN}│{Colors.NC} {'Drift Summary':<56} {Colors.CYAN}│{Colors.NC}")
    print(f"{Colors.CYAN}├{'─' * 58}┤{Colors.NC}")
    print(f"{Colors.CYAN}│{Colors.NC}  Max PSI Score:      {summary['max_psi']:<34} {Colors.CYAN}│{Colors.NC}")
    print(f"{Colors.CYAN}│{Colors.NC}  Mean PSI Score:     {summary['mean_psi']:<34} {Colors.CYAN}│{Colors.NC}")
    print(f"{Colors.CYAN}│{Colors.NC}  Drifted Features:   {summary['num_drifted']}/{summary['total_features']:<32} {Colors.CYAN}│{Colors.NC}")
    print(f"{Colors.CYAN}└{'─' * 58}┘{Colors.NC}")
    print()

    # PSI thresholds legend
    print(f"{Colors.BOLD}PSI Interpretation:{Colors.NC}")
    print(f"  {Colors.GREEN}< 0.10{Colors.NC}: No significant drift")
    print(f"  {Colors.YELLOW}0.10 - 0.25{Colors.NC}: Moderate drift (monitoring recommended)")
    print(f"  {Colors.RED}>= 0.25{Colors.NC}: Significant drift (retraining recommended)")
    print()

    # Feature details
    print(f"{Colors.BOLD}Feature-wise Analysis:{Colors.NC}")
    print(f"{'Feature':<12} {'PSI':<10} {'Ref Mean':<12} {'Cur Mean':<12} {'Status'}")
    print("-" * 60)

    for i, sensor_name in enumerate(SENSOR_NAMES):
        feature = drift_analysis["features"][sensor_name]
        psi = feature["psi"]

        if psi >= 0.25:
            status = f"{Colors.RED}DRIFT{Colors.NC}"
            psi_str = f"{Colors.RED}{psi:.4f}{Colors.NC}"
        elif psi >= 0.1:
            status = f"{Colors.YELLOW}WARN{Colors.NC}"
            psi_str = f"{Colors.YELLOW}{psi:.4f}{Colors.NC}"
        else:
            status = f"{Colors.GREEN}OK{Colors.NC}"
            psi_str = f"{Colors.GREEN}{psi:.4f}{Colors.NC}"

        print(f"{sensor_name:<12} {psi_str:<20} {feature['ref_mean']:<12.2f} {feature['cur_mean']:<12.2f} {status}")

    print()

    # Workflow status
    print(f"{Colors.BOLD}Retraining Status:{Colors.NC}")
    if workflow_triggered:
        print_success("Retraining workflow triggered!")
    else:
        if summary["max_psi"] >= 0.25:
            print_warning("Drift exceeds threshold - retraining recommended")
            print_info("Trigger manually: kubectl create -f k8s/argo-workflows/retraining-workflow.yaml")
        else:
            print_info("Drift below threshold - no retraining needed")


# =============================================================================
# Main Functions
# =============================================================================
def run_drift_simulation(
    api_url: str,
    num_requests: int,
    drift_magnitude: float = 0.15
) -> None:
    """Run the drift simulation."""
    print_header("Data Drift Simulation")

    # Check prerequisites
    if not HAS_REQUESTS:
        print_error("requests library required. Install with: pip install requests")
        sys.exit(1)

    # Generate data
    print_info(f"Generating reference data...")
    reference_data = generate_reference_data(n_samples=5000)
    print_success(f"Generated {len(reference_data)} reference samples")

    print_info(f"Generating drifted data (magnitude: {drift_magnitude})...")
    drifted_data, drifted_features = generate_drifted_data(
        reference_data,
        drift_magnitude=drift_magnitude,
        drift_type="mixed"
    )
    print_success(f"Generated drifted data affecting {len(drifted_features)} features")

    # Analyze drift
    print_info("Analyzing drift...")
    drift_analysis = analyze_drift(reference_data, drifted_data)

    # Send predictions
    print_header("Sending Predictions")
    print_info(f"Sending {num_requests} predictions to {api_url}")

    successful = 0
    failed = 0
    predictions = []

    start_time = time.time()

    for i in range(num_requests):
        # Use drifted data
        sample_idx = i % len(drifted_data)
        sensor_values = drifted_data[sample_idx]

        payload = create_prediction_payload(
            sensor_values=sensor_values,
            unit_id=random.randint(1, 100),
            cycle=random.randint(1, 300)
        )

        result = send_prediction(api_url, payload)

        if result and "error" not in result:
            successful += 1
            predictions.append(result)
        else:
            failed += 1

        # Progress update
        if (i + 1) % 100 == 0 or i == num_requests - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"\r  Progress: {i + 1}/{num_requests} ({rate:.1f} req/s)", end="", flush=True)

    print()

    elapsed = time.time() - start_time
    print_success(f"Completed in {elapsed:.1f}s ({successful} successful, {failed} failed)")

    # Trigger drift detection webhook
    print_header("Triggering Drift Detection")

    max_psi = drift_analysis["summary"]["max_psi"]

    if max_psi >= 0.25:
        print_info(f"Drift score ({max_psi:.4f}) exceeds threshold (0.25)")
        print_info("Sending drift webhook to trigger retraining...")

        webhook_sent = send_drift_webhook(
            drift_score=max_psi,
            drift_type="psi",
            drifted_features=drifted_features
        )

        if webhook_sent:
            print_success("Drift webhook sent successfully")
        else:
            print_warning("Could not send drift webhook (endpoint may not be available)")
    else:
        print_info(f"Drift score ({max_psi:.4f}) below threshold (0.25)")
        webhook_sent = False

    # Check for Argo workflows
    print_info("Checking for triggered workflows...")
    time.sleep(2)  # Wait for workflow to appear

    workflows = check_argo_workflows()
    workflow_triggered = any(
        "retrain" in w["name"].lower() and
        w["status"] in ["Running", "Pending", "Succeeded"]
        for w in workflows
    )

    if workflows:
        print(f"\n  Recent workflows:")
        for wf in workflows[:5]:
            status_color = Colors.GREEN if wf["status"] == "Succeeded" else (
                Colors.YELLOW if wf["status"] in ["Running", "Pending"] else Colors.RED
            )
            print(f"    - {wf['name']}: {status_color}{wf['status']}{Colors.NC}")

    # Print report
    print_drift_report(drift_analysis, drifted_features, workflow_triggered)

    # Save report
    report_file = PROJECT_ROOT / "drift_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_requests": num_requests,
            "successful": successful,
            "failed": failed,
            "drift_analysis": drift_analysis,
            "workflow_triggered": workflow_triggered
        }, f, indent=2)
    print_info(f"Report saved to: {report_file}")


def cleanup() -> None:
    """Clean up generated files and reset state."""
    print_header("Cleanup")

    files_to_remove = [
        PROJECT_ROOT / "drift_report.json",
        PROJECT_ROOT / ".current_run_id",
    ]

    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            print_success(f"Removed: {file_path}")

    print_info("Cleanup complete")


# =============================================================================
# Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Simulate data drift for testing the MLOps pipeline"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Model API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--num-requests", "-n",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f"Number of prediction requests (default: {DEFAULT_NUM_REQUESTS})"
    )
    parser.add_argument(
        "--drift-magnitude",
        type=float,
        default=0.15,
        help="Drift magnitude (0.0-1.0, default: 0.15)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up generated files"
    )

    args = parser.parse_args()

    if args.cleanup:
        cleanup()
    else:
        run_drift_simulation(
            api_url=args.api_url,
            num_requests=args.num_requests,
            drift_magnitude=args.drift_magnitude
        )


if __name__ == "__main__":
    main()
