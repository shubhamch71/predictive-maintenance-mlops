"""
Drift Detection System for Predictive Maintenance Models.

This module provides comprehensive drift detection capabilities:
- Data drift: Detects changes in input feature distributions
- Concept drift: Detects changes in the relationship between features and target
- Prediction drift: Detects changes in model output distributions

Statistical Methods Used:
- Kolmogorov-Smirnov (KS) test: Non-parametric test for distribution differences
- Population Stability Index (PSI): Measures distribution shift magnitude
- Jensen-Shannon (JS) divergence: Symmetric measure of distribution similarity
- Page-Hinkley test: Sequential change detection for concept drift

Usage:
    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector(reference_data_path="data/features/features.csv")

    # Check for data drift
    drift_result = detector.detect_data_drift(current_data)

    # Check for concept drift
    concept_drift = detector.detect_concept_drift(predictions, actuals)

    # Generate comprehensive report
    report = detector.generate_drift_report()
"""

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("drift_detector")


# =============================================================================
# Data Classes for Drift Results
# =============================================================================

@dataclass
class FeatureDriftResult:
    """Drift detection result for a single feature."""
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi_score: float
    js_divergence: float
    drift_detected: bool
    drift_severity: str  # "none", "warning", "critical"


@dataclass
class DataDriftResult:
    """Overall data drift detection result."""
    timestamp: str
    overall_drift_detected: bool
    drifted_feature_count: int
    total_features: int
    drift_percentage: float
    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptDriftResult:
    """Concept drift detection result."""
    timestamp: str
    drift_detected: bool
    current_error: float
    baseline_error: float
    error_increase_pct: float
    window_size: int
    page_hinkley_statistic: Optional[float] = None
    trend: str = "stable"  # "stable", "increasing", "decreasing"


@dataclass
class PredictionDriftResult:
    """Prediction distribution drift result."""
    timestamp: str
    drift_detected: bool
    ks_statistic: float
    ks_pvalue: float
    mean_shift: float
    std_shift: float
    drift_score: float


@dataclass
class DriftReport:
    """Comprehensive drift report."""
    report_id: str
    timestamp: str
    data_drift: Optional[DataDriftResult]
    concept_drift: Optional[ConceptDriftResult]
    prediction_drift: Optional[PredictionDriftResult]
    retraining_recommended: bool
    recommendations: List[str]
    visualization_paths: List[str] = field(default_factory=list)


# =============================================================================
# Thresholds Configuration
# =============================================================================

@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""
    # PSI thresholds
    psi_warning: float = 0.1
    psi_critical: float = 0.25

    # KS test thresholds
    ks_pvalue_threshold: float = 0.05

    # JS divergence thresholds
    js_warning: float = 0.1
    js_critical: float = 0.2

    # Concept drift thresholds
    error_increase_warning: float = 0.10  # 10%
    error_increase_critical: float = 0.20  # 20%

    # Prediction drift thresholds
    prediction_ks_threshold: float = 0.05

    # Feature drift percentage for overall drift
    feature_drift_pct_threshold: float = 0.30  # 30% of features


# =============================================================================
# Main Drift Detector Class
# =============================================================================

class DriftDetector:
    """
    Comprehensive drift detection system for ML models.

    Detects three types of drift:
    1. Data drift: Changes in input feature distributions
    2. Concept drift: Changes in the feature-target relationship
    3. Prediction drift: Changes in model output distribution

    Attributes:
        reference_stats: Statistics from training data
        thresholds: Configurable drift thresholds
        history: Historical drift scores for tracking
    """

    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        thresholds: Optional[DriftThresholds] = None,
        output_dir: str = "./drift_reports",
    ):
        """
        Initialize the drift detector.

        Args:
            reference_data_path: Path to CSV with reference (training) features
            reference_data: DataFrame with reference features (alternative to path)
            reference_predictions: Array of reference predictions for prediction drift
            thresholds: Custom drift thresholds
            output_dir: Directory to save drift reports
        """
        self.thresholds = thresholds or DriftThresholds()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reference statistics
        self.reference_stats: Dict[str, Dict[str, Any]] = {}
        self.reference_histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.reference_predictions: Optional[np.ndarray] = reference_predictions
        self.feature_columns: List[str] = []

        # Baseline error for concept drift
        self.baseline_error: Optional[float] = None

        # Sliding window for concept drift detection
        self.error_window: deque = deque(maxlen=10000)
        self.prediction_window: deque = deque(maxlen=10000)

        # Page-Hinkley test state
        self._ph_sum: float = 0.0
        self._ph_min: float = float('inf')
        self._ph_count: int = 0

        # History for tracking
        self.drift_history: List[Dict[str, Any]] = []

        # Latest results
        self._last_data_drift: Optional[DataDriftResult] = None
        self._last_concept_drift: Optional[ConceptDriftResult] = None
        self._last_prediction_drift: Optional[PredictionDriftResult] = None

        # Load reference data
        if reference_data_path:
            self._load_reference_data(reference_data_path)
        elif reference_data is not None:
            self._compute_reference_stats(reference_data)

        logger.info("DriftDetector initialized")

    def _load_reference_data(self, path: str) -> None:
        """Load reference data from CSV file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Reference data not found: {path}")

        logger.info(f"Loading reference data from {path}")
        df = pd.read_csv(path)
        self._compute_reference_stats(df)

    def _compute_reference_stats(self, df: pd.DataFrame) -> None:
        """Compute and store statistics from reference data."""
        self.feature_columns = list(df.columns)

        logger.info(f"Computing reference statistics for {len(self.feature_columns)} features")

        for col in self.feature_columns:
            values = df[col].dropna().values

            if len(values) == 0:
                continue

            # Basic statistics
            self.reference_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "n_samples": len(values),
            }

            # Histogram for PSI calculation (10 bins)
            hist, bin_edges = np.histogram(values, bins=10, density=True)
            self.reference_histograms[col] = (hist, bin_edges)

        logger.info(f"Reference statistics computed for {len(self.reference_stats)} features")

    def set_baseline_error(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Set the baseline error for concept drift detection.

        Args:
            predictions: Model predictions on validation set
            actuals: Actual values

        Returns:
            Baseline RMSE
        """
        self.baseline_error = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        logger.info(f"Baseline error set: {self.baseline_error:.4f}")
        return self.baseline_error

    def set_reference_predictions(self, predictions: np.ndarray) -> None:
        """Set reference predictions for prediction drift detection."""
        self.reference_predictions = np.array(predictions).flatten()
        logger.info(f"Reference predictions set: {len(self.reference_predictions)} samples")

    # =========================================================================
    # Statistical Tests
    # =========================================================================

    @staticmethod
    def _ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.

        Args:
            reference: Reference distribution samples
            current: Current distribution samples

        Returns:
            Tuple of (KS statistic, p-value)
        """
        statistic, pvalue = stats.ks_2samp(reference, current)
        return float(statistic), float(pvalue)

    @staticmethod
    def _calculate_psi(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI = Σ (actual% - expected%) * ln(actual% / expected%)

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change, monitor
        - PSI >= 0.25: Significant change, action required

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram
            epsilon: Small value to avoid log(0)

        Returns:
            PSI score
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Ensure current values fall within bin range
        bin_edges[0] = min(bin_edges[0], np.min(current))
        bin_edges[-1] = max(bin_edges[-1], np.max(current))

        # Calculate histograms (as proportions)
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        ref_pct = ref_counts / (len(reference) + epsilon)
        cur_pct = cur_counts / (len(current) + epsilon)

        # Add epsilon to avoid division by zero
        ref_pct = np.clip(ref_pct, epsilon, 1)
        cur_pct = np.clip(cur_pct, epsilon, 1)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    @staticmethod
    def _calculate_js_divergence(
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate Jensen-Shannon divergence.

        JS divergence is a symmetric measure of similarity between distributions.
        Range: [0, 1] where 0 = identical distributions

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram

        Returns:
            JS divergence score
        """
        # Create common bin edges
        all_data = np.concatenate([reference, current])
        _, bin_edges = np.histogram(all_data, bins=bins)

        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)

        # Normalize to probability distributions
        ref_prob = ref_hist / (ref_hist.sum() + 1e-10)
        cur_prob = cur_hist / (cur_hist.sum() + 1e-10)

        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, cur_prob)

        return float(js_div)

    # =========================================================================
    # Data Drift Detection
    # =========================================================================

    def detect_data_drift(
        self,
        current_data: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]],
    ) -> DataDriftResult:
        """
        Detect data drift using multiple statistical tests.

        For each feature, performs:
        - Kolmogorov-Smirnov test
        - Population Stability Index (PSI) calculation
        - Jensen-Shannon divergence

        Args:
            current_data: Current data as DataFrame, array, or dict

        Returns:
            DataDriftResult with per-feature and overall drift information
        """
        if not self.reference_stats:
            raise RuntimeError("Reference statistics not set. Load reference data first.")

        # Convert to DataFrame if needed
        if isinstance(current_data, np.ndarray):
            current_data = pd.DataFrame(current_data, columns=self.feature_columns)
        elif isinstance(current_data, dict):
            current_data = pd.DataFrame(current_data)

        logger.info(f"Detecting data drift for {len(current_data)} samples")

        feature_results = {}
        drifted_features = []

        for col in self.feature_columns:
            if col not in current_data.columns:
                logger.warning(f"Feature {col} not found in current data")
                continue

            if col not in self.reference_stats:
                continue

            current_values = current_data[col].dropna().values

            if len(current_values) < 10:
                logger.warning(f"Insufficient samples for {col}: {len(current_values)}")
                continue

            # Get reference data from stored statistics
            ref_stats = self.reference_stats[col]

            # Generate reference samples from stored statistics
            # (approximation using normal distribution)
            ref_samples = np.random.normal(
                ref_stats["mean"],
                ref_stats["std"],
                size=ref_stats["n_samples"],
            )

            # KS test
            ks_stat, ks_pval = self._ks_test(ref_samples, current_values)

            # PSI
            psi = self._calculate_psi(ref_samples, current_values)

            # JS divergence
            js_div = self._calculate_js_divergence(ref_samples, current_values)

            # Determine drift severity
            drift_detected = False
            severity = "none"

            if psi >= self.thresholds.psi_critical or js_div >= self.thresholds.js_critical:
                drift_detected = True
                severity = "critical"
                drifted_features.append(col)
            elif psi >= self.thresholds.psi_warning or js_div >= self.thresholds.js_warning:
                drift_detected = True
                severity = "warning"
                drifted_features.append(col)
            elif ks_pval < self.thresholds.ks_pvalue_threshold:
                drift_detected = True
                severity = "warning"
                drifted_features.append(col)

            feature_results[col] = FeatureDriftResult(
                feature_name=col,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                psi_score=psi,
                js_divergence=js_div,
                drift_detected=drift_detected,
                drift_severity=severity,
            )

            if drift_detected:
                logger.info(
                    f"Drift detected in {col}: PSI={psi:.4f}, JS={js_div:.4f}, "
                    f"KS p-value={ks_pval:.4f}, severity={severity}"
                )

        # Overall drift assessment
        total_features = len(feature_results)
        drifted_count = len(drifted_features)
        drift_pct = drifted_count / total_features if total_features > 0 else 0

        overall_drift = drift_pct >= self.thresholds.feature_drift_pct_threshold

        result = DataDriftResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_drift_detected=overall_drift,
            drifted_feature_count=drifted_count,
            total_features=total_features,
            drift_percentage=drift_pct,
            feature_results=feature_results,
            summary={
                "drifted_features": drifted_features,
                "critical_features": [
                    f for f, r in feature_results.items()
                    if r.drift_severity == "critical"
                ],
                "avg_psi": np.mean([r.psi_score for r in feature_results.values()]),
                "max_psi": max([r.psi_score for r in feature_results.values()]) if feature_results else 0,
                "avg_js": np.mean([r.js_divergence for r in feature_results.values()]),
            },
        )

        self._last_data_drift = result

        logger.info(
            f"Data drift result: {drifted_count}/{total_features} features drifted "
            f"({drift_pct:.1%}), overall_drift={overall_drift}"
        )

        return result

    # =========================================================================
    # Concept Drift Detection
    # =========================================================================

    def detect_concept_drift(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        window_size: int = 1000,
    ) -> ConceptDriftResult:
        """
        Detect concept drift by monitoring prediction error over time.

        Uses:
        - Sliding window error comparison vs baseline
        - Page-Hinkley test for sequential change detection

        Args:
            predictions: Recent model predictions
            actuals: Actual target values
            window_size: Size of sliding window for comparison

        Returns:
            ConceptDriftResult with drift detection and error metrics
        """
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        # Calculate errors
        errors = (predictions - actuals) ** 2

        # Add to sliding window
        for error in errors:
            self.error_window.append(error)

        # Calculate current error (RMSE over window)
        window_errors = list(self.error_window)[-window_size:]
        current_error = float(np.sqrt(np.mean(window_errors)))

        # Set baseline if not set
        if self.baseline_error is None:
            self.baseline_error = current_error
            logger.info(f"Auto-set baseline error: {self.baseline_error:.4f}")

        # Calculate error increase
        error_increase = (current_error - self.baseline_error) / (self.baseline_error + 1e-10)

        # Page-Hinkley test update
        ph_statistic = self._update_page_hinkley(current_error)

        # Determine drift
        drift_detected = error_increase >= self.thresholds.error_increase_critical

        # Determine trend
        if error_increase > self.thresholds.error_increase_warning:
            trend = "increasing"
        elif error_increase < -self.thresholds.error_increase_warning:
            trend = "decreasing"
        else:
            trend = "stable"

        result = ConceptDriftResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            drift_detected=drift_detected,
            current_error=current_error,
            baseline_error=self.baseline_error,
            error_increase_pct=error_increase * 100,
            window_size=len(window_errors),
            page_hinkley_statistic=ph_statistic,
            trend=trend,
        )

        self._last_concept_drift = result

        if drift_detected:
            logger.warning(
                f"Concept drift detected! Error increased by {error_increase:.1%} "
                f"(current: {current_error:.4f}, baseline: {self.baseline_error:.4f})"
            )
        else:
            logger.info(
                f"Concept drift check: error_increase={error_increase:.1%}, "
                f"trend={trend}, drift={drift_detected}"
            )

        return result

    def _update_page_hinkley(self, error: float, delta: float = 0.005) -> float:
        """
        Update Page-Hinkley test statistic.

        The Page-Hinkley test detects change points in sequential data.

        Args:
            error: Current error value
            delta: Magnitude of allowed change

        Returns:
            Current Page-Hinkley statistic (PHT)
        """
        self._ph_count += 1

        # Running mean
        if self._ph_count == 1:
            self._ph_mean = error
        else:
            self._ph_mean = self._ph_mean + (error - self._ph_mean) / self._ph_count

        # Cumulative sum
        self._ph_sum = self._ph_sum + (error - self._ph_mean - delta)

        # Update minimum
        self._ph_min = min(self._ph_min, self._ph_sum)

        # Page-Hinkley statistic
        pht = self._ph_sum - self._ph_min

        return pht

    # =========================================================================
    # Prediction Drift Detection
    # =========================================================================

    def detect_prediction_drift(
        self,
        recent_predictions: np.ndarray,
    ) -> PredictionDriftResult:
        """
        Detect drift in prediction distribution.

        Compares the distribution of recent predictions against
        the reference prediction distribution.

        Args:
            recent_predictions: Array of recent model predictions

        Returns:
            PredictionDriftResult with drift score and detection
        """
        recent_predictions = np.array(recent_predictions).flatten()

        if self.reference_predictions is None:
            # Use historical predictions as reference
            if len(self.prediction_window) < 100:
                # Not enough history, store current and return no drift
                for pred in recent_predictions:
                    self.prediction_window.append(pred)

                return PredictionDriftResult(
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    drift_detected=False,
                    ks_statistic=0.0,
                    ks_pvalue=1.0,
                    mean_shift=0.0,
                    std_shift=0.0,
                    drift_score=0.0,
                )

            reference = np.array(list(self.prediction_window))
        else:
            reference = self.reference_predictions

        # Add to history
        for pred in recent_predictions:
            self.prediction_window.append(pred)

        # KS test
        ks_stat, ks_pval = self._ks_test(reference, recent_predictions)

        # Distribution statistics shift
        ref_mean = np.mean(reference)
        ref_std = np.std(reference)
        cur_mean = np.mean(recent_predictions)
        cur_std = np.std(recent_predictions)

        mean_shift = (cur_mean - ref_mean) / (ref_mean + 1e-10)
        std_shift = (cur_std - ref_std) / (ref_std + 1e-10)

        # Combined drift score
        drift_score = ks_stat + abs(mean_shift) * 0.5 + abs(std_shift) * 0.3

        # Determine drift
        drift_detected = ks_pval < self.thresholds.prediction_ks_threshold

        result = PredictionDriftResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            drift_detected=drift_detected,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            mean_shift=mean_shift,
            std_shift=std_shift,
            drift_score=drift_score,
        )

        self._last_prediction_drift = result

        if drift_detected:
            logger.warning(
                f"Prediction drift detected! KS={ks_stat:.4f}, p={ks_pval:.4f}, "
                f"mean_shift={mean_shift:.2%}, std_shift={std_shift:.2%}"
            )
        else:
            logger.info(
                f"Prediction drift check: KS={ks_stat:.4f}, drift_score={drift_score:.4f}"
            )

        return result

    # =========================================================================
    # Reporting and Decision Making
    # =========================================================================

    def generate_drift_report(
        self,
        save_to_file: bool = True,
    ) -> DriftReport:
        """
        Generate comprehensive drift report.

        Combines all drift detection results into a single report
        with recommendations.

        Args:
            save_to_file: Whether to save report to JSON file

        Returns:
            DriftReport with all drift information
        """
        report_id = f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Collect recommendations
        recommendations = []
        retraining_recommended = False

        # Analyze data drift
        if self._last_data_drift:
            if self._last_data_drift.overall_drift_detected:
                retraining_recommended = True
                recommendations.append(
                    f"Data drift detected in {self._last_data_drift.drifted_feature_count} features "
                    f"({self._last_data_drift.drift_percentage:.1%}). Consider retraining."
                )

                critical_features = self._last_data_drift.summary.get("critical_features", [])
                if critical_features:
                    recommendations.append(
                        f"Critical drift in features: {', '.join(critical_features[:5])}"
                    )

        # Analyze concept drift
        if self._last_concept_drift:
            if self._last_concept_drift.drift_detected:
                retraining_recommended = True
                recommendations.append(
                    f"Concept drift detected: error increased by "
                    f"{self._last_concept_drift.error_increase_pct:.1f}%. "
                    "Model performance degraded."
                )
            elif self._last_concept_drift.trend == "increasing":
                recommendations.append(
                    "Error trend is increasing. Monitor closely."
                )

        # Analyze prediction drift
        if self._last_prediction_drift:
            if self._last_prediction_drift.drift_detected:
                recommendations.append(
                    f"Prediction distribution shift detected. "
                    f"Mean shift: {self._last_prediction_drift.mean_shift:.2%}"
                )

        # Default recommendation if nothing detected
        if not recommendations:
            recommendations.append("No significant drift detected. Model is performing as expected.")

        # Create visualizations (placeholder paths)
        viz_paths = []
        if save_to_file:
            viz_paths = [
                str(self.output_dir / f"{report_id}_data_drift.png"),
                str(self.output_dir / f"{report_id}_error_trend.png"),
                str(self.output_dir / f"{report_id}_prediction_dist.png"),
            ]

        report = DriftReport(
            report_id=report_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            data_drift=self._last_data_drift,
            concept_drift=self._last_concept_drift,
            prediction_drift=self._last_prediction_drift,
            retraining_recommended=retraining_recommended,
            recommendations=recommendations,
            visualization_paths=viz_paths,
        )

        # Save to file
        if save_to_file:
            report_path = self.output_dir / f"{report_id}.json"
            self._save_report(report, report_path)
            logger.info(f"Drift report saved to {report_path}")

        # Add to history
        self.drift_history.append({
            "report_id": report_id,
            "timestamp": report.timestamp,
            "retraining_recommended": retraining_recommended,
            "data_drift": self._last_data_drift.overall_drift_detected if self._last_data_drift else None,
            "concept_drift": self._last_concept_drift.drift_detected if self._last_concept_drift else None,
            "prediction_drift": self._last_prediction_drift.drift_detected if self._last_prediction_drift else None,
        })

        return report

    def _save_report(self, report: DriftReport, path: Path) -> None:
        """Save drift report to JSON file."""
        def serialize(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        report_dict = asdict(report)

        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2, default=serialize)

    def should_trigger_retraining(self) -> Tuple[bool, List[str]]:
        """
        Determine if retraining should be triggered.

        Combines all drift signals to make a decision.

        Returns:
            Tuple of (should_retrain, reasons)
        """
        reasons = []

        # Check data drift
        if self._last_data_drift and self._last_data_drift.overall_drift_detected:
            reasons.append(
                f"Data drift: {self._last_data_drift.drift_percentage:.1%} features drifted"
            )

        # Check concept drift
        if self._last_concept_drift and self._last_concept_drift.drift_detected:
            reasons.append(
                f"Concept drift: error increased {self._last_concept_drift.error_increase_pct:.1f}%"
            )

        # Check prediction drift
        if self._last_prediction_drift and self._last_prediction_drift.drift_detected:
            reasons.append(
                f"Prediction drift: KS statistic {self._last_prediction_drift.ks_statistic:.4f}"
            )

        should_retrain = len(reasons) > 0

        if should_retrain:
            logger.warning(f"Retraining recommended: {reasons}")
        else:
            logger.info("No retraining needed at this time")

        return should_retrain, reasons

    def reset_baseline(
        self,
        new_predictions: Optional[np.ndarray] = None,
        new_actuals: Optional[np.ndarray] = None,
    ) -> None:
        """
        Reset baseline after retraining.

        Call this after model retraining to establish new baselines.

        Args:
            new_predictions: New validation predictions for baseline error
            new_actuals: New actual values for baseline error
        """
        # Reset concept drift tracking
        self.error_window.clear()
        self._ph_sum = 0.0
        self._ph_min = float('inf')
        self._ph_count = 0

        # Reset prediction drift tracking
        self.prediction_window.clear()

        # Set new baseline error if provided
        if new_predictions is not None and new_actuals is not None:
            self.set_baseline_error(new_predictions, new_actuals)
        else:
            self.baseline_error = None

        # Clear latest results
        self._last_data_drift = None
        self._last_concept_drift = None
        self._last_prediction_drift = None

        logger.info("Drift detector baseline reset")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Create sample reference data
    np.random.seed(42)
    n_features = 10
    n_samples = 1000

    reference_data = pd.DataFrame({
        f"feature_{i}": np.random.normal(0, 1, n_samples)
        for i in range(n_features)
    })

    # Initialize detector
    detector = DriftDetector(
        reference_data=reference_data,
        output_dir="./drift_reports",
    )

    # Set baseline error
    ref_predictions = np.random.normal(50, 10, 500)
    ref_actuals = ref_predictions + np.random.normal(0, 5, 500)
    detector.set_baseline_error(ref_predictions, ref_actuals)
    detector.set_reference_predictions(ref_predictions)

    print("\n" + "=" * 60)
    print("Testing Data Drift Detection")
    print("=" * 60)

    # Test with no drift
    current_data_no_drift = pd.DataFrame({
        f"feature_{i}": np.random.normal(0, 1, 500)
        for i in range(n_features)
    })
    result1 = detector.detect_data_drift(current_data_no_drift)
    print(f"No drift scenario: drift_detected={result1.overall_drift_detected}")

    # Test with drift (shifted mean)
    current_data_with_drift = pd.DataFrame({
        f"feature_{i}": np.random.normal(0.5, 1.2, 500)  # Shifted mean and std
        for i in range(n_features)
    })
    result2 = detector.detect_data_drift(current_data_with_drift)
    print(f"With drift scenario: drift_detected={result2.overall_drift_detected}")
    print(f"Drifted features: {result2.drifted_feature_count}/{result2.total_features}")

    print("\n" + "=" * 60)
    print("Testing Concept Drift Detection")
    print("=" * 60)

    # Test with stable error
    stable_preds = np.random.normal(50, 10, 100)
    stable_actuals = stable_preds + np.random.normal(0, 5, 100)
    result3 = detector.detect_concept_drift(stable_preds, stable_actuals)
    print(f"Stable error: drift_detected={result3.drift_detected}, increase={result3.error_increase_pct:.1f}%")

    # Test with increased error (concept drift)
    drift_preds = np.random.normal(50, 10, 100)
    drift_actuals = drift_preds + np.random.normal(0, 15, 100)  # Larger error
    result4 = detector.detect_concept_drift(drift_preds, drift_actuals)
    print(f"Increased error: drift_detected={result4.drift_detected}, increase={result4.error_increase_pct:.1f}%")

    print("\n" + "=" * 60)
    print("Testing Prediction Drift Detection")
    print("=" * 60)

    # Test with stable predictions
    stable_recent = np.random.normal(50, 10, 200)
    result5 = detector.detect_prediction_drift(stable_recent)
    print(f"Stable predictions: drift_detected={result5.drift_detected}")

    # Test with shifted predictions
    shifted_recent = np.random.normal(70, 15, 200)  # Different distribution
    result6 = detector.detect_prediction_drift(shifted_recent)
    print(f"Shifted predictions: drift_detected={result6.drift_detected}")

    print("\n" + "=" * 60)
    print("Generating Drift Report")
    print("=" * 60)

    report = detector.generate_drift_report(save_to_file=True)
    print(f"Report ID: {report.report_id}")
    print(f"Retraining recommended: {report.retraining_recommended}")
    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    print("\n" + "=" * 60)
    print("Checking Retraining Trigger")
    print("=" * 60)

    should_retrain, reasons = detector.should_trigger_retraining()
    print(f"Should retrain: {should_retrain}")
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
