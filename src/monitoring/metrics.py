"""
Prometheus Metrics for Predictive Maintenance Service.

This module defines Prometheus metrics for monitoring:
- Model inference performance
- Drift detection scores
- Error tracking
- Business metrics

Usage:
    from src.monitoring.metrics import metrics

    # Record a prediction
    metrics.record_prediction(latency=0.05, model="ensemble")

    # Update drift score
    metrics.set_drift_score("sensor_2", 0.15)

    # Get metrics in Prometheus format
    output = metrics.export()

Integration with FastAPI:
    @app.get("/metrics")
    def get_metrics():
        return PlainTextResponse(metrics.export(), media_type="text/plain")

Integration with prometheus_client library:
    If prometheus_client is installed, this module uses it for full
    Prometheus compatibility. Otherwise, it provides a lightweight
    implementation.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("metrics")


# =============================================================================
# Try to use prometheus_client if available
# =============================================================================

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_CLIENT_AVAILABLE = True
    logger.info("Using prometheus_client library for metrics")
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False
    logger.info("prometheus_client not available, using lightweight implementation")


# =============================================================================
# Lightweight Metrics Implementation (fallback)
# =============================================================================

class LightweightCounter:
    """Simple counter metric."""

    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "LightweightCounter":
        """Return labeled counter."""
        return _LabeledCounter(self, tuple(kwargs.get(l, "") for l in self.labels))

    def inc(self, amount: float = 1) -> None:
        """Increment counter."""
        with self._lock:
            self._values[()] += amount

    def get_value(self, labels: Tuple = ()) -> float:
        """Get current value."""
        return self._values[labels]

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        for label_values, value in self._values.items():
            if label_values and self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, label_values)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class _LabeledCounter:
    """Labeled counter helper."""

    def __init__(self, parent: LightweightCounter, labels: Tuple):
        self._parent = parent
        self._labels = labels

    def inc(self, amount: float = 1) -> None:
        with self._parent._lock:
            self._parent._values[self._labels] += amount


class LightweightGauge:
    """Simple gauge metric."""

    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self._labels = labels or []
        self._values: Dict[Tuple, float] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_LabeledGauge":
        """Return labeled gauge."""
        return _LabeledGauge(
            self,
            tuple(kwargs.get(l, "") for l in self._labels)
        )

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._values[()] = value

    def get_value(self, labels: Tuple = ()) -> float:
        """Get current value."""
        return self._values.get(labels, 0)

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        for label_values, value in self._values.items():
            if label_values and self._labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self._labels, label_values)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class _LabeledGauge:
    """Labeled gauge helper."""

    def __init__(self, parent: LightweightGauge, labels: Tuple):
        self._parent = parent
        self._labels = labels

    def set(self, value: float) -> None:
        with self._parent._lock:
            self._parent._values[self._labels] = value

    def inc(self, amount: float = 1) -> None:
        with self._parent._lock:
            current = self._parent._values.get(self._labels, 0)
            self._parent._values[self._labels] = current + amount

    def dec(self, amount: float = 1) -> None:
        with self._parent._lock:
            current = self._parent._values.get(self._labels, 0)
            self._parent._values[self._labels] = current - amount


class LightweightHistogram:
    """Simple histogram metric."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        buckets: Tuple[float, ...] = None,
    ):
        self.name = name
        self.description = description
        self._labels = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: Dict[Tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sums: Dict[Tuple, float] = defaultdict(float)
        self._counts: Dict[Tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "_LabeledHistogram":
        """Return labeled histogram."""
        return _LabeledHistogram(
            self,
            tuple(kwargs.get(l, "") for l in self._labels)
        )

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            labels = ()
            self._sums[labels] += value
            self._counts[labels] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[labels][bucket] += 1

    def export(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]

        for label_values in set(list(self._bucket_counts.keys()) + [()]):
            label_str = ""
            if label_values and self._labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self._labels, label_values)
                )

            # Bucket counts (cumulative)
            cumulative = 0
            for bucket in sorted(self.buckets):
                cumulative += self._bucket_counts[label_values].get(bucket, 0)
                le_label = f'le="{bucket}"' if bucket != float("inf") else 'le="+Inf"'
                if label_str:
                    lines.append(f"{self.name}_bucket{{{label_str},{le_label}}} {cumulative}")
                else:
                    lines.append(f"{self.name}_bucket{{{le_label}}} {cumulative}")

            # Sum and count
            if label_str:
                lines.append(f"{self.name}_sum{{{label_str}}} {self._sums[label_values]}")
                lines.append(f"{self.name}_count{{{label_str}}} {self._counts[label_values]}")
            else:
                lines.append(f"{self.name}_sum {self._sums[label_values]}")
                lines.append(f"{self.name}_count {self._counts[label_values]}")

        return "\n".join(lines)


class _LabeledHistogram:
    """Labeled histogram helper."""

    def __init__(self, parent: LightweightHistogram, labels: Tuple):
        self._parent = parent
        self._labels = labels

    def observe(self, value: float) -> None:
        with self._parent._lock:
            self._parent._sums[self._labels] += value
            self._parent._counts[self._labels] += 1
            for bucket in self._parent.buckets:
                if value <= bucket:
                    self._parent._bucket_counts[self._labels][bucket] += 1


# =============================================================================
# Metrics Registry
# =============================================================================

@dataclass
class MetricDefinition:
    """Definition for a metric."""
    name: str
    description: str
    metric_type: str  # "counter", "gauge", "histogram"
    labels: List[str] = field(default_factory=list)
    buckets: Optional[Tuple[float, ...]] = None


class MetricsRegistry:
    """
    Central registry for all Prometheus metrics.

    Provides a unified interface for metric operations regardless
    of whether prometheus_client is installed.
    """

    # Metric definitions
    METRIC_DEFINITIONS = {
        # Prediction metrics
        "predictions_total": MetricDefinition(
            name="predictions_total",
            description="Total number of predictions made",
            metric_type="counter",
            labels=["model_type", "status"],
        ),
        "prediction_latency_seconds": MetricDefinition(
            name="prediction_latency_seconds",
            description="Prediction latency in seconds",
            metric_type="histogram",
            labels=["model_type"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        ),
        "batch_predictions_total": MetricDefinition(
            name="batch_predictions_total",
            description="Total number of batch predictions (sample count)",
            metric_type="counter",
            labels=["status"],
        ),

        # Error metrics
        "model_errors_total": MetricDefinition(
            name="model_errors_total",
            description="Total number of model errors",
            metric_type="counter",
            labels=["error_type", "model_type"],
        ),

        # Drift metrics
        "drift_score": MetricDefinition(
            name="drift_score",
            description="Current drift score per feature",
            metric_type="gauge",
            labels=["feature", "drift_type"],
        ),
        "data_drift_detected": MetricDefinition(
            name="data_drift_detected",
            description="Whether data drift is currently detected (1=yes, 0=no)",
            metric_type="gauge",
            labels=[],
        ),
        "concept_drift_detected": MetricDefinition(
            name="concept_drift_detected",
            description="Whether concept drift is currently detected (1=yes, 0=no)",
            metric_type="gauge",
            labels=[],
        ),
        "prediction_drift_detected": MetricDefinition(
            name="prediction_drift_detected",
            description="Whether prediction drift is currently detected (1=yes, 0=no)",
            metric_type="gauge",
            labels=[],
        ),
        "drift_check_timestamp": MetricDefinition(
            name="drift_check_timestamp",
            description="Timestamp of last drift check (Unix epoch)",
            metric_type="gauge",
            labels=[],
        ),

        # Model metrics
        "model_version_info": MetricDefinition(
            name="model_version_info",
            description="Model version information",
            metric_type="gauge",
            labels=["model_type", "version"],
        ),
        "model_reload_total": MetricDefinition(
            name="model_reload_total",
            description="Number of times models have been reloaded",
            metric_type="counter",
            labels=["model_type", "reason"],
        ),

        # Business metrics
        "rul_prediction_value": MetricDefinition(
            name="rul_prediction_value",
            description="Current RUL prediction value",
            metric_type="gauge",
            labels=["unit_id"],
        ),
        "critical_rul_alerts_total": MetricDefinition(
            name="critical_rul_alerts_total",
            description="Number of critical RUL alerts (RUL < threshold)",
            metric_type="counter",
            labels=["severity"],
        ),

        # System metrics
        "uptime_seconds": MetricDefinition(
            name="uptime_seconds",
            description="Service uptime in seconds",
            metric_type="gauge",
            labels=[],
        ),
        "active_connections": MetricDefinition(
            name="active_connections",
            description="Number of active connections",
            metric_type="gauge",
            labels=[],
        ),
    }

    def __init__(self, namespace: str = "predictive_maintenance"):
        """
        Initialize the metrics registry.

        Args:
            namespace: Prefix for all metric names
        """
        self.namespace = namespace
        self._metrics: Dict[str, Any] = {}
        self._start_time = time.time()
        self._lock = threading.Lock()

        # Initialize metrics
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._registry = CollectorRegistry()
            self._init_prometheus_client_metrics()
        else:
            self._registry = None
            self._init_lightweight_metrics()

        logger.info(f"Metrics registry initialized with namespace: {namespace}")

    def _init_prometheus_client_metrics(self) -> None:
        """Initialize metrics using prometheus_client library."""
        for name, definition in self.METRIC_DEFINITIONS.items():
            full_name = f"{self.namespace}_{definition.name}"

            if definition.metric_type == "counter":
                self._metrics[name] = Counter(
                    full_name,
                    definition.description,
                    definition.labels,
                    registry=self._registry,
                )
            elif definition.metric_type == "gauge":
                self._metrics[name] = Gauge(
                    full_name,
                    definition.description,
                    definition.labels,
                    registry=self._registry,
                )
            elif definition.metric_type == "histogram":
                self._metrics[name] = Histogram(
                    full_name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets or Histogram.DEFAULT_BUCKETS,
                    registry=self._registry,
                )

    def _init_lightweight_metrics(self) -> None:
        """Initialize metrics using lightweight implementation."""
        for name, definition in self.METRIC_DEFINITIONS.items():
            full_name = f"{self.namespace}_{definition.name}"

            if definition.metric_type == "counter":
                self._metrics[name] = LightweightCounter(
                    full_name,
                    definition.description,
                    definition.labels,
                )
            elif definition.metric_type == "gauge":
                self._metrics[name] = LightweightGauge(
                    full_name,
                    definition.description,
                    definition.labels,
                )
            elif definition.metric_type == "histogram":
                self._metrics[name] = LightweightHistogram(
                    full_name,
                    definition.description,
                    definition.labels,
                    definition.buckets,
                )

    # =========================================================================
    # High-Level Recording Methods
    # =========================================================================

    def record_prediction(
        self,
        latency: float,
        model_type: str = "ensemble",
        success: bool = True,
    ) -> None:
        """
        Record a prediction event.

        Args:
            latency: Prediction latency in seconds
            model_type: Type of model used
            success: Whether prediction was successful
        """
        status = "success" if success else "error"

        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["predictions_total"].labels(
                model_type=model_type, status=status
            ).inc()
            self._metrics["prediction_latency_seconds"].labels(
                model_type=model_type
            ).observe(latency)
        else:
            self._metrics["predictions_total"].labels(
                model_type=model_type, status=status
            ).inc()
            self._metrics["prediction_latency_seconds"].labels(
                model_type=model_type
            ).observe(latency)

        if not success:
            self.record_error("prediction_failed", model_type)

    def record_batch_prediction(
        self,
        sample_count: int,
        latency: float,
        success: bool = True,
    ) -> None:
        """
        Record a batch prediction event.

        Args:
            sample_count: Number of samples in batch
            latency: Total batch latency in seconds
            success: Whether batch prediction was successful
        """
        status = "success" if success else "error"

        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["batch_predictions_total"].labels(status=status).inc(sample_count)
        else:
            self._metrics["batch_predictions_total"].labels(status=status).inc(sample_count)

        # Also record individual prediction metrics
        self.record_prediction(latency / max(sample_count, 1), "ensemble", success)

    def record_error(
        self,
        error_type: str,
        model_type: str = "unknown",
    ) -> None:
        """
        Record a model error.

        Args:
            error_type: Type of error (e.g., "prediction_failed", "preprocessing_error")
            model_type: Type of model that caused the error
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["model_errors_total"].labels(
                error_type=error_type, model_type=model_type
            ).inc()
        else:
            self._metrics["model_errors_total"].labels(
                error_type=error_type, model_type=model_type
            ).inc()

    def set_drift_score(
        self,
        feature: str,
        score: float,
        drift_type: str = "psi",
    ) -> None:
        """
        Set drift score for a feature.

        Args:
            feature: Feature name
            score: Drift score value
            drift_type: Type of drift metric (psi, ks, js)
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["drift_score"].labels(
                feature=feature, drift_type=drift_type
            ).set(score)
        else:
            self._metrics["drift_score"].labels(
                feature=feature, drift_type=drift_type
            ).set(score)

    def set_drift_detected(
        self,
        data_drift: bool = False,
        concept_drift: bool = False,
        prediction_drift: bool = False,
    ) -> None:
        """
        Set drift detection status flags.

        Args:
            data_drift: Whether data drift is detected
            concept_drift: Whether concept drift is detected
            prediction_drift: Whether prediction drift is detected
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["data_drift_detected"].set(1 if data_drift else 0)
            self._metrics["concept_drift_detected"].set(1 if concept_drift else 0)
            self._metrics["prediction_drift_detected"].set(1 if prediction_drift else 0)
            self._metrics["drift_check_timestamp"].set(time.time())
        else:
            self._metrics["data_drift_detected"].set(1 if data_drift else 0)
            self._metrics["concept_drift_detected"].set(1 if concept_drift else 0)
            self._metrics["prediction_drift_detected"].set(1 if prediction_drift else 0)
            self._metrics["drift_check_timestamp"].set(time.time())

    def set_model_version(self, model_type: str, version: str) -> None:
        """
        Set model version information.

        Args:
            model_type: Type of model
            version: Version string
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["model_version_info"].labels(
                model_type=model_type, version=version
            ).set(1)
        else:
            self._metrics["model_version_info"].labels(
                model_type=model_type, version=version
            ).set(1)

    def record_model_reload(self, model_type: str, reason: str = "manual") -> None:
        """
        Record a model reload event.

        Args:
            model_type: Type of model reloaded
            reason: Reason for reload (manual, drift, scheduled)
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["model_reload_total"].labels(
                model_type=model_type, reason=reason
            ).inc()
        else:
            self._metrics["model_reload_total"].labels(
                model_type=model_type, reason=reason
            ).inc()

    def set_rul_prediction(self, unit_id: str, rul: float) -> None:
        """
        Set current RUL prediction for a unit.

        Args:
            unit_id: Unit identifier
            rul: Predicted RUL value
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["rul_prediction_value"].labels(unit_id=unit_id).set(rul)
        else:
            self._metrics["rul_prediction_value"].labels(unit_id=unit_id).set(rul)

        # Check for critical RUL
        if rul < 10:
            self.record_critical_alert("critical")
        elif rul < 25:
            self.record_critical_alert("warning")

    def record_critical_alert(self, severity: str) -> None:
        """
        Record a critical RUL alert.

        Args:
            severity: Alert severity (warning, critical)
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["critical_rul_alerts_total"].labels(severity=severity).inc()
        else:
            self._metrics["critical_rul_alerts_total"].labels(severity=severity).inc()

    def update_uptime(self) -> None:
        """Update the uptime metric."""
        uptime = time.time() - self._start_time
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["uptime_seconds"].set(uptime)
        else:
            self._metrics["uptime_seconds"].set(uptime)

    def set_active_connections(self, count: int) -> None:
        """
        Set number of active connections.

        Args:
            count: Number of active connections
        """
        if PROMETHEUS_CLIENT_AVAILABLE:
            self._metrics["active_connections"].set(count)
        else:
            self._metrics["active_connections"].set(count)

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export(self) -> str:
        """
        Export all metrics in Prometheus exposition format.

        Returns:
            String with metrics in Prometheus format
        """
        self.update_uptime()

        if PROMETHEUS_CLIENT_AVAILABLE:
            return generate_latest(self._registry).decode("utf-8")
        else:
            lines = []
            for metric in self._metrics.values():
                lines.append(metric.export())
            return "\n\n".join(lines)

    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        if PROMETHEUS_CLIENT_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain; version=0.0.4; charset=utf-8"

    def get_metric(self, name: str) -> Any:
        """
        Get a metric by name.

        Args:
            name: Metric name (without namespace)

        Returns:
            The metric object
        """
        return self._metrics.get(name)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metric values.

        Returns:
            Dictionary with metric summaries
        """
        self.update_uptime()

        summary = {
            "uptime_seconds": time.time() - self._start_time,
            "metrics_count": len(self._metrics),
            "prometheus_client_available": PROMETHEUS_CLIENT_AVAILABLE,
        }

        return summary


# =============================================================================
# Global Metrics Instance
# =============================================================================

# Create a global metrics instance
metrics = MetricsRegistry()


# =============================================================================
# Context Manager for Timing
# =============================================================================

class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        metric_name: str = "prediction_latency_seconds",
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize timer.

        Args:
            metric_name: Name of histogram metric to record to
            labels: Labels for the metric
        """
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.time() - self.start_time

        metric = metrics.get_metric(self.metric_name)
        if metric:
            if self.labels:
                metric.labels(**self.labels).observe(self.elapsed)
            else:
                metric.observe(self.elapsed)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Prometheus Metrics Demo")
    print("=" * 60)

    # Record some sample metrics
    print("\nRecording sample predictions...")
    for i in range(10):
        latency = 0.05 + (i * 0.01)
        metrics.record_prediction(latency, model_type="ensemble", success=True)

    # Record a failed prediction
    metrics.record_prediction(0.1, model_type="ensemble", success=False)

    # Record batch prediction
    metrics.record_batch_prediction(sample_count=100, latency=0.5, success=True)

    # Set drift scores
    print("\nSetting drift scores...")
    for i in range(5):
        metrics.set_drift_score(f"feature_{i}", score=0.1 + (i * 0.05), drift_type="psi")
        metrics.set_drift_score(f"feature_{i}", score=0.05 + (i * 0.03), drift_type="ks")

    # Set drift detection status
    metrics.set_drift_detected(data_drift=True, concept_drift=False, prediction_drift=False)

    # Set model versions
    metrics.set_model_version("random_forest", "v1.2.3")
    metrics.set_model_version("xgboost", "v2.0.1")
    metrics.set_model_version("lstm", "v1.0.0")

    # Set RUL predictions
    metrics.set_rul_prediction("unit_1", 85.5)
    metrics.set_rul_prediction("unit_2", 12.3)  # Will trigger warning
    metrics.set_rul_prediction("unit_3", 5.1)   # Will trigger critical

    # Use timer context manager
    print("\nUsing timer context manager...")
    with Timer("prediction_latency_seconds", {"model_type": "lstm"}) as t:
        time.sleep(0.1)  # Simulate work
    print(f"Operation took {t.elapsed:.3f} seconds")

    # Export metrics
    print("\n" + "=" * 60)
    print("Exported Metrics (Prometheus Format)")
    print("=" * 60)
    print(metrics.export())

    # Print summary
    print("\n" + "=" * 60)
    print("Metrics Summary")
    print("=" * 60)
    summary = metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
