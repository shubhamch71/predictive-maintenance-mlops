"""
FastAPI Serving Application for Predictive Maintenance Models

Production-ready REST API for RUL (Remaining Useful Life) predictions
using an ensemble of RF, XGBoost, and LSTM models.

Endpoints:
    POST /predict         - Single sample prediction
    POST /predict_batch   - Batch predictions
    GET  /health          - Health check
    GET  /metrics         - Prometheus metrics
    GET  /explain         - SHAP explanation for last prediction
    GET  /model-info      - Model information

Usage:
    # Start server
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

    # With auto-reload for development
    uvicorn src.serving.api:app --reload

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URI (default: http://localhost:5000)
    MODEL_DIR: Local model directory (fallback if MLflow unavailable)
    SCALER_PATH: Path to scaler pickle file
    LOG_LEVEL: Logging level (default: INFO)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .predictor import Predictor, PredictionResult, get_predictor, initialize_predictor

# =============================================================================
# Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
SCALER_PATH = os.getenv("SCALER_PATH", "./models/scaler.pkl")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")


# =============================================================================
# Prometheus Metrics
# =============================================================================

class PrometheusMetrics:
    """Simple Prometheus metrics collector."""

    def __init__(self):
        self.prediction_count = 0
        self.prediction_errors = 0
        self.batch_prediction_count = 0
        self.latency_sum = 0.0
        self.latency_count = 0
        self.latency_buckets = {
            0.01: 0, 0.025: 0, 0.05: 0, 0.1: 0, 0.25: 0,
            0.5: 0, 1.0: 0, 2.5: 0, 5.0: 0, 10.0: 0, float("inf"): 0
        }
        self._start_time = time.time()

    def record_prediction(self, latency: float, success: bool = True):
        """Record a prediction event."""
        if success:
            self.prediction_count += 1
        else:
            self.prediction_errors += 1

        self.latency_sum += latency
        self.latency_count += 1

        # Update histogram buckets
        for bucket in sorted(self.latency_buckets.keys()):
            if latency <= bucket:
                self.latency_buckets[bucket] += 1

    def record_batch_prediction(self, batch_size: int, latency: float):
        """Record a batch prediction event."""
        self.batch_prediction_count += batch_size
        self.record_prediction(latency, success=True)

    def format_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []

        # Counter: prediction count
        lines.append("# HELP predictions_total Total number of predictions made")
        lines.append("# TYPE predictions_total counter")
        lines.append(f"predictions_total {self.prediction_count}")

        # Counter: errors
        lines.append("# HELP prediction_errors_total Total prediction errors")
        lines.append("# TYPE prediction_errors_total counter")
        lines.append(f"prediction_errors_total {self.prediction_errors}")

        # Counter: batch predictions
        lines.append("# HELP batch_predictions_total Total samples in batch predictions")
        lines.append("# TYPE batch_predictions_total counter")
        lines.append(f"batch_predictions_total {self.batch_prediction_count}")

        # Histogram: latency
        lines.append("# HELP prediction_latency_seconds Prediction latency in seconds")
        lines.append("# TYPE prediction_latency_seconds histogram")

        cumulative = 0
        for bucket, count in sorted(self.latency_buckets.items()):
            cumulative += count
            if bucket == float("inf"):
                lines.append(f'prediction_latency_seconds_bucket{{le="+Inf"}} {cumulative}')
            else:
                lines.append(f'prediction_latency_seconds_bucket{{le="{bucket}"}} {cumulative}')

        lines.append(f"prediction_latency_seconds_sum {self.latency_sum}")
        lines.append(f"prediction_latency_seconds_count {self.latency_count}")

        # Gauge: uptime
        uptime = time.time() - self._start_time
        lines.append("# HELP uptime_seconds Time since server start")
        lines.append("# TYPE uptime_seconds gauge")
        lines.append(f"uptime_seconds {uptime:.2f}")

        return "\n".join(lines)


# Global metrics instance
metrics = PrometheusMetrics()


# =============================================================================
# Pydantic Models
# =============================================================================

class SensorInput(BaseModel):
    """Input model for sensor readings."""

    # Operational settings
    op_setting_1: float = Field(..., description="Operational setting 1")
    op_setting_2: float = Field(..., description="Operational setting 2")
    op_setting_3: float = Field(..., description="Operational setting 3")

    # 21 sensor readings
    sensor_1: float = Field(..., description="Sensor 1 reading")
    sensor_2: float = Field(..., description="Sensor 2 reading")
    sensor_3: float = Field(..., description="Sensor 3 reading")
    sensor_4: float = Field(..., description="Sensor 4 reading")
    sensor_5: float = Field(..., description="Sensor 5 reading")
    sensor_6: float = Field(..., description="Sensor 6 reading")
    sensor_7: float = Field(..., description="Sensor 7 reading")
    sensor_8: float = Field(..., description="Sensor 8 reading")
    sensor_9: float = Field(..., description="Sensor 9 reading")
    sensor_10: float = Field(..., description="Sensor 10 reading")
    sensor_11: float = Field(..., description="Sensor 11 reading")
    sensor_12: float = Field(..., description="Sensor 12 reading")
    sensor_13: float = Field(..., description="Sensor 13 reading")
    sensor_14: float = Field(..., description="Sensor 14 reading")
    sensor_15: float = Field(..., description="Sensor 15 reading")
    sensor_16: float = Field(..., description="Sensor 16 reading")
    sensor_17: float = Field(..., description="Sensor 17 reading")
    sensor_18: float = Field(..., description="Sensor 18 reading")
    sensor_19: float = Field(..., description="Sensor 19 reading")
    sensor_20: float = Field(..., description="Sensor 20 reading")
    sensor_21: float = Field(..., description="Sensor 21 reading")

    @field_validator("*", mode="before")
    @classmethod
    def validate_numeric(cls, v):
        """Ensure all values are valid numbers."""
        if v is None:
            raise ValueError("Sensor values cannot be null")
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value: {v}")

    model_config = {
        "json_schema_extra": {
            "example": {
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 641.82,
                "sensor_3": 1589.70,
                "sensor_4": 1400.60,
                "sensor_5": 14.62,
                "sensor_6": 21.61,
                "sensor_7": 554.36,
                "sensor_8": 2388.06,
                "sensor_9": 9046.19,
                "sensor_10": 1.30,
                "sensor_11": 47.47,
                "sensor_12": 521.66,
                "sensor_13": 2388.02,
                "sensor_14": 8138.62,
                "sensor_15": 8.4195,
                "sensor_16": 0.03,
                "sensor_17": 392.0,
                "sensor_18": 2388.0,
                "sensor_19": 100.0,
                "sensor_20": 39.06,
                "sensor_21": 23.4190,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    rul: float = Field(..., description="Predicted Remaining Useful Life (cycles)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    model_version: str = Field(..., description="Model version used for prediction")
    model_type: str = Field(default="ensemble", description="Type of model used")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    individual_predictions: Optional[Dict[str, float]] = Field(
        default=None,
        description="Individual model predictions (if requested)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rul": 85.5,
                "confidence": 0.92,
                "model_version": "random_forest:v3, xgboost:v2, lstm:v1",
                "model_type": "ensemble",
                "timestamp": "2024-01-15T10:30:00.000Z",
                "individual_predictions": {
                    "random_forest": 87.2,
                    "xgboost": 84.1,
                    "lstm": 85.3
                }
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    samples: List[SensorInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of sensor readings (max 1000)"
    )
    return_individual: bool = Field(
        default=False,
        description="Include individual model predictions"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status (healthy/degraded/unhealthy)")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    model_count: int = Field(..., description="Number of models loaded")
    scaler_loaded: bool = Field(..., description="Whether scaler is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    models: Dict[str, Dict[str, Any]]
    ensemble_weights: Dict[str, float]
    feature_count: int
    sequence_length: int


class ExplainResponse(BaseModel):
    """Response model for prediction explanation."""

    prediction: float
    confidence: float
    method: str
    top_features: List[Dict[str, Any]]
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    timestamp: str


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Predictive Maintenance API...")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Model Dir: {MODEL_DIR}")

    try:
        # Try MLflow first, fall back to local
        try:
            initialize_predictor(
                mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                scaler_path=SCALER_PATH,
            )
            logger.info("Models loaded from MLflow")
        except Exception as e:
            logger.warning(f"MLflow load failed: {e}, trying local models")
            initialize_predictor(
                local_model_dir=MODEL_DIR,
                scaler_path=SCALER_PATH,
            )
            logger.info("Models loaded from local directory")

        predictor = get_predictor()
        if predictor.is_loaded:
            logger.info(f"Loaded models: {list(predictor.model_info.keys())}")
        else:
            logger.warning("No models loaded! Predictions will fail.")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API starting without models - predictions will fail")

    logger.info("API startup complete")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="""
    REST API for predicting Remaining Useful Life (RUL) of turbofan engines
    using an ensemble of ML models (Random Forest, XGBoost, LSTM).

    ## Features
    - Single and batch predictions
    - Prometheus metrics
    - SHAP-based explainability
    - Health monitoring

    ## Models
    Trained on NASA C-MAPSS Turbofan Engine Degradation dataset.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Log request
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    return response


# =============================================================================
# Endpoints
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Predictions"],
    summary="Single Sample Prediction",
    description="Predict RUL for a single set of sensor readings",
)
async def predict(
    input_data: SensorInput,
    return_individual: bool = False,
) -> PredictionResponse:
    """
    Make a RUL prediction for a single sample.

    **Input**: 21 sensor readings + 3 operational settings

    **Output**: Predicted RUL (cycles), confidence score, model info
    """
    start_time = time.time()

    predictor = get_predictor()

    if not predictor.is_loaded:
        metrics.record_prediction(0, success=False)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Service is starting up.",
        )

    try:
        # Convert Pydantic model to dict
        sensor_data = input_data.model_dump()

        # Make prediction
        result = predictor.predict(
            sensor_data,
            return_individual=return_individual,
        )

        latency = time.time() - start_time
        metrics.record_prediction(latency, success=True)

        return PredictionResponse(
            rul=result.rul,
            confidence=result.confidence,
            model_version=result.model_version,
            model_type=result.model_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            individual_predictions=result.individual_predictions or None,
        )

    except Exception as e:
        latency = time.time() - start_time
        metrics.record_prediction(latency, success=False)
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Predictions"],
    summary="Batch Prediction",
    description="Predict RUL for multiple samples (max 1000)",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make RUL predictions for multiple samples.

    Useful for bulk inference on historical data.
    Maximum 1000 samples per request.
    """
    start_time = time.time()

    predictor = get_predictor()

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded",
        )

    try:
        # Convert to list of dicts
        samples = [s.model_dump() for s in request.samples]

        # Clear sequence buffer for fresh batch
        predictor.clear_sequence_buffer()

        # Make predictions
        results = predictor.predict_batch(samples)

        latency = time.time() - start_time
        metrics.record_batch_prediction(len(samples), latency)

        predictions = [
            PredictionResponse(
                rul=r.rul,
                confidence=r.confidence,
                model_version=r.model_version,
                model_type=r.model_type,
                timestamp=datetime.utcnow().isoformat() + "Z",
                individual_predictions=r.individual_predictions if request.return_individual else None,
            )
            for r in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            processing_time_ms=latency * 1000,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health Check",
    description="Check service health and model status",
)
async def health() -> HealthResponse:
    """
    Health check endpoint for monitoring and orchestration.

    Returns service status and loaded model information.
    """
    predictor = get_predictor()
    health_info = predictor.get_health_info()

    # Determine overall status
    if not predictor.is_loaded:
        status_str = "unhealthy"
    elif health_info["model_count"] < 3:
        status_str = "degraded"
    else:
        status_str = "healthy"

    return HealthResponse(
        status=status_str,
        models_loaded=health_info["models_loaded"],
        model_count=health_info["model_count"],
        scaler_loaded=health_info["scaler_loaded"],
        uptime_seconds=time.time() - metrics._start_time,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["Monitoring"],
    summary="Prometheus Metrics",
    description="Metrics in Prometheus exposition format",
)
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.

    Exposes:
    - predictions_total: Counter of predictions
    - prediction_errors_total: Counter of errors
    - prediction_latency_seconds: Histogram of latencies
    - uptime_seconds: Gauge of service uptime
    """
    return PlainTextResponse(
        content=metrics.format_prometheus(),
        media_type="text/plain; version=0.0.4",
    )


@app.get(
    "/explain",
    response_model=ExplainResponse,
    responses={
        200: {"description": "Explanation generated"},
        400: {"model": ErrorResponse, "description": "No prediction to explain"},
        500: {"model": ErrorResponse, "description": "Explanation failed"},
    },
    tags=["Explainability"],
    summary="Explain Last Prediction",
    description="Get SHAP-based explanation for the last prediction",
)
async def explain(top_k: int = 5) -> ExplainResponse:
    """
    Explain the last prediction using feature importance.

    Returns the top K most influential features for the prediction.
    Uses SHAP values if available, otherwise falls back to model feature importance.
    """
    predictor = get_predictor()

    try:
        explanation = predictor.explain_prediction(top_k=top_k)

        return ExplainResponse(
            prediction=explanation["prediction"],
            confidence=explanation["confidence"],
            method=explanation.get("method", "unknown"),
            top_features=explanation["top_features"],
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}",
        )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Monitoring"],
    summary="Model Information",
    description="Get detailed information about loaded models",
)
async def model_info() -> ModelInfoResponse:
    """
    Get detailed information about the loaded models.

    Returns model versions, ensemble weights, and configuration.
    """
    predictor = get_predictor()
    health_info = predictor.get_health_info()

    return ModelInfoResponse(
        models=health_info.get("model_info", {}),
        ensemble_weights=predictor.model_weights,
        feature_count=health_info.get("feature_count", 0),
        sequence_length=predictor.sequence_length,
    )


@app.get(
    "/",
    tags=["Info"],
    summary="API Info",
    include_in_schema=False,
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=LOG_LEVEL.lower(),
    )
