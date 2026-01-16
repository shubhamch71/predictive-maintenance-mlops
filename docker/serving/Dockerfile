# =============================================================================
# Dockerfile for Predictive Maintenance Serving API
# =============================================================================
#
# Optimized FastAPI serving image for RUL predictions.
# Target: Fast startup (<5s), small footprint, production-ready.
#
# Usage:
#   # Build image
#   docker build -t pm-serving:latest -f docker/serving/Dockerfile .
#
#   # Run container
#   docker run -p 8000:8000 \
#       -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#       -e MODEL_DIR=/models \
#       pm-serving:latest
#
#   # Run with volume mount for local models
#   docker run -p 8000:8000 \
#       -v /path/to/models:/models:ro \
#       pm-serving:latest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY docker/serving/requirements.txt /tmp/requirements.txt

# Install dependencies in optimal order for caching
RUN pip install --upgrade pip wheel && \
    pip install -r /tmp/requirements.txt && \
    pip cache purge

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="ML Platform Team" \
      version="1.0.0" \
      description="Predictive Maintenance Serving API" \
      org.opencontainers.image.source="https://github.com/org/predictive-maintenance-mlops"

# Set environment variables for optimal performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Disable TensorFlow warnings
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    # Application settings
    APP_HOME=/app \
    MODEL_DIR=/models \
    PORT=8000 \
    # Uvicorn settings for fast startup
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_WORKERS=1 \
    UVICORN_LOOP=uvloop \
    UVICORN_HTTP=httptools \
    # MLflow settings
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    # Preload models on startup
    PRELOAD_MODELS=true

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for numpy/scipy
    libopenblas0 \
    libgomp1 \
    # Health check utility
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories
RUN mkdir -p ${APP_HOME} ${MODEL_DIR} && \
    chown -R appuser:appuser ${APP_HOME} ${MODEL_DIR}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code (minimal set for serving)
COPY --chown=appuser:appuser src/serving/ ${APP_HOME}/src/serving/
COPY --chown=appuser:appuser src/models/ ${APP_HOME}/src/models/
COPY --chown=appuser:appuser src/monitoring/ ${APP_HOME}/src/monitoring/
COPY --chown=appuser:appuser src/__init__.py ${APP_HOME}/src/
COPY --chown=appuser:appuser src/config.py ${APP_HOME}/src/

# Create __init__.py files
RUN touch ${APP_HOME}/__init__.py && \
    touch ${APP_HOME}/src/serving/__init__.py && \
    touch ${APP_HOME}/src/models/__init__.py && \
    touch ${APP_HOME}/src/monitoring/__init__.py

# Set PYTHONPATH
ENV PYTHONPATH="${APP_HOME}:${PYTHONPATH}"

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check - fast endpoint
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Precompile Python files for faster startup
RUN python -m compileall -q ${APP_HOME}/src/

# Default command - optimized for fast startup
CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]

# -----------------------------------------------------------------------------
# Stage 3: Development image (optional)
# -----------------------------------------------------------------------------
FROM runtime as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    isort

USER appuser

# Enable auto-reload for development
CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/src"]
