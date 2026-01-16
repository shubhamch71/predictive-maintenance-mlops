# =============================================================================
# Multi-stage Dockerfile for Predictive Maintenance Training Components
# =============================================================================
#
# This Dockerfile builds optimized images for Kubeflow pipeline components:
# - Data ingestion
# - Feature engineering
# - Model training (RF, XGBoost, LSTM)
# - Model evaluation
#
# Usage:
#   # Build image
#   docker build -t predictive-maintenance-training:latest -f docker/training/Dockerfile .
#
#   # Run specific component
#   docker run predictive-maintenance-training:latest python -m kubeflow.components.data_ingestion --help
#
#   # Override command
#   docker run predictive-maintenance-training:latest python -m kubeflow.components.train_rf \
#       --features-path /data/features.csv --labels-path /data/labels.csv
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile packages
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
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY docker/training/requirements.txt /tmp/requirements.txt

# Install Python packages in stages to optimize caching
# Stage 1: Core data science packages
RUN pip install --upgrade pip wheel setuptools && \
    pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.11.4

# Stage 2: ML frameworks
RUN pip install \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    joblib==1.3.2

# Stage 3: TensorFlow (CPU-only for smaller image)
RUN pip install \
    tensorflow-cpu==2.15.0

# Stage 4: MLflow and other tools
RUN pip install \
    mlflow==2.9.2 \
    kfp==2.4.0 \
    pyyaml==6.0.1 \
    requests==2.31.0

# Clean up pip cache
RUN pip cache purge

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Labels for container metadata
LABEL maintainer="ML Platform Team" \
      version="1.0.0" \
      description="Predictive Maintenance Training Components" \
      org.opencontainers.image.source="https://github.com/org/predictive-maintenance-mlops"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # TensorFlow settings
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    # Application settings
    APP_HOME=/app \
    DATA_DIR=/data \
    MODEL_DIR=/models

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for numpy/scipy
    libopenblas0 \
    libgomp1 \
    # Useful utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 mluser && \
    useradd --uid 1000 --gid mluser --shell /bin/bash --create-home mluser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories
RUN mkdir -p ${APP_HOME} ${DATA_DIR} ${MODEL_DIR} && \
    chown -R mluser:mluser ${APP_HOME} ${DATA_DIR} ${MODEL_DIR}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=mluser:mluser src/ ${APP_HOME}/src/
COPY --chown=mluser:mluser kubeflow/ ${APP_HOME}/kubeflow/
COPY --chown=mluser:mluser data/download_data.py ${APP_HOME}/data/

# Create __init__.py files if missing
RUN touch ${APP_HOME}/__init__.py && \
    touch ${APP_HOME}/src/__init__.py && \
    touch ${APP_HOME}/kubeflow/__init__.py && \
    touch ${APP_HOME}/kubeflow/components/__init__.py

# Set PYTHONPATH
ENV PYTHONPATH="${APP_HOME}:${PYTHONPATH}"

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow; import sklearn; import xgboost; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-c", "print('Predictive Maintenance Training Container Ready. Use: python -m kubeflow.components.<component_name>')"]

# -----------------------------------------------------------------------------
# Stage 3: Component-specific images (optional - build with --target)
# -----------------------------------------------------------------------------

# Data Ingestion Component
FROM runtime as data-ingestion
CMD ["python", "-m", "kubeflow.components.data_ingestion"]

# Feature Engineering Component
FROM runtime as feature-engineering
CMD ["python", "-m", "kubeflow.components.feature_engineering"]

# Random Forest Training Component
FROM runtime as train-rf
CMD ["python", "-m", "kubeflow.components.train_rf"]

# XGBoost Training Component
FROM runtime as train-xgboost
CMD ["python", "-m", "kubeflow.components.train_xgboost"]

# LSTM Training Component
FROM runtime as train-lstm
CMD ["python", "-m", "kubeflow.components.train_lstm"]

# Model Evaluation Component
FROM runtime as evaluate
CMD ["python", "-m", "kubeflow.components.evaluate_models"]
