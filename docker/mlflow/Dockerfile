# =============================================================================
# MLflow Tracking Server Dockerfile
# =============================================================================
#
# Lightweight MLflow server with PostgreSQL backend support.
#
# Usage:
#   docker build -t pm-mlflow:latest -f docker/mlflow/Dockerfile .
#   docker run -p 5000:5000 pm-mlflow:latest
#
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="ML Platform Team" \
      version="1.0.0" \
      description="MLflow Tracking Server"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_HOST=0.0.0.0 \
    MLFLOW_PORT=5000 \
    MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db \
    MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.14 \
    google-cloud-storage==2.14.0

# Create application directories
RUN mkdir -p /mlflow/artifacts && \
    useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow /mlflow

# Switch to non-root user
USER mlflow
WORKDIR /mlflow

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server
CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
    --host ${MLFLOW_HOST} \
    --port ${MLFLOW_PORT}
