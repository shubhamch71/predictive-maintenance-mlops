# Industrial Equipment Health Monitoring Platform

Production-grade MLOps platform for predictive maintenance using the NASA Turbofan Engine dataset.

## Overview

This platform predicts equipment failures using machine learning models deployed with Kubeflow, MLflow, and KServe, with built-in drift detection and monitoring.

## Features

- Data ingestion and preprocessing pipeline
- Multiple ML models (Random Forest, XGBoost, LSTM, Ensemble)
- Kubeflow pipelines for automated training
- MLflow for experiment tracking and model registry
- KServe for model serving
- Evidently for drift detection
- Prometheus metrics for monitoring

## Project Structure

```
predictive-maintenance-mlops/
├── data/               # Raw and processed data
├── notebooks/          # Exploratory data analysis
├── src/                # Source code
├── kubeflow/           # Kubeflow pipeline definitions
├── docker/             # Docker configurations
├── k8s/                # Kubernetes manifests
├── scripts/            # Utility scripts
└── tests/              # Unit tests
```

## Setup

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Kubernetes cluster (local: minikube/kind)
- kubectl configured

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python data/download_data.py
```

## Usage

### Local Development

```bash
docker-compose up -d
```

### Run Training Pipeline

```bash
./scripts/run_training_pipeline.sh
```

## Architecture

TODO: Add architecture diagram

## Contributing

TODO: Add contribution guidelines

## License

TODO: Add license information
