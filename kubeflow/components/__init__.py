"""
Kubeflow Pipeline Components for Predictive Maintenance.

This package contains standalone Python scripts that can be
containerized and used as Kubeflow pipeline components.

Components:
    - data_ingestion: Load and process NASA Turbofan dataset
    - feature_engineering: Apply feature transformations
    - train_rf: Train Random Forest model
    - train_xgboost: Train XGBoost model
    - train_lstm: Train LSTM model
    - evaluate_models: Compare and select best model

Each component is designed to:
    1. Run independently as a standalone script
    2. Accept inputs via argparse CLI arguments
    3. Output artifacts to specified paths
    4. Log to MLflow when available
    5. Handle errors gracefully with proper logging
"""

__version__ = "1.0.0"
