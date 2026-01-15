"""
Unit tests for training orchestrator and model evaluation.

Run tests:
    pytest tests/test_training.py -v
    pytest tests/test_training.py -v -k "test_orchestrator"
    pytest tests/test_training.py -v -m "not slow"  # Skip slow tests
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.training.evaluate import ModelEvaluator, EvaluationError
from src.config import COLUMN_NAMES

# Conditional import of train module - may fail if model dependencies missing
try:
    from src.training.train import (
        TrainingOrchestrator,
        parse_args,
        main,
        TrainingError,
        DataPreparationError,
        ModelTrainingError,
        RF_AVAILABLE,
        XGB_AVAILABLE,
        LSTM_AVAILABLE,
    )
    TRAIN_MODULE_AVAILABLE = True
except Exception as e:
    # Catch all exceptions (ImportError, XGBoostError, etc.)
    TrainingOrchestrator = None
    parse_args = None
    main = None
    TrainingError = Exception
    DataPreparationError = Exception
    ModelTrainingError = Exception
    RF_AVAILABLE = False
    XGB_AVAILABLE = False
    LSTM_AVAILABLE = False
    TRAIN_MODULE_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_train_data() -> pd.DataFrame:
    """
    Create small synthetic dataset for fast testing.

    Creates 5 units with 80 cycles each = 400 samples.
    Must have enough cycles (>50) to create LSTM sequences.
    Includes all required columns: unit_id, time_cycle, sensors, op_settings.
    """
    np.random.seed(42)
    n_units = 5
    cycles_per_unit = 80  # Must be > sequence_length (50) for LSTM
    n_samples = n_units * cycles_per_unit

    data = {
        "unit_id": np.repeat(range(1, n_units + 1), cycles_per_unit),
        "time_cycle": np.tile(range(1, cycles_per_unit + 1), n_units),
    }

    # Op settings
    for i in range(1, 4):
        data[f"op_setting_{i}"] = np.random.randn(n_samples) * 0.1

    # All 21 sensors
    for i in range(1, 22):
        data[f"sensor_{i}"] = np.random.randn(n_samples) * 10 + 500

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_2d_data():
    """Create 2D feature array for RF/XGBoost testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 24

    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 100 + 25  # RUL values: 25-125

    return X, y


@pytest.fixture
def synthetic_3d_data():
    """Create 3D sequence array for LSTM testing."""
    np.random.seed(42)
    n_samples = 50
    sequence_length = 50  # Must match model's expected sequence length
    n_features = 24

    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.rand(n_samples) * 100 + 25

    return X, y


@pytest.fixture
def mock_mlflow():
    """Mock MLflow to avoid server dependency."""
    with patch('src.training.train.mlflow') as mock:
        # Setup context manager returns
        mock_run = MagicMock()
        mock_run.info.run_id = "test-parent-run-id"

        mock_child_run = MagicMock()
        mock_child_run.info.run_id = "test-child-run-id"

        mock.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock.start_run.return_value.__exit__ = Mock(return_value=False)

        mock.active_run.return_value = mock_run

        yield mock


@pytest.fixture
def mock_mlflow_evaluate():
    """Mock MLflow for evaluate module."""
    with patch('src.training.evaluate.mlflow') as mock:
        mock.log_artifact = Mock()
        mock.log_figure = Mock()
        mock.active_run.return_value = MagicMock()
        yield mock


@pytest.fixture
def temp_data_dir(synthetic_train_data):
    """Create temp directory with synthetic data files in NASA format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Ensure columns are in correct order
        ordered_cols = COLUMN_NAMES
        df = synthetic_train_data.copy()

        # Ensure all columns exist (add missing ones with zeros)
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df[ordered_cols]

        # Save as space-delimited text (NASA format) without header
        train_str = df.to_csv(sep=" ", index=False, header=False)
        (tmpdir_path / "train_FD001.txt").write_text(train_str)
        (tmpdir_path / "test_FD001.txt").write_text(train_str)

        # RUL file - one value per unit
        n_units = df["unit_id"].nunique()
        np.savetxt(
            tmpdir_path / "RUL_FD001.txt",
            np.random.randint(20, 100, size=n_units),
            fmt="%d"
        )

        yield tmpdir_path


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


# =============================================================================
# TestParseArgs
# =============================================================================

@pytest.mark.skipif(not TRAIN_MODULE_AVAILABLE, reason="Training module not available")
class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_arguments(self):
        """Test default argument values."""
        with patch.object(sys, 'argv', ['train.py']):
            args = parse_args()
            assert args.experiment_name == "predictive-maintenance"
            assert args.model_type == "all"
            assert args.dataset == "FD001"
            assert args.no_mlflow is False
            assert args.data_path is None

    def test_custom_experiment_name(self):
        """Test custom experiment name."""
        with patch.object(sys, 'argv', ['train.py', '--experiment-name', 'my-exp']):
            args = parse_args()
            assert args.experiment_name == "my-exp"

    def test_model_type_choices(self):
        """Test valid model type choices."""
        for model_type in ["all", "rf", "xgb", "lstm"]:
            with patch.object(sys, 'argv', ['train.py', '--model-type', model_type]):
                args = parse_args()
                assert args.model_type == model_type

    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with patch.object(sys, 'argv', ['train.py', '--model-type', 'invalid']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_dataset_choices(self):
        """Test valid dataset choices."""
        for dataset in ["FD001", "FD002", "FD003", "FD004"]:
            with patch.object(sys, 'argv', ['train.py', '--dataset', dataset]):
                args = parse_args()
                assert args.dataset == dataset

    def test_invalid_dataset(self):
        """Test invalid dataset raises error."""
        with patch.object(sys, 'argv', ['train.py', '--dataset', 'FD005']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_no_mlflow_flag(self):
        """Test no-mlflow flag."""
        with patch.object(sys, 'argv', ['train.py', '--no-mlflow']):
            args = parse_args()
            assert args.no_mlflow is True

    def test_verbose_flag(self):
        """Test verbose flag counting."""
        with patch.object(sys, 'argv', ['train.py', '-v']):
            args = parse_args()
            assert args.verbose == 1

        with patch.object(sys, 'argv', ['train.py', '-vv']):
            args = parse_args()
            assert args.verbose == 2

    def test_custom_data_path(self):
        """Test custom data path."""
        with patch.object(sys, 'argv', ['train.py', '--data-path', '/custom/path']):
            args = parse_args()
            assert args.data_path == "/custom/path"


# =============================================================================
# TestModelEvaluator
# =============================================================================

class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.results == {}
        assert evaluator.figures == []

    def test_compute_metrics(self):
        """Test metric computation."""
        evaluator = ModelEvaluator()
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])

        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics for perfect predictions."""
        evaluator = ModelEvaluator()
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = y_true.copy()

        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert metrics["rmse"] == pytest.approx(0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0, abs=1e-10)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-10)
        assert metrics["mape"] == pytest.approx(0, abs=1e-10)

    def test_compute_metrics_mape_zero_handling(self):
        """Test MAPE handles zero values gracefully."""
        evaluator = ModelEvaluator()
        y_true = np.array([0.0, 10.0, 20.0])  # Contains zero
        y_pred = np.array([1.0, 11.0, 21.0])

        # Should not raise, should handle gracefully
        metrics = evaluator.compute_metrics(y_true, y_pred)
        assert "mape" in metrics
        # MAPE should be computed only on non-zero values
        assert metrics["mape"] != float('inf')

    def test_compute_metrics_all_zeros(self):
        """Test MAPE returns inf when all true values are zero."""
        evaluator = ModelEvaluator()
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = evaluator.compute_metrics(y_true, y_pred)
        assert metrics["mape"] == float('inf')

    def test_compute_metrics_shape_mismatch(self):
        """Test error on shape mismatch."""
        evaluator = ModelEvaluator()
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18])

        with pytest.raises(EvaluationError, match="Shape mismatch"):
            evaluator.compute_metrics(y_true, y_pred)

    def test_compute_metrics_empty_arrays(self):
        """Test error on empty arrays."""
        evaluator = ModelEvaluator()

        with pytest.raises(EvaluationError, match="empty arrays"):
            evaluator.compute_metrics(np.array([]), np.array([]))

    def test_evaluate_model(self, synthetic_2d_data):
        """Test model evaluation."""
        X, y = synthetic_2d_data

        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = y + np.random.randn(len(y)) * 5

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(mock_model, X, y, model_name="test_model")

        assert "rmse" in metrics
        mock_model.predict.assert_called_once()
        assert "test_model" in evaluator.results

    def test_evaluate_model_prediction_failure(self, synthetic_2d_data):
        """Test evaluation handles prediction failure."""
        X, y = synthetic_2d_data

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Prediction failed")

        evaluator = ModelEvaluator()
        with pytest.raises(EvaluationError, match="Model prediction failed"):
            evaluator.evaluate_model(mock_model, X, y)

    def test_plot_predictions(self):
        """Test prediction plot generation."""
        evaluator = ModelEvaluator()
        y_true = np.random.randn(100) * 50 + 75
        y_pred = y_true + np.random.randn(100) * 10

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "predictions.png"
            fig = evaluator.plot_predictions(y_true, y_pred, save_path=save_path)

            assert save_path.exists()
            assert save_path in evaluator.figures

    def test_plot_residuals(self):
        """Test residual plot generation."""
        evaluator = ModelEvaluator()
        y_true = np.random.randn(100) * 50 + 75
        y_pred = y_true + np.random.randn(100) * 10

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "residuals.png"
            fig = evaluator.plot_residuals(y_true, y_pred, save_path=save_path)

            assert save_path.exists()
            assert save_path in evaluator.figures

    def test_compare_models(self, synthetic_2d_data):
        """Test model comparison."""
        X, y = synthetic_2d_data

        # Create mock models with different predictions
        mock_rf = MagicMock()
        mock_rf.predict.return_value = y + np.random.randn(len(y)) * 5

        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = y + np.random.randn(len(y)) * 3

        evaluator = ModelEvaluator()
        results = evaluator.compare_models(
            {"rf": mock_rf, "xgb": mock_xgb},
            X, y,
        )

        assert "rf" in results
        assert "xgb" in results
        assert "rmse" in results["rf"]
        assert "rmse" in results["xgb"]

    def test_compare_models_with_failure(self, synthetic_2d_data):
        """Test model comparison handles failures gracefully."""
        X, y = synthetic_2d_data

        mock_good = MagicMock()
        mock_good.predict.return_value = y + np.random.randn(len(y))

        mock_bad = MagicMock()
        mock_bad.predict.side_effect = RuntimeError("Model failed")

        evaluator = ModelEvaluator()
        results = evaluator.compare_models(
            {"good": mock_good, "bad": mock_bad},
            X, y,
        )

        assert "good" in results
        assert "bad" in results
        assert "error" in results["bad"]
        assert results["bad"]["rmse"] == float('inf')

    def test_generate_evaluation_report(self):
        """Test JSON report generation."""
        evaluator = ModelEvaluator()
        metrics_dict = {
            "rf": {"rmse": 15.0, "mae": 12.0, "r2": 0.85, "mape": 10.5},
            "xgb": {"rmse": 12.0, "mae": 9.0, "r2": 0.90, "mape": 8.2},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "report.json"
            result_path = evaluator.generate_evaluation_report(
                metrics_dict, save_path, dataset="FD001"
            )

            assert result_path.exists()

            # Verify JSON structure
            with open(result_path) as f:
                report = json.load(f)

            assert "models" in report
            assert "best_model" in report
            assert "timestamp" in report
            assert report["dataset"] == "FD001"
            assert report["best_model"]["name"] == "xgb"
            assert report["best_model"]["value"] == 12.0

    def test_plot_comparison_bar_chart(self):
        """Test comparison bar chart generation."""
        evaluator = ModelEvaluator()
        metrics_dict = {
            "rf": {"rmse": 15.0, "mae": 12.0},
            "xgb": {"rmse": 12.0, "mae": 9.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "comparison.png"
            fig = evaluator.plot_comparison_bar_chart(
                metrics_dict, save_path=save_path
            )

            assert save_path.exists()

    def test_clear_results(self):
        """Test clearing results."""
        evaluator = ModelEvaluator()
        evaluator.results = {"test": {"rmse": 10.0}}
        evaluator.figures = [Path("/tmp/test.png")]

        evaluator.clear_results()

        assert evaluator.results == {}
        assert evaluator.figures == []


# =============================================================================
# TestTrainingOrchestrator
# =============================================================================

@pytest.mark.skipif(not TRAIN_MODULE_AVAILABLE, reason="Training module not available")
class TestTrainingOrchestrator:
    """Tests for TrainingOrchestrator class."""

    def test_init_default_params(self):
        """Test orchestrator initialization with defaults."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)
        assert orchestrator.experiment_name == "predictive-maintenance"
        assert orchestrator.dataset == "FD001"
        assert orchestrator.trained_models == {}
        assert orchestrator.results == {}

    def test_init_custom_params(self, temp_data_dir):
        """Test orchestrator initialization with custom params."""
        orchestrator = TrainingOrchestrator(
            experiment_name="custom-exp",
            dataset="FD001",
            data_path=temp_data_dir,
            use_mlflow=False,
        )
        assert orchestrator.experiment_name == "custom-exp"
        assert orchestrator.data_path == temp_data_dir

    def test_init_invalid_dataset(self):
        """Test initialization with invalid dataset raises error."""
        with pytest.raises(ValueError, match="Dataset must be one of"):
            TrainingOrchestrator(dataset="FD005", use_mlflow=False)

    def test_load_and_preprocess_data(self, temp_data_dir):
        """Test data loading and preprocessing."""
        orchestrator = TrainingOrchestrator(
            data_path=temp_data_dir,
            use_mlflow=False,
        )
        X_train, X_val, y_train, y_val = orchestrator.load_and_preprocess_data()

        assert X_train.ndim == 2
        assert y_train.ndim == 1
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_train) > len(X_val)  # Train should be larger

    def test_prepare_lstm_data(self, temp_data_dir):
        """Test LSTM data preparation produces correct shapes."""
        orchestrator = TrainingOrchestrator(
            data_path=temp_data_dir,
            use_mlflow=False,
        )
        # Must load data first
        orchestrator.load_and_preprocess_data()
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = orchestrator.prepare_lstm_data()

        # Check 3D shape
        assert X_train_seq.ndim == 3
        assert X_val_seq.ndim == 3
        # Check sequence length is second dimension
        assert X_train_seq.shape[1] > 0
        # Check target shapes
        assert y_train_seq.ndim == 1
        assert y_val_seq.ndim == 1

    def test_prepare_lstm_data_without_load(self):
        """Test LSTM data prep fails without loading data first."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)

        with pytest.raises(DataPreparationError, match="Data not loaded"):
            orchestrator.prepare_lstm_data()

    @pytest.mark.skipif(not RF_AVAILABLE, reason="RandomForest not available")
    def test_train_random_forest(self, synthetic_2d_data):
        """Test Random Forest training."""
        X, y = synthetic_2d_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        orchestrator = TrainingOrchestrator(use_mlflow=False)
        orchestrator._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        model, metrics = orchestrator.train_random_forest(
            X_train, y_train, X_val, y_val,
            log_to_mlflow=False,
        )

        assert model.is_trained
        assert "val_rmse" in metrics
        assert "val_mae" in metrics
        assert "val_r2" in metrics
        assert "rf" in orchestrator.trained_models

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not available")
    def test_train_xgboost(self, synthetic_2d_data):
        """Test XGBoost training."""
        X, y = synthetic_2d_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        orchestrator = TrainingOrchestrator(use_mlflow=False)
        orchestrator._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        model, metrics = orchestrator.train_xgboost(
            X_train, y_train, X_val, y_val,
            log_to_mlflow=False,
        )

        assert model.is_trained
        assert "val_rmse" in metrics
        assert "xgb" in orchestrator.trained_models

    @pytest.mark.skipif(not LSTM_AVAILABLE, reason="LSTM/TensorFlow not available")
    def test_train_lstm(self, synthetic_3d_data):
        """Test LSTM training with 3D input."""
        X, y = synthetic_3d_data
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]

        orchestrator = TrainingOrchestrator(use_mlflow=False)

        model, metrics = orchestrator.train_lstm(
            X_train, y_train, X_val, y_val,
            log_to_mlflow=False,
        )

        assert model.is_trained
        assert "val_rmse" in metrics
        assert "lstm" in orchestrator.trained_models

    def test_select_best_model(self):
        """Test best model selection."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)

        # Manually set results
        orchestrator.results = {
            "rf": {"model": MagicMock(is_trained=True), "metrics": {"val_rmse": 15.0}},
            "xgb": {"model": MagicMock(is_trained=True), "metrics": {"val_rmse": 12.0}},
        }

        best_type, best_model, best_metrics = orchestrator.select_best_model(
            metric="val_rmse",
            lower_is_better=True,
        )

        assert best_type == "xgb"
        assert best_metrics["val_rmse"] == 12.0

    def test_select_best_model_higher_is_better(self):
        """Test best model selection with higher is better."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)

        orchestrator.results = {
            "rf": {"model": MagicMock(), "metrics": {"val_r2": 0.85}},
            "xgb": {"model": MagicMock(), "metrics": {"val_r2": 0.90}},
        }

        best_type, _, best_metrics = orchestrator.select_best_model(
            metric="val_r2",
            lower_is_better=False,
        )

        assert best_type == "xgb"
        assert best_metrics["val_r2"] == 0.90

    def test_select_best_model_no_results(self):
        """Test best model selection with no results raises error."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)

        with pytest.raises(ValueError, match="No models have been trained"):
            orchestrator.select_best_model()

    def test_select_best_model_skips_failed(self):
        """Test best model selection skips failed models."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)

        orchestrator.results = {
            "rf": {"model": None, "metrics": {"error": "Failed"}},
            "xgb": {"model": MagicMock(), "metrics": {"val_rmse": 12.0}},
        }

        best_type, _, _ = orchestrator.select_best_model()
        assert best_type == "xgb"

    def test_train_all_models_invalid_type(self):
        """Test training with invalid model type raises error."""
        orchestrator = TrainingOrchestrator(use_mlflow=False)

        with pytest.raises(ValueError, match="Invalid model type"):
            orchestrator.train_all_models(model_types=["invalid"])


# =============================================================================
# TestIntegration
# =============================================================================

@pytest.mark.skipif(not TRAIN_MODULE_AVAILABLE, reason="Training module not available")
class TestIntegration:
    """Integration tests for full training pipeline."""

    @pytest.mark.slow
    @pytest.mark.skipif(not RF_AVAILABLE, reason="RandomForest not available")
    def test_full_training_pipeline_rf(self, temp_data_dir):
        """Test complete training pipeline for RF only (for speed)."""
        orchestrator = TrainingOrchestrator(
            experiment_name="test-experiment",
            dataset="FD001",
            data_path=temp_data_dir,
            use_mlflow=False,
        )

        results = orchestrator.train_all_models(model_types=["rf"])

        assert "rf" in results
        assert results["rf"]["model"] is not None
        assert results["rf"]["model"].is_trained

        # Test best model selection
        best_type, best_model, best_metrics = orchestrator.select_best_model()
        assert best_type == "rf"

    @pytest.mark.slow
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not available")
    def test_full_training_pipeline_xgb(self, temp_data_dir):
        """Test complete training pipeline for XGBoost."""
        orchestrator = TrainingOrchestrator(
            experiment_name="test-experiment",
            dataset="FD001",
            data_path=temp_data_dir,
            use_mlflow=False,
        )

        results = orchestrator.train_all_models(model_types=["xgb"])

        assert "xgb" in results
        assert results["xgb"]["model"].is_trained

    @pytest.mark.slow
    @pytest.mark.skipif(not RF_AVAILABLE, reason="RandomForest not available")
    def test_save_best_model(self, temp_data_dir):
        """Test saving best model."""
        orchestrator = TrainingOrchestrator(
            experiment_name="test-experiment",
            dataset="FD001",
            data_path=temp_data_dir,
            use_mlflow=False,
        )

        orchestrator.train_all_models(model_types=["rf"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = orchestrator.save_best_model(save_dir=Path(tmpdir))
            assert save_path.exists()


# =============================================================================
# TestErrorHandling
# =============================================================================

@pytest.mark.skipif(not TRAIN_MODULE_AVAILABLE, reason="Training module not available")
class TestErrorHandling:
    """Tests for error handling."""

    def test_data_load_error_handling(self):
        """Test handling of data load errors."""
        orchestrator = TrainingOrchestrator(
            data_path=Path("/nonexistent/path"),
            use_mlflow=False,
        )

        with pytest.raises(DataPreparationError):
            orchestrator.load_and_preprocess_data()

    @pytest.mark.skipif(not RF_AVAILABLE, reason="RandomForest not available")
    def test_model_training_error_handling(self, synthetic_2d_data):
        """Test handling of model training errors."""
        X, y = synthetic_2d_data

        with patch('src.training.train.RandomForestModel') as mock_rf_class:
            mock_rf = MagicMock()
            mock_rf.train.side_effect = RuntimeError("Training failed")
            mock_rf_class.return_value = mock_rf

            orchestrator = TrainingOrchestrator(use_mlflow=False)
            orchestrator._feature_names = [f"f{i}" for i in range(X.shape[1])]

            with pytest.raises(ModelTrainingError, match="Random Forest training failed"):
                orchestrator.train_random_forest(X, y, X, y, log_to_mlflow=False)

    def test_mlflow_unavailable_graceful(self, temp_data_dir):
        """Test graceful handling when MLflow is unavailable."""
        with patch('src.training.train.MLFLOW_AVAILABLE', False):
            orchestrator = TrainingOrchestrator(
                data_path=temp_data_dir,
                use_mlflow=True,  # Request MLflow but it's unavailable
            )

            # Should not use MLflow even though requested
            assert orchestrator.use_mlflow is False


# =============================================================================
# TestMain
# =============================================================================

@pytest.mark.skipif(not TRAIN_MODULE_AVAILABLE, reason="Training module not available")
class TestMain:
    """Tests for main function."""

    @pytest.mark.slow
    @pytest.mark.skipif(not RF_AVAILABLE, reason="RandomForest not available")
    def test_main_function_rf(self, temp_data_dir):
        """Test main() entry point with RF model."""
        with patch.object(sys, 'argv', [
            'train.py',
            '--experiment-name', 'test',
            '--model-type', 'rf',
            '--dataset', 'FD001',
            '--data-path', str(temp_data_dir),
            '--no-mlflow',
            '-v',
        ]):
            exit_code = main()
            assert exit_code == 0

    def test_main_function_data_error(self):
        """Test main() handles data errors."""
        with patch.object(sys, 'argv', [
            'train.py',
            '--data-path', '/nonexistent/path',
            '--no-mlflow',
        ]):
            exit_code = main()
            assert exit_code == 1
