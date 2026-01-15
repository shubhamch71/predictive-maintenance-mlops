#!/usr/bin/env python3
"""
Compile Kubeflow Pipeline to YAML

This script compiles the predictive maintenance pipeline to a YAML file
that can be uploaded to the Kubeflow Pipelines UI or submitted via CLI.

Usage:
    # Basic compilation
    python compile_pipeline.py

    # With custom output path
    python compile_pipeline.py --output my_pipeline.yaml

    # Compile and submit to Kubeflow
    python compile_pipeline.py --submit --endpoint http://kubeflow.example.com

Example workflow:
    1. Compile: python compile_pipeline.py
    2. Upload: Go to Kubeflow UI -> Pipelines -> Upload Pipeline -> Select pipeline.yaml
    3. Run: Create experiment -> Create run -> Set parameters -> Start

Requirements:
    pip install kfp>=2.0.0
"""

import argparse
import os
import sys
from pathlib import Path


def compile_pipeline(output_path: str = "pipeline.yaml") -> str:
    """
    Compile the predictive maintenance pipeline to YAML.

    Args:
        output_path: Path for the output YAML file

    Returns:
        Path to the compiled YAML file
    """
    from kfp import compiler

    from pipeline import predictive_maintenance_pipeline

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compiling pipeline to {output_path}...")
    compiler.Compiler().compile(
        pipeline_func=predictive_maintenance_pipeline,
        package_path=str(output_path),
    )
    print(f"Pipeline compiled successfully!")
    print(f"Output: {output_path.absolute()}")

    return str(output_path)


def submit_pipeline(
    pipeline_path: str,
    endpoint: str,
    experiment_name: str = "predictive-maintenance",
    run_name: str = None,
    **params,
) -> str:
    """
    Submit compiled pipeline to Kubeflow Pipelines.

    Args:
        pipeline_path: Path to compiled pipeline YAML
        endpoint: Kubeflow Pipelines endpoint URL
        experiment_name: Name of the experiment
        run_name: Name for this run (auto-generated if not provided)
        **params: Pipeline parameters to override

    Returns:
        Run ID
    """
    import kfp

    client = kfp.Client(host=endpoint)

    # Get or create experiment
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
    except Exception:
        experiment = client.create_experiment(name=experiment_name)

    # Generate run name if not provided
    if run_name is None:
        from datetime import datetime
        run_name = f"pm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Submit run
    print(f"Submitting pipeline to {endpoint}...")
    print(f"Experiment: {experiment_name}")
    print(f"Run name: {run_name}")

    run = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_path,
        experiment_name=experiment_name,
        run_name=run_name,
        arguments=params or None,
    )

    print(f"Run submitted successfully!")
    print(f"Run ID: {run.run_id}")
    print(f"View at: {endpoint}/#/runs/details/{run.run_id}")

    return run.run_id


def create_run_config(output_path: str = "run_config.yaml") -> None:
    """
    Create a sample run configuration file with default parameters.
    """
    import yaml

    config = {
        "pipeline": {
            "name": "predictive-maintenance-training",
            "description": "Training pipeline for predictive maintenance models",
        },
        "parameters": {
            # Data parameters
            "data_path": "/data/raw",
            "dataset": "FD001",
            "max_rul": 125,

            # Feature engineering
            "rolling_windows": "5,10,20",
            "lag_periods": "1,5,10",

            # MLflow
            "mlflow_tracking_uri": "http://mlflow:5000",
            "experiment_name": "predictive-maintenance",

            # Random Forest
            "rf_n_estimators": 100,
            "rf_max_depth": 0,  # 0 means unlimited

            # XGBoost
            "xgb_n_estimators": 100,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.1,

            # LSTM
            "lstm_sequence_length": 50,
            "lstm_epochs": 100,
            "lstm_batch_size": 32,
            "lstm_learning_rate": 0.001,

            # Evaluation
            "selection_metric": "rmse",

            # General
            "random_state": 42,
        },
        "resources": {
            "data_ingestion": {
                "cpu": "1",
                "memory": "2Gi",
            },
            "feature_engineering": {
                "cpu": "2",
                "memory": "4Gi",
            },
            "training": {
                "cpu": "4",
                "memory": "8Gi",
            },
            "lstm_training": {
                "cpu": "4",
                "memory": "16Gi",
                "gpu": "1",
            },
            "evaluation": {
                "cpu": "2",
                "memory": "4Gi",
            },
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Run configuration saved to {output_path}")


def validate_pipeline() -> bool:
    """
    Validate the pipeline by checking component imports and structure.
    """
    print("Validating pipeline...")

    try:
        from pipeline import predictive_maintenance_pipeline

        # Check that pipeline can be compiled
        from kfp import compiler
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=True) as f:
            compiler.Compiler().compile(
                pipeline_func=predictive_maintenance_pipeline,
                package_path=f.name,
            )
            # Read and check basic structure
            with open(f.name, "r") as yaml_file:
                content = yaml_file.read()
                if "pipelineSpec" not in content and "deploymentSpec" not in content:
                    print("Warning: Pipeline YAML may have unexpected structure")

        print("Pipeline validation successful!")
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure kfp is installed: pip install kfp>=2.0.0")
        return False
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compile and optionally submit Kubeflow Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compile pipeline to YAML
    python compile_pipeline.py

    # Compile with custom output path
    python compile_pipeline.py --output my_pipeline.yaml

    # Validate pipeline without compiling
    python compile_pipeline.py --validate

    # Compile and submit to Kubeflow
    python compile_pipeline.py --submit --endpoint http://kubeflow.example.com

    # Submit with custom parameters
    python compile_pipeline.py --submit --endpoint http://kubeflow.example.com \\
        --param dataset=FD002 --param rf_n_estimators=200

    # Create run configuration file
    python compile_pipeline.py --create-config
        """,
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="pipeline.yaml",
        help="Output path for compiled pipeline YAML",
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate pipeline without compiling",
    )
    parser.add_argument(
        "--submit", "-s",
        action="store_true",
        help="Submit pipeline to Kubeflow after compiling",
    )
    parser.add_argument(
        "--endpoint", "-e",
        type=str,
        default=os.environ.get("KUBEFLOW_ENDPOINT", "http://localhost:8080"),
        help="Kubeflow Pipelines endpoint URL",
    )
    parser.add_argument(
        "--experiment", "-x",
        type=str,
        default="predictive-maintenance",
        help="Experiment name for the run",
    )
    parser.add_argument(
        "--run-name", "-r",
        type=str,
        default=None,
        help="Name for this pipeline run",
    )
    parser.add_argument(
        "--param", "-p",
        action="append",
        default=[],
        help="Pipeline parameter override (format: key=value)",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a sample run configuration file",
    )

    args = parser.parse_args()

    # Create config and exit
    if args.create_config:
        create_run_config()
        return 0

    # Validate and exit
    if args.validate:
        success = validate_pipeline()
        return 0 if success else 1

    # Compile pipeline
    try:
        pipeline_path = compile_pipeline(args.output)
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure required packages are installed:")
        print("  pip install kfp>=2.0.0")
        return 1
    except Exception as e:
        print(f"Compilation failed: {e}")
        return 1

    # Submit if requested
    if args.submit:
        # Parse parameters
        params = {}
        for param in args.param:
            if "=" in param:
                key, value = param.split("=", 1)
                # Try to convert to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                params[key] = value

        try:
            run_id = submit_pipeline(
                pipeline_path=pipeline_path,
                endpoint=args.endpoint,
                experiment_name=args.experiment,
                run_name=args.run_name,
                **params,
            )
        except Exception as e:
            print(f"Submission failed: {e}")
            return 1

    print("\nNext steps:")
    print("  1. Upload to Kubeflow UI: Pipelines -> Upload Pipeline")
    print("  2. Or submit via CLI: kfp run create -f pipeline.yaml")
    print("  3. Or use this script: python compile_pipeline.py --submit --endpoint <URL>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
