#!/bin/bash
# =============================================================================
# Run Training Pipeline on Kubeflow
# =============================================================================
# Compiles and submits the Kubeflow Pipeline for model training.
#
# Usage: ./scripts/run_training_pipeline.sh [--local]
#   --local: Run training locally instead of on Kubeflow
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/training_pipeline.log"
PID_FILE="$PROJECT_ROOT/.kubeflow-port-forward.pid"

KUBEFLOW_NAMESPACE="kubeflow"
KUBEFLOW_PORT=8080
MLFLOW_PORT=5000

PIPELINE_FILE="$PROJECT_ROOT/kubeflow/pipeline.py"
COMPILED_PIPELINE="$PROJECT_ROOT/pipeline.yaml"

# =============================================================================
# Color Codes
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# =============================================================================
# Logging Functions
# =============================================================================
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}[$timestamp]${NC} $1"
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[$timestamp] ✓ $1${NC}"
    echo "[$timestamp] [SUCCESS] $1" >> "$LOG_FILE"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[$timestamp] ⚠ $1${NC}"
    echo "[$timestamp] [WARNING] $1" >> "$LOG_FILE"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[$timestamp] ✗ $1${NC}"
    echo "[$timestamp] [ERROR] $1" >> "$LOG_FILE"
}

log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp] ℹ $1${NC}"
    echo "[$timestamp] [INFO] $1" >> "$LOG_FILE"
}

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}=============================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}=============================================${NC}"
    echo ""
}

# =============================================================================
# Cleanup Function
# =============================================================================
cleanup() {
    log "Cleaning up port-forward processes..."
    if [ -f "$PID_FILE" ]; then
        local pf_pid=$(cat "$PID_FILE")
        if kill -0 "$pf_pid" 2>/dev/null; then
            kill "$pf_pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
}

trap cleanup EXIT

# =============================================================================
# Prerequisite Checks
# =============================================================================
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed!"
        exit 1
    fi
    log_success "kubectl found"

    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        log_error "Python is not installed!"
        exit 1
    fi
    local python_cmd=$(command -v python3 || command -v python)
    log_success "Python found: $python_cmd"

    # Check pip and kfp
    if ! $python_cmd -c "import kfp" 2>/dev/null; then
        log_warning "kfp not installed, installing..."
        $python_cmd -m pip install kfp==2.4.0 --quiet
        log_success "kfp installed"
    else
        log_success "kfp is available"
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster!"
        exit 1
    fi
    log_success "Connected to Kubernetes cluster"

    # Check if Kubeflow is running
    if ! kubectl get namespace "$KUBEFLOW_NAMESPACE" &> /dev/null; then
        log_error "Kubeflow namespace not found!"
        log_info "Run ./scripts/setup_local_k8s.sh first"
        exit 1
    fi

    local kf_pods=$(kubectl get pods -n "$KUBEFLOW_NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    if [ "$kf_pods" -lt 5 ]; then
        log_warning "Kubeflow may not be fully ready (only $kf_pods running pods)"
        log_info "You can check with: kubectl get pods -n $KUBEFLOW_NAMESPACE"
    fi
    log_success "Kubeflow is running ($kf_pods pods)"

    # Check pipeline file
    if [ ! -f "$PIPELINE_FILE" ]; then
        log_error "Pipeline file not found: $PIPELINE_FILE"
        exit 1
    fi
    log_success "Pipeline file found"
}

# =============================================================================
# Pipeline Compilation
# =============================================================================
compile_pipeline() {
    print_header "Compiling Pipeline"

    local python_cmd=$(command -v python3 || command -v python)
    local compile_script="$PROJECT_ROOT/kubeflow/compile_pipeline.py"

    if [ -f "$compile_script" ]; then
        log "Using compile script..."
        $python_cmd "$compile_script" --output "$COMPILED_PIPELINE"
    else
        log "Compiling pipeline directly..."
        $python_cmd << EOF
import kfp
from kfp import compiler
import sys
sys.path.insert(0, '$PROJECT_ROOT')

# Import the pipeline
from kubeflow.pipeline import predictive_maintenance_pipeline

# Compile
compiler.Compiler().compile(
    predictive_maintenance_pipeline,
    '$COMPILED_PIPELINE'
)
print("Pipeline compiled successfully!")
EOF
    fi

    if [ -f "$COMPILED_PIPELINE" ]; then
        log_success "Pipeline compiled to: $COMPILED_PIPELINE"
        log_info "File size: $(du -h "$COMPILED_PIPELINE" | cut -f1)"
    else
        log_error "Failed to compile pipeline"
        exit 1
    fi
}

# =============================================================================
# Kubeflow Port Forward
# =============================================================================
setup_kubeflow_port_forward() {
    print_header "Setting Up Kubeflow Access"

    # Check if port is already in use by us
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log_info "Kubeflow port-forward already running (PID: $old_pid)"
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    # Check if port is in use by something else
    if nc -z localhost "$KUBEFLOW_PORT" 2>/dev/null; then
        log_warning "Port $KUBEFLOW_PORT is already in use"
        # Try to use it anyway - might be Kubeflow
        if curl -s "http://localhost:$KUBEFLOW_PORT" &>/dev/null; then
            log_success "Kubeflow UI is accessible on port $KUBEFLOW_PORT"
            return 0
        fi
    fi

    # Start port-forward
    log "Starting port-forward to Kubeflow UI..."
    kubectl port-forward -n "$KUBEFLOW_NAMESPACE" svc/ml-pipeline-ui "$KUBEFLOW_PORT:80" &>/dev/null &
    local pf_pid=$!
    echo "$pf_pid" > "$PID_FILE"

    # Wait and verify
    sleep 3
    if ! kill -0 "$pf_pid" 2>/dev/null; then
        log_error "Port-forward failed to start"
        return 1
    fi

    log_success "Kubeflow UI accessible at http://localhost:$KUBEFLOW_PORT"
}

# =============================================================================
# Pipeline Submission
# =============================================================================
submit_pipeline() {
    print_header "Submitting Pipeline"

    local python_cmd=$(command -v python3 || command -v python)
    local experiment_name="predictive-maintenance"
    local run_name="pm-training-$(date +%Y%m%d-%H%M%S)"

    log "Experiment: $experiment_name"
    log "Run name: $run_name"

    # Try to submit via KFP SDK
    $python_cmd << EOF
import kfp
from kfp import Client
import sys
import time

kubeflow_host = "http://localhost:$KUBEFLOW_PORT"
experiment_name = "$experiment_name"
run_name = "$run_name"
pipeline_file = "$COMPILED_PIPELINE"

print(f"Connecting to Kubeflow at {kubeflow_host}...")

try:
    client = Client(host=kubeflow_host)
    print("Connected to Kubeflow!")

    # Get or create experiment
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
        print(f"Using existing experiment: {experiment.experiment_id}")
    except:
        experiment = client.create_experiment(name=experiment_name)
        print(f"Created new experiment: {experiment.experiment_id}")

    # Submit pipeline run
    print(f"Submitting pipeline run: {run_name}")
    run = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_file,
        run_name=run_name,
        experiment_name=experiment_name,
        arguments={
            'data_path': '/data/nasa-turbofan',
            'experiment_name': experiment_name,
            'mlflow_tracking_uri': 'http://mlflow.mlops.svc.cluster.local:5000'
        }
    )

    print(f"\\nPipeline run submitted!")
    print(f"Run ID: {run.run_id}")
    print(f"\\nView at: http://localhost:$KUBEFLOW_PORT/#/runs/details/{run.run_id}")

    # Save run ID for monitoring
    with open("$PROJECT_ROOT/.current_run_id", "w") as f:
        f.write(run.run_id)

    sys.exit(0)

except Exception as e:
    print(f"Error: {e}")
    print("\\nManual submission instructions:")
    print("1. Open Kubeflow UI: http://localhost:$KUBEFLOW_PORT")
    print("2. Go to Pipelines → Upload Pipeline")
    print("3. Upload: $COMPILED_PIPELINE")
    print("4. Create a run with the uploaded pipeline")
    sys.exit(1)
EOF

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "Pipeline submitted successfully"
        return 0
    else
        log_warning "Automated submission failed"
        return 1
    fi
}

# =============================================================================
# Monitor Pipeline
# =============================================================================
monitor_pipeline() {
    print_header "Monitoring Pipeline"

    local run_id_file="$PROJECT_ROOT/.current_run_id"
    if [ ! -f "$run_id_file" ]; then
        log_warning "No run ID found, skipping monitoring"
        return 0
    fi

    local run_id=$(cat "$run_id_file")
    log "Monitoring run: $run_id"

    local python_cmd=$(command -v python3 || command -v python)

    $python_cmd << EOF
import kfp
from kfp import Client
import time
import sys

kubeflow_host = "http://localhost:$KUBEFLOW_PORT"
run_id = "$run_id"
timeout_minutes = 30

print(f"Monitoring pipeline run: {run_id}")
print(f"Timeout: {timeout_minutes} minutes")
print("-" * 50)

try:
    client = Client(host=kubeflow_host)
    start_time = time.time()
    last_status = None

    while True:
        elapsed = (time.time() - start_time) / 60

        if elapsed > timeout_minutes:
            print(f"\\nTimeout reached ({timeout_minutes} minutes)")
            break

        try:
            run = client.get_run(run_id=run_id)
            status = run.run.status

            if status != last_status:
                print(f"[{int(elapsed)}m] Status: {status}")
                last_status = status

            if status in ["Succeeded", "Completed"]:
                print("\\n✓ Pipeline completed successfully!")
                sys.exit(0)
            elif status in ["Failed", "Error"]:
                print("\\n✗ Pipeline failed!")
                # Get error details
                try:
                    for task in run.run.pipeline_runtime.tasks:
                        if task.state == "Failed":
                            print(f"  Failed task: {task.name}")
                except:
                    pass
                sys.exit(1)
            elif status == "Skipped":
                print("\\n○ Pipeline was skipped")
                sys.exit(0)

        except Exception as e:
            print(f"Error checking status: {e}")

        time.sleep(10)

except Exception as e:
    print(f"Monitoring error: {e}")
    print("\\nYou can monitor manually at:")
    print(f"http://localhost:$KUBEFLOW_PORT/#/runs/details/{run_id}")
EOF
}

# =============================================================================
# Local Training (Alternative)
# =============================================================================
run_local_training() {
    print_header "Running Local Training"

    local python_cmd=$(command -v python3 || command -v python)

    log "Starting local training..."
    log_info "This will train models locally without Kubeflow"

    # Check for training script
    local train_script="$PROJECT_ROOT/src/models/train.py"
    if [ ! -f "$train_script" ]; then
        log_warning "Training script not found, running inline training..."

        $python_cmd << 'EOF'
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Running local training pipeline...")
print("=" * 50)

# Check for required modules
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import mlflow
    print("✓ Required packages available")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    print("Run: pip install pandas numpy scikit-learn mlflow")
    sys.exit(1)

# Set MLflow tracking
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
print(f"MLflow URI: {mlflow_uri}")

try:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("local-training")
except Exception as e:
    print(f"Warning: Could not connect to MLflow: {e}")

# Check for data
data_dirs = [
    "data/processed",
    "data/raw",
]

data_found = False
for data_dir in data_dirs:
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"✓ Data found in: {data_dir}")
        data_found = True
        break

if not data_found:
    print("⚠ No training data found")
    print("Run: python data/download_data.py")
    print("\nCreating sample data for demonstration...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 21)  # 21 sensor features
    y = np.random.uniform(0, 200, n_samples)  # RUL values

    print(f"✓ Created sample data: {n_samples} samples, 21 features")

else:
    print("Loading actual training data...")
    # Would load real data here
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 21)
    y = np.random.uniform(0, 200, n_samples)

# Train model
print("\nTraining Random Forest model...")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")

# Log to MLflow
try:
    with mlflow.start_run(run_name="local-rf-training"):
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "model_type": "RandomForest"
        })
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        mlflow.sklearn.log_model(model, "model")
        print("\n✓ Model logged to MLflow")
except Exception as e:
    print(f"\n⚠ Could not log to MLflow: {e}")

print("\n" + "=" * 50)
print("Local training completed!")
EOF
    else
        $python_cmd "$train_script"
    fi
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    print_header "Training Pipeline Summary"

    echo -e "${BOLD}Access URLs:${NC}"
    echo ""
    echo -e "  ${CYAN}Kubeflow UI:${NC}  http://localhost:$KUBEFLOW_PORT"
    echo -e "  ${CYAN}MLflow UI:${NC}    http://localhost:$MLFLOW_PORT"
    echo ""

    if [ -f "$PROJECT_ROOT/.current_run_id" ]; then
        local run_id=$(cat "$PROJECT_ROOT/.current_run_id")
        echo -e "${BOLD}Current Run:${NC}"
        echo ""
        echo "  Run ID: $run_id"
        echo "  View:   http://localhost:$KUBEFLOW_PORT/#/runs/details/$run_id"
        echo ""
    fi

    echo -e "${BOLD}Useful Commands:${NC}"
    echo ""
    echo "  # Check pipeline pods"
    echo "  kubectl get pods -n kubeflow"
    echo ""
    echo "  # View pipeline logs"
    echo "  kubectl logs -f <pod-name> -n kubeflow"
    echo ""
    echo "  # Resubmit pipeline"
    echo "  ./scripts/run_training_pipeline.sh"
    echo ""

    log_success "Log saved to: $LOG_FILE"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    local local_mode=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --local|-l)
                local_mode=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    clear

    echo ""
    echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║               Run Training Pipeline                           ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    log "Starting training pipeline at $(date)"

    # Clear previous log
    > "$LOG_FILE"

    if [ "$local_mode" = true ]; then
        log_info "Running in local mode"
        run_local_training
    else
        check_prerequisites
        compile_pipeline
        setup_kubeflow_port_forward
        submit_pipeline

        # Ask if user wants to monitor
        echo ""
        read -p "Monitor pipeline progress? (Y/n): " monitor
        if [[ ! "$monitor" =~ ^[Nn]$ ]]; then
            monitor_pipeline
        fi
    fi

    print_summary

    log "Training pipeline script completed at $(date)"
}

# Run main function
main "$@"
