#!/bin/bash
# =============================================================================
# Access All Services - Port Forward Setup
# =============================================================================
# Starts port-forwards for all MLOps platform services.
#
# Usage: ./scripts/access_all_services.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/.port-forward-pids"

# Service ports
MLFLOW_PORT=5000
KUBEFLOW_PORT=8080
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090
MODEL_API_PORT=8000
ARGO_PORT=2746

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
log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Helper Functions
# =============================================================================
check_port() {
    local port=$1
    if nc -z localhost "$port" 2>/dev/null; then
        return 0  # Port in use
    else
        return 1  # Port free
    fi
}

start_port_forward() {
    local service_name=$1
    local namespace=$2
    local service=$3
    local local_port=$4
    local remote_port=$5

    # Check if port is already in use
    if check_port "$local_port"; then
        log_warning "$service_name: Port $local_port already in use"
        return 0
    fi

    # Check if service exists
    if ! kubectl get "$service" -n "$namespace" &>/dev/null; then
        log_warning "$service_name: Service not found ($service in $namespace)"
        return 1
    fi

    # Start port-forward
    kubectl port-forward -n "$namespace" "$service" "$local_port:$remote_port" &>/dev/null &
    local pid=$!

    # Record PID
    echo "$pid:$service_name:$local_port" >> "$PID_FILE"

    # Wait and verify
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        log_success "$service_name: localhost:$local_port (PID: $pid)"
        return 0
    else
        log_error "$service_name: Failed to start port-forward"
        return 1
    fi
}

# =============================================================================
# Stop Existing Port-Forwards
# =============================================================================
stop_existing() {
    if [ -f "$PID_FILE" ]; then
        echo "Stopping existing port-forwards..."
        while IFS=: read -r pid name port; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
        sleep 1
    fi
}

# =============================================================================
# Start All Port-Forwards
# =============================================================================
start_all_port_forwards() {
    echo ""
    echo -e "${BOLD}Starting port-forwards...${NC}"
    echo ""

    # Initialize PID file
    > "$PID_FILE"

    # MLflow
    start_port_forward "MLflow" "mlops" "svc/mlflow" "$MLFLOW_PORT" "5000"

    # Kubeflow Pipelines UI
    start_port_forward "Kubeflow" "kubeflow" "svc/ml-pipeline-ui" "$KUBEFLOW_PORT" "80"

    # Grafana (check both Helm and custom deployment)
    if kubectl get svc prometheus-grafana -n monitoring &>/dev/null; then
        start_port_forward "Grafana" "monitoring" "svc/prometheus-grafana" "$GRAFANA_PORT" "80"
    else
        start_port_forward "Grafana" "monitoring" "svc/grafana" "$GRAFANA_PORT" "3000"
    fi

    # Prometheus
    if kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring &>/dev/null; then
        start_port_forward "Prometheus" "monitoring" "svc/prometheus-kube-prometheus-prometheus" "$PROMETHEUS_PORT" "9090"
    else
        start_port_forward "Prometheus" "monitoring" "svc/prometheus" "$PROMETHEUS_PORT" "9090"
    fi

    # Model API (InferenceService)
    if kubectl get svc -n mlops -l serving.kserve.io/inferenceservice=rul-ensemble -o name &>/dev/null 2>&1; then
        local model_svc=$(kubectl get svc -n mlops -l serving.kserve.io/inferenceservice=rul-ensemble -o name 2>/dev/null | head -1)
        if [ -n "$model_svc" ]; then
            start_port_forward "Model API" "mlops" "$model_svc" "$MODEL_API_PORT" "8000"
        fi
    fi

    # Argo Workflows
    start_port_forward "Argo" "argo" "svc/argo-server" "$ARGO_PORT" "2746"
}

# =============================================================================
# Create Dashboard HTML
# =============================================================================
create_dashboard() {
    local dashboard_file="$PROJECT_ROOT/dashboard.html"

    cat > "$dashboard_file" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>MLOps Platform Dashboard</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 40px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #888;
            text-align: center;
            margin-bottom: 40px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        }
        .card-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .card h2 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.4em;
        }
        .card p {
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .card-url {
            background: #f5f5f5;
            padding: 10px 15px;
            border-radius: 8px;
            font-family: monospace;
            margin-bottom: 15px;
            font-size: 14px;
            word-break: break-all;
        }
        .card a {
            display: inline-block;
            background: #0066FF;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.3s;
        }
        .card a:hover {
            background: #0052CC;
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
        }
        .status.running {
            background: #d4edda;
            color: #155724;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #666;
        }
        .footer a {
            color: #0066FF;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 MLOps Platform</h1>
        <p class="subtitle">Predictive Maintenance - All Services</p>

        <div class="grid">
            <div class="card">
                <div class="card-icon">🔬</div>
                <h2>MLflow <span class="status running">Running</span></h2>
                <p>Experiment tracking, model registry, and artifact storage.</p>
                <div class="card-url">http://localhost:$MLFLOW_PORT</div>
                <a href="http://localhost:$MLFLOW_PORT" target="_blank">Open MLflow</a>
            </div>

            <div class="card">
                <div class="card-icon">🔄</div>
                <h2>Kubeflow Pipelines <span class="status running">Running</span></h2>
                <p>ML pipeline orchestration and workflow management.</p>
                <div class="card-url">http://localhost:$KUBEFLOW_PORT</div>
                <a href="http://localhost:$KUBEFLOW_PORT" target="_blank">Open Kubeflow</a>
            </div>

            <div class="card">
                <div class="card-icon">📊</div>
                <h2>Grafana <span class="status running">Running</span></h2>
                <p>Dashboards for model performance and system metrics.</p>
                <div class="card-url">http://localhost:$GRAFANA_PORT</div>
                <a href="http://localhost:$GRAFANA_PORT" target="_blank">Open Grafana</a>
            </div>

            <div class="card">
                <div class="card-icon">📈</div>
                <h2>Prometheus <span class="status running">Running</span></h2>
                <p>Metrics collection, alerting, and time series database.</p>
                <div class="card-url">http://localhost:$PROMETHEUS_PORT</div>
                <a href="http://localhost:$PROMETHEUS_PORT" target="_blank">Open Prometheus</a>
            </div>

            <div class="card">
                <div class="card-icon">🤖</div>
                <h2>Model API <span class="status running">Running</span></h2>
                <p>RUL prediction endpoint for inference requests.</p>
                <div class="card-url">http://localhost:$MODEL_API_PORT</div>
                <a href="http://localhost:$MODEL_API_PORT/docs" target="_blank">Open API Docs</a>
            </div>

            <div class="card">
                <div class="card-icon">⚡</div>
                <h2>Argo Workflows <span class="status running">Running</span></h2>
                <p>Workflow automation and retraining pipelines.</p>
                <div class="card-url">http://localhost:$ARGO_PORT</div>
                <a href="http://localhost:$ARGO_PORT" target="_blank">Open Argo</a>
            </div>
        </div>

        <div class="footer">
            <p>Stop all services: <code>./scripts/stop_all_services.sh</code></p>
            <p style="margin-top: 10px;">
                <a href="https://github.com/your-repo/predictive-maintenance-mlops">Documentation</a>
            </p>
        </div>
    </div>

    <script>
        // Auto-refresh status every 30 seconds
        setInterval(() => {
            document.querySelectorAll('.card a').forEach(link => {
                fetch(link.href, {mode: 'no-cors'})
                    .then(() => {
                        link.closest('.card').querySelector('.status').className = 'status running';
                        link.closest('.card').querySelector('.status').textContent = 'Running';
                    })
                    .catch(() => {
                        link.closest('.card').querySelector('.status').className = 'status';
                        link.closest('.card').querySelector('.status').textContent = 'Checking...';
                    });
            });
        }, 30000);
    </script>
</body>
</html>
EOF

    log_success "Dashboard created: $dashboard_file"
}

# =============================================================================
# Print Access Information
# =============================================================================
print_access_info() {
    echo ""
    echo -e "${BOLD}${CYAN}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${CYAN}│              MLOps Platform Access URLs                     │${NC}"
    echo -e "${BOLD}${CYAN}├─────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  ${BOLD}MLflow:${NC}       http://localhost:$MLFLOW_PORT                     ${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  ${BOLD}Kubeflow:${NC}     http://localhost:$KUBEFLOW_PORT                     ${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  ${BOLD}Grafana:${NC}      http://localhost:$GRAFANA_PORT                      ${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  ${BOLD}Prometheus:${NC}   http://localhost:$PROMETHEUS_PORT                     ${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  ${BOLD}Model API:${NC}    http://localhost:$MODEL_API_PORT                      ${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}│${NC}  ${BOLD}Argo:${NC}         http://localhost:$ARGO_PORT                      ${CYAN}│${NC}"
    echo -e "${BOLD}${CYAN}└─────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    echo -e "${BOLD}Default Credentials:${NC}"
    echo "  Grafana: admin / admin123 (or prom-operator if using Helm)"
    echo ""

    echo -e "${BOLD}Quick Commands:${NC}"
    echo "  Stop services:  ./scripts/stop_all_services.sh"
    echo "  View dashboard: open $PROJECT_ROOT/dashboard.html"
    echo ""
}

# =============================================================================
# Open Dashboard
# =============================================================================
open_dashboard() {
    local dashboard_file="$PROJECT_ROOT/dashboard.html"

    if [ -f "$dashboard_file" ]; then
        if command -v open &>/dev/null; then
            open "$dashboard_file" 2>/dev/null &
        elif command -v xdg-open &>/dev/null; then
            xdg-open "$dashboard_file" 2>/dev/null &
        elif command -v start &>/dev/null; then
            start "$dashboard_file" 2>/dev/null &
        fi
    fi
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    clear

    echo ""
    echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║             Access All MLOps Services                         ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Check kubectl
    if ! command -v kubectl &>/dev/null; then
        log_error "kubectl is not installed!"
        exit 1
    fi

    # Check cluster
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster!"
        exit 1
    fi

    # Stop existing port-forwards
    stop_existing

    # Start all port-forwards
    start_all_port_forwards

    # Create dashboard
    create_dashboard

    # Print access info
    print_access_info

    # Open dashboard
    open_dashboard

    echo -e "${GREEN}All services are now accessible!${NC}"
    echo ""
    echo "Press Ctrl+C to stop all port-forwards, or run:"
    echo "  ./scripts/stop_all_services.sh"
    echo ""

    # Keep script running
    echo "Port-forwards are running in the background."
    echo "Dashboard opened in your default browser."
}

# Run main function
main "$@"
