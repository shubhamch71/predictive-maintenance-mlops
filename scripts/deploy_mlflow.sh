#!/bin/bash
# =============================================================================
# Deploy MLflow to Kubernetes
# =============================================================================
# Deploys MLflow tracking server with PostgreSQL backend.
#
# Usage: ./scripts/deploy_mlflow.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/deploy_mlflow.log"
PID_FILE="$PROJECT_ROOT/.mlflow-port-forward.pid"

NAMESPACE="mlops"
MLFLOW_PORT=5000

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

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster!"
        exit 1
    fi
    log_success "Connected to Kubernetes cluster"

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace $NAMESPACE..."
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace created: $NAMESPACE"
    else
        log_info "Namespace $NAMESPACE already exists"
    fi
}

# =============================================================================
# Deployment Functions
# =============================================================================
deploy_postgres() {
    print_header "Deploying PostgreSQL"

    local postgres_manifest="$PROJECT_ROOT/k8s/mlflow/postgres.yaml"

    if [ ! -f "$postgres_manifest" ]; then
        log_error "PostgreSQL manifest not found: $postgres_manifest"
        log_info "Creating minimal PostgreSQL deployment..."

        # Create minimal PostgreSQL deployment inline
        kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: postgres-credentials
type: Opaque
stringData:
  POSTGRES_USER: mlflow
  POSTGRES_PASSWORD: mlflow123
  POSTGRES_DB: mlflow
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:15-alpine
          ports:
            - containerPort: 5432
          envFrom:
            - secretRef:
                name: postgres-credentials
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
      volumes:
        - name: postgres-storage
          persistentVolumeClaim:
            claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  clusterIP: None
EOF
    else
        log "Applying PostgreSQL manifest..."
        kubectl apply -f "$postgres_manifest" -n "$NAMESPACE"
    fi

    # Wait for PostgreSQL to be ready
    log "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=Ready pod -l app=postgres -n "$NAMESPACE" --timeout=180s 2>/dev/null || {
        # Fallback: wait for StatefulSet
        local retries=30
        while [ $retries -gt 0 ]; do
            local ready=$(kubectl get statefulset postgres -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            if [ "$ready" == "1" ]; then
                break
            fi
            log_info "Waiting for PostgreSQL... ($retries retries left)"
            sleep 5
            ((retries--))
        done
    }

    log_success "PostgreSQL deployed and ready"
}

deploy_pvc() {
    print_header "Creating Persistent Volume Claims"

    local pvc_manifest="$PROJECT_ROOT/k8s/mlflow/pvc.yaml"

    if [ -f "$pvc_manifest" ]; then
        log "Applying PVC manifest..."
        kubectl apply -f "$pvc_manifest" -n "$NAMESPACE"
    else
        log "Creating MLflow artifacts PVC..."
        kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-artifacts
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF
    fi

    log_success "PVCs created"
}

deploy_mlflow() {
    print_header "Deploying MLflow"

    local mlflow_manifest="$PROJECT_ROOT/k8s/mlflow/deployment.yaml"

    if [ ! -f "$mlflow_manifest" ]; then
        log_warning "MLflow manifest not found: $mlflow_manifest"
        log_info "Creating MLflow deployment..."

        kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  labels:
    app: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:v2.9.0
          command:
            - mlflow
            - server
            - --host=0.0.0.0
            - --port=5000
            - --backend-store-uri=postgresql://mlflow:mlflow123@postgres:5432/mlflow
            - --default-artifact-root=/mlflow/artifacts
          ports:
            - containerPort: 5000
          volumeMounts:
            - name: mlflow-artifacts
              mountPath: /mlflow/artifacts
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "1Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: mlflow-artifacts
          persistentVolumeClaim:
            claimName: mlflow-artifacts
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  labels:
    app: mlflow
spec:
  type: ClusterIP
  selector:
    app: mlflow
  ports:
    - name: http
      port: 5000
      targetPort: 5000
EOF
    else
        log "Applying MLflow manifest..."
        kubectl apply -f "$mlflow_manifest" -n "$NAMESPACE"
    fi

    # Wait for MLflow to be ready
    log "Waiting for MLflow to be ready..."
    kubectl wait --for=condition=Available deployment/mlflow -n "$NAMESPACE" --timeout=180s 2>/dev/null || {
        local retries=30
        while [ $retries -gt 0 ]; do
            local ready=$(kubectl get deployment mlflow -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            if [ "$ready" == "1" ]; then
                break
            fi
            log_info "Waiting for MLflow... ($retries retries left)"
            sleep 5
            ((retries--))
        done
    }

    log_success "MLflow deployed and ready"
}

setup_port_forward() {
    print_header "Setting Up Port Forward"

    # Kill existing port-forward if any
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log "Stopping existing port-forward (PID: $old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            sleep 2
        fi
        rm -f "$PID_FILE"
    fi

    # Check if port is already in use
    if lsof -i ":$MLFLOW_PORT" &>/dev/null || nc -z localhost "$MLFLOW_PORT" 2>/dev/null; then
        log_warning "Port $MLFLOW_PORT is already in use"
        log_info "Checking if it's already MLflow..."

        # Try to access MLflow
        if curl -s "http://localhost:$MLFLOW_PORT/health" &>/dev/null; then
            log_success "MLflow is already accessible on port $MLFLOW_PORT"
            return 0
        else
            log_error "Port $MLFLOW_PORT is in use by another process"
            log_info "Please free port $MLFLOW_PORT and try again"
            return 1
        fi
    fi

    # Start port-forward
    log "Starting port-forward to MLflow..."
    kubectl port-forward -n "$NAMESPACE" service/mlflow "$MLFLOW_PORT:5000" &>/dev/null &
    local pf_pid=$!

    # Save PID
    echo "$pf_pid" > "$PID_FILE"

    # Wait for port-forward to be ready
    sleep 3

    # Verify port-forward is working
    if ! kill -0 "$pf_pid" 2>/dev/null; then
        log_error "Port-forward failed to start"
        return 1
    fi

    # Test connection
    local retries=10
    while [ $retries -gt 0 ]; do
        if curl -s "http://localhost:$MLFLOW_PORT/health" &>/dev/null; then
            log_success "Port-forward established (PID: $pf_pid)"
            log_success "MLflow accessible at http://localhost:$MLFLOW_PORT"
            return 0
        fi
        sleep 1
        ((retries--))
    done

    log_error "Could not verify MLflow connection"
    return 1
}

setup_environment() {
    print_header "Configuring Environment"

    local mlflow_uri="http://localhost:$MLFLOW_PORT"

    # Detect shell config file
    local shell_config=""
    if [ -f "$HOME/.zshrc" ]; then
        shell_config="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        shell_config="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        shell_config="$HOME/.bash_profile"
    fi

    if [ -n "$shell_config" ]; then
        # Check if already configured
        if grep -q "MLFLOW_TRACKING_URI" "$shell_config" 2>/dev/null; then
            log_info "MLFLOW_TRACKING_URI already configured in $shell_config"
        else
            log "Adding MLFLOW_TRACKING_URI to $shell_config..."
            echo "" >> "$shell_config"
            echo "# MLflow Configuration (added by deploy_mlflow.sh)" >> "$shell_config"
            echo "export MLFLOW_TRACKING_URI=$mlflow_uri" >> "$shell_config"
            log_success "Environment variable added to $shell_config"
        fi
    fi

    # Export for current session
    export MLFLOW_TRACKING_URI="$mlflow_uri"
    log_success "MLFLOW_TRACKING_URI set to $mlflow_uri"
}

create_access_file() {
    print_header "Creating Access File"

    local html_file="$PROJECT_ROOT/mlflow-access.html"

    cat > "$html_file" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>MLflow Access</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card {
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
            max-width: 400px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .logo {
            font-size: 60px;
            margin-bottom: 20px;
        }
        .url {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-family: monospace;
            font-size: 16px;
        }
        a.button {
            display: inline-block;
            background: #0066FF;
            color: white;
            padding: 15px 40px;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin-top: 20px;
            transition: background 0.3s;
        }
        a.button:hover {
            background: #0052CC;
        }
        .status {
            color: #28a745;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="logo">🔬</div>
        <h1>MLflow Tracking Server</h1>
        <p class="status">● Running</p>
        <div class="url">http://localhost:$MLFLOW_PORT</div>
        <a href="http://localhost:$MLFLOW_PORT" class="button" target="_blank">Open MLflow UI</a>
        <p style="margin-top: 30px; color: #666; font-size: 14px;">
            Track experiments, compare runs, and manage models.
        </p>
    </div>
</body>
</html>
EOF

    log_success "Access file created: $html_file"

    # Try to open in browser
    if command -v open &>/dev/null; then
        open "$html_file" 2>/dev/null || true
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$html_file" 2>/dev/null || true
    fi
}

print_summary() {
    print_header "Deployment Complete!"

    echo -e "${GREEN}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   MLflow Deployed Successfully!                               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo -e "${BOLD}Access Information:${NC}"
    echo ""
    echo -e "  ${CYAN}MLflow UI:${NC}  http://localhost:$MLFLOW_PORT"
    echo ""
    echo -e "  ${CYAN}Tracking URI:${NC}  http://localhost:$MLFLOW_PORT"
    echo "                  (for Python SDK)"
    echo ""

    echo -e "${BOLD}Usage in Python:${NC}"
    echo ""
    echo -e "  ${YELLOW}import mlflow${NC}"
    echo -e "  ${YELLOW}mlflow.set_tracking_uri('http://localhost:$MLFLOW_PORT')${NC}"
    echo ""

    echo -e "${BOLD}Kubernetes Resources:${NC}"
    echo ""
    kubectl get pods -n "$NAMESPACE" -l 'app in (mlflow,postgres)' --no-headers | while read line; do
        local name=$(echo "$line" | awk '{print $1}')
        local status=$(echo "$line" | awk '{print $3}')
        if [[ "$status" == "Running" ]]; then
            echo -e "  ${GREEN}✓${NC} $name ($status)"
        else
            echo -e "  ${YELLOW}○${NC} $name ($status)"
        fi
    done
    echo ""

    echo -e "${BOLD}To stop port-forward:${NC}"
    echo "  ./scripts/stop_all_services.sh"
    echo "  # or"
    echo "  kill \$(cat $PID_FILE)"
    echo ""

    log_success "Deployment log saved to: $LOG_FILE"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    clear

    echo ""
    echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║                   Deploy MLflow to Kubernetes                 ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    log "Starting MLflow deployment at $(date)"

    # Clear previous log
    > "$LOG_FILE"

    # Run deployment steps
    check_prerequisites
    deploy_postgres
    deploy_pvc
    deploy_mlflow
    setup_port_forward
    setup_environment
    create_access_file
    print_summary

    log "Deployment completed at $(date)"
}

# Run main function
main "$@"
