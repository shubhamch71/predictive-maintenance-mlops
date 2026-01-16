#!/bin/bash
# =============================================================================
# Deploy ML Models to Kubernetes
# =============================================================================
# Builds Docker images, pushes to registry, and deploys KServe InferenceServices.
#
# Usage: ./scripts/deploy_models.sh [--registry <registry>]
#   --registry: Docker registry (default: localhost:5000)
#   --dockerhub: Use DockerHub (prompts for username)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/deploy_models.log"
PID_FILE="$PROJECT_ROOT/.model-port-forward.pid"

NAMESPACE="mlops"
MODEL_API_PORT=8000
DEFAULT_REGISTRY="localhost:5000"
IMAGE_NAME="predictive-maintenance-api"
IMAGE_TAG="latest"

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
# Global Variables
# =============================================================================
REGISTRY=""
USE_LOCAL_REGISTRY=true
DOCKERHUB_USERNAME=""

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --registry)
                REGISTRY="$2"
                USE_LOCAL_REGISTRY=false
                shift 2
                ;;
            --dockerhub)
                USE_LOCAL_REGISTRY=false
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    # Set default registry
    if [ -z "$REGISTRY" ] && [ "$USE_LOCAL_REGISTRY" = true ]; then
        REGISTRY="$DEFAULT_REGISTRY"
    fi
}

# =============================================================================
# Prerequisite Checks
# =============================================================================
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        exit 1
    fi
    log_success "Docker found"

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running!"
        exit 1
    fi
    log_success "Docker is running"

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

    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace $NAMESPACE..."
        kubectl create namespace "$NAMESPACE"
    fi
    log_success "Namespace $NAMESPACE exists"

    # Check Dockerfile
    local dockerfile="$PROJECT_ROOT/docker/serving/Dockerfile"
    if [ ! -f "$dockerfile" ]; then
        log_warning "Dockerfile not found: $dockerfile"
        log_info "Will create a minimal serving container"
    else
        log_success "Dockerfile found"
    fi
}

# =============================================================================
# Local Registry Setup
# =============================================================================
setup_local_registry() {
    print_header "Setting Up Local Registry"

    # Check if registry is already running
    if docker ps --format '{{.Names}}' | grep -q '^registry$'; then
        log_info "Local registry already running"
        return 0
    fi

    # Check if registry container exists but stopped
    if docker ps -a --format '{{.Names}}' | grep -q '^registry$'; then
        log "Starting existing registry container..."
        docker start registry
    else
        log "Starting local Docker registry..."
        docker run -d -p 5000:5000 --name registry --restart=always registry:2
    fi

    # Wait for registry to be ready
    sleep 2

    # Verify registry is accessible
    local retries=10
    while [ $retries -gt 0 ]; do
        if curl -s "http://localhost:5000/v2/" &>/dev/null; then
            log_success "Local registry is running at localhost:5000"
            return 0
        fi
        sleep 1
        ((retries--))
    done

    log_error "Could not verify local registry"
    return 1
}

# =============================================================================
# DockerHub Setup
# =============================================================================
setup_dockerhub() {
    print_header "DockerHub Setup"

    # Check if already logged in
    if docker info 2>/dev/null | grep -q "Username:"; then
        DOCKERHUB_USERNAME=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
        log_info "Already logged in as: $DOCKERHUB_USERNAME"
    else
        log "Please log in to DockerHub..."
        docker login

        DOCKERHUB_USERNAME=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    fi

    if [ -z "$DOCKERHUB_USERNAME" ]; then
        log_error "Could not determine DockerHub username"
        exit 1
    fi

    REGISTRY="$DOCKERHUB_USERNAME"
    log_success "Using DockerHub registry: $REGISTRY"
}

# =============================================================================
# Build Docker Images
# =============================================================================
build_images() {
    print_header "Building Docker Images"

    local dockerfile="$PROJECT_ROOT/docker/serving/Dockerfile"
    local full_image="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    if [ ! -f "$dockerfile" ]; then
        log "Creating minimal serving Dockerfile..."

        mkdir -p "$PROJECT_ROOT/docker/serving"
        cat > "$dockerfile" <<'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.0 \
    uvicorn[standard]==0.24.0 \
    mlflow==2.9.0 \
    scikit-learn==1.3.2 \
    xgboost==2.0.2 \
    numpy==1.26.2 \
    pandas==2.1.3 \
    prometheus-client==0.19.0

# Copy application
COPY src/serving/ /app/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    fi

    log "Building image: $full_image"
    log_info "This may take a few minutes..."

    # Build the image
    docker build \
        -t "$full_image" \
        -f "$dockerfile" \
        "$PROJECT_ROOT" 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Image built: $full_image"
    else
        log_error "Failed to build image"
        exit 1
    fi

    # Also tag without registry for local use
    docker tag "$full_image" "$IMAGE_NAME:$IMAGE_TAG"
}

# =============================================================================
# Push Docker Images
# =============================================================================
push_images() {
    print_header "Pushing Docker Images"

    local full_image="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    log "Pushing image: $full_image"

    docker push "$full_image" 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Image pushed: $full_image"
    else
        log_error "Failed to push image"
        log_info "If using local registry, ensure it's running: docker ps | grep registry"
        exit 1
    fi
}

# =============================================================================
# Deploy KServe InferenceServices
# =============================================================================
deploy_inferenceservices() {
    print_header "Deploying InferenceServices"

    local full_image="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    # Create a simple InferenceService that works with Docker Desktop
    log "Creating InferenceService for ensemble model..."

    kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: rul-ensemble
  labels:
    app.kubernetes.io/name: rul-ensemble
    app.kubernetes.io/component: inference-service
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 3
    containers:
      - name: predictor
        image: $full_image
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000
            protocol: TCP
        env:
          - name: MODEL_TYPE
            value: "ensemble"
          - name: MLFLOW_TRACKING_URI
            value: "http://mlflow.mlops.svc.cluster.local:5000"
        resources:
          requests:
            cpu: "200m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "2Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
EOF

    log_success "InferenceService 'rul-ensemble' created"

    # Apply existing KServe manifests if available
    local kserve_dir="$PROJECT_ROOT/k8s/kserve"
    if [ -d "$kserve_dir" ]; then
        for manifest in "$kserve_dir"/inferenceservice-*.yaml; do
            if [ -f "$manifest" ]; then
                local name=$(basename "$manifest" .yaml)
                log "Applying $name..."

                # Update image in manifest and apply
                sed "s|image:.*|image: $full_image|g" "$manifest" | \
                    kubectl apply -n "$NAMESPACE" -f - 2>/dev/null || {
                    log_warning "Could not apply $manifest"
                }
            fi
        done
    fi
}

# =============================================================================
# Wait for Deployments
# =============================================================================
wait_for_deployments() {
    print_header "Waiting for Deployments"

    log "Waiting for InferenceServices to be ready..."
    log_info "This may take 2-3 minutes..."

    # Wait for the InferenceService to be ready
    local retries=30
    while [ $retries -gt 0 ]; do
        local status=$(kubectl get inferenceservice rul-ensemble -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")

        if [ "$status" == "True" ]; then
            log_success "InferenceService 'rul-ensemble' is ready"
            break
        fi

        local reason=$(kubectl get inferenceservice rul-ensemble -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].reason}' 2>/dev/null || echo "Pending")
        log_info "Status: $reason ($retries retries left)"

        sleep 10
        ((retries--))
    done

    if [ $retries -eq 0 ]; then
        log_warning "Timeout waiting for InferenceService"
        log_info "Check status with: kubectl get inferenceservice -n $NAMESPACE"
    fi

    # Show all InferenceServices
    echo ""
    log "InferenceServices status:"
    kubectl get inferenceservice -n "$NAMESPACE" 2>/dev/null || true
}

# =============================================================================
# Setup Port Forward
# =============================================================================
setup_model_port_forward() {
    print_header "Setting Up Model Access"

    # Kill existing port-forward
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            kill "$old_pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi

    # Find the predictor service
    local svc_name=$(kubectl get svc -n "$NAMESPACE" -l serving.kserve.io/inferenceservice=rul-ensemble -o name 2>/dev/null | head -1)

    if [ -z "$svc_name" ]; then
        # Fallback to pod port-forward
        local pod_name=$(kubectl get pods -n "$NAMESPACE" -l serving.kserve.io/inferenceservice=rul-ensemble -o name 2>/dev/null | head -1)
        if [ -z "$pod_name" ]; then
            log_warning "No inference service pods found"
            log_info "Creating direct port-forward to deployment..."

            kubectl port-forward -n "$NAMESPACE" deployment/rul-ensemble-predictor "$MODEL_API_PORT:8000" &>/dev/null &
        else
            kubectl port-forward -n "$NAMESPACE" "$pod_name" "$MODEL_API_PORT:8000" &>/dev/null &
        fi
    else
        kubectl port-forward -n "$NAMESPACE" "$svc_name" "$MODEL_API_PORT:8000" &>/dev/null &
    fi

    local pf_pid=$!
    echo "$pf_pid" > "$PID_FILE"

    sleep 3

    if kill -0 "$pf_pid" 2>/dev/null; then
        log_success "Model API accessible at http://localhost:$MODEL_API_PORT"
    else
        log_warning "Port-forward may have failed"
    fi
}

# =============================================================================
# Run Smoke Test
# =============================================================================
run_smoke_test() {
    print_header "Running Smoke Test"

    local api_url="http://localhost:$MODEL_API_PORT"

    # Test health endpoint
    log "Testing health endpoint..."
    local health_response=$(curl -s "$api_url/health" 2>/dev/null || echo "")

    if [ -n "$health_response" ]; then
        log_success "Health check passed"
        echo "  Response: $health_response"
    else
        log_warning "Health check failed or service not ready"
        return 1
    fi

    # Test prediction endpoint
    log "Testing prediction endpoint..."

    local test_payload='{
        "unit_id": 1,
        "cycle": 100,
        "operational_setting_1": 0.0023,
        "operational_setting_2": 0.0003,
        "operational_setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 642.15,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.36,
        "sensor_8": 2388.02,
        "sensor_9": 9046.19,
        "sensor_10": 1.30,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_16": 0.03,
        "sensor_17": 392,
        "sensor_18": 2388,
        "sensor_19": 100.0,
        "sensor_20": 39.06,
        "sensor_21": 23.4190
    }'

    local predict_response=$(curl -s -X POST "$api_url/predict" \
        -H "Content-Type: application/json" \
        -d "$test_payload" 2>/dev/null || echo "")

    if [ -n "$predict_response" ]; then
        log_success "Prediction test passed"
        echo "  Response: $predict_response"
    else
        log_warning "Prediction test failed"
        log_info "The model endpoint may not be fully ready"
    fi

    # Create test script
    create_test_script
}

# =============================================================================
# Create Test Script
# =============================================================================
create_test_script() {
    local test_script="$PROJECT_ROOT/test-prediction.sh"

    cat > "$test_script" <<'EOF'
#!/bin/bash
# Test prediction endpoint

API_URL="${1:-http://localhost:8000}"

echo "Testing prediction API at $API_URL"
echo ""

# Health check
echo "1. Health Check:"
curl -s "$API_URL/health" | python3 -m json.tool 2>/dev/null || curl -s "$API_URL/health"
echo ""

# Sample prediction
echo "2. Sample Prediction:"
curl -s -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "unit_id": 1,
        "cycle": 100,
        "operational_setting_1": 0.0023,
        "operational_setting_2": 0.0003,
        "operational_setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 642.15,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.36,
        "sensor_8": 2388.02,
        "sensor_9": 9046.19,
        "sensor_10": 1.30,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_16": 0.03,
        "sensor_17": 392,
        "sensor_18": 2388,
        "sensor_19": 100.0,
        "sensor_20": 39.06,
        "sensor_21": 23.4190
    }' | python3 -m json.tool 2>/dev/null || echo "Request sent"
echo ""

# Metrics
echo "3. Metrics:"
curl -s "$API_URL/metrics" | head -20
echo "..."
EOF

    chmod +x "$test_script"
    log_success "Test script created: $test_script"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    print_header "Deployment Summary"

    local full_image="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    echo -e "${GREEN}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   Model Deployment Complete!                                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo -e "${BOLD}Docker Image:${NC}"
    echo "  $full_image"
    echo ""

    echo -e "${BOLD}Model API:${NC}"
    echo "  http://localhost:$MODEL_API_PORT"
    echo ""

    echo -e "${BOLD}Available Endpoints:${NC}"
    echo "  POST /predict       - Single prediction"
    echo "  POST /predict_batch - Batch predictions"
    echo "  GET  /health        - Health check"
    echo "  GET  /metrics       - Prometheus metrics"
    echo ""

    echo -e "${BOLD}Test the API:${NC}"
    echo "  ./test-prediction.sh"
    echo ""

    echo -e "${BOLD}Kubernetes Resources:${NC}"
    kubectl get inferenceservice -n "$NAMESPACE" --no-headers 2>/dev/null | while read line; do
        local name=$(echo "$line" | awk '{print $1}')
        local ready=$(echo "$line" | awk '{print $2}')
        if [ "$ready" == "True" ]; then
            echo -e "  ${GREEN}✓${NC} $name (Ready)"
        else
            echo -e "  ${YELLOW}○${NC} $name (Not Ready)"
        fi
    done
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
    echo -e "${BOLD}${CYAN}║                   Deploy ML Models                            ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Parse arguments
    parse_args "$@"

    log "Starting model deployment at $(date)"

    # Clear previous log
    > "$LOG_FILE"

    # Run deployment steps
    check_prerequisites

    if [ "$USE_LOCAL_REGISTRY" = true ]; then
        setup_local_registry
    else
        setup_dockerhub
    fi

    build_images
    push_images
    deploy_inferenceservices
    wait_for_deployments
    setup_model_port_forward
    run_smoke_test
    print_summary

    log "Model deployment completed at $(date)"
}

# Run main function
main "$@"
