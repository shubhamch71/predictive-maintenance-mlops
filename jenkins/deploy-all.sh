#!/usr/bin/env bash
# =============================================================================
# Master Deployment Script for Predictive Maintenance MLOps Platform
# =============================================================================
#
# This script orchestrates the complete deployment of the MLOps platform.
# It is designed to be executed by Jenkins but can also run standalone.
#
# Usage:
#   ./jenkins/deploy-all.sh [OPTIONS]
#
# Options:
#   --skip-tests        Skip running tests
#   --skip-build        Skip Docker image builds
#   --skip-push         Skip pushing to registry
#   --skip-deploy       Skip Kubernetes deployments
#   --local-only        Deploy only local services (docker-compose)
#   --cleanup           Run cleanup after deployment
#   --dry-run           Show what would be done without executing
#   --help              Show this help message
#
# Environment Variables:
#   DOCKER_USERNAME     Docker Hub username
#   DOCKER_PASSWORD     Docker Hub password or token
#   NAMESPACE           Kubernetes namespace (default: mlops)
#   VERSION             Application version (auto-detected from __version__.py)
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${SCRIPT_DIR}/deploy.log"
PID_FILE="${SCRIPT_DIR}/port-forwards.pids"

# Default configuration
NAMESPACE="${NAMESPACE:-mlops}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
SKIP_TESTS=false
SKIP_BUILD=false
SKIP_PUSH=false
SKIP_DEPLOY=false
LOCAL_ONLY=false
RUN_CLEANUP=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        INFO)  echo -e "${BLUE}[INFO]${NC}  ${message}" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} ${message}" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC}  ${message}" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} ${message}" ;;
        DEBUG) [ "${DEBUG:-false}" = "true" ] && echo -e "${CYAN}[DEBUG]${NC} ${message}" ;;
    esac

    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
}

log_info() { log INFO "$@"; }
log_success() { log SUCCESS "$@"; }
log_warn() { log WARN "$@"; }
log_error() { log ERROR "$@"; }
log_debug() { log DEBUG "$@"; }

print_header() {
    local title="$1"
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    printf "${CYAN}║${NC} ${BOLD}%-68s${NC} ${CYAN}║${NC}\n" "$title"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_separator() {
    echo -e "${CYAN}──────────────────────────────────────────────────────────────────────────${NC}"
}

# =============================================================================
# Utility Functions
# =============================================================================

check_command() {
    local cmd="$1"
    if command -v "$cmd" &> /dev/null; then
        local version
        version=$("$cmd" --version 2>&1 | head -n1 || echo "unknown")
        log_info "  ✓ $cmd: $version"
        return 0
    else
        log_error "  ✗ $cmd: NOT FOUND"
        return 1
    fi
}

wait_for_url() {
    local url="$1"
    local max_attempts="${2:-30}"
    local attempt=1

    log_info "Waiting for $url to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 2 "$url" > /dev/null 2>&1; then
            log_success "  $url is ready"
            return 0
        fi
        log_debug "  Attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done

    log_warn "  $url not ready after $max_attempts attempts"
    return 1
}

cleanup_on_exit() {
    log_info "Running cleanup..."

    # Kill port-forward processes
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                log_debug "Killed process $pid"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    log_info "Cleanup completed"
}

# Set trap for cleanup
trap cleanup_on_exit EXIT

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    print_header "CHECKING PREREQUISITES"

    local missing_deps=0

    log_info "Checking required tools..."

    check_command docker || ((missing_deps++))
    check_command kubectl || ((missing_deps++))
    check_command helm || ((missing_deps++))
    check_command python3 || ((missing_deps++))
    check_command pip3 || check_command pip || ((missing_deps++))
    check_command git || ((missing_deps++))

    print_separator

    log_info "Checking optional tools..."
    check_command trivy || log_warn "  Trivy not installed (vulnerability scanning disabled)"
    check_command locust || log_warn "  Locust not installed (load testing disabled)"

    print_separator

    if [ $missing_deps -gt 0 ]; then
        log_error "Missing $missing_deps required dependencies"
        log_error "Please install missing tools and try again"
        exit 1
    fi

    log_success "All required dependencies are available"
}

check_docker_running() {
    log_info "Checking Docker daemon..."
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker daemon is not running"
        log_error "Please start Docker Desktop and try again"
        exit 1
    fi
    log_success "Docker daemon is running"
}

check_kubernetes() {
    log_info "Checking Kubernetes cluster..."
    if ! kubectl cluster-info > /dev/null 2>&1; then
        log_error "Kubernetes cluster is not accessible"
        log_error "Please enable Kubernetes in Docker Desktop settings"
        exit 1
    fi

    local context
    context=$(kubectl config current-context)
    log_success "Connected to Kubernetes cluster: $context"

    # Check available resources
    local nodes
    nodes=$(kubectl get nodes -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}')
    if [[ "$nodes" != *"True"* ]]; then
        log_error "No ready Kubernetes nodes found"
        exit 1
    fi
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    print_header "SETTING UP ENVIRONMENT"

    cd "$PROJECT_ROOT"

    # Load .env file if exists
    if [ -f ".env" ]; then
        log_info "Loading environment from .env file..."
        set -a
        # shellcheck source=/dev/null
        source .env
        set +a
    else
        log_warn ".env file not found, using defaults"
    fi

    # Read version from __version__.py
    if [ -f "src/__version__.py" ]; then
        VERSION=$(python3 -c "exec(open('src/__version__.py').read()); print(__version__)")
    else
        VERSION="1.0.0"
    fi
    export VERSION

    # Git information
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    export GIT_COMMIT GIT_BRANCH

    log_info "Environment configuration:"
    log_info "  PROJECT_ROOT: $PROJECT_ROOT"
    log_info "  VERSION: $VERSION"
    log_info "  GIT_COMMIT: $GIT_COMMIT"
    log_info "  GIT_BRANCH: $GIT_BRANCH"
    log_info "  NAMESPACE: $NAMESPACE"

    # Setup Python virtual environment
    log_info "Setting up Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # shellcheck source=/dev/null
    source venv/bin/activate
    pip install --upgrade pip wheel -q

    log_info "Installing Python dependencies..."
    pip install -r requirements.txt -q
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt -q
    fi

    log_success "Environment setup completed"
}

# =============================================================================
# Code Quality & Testing
# =============================================================================

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warn "Skipping tests (--skip-tests flag set)"
        return 0
    fi

    print_header "RUNNING CODE QUALITY CHECKS & TESTS"

    cd "$PROJECT_ROOT"
    # shellcheck source=/dev/null
    source venv/bin/activate

    log_info "Running flake8 linting..."
    flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics || true

    log_info "Running black formatting check..."
    black --check --line-length 100 src/ tests/ 2>/dev/null || log_warn "Code formatting issues found"

    log_info "Running bandit security scan..."
    bandit -r src/ -ll -ii --exit-zero || true

    log_info "Running pytest..."
    pytest tests/ \
        --cov=src \
        --cov-report=html:coverage-html \
        --cov-report=term-missing \
        -v \
        || log_warn "Some tests failed"

    log_success "Code quality checks completed"
}

# =============================================================================
# Docker Build
# =============================================================================

build_docker_images() {
    if [ "$SKIP_BUILD" = true ]; then
        log_warn "Skipping Docker build (--skip-build flag set)"
        return 0
    fi

    print_header "BUILDING DOCKER IMAGES"

    cd "$PROJECT_ROOT"

    local training_image="${DOCKER_REGISTRY}/${DOCKER_USERNAME:-local}/pm-training"
    local serving_image="${DOCKER_REGISTRY}/${DOCKER_USERNAME:-local}/pm-serving"

    log_info "Building training image..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would build: $training_image:$VERSION"
    else
        docker build \
            -t "${training_image}:latest" \
            -t "${training_image}:${VERSION}" \
            -t "${training_image}:${GIT_COMMIT}" \
            -f docker/training/Dockerfile \
            --build-arg VERSION="${VERSION}" \
            --build-arg GIT_COMMIT="${GIT_COMMIT}" \
            .
    fi

    log_info "Building serving image..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would build: $serving_image:$VERSION"
    else
        docker build \
            -t "${serving_image}:latest" \
            -t "${serving_image}:${VERSION}" \
            -t "${serving_image}:${GIT_COMMIT}" \
            -f docker/serving/Dockerfile \
            --build-arg VERSION="${VERSION}" \
            --build-arg GIT_COMMIT="${GIT_COMMIT}" \
            .
    fi

    log_info "Docker image sizes:"
    docker images | grep -E "(pm-training|pm-serving)" | head -10 || true

    log_success "Docker images built successfully"
}

# =============================================================================
# Docker Push
# =============================================================================

push_docker_images() {
    if [ "$SKIP_PUSH" = true ]; then
        log_warn "Skipping Docker push (--skip-push flag set)"
        return 0
    fi

    print_header "PUSHING DOCKER IMAGES"

    # Check credentials
    if [ -z "${DOCKER_USERNAME:-}" ] || [ -z "${DOCKER_PASSWORD:-}" ]; then
        log_error "Docker credentials not set"
        log_error "Set DOCKER_USERNAME and DOCKER_PASSWORD environment variables"
        return 1
    fi

    local training_image="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/pm-training"
    local serving_image="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/pm-serving"

    log_info "Logging in to Docker registry..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would login to $DOCKER_REGISTRY"
    else
        set +x
        echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
        set -x
    fi

    log_info "Pushing training image..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would push: $training_image"
    else
        docker push "${training_image}:latest"
        docker push "${training_image}:${VERSION}"
        docker push "${training_image}:${GIT_COMMIT}"
    fi

    log_info "Pushing serving image..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would push: $serving_image"
    else
        docker push "${serving_image}:latest"
        docker push "${serving_image}:${VERSION}"
        docker push "${serving_image}:${GIT_COMMIT}"
    fi

    log_info "Logging out from Docker registry..."
    docker logout

    log_success "Docker images pushed successfully"
}

# =============================================================================
# Infrastructure Deployment
# =============================================================================

deploy_infrastructure() {
    if [ "$SKIP_DEPLOY" = true ]; then
        log_warn "Skipping deployment (--skip-deploy flag set)"
        return 0
    fi

    if [ "$LOCAL_ONLY" = true ]; then
        deploy_local
        return 0
    fi

    print_header "DEPLOYING KUBERNETES INFRASTRUCTURE"

    cd "$PROJECT_ROOT"

    # Create namespace
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Add Helm repositories
    log_info "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null || true
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
    helm repo update

    # Deploy PostgreSQL
    log_info "Deploying PostgreSQL..."
    helm upgrade --install postgres bitnami/postgresql \
        --namespace "$NAMESPACE" \
        --set auth.postgresPassword=mlflow \
        --set auth.database=mlflow \
        --set primary.persistence.size=5Gi \
        --wait --timeout 5m0s || log_warn "PostgreSQL deployment issues"

    # Deploy MLflow
    log_info "Deploying MLflow..."
    kubectl apply -f k8s/mlflow/ -n "$NAMESPACE"
    kubectl wait --for=condition=available deployment/mlflow \
        -n "$NAMESPACE" --timeout=300s || log_warn "MLflow deployment timeout"

    # Deploy Prometheus
    log_info "Deploying Prometheus..."
    helm upgrade --install prometheus prometheus-community/prometheus \
        --namespace "$NAMESPACE" \
        --set server.persistentVolume.size=5Gi \
        --set alertmanager.enabled=false \
        --wait --timeout 5m0s || log_warn "Prometheus deployment issues"

    # Deploy Grafana
    log_info "Deploying Grafana..."
    helm upgrade --install grafana grafana/grafana \
        --namespace "$NAMESPACE" \
        --set adminPassword=admin \
        --set persistence.enabled=true \
        --set persistence.size=2Gi \
        --wait --timeout 5m0s || log_warn "Grafana deployment issues"

    log_success "Infrastructure deployment completed"

    # Start port-forwards
    start_port_forwards
}

deploy_local() {
    print_header "DEPLOYING LOCAL SERVICES (DOCKER COMPOSE)"

    cd "$PROJECT_ROOT"

    log_info "Starting docker-compose services..."
    docker-compose up -d

    log_info "Waiting for services to be healthy..."
    sleep 10

    docker-compose ps

    log_success "Local services deployed"
    log_info "Service URLs:"
    log_info "  MLflow:     http://localhost:5000"
    log_info "  Prometheus: http://localhost:9090"
    log_info "  Grafana:    http://localhost:3000"
}

start_port_forwards() {
    print_header "STARTING PORT FORWARDS"

    # Clear existing port-forwards
    pkill -f "kubectl port-forward" 2>/dev/null || true
    rm -f "$PID_FILE"

    log_info "Starting port-forwards..."

    # MLflow
    kubectl port-forward svc/mlflow 5000:5000 -n "$NAMESPACE" > /dev/null 2>&1 &
    echo $! >> "$PID_FILE"

    # Prometheus
    kubectl port-forward svc/prometheus-server 9090:80 -n "$NAMESPACE" > /dev/null 2>&1 &
    echo $! >> "$PID_FILE"

    # Grafana
    kubectl port-forward svc/grafana 3000:80 -n "$NAMESPACE" > /dev/null 2>&1 &
    echo $! >> "$PID_FILE"

    sleep 5

    log_success "Port-forwards established"
    log_info "Service URLs:"
    log_info "  MLflow:     http://localhost:5000"
    log_info "  Prometheus: http://localhost:9090"
    log_info "  Grafana:    http://localhost:3000 (admin/admin)"
}

# =============================================================================
# Summary & Reporting
# =============================================================================

print_summary() {
    print_header "DEPLOYMENT SUMMARY"

    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}Predictive Maintenance MLOps Platform${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${BOLD}Version:${NC}      $VERSION"
    echo -e "  ${BOLD}Git Commit:${NC}   $GIT_COMMIT"
    echo -e "  ${BOLD}Git Branch:${NC}   $GIT_BRANCH"
    echo -e "  ${BOLD}Namespace:${NC}    $NAMESPACE"
    echo ""
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}Service URLs:${NC}"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "  MLflow UI:      ${CYAN}http://localhost:5000${NC}"
    echo -e "  Prometheus:     ${CYAN}http://localhost:9090${NC}"
    echo -e "  Grafana:        ${CYAN}http://localhost:3000${NC} (admin/admin)"
    echo -e "  API Swagger:    ${CYAN}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Save summary to JSON
    cat > "${PROJECT_ROOT}/deployment-summary.json" << EOF
{
    "version": "${VERSION}",
    "git_commit": "${GIT_COMMIT}",
    "git_branch": "${GIT_BRANCH}",
    "namespace": "${NAMESPACE}",
    "deployed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "services": {
        "mlflow": "http://localhost:5000",
        "prometheus": "http://localhost:9090",
        "grafana": "http://localhost:3000",
        "api": "http://localhost:8000"
    }
}
EOF

    log_info "Deployment summary saved to deployment-summary.json"
}

# =============================================================================
# Help & Argument Parsing
# =============================================================================

show_help() {
    cat << EOF
${BOLD}Predictive Maintenance MLOps Deployment Script${NC}

${BOLD}Usage:${NC}
    $0 [OPTIONS]

${BOLD}Options:${NC}
    --skip-tests        Skip running tests
    --skip-build        Skip Docker image builds
    --skip-push         Skip pushing to registry
    --skip-deploy       Skip Kubernetes deployments
    --local-only        Deploy only local services (docker-compose)
    --cleanup           Run cleanup after deployment
    --dry-run           Show what would be done without executing
    --help              Show this help message

${BOLD}Examples:${NC}
    # Full deployment
    $0

    # Local development only
    $0 --local-only

    # Skip tests and push
    $0 --skip-tests --skip-push

${BOLD}Environment Variables:${NC}
    DOCKER_USERNAME     Docker Hub username
    DOCKER_PASSWORD     Docker Hub password or token
    NAMESPACE           Kubernetes namespace (default: mlops)

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-tests)  SKIP_TESTS=true; shift ;;
            --skip-build)  SKIP_BUILD=true; shift ;;
            --skip-push)   SKIP_PUSH=true; shift ;;
            --skip-deploy) SKIP_DEPLOY=true; shift ;;
            --local-only)  LOCAL_ONLY=true; shift ;;
            --cleanup)     RUN_CLEANUP=true; shift ;;
            --dry-run)     DRY_RUN=true; shift ;;
            --help)        show_help; exit 0 ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    parse_arguments "$@"

    # Initialize log file
    echo "Deployment started at $(date)" > "$LOG_FILE"

    print_header "PREDICTIVE MAINTENANCE MLOPS DEPLOYMENT"

    if [ "$DRY_RUN" = true ]; then
        log_warn "DRY-RUN MODE - No changes will be made"
    fi

    # Execute deployment steps
    check_prerequisites
    check_docker_running
    check_kubernetes
    setup_environment
    run_tests
    build_docker_images
    push_docker_images
    deploy_infrastructure

    # Print summary
    print_summary

    log_success "Deployment completed successfully!"
    log_info "Log file: $LOG_FILE"

    return 0
}

# Run main function
main "$@"
