#!/bin/bash
# =============================================================================
# Setup Local Kubernetes Environment (Docker Desktop)
# =============================================================================
# This script sets up a complete MLOps environment on Docker Desktop Kubernetes
# including Kubeflow Pipelines, KServe, Argo Workflows, and Prometheus.
#
# Usage: ./scripts/setup_local_k8s.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/setup.log"

# Component versions
KUBEFLOW_VERSION="2.0.0"
KSERVE_VERSION="0.11.0"
ARGO_VERSION="3.5.0"

# =============================================================================
# Color Codes
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
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

print_box() {
    local message="$1"
    local length=${#message}
    local border=$(printf '%*s' "$((length + 4))" | tr ' ' '─')
    echo -e "${CYAN}┌${border}┐${NC}"
    echo -e "${CYAN}│  ${NC}${BOLD}$message${NC}${CYAN}  │${NC}"
    echo -e "${CYAN}└${border}┘${NC}"
}

# =============================================================================
# Prerequisite Check Functions
# =============================================================================
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_docker_desktop() {
    print_header "Checking Docker Desktop"

    # Check if Docker is installed
    if ! check_command docker; then
        log_error "Docker is not installed!"
        echo ""
        echo -e "${YELLOW}Please install Docker Desktop from:${NC}"
        echo "  https://www.docker.com/products/docker-desktop"
        echo ""
        exit 1
    fi
    log_success "Docker CLI found"

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running!"
        echo ""
        echo -e "${YELLOW}Please start Docker Desktop and try again.${NC}"
        echo ""
        exit 1
    fi
    log_success "Docker daemon is running"

    # Check Docker version
    local docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    log_info "Docker version: $docker_version"

    # Check available resources
    local memory_gb=$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{printf "%.1f", $1/1024/1024/1024}')
    local cpus=$(docker info --format '{{.NCPU}}' 2>/dev/null || echo "unknown")

    log_info "Available resources: ${cpus} CPUs, ${memory_gb}GB RAM"

    # Warn if resources are low
    if (( $(echo "$memory_gb < 8" | bc -l 2>/dev/null || echo 0) )); then
        log_warning "Less than 8GB RAM allocated to Docker!"
        echo ""
        echo -e "${YELLOW}Recommended: Allocate at least 8GB RAM to Docker Desktop${NC}"
        echo "Go to: Docker Desktop → Settings → Resources → Memory"
        echo ""
    fi
}

check_kubernetes_enabled() {
    print_header "Checking Kubernetes"

    # Check if kubectl is installed
    if ! check_command kubectl; then
        log_error "kubectl is not installed!"
        echo ""
        echo -e "${YELLOW}kubectl should be included with Docker Desktop.${NC}"
        echo "If not, install it from: https://kubernetes.io/docs/tasks/tools/"
        echo ""
        exit 1
    fi
    log_success "kubectl found"

    # Check if Kubernetes is enabled in Docker Desktop
    if ! kubectl cluster-info &> /dev/null; then
        log_warning "Kubernetes is not enabled or not responding!"
        echo ""
        echo -e "${YELLOW}Please enable Kubernetes in Docker Desktop:${NC}"
        echo ""
        echo "  1. Open Docker Desktop"
        echo "  2. Go to Settings (gear icon)"
        echo "  3. Click on 'Kubernetes' in the left sidebar"
        echo "  4. Check 'Enable Kubernetes'"
        echo "  5. Click 'Apply & Restart'"
        echo "  6. Wait for Kubernetes to start (this may take a few minutes)"
        echo ""
        read -p "Press Enter once Kubernetes is enabled and running..."

        # Retry
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Kubernetes is still not responding. Please check Docker Desktop."
            exit 1
        fi
    fi
    log_success "Kubernetes cluster is running"

    # Set context to docker-desktop
    log "Setting kubectl context to docker-desktop..."
    if kubectl config get-contexts docker-desktop &> /dev/null; then
        kubectl config use-context docker-desktop
        log_success "kubectl context set to docker-desktop"
    else
        log_warning "docker-desktop context not found, using current context"
    fi

    # Display cluster info
    local k8s_version=$(kubectl version --short 2>/dev/null | grep Server | awk '{print $3}' || kubectl version -o json 2>/dev/null | grep -o '"gitVersion": "[^"]*"' | head -1 | cut -d'"' -f4)
    log_info "Kubernetes version: $k8s_version"

    # Check nodes
    local node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    log_info "Cluster nodes: $node_count"
}

# =============================================================================
# Installation Functions
# =============================================================================
create_namespaces() {
    print_header "Creating Namespaces"

    local namespaces=("mlops" "monitoring" "argo")

    for ns in "${namespaces[@]}"; do
        if kubectl get namespace "$ns" &> /dev/null; then
            log_info "Namespace '$ns' already exists"
        else
            kubectl create namespace "$ns"
            log_success "Created namespace: $ns"
        fi
    done
}

install_kubeflow_pipelines() {
    print_header "Installing Kubeflow Pipelines"

    log "Installing Kubeflow Pipelines v${KUBEFLOW_VERSION}..."
    log_info "This may take several minutes..."

    # Check if already installed
    if kubectl get namespace kubeflow &> /dev/null; then
        local existing_pods=$(kubectl get pods -n kubeflow --no-headers 2>/dev/null | wc -l)
        if [ "$existing_pods" -gt 0 ]; then
            log_warning "Kubeflow Pipelines appears to be already installed"
            read -p "Do you want to reinstall? (y/N): " reinstall
            if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
                log_info "Skipping Kubeflow Pipelines installation"
                return 0
            fi
        fi
    fi

    # Install cluster-scoped resources
    log "Installing cluster-scoped resources..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${KUBEFLOW_VERSION}" 2>&1 | tee -a "$LOG_FILE"

    # Wait for CRDs to be established
    log "Waiting for CRDs to be established..."
    sleep 10

    # Install platform-agnostic resources
    log "Installing platform-agnostic resources..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${KUBEFLOW_VERSION}" 2>&1 | tee -a "$LOG_FILE"

    # Wait for pods to be ready
    log "Waiting for Kubeflow pods to be ready (timeout: 5 minutes)..."
    kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=300s 2>/dev/null || {
        log_warning "Some Kubeflow pods may not be ready yet"
        log_info "You can check status with: kubectl get pods -n kubeflow"
    }

    log_success "Kubeflow Pipelines installed"
}

install_kserve() {
    print_header "Installing KServe"

    log "Installing KServe v${KSERVE_VERSION}..."

    # Check if already installed
    if kubectl get crd inferenceservices.serving.kserve.io &> /dev/null; then
        log_warning "KServe CRDs already exist"
        read -p "Do you want to reinstall? (y/N): " reinstall
        if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
            log_info "Skipping KServe installation"
            return 0
        fi
    fi

    # Install KServe
    log "Installing KServe core components..."
    kubectl apply -f "https://github.com/kserve/kserve/releases/download/v${KSERVE_VERSION}/kserve.yaml" 2>&1 | tee -a "$LOG_FILE"

    # Wait for controller to be ready
    log "Waiting for KServe controller..."
    sleep 10

    # Install runtimes
    log "Installing KServe runtimes..."
    kubectl apply -f "https://github.com/kserve/kserve/releases/download/v${KSERVE_VERSION}/kserve-runtimes.yaml" 2>&1 | tee -a "$LOG_FILE"

    log_success "KServe installed"
}

install_argo_workflows() {
    print_header "Installing Argo Workflows"

    log "Installing Argo Workflows v${ARGO_VERSION}..."

    # Ensure namespace exists
    kubectl create namespace argo 2>/dev/null || true

    # Check if already installed
    if kubectl get deployment workflow-controller -n argo &> /dev/null; then
        log_warning "Argo Workflows appears to be already installed"
        read -p "Do you want to reinstall? (y/N): " reinstall
        if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
            log_info "Skipping Argo Workflows installation"
            return 0
        fi
    fi

    # Install Argo Workflows
    kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/v${ARGO_VERSION}/install.yaml" 2>&1 | tee -a "$LOG_FILE"

    # Patch auth mode for local development (no auth required)
    log "Configuring Argo for local development..."
    kubectl patch deployment argo-server -n argo --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--auth-mode=server"}]' 2>/dev/null || true

    # Wait for deployment
    log "Waiting for Argo Workflows to be ready..."
    kubectl wait --for=condition=Available deployment/argo-server -n argo --timeout=120s 2>/dev/null || {
        log_warning "Argo server may not be ready yet"
    }

    log_success "Argo Workflows installed"
}

install_argo_events() {
    print_header "Installing Argo Events"

    log "Installing Argo Events..."

    # Install Argo Events
    kubectl create namespace argo-events 2>/dev/null || true
    kubectl apply -f https://raw.githubusercontent.com/argoproj/argo-events/stable/manifests/install.yaml 2>&1 | tee -a "$LOG_FILE"
    kubectl apply -n argo-events -f https://raw.githubusercontent.com/argoproj/argo-events/stable/manifests/install-validating-webhook.yaml 2>&1 | tee -a "$LOG_FILE" || true

    log_success "Argo Events installed"
}

install_prometheus() {
    print_header "Installing Prometheus Stack"

    # Check if Helm is installed
    if check_command helm; then
        log "Installing Prometheus using Helm..."

        # Add Prometheus Helm repo
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
        helm repo update

        # Install kube-prometheus-stack
        if helm list -n monitoring | grep -q prometheus; then
            log_warning "Prometheus appears to be already installed"
        else
            helm install prometheus prometheus-community/kube-prometheus-stack \
                --namespace monitoring \
                --create-namespace \
                --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
                --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
                --set grafana.adminPassword=admin123 \
                --wait --timeout 5m 2>&1 | tee -a "$LOG_FILE" || {
                log_warning "Helm installation had issues, check logs"
            }
        fi

        log_success "Prometheus Stack installed via Helm"
    else
        log_warning "Helm not found, installing minimal Prometheus..."

        # Apply our custom Prometheus manifests
        if [ -f "$PROJECT_ROOT/k8s/monitoring/prometheus/deployment.yaml" ]; then
            kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/prometheus/configmap.yaml" -n monitoring 2>&1 | tee -a "$LOG_FILE"
            kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/prometheus/deployment.yaml" -n monitoring 2>&1 | tee -a "$LOG_FILE"
            log_success "Custom Prometheus manifests applied"
        else
            log_warning "Prometheus manifests not found, skipping Prometheus installation"
            log_info "You can install Prometheus later using: helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring"
        fi
    fi
}

install_grafana() {
    print_header "Installing Grafana"

    # Check if already installed via Helm (kube-prometheus-stack includes Grafana)
    if kubectl get deployment prometheus-grafana -n monitoring &> /dev/null; then
        log_info "Grafana already installed via Prometheus Stack"
        return 0
    fi

    # Apply our custom Grafana manifests
    if [ -f "$PROJECT_ROOT/k8s/monitoring/grafana/deployment.yaml" ]; then
        kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/grafana/configmap.yaml" -n monitoring 2>&1 | tee -a "$LOG_FILE"
        kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/grafana/deployment.yaml" -n monitoring 2>&1 | tee -a "$LOG_FILE"

        # Apply dashboards
        if [ -f "$PROJECT_ROOT/k8s/monitoring/grafana/dashboards/configmap.yaml" ]; then
            kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/grafana/dashboards/configmap.yaml" -n monitoring 2>&1 | tee -a "$LOG_FILE"
        fi

        log_success "Custom Grafana manifests applied"
    else
        log_warning "Grafana manifests not found"
    fi
}

# =============================================================================
# Verification Functions
# =============================================================================
verify_installation() {
    print_header "Verifying Installation"

    echo ""
    log "Checking all pods across namespaces..."
    echo ""

    # Define namespaces to check
    local namespaces=("kube-system" "kubeflow" "kserve" "argo" "mlops" "monitoring")

    for ns in "${namespaces[@]}"; do
        if kubectl get namespace "$ns" &> /dev/null; then
            echo -e "${BOLD}Namespace: $ns${NC}"
            kubectl get pods -n "$ns" --no-headers 2>/dev/null | while read line; do
                local name=$(echo "$line" | awk '{print $1}')
                local ready=$(echo "$line" | awk '{print $2}')
                local status=$(echo "$line" | awk '{print $3}')

                if [[ "$status" == "Running" || "$status" == "Completed" ]]; then
                    echo -e "  ${GREEN}✓${NC} $name ($status)"
                elif [[ "$status" == "Pending" || "$status" == "ContainerCreating" ]]; then
                    echo -e "  ${YELLOW}○${NC} $name ($status)"
                else
                    echo -e "  ${RED}✗${NC} $name ($status)"
                fi
            done
            echo ""
        fi
    done
}

print_summary() {
    print_header "Setup Complete!"

    echo -e "${BOLD}${GREEN}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   MLOps Platform Setup Complete!                              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo -e "${BOLD}Installed Components:${NC}"
    echo "  • Kubeflow Pipelines v${KUBEFLOW_VERSION}"
    echo "  • KServe v${KSERVE_VERSION}"
    echo "  • Argo Workflows v${ARGO_VERSION}"
    echo "  • Prometheus & Grafana"
    echo ""

    echo -e "${BOLD}Namespaces Created:${NC}"
    echo "  • mlops        - Main application namespace"
    echo "  • monitoring   - Prometheus, Grafana"
    echo "  • argo         - Argo Workflows"
    echo "  • kubeflow     - Kubeflow Pipelines"
    echo ""

    echo -e "${BOLD}Next Steps:${NC}"
    echo "  1. Deploy MLflow:       ./scripts/deploy_mlflow.sh"
    echo "  2. Run training:        ./scripts/run_training_pipeline.sh"
    echo "  3. Deploy models:       ./scripts/deploy_models.sh"
    echo "  4. Access all services: ./scripts/access_all_services.sh"
    echo ""

    echo -e "${BOLD}Resource Recommendations:${NC}"
    echo -e "  ${YELLOW}For optimal performance, allocate to Docker Desktop:${NC}"
    echo "  • Memory: 12GB (minimum 8GB)"
    echo "  • CPUs: 6 (minimum 4)"
    echo "  • Disk: 50GB"
    echo ""
    echo "  Configure at: Docker Desktop → Settings → Resources"
    echo ""

    echo -e "${BOLD}Useful Commands:${NC}"
    echo "  kubectl get pods -A                    # View all pods"
    echo "  kubectl get svc -A                     # View all services"
    echo "  kubectl logs -f <pod-name> -n <ns>     # View pod logs"
    echo "  kubectl describe pod <pod-name> -n <ns> # Debug pod issues"
    echo ""

    log_success "Setup log saved to: $LOG_FILE"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    clear

    print_box "MLOps Platform Setup for Docker Desktop Kubernetes"

    echo ""
    log "Starting setup at $(date)"
    log "Log file: $LOG_FILE"
    echo ""

    # Clear previous log
    > "$LOG_FILE"

    # Prerequisite checks
    check_docker_desktop
    check_kubernetes_enabled

    # Create namespaces
    create_namespaces

    # Install components
    install_kubeflow_pipelines
    install_kserve
    install_argo_workflows
    install_argo_events
    install_prometheus
    install_grafana

    # Verify installation
    verify_installation

    # Print summary
    print_summary

    log "Setup completed at $(date)"
}

# Run main function
main "$@"
