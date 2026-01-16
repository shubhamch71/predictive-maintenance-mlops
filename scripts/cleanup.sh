#!/bin/bash
# =============================================================================
# Complete Cleanup Script
# =============================================================================
# Removes all deployed resources and cleans up the environment.
#
# Usage: ./scripts/cleanup.sh [--all] [--keep-data]
#   --all: Remove everything including Docker images
#   --keep-data: Keep raw data files
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
    echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"
}

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

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}=============================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}=============================================${NC}"
    echo ""
}

# =============================================================================
# Parse Arguments
# =============================================================================
REMOVE_ALL=false
KEEP_DATA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            REMOVE_ALL=true
            shift
            ;;
        --keep-data)
            KEEP_DATA=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# =============================================================================
# Confirmation
# =============================================================================
confirm_cleanup() {
    echo ""
    echo -e "${BOLD}${RED}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${RED}║                    WARNING: CLEANUP                           ║${NC}"
    echo -e "${BOLD}${RED}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "This will remove:"
    echo "  • All Kubernetes namespaces (mlops, monitoring, argo, kubeflow)"
    echo "  • All deployed applications and data"
    echo "  • Port-forward processes"
    echo "  • Generated files and reports"

    if [ "$REMOVE_ALL" = true ]; then
        echo "  • Local Docker images"
        echo "  • Local Docker registry"
    fi

    if [ "$KEEP_DATA" = false ]; then
        echo "  • Processed data files"
    fi

    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
}

# =============================================================================
# Stop Port Forwards
# =============================================================================
stop_port_forwards() {
    print_header "Stopping Port Forwards"

    if [ -f "$SCRIPT_DIR/stop_all_services.sh" ]; then
        bash "$SCRIPT_DIR/stop_all_services.sh" 2>/dev/null || true
    fi

    # Kill any remaining kubectl port-forward processes
    pkill -f "kubectl port-forward" 2>/dev/null || true

    log_success "Port forwards stopped"
}

# =============================================================================
# Delete Kubernetes Resources
# =============================================================================
delete_kubernetes_resources() {
    print_header "Deleting Kubernetes Resources"

    # Check if kubectl is available
    if ! command -v kubectl &>/dev/null; then
        log_warning "kubectl not found, skipping Kubernetes cleanup"
        return
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        log_warning "Cannot connect to Kubernetes cluster, skipping"
        return
    fi

    # Namespaces to delete
    local namespaces=("mlops" "monitoring" "argo" "kubeflow" "argo-events" "kserve")

    for ns in "${namespaces[@]}"; do
        if kubectl get namespace "$ns" &>/dev/null; then
            log "Deleting namespace: $ns"

            # Delete all resources in namespace first (faster)
            kubectl delete all --all -n "$ns" --timeout=60s 2>/dev/null || true

            # Delete PVCs
            kubectl delete pvc --all -n "$ns" --timeout=60s 2>/dev/null || true

            # Delete the namespace
            kubectl delete namespace "$ns" --grace-period=10 --timeout=120s 2>/dev/null || {
                log_warning "Namespace $ns deletion timed out, forcing..."
                kubectl delete namespace "$ns" --force --grace-period=0 2>/dev/null || true
            }

            log_success "Deleted namespace: $ns"
        else
            log_info "Namespace $ns does not exist"
        fi
    done

    # Delete cluster-scoped resources
    log "Cleaning up cluster-scoped resources..."

    # Delete CRDs if they exist
    kubectl delete crd inferenceservices.serving.kserve.io 2>/dev/null || true
    kubectl delete crd trainedmodels.serving.kserve.io 2>/dev/null || true
    kubectl delete crd workflows.argoproj.io 2>/dev/null || true
    kubectl delete crd eventsources.argoproj.io 2>/dev/null || true
    kubectl delete crd sensors.argoproj.io 2>/dev/null || true

    # Delete ClusterRoles and ClusterRoleBindings
    kubectl delete clusterrole prometheus 2>/dev/null || true
    kubectl delete clusterrolebinding prometheus 2>/dev/null || true
    kubectl delete clusterrole argo-workflow-cluster-role 2>/dev/null || true
    kubectl delete clusterrolebinding argo-workflow-cluster-rolebinding 2>/dev/null || true

    log_success "Kubernetes resources cleaned up"
}

# =============================================================================
# Remove Docker Resources
# =============================================================================
remove_docker_resources() {
    print_header "Removing Docker Resources"

    if ! command -v docker &>/dev/null; then
        log_warning "Docker not found, skipping Docker cleanup"
        return
    fi

    if [ "$REMOVE_ALL" = true ]; then
        # Stop and remove local registry
        if docker ps -a --format '{{.Names}}' | grep -q '^registry$'; then
            log "Stopping local registry..."
            docker stop registry 2>/dev/null || true
            docker rm registry 2>/dev/null || true
            log_success "Local registry removed"
        fi

        # Remove project images
        log "Removing project Docker images..."

        local images=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E "(predictive-maintenance|pm-serving|pm-training)" 2>/dev/null || true)

        if [ -n "$images" ]; then
            echo "$images" | while read -r image; do
                docker rmi "$image" 2>/dev/null && log_success "Removed: $image" || true
            done
        else
            log_info "No project images found"
        fi

        # Remove dangling images
        log "Removing dangling images..."
        docker image prune -f 2>/dev/null || true
    else
        log_info "Skipping Docker image removal (use --all to include)"
    fi
}

# =============================================================================
# Clean Local Files
# =============================================================================
clean_local_files() {
    print_header "Cleaning Local Files"

    # Files to remove
    local files_to_remove=(
        "$PROJECT_ROOT/.port-forward-pids"
        "$PROJECT_ROOT/.mlflow-port-forward.pid"
        "$PROJECT_ROOT/.kubeflow-port-forward.pid"
        "$PROJECT_ROOT/.model-port-forward.pid"
        "$PROJECT_ROOT/.current_run_id"
        "$PROJECT_ROOT/setup.log"
        "$PROJECT_ROOT/deploy_mlflow.log"
        "$PROJECT_ROOT/deploy_models.log"
        "$PROJECT_ROOT/training_pipeline.log"
        "$PROJECT_ROOT/drift_report.json"
        "$PROJECT_ROOT/dashboard.html"
        "$PROJECT_ROOT/mlflow-access.html"
        "$PROJECT_ROOT/pipeline.yaml"
        "$PROJECT_ROOT/test-prediction.sh"
    )

    for file in "${files_to_remove[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            log_success "Removed: $(basename "$file")"
        fi
    done

    # Directories to clean
    if [ "$KEEP_DATA" = false ]; then
        local dirs_to_clean=(
            "$PROJECT_ROOT/data/processed"
            "$PROJECT_ROOT/data/features"
            "$PROJECT_ROOT/models"
            "$PROJECT_ROOT/mlruns"
        )

        for dir in "${dirs_to_clean[@]}"; do
            if [ -d "$dir" ]; then
                rm -rf "$dir"
                log_success "Removed: $(basename "$dir")/"
            fi
        done
    else
        log_info "Keeping data files (--keep-data specified)"
    fi

    # Clean Python cache
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    log_success "Python cache cleaned"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   Cleanup Complete!                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${BOLD}Cleaned:${NC}"
    echo "  • Kubernetes namespaces and resources"
    echo "  • Port-forward processes"
    echo "  • Generated files and logs"

    if [ "$REMOVE_ALL" = true ]; then
        echo "  • Docker images and registry"
    fi

    if [ "$KEEP_DATA" = false ]; then
        echo "  • Processed data and models"
    fi

    echo ""
    echo -e "${BOLD}To set up again:${NC}"
    echo "  1. ./scripts/setup_local_k8s.sh"
    echo "  2. ./scripts/deploy_mlflow.sh"
    echo "  3. ./scripts/run_training_pipeline.sh"
    echo "  4. ./scripts/deploy_models.sh"
    echo "  5. ./scripts/access_all_services.sh"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    clear

    echo ""
    echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║                   MLOps Platform Cleanup                      ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Confirm cleanup
    confirm_cleanup

    # Run cleanup steps
    stop_port_forwards
    delete_kubernetes_resources
    remove_docker_resources
    clean_local_files
    print_summary
}

# Run main function
main "$@"
