#!/usr/bin/env bash
# =============================================================================
# Comprehensive Cleanup Script for Predictive Maintenance MLOps Platform
# =============================================================================
#
# This script cleans up all resources created by the deployment:
# - Port-forward processes
# - Kubernetes deployments
# - Docker containers and images
# - Temporary files
#
# Usage:
#   ./jenkins/cleanup.sh [OPTIONS]
#
# Options:
#   --all               Clean everything (Kubernetes, Docker, files)
#   --k8s               Clean only Kubernetes resources
#   --docker            Clean only Docker resources
#   --files             Clean only temporary files
#   --ports             Kill only port-forward processes
#   --force             Skip confirmation prompts
#   --keep-logs         Preserve log files
#   --help              Show this help message
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${SCRIPT_DIR}/port-forwards.pids"
NAMESPACE="${NAMESPACE:-mlops}"

# Cleanup flags
CLEAN_ALL=false
CLEAN_K8S=false
CLEAN_DOCKER=false
CLEAN_FILES=false
CLEAN_PORTS=false
FORCE=false
KEEP_LOGS=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

print_header() {
    local title="$1"
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    printf "${CYAN}║${NC} ${BOLD}%-60s${NC} ${CYAN}║${NC}\n" "$title"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
# Confirmation Function
# =============================================================================

confirm() {
    local prompt="$1"
    local default="${2:-n}"

    if [ "$FORCE" = true ]; then
        return 0
    fi

    local yn
    if [ "$default" = "y" ]; then
        read -r -p "$prompt [Y/n]: " yn
        yn="${yn:-y}"
    else
        read -r -p "$prompt [y/N]: " yn
        yn="${yn:-n}"
    fi

    case "$yn" in
        [Yy]* ) return 0 ;;
        * ) return 1 ;;
    esac
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup_port_forwards() {
    print_header "CLEANING UP PORT FORWARDS"

    local killed=0

    # Kill processes from PID file
    if [ -f "$PID_FILE" ]; then
        log_info "Reading PID file: $PID_FILE"
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                log_info "Killing process: $pid"
                kill "$pid" 2>/dev/null || true
                ((killed++))
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    # Find and kill kubectl port-forward processes
    log_info "Looking for kubectl port-forward processes..."
    local pids
    pids=$(pgrep -f "kubectl port-forward" 2>/dev/null || echo "")

    if [ -n "$pids" ]; then
        for pid in $pids; do
            log_info "Killing kubectl port-forward process: $pid"
            kill "$pid" 2>/dev/null || true
            ((killed++))
        done
    fi

    if [ $killed -gt 0 ]; then
        log_success "Killed $killed port-forward process(es)"
    else
        log_info "No port-forward processes found"
    fi
}

cleanup_kubernetes() {
    print_header "CLEANING UP KUBERNETES RESOURCES"

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl not found, skipping Kubernetes cleanup"
        return 0
    fi

    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_warn "Kubernetes cluster not accessible, skipping cleanup"
        return 0
    fi

    # List what will be deleted
    log_info "Resources in namespace '$NAMESPACE':"
    kubectl get all -n "$NAMESPACE" 2>/dev/null || true

    if ! confirm "Delete all resources in namespace '$NAMESPACE'?"; then
        log_info "Skipping Kubernetes cleanup"
        return 0
    fi

    # Delete namespace (this removes all resources within it)
    log_info "Deleting namespace: $NAMESPACE"
    kubectl delete namespace "$NAMESPACE" --grace-period=30 --ignore-not-found=true

    # Delete Kubeflow pipelines if present
    if kubectl get namespace kubeflow &> /dev/null; then
        if confirm "Delete Kubeflow namespace?"; then
            log_info "Deleting Kubeflow namespace..."
            kubectl delete namespace kubeflow --grace-period=30 --ignore-not-found=true
        fi
    fi

    # Clean up Helm releases
    log_info "Cleaning up Helm releases..."
    for release in postgres prometheus grafana; do
        helm uninstall "$release" -n "$NAMESPACE" 2>/dev/null || true
    done

    log_success "Kubernetes cleanup completed"
}

cleanup_docker_compose() {
    print_header "CLEANING UP DOCKER COMPOSE"

    cd "$PROJECT_ROOT"

    if [ ! -f "docker-compose.yml" ]; then
        log_warn "docker-compose.yml not found, skipping"
        return 0
    fi

    # Show running containers
    log_info "Current Docker Compose services:"
    docker-compose ps 2>/dev/null || true

    if ! confirm "Stop and remove Docker Compose services?"; then
        log_info "Skipping Docker Compose cleanup"
        return 0
    fi

    log_info "Stopping Docker Compose services..."
    docker-compose down -v --remove-orphans 2>/dev/null || true

    log_success "Docker Compose cleanup completed"
}

cleanup_docker_images() {
    print_header "CLEANING UP DOCKER IMAGES"

    # List project images
    log_info "Project Docker images:"
    docker images | grep -E "(pm-training|pm-serving|predictive-maintenance)" | head -20 || true

    if ! confirm "Remove project Docker images?"; then
        log_info "Skipping Docker image cleanup"
        return 0
    fi

    log_info "Removing project Docker images..."

    # Remove project images
    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(pm-training|pm-serving|predictive-maintenance)" || echo "")

    if [ -n "$images" ]; then
        for image in $images; do
            log_info "Removing: $image"
            docker rmi "$image" 2>/dev/null || true
        done
    fi

    # Clean dangling images
    if confirm "Remove dangling Docker images?"; then
        log_info "Removing dangling images..."
        docker image prune -f
    fi

    # Clean unused volumes
    if confirm "Remove unused Docker volumes?"; then
        log_info "Removing unused volumes..."
        docker volume prune -f
    fi

    log_success "Docker image cleanup completed"
}

cleanup_temporary_files() {
    print_header "CLEANING UP TEMPORARY FILES"

    cd "$PROJECT_ROOT"

    log_info "Files that will be cleaned:"
    echo "  - Python cache (__pycache__, *.pyc, *.pyo)"
    echo "  - Pytest cache (.pytest_cache)"
    echo "  - Coverage files (.coverage, htmlcov, coverage.xml)"
    echo "  - Build artifacts (dist, build, *.egg-info)"
    echo "  - Temporary files (*.tmp, *.temp)"
    echo "  - IDE files (.idea, .vscode - excluding settings)"

    if ! confirm "Clean temporary files?"; then
        log_info "Skipping file cleanup"
        return 0
    fi

    log_info "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true

    log_info "Removing pytest cache..."
    rm -rf .pytest_cache 2>/dev/null || true

    log_info "Removing coverage files..."
    rm -rf .coverage htmlcov coverage.xml coverage-html 2>/dev/null || true

    log_info "Removing build artifacts..."
    rm -rf dist build *.egg-info 2>/dev/null || true

    log_info "Removing test results..."
    rm -rf test-results.xml bandit-report.json 2>/dev/null || true

    log_info "Removing temporary files..."
    find . -type f \( -name "*.tmp" -o -name "*.temp" \) -delete 2>/dev/null || true

    # Clean logs (optional)
    if [ "$KEEP_LOGS" = false ]; then
        if confirm "Remove log files?"; then
            log_info "Removing log files..."
            rm -f jenkins/deploy.log 2>/dev/null || true
            rm -f deployment-summary.json 2>/dev/null || true
        fi
    else
        log_info "Keeping log files (--keep-logs flag set)"
    fi

    log_success "Temporary file cleanup completed"
}

cleanup_virtual_environment() {
    print_header "CLEANING UP VIRTUAL ENVIRONMENT"

    cd "$PROJECT_ROOT"

    if [ ! -d "venv" ]; then
        log_info "No virtual environment found"
        return 0
    fi

    if ! confirm "Remove Python virtual environment?"; then
        log_info "Skipping virtual environment cleanup"
        return 0
    fi

    log_info "Removing virtual environment..."
    rm -rf venv

    log_success "Virtual environment removed"
}

# =============================================================================
# Summary
# =============================================================================

print_summary() {
    print_header "CLEANUP SUMMARY"

    log_success "Cleanup completed successfully!"
    echo ""

    # Check remaining resources
    echo "Remaining resources:"
    echo ""

    echo "  Kubernetes namespaces:"
    kubectl get namespaces 2>/dev/null | grep -E "(mlops|kubeflow)" || echo "    None related to project"

    echo ""
    echo "  Docker containers:"
    docker ps -a --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -E "(mlops|mlflow|prometheus|grafana)" || echo "    None related to project"

    echo ""
    echo "  Docker images:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null | grep -E "(pm-training|pm-serving)" || echo "    None related to project"

    echo ""
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << EOF
${BOLD}Cleanup Script for Predictive Maintenance MLOps Platform${NC}

${YELLOW}Usage:${NC}
    $0 [OPTIONS]

${YELLOW}Options:${NC}
    --all               Clean everything (Kubernetes, Docker, files)
    --k8s               Clean only Kubernetes resources
    --docker            Clean only Docker resources
    --files             Clean only temporary files
    --ports             Kill only port-forward processes
    --force             Skip confirmation prompts
    --keep-logs         Preserve log files
    --help              Show this help message

${YELLOW}Examples:${NC}
    # Interactive full cleanup
    $0 --all

    # Force cleanup without prompts
    $0 --all --force

    # Clean only Kubernetes
    $0 --k8s

    # Kill port-forwards only
    $0 --ports

${YELLOW}Environment Variables:${NC}
    NAMESPACE           Kubernetes namespace to clean (default: mlops)

EOF
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)       CLEAN_ALL=true; shift ;;
            --k8s)       CLEAN_K8S=true; shift ;;
            --docker)    CLEAN_DOCKER=true; shift ;;
            --files)     CLEAN_FILES=true; shift ;;
            --ports)     CLEAN_PORTS=true; shift ;;
            --force)     FORCE=true; shift ;;
            --keep-logs) KEEP_LOGS=true; shift ;;
            --help|-h)   show_help; exit 0 ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # If no specific cleanup selected, show interactive menu
    if [ "$CLEAN_ALL" = false ] && [ "$CLEAN_K8S" = false ] && \
       [ "$CLEAN_DOCKER" = false ] && [ "$CLEAN_FILES" = false ] && \
       [ "$CLEAN_PORTS" = false ]; then
        CLEAN_ALL=true
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_arguments "$@"

    print_header "MLOPS PLATFORM CLEANUP"

    log_info "Namespace: $NAMESPACE"
    log_info "Force mode: $FORCE"
    log_info "Keep logs: $KEEP_LOGS"

    if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_PORTS" = true ]; then
        cleanup_port_forwards
    fi

    if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_DOCKER" = true ]; then
        cleanup_docker_compose
    fi

    if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_K8S" = true ]; then
        cleanup_kubernetes
    fi

    if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_DOCKER" = true ]; then
        cleanup_docker_images
    fi

    if [ "$CLEAN_ALL" = true ] || [ "$CLEAN_FILES" = true ]; then
        cleanup_temporary_files
    fi

    if [ "$CLEAN_ALL" = true ]; then
        if confirm "Remove Python virtual environment?"; then
            cleanup_virtual_environment
        fi
    fi

    print_summary
}

main "$@"
