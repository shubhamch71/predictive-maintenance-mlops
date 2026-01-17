#!/usr/bin/env bash
# =============================================================================
# Quick Start Script for Local Development
# =============================================================================
#
# This script sets up and starts the local development environment using
# Docker Compose. It handles prerequisites, configuration, and service startup.
#
# Usage:
#   ./scripts/run-local.sh [OPTIONS]
#
# Options:
#   --build       Force rebuild of Docker images
#   --logs        Tail logs after starting
#   --detach      Run in detached mode (default)
#   --clean       Clean up before starting
#   --help        Show this help message
#
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Flags
BUILD=false
TAIL_LOGS=false
DETACH=true
CLEAN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    printf "${BLUE}║${NC} %-60s ${BLUE}║${NC}\n" "$1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_help() {
    cat << EOF
Quick Start Script for Predictive Maintenance MLOps Platform

Usage:
    $0 [OPTIONS]

Options:
    --build       Force rebuild of Docker images
    --logs        Tail logs after starting
    --detach      Run in detached mode (default)
    --clean       Clean up before starting
    --help        Show this help message

Examples:
    # Start all services
    $0

    # Rebuild and start
    $0 --build

    # Start and tail logs
    $0 --logs

    # Clean start
    $0 --clean --build
EOF
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    local missing=0

    # Check Docker
    if command -v docker &> /dev/null; then
        log_info "Docker: $(docker --version)"
    else
        log_error "Docker is not installed"
        ((missing++))
    fi

    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        log_info "Docker Compose: $(docker-compose --version)"
    elif docker compose version &> /dev/null; then
        log_info "Docker Compose: $(docker compose version)"
    else
        log_error "Docker Compose is not installed"
        ((missing++))
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        log_error "Please start Docker Desktop"
        ((missing++))
    fi

    if [ $missing -gt 0 ]; then
        log_error "Missing $missing prerequisites"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

setup_environment() {
    print_header "Setting Up Environment"

    cd "$PROJECT_ROOT"

    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            log_info "Creating .env from .env.example..."
            cp .env.example .env
            log_warn "Please review and update .env with your settings"
        else
            log_warn ".env.example not found, using defaults"
        fi
    else
        log_info ".env file exists"
    fi

    # Create required directories
    mkdir -p data/raw data/processed models logs mlflow-artifacts

    log_success "Environment setup complete"
}

cleanup() {
    print_header "Cleaning Up"

    cd "$PROJECT_ROOT"

    log_info "Stopping existing services..."
    docker-compose down -v --remove-orphans 2>/dev/null || true

    log_info "Removing old containers..."
    docker container prune -f 2>/dev/null || true

    log_success "Cleanup complete"
}

start_services() {
    print_header "Starting Services"

    cd "$PROJECT_ROOT"

    local compose_cmd="docker-compose"
    if ! command -v docker-compose &> /dev/null; then
        compose_cmd="docker compose"
    fi

    local args=""
    if [ "$BUILD" = true ]; then
        args="--build"
        log_info "Building images..."
    fi

    if [ "$DETACH" = true ]; then
        args="$args -d"
    fi

    log_info "Starting docker-compose services..."
    $compose_cmd up $args

    if [ "$DETACH" = true ]; then
        log_info "Waiting for services to be ready..."
        sleep 10

        # Check service health
        log_info "Checking service health..."
        $compose_cmd ps
    fi
}

wait_for_services() {
    print_header "Waiting for Services"

    local services=("localhost:5000" "localhost:8000" "localhost:9090" "localhost:3000")
    local names=("MLflow" "API" "Prometheus" "Grafana")

    for i in "${!services[@]}"; do
        local url="${services[$i]}"
        local name="${names[$i]}"
        local max_attempts=30
        local attempt=1

        log_info "Waiting for $name ($url)..."
        while [ $attempt -le $max_attempts ]; do
            if curl -s "http://$url" > /dev/null 2>&1; then
                log_success "$name is ready"
                break
            fi
            sleep 2
            ((attempt++))
        done

        if [ $attempt -gt $max_attempts ]; then
            log_warn "$name not ready after $max_attempts attempts"
        fi
    done
}

print_urls() {
    print_header "Service URLs"

    echo ""
    echo -e "  ${GREEN}MLflow UI:${NC}      http://localhost:5000"
    echo -e "  ${GREEN}API Swagger:${NC}    http://localhost:8000/docs"
    echo -e "  ${GREEN}Prometheus:${NC}     http://localhost:9090"
    echo -e "  ${GREEN}Grafana:${NC}        http://localhost:3000 (admin/admin)"
    echo ""
    echo -e "  ${BLUE}API Health:${NC}     curl http://localhost:8000/health"
    echo -e "  ${BLUE}Predict:${NC}        python examples/predict.py"
    echo ""
}

tail_logs() {
    print_header "Tailing Logs"

    cd "$PROJECT_ROOT"

    local compose_cmd="docker-compose"
    if ! command -v docker-compose &> /dev/null; then
        compose_cmd="docker compose"
    fi

    $compose_cmd logs -f
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)  BUILD=true; shift ;;
        --logs)   TAIL_LOGS=true; shift ;;
        --detach) DETACH=true; shift ;;
        --clean)  CLEAN=true; shift ;;
        --help)   show_help; exit 0 ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header "PREDICTIVE MAINTENANCE - LOCAL DEVELOPMENT"

    check_prerequisites

    if [ "$CLEAN" = true ]; then
        cleanup
    fi

    setup_environment
    start_services

    if [ "$DETACH" = true ]; then
        wait_for_services
        print_urls
    fi

    if [ "$TAIL_LOGS" = true ]; then
        tail_logs
    fi

    log_success "Local development environment is ready!"
}

main
