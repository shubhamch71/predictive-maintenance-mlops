#!/bin/bash
# =============================================================================
# Stop All Services - Kill Port Forwards
# =============================================================================
# Stops all port-forward processes started by access_all_services.sh
#
# Usage: ./scripts/stop_all_services.sh
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/.port-forward-pids"

# Additional PID files that might exist
ADDITIONAL_PID_FILES=(
    "$PROJECT_ROOT/.mlflow-port-forward.pid"
    "$PROJECT_ROOT/.kubeflow-port-forward.pid"
    "$PROJECT_ROOT/.model-port-forward.pid"
)

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

log_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Stop Functions
# =============================================================================
stop_from_pid_file() {
    local pid_file=$1
    local stopped=0

    if [ -f "$pid_file" ]; then
        echo "Reading PIDs from: $pid_file"

        while IFS=: read -r pid name port || [ -n "$pid" ]; do
            if [ -n "$pid" ]; then
                # Handle both formats: "pid:name:port" and just "pid"
                if [ -z "$name" ]; then
                    name="unknown"
                fi

                if kill -0 "$pid" 2>/dev/null; then
                    kill "$pid" 2>/dev/null
                    if [ $? -eq 0 ]; then
                        log_success "Stopped $name (PID: $pid)"
                        ((stopped++))
                    else
                        log_warning "Could not stop $name (PID: $pid)"
                    fi
                else
                    log_info "Process already stopped: $name (PID: $pid)"
                fi
            fi
        done < "$pid_file"

        rm -f "$pid_file"
        return $stopped
    else
        return 0
    fi
}

stop_kubectl_port_forwards() {
    echo ""
    echo "Stopping any remaining kubectl port-forward processes..."

    # Find and kill all kubectl port-forward processes
    local pids=$(pgrep -f "kubectl port-forward" 2>/dev/null)

    if [ -n "$pids" ]; then
        for pid in $pids; do
            # Get the command for logging
            local cmd=$(ps -p "$pid" -o args= 2>/dev/null | head -c 80)
            kill "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                log_success "Stopped: $cmd..."
            fi
        done
    else
        log_info "No kubectl port-forward processes found"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║                   Stop All Services                           ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    local total_stopped=0

    # Stop from main PID file
    if [ -f "$PID_FILE" ]; then
        echo "Stopping services from main PID file..."
        stop_from_pid_file "$PID_FILE"
        ((total_stopped += $?))
    fi

    # Stop from additional PID files
    for pid_file in "${ADDITIONAL_PID_FILES[@]}"; do
        if [ -f "$pid_file" ]; then
            stop_from_pid_file "$pid_file"
            ((total_stopped += $?))
        fi
    done

    # Stop any remaining kubectl port-forwards
    stop_kubectl_port_forwards

    # Verify ports are free
    echo ""
    echo "Verifying ports are free..."

    local ports=(5000 8080 3000 9090 8000 2746)
    local all_free=true

    for port in "${ports[@]}"; do
        if nc -z localhost "$port" 2>/dev/null; then
            log_warning "Port $port is still in use"
            all_free=false
        fi
    done

    if [ "$all_free" = true ]; then
        log_success "All ports are free"
    fi

    # Summary
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   All Services Stopped                        ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${BOLD}To restart services:${NC}"
    echo "  ./scripts/access_all_services.sh"
    echo ""
}

# Run main function
main "$@"
