#!/usr/bin/env bash
# =============================================================================
# Development Environment Setup Script
# =============================================================================
#
# This script sets up the complete development environment including:
# - Python virtual environment
# - All dependencies
# - Pre-commit hooks
# - Sample data download
#
# Usage:
#   ./scripts/setup-dev.sh
#
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="3.11"
VENV_DIR="${PROJECT_ROOT}/venv"

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

check_python_version() {
    print_header "Checking Python Version"

    # Check for python3
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        log_error "Please install Python 3.11 or higher"
        exit 1
    fi

    # Get version
    local version
    version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    log_info "Python version: $version"

    # Check minimum version (3.9)
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)

    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 9 ]); then
        log_error "Python 3.9+ is required (found $version)"
        exit 1
    fi

    log_success "Python version $version is compatible"
}

create_virtual_environment() {
    print_header "Creating Virtual Environment"

    cd "$PROJECT_ROOT"

    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment already exists"
        read -p "Recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi

    log_info "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"

    log_success "Virtual environment created"
}

install_dependencies() {
    print_header "Installing Dependencies"

    cd "$PROJECT_ROOT"

    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    log_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools

    log_info "Installing production dependencies..."
    pip install -r requirements.txt

    log_info "Installing development dependencies..."
    pip install -r requirements-dev.txt

    if [ -f "requirements-k8s.txt" ]; then
        log_info "Installing Kubernetes dependencies..."
        pip install -r requirements-k8s.txt
    fi

    log_success "All dependencies installed"
}

setup_pre_commit() {
    print_header "Setting Up Pre-commit Hooks"

    cd "$PROJECT_ROOT"

    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    # Create pre-commit config if it doesn't exist
    if [ ! -f ".pre-commit-config.yaml" ]; then
        log_info "Creating pre-commit configuration..."
        cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: ['--line-length=100']

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ['--profile=black']

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503']

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ['-r', 'src/', '-ll']
        exclude: tests/
EOF
    fi

    log_info "Installing pre-commit hooks..."
    pre-commit install

    log_success "Pre-commit hooks installed"
}

setup_environment_file() {
    print_header "Setting Up Environment File"

    cd "$PROJECT_ROOT"

    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            log_info "Creating .env from .env.example..."
            cp .env.example .env
            log_warn "Please update .env with your configuration"
        fi
    else
        log_info ".env file already exists"
    fi
}

create_directories() {
    print_header "Creating Project Directories"

    cd "$PROJECT_ROOT"

    local dirs=(
        "data/raw"
        "data/processed"
        "models"
        "logs"
        "mlflow-artifacts"
        "notebooks"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_info "Creating $dir/"
            mkdir -p "$dir"
        fi
    done

    log_success "Project directories created"
}

download_sample_data() {
    print_header "Downloading Sample Data"

    cd "$PROJECT_ROOT"

    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    if [ -f "data/download_data.py" ]; then
        log_info "Running data download script..."
        python data/download_data.py || log_warn "Data download failed - you may need to download manually"
    else
        log_warn "Data download script not found"
        log_info "Please download the NASA Turbofan dataset manually"
        log_info "See: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository"
    fi
}

verify_installation() {
    print_header "Verifying Installation"

    cd "$PROJECT_ROOT"

    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    log_info "Checking key packages..."

    local packages=("pandas" "numpy" "sklearn" "tensorflow" "mlflow" "fastapi" "pytest")
    local failed=0

    for pkg in "${packages[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            log_success "  $pkg: OK"
        else
            log_error "  $pkg: FAILED"
            ((failed++))
        fi
    done

    if [ $failed -gt 0 ]; then
        log_error "$failed packages failed to import"
        exit 1
    fi

    log_success "All packages verified"
}

print_next_steps() {
    print_header "Next Steps"

    echo ""
    echo "Development environment is ready!"
    echo ""
    echo "To activate the virtual environment:"
    echo -e "  ${GREEN}source venv/bin/activate${NC}"
    echo ""
    echo "Common commands:"
    echo -e "  ${BLUE}pytest tests/${NC}                    # Run tests"
    echo -e "  ${BLUE}black src/ tests/${NC}                # Format code"
    echo -e "  ${BLUE}flake8 src/ tests/${NC}               # Lint code"
    echo -e "  ${BLUE}python examples/predict.py${NC}       # Run prediction example"
    echo ""
    echo "To start local services:"
    echo -e "  ${GREEN}./scripts/run-local.sh${NC}"
    echo ""
    echo "Documentation:"
    echo -e "  ${BLUE}docs/JENKINS_SETUP.md${NC}            # Jenkins CI/CD setup"
    echo -e "  ${BLUE}docs/DOCKER_HUB_SETUP.md${NC}         # Docker Hub authentication"
    echo -e "  ${BLUE}CONTRIBUTING.md${NC}                  # Contribution guidelines"
    echo ""
}

# Main execution
main() {
    print_header "DEVELOPMENT ENVIRONMENT SETUP"

    check_python_version
    create_virtual_environment
    install_dependencies
    setup_pre_commit
    setup_environment_file
    create_directories
    download_sample_data
    verify_installation
    print_next_steps

    log_success "Setup completed successfully!"
}

main
