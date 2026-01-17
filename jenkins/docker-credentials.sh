#!/usr/bin/env bash
# =============================================================================
# Secure Docker Authentication Script
# =============================================================================
#
# This script handles secure Docker Hub authentication using access tokens.
# It is designed to be called from Jenkins pipelines or CI/CD systems.
#
# SECURITY NOTES:
# - Always use access tokens, never passwords
# - Tokens should be passed as environment variables or parameters
# - Never echo or log credentials
# - Use --password-stdin to avoid command-line exposure
#
# Usage:
#   # Method 1: Pass credentials as parameters
#   ./docker-credentials.sh login <username> <token>
#   ./docker-credentials.sh logout
#
#   # Method 2: Use environment variables
#   export DOCKER_USERNAME="your-username"
#   export DOCKER_PASSWORD="your-access-token"
#   ./docker-credentials.sh login
#
#   # Method 3: Read from file (for local development)
#   ./docker-credentials.sh login --from-env-file .env
#
# Exit Codes:
#   0 - Success
#   1 - Authentication failed
#   2 - Missing required parameters
#   3 - Docker daemon not running
#
# Examples:
#   # Login with parameters
#   ./docker-credentials.sh login myuser dckr_pat_xxxxxxxxxxxx
#
#   # Login with environment variables (Jenkins style)
#   withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials',
#                   usernameVariable: 'DOCKER_USERNAME',
#                   passwordVariable: 'DOCKER_PASSWORD')]) {
#       sh './jenkins/docker-credentials.sh login'
#   }
#
#   # Logout
#   ./docker-credentials.sh logout
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_NAME="$(basename "$0")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"

# Colors (disabled in non-interactive mode)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

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

# =============================================================================
# Utility Functions
# =============================================================================

check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker daemon is not running"
        log_error "Please start Docker and try again"
        exit 3
    fi
}

validate_credentials() {
    local username="$1"
    local password="$2"

    if [ -z "$username" ]; then
        log_error "Docker username is required"
        return 1
    fi

    if [ -z "$password" ]; then
        log_error "Docker password/token is required"
        return 1
    fi

    # Basic validation - token should be non-empty and not contain spaces
    if [[ "$password" =~ [[:space:]] ]]; then
        log_error "Password/token contains invalid characters"
        return 1
    fi

    return 0
}

# =============================================================================
# Authentication Functions
# =============================================================================

docker_login() {
    local username="${1:-${DOCKER_USERNAME:-}}"
    local password="${2:-${DOCKER_PASSWORD:-}}"
    local registry="${3:-${DOCKER_REGISTRY}}"

    # Validate credentials
    if ! validate_credentials "$username" "$password"; then
        log_error "Missing or invalid credentials"
        echo ""
        echo "Usage: $SCRIPT_NAME login <username> <token>"
        echo "   or: Set DOCKER_USERNAME and DOCKER_PASSWORD environment variables"
        exit 2
    fi

    # Check Docker daemon
    check_docker

    log_info "Authenticating with Docker registry: $registry"

    # CRITICAL: Disable command echoing before handling credentials
    set +x

    # Use --password-stdin to avoid exposing token in process list
    if echo "$password" | docker login -u "$username" --password-stdin "$registry" 2>/dev/null; then
        log_success "Successfully authenticated to Docker registry"

        # Verify login
        local logged_in_user
        logged_in_user=$(docker info 2>/dev/null | grep -i "Username:" | awk '{print $2}' || echo "")

        if [ -n "$logged_in_user" ]; then
            log_info "Logged in as: $logged_in_user"
        fi

        return 0
    else
        log_error "Failed to authenticate to Docker registry"
        log_error "Please verify your credentials and try again"

        # Common troubleshooting hints
        echo ""
        echo "Troubleshooting:"
        echo "  1. Verify your Docker Hub username is correct"
        echo "  2. Ensure you're using an access token (not password)"
        echo "  3. Check token permissions (Read, Write, Delete)"
        echo "  4. Verify token hasn't expired"
        echo "  5. Try generating a new token at hub.docker.com/settings/security"

        return 1
    fi
}

docker_logout() {
    local registry="${1:-${DOCKER_REGISTRY}}"

    check_docker

    log_info "Logging out from Docker registry: $registry"

    if docker logout "$registry" 2>/dev/null; then
        log_success "Successfully logged out"
        return 0
    else
        log_warn "Logout may have failed or was not needed"
        return 0
    fi
}

verify_login() {
    check_docker

    log_info "Verifying Docker authentication..."

    local username
    username=$(docker info 2>/dev/null | grep -i "Username:" | awk '{print $2}' || echo "")

    if [ -n "$username" ]; then
        log_success "Currently authenticated as: $username"
        return 0
    else
        log_warn "Not currently authenticated to Docker Hub"
        return 1
    fi
}

load_from_env_file() {
    local env_file="$1"

    if [ ! -f "$env_file" ]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi

    log_info "Loading credentials from: $env_file"

    # Source the file carefully
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue

        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"

        case "$key" in
            DOCKER_USERNAME) export DOCKER_USERNAME="$value" ;;
            DOCKER_PASSWORD) export DOCKER_PASSWORD="$value" ;;
        esac
    done < "$env_file"

    return 0
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << EOF
${GREEN}Docker Credentials Helper${NC}

${YELLOW}Usage:${NC}
    $SCRIPT_NAME login [username] [token] [registry]
    $SCRIPT_NAME logout [registry]
    $SCRIPT_NAME verify
    $SCRIPT_NAME --help

${YELLOW}Commands:${NC}
    login       Authenticate to Docker registry
    logout      Remove Docker authentication
    verify      Check current authentication status

${YELLOW}Options:${NC}
    --from-env-file FILE    Load credentials from environment file

${YELLOW}Environment Variables:${NC}
    DOCKER_USERNAME     Docker Hub username
    DOCKER_PASSWORD     Docker Hub access token
    DOCKER_REGISTRY     Docker registry URL (default: docker.io)

${YELLOW}Security Best Practices:${NC}
    - Always use access tokens instead of passwords
    - Generate tokens at: hub.docker.com/settings/security
    - Set token permissions to minimum required (Read, Write)
    - Rotate tokens every 90 days
    - Never commit credentials to version control
    - Use Jenkins credentials or secret management

${YELLOW}Examples:${NC}
    # Login with parameters
    $SCRIPT_NAME login myuser dckr_pat_xxxxxxxxxxxxx

    # Login with environment variables
    export DOCKER_USERNAME=myuser
    export DOCKER_PASSWORD=dckr_pat_xxxxxxxxxxxxx
    $SCRIPT_NAME login

    # Login with env file
    $SCRIPT_NAME login --from-env-file .env

    # Verify authentication
    $SCRIPT_NAME verify

    # Logout
    $SCRIPT_NAME logout

EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    local command="${1:-}"

    case "$command" in
        login)
            shift

            # Check for --from-env-file option
            if [ "${1:-}" = "--from-env-file" ]; then
                shift
                load_from_env_file "${1:-}"
                docker_login
            else
                docker_login "${1:-}" "${2:-}" "${3:-}"
            fi
            ;;
        logout)
            shift
            docker_logout "${1:-}"
            ;;
        verify)
            verify_login
            ;;
        --help|-h|help)
            show_help
            ;;
        "")
            log_error "No command specified"
            echo ""
            show_help
            exit 2
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 2
            ;;
    esac
}

main "$@"
