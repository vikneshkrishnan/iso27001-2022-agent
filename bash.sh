#!/bin/bash

# ISO 27001:2022 Expert Agent Startup Script
# Comprehensive startup script for the ISO compliance analysis system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_WORKERS="1"
DEBUG_MODE="true"
INSTALL_DEPS=false
HEALTH_CHECK_ONLY=false
PRODUCTION_MODE=false
BACKGROUND_MODE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PATH="$PROJECT_ROOT/venv"
SRC_PATH="$PROJECT_ROOT/src"
ENV_FILE="$PROJECT_ROOT/.env"

# Function to display banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    ISO 27001:2022 Expert Agent System                       ║"
    echo "║                                                                              ║"
    echo "║  🎯 Comprehensive compliance analysis for ISO 27001:2022                    ║"
    echo "║  📊 93 Annex A Controls + Management Clauses 4-10                          ║"
    echo "║  🤖 AI-powered document analysis and expert consultation                     ║"
    echo "║  🔍 Gap analysis, recommendations, and implementation roadmaps              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Function to display help
show_help() {
    echo -e "${BLUE}ISO 27001:2022 Expert Agent - Startup Script${NC}"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --host HOST        Set the host address (default: $DEFAULT_HOST)"
    echo "  --port PORT        Set the port number (default: $DEFAULT_PORT)"
    echo "  --workers NUM      Set number of workers (default: $DEFAULT_WORKERS)"
    echo "  --debug            Enable debug mode"
    echo "  --prod             Enable production mode (disables debug, reload)"
    echo "  --install          Install/update dependencies before starting"
    echo "  --check            Run health checks only (don't start server)"
    echo "  --background       Run in background mode"
    echo "  --help             Show this help message"
    echo
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY     OpenAI API key for enhanced LLM analysis"
    echo "  ANTHROPIC_API_KEY  Anthropic API key for enhanced LLM analysis"
    echo
    echo "Examples:"
    echo "  $0                          # Start with default settings"
    echo "  $0 --port 8080 --debug     # Start on port 8080 in debug mode"
    echo "  $0 --prod --workers 4      # Start in production mode with 4 workers"
    echo "  $0 --install --check       # Install deps and run health check"
    echo
    exit 0
}

# Function to log messages
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "DEBUG")
            echo -e "${PURPLE}[DEBUG]${NC} ${timestamp} - $message"
            ;;
        *)
            echo -e "${timestamp} - $message"
            ;;
    esac
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log "ERROR" "Port $port is already in use"
        echo -e "${YELLOW}To find what's using the port: ${NC}lsof -i :$port"
        echo -e "${YELLOW}To kill the process: ${NC}kill -9 \$(lsof -t -i :$port)"
        return 1
    fi
    log "INFO" "Port $port is available"
    return 0
}

# Function to check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python 3 is not installed"
        echo -e "${YELLOW}Please install Python 3.9 or higher${NC}"
        return 1
    fi

    local python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local major_version=$(echo $python_version | cut -d. -f1)
    local minor_version=$(echo $python_version | cut -d. -f2)

    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 9 ]]; then
        log "ERROR" "Python version $python_version is not supported. Requires Python 3.9+"
        return 1
    fi

    log "INFO" "Python version $python_version is supported"
    return 0
}

# Function to setup virtual environment
setup_venv() {
    if [[ ! -d "$VENV_PATH" ]]; then
        log "INFO" "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    fi

    log "INFO" "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"

    # Verify activation (check if VIRTUAL_ENV is set, path might differ)
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log "ERROR" "Failed to activate virtual environment"
        return 1
    fi

    log "INFO" "Virtual environment activated: $VIRTUAL_ENV"

    # Additional check - make sure we can run python from venv
    if ! which python3 | grep -q "$VIRTUAL_ENV\|venv"; then
        log "WARN" "Python path might not be from virtual environment"
        log "INFO" "Current Python: $(which python3)"
    fi

    return 0
}

# Function to install dependencies
install_dependencies() {
    log "INFO" "Installing/updating dependencies..."

    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log "ERROR" "requirements.txt not found in $PROJECT_ROOT"
        return 1
    fi

    pip install --upgrade pip
    pip install -r "$PROJECT_ROOT/requirements.txt"

    log "INFO" "Dependencies installed successfully"
    return 0
}

# Function to validate environment
validate_environment() {
    log "INFO" "Validating environment..."

    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log "WARN" ".env file not found. Using default configuration."
    else
        log "INFO" "Environment file found: $ENV_FILE"
        # Source the .env file safely, ignoring lines with JSON arrays that bash can't handle
        if set -a; grep -v '\[.*\]' "$ENV_FILE" | source /dev/stdin; set +a; then
            log "DEBUG" "Environment variables loaded from .env file (JSON arrays skipped for bash)"
        else
            log "WARN" "Failed to load some environment variables from .env file"
        fi

        # Manually extract API keys since they're simple string values
        export OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" "$ENV_FILE" | cut -d'=' -f2-)
        export ANTHROPIC_API_KEY=$(grep "^ANTHROPIC_API_KEY=" "$ENV_FILE" | cut -d'=' -f2- 2>/dev/null || true)
        export PINECONE_API_KEY=$(grep "^PINECONE_API_KEY=" "$ENV_FILE" | cut -d'=' -f2- 2>/dev/null || true)
    fi

    # Check if source directory exists
    if [[ ! -d "$SRC_PATH" ]]; then
        log "ERROR" "Source directory not found: $SRC_PATH"
        return 1
    fi

    # Check if main module exists
    if [[ ! -f "$SRC_PATH/document_extraction_system/main_simple.py" ]]; then
        log "ERROR" "Main application module not found"
        return 1
    fi

    # Check for LLM API keys (now properly loaded from .env)
    if [[ -n "$OPENAI_API_KEY" ]]; then
        log "INFO" "OpenAI API key detected - enhanced LLM analysis available"
    fi

    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        log "INFO" "Anthropic API key detected - enhanced LLM analysis available"
    fi

    if [[ -z "$OPENAI_API_KEY" && -z "$ANTHROPIC_API_KEY" ]]; then
        log "WARN" "No LLM API keys detected - using rule-based analysis only"
        echo -e "${YELLOW}For enhanced AI analysis, set OPENAI_API_KEY or ANTHROPIC_API_KEY${NC}"
    fi

    log "INFO" "Environment validation completed"
    return 0
}

# Function to run health check
health_check() {
    log "INFO" "Running system health checks..."

    # Check Python
    check_python || return 1

    # Setup and activate virtual environment
    setup_venv || return 1

    # Validate environment
    validate_environment || return 1

    # Check if dependencies are installed
    log "INFO" "Checking Python dependencies..."
    python3 -c "
import sys
try:
    import fastapi
    import uvicorn
    import fitz
    print('✓ Core dependencies are installed')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
" || {
        log "ERROR" "Dependencies check failed"
        echo -e "${YELLOW}Run with --install to install dependencies${NC}"
        return 1
    }

    # Check port availability
    check_port $DEFAULT_PORT || return 1

    log "INFO" "All health checks passed ✓"
    return 0
}

# Function to start the server
start_server() {
    local host=${1:-$DEFAULT_HOST}
    local port=${2:-$DEFAULT_PORT}
    local workers=${3:-$DEFAULT_WORKERS}
    local debug=${4:-$DEBUG_MODE}

    log "INFO" "Starting ISO 27001:2022 Expert Agent..."
    log "INFO" "Host: $host"
    log "INFO" "Port: $port"
    log "INFO" "Workers: $workers"
    log "INFO" "Debug Mode: $debug"

    # Change to source directory
    cd "$SRC_PATH"

    # Start the application
    if [[ "$PRODUCTION_MODE" == "true" ]]; then
        log "INFO" "Starting in PRODUCTION mode..."
        uvicorn document_extraction_system.main_simple:app \
            --host "$host" \
            --port "$port" \
            --workers "$workers" \
            --log-level info \
            --no-reload
    else
        log "INFO" "Starting in DEVELOPMENT mode..."
        python3 -m document_extraction_system.main_simple
    fi
}

# Function to display service information
show_service_info() {
    local host=${1:-$DEFAULT_HOST}
    local port=${2:-$DEFAULT_PORT}

    echo
    echo -e "${GREEN}🚀 ISO 27001:2022 Expert Agent is running!${NC}"
    echo
    echo -e "${CYAN}📍 Service URLs:${NC}"
    echo -e "   Main Application: http://$host:$port/"
    echo -e "   API Documentation: http://$host:$port/docs"
    echo -e "   Alternative Docs: http://$host:$port/redoc"
    echo -e "   Health Check: http://$host:$port/health"
    echo
    echo -e "${CYAN}🔧 Main API Endpoints:${NC}"
    echo -e "   Document Analysis: POST /api/v1/iso-analyze"
    echo -e "   Expert Consultation: POST /api/v1/iso-consult"
    echo -e "   ISO Controls: GET /api/v1/iso-controls"
    echo -e "   Text Extraction: POST /api/v1/extract"
    echo
    echo -e "${CYAN}💡 Example Usage:${NC}"
    echo -e "   curl -X POST 'http://$host:$port/api/v1/iso-analyze' -F 'file=@policy.pdf'"
    echo -e "   curl -X POST 'http://$host:$port/api/v1/iso-consult' -d '{\"question\":\"How to implement access control?\"}'"
    echo
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo
}

# Function to cleanup on exit
cleanup() {
    log "INFO" "Shutting down ISO 27001:2022 Expert Agent..."
    # Kill any background processes if needed
    jobs -p | xargs -r kill 2>/dev/null || true
    log "INFO" "Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            DEFAULT_HOST="$2"
            shift 2
            ;;
        --port)
            DEFAULT_PORT="$2"
            shift 2
            ;;
        --workers)
            DEFAULT_WORKERS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="true"
            PRODUCTION_MODE=false
            shift
            ;;
        --prod)
            PRODUCTION_MODE=true
            DEBUG_MODE="false"
            shift
            ;;
        --install)
            INSTALL_DEPS=true
            shift
            ;;
        --check)
            HEALTH_CHECK_ONLY=true
            shift
            ;;
        --background)
            BACKGROUND_MODE=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    show_banner

    # Install dependencies if requested
    if [[ "$INSTALL_DEPS" == "true" ]]; then
        check_python || exit 1
        setup_venv || exit 1
        install_dependencies || exit 1
    fi

    # Run health check
    health_check || exit 1

    # Exit if only health check was requested
    if [[ "$HEALTH_CHECK_ONLY" == "true" ]]; then
        log "INFO" "Health check completed successfully ✓"
        exit 0
    fi

    # Show service information
    show_service_info "$DEFAULT_HOST" "$DEFAULT_PORT"

    # Start the server
    if [[ "$BACKGROUND_MODE" == "true" ]]; then
        log "INFO" "Starting server in background mode..."
        start_server "$DEFAULT_HOST" "$DEFAULT_PORT" "$DEFAULT_WORKERS" "$DEBUG_MODE" &
        local server_pid=$!
        log "INFO" "Server started with PID: $server_pid"
        echo $server_pid > "$PROJECT_ROOT/server.pid"
        log "INFO" "Server PID saved to server.pid"
    else
        start_server "$DEFAULT_HOST" "$DEFAULT_PORT" "$DEFAULT_WORKERS" "$DEBUG_MODE"
    fi
}

# Execute main function
main "$@"