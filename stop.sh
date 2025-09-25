#!/bin/bash

# ISO 27001:2022 Expert Agent Stop Script
# Script to stop the running ISO compliance analysis system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PID_FILE="$PROJECT_ROOT/server.pid"

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
    esac
}

# Function to stop server by PID file
stop_by_pid_file() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Stopping ISO Agent server (PID: $pid)..."
            kill "$pid"
            sleep 2

            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log "WARN" "Server still running, force killing..."
                kill -9 "$pid"
            fi

            rm -f "$PID_FILE"
            log "INFO" "Server stopped successfully"
            return 0
        else
            log "WARN" "Server with PID $pid is not running"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        log "WARN" "PID file not found: $PID_FILE"
        return 1
    fi
}

# Function to stop server by port
stop_by_port() {
    local port=${1:-8000}
    local pids=$(lsof -t -i :$port 2>/dev/null || true)

    if [[ -n "$pids" ]]; then
        log "INFO" "Stopping processes on port $port..."
        echo $pids | xargs -r kill
        sleep 2

        # Force kill if still running
        local remaining_pids=$(lsof -t -i :$port 2>/dev/null || true)
        if [[ -n "$remaining_pids" ]]; then
            log "WARN" "Force killing remaining processes..."
            echo $remaining_pids | xargs -r kill -9
        fi

        log "INFO" "Processes on port $port stopped"
        return 0
    else
        log "WARN" "No processes found on port $port"
        return 1
    fi
}

# Function to stop all python processes with our app name
stop_by_process_name() {
    local pids=$(pgrep -f "document_extraction_system.main_simple" || true)

    if [[ -n "$pids" ]]; then
        log "INFO" "Stopping ISO Agent processes..."
        echo $pids | xargs -r kill
        sleep 2

        # Force kill if still running
        local remaining_pids=$(pgrep -f "document_extraction_system.main_simple" || true)
        if [[ -n "$remaining_pids" ]]; then
            log "WARN" "Force killing remaining processes..."
            echo $remaining_pids | xargs -r kill -9
        fi

        log "INFO" "ISO Agent processes stopped"
        return 0
    else
        log "WARN" "No ISO Agent processes found"
        return 1
    fi
}

# Function to show help
show_help() {
    echo -e "${BLUE}ISO 27001:2022 Expert Agent - Stop Script${NC}"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --port PORT    Stop process on specific port (default: 8000)"
    echo "  --force        Force kill all related processes"
    echo "  --help         Show this help message"
    echo
    echo "Examples:"
    echo "  $0              # Stop using PID file, then fallback methods"
    echo "  $0 --port 8080 # Stop process on port 8080"
    echo "  $0 --force     # Force stop all related processes"
    echo
    exit 0
}

# Main function
main() {
    local port=8000
    local force_stop=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                port="$2"
                shift 2
                ;;
            --force)
                force_stop=true
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

    echo -e "${BLUE}🛑 Stopping ISO 27001:2022 Expert Agent...${NC}"
    echo

    local stopped=false

    # Try to stop by PID file first
    if stop_by_pid_file; then
        stopped=true
    fi

    # If PID file method failed, try by port
    if [[ "$stopped" == "false" ]]; then
        if stop_by_port "$port"; then
            stopped=true
        fi
    fi

    # If port method failed, try by process name
    if [[ "$stopped" == "false" ]]; then
        if stop_by_process_name; then
            stopped=true
        fi
    fi

    # Force stop if requested
    if [[ "$force_stop" == "true" ]]; then
        log "INFO" "Force stopping all related processes..."

        # Kill any remaining uvicorn processes
        pkill -f uvicorn 2>/dev/null || true

        # Kill any remaining python processes with our module
        pkill -f "document_extraction_system" 2>/dev/null || true

        # Clean up PID file
        rm -f "$PID_FILE"

        stopped=true
    fi

    if [[ "$stopped" == "true" ]]; then
        echo -e "${GREEN}✅ ISO 27001:2022 Expert Agent stopped successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  No running ISO Agent processes found${NC}"
    fi

    echo
}

# Execute main function
main "$@"