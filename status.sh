#!/bin/bash

# ISO 27001:2022 Expert Agent Status Script
# Script to check the status of the ISO compliance analysis system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PID_FILE="$PROJECT_ROOT/server.pid"

# Default port
DEFAULT_PORT=8000

# Function to log messages
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Function to check port status
check_port_status() {
    local port=${1:-$DEFAULT_PORT}
    local pids=$(lsof -t -i :$port 2>/dev/null || true)

    if [[ -n "$pids" ]]; then
        echo -e "${GREEN}✅ Service is running on port $port${NC}"
        echo -e "   Process IDs: $pids"
        return 0
    else
        echo -e "${RED}❌ No service found on port $port${NC}"
        return 1
    fi
}

# Function to check PID file status
check_pid_status() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✅ Server running with PID: $pid${NC}"
            return 0
        else
            echo -e "${RED}❌ PID file exists but process is not running${NC}"
            echo -e "   Stale PID file: $PID_FILE (PID: $pid)"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  No PID file found${NC}"
        return 1
    fi
}

# Function to check process by name
check_process_status() {
    local pids=$(pgrep -f "document_extraction_system.main_simple" 2>/dev/null || true)

    if [[ -n "$pids" ]]; then
        echo -e "${GREEN}✅ ISO Agent processes found${NC}"
        echo -e "   Process IDs: $pids"
        return 0
    else
        echo -e "${RED}❌ No ISO Agent processes found${NC}"
        return 1
    fi
}

# Function to test API endpoints
test_endpoints() {
    local port=${1:-$DEFAULT_PORT}
    local base_url="http://localhost:$port"

    echo -e "${CYAN}🔍 Testing API endpoints...${NC}"

    # Test health endpoint
    if curl -s -f "$base_url/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Health check endpoint: HEALTHY${NC}"
    else
        echo -e "${RED}❌ Health check endpoint: FAILED${NC}"
    fi

    # Test main endpoint
    if curl -s -f "$base_url/" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Main endpoint: ACCESSIBLE${NC}"
    else
        echo -e "${RED}❌ Main endpoint: FAILED${NC}"
    fi

    # Test docs endpoint
    if curl -s -f "$base_url/docs" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ API documentation: AVAILABLE${NC}"
    else
        echo -e "${RED}❌ API documentation: FAILED${NC}"
    fi

    # Test ISO controls endpoint
    if curl -s -f "$base_url/api/v1/iso-controls" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ ISO controls endpoint: WORKING${NC}"
    else
        echo -e "${RED}❌ ISO controls endpoint: FAILED${NC}"
    fi
}

# Function to show detailed service information
show_service_info() {
    local port=${1:-$DEFAULT_PORT}

    echo -e "${CYAN}📊 Service Information:${NC}"
    echo -e "   Base URL: http://localhost:$port"
    echo -e "   API Docs: http://localhost:$port/docs"
    echo -e "   Health Check: http://localhost:$port/health"
    echo
    echo -e "${CYAN}🚀 Key Features Available:${NC}"
    echo -e "   • Document Analysis (93 Annex A Controls)"
    echo -e "   • Expert Consultation (Management Clauses 4-10)"
    echo -e "   • Gap Analysis & Recommendations"
    echo -e "   • Implementation Roadmaps"
    echo
}

# Function to show help
show_help() {
    echo -e "${BLUE}ISO 27001:2022 Expert Agent - Status Script${NC}"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --port PORT    Check specific port (default: $DEFAULT_PORT)"
    echo "  --test         Run API endpoint tests"
    echo "  --detailed     Show detailed service information"
    echo "  --help         Show this help message"
    echo
    echo "Examples:"
    echo "  $0              # Check basic status"
    echo "  $0 --test       # Check status and test endpoints"
    echo "  $0 --port 8080  # Check status on port 8080"
    echo "  $0 --detailed   # Show detailed information"
    echo
    exit 0
}

# Main function
main() {
    local port=$DEFAULT_PORT
    local run_tests=false
    local show_detailed=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                port="$2"
                shift 2
                ;;
            --test)
                run_tests=true
                shift
                ;;
            --detailed)
                show_detailed=true
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

    echo -e "${BLUE}📋 ISO 27001:2022 Expert Agent - Status Check${NC}"
    echo

    local service_running=false

    # Check various status indicators
    echo -e "${CYAN}🔍 Checking service status...${NC}"

    if check_port_status "$port"; then
        service_running=true
    fi

    echo

    if check_pid_status; then
        service_running=true
    fi

    echo

    if check_process_status; then
        service_running=true
    fi

    echo

    # Run API tests if requested and service is running
    if [[ "$run_tests" == "true" && "$service_running" == "true" ]]; then
        test_endpoints "$port"
        echo
    fi

    # Show detailed info if requested and service is running
    if [[ "$show_detailed" == "true" && "$service_running" == "true" ]]; then
        show_service_info "$port"
    fi

    # Final status summary
    if [[ "$service_running" == "true" ]]; then
        echo -e "${GREEN}🎯 Overall Status: ISO 27001:2022 Expert Agent is RUNNING${NC}"
        if [[ "$run_tests" == "false" ]]; then
            echo -e "${YELLOW}💡 Tip: Use --test to verify API endpoints${NC}"
        fi
    else
        echo -e "${RED}🚫 Overall Status: ISO 27001:2022 Expert Agent is NOT RUNNING${NC}"
        echo -e "${YELLOW}💡 Start the service with: ./bash.sh${NC}"
    fi

    echo
}

# Execute main function
main "$@"